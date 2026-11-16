// Virtual sequences: iDMA transfers through the RAL, random traffic, end-of-test memory check.

class tpu_base_vseq extends uvm_sequence;
  `uvm_object_utils(tpu_base_vseq)
  `uvm_declare_p_sequencer(tpu_vsqr)
  bit [31:0] conf_extra = 0;     // OR-ed into conf (burst reduction / decoupling experiments)
  function new(string name = "tpu_base_vseq"); super.new(name); endfunction

  // program + launch (one next_id read) - returns the transfer id
  task launch(dma_xfer x, output bit [31:0] id);
    uvm_status_e st;
    uvm_reg_data_t v;
    p_sequencer.regs.src_addr  .write(st, x.src_addr(), .parent(this));
    p_sequencer.regs.dst_addr  .write(st, x.dst_addr(), .parent(this));
    p_sequencer.regs.length    .write(st, x.len,       .parent(this));
    p_sequencer.regs.src_stride.write(st, x.sstride,   .parent(this));
    p_sequencer.regs.dst_stride.write(st, x.dstride,   .parent(this));
    p_sequencer.regs.reps      .write(st, x.reps,      .parent(this));
    p_sequencer.regs.conf      .write(st, ((x.reps > 1) ? 'h400 : 'h0) | conf_extra, .parent(this));
    p_sequencer.regs.next_id   .read (st, v,           .parent(this));   // launch
    id = v[31:0];
    if (id == 0) `uvm_error("DMA", $sformatf("next_id = 0 for %s", x.brief()))
    x.id = id;
  endtask

  task wait_done(bit [31:0] id, int unsigned max_polls = 60000);
    uvm_status_e st; uvm_reg_data_t v; int unsigned n = 0;
    do begin p_sequencer.regs.done_id.read(st, v, .parent(this)); n++; end
    while ($signed(v[31:0] - id) < 0 && n < max_polls);
    if ($signed(v[31:0] - id) < 0)
      `uvm_fatal("DMA", $sformatf("id %0d HUNG (done=%0d) busy=%0b dma_err=%b fab_err=%b wf_level=%0d",
                 id, v[31:0], p_sequencer.vif.dma_busy, p_sequencer.vif.dma_err,
                 p_sequencer.vif.fab_err, p_sequencer.vif.wf_level))
  endtask

  // fill DDR source rows with random bytes (backdoor + model), poison DDR destination rows
  function void prep(dma_xfer x);
    if (x.src_sp == SP_DDR)
      for (int unsigned r = 0; r < x.reps; r++)
        for (int unsigned i = 0; i < x.len; i++) begin
          longint unsigned a = x.src_addr() + r * x.sstride + i;
          byte unsigned b = $urandom();
          p_sequencer.ddr.write(a, b); p_sequencer.model.wr(SP_DDR, a, b);
        end
    if (x.dst_sp == SP_DDR)
      for (int unsigned r = 0; r < x.reps; r++)
        for (int unsigned i = 0; i < x.len; i++) p_sequencer.ddr.write(x.dst_addr() + r * x.dstride + i, 8'hEE);
  endfunction

  task do_xfer(dma_xfer x);
    bit [31:0] id;
    prep(x);
    // predict BEFORE launch: FIFO words are popped (and checked) while the transfer runs.
    // Safe because source and destination ranges are disjoint by constraint.
    p_sequencer.model.apply(x);
    `uvm_info("XFER", $sformatf("launch %s", x.brief()), UVM_MEDIUM)
    launch(x, id);
    wait_done(id);
    // every transfer must leave all error flags clear (a DECERR / partial FIFO beat / ownership
    // block is never silent); tests that provoke errors call launch/wait_done directly
    if (p_sequencer.vif.dma_err != 0 || p_sequencer.vif.fab_err != 0) begin
      `uvm_error("FLAGS", $sformatf("xfer %0d %s left dma_err=%b fab_err=%b", id, x.brief(),
                                    p_sequencer.vif.dma_err, p_sequencer.vif.fab_err))
      set_ctrl("err_clr", 1); set_ctrl("err_clr", 0);
    end
    if (x.dst_sp == SP_DDR) p_sequencer.sb.check_ddr(x);
    p_sequencer.cov.sample_xfer(x);
  endtask

  task set_ctrl(string what, bit v);
    @(posedge p_sequencer.vif.clk); #2ns;
    case (what)
      "ub_own":  p_sequencer.vif.ub_own  = v;
      "acc_own": p_sequencer.vif.acc_own = v;
      "err_clr": p_sequencer.vif.err_clr = v;
      default: `uvm_fatal("CTRL", what)
    endcase
  endtask
endclass

class dma_random_vseq extends tpu_base_vseq;
  `uvm_object_utils(dma_random_vseq)
  int unsigned n = 200;
  function new(string name = "dma_random_vseq"); super.new(name); endfunction
  task body();
    for (int unsigned k = 0; k < n; k++) begin
      dma_xfer x = dma_xfer::type_id::create($sformatf("x%0d", k));
      if (!x.randomize()) `uvm_fatal("RAND", "dma_xfer randomize failed")
      do_xfer(x);
    end
  endtask
endclass

// end of test: drain the Weight FIFO, then compare every known UB/ACC byte via the native ports
class final_check_vseq extends tpu_base_vseq;
  `uvm_object_utils(final_check_vseq)
  function new(string name = "final_check_vseq"); super.new(name); endfunction
  task body();
    int unsigned rows[$], bad = 0, nb = 0, n = 0;
    nat_item it;
    while (p_sequencer.model.wf_q.size() != 0 && n < 200000) begin @(posedge p_sequencer.vif.clk); n++; end
    // UB
    foreach (p_sequencer.model.ub[o]) if (rows.size() == 0 || rows[$] != o / 16) rows.push_back(o / 16);
    foreach (rows[k]) begin
      it = nat_item::type_id::create("rd");
      start_item(it, -1, p_sequencer.nat_sqr);
      it.mem = NAT_UB; it.write = 0; it.row = rows[k];
      finish_item(it);
      for (int b = 0; b < 16; b++) if (p_sequencer.model.ub.exists(rows[k]*16 + b)) begin
        nb++; if (it.data[b*8 +: 8] != p_sequencer.model.ub[rows[k]*16 + b]) bad++;
      end
    end
    `uvm_info("CHECK", $sformatf("UB: %0d rows, %0d known bytes compared via native port", rows.size(), nb), UVM_NONE)
    if (bad) `uvm_error("CHECK", $sformatf("UB: %0d bytes wrong", bad))
    // ACC
    rows.delete(); bad = 0; nb = 0;
    foreach (p_sequencer.model.acc[o]) if (rows.size() == 0 || rows[$] != o / 64) rows.push_back(o / 64);
    foreach (rows[k]) begin
      it = nat_item::type_id::create("rd");
      start_item(it, -1, p_sequencer.nat_sqr);
      it.mem = NAT_ACC; it.write = 0; it.row = rows[k];
      finish_item(it);
      for (int b = 0; b < 64; b++) if (p_sequencer.model.acc.exists(rows[k]*64 + b)) begin
        nb++; if (it.data[b*8 +: 8] != p_sequencer.model.acc[rows[k]*64 + b]) bad++;
      end
    end
    `uvm_info("CHECK", $sformatf("ACC: %0d rows, %0d known bytes compared via native port", rows.size(), nb), UVM_NONE)
    if (bad) `uvm_error("CHECK", $sformatf("ACC: %0d bytes wrong", bad))
  endtask
endclass
