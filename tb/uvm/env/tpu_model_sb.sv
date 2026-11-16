// Reference model of every memory space + scoreboard + coverage.

// Backdoor into the DDR model (implemented in the top module, which can see axi_sim_mem).
virtual class ddr_backdoor extends uvm_object;
  function new(string name = "ddr_backdoor"); super.new(name); endfunction
  pure virtual function void write(longint unsigned a, byte unsigned d);
  pure virtual function int  read(longint unsigned a);   // -1 = never written
  pure virtual function void set_rerr(longint unsigned a, axi_pkg::resp_t r);
endclass

class tpu_ref_model extends uvm_object;
  `uvm_object_utils(tpu_ref_model)
  // known bytes only (absent = unknown/uninitialised)
  byte unsigned ddr [longint unsigned];
  byte unsigned ub  [int unsigned];
  byte unsigned acc [int unsigned];
  int           wf_q [$][16];             // expected Weight FIFO words (-1 = don't care)
  local int     wf_part [16];
  local int     wf_fill;
  function new(string name = "tpu_ref_model"); super.new(name); endfunction

  function int rd(mem_space_e sp, longint unsigned a);
    case (sp)
      SP_DDR: return ddr.exists(a) ? ddr[a] : -1;
      SP_UB:  begin int unsigned o = a - tpu_pkg::UB_BASE;  return ub.exists(o)  ? ub[o]  : -1; end
      SP_ACC: begin int unsigned o = a - tpu_pkg::ACC_BASE; return acc.exists(o) ? acc[o] : -1; end
      default: return -1;
    endcase
  endfunction

  function void wr(mem_space_e sp, longint unsigned a, int v);
    case (sp)
      SP_DDR: if (v < 0) ddr.delete(a); else ddr[a] = v;
      SP_UB:  begin int unsigned o = a - tpu_pkg::UB_BASE;  if (v < 0) ub.delete(o);  else ub[o]  = v; end
      SP_ACC: begin int unsigned o = a - tpu_pkg::ACC_BASE; if (v < 0) acc.delete(o); else acc[o] = v; end
      SP_WFIFO: begin
        wf_part[wf_fill++] = v;
        if (wf_fill == 16) begin wf_q.push_back(wf_part); wf_fill = 0; end
      end
    endcase
  endfunction

  function void apply(dma_xfer x);
    for (int unsigned r = 0; r < x.reps; r++)
      for (int unsigned i = 0; i < x.len; i++)
        wr(x.dst_sp, (x.dst_sp == SP_WFIFO) ? 0 : x.dst_addr() + r * x.dstride + i,
           rd(x.src_sp, x.src_addr() + r * x.sstride + i));
  endfunction
endclass

class tpu_scoreboard extends uvm_component;
  `uvm_component_utils(tpu_scoreboard)
  `uvm_analysis_imp_decl(_wf)
  uvm_analysis_imp_wf #(fifo_word, tpu_scoreboard) wf_imp;
  tpu_ref_model model;
  ddr_backdoor  ddr;
  int unsigned  n_xfer_checked, n_bytes_checked, n_words_checked, n_errors;

  function new(string name, uvm_component parent); super.new(name, parent); endfunction
  function void build_phase(uvm_phase phase); wf_imp = new("wf_imp", this); endfunction

  function void write_wf(fifo_word w);
    int exp[16];
    if (model.wf_q.size() == 0) begin
      n_errors++; `uvm_error("SB", $sformatf("unexpected Weight FIFO word %h", w.data)) return;
    end
    exp = model.wf_q.pop_front();
    foreach (exp[b]) if (exp[b] >= 0 && w.data[b*8 +: 8] != exp[b][7:0]) begin
      n_errors++;
      `uvm_error("SB", $sformatf("Weight FIFO word %0d byte %0d: got %h exp %h",
                                 n_words_checked, b, w.data[b*8 +: 8], exp[b][7:0]))
      break;
    end
    n_words_checked++;
  endfunction

  function void check_ddr(dma_xfer x);
    int unsigned bad = 0;
    for (int unsigned r = 0; r < x.reps; r++)
      for (int unsigned i = 0; i < x.len; i++) begin
        longint unsigned a = x.dst_addr() + r * x.dstride + i;
        int e = model.rd(SP_DDR, a);
        if (e >= 0) begin n_bytes_checked++; if (ddr.read(a) != e) bad++; end
      end
    if (bad) begin n_errors++; `uvm_error("SB", $sformatf("xfer %0d %s: %0d DDR bytes wrong", x.id, x.brief(), bad)) end
    n_xfer_checked++;
  endfunction

  function void report_phase(uvm_phase phase);
    if (model.wf_q.size() != 0) begin
      n_errors++; `uvm_error("SB", $sformatf("%0d Weight FIFO words never popped", model.wf_q.size()))
    end
    `uvm_info("SB", $sformatf("checked: %0d DDR transfers (%0d bytes), %0d FIFO words; errors=%0d",
                              n_xfer_checked, n_bytes_checked, n_words_checked, n_errors), UVM_NONE)
  endfunction
endclass

class tpu_coverage extends uvm_component;
  `uvm_component_utils(tpu_coverage)
  dma_xfer x;
  bit backpressure_seen, own_block_seen, rd_err_seen;

  covergroup cg_xfer;
    option.per_instance = 1;
    cp_src:  coverpoint x.src_sp { bins sp[] = {SP_DDR, SP_UB, SP_ACC}; }
    cp_dst:  coverpoint x.dst_sp;
    cp_path: cross cp_src, cp_dst;
    cp_len:  coverpoint x.len { bins one = {1}; bins sub16 = {[2:15]}; bins b16 = {16};
                                bins sml = {[17:255]}; bins mid = {[256:4095]};
                                bins b4k = {4096}; bins big = {[4097:$]}; }
    cp_2d:   coverpoint (x.reps > 1);
    cp_salign: coverpoint (x.src_addr() % 16 == 0);
    cp_dalign: coverpoint (x.dst_addr() % 16 == 0);
    cp_4k:   coverpoint x.crosses_4k();
    cp_hi:   coverpoint (x.src_sp == SP_DDR && x.src_hi);
    x_len_2d: cross cp_len, cp_2d;
  endgroup

  covergroup cg_events;
    cp_bp:  coverpoint backpressure_seen { bins hit = {1}; }
    cp_own: coverpoint own_block_seen    { bins hit = {1}; }
    cp_err: coverpoint rd_err_seen       { bins hit = {1}; }
  endgroup

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cg_xfer = new(); cg_events = new();
  endfunction
  function void sample_xfer(dma_xfer t); x = t; cg_xfer.sample(); endfunction
  function void sample_events(); cg_events.sample(); endfunction
  function void report_phase(uvm_phase phase);
    `uvm_info("COV", $sformatf("transfer coverage %0.1f%%, event coverage %0.1f%%",
                               cg_xfer.get_inst_coverage(), cg_events.get_inst_coverage()), UVM_NONE)
  endfunction
endclass
