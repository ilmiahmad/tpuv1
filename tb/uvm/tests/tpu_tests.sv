// Tests.  +UVM_TESTNAME=<name>   +NXFER=<n> (random test)

class tpu_base_test extends uvm_test;
  `uvm_component_utils(tpu_base_test)
  tpu_env env;
  function new(string name, uvm_component parent); super.new(name, parent); endfunction
  function void build_phase(uvm_phase phase); env = tpu_env::type_id::create("env", this); endfunction
  task wait_reset();
    wait (env.vsqr.vif.rst_n === 1'b1);
    repeat (5) @(posedge env.vsqr.vif.clk);
  endtask
  function void report_phase(uvm_phase phase);
    uvm_report_server rs = uvm_report_server::get_server();
    int n_err = rs.get_severity_count(UVM_ERROR) + rs.get_severity_count(UVM_FATAL);
    `uvm_info("RESULT", n_err == 0 ? "TPU_UVM_PASS" : $sformatf("TPU_UVM_FAIL errors=%0d", n_err), UVM_NONE)
  endfunction
endclass

// random iDMA traffic between DDR (incl. >4 GiB), UB, ACC and the Weight FIFO, random stalls on
// HP0 and a randomly stalling Weight FIFO consumer; end-of-test UB/ACC check via native ports
class tpu_dma_random_test extends tpu_base_test;
  `uvm_component_utils(tpu_dma_random_test)
  function new(string name, uvm_component parent); super.new(name, parent); endfunction
  task run_phase(uvm_phase phase);
    dma_random_vseq  rs = dma_random_vseq::type_id::create("rs");
    final_check_vseq fc = final_check_vseq::type_id::create("fc");
    phase.raise_objection(this);
    wait_reset();
    if (!$value$plusargs("NXFER=%d", rs.n)) rs.n = 200;
    rs.start(env.vsqr);
    fc.start(env.vsqr);
    phase.drop_objection(this);
  endtask
endclass

// Weight FIFO backpressure: consumer stalled -> DMA held; release -> everything delivered in order
class tpu_fifo_backpressure_test extends tpu_base_test;
  `uvm_component_utils(tpu_fifo_backpressure_test)
  function new(string name, uvm_component parent); super.new(name, parent); endfunction
  task run_phase(uvm_phase phase);
    tpu_base_vseq v = tpu_base_vseq::type_id::create("v");
    dma_xfer x = dma_xfer::type_id::create("x");
    bit [31:0] id;
    uvm_status_e st; uvm_reg_data_t d;
    phase.raise_objection(this);
    wait_reset();
    env.wf_sink.ready_pct = 0;
    x.c_len.constraint_mode(0); x.c_reps.constraint_mode(0);
    if (!x.randomize() with { src_sp == SP_DDR; dst_sp == SP_WFIFO; len == 1280 * 16; reps == 1; src_off < 4096; })
      `uvm_fatal("RAND", "x")
    v.set_sequencer(env.vsqr);
    v.prep(x);
    env.model.apply(x);                          // expected words, consumed in order by the sink
    v.launch(x, id);
    repeat (30000) @(posedge env.vsqr.vif.clk);
    env.regs.done_id.read(st, d);
    if ($signed(d[31:0] - id) >= 0) `uvm_error("BP", "DMA completed although the FIFO is full")
    if (env.vsqr.vif.wf_level < tpu_pkg::WfifoWords) `uvm_error("BP", $sformatf("level %0d", env.vsqr.vif.wf_level))
    else begin env.cov.backpressure_seen = 1; env.cov.sample_events(); end
    `uvm_info("BP", $sformatf("held: level=%0d", env.vsqr.vif.wf_level), UVM_NONE)
    env.wf_sink.ready_pct = 60;
    v.wait_done(id);
    begin
      final_check_vseq fc = final_check_vseq::type_id::create("fc");
      fc.start(env.vsqr);
    end
    `uvm_info("BP", $sformatf("released: %0d words popped", env.wf_sink.popped), UVM_NONE)
    phase.drop_objection(this);
  endtask
endclass

// ownership + error flags: TPU-owned UB rejects iDMA writes; a DDR read error is flagged
class tpu_ownership_err_test extends tpu_base_test;
  `uvm_component_utils(tpu_ownership_err_test)
  function new(string name, uvm_component parent); super.new(name, parent); endfunction
  task run_phase(uvm_phase phase);
    tpu_base_vseq v = tpu_base_vseq::type_id::create("v");
    dma_xfer a = dma_xfer::type_id::create("a");
    dma_xfer b = dma_xfer::type_id::create("b");
    dma_xfer c = dma_xfer::type_id::create("c");
    bit [31:0] id;
    phase.raise_objection(this);
    wait_reset();
    v.set_sequencer(env.vsqr);
    // 1. fill a UB area normally
    if (!a.randomize() with { src_sp == SP_DDR; dst_sp == SP_UB; reps == 1; len == 512; dst_off == 0; })
      `uvm_fatal("RAND", "a")
    v.do_xfer(a);
    // 2. same area while TPU-owned: must be blocked (model NOT updated)
    if (!b.randomize() with { src_sp == SP_DDR; dst_sp == SP_UB; reps == 1; len == 512; dst_off == 0; })
      `uvm_fatal("RAND", "b")
    v.set_ctrl("ub_own", 1);
    v.prep(b); v.launch(b, id); v.wait_done(id);
    v.set_ctrl("ub_own", 0);
    if (!env.vsqr.vif.fab_err[0] || !env.vsqr.vif.dma_err[1])
      `uvm_error("OWN", $sformatf("flags not set: fab_err=%b dma_err=%b", env.vsqr.vif.fab_err, env.vsqr.vif.dma_err))
    else begin env.cov.own_block_seen = 1; env.cov.sample_events(); end
    v.set_ctrl("err_clr", 1); v.set_ctrl("err_clr", 0);
    // 3. read error injected on a DDR source byte: transfer completes, flag set
    if (!c.randomize() with { src_sp == SP_DDR; dst_sp == SP_UB; reps == 1; len == 256; dst_off == 4096; })
      `uvm_fatal("RAND", "c")
    v.prep(c);
    env.vsqr.ddr.set_rerr(c.src_addr() + 17, axi_pkg::RESP_SLVERR);
    v.launch(c, id); v.wait_done(id);
    if (!env.vsqr.vif.dma_err[0]) `uvm_error("ERR", "read SLVERR not flagged")
    else begin env.cov.rd_err_seen = 1; env.cov.sample_events(); end
    v.set_ctrl("err_clr", 1); v.set_ctrl("err_clr", 0);
    begin   // UB area still holds transfer a's data
      final_check_vseq fc = final_check_vseq::type_id::create("fc");
      fc.start(env.vsqr);
    end
    phase.drop_objection(this);
  endtask
endclass

// Same-memory copy (ACC->ACC, UB->UB). +CONF=<hex> ORs iDMA conf bits:
//   0x00 default (expected to deadlock), 0x0C reduce src+dst bursts to 1 beat, 0x03 decouple
class tpu_same_mem_test extends tpu_base_test;
  `uvm_component_utils(tpu_same_mem_test)
  function new(string name, uvm_component parent); super.new(name, parent); endfunction
  task run_phase(uvm_phase phase);
    tpu_base_vseq v = tpu_base_vseq::type_id::create("v");
    dma_xfer a = dma_xfer::type_id::create("a");
    dma_xfer b = dma_xfer::type_id::create("b");
    int unsigned conf = 0;
    phase.raise_objection(this);
    wait_reset();
    v.set_sequencer(env.vsqr);
    void'($value$plusargs("CONF=%h", conf));
    // seed the ACC with known data, then copy inside it
    if (!a.randomize() with { src_sp == SP_DDR; dst_sp == SP_ACC; reps == 1; len == 4096; dst_off == 0; })
      `uvm_fatal("RAND", "a")
    v.do_xfer(a);
    b.c_nosame.constraint_mode(0);
    if (!b.randomize() with { src_sp == SP_ACC; dst_sp == SP_ACC; reps == 1; len == 2048; src_off == 0; dst_off == 8192; })
      `uvm_fatal("RAND", "b")
    v.conf_extra = conf;
    `uvm_info("SAME", $sformatf("ACC->ACC 2048 B with conf |= 0x%0h", conf), UVM_NONE)
    v.do_xfer(b);
    `uvm_info("SAME", "ACC->ACC completed", UVM_NONE)
    begin final_check_vseq fc = final_check_vseq::type_id::create("fc"); fc.start(env.vsqr); end
    phase.drop_objection(this);
  endtask
endclass
