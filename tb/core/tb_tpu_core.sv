// M4 - tpu_core + tpu_fabric with real memories, running the golden-model golden programs.
//
//   port 1 (HPM data window): UB init, ACC clear, program -> Instruction FIFO, result read-back
//   port 0 (second fabric slave, models an interleaving host DMA): weights -> Weight FIFO
// Two masters because a program longer than the 512-entry IFIFO plus a weight stream larger than
// the WFIFO deadlock on one sequential master (finding).
//
// Per case: flush FIFOs, clear errors/perf, write ub_init, zero ACC rows 0..acc_rows-1,
// run=1, push program and weights concurrently, wait for INTERRUPT_HOST (or cfg_err), wait idle,
// compare UB output region with ub_expect.hex and ACC rows with acc_expect.hex bit-exactly,
// check irq tag / cfg_err opcode, print perf counters. All cases back-to-back without reset
// (L14), then a reset in the middle of mlp3 and a clean re-run.
// Plusargs: +CASES=<cases.txt> (default build/m4/cases/cases.txt)  +ONLY=<case name>

`timescale 1ns/1ps
`include "axi/typedef.svh"

module tb_tpu_core;
  import tpu_pkg::*;

  localparam time TCK = 10ns, TA = 2ns;
  localparam int unsigned N = 16;

  logic clk = 1'b0, rst_n = 1'b0;
  always #(TCK/2) clk = ~clk;

  // ---------------------------------------------------------------------------------------------
  // DUT
  // ---------------------------------------------------------------------------------------------
  fs_req_t  mreq [2];
  fs_resp_t mrsp [2];
  fm_req_t  hp0_req;
  fm_resp_t hp0_rsp;
  assign hp0_rsp = '0;            // no DDR traffic in this bench

  logic ub_rd_en, ub_wr_en, acc_rd_en, acc_wr_en;
  logic [15:0] ub_rd_addr, ub_wr_addr;
  logic [10:0] acc_rd_addr, acc_wr_addr;
  data_t ub_rd_data, ub_wr_data;
  logic [15:0] ub_wr_be;
  acc_row_t acc_rd_data, acc_wr_data;
  logic [3:0] acc_wr_be;
  logic wf_valid, wf_ready, if_valid, if_ready;
  data_t wf_data, if_data;
  logic wf_flush = 0, if_flush = 0, fab_err_clr = 0;
  logic [3:0] fab_err;
  logic [15:0] wf_level, if_level;

  tpu_fabric i_fab (
    .clk_i(clk), .rst_ni(rst_n),
    .dma_req_i(mreq[0]), .dma_rsp_o(mrsp[0]), .hpm_req_i(mreq[1]), .hpm_rsp_o(mrsp[1]),
    .hp0_req_o(hp0_req), .hp0_rsp_i(hp0_rsp),
    .ub_rd_en_i(ub_rd_en), .ub_rd_addr_i(ub_rd_addr), .ub_rd_data_o(ub_rd_data),
    .ub_wr_en_i(ub_wr_en), .ub_wr_addr_i(ub_wr_addr), .ub_wr_data_i(ub_wr_data), .ub_wr_be_i(ub_wr_be),
    .acc_rd_en_i(acc_rd_en), .acc_rd_addr_i(acc_rd_addr), .acc_rd_data_o(acc_rd_data),
    .acc_wr_en_i(acc_wr_en), .acc_wr_addr_i(acc_wr_addr), .acc_wr_data_i(acc_wr_data), .acc_wr_be_i(acc_wr_be),
    .wf_valid_o(wf_valid), .wf_ready_i(wf_ready), .wf_data_o(wf_data),
    .if_valid_o(if_valid), .if_ready_i(if_ready), .if_data_o(if_data),
    .ub_owner_tpu_i(1'b0), .acc_owner_tpu_i(1'b0), .wf_dbg_pop_i(1'b0), .if_dbg_pop_i(1'b0),
    .wf_flush_i(wf_flush), .if_flush_i(if_flush), .err_clr_i(fab_err_clr),
    .err_o(fab_err), .wf_level_o(wf_level), .if_level_o(if_level)
  );

  logic run = 0, err_clr = 0, perf_clr = 0;
  logic busy, halted, cfg_err, irq;
  logic [7:0]  cfg_err_op;
  logic [31:0] irq_tag, debug_tag;
  logic [31:0] perf [8];

  tpu_core #(.N(N)) i_core (
    .clk_i(clk), .rst_ni(rst_n),
    .if_valid_i(if_valid), .if_ready_o(if_ready), .if_data_i(if_data),
    .wf_valid_i(wf_valid), .wf_ready_o(wf_ready), .wf_data_i(wf_data),
    .ub_rd_en_o(ub_rd_en), .ub_rd_addr_o(ub_rd_addr), .ub_rd_data_i(ub_rd_data),
    .ub_wr_en_o(ub_wr_en), .ub_wr_addr_o(ub_wr_addr), .ub_wr_data_o(ub_wr_data), .ub_wr_be_o(ub_wr_be),
    .acc_rd_en_o(acc_rd_en), .acc_rd_addr_o(acc_rd_addr), .acc_rd_data_i(acc_rd_data),
    .acc_wr_en_o(acc_wr_en), .acc_wr_addr_o(acc_wr_addr), .acc_wr_data_o(acc_wr_data), .acc_wr_be_o(acc_wr_be),
    .run_i(run), .err_clr_i(err_clr), .perf_clr_i(perf_clr),
    .busy_o(busy), .halted_o(halted), .cfg_err_o(cfg_err), .cfg_err_op_o(cfg_err_op),
    .irq_o(irq), .irq_tag_o(irq_tag), .debug_tag_o(debug_tag), .perf_o(perf),
    .dma_req_valid_o(), .dma_req_ready_i(1'b0), .dma_src_addr_o(), .dma_dst_addr_o(),
    .dma_length_o(), .dma_reps_o(), .dma_src_stride_o(), .dma_dst_stride_o(),
    .dma_rsp_valid_i(1'b0), .dma_rsp_ready_o(), .dma_rsp_error_i(1'b0),
    .dma_bus_error_i(1'b0), .dma_error_o()
  );

  int unsigned irq_cnt = 0;
  always @(posedge clk) if (rst_n && irq) irq_cnt++;

  // ---------------------------------------------------------------------------------------------
  // AXI4 single-beat master tasks (port 0 / 1)
  // ---------------------------------------------------------------------------------------------
  int unsigned errors = 0;
  task automatic fail(input string m); $error("%s", m); errors++; endtask

  task automatic axi_write(input int p, input logic [63:0] a, input data_t d, output axi_pkg::resp_t r);
    mreq[p].aw = '0; mreq[p].aw.addr = a; mreq[p].aw.size = 3'd4; mreq[p].aw.burst = axi_pkg::BURST_INCR;
    mreq[p].aw_valid = 1;
    mreq[p].w = '0; mreq[p].w.data = d; mreq[p].w.strb = '1; mreq[p].w.last = 1; mreq[p].w_valid = 1;
    mreq[p].b_ready = 1;
    fork
      begin do @(posedge clk); while (!mrsp[p].aw_ready); #TA mreq[p].aw_valid = 0; end
      begin do @(posedge clk); while (!mrsp[p].w_ready);  #TA mreq[p].w_valid  = 0; end
    join
    while (!mrsp[p].b_valid) @(posedge clk);
    r = mrsp[p].b.resp;
    @(posedge clk); #TA mreq[p].b_ready = 0;
  endtask
  task automatic axi_read(input int p, input logic [63:0] a, output data_t d);
    mreq[p].ar = '0; mreq[p].ar.addr = a; mreq[p].ar.size = 3'd4; mreq[p].ar.burst = axi_pkg::BURST_INCR;
    mreq[p].ar_valid = 1; mreq[p].r_ready = 1;
    do @(posedge clk); while (!mrsp[p].ar_ready);
    #TA mreq[p].ar_valid = 0;
    while (!mrsp[p].r_valid) @(posedge clk);
    d = mrsp[p].r.data;
    if (mrsp[p].r.resp != axi_pkg::RESP_OKAY) fail($sformatf("read %h resp %0d", a, mrsp[p].r.resp));
    @(posedge clk); #TA mreq[p].r_ready = 0;
  endtask
  task automatic wr(input int p, input logic [63:0] a, input data_t d);
    axi_pkg::resp_t r;
    axi_write(p, a, d, r);
    if (r != axi_pkg::RESP_OKAY) fail($sformatf("write %h resp %0d", a, r));
  endtask

  task automatic pulse(ref logic s); @(posedge clk); #TA s = 1; @(posedge clk); #TA s = 0; endtask

  // ---------------------------------------------------------------------------------------------
  // Case data
  // ---------------------------------------------------------------------------------------------
  logic [127:0] prog      [0:8191];
  logic [127:0] wmem      [0:65535];
  logic [127:0] ub_init   [0:65535];
  logic [127:0] ub_expect [0:65535];
  logic [511:0] acc_expect[0:2047];
  int unsigned  m_prog, m_wwords, m_init, m_out_first, m_out_rows, m_acc_rows, m_irq_exp, m_cfg_err;
  logic [31:0]  m_irq_tag;
  logic [7:0]   m_err_op;
  bit           has_acc;

  task automatic load_case(input string dir);
    int fd, rc;
    string k, v;
    fd = $fopen({dir, "/meta.txt"}, "r");
    if (fd == 0) begin fail({"cannot open ", dir, "/meta.txt"}); return; end
    m_acc_rows = 0; m_irq_exp = 0; m_cfg_err = 0; m_irq_tag = '0; m_err_op = '0;
    while (!$feof(fd)) begin
      rc = $fscanf(fd, "%s %s\n", k, v);
      if (rc != 2) continue;
      case (k)
        "prog_instrs":   m_prog      = v.atoi();
        "weight_words":  m_wwords    = v.atoi();
        "ub_init_rows":  m_init      = v.atoi();
        "out_first_row": m_out_first = v.atoi();
        "out_rows":      m_out_rows  = v.atoi();
        "acc_rows":      m_acc_rows  = v.atoi();
        "irq_expected":  m_irq_exp   = v.atoi();
        "irq_tag":       m_irq_tag   = v.atohex();
        "cfg_err":       m_cfg_err   = v.atoi();
        "cfg_err_op":    m_err_op    = v.atohex();
        default: ;
      endcase
    end
    $fclose(fd);
    $readmemh({dir, "/prog.hex"}, prog, 0, m_prog - 1);
    if (m_wwords > 0) $readmemh({dir, "/wmem.hex"}, wmem, 0, m_wwords - 1);
    if (m_init > 0)   $readmemh({dir, "/ub_init.hex"}, ub_init, 0, m_init - 1);
    if (m_out_rows > 0) $readmemh({dir, "/ub_expect.hex"}, ub_expect, 0, m_out_rows - 1);
    fd = $fopen({dir, "/acc_expect.hex"}, "r");
    has_acc = (fd != 0) && (m_acc_rows > 0);
    if (fd != 0) $fclose(fd);
    if (has_acc) $readmemh({dir, "/acc_expect.hex"}, acc_expect, 0, m_acc_rows - 1);
  endtask

  // ---------------------------------------------------------------------------------------------
  // One program run
  // ---------------------------------------------------------------------------------------------
  int unsigned n_pass = 0, n_cases = 0;

  task automatic run_case(input string dir, input string name);
    int unsigned bad_ub, bad_acc, t;
    bit pushed_p, pushed_w;
    data_t d;
    n_cases++;
    load_case(dir);
    // clean state (no reset between cases)
    #TA run = 0;
    pulse(wf_flush); pulse(if_flush); pulse(err_clr); pulse(fab_err_clr);
    for (int r = 0; r < m_init; r++) wr(1, UB_BASE + 64'(r) * 16, ub_init[r]);
    for (int r = 0; r < m_acc_rows; r++) for (int b = 0; b < 4; b++) wr(1, ACC_BASE + 64'(r) * 64 + 64'(b) * 16, '0);
    t = irq_cnt;
    pulse(perf_clr);                                   // counters measure the program only
    @(posedge clk); #TA run = 1;
    pushed_p = 0; pushed_w = 0;
    fork
      begin for (int i = 0; i < m_prog; i++)   wr(1, IFIFO_BASE + 64'(i % 4096) * 16, prog[i]); pushed_p = 1; end
      begin for (int i = 0; i < m_wwords; i++) wr(0, WFIFO_BASE + 64'(i % 65536) * 16, wmem[i]); pushed_w = 1; end
    join_none
    // wait for the end of the program
    begin
      int unsigned c = 0;
      while (c < 4_000_000 && !(m_cfg_err ? cfg_err : (irq_cnt != t))) begin @(posedge clk); c++; end
      if (c >= 4_000_000) begin
        fail($sformatf("%s: timeout (busy=%0b halted=%0b cfg_err=%0b if_lvl=%0d wf_lvl=%0d)",
                       name, busy, halted, cfg_err, if_level, wf_level));
        $display("TPU_CORE_FAIL"); $finish;
      end
      c = 0;
      while (c < 200_000 && (busy || !(pushed_p && pushed_w) && !m_cfg_err)) begin @(posedge clk); c++; end
      if (m_cfg_err) begin   // pushes may be left over; they finish into the FIFO (flushed next case)
        c = 0;
        while (c < 200_000 && !(pushed_p && pushed_w)) begin @(posedge clk); c++; end
        if (!(pushed_p && pushed_w)) begin fail({name, ": pushes stuck after cfg_err"}); $finish; end
      end
    end
    repeat (20) @(posedge clk);
    // checks
    if (m_cfg_err) begin
      if (!cfg_err) fail({name, ": cfg_err expected"});
      else if (cfg_err_op != m_err_op) fail($sformatf("%s: cfg_err_op %h exp %h", name, cfg_err_op, m_err_op));
      if (irq_cnt != t) fail({name, ": unexpected irq"});
    end else begin
      if (cfg_err) fail($sformatf("%s: unexpected cfg_err op=%h", name, cfg_err_op));
      if (m_irq_exp && irq_tag != m_irq_tag) fail($sformatf("%s: irq tag %h exp %h", name, irq_tag, m_irq_tag));
      if (!halted) fail({name, ": not halted after the program"});
    end
    bad_ub = 0;
    for (int r = 0; r < m_out_rows; r++) begin
      axi_read(1, UB_BASE + 64'(m_out_first + r) * 16, d);
      if (d !== ub_expect[r]) begin
        if (bad_ub < 4) $display("  %s UB row %0d: got %h exp %h", name, m_out_first + r, d, ub_expect[r]);
        bad_ub++;
      end
    end
    bad_acc = 0;
    if (has_acc) for (int r = 0; r < m_acc_rows; r++) begin
      logic [511:0] row;
      for (int b = 0; b < 4; b++) begin axi_read(1, ACC_BASE + 64'(r) * 64 + 64'(b) * 16, d); row[128*b +: 128] = d; end
      if (row !== acc_expect[r]) begin
        if (bad_acc < 3) $display("  %s ACC row %0d: got %h\n      exp %h", name, r, row, acc_expect[r]);
        bad_acc++;
      end
    end
    if (bad_ub || bad_acc) fail($sformatf("%s: %0d UB rows, %0d ACC rows wrong", name, bad_ub, bad_acc));
    else begin
      n_pass++;
      $display("[PASS] %-20s prog=%0d w=%0d out=%0d acc=%0d | total=%0d array=%0d wstall=%0d wshift=%0d nonmx=%0d sync=%0d act=%0d instr=%0d",
               name, m_prog, m_wwords, m_out_rows, m_acc_rows,
               perf[0], perf[1], perf[2], perf[3], perf[4], perf[5], perf[6], perf[7]);
    end
    #TA run = 0;
  endtask

  // ---------------------------------------------------------------------------------------------
  // Main
  // ---------------------------------------------------------------------------------------------
  initial begin
    string cases_file, only, line;
    string dirs[$];
    int fd;
    mreq[0] = '0; mreq[1] = '0;
    if (!$value$plusargs("CASES=%s", cases_file)) cases_file = "build/m4/cases/cases.txt";
    if (!$value$plusargs("ONLY=%s", only)) only = "";
    fd = $fopen(cases_file, "r");
    if (fd == 0) begin $display("TPU_CORE_FAIL no cases file %s", cases_file); $finish; end
    while ($fgets(line, fd)) begin
      string s;
      s = line;
      while (s.len() > 0 && (s[s.len()-1] == "\n" || s[s.len()-1] == "\r" || s[s.len()-1] == " ")) s = s.substr(0, s.len()-2);
      if (s.len() > 0) dirs.push_back(s);
    end
    $fclose(fd);

    repeat (5) @(posedge clk); #TA rst_n = 1; repeat (5) @(posedge clk);

    // all cases back to back, no reset (L14); MLP cases twice
    for (int pass = 0; pass < 2; pass++)
      foreach (dirs[i]) begin
        string name;
        name = dirs[i].substr(dirs[i].len() - 1, dirs[i].len() - 1);
        for (int j = dirs[i].len() - 1; j >= 0; j--) if (dirs[i][j] == "/") begin name = dirs[i].substr(j + 1, dirs[i].len() - 1); break; end
        if (only != "" && name != only) continue;
        if (pass == 1 && name.substr(0, 5) == "cfgerr") continue;
        run_case(dirs[i], name);
      end

    // reset in the middle of a run, then a clean re-run (L14)
    if (only == "") begin
      string d3;
      foreach (dirs[i]) if (dirs[i].len() >= 4 && dirs[i].substr(dirs[i].len() - 4, dirs[i].len() - 1) == "mlp3") d3 = dirs[i];
      if (d3 != "") begin
        load_case(d3);
        for (int r = 0; r < m_init; r++) wr(1, UB_BASE + 64'(r) * 16, ub_init[r]);
        @(posedge clk); #TA run = 1;
        fork
          begin for (int i = 0; i < m_prog; i++) wr(1, IFIFO_BASE + 64'(i) * 16, prog[i]); end
          begin for (int i = 0; i < m_wwords; i++) wr(0, WFIFO_BASE + 64'(i) * 16, wmem[i]); end
          begin repeat (3000) @(posedge clk); end
        join_any
        disable fork;
        $display("[INFO] reset during mlp3 at instr %0d (busy=%0b), re-running", perf[7], busy);
        #TA rst_n = 0; run = 0; mreq[0] = '0; mreq[1] = '0;
        repeat (5) @(posedge clk); #TA rst_n = 1; repeat (5) @(posedge clk);
        run_case(d3, "mlp3_after_reset");
      end
    end

    $display("tb_tpu_core: %0d/%0d case runs passed, errors=%0d", n_pass, n_cases, errors);
    if (errors == 0 && n_pass == n_cases) $display("TPU_CORE_PASS"); else $display("TPU_CORE_FAIL");
    $finish;
  end

endmodule
