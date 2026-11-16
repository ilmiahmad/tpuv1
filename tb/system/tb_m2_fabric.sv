// M2 - data fabric + on-chip memories (Xcelium).
//
//   AXI-Lite tasks -> idma_subsystem -+
//   HPM0 AXI4 tasks (CPU debug) ------+-> tpu_fabric -> axi_delayer (random stalls) -> axi_sim_mem (DDR)
//   native-port tasks (TPU side) -----+        +-> UB / ACC / Weight FIFO / Instruction FIFO
//
// A  DDR->UB->DDR via iDMA (1D/2D, random offsets), contents cross-checked through UB port B
// B  TPU writes UB (port B) -> iDMA UB->DDR
// C  ACC: iDMA write + native row read (bank mapping), native row write -> iDMA read
// D  Weight FIFO: stream in order with a randomly stalling consumer; overfill -> DMA stalls
//    until the consumer drains (backpressure, no loss)
// E  Instruction FIFO: HPM0 pushes, level peek, debug-pop loopback IFIFO->DDR via iDMA,
//    empty debug-pop -> SLVERR
// F  partial-strobe push -> SLVERR + sticky flag
// G  ownership: UB/ACC writes blocked (SLVERR + flag) while TPU-owned
// H  HPM0 -> DDR blocked, unmapped hole -> DECERR
// I  HPM0 debug read/write of UB and ACC
// Everything twice (L14) where data is involved. Plusarg +SEED=<n>.

`timescale 1ns/1ps
`include "axi/typedef.svh"

module tb_m2_fabric;
  import tpu_pkg::*;

  localparam time TCK = 10ns, TA = 2ns, TT = 8ns;
  // iDMA reg64_2d map (see docs/idma_integration.md)
  localparam logic [31:0] R_CONF = 32'h000, R_NEXT_ID = 32'h044, R_DONE_ID = 32'h084,
                          R_DST = 32'h0D0, R_SRC = 32'h0D8, R_LEN = 32'h0E0,
                          R_DSTRIDE = 32'h100, R_SSTRIDE = 32'h108, R_REPS = 32'h110;

  logic clk = 1'b0, rst_n = 1'b0;
  always #(TCK/2) clk = ~clk;

  // ---------------------------------------------------------------------------------------------
  // DUT
  // ---------------------------------------------------------------------------------------------
  tpu_m0_pkg::axil_req_t  cfg_req;
  tpu_m0_pkg::axil_resp_t cfg_rsp;
  fs_req_t  dma_req, hpm_req;
  fs_resp_t dma_rsp, hpm_rsp;
  fm_req_t  hp0_req, mem_req;
  fm_resp_t hp0_rsp, mem_rsp;
  logic [31:0] done_id;
  logic        dma_busy;
  logic [2:0]  dma_err;
  logic        err_clr = 1'b0;

  idma_subsystem #(
    .DataWidth(DataWidth), .AddrWidth(AddrWidth), .AxiIdWidth(SlvIdWidth), .UserWidth(UserWidth),
    .axi_req_t(fs_req_t), .axi_rsp_t(fs_resp_t),
    .axi_ar_chan_t(fs_ar_chan_t), .axi_aw_chan_t(fs_aw_chan_t),
    .axil_req_t(tpu_m0_pkg::axil_req_t), .axil_rsp_t(tpu_m0_pkg::axil_resp_t)
  ) i_idma (
    .clk_i(clk), .rst_ni(rst_n), .cfg_req_i(cfg_req), .cfg_rsp_o(cfg_rsp),
    .axi_req_o(dma_req), .axi_rsp_i(dma_rsp), .done_id_o(done_id), .busy_o(dma_busy),
    .err_o(dma_err), .err_clr_i(err_clr),
    .core_req_valid_i(1'b0), .core_req_ready_o(),
    .core_src_addr_i('0), .core_dst_addr_i('0), .core_length_i('0), .core_reps_i('0),
    .core_src_stride_i('0), .core_dst_stride_i('0),
    .core_rsp_valid_o(), .core_rsp_ready_i(1'b1), .core_rsp_error_o()
  );

  logic ub_rd_en = 0, ub_wr_en = 0;   logic [StrbWidth-1:0] ub_wr_be = '0;
  ub_addr_t ub_rd_addr = '0, ub_wr_addr = '0;  data_t ub_wr_data = '0, ub_rd_data;
  logic acc_rd_en = 0, acc_wr_en = 0; logic [AccBanks-1:0] acc_wr_be = '0;
  acc_addr_t acc_rd_addr = '0, acc_wr_addr = '0; acc_row_t acc_wr_data = '0, acc_rd_data;
  logic                 wf_valid, wf_ready = 0, if_valid, if_ready = 0;
  data_t                wf_data, if_data;
  logic ub_own = 0, acc_own = 0, wf_dbg = 0, if_dbg = 0, wf_flush = 0, if_flush = 0;
  logic [3:0]  fab_err;
  logic [15:0] wf_level, if_level;

  tpu_fabric i_fab (
    .clk_i(clk), .rst_ni(rst_n),
    .dma_req_i(dma_req), .dma_rsp_o(dma_rsp), .hpm_req_i(hpm_req), .hpm_rsp_o(hpm_rsp),
    .hp0_req_o(hp0_req), .hp0_rsp_i(hp0_rsp),
    .ub_rd_en_i(ub_rd_en), .ub_rd_addr_i(ub_rd_addr), .ub_rd_data_o(ub_rd_data),
    .ub_wr_en_i(ub_wr_en), .ub_wr_addr_i(ub_wr_addr), .ub_wr_data_i(ub_wr_data), .ub_wr_be_i(ub_wr_be),
    .acc_rd_en_i(acc_rd_en), .acc_rd_addr_i(acc_rd_addr), .acc_rd_data_o(acc_rd_data),
    .acc_wr_en_i(acc_wr_en), .acc_wr_addr_i(acc_wr_addr), .acc_wr_data_i(acc_wr_data), .acc_wr_be_i(acc_wr_be),
    .wf_valid_o(wf_valid), .wf_ready_i(wf_ready), .wf_data_o(wf_data),
    .if_valid_o(if_valid), .if_ready_i(if_ready), .if_data_o(if_data),
    .ub_owner_tpu_i(ub_own), .acc_owner_tpu_i(acc_own), .wf_dbg_pop_i(wf_dbg), .if_dbg_pop_i(if_dbg),
    .wf_flush_i(wf_flush), .if_flush_i(if_flush), .err_clr_i(err_clr),
    .err_o(fab_err), .wf_level_o(wf_level), .if_level_o(if_level)
  );

  axi_delayer #(
    .aw_chan_t(fm_aw_chan_t), .w_chan_t(fm_w_chan_t), .b_chan_t(fm_b_chan_t),
    .ar_chan_t(fm_ar_chan_t), .r_chan_t(fm_r_chan_t), .axi_req_t(fm_req_t), .axi_resp_t(fm_resp_t),
    .StallRandomInput(1'b1), .StallRandomOutput(1'b1), .FixedDelayInput(0), .FixedDelayOutput(0)
  ) i_delay (
    .clk_i(clk), .rst_ni(rst_n), .slv_req_i(hp0_req), .slv_resp_o(hp0_rsp),
    .mst_req_o(mem_req), .mst_resp_i(mem_rsp)
  );

  axi_sim_mem #(
    .AddrWidth(AddrWidth), .DataWidth(DataWidth), .IdWidth(MstIdWidth), .UserWidth(UserWidth),
    .axi_req_t(fm_req_t), .axi_rsp_t(fm_resp_t),
    .WarnUninitialized(1'b0), .ClearErrOnAccess(1'b1), .ApplDelay(TA), .AcqDelay(TT)
  ) i_ddr (
    .clk_i(clk), .rst_ni(rst_n), .axi_req_i(mem_req), .axi_rsp_o(mem_rsp),
    .mon_r_last_o(), .mon_r_beat_count_o(), .mon_r_user_o(), .mon_r_id_o(),
    .mon_r_data_o(), .mon_r_addr_o(), .mon_r_valid_o(),
    .mon_w_last_o(), .mon_w_beat_count_o(), .mon_w_user_o(), .mon_w_id_o(),
    .mon_w_data_o(), .mon_w_addr_o(), .mon_w_valid_o()
  );

  // HP0 rules carried over from M0/M1
  a_l2: assert property (@(posedge clk) disable iff (!rst_n)
      (hp0_req.ar_valid && hp0_req.ar.len != 0) |-> hp0_req.ar.size == 3'd4)
    else $error("L2: narrow read burst on HP0");
  a_hp0_only_ddr: assert property (@(posedge clk) disable iff (!rst_n)
      hp0_req.ar_valid |-> (hp0_req.ar.addr < DDR_LO_END || hp0_req.ar.addr >= DDR_HI_BASE))
    else $error("non-DDR address on HP0: %h", hp0_req.ar.addr);

  // ---------------------------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------------------------
  int unsigned errors = 0, checks = 0, seed = 1;
  task automatic fail(input string m); $error("%s", m); errors++; endtask
  task automatic pass(input string m); checks++; $display("[PASS] %s", m); endtask

  task automatic lite_write(input logic [31:0] a, input logic [31:0] d);
    cfg_req.aw.addr = a; cfg_req.aw.prot = '0; cfg_req.aw_valid = 1;
    cfg_req.w.data = d; cfg_req.w.strb = '1; cfg_req.w_valid = 1; cfg_req.b_ready = 1;
    fork
      begin do @(posedge clk); while (!cfg_rsp.aw_ready); #TA cfg_req.aw_valid = 0; end
      begin do @(posedge clk); while (!cfg_rsp.w_ready);  #TA cfg_req.w_valid  = 0; end
    join
    while (!cfg_rsp.b_valid) @(posedge clk);
    @(posedge clk); #TA cfg_req.b_ready = 0;
  endtask
  task automatic lite_read(input logic [31:0] a, output logic [31:0] d);
    cfg_req.ar.addr = a; cfg_req.ar.prot = '0; cfg_req.ar_valid = 1; cfg_req.r_ready = 1;
    do @(posedge clk); while (!cfg_rsp.ar_ready);
    #TA cfg_req.ar_valid = 0;
    while (!cfg_rsp.r_valid) @(posedge clk);
    d = cfg_rsp.r.data;
    @(posedge clk); #TA cfg_req.r_ready = 0;
  endtask
  task automatic w64(input logic [31:0] a, input logic [63:0] v); lite_write(a, v[31:0]); lite_write(a+4, v[63:32]); endtask

  // iDMA copy; returns completion. Does not wait if wait_done = 0 (for backpressure tests).
  task automatic dma(input logic [63:0] src, dst, len, input int unsigned reps = 1,
                     input logic [63:0] ss = 0, ds = 0, output logic [31:0] id);
    w64(R_SRC, src); w64(R_DST, dst); w64(R_LEN, len);
    w64(R_SSTRIDE, ss); w64(R_DSTRIDE, ds); w64(R_REPS, 64'(reps));
    lite_write(R_CONF, reps > 1 ? 32'h400 : 32'h0);
    lite_read(R_NEXT_ID, id);
  endtask
  task automatic dma_wait(input logic [31:0] id, input string name, input int unsigned max = 400000);
    logic [31:0] d; int unsigned n;
    n = 0;
    do begin lite_read(R_DONE_ID, d); n++; end while ($signed(d - id) < 0 && n < max);
    if ($signed(d - id) < 0) fail($sformatf("%s: DMA id %0d timeout (done=%0d)", name, id, d));
  endtask
  task automatic dma_run(input string name, input logic [63:0] src, dst, len,
                         input int unsigned reps = 1, input logic [63:0] ss = 0, ds = 0);
    logic [31:0] id;
    dma(src, dst, len, reps, ss, ds, id);
    dma_wait(id, name);
  endtask

  // HPM0 single-beat 128-bit accesses (CPU debug path)
  task automatic hpm_write(input logic [63:0] a, input data_t d, input strb_t s, output axi_pkg::resp_t r);
    hpm_req.aw = '0; hpm_req.aw.addr = a; hpm_req.aw.size = 3'd4; hpm_req.aw.burst = axi_pkg::BURST_INCR;
    hpm_req.aw_valid = 1;
    hpm_req.w = '0; hpm_req.w.data = d; hpm_req.w.strb = s; hpm_req.w.last = 1; hpm_req.w_valid = 1;
    hpm_req.b_ready = 1;
    fork
      begin do @(posedge clk); while (!hpm_rsp.aw_ready); #TA hpm_req.aw_valid = 0; end
      begin do @(posedge clk); while (!hpm_rsp.w_ready);  #TA hpm_req.w_valid  = 0; end
    join
    while (!hpm_rsp.b_valid) @(posedge clk);
    r = hpm_rsp.b.resp;
    @(posedge clk); #TA hpm_req.b_ready = 0;
  endtask
  task automatic hpm_read(input logic [63:0] a, output data_t d, output axi_pkg::resp_t r);
    hpm_req.ar = '0; hpm_req.ar.addr = a; hpm_req.ar.size = 3'd4; hpm_req.ar.burst = axi_pkg::BURST_INCR;
    hpm_req.ar_valid = 1; hpm_req.r_ready = 1;
    do @(posedge clk); while (!hpm_rsp.ar_ready);
    #TA hpm_req.ar_valid = 0;
    while (!hpm_rsp.r_valid) @(posedge clk);
    d = hpm_rsp.r.data; r = hpm_rsp.r.resp;
    @(posedge clk); #TA hpm_req.r_ready = 0;
  endtask

  // native ports
  task automatic ub_native_write(input ub_addr_t a, input data_t d);
    @(posedge clk); #TA ub_wr_en = 1; ub_wr_be = '1; ub_wr_addr = a; ub_wr_data = d;
    @(posedge clk); #TA ub_wr_en = 0; ub_wr_be = '0;
  endtask
  task automatic ub_native_read(input ub_addr_t a, output data_t d);
    @(posedge clk); #TA ub_rd_en = 1; ub_rd_addr = a;
    @(posedge clk); #TA ub_rd_en = 0;
    repeat (UbLatency - 1) @(posedge clk);
    #1 d = ub_rd_data;                                     // latency UbLatency
  endtask
  task automatic acc_native_write(input acc_addr_t a, input acc_row_t d);
    @(posedge clk); #TA acc_wr_en = 1; acc_wr_be = '1; acc_wr_addr = a; acc_wr_data = d;
    @(posedge clk); #TA acc_wr_en = 0; acc_wr_be = '0;
  endtask
  task automatic acc_native_read(input acc_addr_t a, output acc_row_t d);
    @(posedge clk); #TA acc_rd_en = 1; acc_rd_addr = a;
    @(posedge clk); #1 d = acc_rd_data;                  // latency 1
    #(TA-1) acc_rd_en = 0;
  endtask

  task automatic clear_errs(); @(posedge clk); #TA err_clr = 1; @(posedge clk); #TA err_clr = 0; endtask

  function automatic byte unsigned pat(input int unsigned s, input int unsigned i);
    return byte'(s * 37 + i * 11 + (i >> 7));
  endfunction
  task automatic fill_ddr(input logic [63:0] a, input int unsigned n, input int unsigned s);
    for (int unsigned i = 0; i < n; i++) i_ddr.mem[a + i] = pat(s, i);
  endtask
  task automatic check_ddr(input string name, input logic [63:0] a, input int unsigned n, input int unsigned s);
    int unsigned bad;
    bad = 0;
    for (int unsigned i = 0; i < n; i++) if (i_ddr.mem[a + i] !== pat(s, i)) bad++;
    if (bad) fail($sformatf("%s: %0d/%0d bytes wrong", name, bad, n)); else pass(name);
  endtask

  // ---------------------------------------------------------------------------------------------
  // Tests
  // ---------------------------------------------------------------------------------------------
  initial begin
    logic [31:0] id;
    data_t d;
    acc_row_t row;
    axi_pkg::resp_t r;
    if ($value$plusargs("SEED=%d", seed)) ;
    void'($urandom(seed));
    cfg_req = '0; hpm_req = '0;
    repeat (5) @(posedge clk); #TA rst_n = 1; repeat (5) @(posedge clk);

    for (int run = 0; run < 2; run++) begin
      // ---------------- A: DDR -> UB -> DDR ----------------
      for (int t = 0; t < 6; t++) begin
        int unsigned len, off, s;
        logic [63:0] src, dst;
        len = $urandom_range(1, 6000);
        off = $urandom_range(0, (1 << 20) - len - 1);
        s   = run * 100 + t;
        src = 64'h3000_0000 + 64'(t) * 64'h10_0000 + $urandom_range(0, 15);
        dst = 64'h4000_0000 + 64'(t) * 64'h10_0000 + $urandom_range(0, 15);
        fill_ddr(src, len, s);
        dma_run("A in",  src, UB_BASE + off, len);
        dma_run("A out", UB_BASE + off, dst, len);
        check_ddr($sformatf("A run%0d: DDR->UB->DDR %0d B @UB+0x%0h", run, len, off), dst, len, s);
      end
      // UB contents as seen by the TPU port (16-byte aligned block)
      fill_ddr(64'h3800_0000, 256, 7 + run);
      dma_run("A2 in", 64'h3800_0000, UB_BASE + 64'h8_0000, 256);
      begin
        int unsigned bad;
        bad = 0;
        for (int rw = 0; rw < 16; rw++) begin
          ub_native_read(ub_addr_t'(16'h8000 + rw), d);
          for (int b = 0; b < 16; b++) if (d[b*8 +: 8] !== pat(7 + run, rw*16 + b)) bad++;
        end
        if (bad) fail($sformatf("A2: UB port B sees %0d wrong bytes", bad));
        else pass($sformatf("A2 run%0d: UB rows written by iDMA read back on the native port (byte order L12)", run));
      end
      // 2D into UB
      fill_ddr(64'h3900_0000, 8 * 512, 9 + run);
      dma_run("A3 in 2D",  64'h3900_0000, UB_BASE + 64'h4_0000, 100, 8, 512, 128);
      dma_run("A3 out 2D", UB_BASE + 64'h4_0000, 64'h4900_0000, 100, 8, 128, 512);
      begin
        int unsigned bad;
        bad = 0;
        for (int rr = 0; rr < 8; rr++) for (int i = 0; i < 100; i++)
          if (i_ddr.mem[64'h4900_0000 + rr*512 + i] !== pat(9 + run, rr*512 + i)) bad++;
        if (bad) fail($sformatf("A3 2D: %0d bad", bad)); else pass($sformatf("A3 run%0d: 2D strided DDR->UB->DDR", run));
      end

      // ---------------- B: TPU writes UB -> iDMA reads ----------------
      for (int rw = 0; rw < 32; rw++) begin
        for (int b = 0; b < 16; b++) d[b*8 +: 8] = pat(50 + run, rw*16 + b);
        ub_native_write(ub_addr_t'(16'hF000 + rw), d);
      end
      dma_run("B out", UB_BASE + 64'hF_0000, 64'h4A00_0000, 512);
      check_ddr($sformatf("B run%0d: native UB write -> iDMA UB->DDR", run), 64'h4A00_0000, 512, 50 + run);

      // ---------------- C: Accumulators ----------------
      fill_ddr(64'h3A00_0000, 64 * 8, 60 + run);
      dma_run("C in", 64'h3A00_0000, ACC_BASE + 64'h40 * 100, 64 * 8);   // rows 100..107
      begin
        int unsigned bad;
        bad = 0;
        for (int rw = 0; rw < 8; rw++) begin
          acc_native_read(acc_addr_t'(100 + rw), row);
          for (int b = 0; b < 64; b++) if (row[b*8 +: 8] !== pat(60 + run, rw*64 + b)) bad++;
        end
        if (bad) fail($sformatf("C1: ACC native rows %0d bad bytes", bad));
        else pass($sformatf("C1 run%0d: iDMA->ACC, native 512-bit row read (bank k = bytes 16k..16k+15)", run));
      end
      for (int rw = 0; rw < 8; rw++) begin
        for (int b = 0; b < 64; b++) row[b*8 +: 8] = pat(70 + run, rw*64 + b);
        acc_native_write(acc_addr_t'(2000 + rw), row);
      end
      dma_run("C out", ACC_BASE + 64'h40 * 2000, 64'h4B00_0000, 64 * 8);
      check_ddr($sformatf("C2 run%0d: native ACC row write -> iDMA ACC->DDR", run), 64'h4B00_0000, 512, 70 + run);

      // ---------------- D: Weight FIFO streaming with a stalling consumer ----------------
      begin
        int unsigned n_words, got, bad;
        n_words = 256;                                   // 16 tiles = 4 KiB
        fill_ddr(64'h3B00_0000, n_words * 16, 80 + run);
        got = 0; bad = 0;
        fork
          dma_run("D push", 64'h3B00_0000, WFIFO_BASE, n_words * 16);
          while (got < n_words) begin
            @(posedge clk); #TA wf_ready = ($urandom_range(0, 2) != 0);
            @(negedge clk);
            if (wf_valid && wf_ready) begin
              for (int b = 0; b < 16; b++) if (wf_data[b*8 +: 8] !== pat(80 + run, got*16 + b)) bad++;
              got++;
            end
          end
        join
        @(posedge clk); #TA wf_ready = 0;          // let the last sampled handshake complete
        if (bad || wf_level != 0) fail($sformatf("D1: %0d bad bytes, level %0d", bad, wf_level));
        else pass($sformatf("D1 run%0d: 256 words DDR->WFIFO, popped in order by a stalling consumer", run));
      end
      begin  // overfill: 1280 words > capacity; DMA must stall, then finish when drained
        int unsigned n_words, got, bad;
        logic [31:0] d2;
        n_words = 1280;
        fill_ddr(64'h3C00_0000, n_words * 16, 90 + run);
        dma(64'h3C00_0000, WFIFO_BASE, n_words * 16, 1, 0, 0, id);
        repeat (20000) @(posedge clk);
        lite_read(R_DONE_ID, d2);
        if ($signed(d2 - id) >= 0) fail("D2: DMA completed although the FIFO is full and nobody pops");
        else if (wf_level < WfifoWords) fail($sformatf("D2: FIFO not full while DMA blocked (level %0d)", wf_level));
        else pass($sformatf("D2 run%0d: FIFO full (level %0d) -> DMA held by backpressure", run, wf_level));
        got = 0; bad = 0;
        #TA wf_ready = 1;
        while (got < n_words) begin
          @(negedge clk);
          if (wf_valid && wf_ready) begin
            for (int b = 0; b < 16; b++) if (wf_data[b*8 +: 8] !== pat(90 + run, got*16 + b)) bad++;
            got++;
          end
        end
        @(posedge clk); #TA wf_ready = 0;          // let the last sampled handshake complete
        dma_wait(id, "D2 drain");
        if (bad) fail($sformatf("D2: %0d bad bytes after drain", bad));
        else pass($sformatf("D2 run%0d: drained 1280 words in order, DMA completed, no loss", run));
      end

      // ---------------- E: Instruction FIFO ----------------
      for (int k = 0; k < 8; k++) begin
        for (int b = 0; b < 16; b++) d[b*8 +: 8] = pat(100 + run, k*16 + b);
        hpm_write(IFIFO_BASE + 64'(k) * 16, d, '1, r);
        if (r != axi_pkg::RESP_OKAY) fail("E: HPM0 instruction push not OKAY");
      end
      hpm_read(IFIFO_BASE, d, r);
      if (r != axi_pkg::RESP_OKAY || d[15:0] != 16'd8 || if_level != 8)
        fail($sformatf("E1: level peek resp=%0d data=%0d level=%0d", r, d[15:0], if_level));
      else pass($sformatf("E1 run%0d: 8 instructions pushed by HPM0, level peek = 8 without popping", run));
      #TA if_dbg = 1;
      dma_run("E2 loopback", IFIFO_BASE, 64'h4C00_0000, 8 * 16);
      check_ddr($sformatf("E2 run%0d: debug-pop loopback IFIFO->DDR in order", run), 64'h4C00_0000, 128, 100 + run);
      hpm_read(IFIFO_BASE, d, r);
      if (r != axi_pkg::RESP_SLVERR || !fab_err[3]) fail("E3: empty debug pop not SLVERR/flagged");
      else pass("E3: debug pop of an empty FIFO -> SLVERR + flag, no stall");
      #TA if_dbg = 0;
      clear_errs();

      // ---------------- F: partial-strobe push ----------------
      hpm_write(WFIFO_BASE, '1, 16'h00FF, r);
      if (r != axi_pkg::RESP_SLVERR || !fab_err[2] || wf_level != 0)
        fail($sformatf("F: partial push resp=%0d err=%b level=%0d", r, fab_err, wf_level));
      else pass("F: partial-strobe push dropped, SLVERR, sticky flag");
      clear_errs();

      // ---------------- G: ownership ----------------
      hpm_write(UB_BASE + 64'h100, {8{16'hCAFE}}, '1, r);
      #TA ub_own = 1; acc_own = 1;
      hpm_write(UB_BASE + 64'h100, '0, '1, r);
      if (r != axi_pkg::RESP_SLVERR || !fab_err[0]) fail("G1: UB write while TPU-owned not blocked");
      hpm_write(ACC_BASE + 64'h40, '0, '1, r);
      if (r != axi_pkg::RESP_SLVERR || !fab_err[1]) fail("G2: ACC write while TPU-owned not blocked");
      hpm_read(UB_BASE + 64'h100, d, r);                 // reads stay allowed
      if (r != axi_pkg::RESP_OKAY || d !== {8{16'hCAFE}}) fail("G3: UB data changed or read blocked");
      #TA ub_own = 0; acc_own = 0;
      clear_errs();
      if (fab_err == 0) pass($sformatf("G run%0d: TPU-owned UB/ACC reject AXI writes (SLVERR + flags), reads allowed", run));

      // ---------------- H: HPM0 must not reach DDR; holes DECERR ----------------
      hpm_read(64'h0000_0000_3000_0000, d, r);
      if (r != axi_pkg::RESP_DECERR) fail($sformatf("H1: HPM0 -> DDR resp=%0d", r));
      hpm_read(64'h0000_0000_A050_0000, d, r);
      if (r != axi_pkg::RESP_DECERR) fail($sformatf("H2: hole resp=%0d", r));
      else pass("H: HPM0->DDR and unmapped hole answer DECERR");

      // ---------------- I: HPM0 debug view of UB/ACC matches iDMA/native views ----------------
      hpm_read(UB_BASE + 64'h8_0000, d, r);               // written by iDMA in A2
      begin
        int unsigned bad;
        bad = 0;
        for (int b = 0; b < 16; b++) if (d[b*8 +: 8] !== pat(7 + run, b)) bad++;
        acc_native_read(acc_addr_t'(100), row);
        hpm_read(ACC_BASE + 64'h40 * 100 + 64'h30, d, r);  // bank 3 of row 100
        if (d !== row[3*128 +: 128]) bad++;
        if (bad) fail("I: HPM0 debug view mismatch"); else pass($sformatf("I run%0d: HPM0 debug reads of UB/ACC match", run));
      end
    end

    if (dma_err != 0) fail($sformatf("iDMA error flags set: %b", dma_err));
    $display("tb_m2_fabric: checks=%0d errors=%0d", checks, errors);
    if (errors == 0) $display("M2_FABRIC_PASS"); else $display("M2_FABRIC_FAIL errors=%0d", errors);
    $finish;
  end

  initial begin #300ms $display("M2_FABRIC_FAIL global timeout"); $finish; end

endmodule
