// M1 - iDMA subsystem regression (Xcelium). Closes the idma_integration.md checklist.
//
//   tb AXI4-Lite master -> idma_subsystem -> axi_delayer (random stalls, L16) -> axi_sim_mem
//
// Tests: T1 directed length/alignment boundaries, T2 random 1D/2D (incl. >4 GiB), T3 1 MiB,
// T4 SLVERR/DECERR injection + midend error, T5 invalid protocol, T6 length overflow (>TFLen),
// T7 reset mid-transfer + recovery, T8 ID wrap (run with ID_W=4).
// SVA on the DUT master port: L2 (no narrow multi-beat reads), no 4 KiB crossing, AW/AR
// stable while valid (AXI handshake rule).
// Plusargs: +SEED=<n>  +NRAND=<n>

`timescale 1ns/1ps
`include "axi/typedef.svh"

module tb_m1_idma #(
  parameter int unsigned ID_W  = 32,
  parameter bit          STALL = 1'b1
);
  import tpu_m0_pkg::*;

  localparam time TCK = 10ns, TA = 2ns, TT = 8ns;
  localparam logic [31:0] R_CONF = 32'h000, R_STATUS0 = 32'h004, R_NEXT_ID = 32'h044,
                          R_DONE_ID = 32'h084, R_DST = 32'h0D0, R_SRC = 32'h0D8, R_LEN = 32'h0E0,
                          R_DSTRIDE = 32'h100, R_SSTRIDE = 32'h108, R_REPS = 32'h110;
  localparam logic [31:0] CONF_2D = 32'h400;

  logic clk = 1'b0, rst_n = 1'b0;       // PL reset (DUT + delayer)
  logic mem_rst_n = 1'b0;                // PS side: NOT reset by a PL reset
  always #(TCK/2) clk = ~clk;

  axil_req_t  cfg_req;
  axil_resp_t cfg_rsp;
  axi_req_t   dut_req, mem_req;
  axi_resp_t  dut_rsp, mem_rsp;
  logic [31:0] done_id;
  logic        busy;
  logic [2:0]  err;
  logic        err_clr = 1'b0;

  idma_subsystem #(
    .DataWidth(DataWidth), .AddrWidth(AddrWidth), .AxiIdWidth(AxiIdWidth), .UserWidth(UserWidth),
    .IdCounterWidth(ID_W),
    .axi_req_t(axi_req_t), .axi_rsp_t(axi_resp_t),
    .axi_ar_chan_t(axi_ar_chan_t), .axi_aw_chan_t(axi_aw_chan_t),
    .axil_req_t(axil_req_t), .axil_rsp_t(axil_resp_t)
  ) i_dut (
    .clk_i(clk), .rst_ni(rst_n), .cfg_req_i(cfg_req), .cfg_rsp_o(cfg_rsp),
    .axi_req_o(dut_req), .axi_rsp_i(dut_rsp), .done_id_o(done_id), .busy_o(busy),
    .err_o(err), .err_clr_i(err_clr),
    .core_req_valid_i(1'b0), .core_req_ready_o(),
    .core_src_addr_i('0), .core_dst_addr_i('0), .core_length_i('0), .core_reps_i('0),
    .core_src_stride_i('0), .core_dst_stride_i('0),
    .core_rsp_valid_o(), .core_rsp_ready_i(1'b1), .core_rsp_error_o()
  );

  axi_delayer #(
    .aw_chan_t(axi_aw_chan_t), .w_chan_t(axi_w_chan_t), .b_chan_t(axi_b_chan_t),
    .ar_chan_t(axi_ar_chan_t), .r_chan_t(axi_r_chan_t),
    .axi_req_t(axi_req_t), .axi_resp_t(axi_resp_t),
    .StallRandomInput(STALL), .StallRandomOutput(STALL),
    .FixedDelayInput(STALL ? 0 : 1), .FixedDelayOutput(STALL ? 0 : 1)
  ) i_delay (
    .clk_i(clk), .rst_ni(rst_n), .slv_req_i(dut_req), .slv_resp_o(dut_rsp),
    .mst_req_o(mem_req), .mst_resp_i(mem_rsp)
  );

  axi_sim_mem #(
    .AddrWidth(AddrWidth), .DataWidth(DataWidth), .IdWidth(AxiIdWidth), .UserWidth(UserWidth),
    .axi_req_t(axi_req_t), .axi_rsp_t(axi_resp_t),
    .WarnUninitialized(1'b0), .ClearErrOnAccess(1'b1), .ApplDelay(TA), .AcqDelay(TT)
  ) i_mem (
    .clk_i(clk), .rst_ni(mem_rst_n), .axi_req_i(mem_req), .axi_rsp_o(mem_rsp),
    .mon_r_last_o(), .mon_r_beat_count_o(), .mon_r_user_o(), .mon_r_id_o(),
    .mon_r_data_o(), .mon_r_addr_o(), .mon_r_valid_o(),
    .mon_w_last_o(), .mon_w_beat_count_o(), .mon_w_user_o(), .mon_w_id_o(),
    .mon_w_data_o(), .mon_w_addr_o(), .mon_w_valid_o()
  );

  // ------------------------------------------------------------------------------------------
  // Assertions on the DUT master port (what HP0 will see)
  // ------------------------------------------------------------------------------------------
  function automatic bit crosses_4k(input logic [63:0] a, input logic [7:0] len, input logic [2:0] sz);
    logic [63:0] last = (a & ~((64'd1 << sz) - 1)) + ((64'(len)) << sz);
    return (a[63:12] != last[63:12]);
  endfunction

  a_l2_narrow_read: assert property (@(posedge clk) disable iff (!rst_n)
      (dut_req.ar_valid && dut_req.ar.len != 0) |-> dut_req.ar.size == 3'd4)
    else $error("L2: narrow read burst size=%0d len=%0d addr=%h", dut_req.ar.size, dut_req.ar.len, dut_req.ar.addr);
  a_ar_4k: assert property (@(posedge clk) disable iff (!rst_n)
      dut_req.ar_valid |-> !crosses_4k(dut_req.ar.addr, dut_req.ar.len, dut_req.ar.size))
    else $error("AR crosses 4 KiB: addr=%h len=%0d", dut_req.ar.addr, dut_req.ar.len);
  a_aw_4k: assert property (@(posedge clk) disable iff (!rst_n)
      dut_req.aw_valid |-> !crosses_4k(dut_req.aw.addr, dut_req.aw.len, dut_req.aw.size))
    else $error("AW crosses 4 KiB: addr=%h len=%0d", dut_req.aw.addr, dut_req.aw.len);
  a_ar_stable: assert property (@(posedge clk) disable iff (!rst_n)
      (dut_req.ar_valid && !dut_rsp.ar_ready) |=> (dut_req.ar_valid && $stable(dut_req.ar)))
    else $error("AR changed/dropped while valid && !ready");
  a_aw_stable: assert property (@(posedge clk) disable iff (!rst_n)
      (dut_req.aw_valid && !dut_rsp.aw_ready) |=> (dut_req.aw_valid && $stable(dut_req.aw)))
    else $error("AW changed/dropped while valid && !ready");

  int unsigned n_ar = 0, n_aw = 0, max_arlen = 0;
  always @(posedge clk) begin
    if (dut_req.ar_valid && dut_rsp.ar_ready) begin n_ar++; if (dut_req.ar.len > max_arlen) max_arlen = dut_req.ar.len; end
    if (dut_req.aw_valid && dut_rsp.aw_ready) n_aw++;
  end

  // ------------------------------------------------------------------------------------------
  // AXI4-Lite master
  // ------------------------------------------------------------------------------------------
  int unsigned errors = 0, passes = 0;
  task automatic fail(input string m); $error("%s", m); errors++; endtask

  task automatic lite_write(input logic [31:0] addr, input logic [31:0] data);
    cfg_req.aw.addr = addr; cfg_req.aw.prot = '0; cfg_req.aw_valid = 1'b1;
    cfg_req.w.data  = data; cfg_req.w.strb  = '1; cfg_req.w_valid  = 1'b1;
    cfg_req.b_ready = 1'b1;
    fork
      begin do @(posedge clk); while (!cfg_rsp.aw_ready); #TA cfg_req.aw_valid = 1'b0; end
      begin do @(posedge clk); while (!cfg_rsp.w_ready);  #TA cfg_req.w_valid  = 1'b0; end
    join
    while (!cfg_rsp.b_valid) @(posedge clk);
    if (cfg_rsp.b.resp != axi_pkg::RESP_OKAY) fail($sformatf("cfg write %h resp %0d", addr, cfg_rsp.b.resp));
    @(posedge clk); #TA cfg_req.b_ready = 1'b0;
  endtask

  task automatic lite_read(input logic [31:0] addr, output logic [31:0] data);
    cfg_req.ar.addr = addr; cfg_req.ar.prot = '0; cfg_req.ar_valid = 1'b1; cfg_req.r_ready = 1'b1;
    do @(posedge clk); while (!cfg_rsp.ar_ready);
    #TA cfg_req.ar_valid = 1'b0;
    while (!cfg_rsp.r_valid) @(posedge clk);
    data = cfg_rsp.r.data;
    if (cfg_rsp.r.resp != axi_pkg::RESP_OKAY) fail($sformatf("cfg read %h resp %0d", addr, cfg_rsp.r.resp));
    @(posedge clk); #TA cfg_req.r_ready = 1'b0;
  endtask

  task automatic write64(input logic [31:0] a, input logic [63:0] v);
    lite_write(a, v[31:0]); lite_write(a + 4, v[63:32]);
  endtask

  // wrap-aware "done has reached id" for an ID_W-bit counter
  function automatic bit reached(input logic [31:0] done, input logic [31:0] id);
    logic [ID_W-1:0] diff = ID_W'(done - id);
    return diff[ID_W-1] == 1'b0;
  endfunction

  // ------------------------------------------------------------------------------------------
  // Launch + wait. Returns id (0 = rejected) and whether it completed.
  // ------------------------------------------------------------------------------------------
  task automatic launch_wait(input logic [63:0] src, dst, len, input int unsigned reps,
                             input logic [63:0] ss, ds, input logic [31:0] conf,
                             output logic [31:0] id, output bit completed,
                             input int unsigned max_polls = 200000);
    logic [31:0] d;
    int unsigned n;
    n = 0;
    wait_idle();
    write64(R_SRC, src); write64(R_DST, dst); write64(R_LEN, len);
    write64(R_SSTRIDE, ss); write64(R_DSTRIDE, ds); write64(R_REPS, 64'(reps));
    lite_write(R_CONF, conf);
    lite_read(R_NEXT_ID, id);
    completed = 1'b0;
    if (id == 0) return;
    do begin lite_read(R_DONE_ID, d); n++; end while (!reached(d, id) && n < max_polls);
    completed = reached(d, id);
  endtask

  task automatic wait_idle();
    int unsigned n;
    n = 0;
    while (busy && n < 5_000_000) begin @(posedge clk); n++; end
    if (busy) fail("DMA still busy before next test");
  endtask

  // fill src rows, poison dst rows + guards, run, check
  task automatic copy_check(input string name, input logic [63:0] src, dst,
                            input int unsigned len, reps, input logic [63:0] ss, ds,
                            input int unsigned seed, input bit verbose = 1'b0,
                            input int unsigned max_polls = 200000);
    logic [31:0] id; bit done;
    int unsigned bad;
    bad = 0;
    for (int unsigned r = 0; r < reps; r++) begin
      for (int unsigned i = 0; i < len; i++) begin
        i_mem.mem[src + r*ss + i] = byte'(seed * 131 + r * 31 + i * 7 + (i >> 9));
        i_mem.mem[dst + r*ds + i] = 8'hEE;
      end
      i_mem.mem[dst + r*ds - 1]   = 8'hA5;
      i_mem.mem[dst + r*ds + len] = 8'h5A;
    end
    launch_wait(src, dst, 64'(len), reps, ss, ds, (reps > 1) ? CONF_2D : 32'h0, id, done, max_polls);
    if (id == 0)  begin fail($sformatf("%s: rejected (next_id=0)", name)); return; end
    if (!done)    begin fail($sformatf("%s: timeout id=%0d", name, id)); return; end
    for (int unsigned r = 0; r < reps; r++) begin
      for (int unsigned i = 0; i < len; i++)
        if (i_mem.mem[dst + r*ds + i] !== byte'(seed * 131 + r * 31 + i * 7 + (i >> 9))) bad++;
      if (i_mem.mem[dst + r*ds - 1] !== 8'hA5 || i_mem.mem[dst + r*ds + len] !== 8'h5A) bad++;
    end
    if (bad) fail($sformatf("%s: %0d bad bytes (src=%h dst=%h len=%0d reps=%0d ss=%0d ds=%0d)",
                            name, bad, src, dst, len, reps, ss, ds));
    else begin
      passes++;
      if (verbose) $display("[PASS] %s", name);
    end
  endtask

  task automatic clear_err();
    @(posedge clk); #TA err_clr = 1'b1; @(posedge clk); #TA err_clr = 1'b0;
  endtask

  // ------------------------------------------------------------------------------------------
  // Tests
  // ------------------------------------------------------------------------------------------
  int unsigned seed = 1, nrand = 300;

  initial begin
    logic [31:0] id, st; bit done;
    if ($value$plusargs("SEED=%d", seed)) ;
    if ($value$plusargs("NRAND=%d", nrand)) ;
    void'($urandom(seed));
    $display("tb_m1_idma: ID_W=%0d STALL=%0b SEED=%0d NRAND=%0d", ID_W, STALL, seed, nrand);
    cfg_req = '0;
    repeat (5) @(posedge clk); #TA rst_n = 1'b1; mem_rst_n = 1'b1; repeat (5) @(posedge clk);

    if (ID_W < 32) begin
      // ---------------- T8: ID wrap (small counter) ----------------
      int unsigned zero_ids;
      zero_ids = 0;
      for (int k = 0; k < 3 * (1 << ID_W); k++) begin
        write64(R_SRC, 64'h3000_0000); write64(R_DST, 64'h3100_0000 + 64'(k) * 64);
        write64(R_LEN, 64'd32); write64(R_REPS, 64'd1); lite_write(R_CONF, 0);
        for (int i = 0; i < 32; i++) i_mem.mem[64'h3000_0000 + i] = byte'(k + i);
        lite_read(R_NEXT_ID, id);
        if (id == 0) begin zero_ids++; continue; end
        begin
          logic [31:0] d; int unsigned n;
          n = 0;
          do begin lite_read(R_DONE_ID, d); n++; end while (!reached(d, id) && n < 10000);
          if (!reached(d, id)) fail($sformatf("T8: id %0d never completed (done=%0d)", id, d));
        end
        for (int i = 0; i < 32; i++)
          if (i_mem.mem[64'h3100_0000 + 64'(k) * 64 + i] !== byte'(k + i))
            begin fail($sformatf("T8: transfer %0d (id %0d) data wrong", k, id)); break; end
      end
      $display("[INFO] T8: %0d launches over %0d-bit IDs; next_id returned 0 %0d time(s) (=> launch REJECTED, data copy skipped)",
               3 * (1 << ID_W), ID_W, zero_ids);
      if (zero_ids != 0)
        $display("[FINDING] T8: when the ID counter wraps, next_id can be 0 for a VALID config; software cannot use 0 as the reject code without checking");
      finish();
    end

    // ---------------- T1: directed boundaries ----------------
    begin
      int unsigned lens[] = '{1, 2, 15, 16, 17, 255, 256, 257, 4095, 4096, 4097, 8192 + 9, 65536 + 5};
      int unsigned offs[] = '{0, 1, 15};
      int k;
      k = 0;
      foreach (lens[a]) foreach (offs[b]) foreach (offs[c]) begin
        copy_check($sformatf("T1 len=%0d so=%0d do=%0d", lens[a], offs[b], offs[c]),
                   64'h1000_0000 + 64'(k) * 64'h2_0000 + offs[b],
                   64'h2000_0000 + 64'(k) * 64'h2_0000 + 64'h10 + offs[c],
                   lens[a], 1, 0, 0, k);
        k++;
      end
      // starts just before a 4 KiB boundary
      copy_check("T1 src 4K-1", 64'h1500_0FFF, 64'h2500_0000, 64, 1, 0, 0, 77, 1);
      copy_check("T1 dst 4K-1", 64'h1501_0000, 64'h2501_0FFF, 64, 1, 0, 0, 78, 1);
      $display("[INFO] T1: %0d directed copies done, max AR len seen = %0d beats", k + 2, max_arlen + 1);
    end

    // ---------------- T2: random 1D/2D, back-to-back ----------------
    for (int unsigned t = 0; t < nrand; t++) begin
      int unsigned len, reps;
      logic [63:0] ss, ds, src, dst;
      bit hi;
      len  = $urandom_range(1, 3000);
      reps = ($urandom_range(0, 2) == 0) ? 1 : $urandom_range(2, 8);
      ss   = 64'(len) + $urandom_range(0, 300);
      ds   = 64'(len) + 2 + $urandom_range(0, 300);
      hi   = $urandom_range(0, 3) == 0;
      src  = (hi ? 64'h8_0000_0000 : 64'h4000_0000) + 64'(t) * 64'h10_0000 + $urandom_range(0, 4095);
      dst  = 64'h6000_0000 + 64'(t) * 64'h10_0000 + 1 + $urandom_range(0, 4095);
      copy_check($sformatf("T2 #%0d", t), src, dst, len, reps, ss, ds, t + 1000);
    end
    $display("[INFO] T2: %0d random 1D/2D copies (passes so far %0d)", nrand, passes);

    // ---------------- T3: 1 MiB + 3 ----------------
    copy_check("T3 1 MiB + 3 B", 64'h7000_0003, 64'h7800_0000, (1 << 20) + 3, 1, 0, 0, 3, 1, 5_000_000);

    // ---------------- T4: error injection ----------------
    clear_err();
    if (err != 0) fail("T4: err not clear at start");
    // read SLVERR on one source byte
    i_mem.rerr[64'h1A00_0040] = axi_pkg::RESP_SLVERR;
    launch_wait(64'h1A00_0000, 64'h2A00_0000, 64'd256, 1, 0, 0, 0, id, done, 20000);
    $display("[INFO] T4a read SLVERR : id=%0d completed=%0b err=%03b", id, done, err);
    if (!done)    fail("T4a: transfer with a read SLVERR did not complete (hang)");
    if (!err[0])  fail("T4a: read error not flagged");
    clear_err();
    // write DECERR on one destination byte
    i_mem.werr[64'h2B00_0080] = axi_pkg::RESP_DECERR;
    launch_wait(64'h1B00_0000, 64'h2B00_0000, 64'd256, 1, 0, 0, 0, id, done, 20000);
    $display("[INFO] T4b write DECERR: id=%0d completed=%0b err=%03b", id, done, err);
    if (!done)    fail("T4b: transfer with a write DECERR did not complete (hang)");
    if (!err[1])  fail("T4b: write error not flagged");
    clear_err();
    // 2D with reps = 0
    launch_wait(64'h1C00_0000, 64'h2C00_0000, 64'd64, 0, 64, 64, CONF_2D, id, done, 20000);
    $display("[INFO] T4c 2D reps=0   : id=%0d completed=%0b err=%03b aw=%0d", id, done, err, n_aw);
    if (id != 0 && !done) fail("T4c: reps=0 transfer never completed");
    clear_err();
    copy_check("T4d clean copy after errors", 64'h1D00_0005, 64'h2D00_0003, 1000, 1, 0, 0, 44, 1);
    if (err != 0) fail($sformatf("T4d: spurious err=%03b", err));
    else $display("[PASS] T4: SLVERR/DECERR complete and are flagged; clean copy after clear");

    // ---------------- T5: invalid protocol (OBI on rw_axi) ----------------
    begin
      int unsigned aw0, ar0;
      aw0 = n_aw; ar0 = n_ar;
      // src_protocol field starts at bit 12 for reg64_2d ([11:10] = enable_nd); OBI = 1
      launch_wait(64'h1E00_0000, 64'h2E00_0000, 64'd64, 1, 0, 0, 32'h1 << 12, id, done, 2000);
      $display("[INFO] T5 src_protocol=OBI: next_id=%0d completed=%0b AR+%0d AW+%0d err=%03b",
               id, done, n_ar - ar0, n_aw - aw0, err);
      if (id != 0 && !done) $display("[FINDING] T5: unsupported protocol accepted and wedged the DMA");
      clear_err();
    end

    // ---------------- T6: length beyond TFLenWidth (24 bit) ----------------
    begin
      i_mem.mem[64'h2F00_0010] = 8'h5A;    // guard right after the low 16 bytes
      for (int i = 0; i < 16; i++) i_mem.mem[64'h1F00_0000 + i] = byte'(i + 1);
      launch_wait(64'h1F00_0000, 64'h2F00_0000, 64'h0100_0010, 1, 0, 0, 0, id, done, 20000);
      $display("[INFO] T6 length=2^24+16: id=%0d completed=%0b dst[15]=%h dst[16]=%h",
               id, done, i_mem.mem[64'h2F00_000F], i_mem.mem[64'h2F00_0010]);
      if (done && i_mem.mem[64'h2F00_0010] === 8'h5A)
        $display("[FINDING] T6: lengths >= 2^TFLenWidth are silently truncated (only len mod 2^24 copied); the runtime must split transfers >= 16 MiB");
    end

    // ---------------- T7a: PL soft reset while idle (supported) ----------------
    begin
      wait_idle();
      #TA rst_n = 1'b0; repeat (4) @(posedge clk); #TA rst_n = 1'b1; repeat (4) @(posedge clk);
      lite_read(R_DONE_ID, st);
      if (busy) fail("T7a: busy after idle reset");
      copy_check("T7a copy after idle reset (1D)", 64'h5100_0001, 64'h5900_0002, 3000, 1, 0, 0, 71, 1);
      copy_check("T7a copy after idle reset (2D)", 64'h5200_0000, 64'h5A00_0000, 200, 5, 512, 256, 72, 1);
      $display("[INFO] T7a: after reset done_id restarted at %0d", st);
    end

    // ---------------- T7b: PL reset in the middle of a burst (documents the hazard) -------------
    begin
      logic [31:0] id2;
      int unsigned stuck;
      wait_idle();
      write64(R_SRC, 64'h5000_0000); write64(R_DST, 64'h5800_0000); write64(R_LEN, 64'd65536);
      write64(R_REPS, 64'd1); lite_write(R_CONF, 0);
      lite_read(R_NEXT_ID, id2);
      wait (dut_req.w_valid && dut_rsp.w_ready && !dut_req.w.last);   // mid-W-burst
      @(posedge clk); #TA rst_n = 1'b0;                                  // PL only; PS keeps state
      repeat (4) @(posedge clk); #TA rst_n = 1'b1;
      stuck = 0;
      // the PS side still owes/expects beats of the interrupted burst
      repeat (2000) @(posedge clk);
      $display("[FINDING] T7b: a PL reset during a W burst leaves the PS-side slave mid-burst (sim-mem expects more W beats); on the board this can wedge S_AXI_HP0. Soft reset must wait for busy==0 (or isolate+drain).");
    end
    finish();
  end

  task automatic finish();
    $display("tb_m1_idma: passes=%0d errors=%0d ARs=%0d AWs=%0d max_arlen=%0d", passes, errors, n_ar, n_aw, max_arlen + 1);
    if (errors == 0) $display("M1_IDMA_PASS");
    else             $display("M1_IDMA_FAIL errors=%0d", errors);
    $finish;
  endtask

  initial begin #200ms $display("M1_IDMA_FAIL global timeout"); $finish; end

endmodule
