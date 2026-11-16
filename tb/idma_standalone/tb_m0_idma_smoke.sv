// M0 smoke test for idma_subsystem (reg64_2d + nd_midend + rw_axi + transfer_id_gen).
// Programs the frontend over AXI4-Lite exactly as the Linux runtime will, copies bytes inside a
// PULP axi_sim_mem, and checks every byte. Not the M1 regression; just proof-of-life before
// the board. Covers: >4 GiB source, misaligned src/dst, 4097-byte length (crosses 4 KiB),
// 2D strided copy, two back-to-back runs per case (L14), next_id launch/ID contract.
// SVA (L2): every multi-beat read burst on the master port must be full-width (16 B/beat).

`timescale 1ns/1ps
`include "axi/typedef.svh"

module tb_m0_idma_smoke;
  import tpu_m0_pkg::*;

  localparam time TCK = 10ns;
  localparam time TA  = 2ns;
  localparam time TT  = 8ns;

  // reg64_2d register map (idma_reg64_2d_addrmap_pkg)
  localparam logic [31:0] R_CONF    = 32'h000;
  localparam logic [31:0] R_STATUS0 = 32'h004;
  localparam logic [31:0] R_NEXT_ID = 32'h044;
  localparam logic [31:0] R_DONE_ID = 32'h084;
  localparam logic [31:0] R_DST     = 32'h0D0;
  localparam logic [31:0] R_SRC     = 32'h0D8;
  localparam logic [31:0] R_LEN     = 32'h0E0;
  localparam logic [31:0] R_DSTRIDE = 32'h100;
  localparam logic [31:0] R_SSTRIDE = 32'h108;
  localparam logic [31:0] R_REPS    = 32'h110;
  localparam logic [31:0] CONF_2D   = 32'h400;   // enable_nd = 1 (bits [11:10])

  logic clk = 1'b0, rst_n = 1'b0;
  always #(TCK/2) clk = ~clk;

  axil_req_t  cfg_req;
  axil_resp_t cfg_rsp;
  axi_req_t   axi_req;
  axi_resp_t  axi_rsp;
  logic [31:0] done_id;
  logic        busy;

  idma_subsystem #(
    .DataWidth(DataWidth), .AddrWidth(AddrWidth), .AxiIdWidth(AxiIdWidth), .UserWidth(UserWidth),
    .axi_req_t(axi_req_t), .axi_rsp_t(axi_resp_t),
    .axi_ar_chan_t(axi_ar_chan_t), .axi_aw_chan_t(axi_aw_chan_t),
    .axil_req_t(axil_req_t), .axil_rsp_t(axil_resp_t)
  ) i_dut (
    .clk_i(clk), .rst_ni(rst_n),
    .cfg_req_i(cfg_req), .cfg_rsp_o(cfg_rsp),
    .axi_req_o(axi_req), .axi_rsp_i(axi_rsp),
    .done_id_o(done_id), .busy_o(busy), .err_o(), .err_clr_i(1'b0),
    .core_req_valid_i(1'b0), .core_req_ready_o(),
    .core_src_addr_i('0), .core_dst_addr_i('0), .core_length_i('0), .core_reps_i('0),
    .core_src_stride_i('0), .core_dst_stride_i('0),
    .core_rsp_valid_o(), .core_rsp_ready_i(1'b1), .core_rsp_error_o()
  );

  axi_sim_mem #(
    .AddrWidth(AddrWidth), .DataWidth(DataWidth), .IdWidth(AxiIdWidth), .UserWidth(UserWidth),
    .axi_req_t(axi_req_t), .axi_rsp_t(axi_resp_t),
    .WarnUninitialized(1'b0), .ClearErrOnAccess(1'b1), .ApplDelay(TA), .AcqDelay(TT)
  ) i_mem (
    .clk_i(clk), .rst_ni(rst_n), .axi_req_i(axi_req), .axi_rsp_o(axi_rsp),
    .mon_r_last_o(), .mon_r_beat_count_o(), .mon_r_user_o(), .mon_r_id_o(),
    .mon_r_data_o(), .mon_r_addr_o(), .mon_r_valid_o(),
    .mon_w_last_o(), .mon_w_beat_count_o(), .mon_w_user_o(), .mon_w_id_o(),
    .mon_w_data_o(), .mon_w_addr_o(), .mon_w_valid_o()
  );

  // L2: no narrow multi-beat read bursts on the HP0-facing port
  property p_full_width_reads;
    @(posedge clk) disable iff (!rst_n)
      (axi_req.ar_valid && axi_req.ar.len != 0) |-> (axi_req.ar.size == 3'd4);
  endproperty
  a_full_width_reads: assert property (p_full_width_reads)
    else $error("L2 violation: narrow read burst size=%0d len=%0d addr=%h",
                axi_req.ar.size, axi_req.ar.len, axi_req.ar.addr);

  int unsigned aw_count = 0, ar_count = 0;
  always @(posedge clk) begin
    if (axi_req.aw_valid && axi_rsp.aw_ready) aw_count++;
    if (axi_req.ar_valid && axi_rsp.ar_ready) ar_count++;
  end

  // ------------------------------------------------------------------------------------------
  // AXI4-Lite master tasks
  // ------------------------------------------------------------------------------------------
  int unsigned errors = 0;

  task automatic lite_write(input logic [31:0] addr, input logic [31:0] data);
    cfg_req.aw.addr = addr; cfg_req.aw.prot = '0; cfg_req.aw_valid = 1'b1;
    cfg_req.w.data  = data; cfg_req.w.strb  = '1; cfg_req.w_valid  = 1'b1;
    cfg_req.b_ready = 1'b1;
    fork
      begin do @(posedge clk); while (!cfg_rsp.aw_ready); #TA cfg_req.aw_valid = 1'b0; end
      begin do @(posedge clk); while (!cfg_rsp.w_ready);  #TA cfg_req.w_valid  = 1'b0; end
    join
    while (!cfg_rsp.b_valid) @(posedge clk);
    if (cfg_rsp.b.resp != axi_pkg::RESP_OKAY) begin
      $error("write %h: resp %0d", addr, cfg_rsp.b.resp); errors++;
    end
    @(posedge clk); #TA cfg_req.b_ready = 1'b0;
  endtask

  task automatic lite_read(input logic [31:0] addr, output logic [31:0] data);
    cfg_req.ar.addr = addr; cfg_req.ar.prot = '0; cfg_req.ar_valid = 1'b1; cfg_req.r_ready = 1'b1;
    do @(posedge clk); while (!cfg_rsp.ar_ready);
    #TA cfg_req.ar_valid = 1'b0;
    while (!cfg_rsp.r_valid) @(posedge clk);
    data = cfg_rsp.r.data;
    if (cfg_rsp.r.resp != axi_pkg::RESP_OKAY) begin
      $error("read %h: resp %0d", addr, cfg_rsp.r.resp); errors++;
    end
    @(posedge clk); #TA cfg_req.r_ready = 1'b0;
  endtask

  task automatic write64(input logic [31:0] addr, input logic [63:0] v);
    lite_write(addr,     v[31:0]);
    lite_write(addr + 4, v[63:32]);
  endtask

  // ------------------------------------------------------------------------------------------
  // One transfer: program, launch by reading next_id once, wait for done_id, check bytes
  // ------------------------------------------------------------------------------------------
  task automatic run_copy(input string name, input logic [63:0] src, input logic [63:0] dst,
                          input int unsigned len, input int unsigned reps,
                          input logic [63:0] sstride, input logic [63:0] dstride,
                          input byte unsigned seed);
    logic [31:0] id, d;
    int unsigned bad = 0, cycles = 0;
    // stimulus + clear destination (with a guard byte on each side)
    for (int unsigned r = 0; r < reps; r++) begin
      for (int unsigned i = 0; i < len; i++) begin
        i_mem.mem[src + r*sstride + i] = byte'(seed + 7*r + 13*i + (i >> 8));
        i_mem.mem[dst + r*dstride + i] = 8'hEE;
      end
      i_mem.mem[dst + r*dstride - 1]   = 8'hA5;
      i_mem.mem[dst + r*dstride + len] = 8'h5A;
    end
    write64(R_SRC, src);
    write64(R_DST, dst);
    write64(R_LEN, 64'(len));
    write64(R_SSTRIDE, sstride);
    write64(R_DSTRIDE, dstride);
    write64(R_REPS, 64'(reps));
    lite_write(R_CONF, (reps > 1) ? CONF_2D : 32'h0);
    lite_read(R_NEXT_ID, id);                       // side effect: launch (exactly once)
    if (id == 0) begin $error("%s: next_id returned 0 (rejected)", name); errors++; return; end
    do begin lite_read(R_DONE_ID, d); cycles++; end
    while ($signed(d - id) < 0 && cycles < 20000);
    if (cycles >= 20000) begin $error("%s: timeout, done_id=%0d id=%0d", name, d, id); errors++; end
    for (int unsigned r = 0; r < reps; r++) begin
      for (int unsigned i = 0; i < len; i++)
        if (i_mem.mem[dst + r*dstride + i] !== byte'(seed + 7*r + 13*i + (i >> 8))) bad++;
      if (i_mem.mem[dst + r*dstride - 1] !== 8'hA5 || i_mem.mem[dst + r*dstride + len] !== 8'h5A)
        begin $error("%s: row %0d guard byte overwritten", name, r); errors++; end
    end
    if (bad) begin $error("%s: %0d bad bytes", name, bad); errors++; end
    else $display("[PASS] %-34s id=%0d len=%0d reps=%0d src=%h dst=%h", name, id, len, reps, src, dst);
  endtask

  initial begin
    logic [31:0] st;
    cfg_req = '0;
    repeat (5) @(posedge clk);
    #TA rst_n = 1'b1;
    repeat (5) @(posedge clk);

    for (int run = 0; run < 2; run++) begin   // L14: everything twice, back to back
      run_copy($sformatf("1D aligned 256B (run %0d)", run),
               64'h0000_0000_3000_0000, 64'h0000_0000_3100_0000, 256, 1, 0, 0, 8'h11 + run);
      run_copy($sformatf("1D >4GiB misaligned 4097B (run %0d)", run),
               64'h0000_0008_0000_1003, 64'h0000_0000_3200_0011, 4097, 1, 0, 0, 8'h22 + run);
      run_copy($sformatf("2D 8 rows x 100B, strides (run %0d)", run),
               64'h0000_0000_3300_0005, 64'h0000_0000_3400_0000, 100, 8, 256, 128, 8'h33 + run);
    end

    // Zero-length: the docs claim next_id returns 0. The v0.7.4 RTL instead issues an ID and
    // the backend (RejectZeroTransfers=1) retires it without bus traffic. Check THAT contract.
    begin
      logic [31:0] zid, d;
      int unsigned aw0, ar0, n;
      aw0 = aw_count; ar0 = ar_count; n = 0;   // static block vars: assign, don't initialize
      write64(R_LEN, 64'd0);
      lite_write(R_CONF, 32'h0);
      lite_read(R_NEXT_ID, zid);
      do begin lite_read(R_DONE_ID, d); n++; end while ($signed(d - zid) < 0 && n < 1000);
      if (zid == 0) $display("[INFO] zero-length: next_id=0 (rejected at frontend)");
      else if (n >= 1000) begin $error("zero-length id=%0d never completed", zid); errors++; end
      else if (aw_count != aw0 || ar_count != ar0) begin
        $error("zero-length transfer caused bus traffic"); errors++; end
      else $display("[PASS] zero-length: id=%0d issued, retired with no AXI traffic", zid);
    end

    lite_read(R_STATUS0, st);
    $display("status0 = %h, busy = %b", st, busy);
    if (errors == 0) $display("M0_IDMA_SMOKE_PASS");
    else             $display("M0_IDMA_SMOKE_FAIL errors=%0d", errors);
    $finish;
  end

  initial begin #5ms $error("global timeout"); $display("M0_IDMA_SMOKE_FAIL timeout"); $finish; end

endmodule
