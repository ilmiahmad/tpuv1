// Phase B transport regression: real register frontend + ND midend + rw_axi backend.
// The scoreboard follows client acceptance, independently of the DUT source FIFO.
`timescale 1ns/1ps

module tb_m5_dma #(
  parameter int unsigned ID_W = 4,
  parameter int unsigned SLOTS = 4
);
  import tpu_m0_pkg::*;
  localparam time TCK = 10ns;
  logic clk = 0, rst_n = 0;
  always #(TCK/2) clk = ~clk;
  axil_req_t cfg_req;
  axil_resp_t cfg_rsp;
  axi_req_t dma_req, delayed_req, mem_req;
  axi_resp_t dma_rsp, delayed_rsp, mem_rsp;
  logic [31:0] done_id;
  logic busy, err_clr = 0;
  logic [2:0] err;
  logic core_valid = 0, core_ready, rsp_valid, rsp_error, rsp_ready;
  logic [63:0] src, dst, ss, ds;
  logic [23:0] length;
  logic [31:0] reps;
  logic allow_responses = 1, allow_core_completion = 1, random_core_ready = 0;
  logic random_ready_q = 1;
  bit random_request_gaps = 0;
  always @(negedge clk) random_ready_q = ($urandom_range(0, 3) != 0);
  assign rsp_ready = allow_core_completion && (!random_core_ready || random_ready_q);

  idma_subsystem #(
    .DataWidth(DataWidth), .AddrWidth(AddrWidth), .AxiIdWidth(AxiIdWidth), .UserWidth(UserWidth),
    .IdCounterWidth(ID_W), .EnableCore(1), .CoreOutstanding(SLOTS),
    .axi_req_t(axi_req_t), .axi_rsp_t(axi_resp_t),
    .axi_ar_chan_t(axi_ar_chan_t), .axi_aw_chan_t(axi_aw_chan_t),
    .axil_req_t(axil_req_t), .axil_rsp_t(axil_resp_t)
  ) dut (
    .clk_i(clk), .rst_ni(rst_n), .cfg_req_i(cfg_req), .cfg_rsp_o(cfg_rsp),
    .axi_req_o(dma_req), .axi_rsp_i(dma_rsp), .done_id_o(done_id), .busy_o(busy),
    .err_o(err), .err_clr_i(err_clr),
    .core_req_valid_i(core_valid), .core_req_ready_o(core_ready),
    .core_src_addr_i(src), .core_dst_addr_i(dst), .core_length_i(length), .core_reps_i(reps),
    .core_src_stride_i(ss), .core_dst_stride_i(ds),
    .core_rsp_valid_o(rsp_valid), .core_rsp_ready_i(rsp_ready), .core_rsp_error_o(rsp_error)
  );

  axi_delayer #(
    .aw_chan_t(axi_aw_chan_t), .w_chan_t(axi_w_chan_t), .b_chan_t(axi_b_chan_t),
    .ar_chan_t(axi_ar_chan_t), .r_chan_t(axi_r_chan_t),
    .axi_req_t(axi_req_t), .axi_resp_t(axi_resp_t),
    .StallRandomInput(1), .StallRandomOutput(1)
  ) i_delay (
    .clk_i(clk), .rst_ni(rst_n), .slv_req_i(dma_req), .slv_resp_o(dma_rsp),
    .mst_req_o(delayed_req), .mst_resp_i(delayed_rsp)
  );
  // Freeze response channels only between drained phases, to deterministically
  // fill the outstanding queue. Request channels continue obeying AXI.
  always_comb begin
    mem_req = delayed_req;
    mem_req.r_ready = delayed_req.r_ready && allow_responses;
    mem_req.b_ready = delayed_req.b_ready && allow_responses;
    delayed_rsp = mem_rsp;
    delayed_rsp.r_valid = mem_rsp.r_valid && allow_responses;
    delayed_rsp.b_valid = mem_rsp.b_valid && allow_responses;
  end
  axi_sim_mem #(
    .AddrWidth(AddrWidth), .DataWidth(DataWidth), .IdWidth(AxiIdWidth), .UserWidth(UserWidth),
    .axi_req_t(axi_req_t), .axi_rsp_t(axi_resp_t),
    .WarnUninitialized(0), .ClearErrOnAccess(1), .ApplDelay(2ns), .AcqDelay(8ns)
  ) i_mem (
    .clk_i(clk), .rst_ni(rst_n), .axi_req_i(mem_req), .axi_rsp_o(mem_rsp),
    .mon_r_last_o(), .mon_r_beat_count_o(), .mon_r_user_o(), .mon_r_id_o(),
    .mon_r_data_o(), .mon_r_addr_o(), .mon_r_valid_o(),
    .mon_w_last_o(), .mon_w_beat_count_o(), .mon_w_user_o(), .mon_w_id_o(),
    .mon_w_data_o(), .mon_w_addr_o(), .mon_w_valid_o()
  );

  typedef struct packed {
    bit core;
    logic [63:0] src, dst, ss, ds;
    logic [23:0] length;
    logic [31:0] reps;
  } desc_t;
  desc_t expected[$];
  int host_accepted = 0, core_accepted = 0, host_completed = 0, core_completed = 0;
  int max_pending = 0, competitions = 0, host_wait_grants = 0, core_wait_grants = 0;
  int ar_count = 0, aw_count = 0;
  int seed = 1;
  logic [ID_W-1:0] expected_done, expected_next;

  function automatic logic [ID_W-1:0] advance_id(input logic [ID_W-1:0] value, input bit event_);
    if (&value) return event_ ? ID_W'(2) : ID_W'(1);
    return value + ID_W'(event_);
  endfunction

  task automatic check_copy(input desc_t d);
    if (d.length == 0 || d.reps == 0) return;
    for (int unsigned r = 0; r < d.reps; r++) begin
      for (int unsigned i = 0; i < d.length; i++)
        assert (i_mem.mem[d.dst + 64'(r)*d.ds + 64'(i)] ===
                i_mem.mem[d.src + 64'(r)*d.ss + 64'(i)])
          else $fatal(1, "copy mismatch client=%0d dst=%h row=%0d byte=%0d", d.core, d.dst, r, i);
      assert (i_mem.mem[d.dst + 64'(r)*d.ds - 1] == 8'ha5 &&
              i_mem.mem[d.dst + 64'(r)*d.ds + 64'(d.length)] == 8'h5a)
        else $fatal(1, "guard overwritten client=%0d dst=%h row=%0d", d.core, d.dst, r);
    end
  endtask

  // Observe the existing frontend's request handshake as the Linux client boundary.
  // Expected completions and IDs do not inspect the arbiter's tags or head.
  always @(posedge clk) begin : scoreboard
    desc_t d;
    bit host_issue, core_issue, host_retire, core_retire;
    host_issue = dut.fe_req_valid && dut.fe_req_ready;
    core_issue = core_valid && core_ready;
    host_retire = dut.host_retire;
    core_retire = rsp_valid && rsp_ready;
    if (!rst_n) begin
      expected.delete(); expected_done = ID_W'(1); expected_next = ID_W'(2);
    end else begin
      assert (done_id == 32'(expected_done)) else $fatal(1, "Linux done_id changed on wrong source");
      assert (dut.next_id == expected_next) else $fatal(1, "Linux next_id changed on wrong source");
      expected_done = advance_id(expected_done, host_retire);
      expected_next = advance_id(expected_next, host_issue);
      assert (!(host_issue && core_issue)) else $fatal(1, "two request captures in one cycle");
      if (host_issue) begin
        d = '{core:0, src:dut.fe_req.burst_req.src_addr, dst:dut.fe_req.burst_req.dst_addr,
              ss:dut.fe_req.d_req[0].src_strides, ds:dut.fe_req.d_req[0].dst_strides,
              length:dut.fe_req.burst_req.length, reps:dut.fe_req.d_req[0].reps[31:0]};
        expected.push_back(d); host_accepted++;
      end
      if (core_issue) begin
        d = '{core:1, src:src, dst:dst, ss:ss, ds:ds, length:length, reps:reps};
        expected.push_back(d); core_accepted++;
      end
      if (host_issue || core_issue) begin
        if (dut.fe_req_valid && core_valid) competitions++;
        if (dut.fe_req_valid && !host_issue) host_wait_grants++;
        else host_wait_grants = 0;
        if (core_valid && !core_issue) core_wait_grants++;
        else core_wait_grants = 0;
        assert (host_wait_grants <= 1 && core_wait_grants <= 1)
          else $fatal(1, "round-robin starvation");
      end
      if (!dut.fe_req_valid) host_wait_grants = 0;
      if (!core_valid) core_wait_grants = 0;
      if (host_retire || core_retire) begin
        assert (!(host_retire && core_retire) && expected.size() > 0)
          else $fatal(1, "duplicate/orphan completion");
        d = expected.pop_front();
        assert (d.core == core_retire) else $fatal(1, "completion delivered to wrong source");
        if (core_retire) begin
          assert (rsp_error == (d.length == 0 || d.reps == 0))
            else $fatal(1, "wrong core response error");
          core_completed++;
        end else host_completed++;
        check_copy(d);
      end
      if (expected.size() > max_pending) max_pending = expected.size();
      assert (expected.size() <= SLOTS) else $fatal(1, "source capacity exceeded");
      if (expected.size() != 0) assert (busy) else $fatal(1, "busy dropped with outstanding work");
      if (dma_req.ar_valid && dma_rsp.ar_ready) ar_count++;
      if (dma_req.aw_valid && dma_rsp.aw_ready) aw_count++;
    end
  end

  a_core_response_stable: assert property (@(posedge clk) disable iff (!rst_n)
    rsp_valid && !rsp_ready |=> rsp_valid && $stable(rsp_error));
  a_core_request_stable: assert property (@(posedge clk) disable iff (!rst_n)
    core_valid && !core_ready |=> core_valid && $stable({src,dst,length,reps,ss,ds}));
  a_ar_stable: assert property (@(posedge clk) disable iff (!rst_n)
    dma_req.ar_valid && !dma_rsp.ar_ready |=> dma_req.ar_valid && $stable(dma_req.ar));
  a_aw_stable: assert property (@(posedge clk) disable iff (!rst_n)
    dma_req.aw_valid && !dma_rsp.aw_ready |=> dma_req.aw_valid && $stable(dma_req.aw));

  task automatic lite_write(input logic [31:0] addr, data);
    @(negedge clk);
    cfg_req.aw.addr = addr; cfg_req.aw_valid = 1;
    cfg_req.w.data = data; cfg_req.w.strb = '1; cfg_req.w_valid = 1; cfg_req.b_ready = 1;
    fork
      begin do @(posedge clk); while (!cfg_rsp.aw_ready); @(negedge clk); cfg_req.aw_valid = 0; end
      begin do @(posedge clk); while (!cfg_rsp.w_ready); @(negedge clk); cfg_req.w_valid = 0; end
    join
    do @(posedge clk); while (!cfg_rsp.b_valid);
    assert (cfg_rsp.b.resp == axi_pkg::RESP_OKAY) else $fatal(1, "cfg write error");
    @(negedge clk); cfg_req.b_ready = 0;
  endtask
  task automatic lite_read(input logic [31:0] addr, output logic [31:0] data);
    @(negedge clk); cfg_req.ar.addr = addr; cfg_req.ar_valid = 1; cfg_req.r_ready = 1;
    do @(posedge clk); while (!cfg_rsp.ar_ready);
    @(negedge clk); cfg_req.ar_valid = 0;
    do @(posedge clk); while (!cfg_rsp.r_valid);
    data = cfg_rsp.r.data;
    assert (cfg_rsp.r.resp == axi_pkg::RESP_OKAY) else $fatal(1, "cfg read error");
    @(negedge clk); cfg_req.r_ready = 0;
  endtask
  task automatic write64(input logic [31:0] addr, input logic [63:0] data);
    lite_write(addr, data[31:0]); lite_write(addr + 4, data[63:32]);
  endtask

  function automatic desc_t make_desc(input bit core, input int unsigned n, len, rows);
    desc_t d;
    d.core = core;
    d.src = 64'h8_0000_0000 + (core ? 64'h1000_0000 : 0) + 64'(n)*64'h20000 + 64'hff0;
    d.dst = d.src + 64'h4000_0000 + 64'd3;
    d.length = 24'(len); d.reps = 32'(rows);
    d.ss = 64'(len) + 64'd48; d.ds = 64'(len) + 64'd80;
    return d;
  endfunction
  task automatic initialize_copy(input desc_t d);
    for (int unsigned r = 0; r < d.reps; r++) begin
      for (int unsigned i = 0; i < d.length; i++) begin
        i_mem.mem[d.src + 64'(r)*d.ss + 64'(i)] = byte'(d.src + 64'(r)*31 + 64'(i)*7);
        i_mem.mem[d.dst + 64'(r)*d.ds + 64'(i)] = 8'hee;
      end
      i_mem.mem[d.dst + 64'(r)*d.ds - 1] = 8'ha5;
      i_mem.mem[d.dst + 64'(r)*d.ds + 64'(d.length)] = 8'h5a;
    end
  endtask
  task automatic send_core(input desc_t d);
    if (random_request_gaps) repeat ($urandom_range(0, 15)) @(negedge clk);
    initialize_copy(d);
    @(negedge clk);
    src = d.src; dst = d.dst; length = d.length; reps = d.reps; ss = d.ss; ds = d.ds;
    core_valid = 1;
    do @(posedge clk); while (!core_ready);
    @(negedge clk); core_valid = 0;
  endtask
  logic [31:0] last_host_id = 0;
  task automatic send_host(input desc_t d);
    if (random_request_gaps) repeat ($urandom_range(0, 15)) @(negedge clk);
    initialize_copy(d);
    write64('hd8, d.src); write64('hd0, d.dst); write64('he0, 64'(d.length));
    write64('h108, d.ss); write64('h100, d.ds); write64('h110, 64'(d.reps));
    lite_write(0, 'h400);
    lite_read('h44, last_host_id);
    assert (last_host_id != 0) else $fatal(1, "valid host request rejected");
    // The APB read launches a held request; wait before reusing frontend registers.
    do @(posedge clk); while (dut.fe_req_valid);
  endtask
  task automatic drain;
    do @(negedge clk); while (busy || expected.size() != 0 || dut.fe_req_valid);
    repeat (8) @(negedge clk);
    assert (host_accepted == host_completed && core_accepted == core_completed)
      else $fatal(1, "lost completion");
  endtask
  task automatic clear_errors;
    @(negedge clk); err_clr = 1;
    @(negedge clk); err_clr = 0;
  endtask

  initial begin : tests
    logic [31:0] host_done;
    int frozen_accepts, ar_before, aw_before;
    desc_t d;
    if ($value$plusargs("SEED=%d", seed)) ;
    void'($urandom(seed));
    cfg_req = '0; src = 0; dst = 0; length = 0; reps = 0; ss = 0; ds = 0;
    repeat (6) @(negedge clk); rst_n = 1;
    repeat (5) @(negedge clk);
    send_core(make_desc(1, 0, 4097, 3)); drain();
    lite_read('h84, host_done);
    assert (host_done == 1 && dut.next_id == ID_W'(2)) else $fatal(1, "core modified Linux IDs");
    send_host(make_desc(0, 0, 4095, 2)); drain();
    $display("[PASS] core-only and Linux-only 2D/page-boundary copies, IDs isolated");

    random_core_ready = 1; random_request_gaps = 1;
    fork
      begin for (int n = 1; n <= 48; n++) send_core(make_desc(1, n, 16 + (n%7)*31, 1+n%3)); end
      begin for (int n = 1; n <= 48; n++) send_host(make_desc(0, n, 17 + (n%5)*29, 1+n%3)); end
    join
    drain(); random_core_ready = 0; random_request_gaps = 0;
    lite_read('h84, host_done);
    assert (host_done == last_host_id) else $fatal(1, "last host ID did not retire");
    assert (err == 0) else $fatal(1, "unexpected DMA errors");
    $display("[PASS] 96 concurrent-source copies, randomized AXI/response stalls, Linux ID wrap");

    // Let AXI data drain while holding the client's final completion. Freezing
    // AXI reads here would exhaust backend read credits before the tag FIFO.
    @(negedge clk); allow_core_completion = 0;
    for (int n = 0; n < SLOTS; n++) send_core(make_desc(1, 100+n, 64, 1));
    frozen_accepts = core_accepted + host_accepted;
    $display("[INFO] capacity filled: slots=%0d pending=%0d rsp=%b backend=%b nd_valid=%b nd_ready=%b",
      SLOTS, expected.size(), rsp_valid, dut.be_busy, dut.nd_req_valid, dut.nd_req_ready);
    fork
      send_core(make_desc(1, 110, 80, 1));
      send_host(make_desc(0, 110, 96, 1));
      begin
        repeat (160) @(negedge clk);
        assert (core_accepted + host_accepted == frozen_accepts && busy)
          else $fatal(1, "full tag FIFO accepted more work or lost busy");
        allow_responses = 1;
        $display("[INFO] resuming response path: rsp=%b backend_rsp=%b nd_rsp=%b",
          rsp_valid, dut.be_rsp_valid, dut.nd_rsp_valid);
        wait (rsp_valid);
        repeat (20) @(negedge clk);
        assert (busy) else $fatal(1, "stalled core completion not busy");
        clear_errors();
        @(negedge clk); allow_core_completion = 1;
      end
    join
    drain();
    assert (max_pending == SLOTS) else $fatal(1, "capacity coverage missing");
    $display("[PASS] tag FIFO full, blocked response, clear during flight, resume and fairness");

    // These requests would bypass older backend work without the ordering guard.
    for (int mode = 0; mode < 2; mode++) begin
      @(negedge clk); allow_responses = 0;
      send_core(make_desc(1, 120+mode*4, 128, 1));
      send_core(make_desc(1, 121+mode*4, mode != 0 ? 0 : 64, mode != 0 ? 3 : 0));
      repeat (80) @(negedge clk);
      assert (!rsp_valid) else $fatal(1, "zero request overtook stalled older transfer");
      allow_responses = 1;
      drain();
      assert (err[2]) else $fatal(1, "zero request error not sticky");
      clear_errors();
    end
    // Same rejection rules apply to Linux, with exactly one ID retirement.
    ar_before = ar_count; aw_before = aw_count;
    send_host(make_desc(0, 130, 32, 0)); drain();
    send_host(make_desc(0, 131, 0, 3)); drain();
    assert (ar_count == ar_before && aw_count == aw_before)
      else $fatal(1, "zero request generated AXI traffic");
    assert (err[2]) else $fatal(1, "host rejection error absent");
    clear_errors();
    @(negedge clk); allow_core_completion = 0;
    send_core(make_desc(1, 132, 0, 3));
    wait (rsp_valid);
    clear_errors();
    repeat (20) @(negedge clk);
    assert (rsp_valid && rsp_error && busy && err == 0)
      else $fatal(1, "clearing errors changed the stalled response or busy");
    allow_core_completion = 1; drain();
    $display("[PASS] ordered zero-repetition and multi-row zero-length errors from both sources");

    d = make_desc(1, 140, 64, 1);
    i_mem.rerr[d.src] = axi_pkg::RESP_SLVERR;
    i_mem.werr[d.dst] = axi_pkg::RESP_DECERR;
    send_core(d); drain();
    assert (err[1:0] == 2'b11) else $fatal(1, "passive AXI error monitor lost core errors");
    clear_errors();
    send_host(make_desc(0, 140, 64, 1)); drain();
    assert (err == 0) else $fatal(1, "error clear/recovery failed");
    $display("[PASS] shared AXI error monitor, clear and clean recovery");

    @(negedge clk); rst_n = 0;
    repeat (5) @(negedge clk); rst_n = 1;
    repeat (5) @(negedge clk);
    send_core(make_desc(1, 150, 32, 2));
    send_host(make_desc(0, 150, 48, 2)); drain();
    assert (max_pending > 1 && competitions > 0) else $fatal(1, "concurrency/fairness coverage missing");
    $display("[PASS] drained reset and repeat run");
    $display("M5_DMA_PASS seed=%0d slots=%0d host=%0d core=%0d max_pending=%0d competitions=%0d",
             seed, SLOTS, host_completed, core_completed, max_pending, competitions);
    $finish;
  end
  initial begin
    #20ms;
    $fatal(1, "M5 transport timeout: host=%0d/%0d core=%0d/%0d pending=%0d busy=%b",
           host_completed, host_accepted, core_completed, core_accepted, expected.size(), busy);
  end
endmodule
