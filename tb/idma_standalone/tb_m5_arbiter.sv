// Exercise default eight-entry tag capacity with the real ND midend and a
// controllable ordered 1D responder. AXI/data correctness is checked by tb_m5_dma.
`timescale 1ns/1ps
`include "idma/typedef.svh"

module tb_m5_arbiter;
  localparam int unsigned NumDim = 2, Depth = 8;
  typedef logic [63:0] addr_t;
  typedef logic [5:0] id_t;
  typedef logic [23:0] len_t;
  `IDMA_TYPEDEF_FULL_REQ_T(req_t, id_t, addr_t, len_t)
  `IDMA_TYPEDEF_FULL_RSP_T(rsp_t, addr_t)
  `IDMA_TYPEDEF_FULL_ND_REQ_T(nd_t, req_t, addr_t, addr_t)
  logic clk = 0, rst_n = 0;
  always #5ns clk = ~clk;
  nd_t host_req, core_req, nd_req;
  logic host_valid = 0, core_valid = 0, host_ready, core_ready;
  logic host_done, core_done, core_done_ready = 0, busy;
  logic nd_valid, nd_ready, rsp_valid, rsp_ready;
  rsp_t core_rsp, nd_rsp, be_rsp;
  req_t be_req;
  logic be_valid, be_ready = 1, be_rsp_valid, be_rsp_ready, mid_busy;
  logic allow_complete = 0;
  rsp_t backend_responses[$];
  typedef struct packed { bit core; bit error; addr_t last_src; } expected_t;
  expected_t expected[$];
  int accepted = 0, completed = 0, max_pending = 0, competition = 0;

  idma_request_arbiter #(.Outstanding(Depth), .req_t(nd_t), .rsp_t(rsp_t)) dut (
    .clk_i(clk), .rst_ni(rst_n),
    .host_req_i(host_req), .host_req_valid_i(host_valid), .host_req_ready_o(host_ready),
    .host_rsp_valid_o(host_done), .host_rsp_ready_i(1'b1),
    .core_req_i(core_req), .core_req_valid_i(core_valid), .core_req_ready_o(core_ready),
    .core_rsp_o(core_rsp), .core_rsp_valid_o(core_done), .core_rsp_ready_i(core_done_ready),
    .nd_req_o(nd_req), .nd_req_valid_o(nd_valid), .nd_req_ready_i(nd_ready),
    .nd_rsp_i(nd_rsp), .nd_rsp_valid_i(rsp_valid), .nd_rsp_ready_o(rsp_ready), .busy_o(busy)
  );
  idma_nd_midend #(
    .NumDim(NumDim), .addr_t(addr_t), .idma_req_t(req_t), .idma_rsp_t(rsp_t),
    .idma_nd_req_t(nd_t), .RepWidths('{default:32'd32})
  ) i_nd (
    .clk_i(clk), .rst_ni(rst_n), .nd_req_i(nd_req), .nd_req_valid_i(nd_valid), .nd_req_ready_o(nd_ready),
    .nd_rsp_o(nd_rsp), .nd_rsp_valid_o(rsp_valid), .nd_rsp_ready_i(rsp_ready),
    .burst_req_o(be_req), .burst_req_valid_o(be_valid), .burst_req_ready_i(be_ready),
    .burst_rsp_i(be_rsp), .burst_rsp_valid_i(be_rsp_valid), .burst_rsp_ready_o(be_rsp_ready),
    .busy_o(mid_busy)
  );
  always_comb begin
    be_rsp = '0;
    be_rsp_valid = allow_complete && backend_responses.size() != 0;
    if (backend_responses.size() != 0) be_rsp = backend_responses[0];
  end
  always @(posedge clk) begin : model_and_scoreboard
    rsp_t r;
    expected_t e;
    nd_t q;
    if (!rst_n) begin
      backend_responses.delete(); expected.delete();
    end else begin
      if (host_valid && host_ready || core_valid && core_ready) begin
        assert (!(host_valid && host_ready && core_valid && core_ready)) else $fatal(1, "double grant");
        q = core_valid && core_ready ? core_req : host_req;
        e.core = core_valid && core_ready;
        e.error = q.burst_req.length == 0 || q.d_req[0].reps == 0;
        e.last_src = q.burst_req.src_addr + (q.d_req[0].reps-1)*q.d_req[0].src_strides;
        expected.push_back(e); accepted++;
        if (core_valid && host_valid) competition++;
      end
      if (host_done || core_done && core_done_ready) begin
        assert (!(host_done && core_done) && expected.size() > 0) else $fatal(1, "orphan response");
        e = expected.pop_front();
        assert (e.core == (core_done && core_done_ready)) else $fatal(1, "response source order");
        if (e.core) begin
          assert (core_rsp.error == e.error) else $fatal(1, "response error payload");
          if (!e.error) assert (core_rsp.pld.burst_addr == e.last_src)
            else $fatal(1, "response address payload");
        end
        completed++;
      end
      if (expected.size() > max_pending) max_pending = expected.size();
      assert (expected.size() <= Depth) else $fatal(1, "capacity exceeded");
      // Sample valid/ready before mutating the model queue.
      if (be_rsp_valid && be_rsp_ready) r = backend_responses.pop_front();
      if (be_valid && be_ready) begin
        r = '0; r.last = be_req.opt.last; r.pld.burst_addr = be_req.src_addr;
        backend_responses.push_back(r);
      end
    end
  end
  function automatic nd_t descriptor(input bit core, input int n);
    nd_t q;
    q = '0;
    q.burst_req.src_addr = (core ? 64'h80000000 : 64'h10000000) + 64'(n)*4096;
    q.burst_req.dst_addr = q.burst_req.src_addr + 64'h100000000;
    q.burst_req.length = 16;
    q.d_req[0].reps = 64'(1+n%3);
    q.d_req[0].src_strides = 64;
    q.d_req[0].dst_strides = 128;
    return q;
  endfunction
  task automatic send_core(input nd_t q);
    @(negedge clk); core_req = q; core_valid = 1;
    do @(posedge clk); while (!core_ready);
    @(negedge clk); core_valid = 0;
  endtask
  task automatic send_host(input nd_t q);
    @(negedge clk); host_req = q; host_valid = 1;
    do @(posedge clk); while (!host_ready);
    @(negedge clk); host_valid = 0;
  endtask
  task automatic drain;
    do @(negedge clk); while (busy || expected.size() != 0 || backend_responses.size() != 0);
    repeat (5) @(negedge clk);
    assert (accepted == completed) else $fatal(1, "lost response");
  endtask
  initial begin
    nd_t q;
    host_req = '0; core_req = '0;
    repeat (5) @(negedge clk); rst_n = 1;
    fork
      begin for (int n=0; n<24; n++) send_host(descriptor(0,n)); end
      begin for (int n=0; n<24; n++) send_core(descriptor(1,n)); end
      begin
        wait (expected.size() == Depth);
        repeat (30) @(negedge clk);
        assert (accepted == Depth && busy) else $fatal(1, "full tag queue not backpressured");
        allow_complete = 1;
        wait (core_done);
        repeat (15) @(negedge clk);
        core_done_ready = 1;
      end
    join
    drain();
    $display("[PASS] eight outstanding tags, full backpressure, concurrent sources, response payloads");
    @(negedge clk); allow_complete = 0;
    send_core(descriptor(1,30));
    q = descriptor(1,31); q.d_req[0].reps = 0;
    send_core(q);
    repeat (30) @(negedge clk);
    assert (!core_done) else $fatal(1, "zero reps overtook an older descriptor");
    allow_complete = 1;
    drain();
    assert (max_pending == Depth && competition > 16) else $fatal(1, "coverage missing");
    $display("M5_ARBITER_PASS accepted=%0d completed=%0d max_pending=%0d competition=%0d",
      accepted, completed, max_pending, competition);
    $finish;
  end
  initial begin #1ms; $fatal(1, "arbiter timeout accepted=%0d completed=%0d", accepted, completed); end
endmodule
