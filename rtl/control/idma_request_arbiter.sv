// Phase B request transport. All data movement remains in PULP iDMA.
// See docs/m5_dma_interface.md for the ND handshake and zero-repetition ordering rule.
`default_nettype none
`timescale 1ns/1ps

module idma_request_arbiter #(
  parameter int unsigned Outstanding = 8,
  parameter type req_t = logic,
  parameter type rsp_t = logic
) (
  input  wire logic clk_i,
  input  wire logic rst_ni,
  input  req_t host_req_i,
  input  wire logic host_req_valid_i,
  output logic host_req_ready_o,
  output logic host_rsp_valid_o,
  input  wire logic host_rsp_ready_i,
  input  req_t core_req_i,
  input  wire logic core_req_valid_i,
  output logic core_req_ready_o,
  output rsp_t core_rsp_o,
  output logic core_rsp_valid_o,
  input  wire logic core_rsp_ready_i,
  output req_t nd_req_o,
  output logic nd_req_valid_o,
  input  wire logic nd_req_ready_i,
  input  rsp_t nd_rsp_i,
  input  wire logic nd_rsp_valid_i,
  output logic nd_rsp_ready_o,
  output logic busy_o
);
  typedef struct packed {
    logic source;  // 0: register frontend, 1: core
    req_t req;
  } tagged_req_t;

  tagged_req_t [1:0] arb_inputs;
  tagged_req_t arb_output;
  logic [1:0] arb_ready;
  logic arb_valid, capture_ready, capture;
  req_t held_req_q;
  logic held_valid_q;
  logic tag_full, tag_empty, tag_head, retire;
  logic [cc_pkg::cnt_width(Outstanding)-1:0] tag_usage;
  rsp_t completion;
  logic completion_valid, completion_ready, response_buffer_ready;

  assign arb_inputs[0] = '{source: 1'b0, req: host_req_i};
  assign arb_inputs[1] = '{source: 1'b1, req: core_req_i};
  cc_stream_arbiter #(.data_t(tagged_req_t), .NumInp(2)) i_arbiter (
    .clk_i, .rst_ni, .clr_i(1'b0),
    .inp_data_i(arb_inputs), .inp_valid_i({core_req_valid_i, host_req_valid_i}),
    .inp_ready_o(arb_ready), .oup_data_o(arb_output),
    .oup_valid_o(arb_valid), .oup_ready_i(capture_ready)
  );
  assign host_req_ready_o = arb_ready[0];
  assign core_req_ready_o = arb_ready[1];
  assign capture_ready = !held_valid_q && !tag_full;
  assign capture = arb_valid && capture_ready;

  // Reserve the tag at capture, before presenting any ND valid. An ND descriptor
  // is held until its final row is expanded; it is not an ordinary input FIFO.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      held_req_q <= '0;
      held_valid_q <= 1'b0;
    end else begin
      if (capture) begin
        held_req_q <= arb_output.req;
        held_valid_q <= 1'b1;
      end else if (nd_req_valid_o && nd_req_ready_i) begin
        held_valid_q <= 1'b0;
      end
    end
  end

  cc_fifo #(.Depth(Outstanding), .data_t(logic)) i_sources (
    .clk_i, .rst_ni, .clr_i(1'b0), .flush_i(1'b0),
    .full_o(tag_full), .empty_o(tag_empty), .usage_o(tag_usage),
    .data_i(arb_output.source), .push_i(capture), .data_o(tag_head), .pop_i(retire)
  );

  // The midend's effective repetition counter is 32 bits. Its zero-reps error
  // bypasses backend responses, so do not let it overtake an older descriptor.
  logic zero_reps, zero_length;
  assign zero_reps = (held_req_q.d_req[0].reps[31:0] == 0);
  assign zero_length = (held_req_q.burst_req.length == 0);
  always_comb begin
    nd_req_o = held_req_q;
    // A zero-length multi-row transfer must produce one ND error, not one
    // backend rejection per row (the backend synthesizes last=1 on each reject).
    if (zero_length) nd_req_o.d_req[0].reps = '0;
  end
  assign nd_req_valid_o = held_valid_q && (!(zero_reps || zero_length) || tag_usage == 1);

  // NO_ERROR_HANDLING ties AXI read readiness to the backend's response-ready
  // input. A client may wait for valid before asserting ready, so cut that path:
  // otherwise a stalled core prevents the reads needed to create its completion.
  cc_spill_register #(.data_t(rsp_t)) i_responses (
    .clk_i, .rst_ni, .clr_i(1'b0),
    .data_i(nd_rsp_i), .valid_i(nd_rsp_valid_i && !tag_empty), .ready_o(response_buffer_ready),
    .data_o(completion), .valid_o(completion_valid), .ready_i(completion_ready)
  );
  assign host_rsp_valid_o = completion_valid && !tag_empty && !tag_head;
  assign core_rsp_valid_o = completion_valid && !tag_empty && tag_head;
  assign core_rsp_o = completion;
  assign completion_ready = !tag_empty && (tag_head ? core_rsp_ready_i : host_rsp_ready_i);
  assign nd_rsp_ready_o = response_buffer_ready && !tag_empty;
  assign retire = completion_valid && completion_ready;
  assign busy_o = !tag_empty || held_valid_q || arb_valid;

`ifndef SYNTHESIS
  initial assert (Outstanding > 0) else $fatal(1, "Outstanding must be positive");
  a_response_has_source: assert property (@(posedge clk_i) disable iff (!rst_ni)
    nd_rsp_valid_i |-> !tag_empty);
  a_held_nd_stable: assert property (@(posedge clk_i) disable iff (!rst_ni)
    nd_req_valid_o && !nd_req_ready_i |=> nd_req_valid_o && $stable(nd_req_o));
  a_core_rsp_stable: assert property (@(posedge clk_i) disable iff (!rst_ni)
    core_rsp_valid_o && !core_rsp_ready_i |=> core_rsp_valid_o && $stable(core_rsp_o));
  a_one_source: assert property (@(posedge clk_i) disable iff (!rst_ni)
    !(host_req_ready_o && host_req_valid_i && core_req_ready_o && core_req_valid_i));
`endif
endmodule

`default_nettype wire
