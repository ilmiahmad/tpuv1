// Activation unit (docs/m4_compute_core.md sec 3): ACC row (N x INT32) -> TFLite-exact int8
// requantisation (+ ReLU/ReLU6 via the clamp range) -> UB row (N x INT8).
//
// Pipeline: entry register S0 (data + parameters of in_qset latched at acceptance) and
// requant_lane stages S1..S7 -> Latency = 8 cycles from acceptance to out_valid (no stall).
// Flow control: one global enable. The whole pipe advances when the output register is empty
// or being read (en = !valid[7] || out_ready). in_ready = en, so 1 row/cycle is sustained
// under out_ready = 1 and a stalled output freezes the pipe without losing data.
// Parameter-write ordering: a SET_CONFIG write in cycle t affects rows accepted in t+1 or later.

module activation_unit #(
  parameter int unsigned N = 16
) (
  input  logic              clk_i,
  input  logic              rst_ni,
  // parameter writes
  input  logic              cfg_we_i,
  input  logic [15:0]       cfg_target_i,
  input  logic [63:0]       cfg_value_i,
  output logic              cfg_err_o,
  // stream in
  input  logic              in_valid_i,
  output logic              in_ready_o,
  input  logic [N*32-1:0]   in_data_i,
  input  logic [1:0]        in_qset_i,
  // stream out
  output logic              out_valid_o,
  input  logic              out_ready_i,
  output logic [N*8-1:0]    out_data_o,
  output logic              busy_o
);

  localparam int unsigned Latency = 8;     // S0 .. S7

  // ---------------- parameters ----------------
  logic [N-1:0][31:0] p_m0, p_bias;
  logic [N-1:0][7:0]  p_sh;
  logic [7:0]         p_zp, p_min, p_max;

  act_params #(.N(N)) i_params (
    .clk_i, .rst_ni,
    .cfg_we_i, .cfg_target_i, .cfg_value_i, .cfg_err_o,
    .rd_q_i (in_qset_i),
    .m0_o (p_m0), .shift_o (p_sh), .bias_o (p_bias),
    .zp_o (p_zp), .amin_o (p_min), .amax_o (p_max)
  );

  // ---------------- flow control ----------------
  logic [Latency-1:0] vld_q;
  logic               en;
  assign en          = !vld_q[Latency-1] || out_ready_i;
  assign in_ready_o  = en;
  assign out_valid_o = vld_q[Latency-1];
  assign busy_o      = |vld_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)  vld_q <= '0;
    else if (en)  vld_q <= {vld_q[Latency-2:0], in_valid_i};
  end

  // ---------------- S0 entry register ----------------
  logic [N-1:0][31:0] acc_s0, m0_s0, bias_s0;
  logic [N-1:0][7:0]  sh_s0;
  logic [7:0]         zp_s0, min_s0, max_s0;
  always_ff @(posedge clk_i) begin
    if (en) begin
      acc_s0  <= in_data_i;
      m0_s0   <= p_m0;   sh_s0  <= p_sh;   bias_s0 <= p_bias;
      zp_s0   <= p_zp;   min_s0 <= p_min;  max_s0  <= p_max;
    end
  end

  // ---------------- lanes ----------------
  for (genvar c = 0; c < N; c++) begin : gen_lane
    requant_lane i_lane (
      .clk_i,
      .en_i    ( en ),
      .acc_i   ( acc_s0[c]  ),
      .m0_i    ( m0_s0[c]   ),
      .shift_i ( sh_s0[c]   ),
      .bias_i  ( bias_s0[c] ),
      .zp_i    ( zp_s0      ),
      .amin_i  ( min_s0     ),
      .amax_i  ( max_s0     ),
      .out_o   ( out_data_o[c*8 +: 8] )
    );
  end

endmodule
