// 2x2 / stride-2 pooling over requantized int8 rows (docs/m6_cnn_contract.md sec 5).
//
// Input: one row of N int8 lanes per accepted beat, in groups of 4 consecutive rows in the order
//   (2i,2j) (2i,2j+1) (2i+1,2j) (2i+1,2j+1).  Output: one pooled row per group, per lane
//   mode 1 (max): max(v0..v3), signed
//   mode 2 (avg): TFLite int8 AveragePool, count 4:  s = v0+v1+v2+v3;
//                 out = s >= 0 ? (s + 2) >> 2 : -(((-s) + 2) >> 2)   (round half away from zero)
//   mode 0      : pass-through, every row is its own group (not used by act_seq, handy for tests)
//   mode 3      : undefined; behaves as max (the decoder rejects pool values 3..7)
// The mode is sampled with the first row of a group and then held in mode_q.
//
// Timing: no output register. The group's fourth row produces out_valid_o in the SAME cycle from
// the accumulator registers plus the row on in_data_i, so a pass-through UB write of a following
// ACTIVATE can never collide with a pooled write (at most one row accepted per cycle).
//   in_ready_o = !(group completes with this row) || out_ready_i
// in_ready_o therefore depends combinationally on out_ready_i (same rule as activation_unit);
// out_ready_i must not depend on in_valid_i.  Rows 1..3 of a group never wait.

module pool_unit #(
  parameter int unsigned N = 16
) (
  input  logic              clk_i,
  input  logic              rst_ni,
  // rows in
  input  logic              in_valid_i,
  output logic              in_ready_o,
  input  logic [N*8-1:0]    in_data_i,
  input  logic [1:0]        in_mode_i,
  // pooled rows out (combinational with the last row of each group)
  output logic              out_valid_o,
  input  logic              out_ready_i,
  output logic [N*8-1:0]    out_data_o,
  // status
  output logic              busy_o              // a group is partially accumulated
);

  logic [1:0] cnt_q;                             // rows of the current group already absorbed
  logic [1:0] mode_q;
  logic signed [7:0] mx_q  [N];                  // running max (mode 1)
  logic signed [9:0] sum_q [N];                  // running sum (mode 2): 4 * [-128,127] fits 10 b

  wire [1:0] mode = (cnt_q == 2'd0) ? in_mode_i : mode_q;
  wire       pass = (mode == 2'd0);
  wire       avg  = (mode == 2'd2);
  wire       last = pass || (cnt_q == 2'd3);     // this row completes its group
  wire       first = pass || (cnt_q == 2'd0);

  assign in_ready_o  = !last || out_ready_i;
  wire   take        = in_valid_i && in_ready_o;
  assign out_valid_o = in_valid_i && last;
  assign busy_o      = (cnt_q != 2'd0);

  // ---- per-lane datapath ------------------------------------------------------------------
  logic signed [7:0]  v      [N];
  logic signed [7:0]  mx_nx  [N];
  logic signed [9:0]  sum_nx [N];
  logic [N*8-1:0]     res;

  for (genvar l = 0; l < N; l++) begin : gen_lane
    assign v[l] = $signed(in_data_i[l*8 +: 8]);
    assign mx_nx[l]  = first ? v[l] : ((v[l] > mx_q[l]) ? v[l] : mx_q[l]);
    assign sum_nx[l] = (first ? 10'sd0 : sum_q[l]) + 10'(v[l]);   // sign-extended add

    // average: round half away from zero of sum/4
    logic        neg;
    logic [10:0] mag;                            // |s| <= 512
    logic [10:0] t;                              // (|s| + 2) >> 2 <= 128 (fits 8 b unsigned)
    logic [7:0]  sgn;
    assign neg = sum_nx[l][9];
    assign mag = neg ? 11'(-{sum_nx[l][9], sum_nx[l]}) : {1'b0, sum_nx[l]};
    assign t   = (mag + 11'd2) >> 2;
    assign sgn = neg ? (8'd0 - t[7:0]) : t[7:0]; // t == 128 only when negative: -128

    always_comb begin
      if (pass)     res[l*8 +: 8] = in_data_i[l*8 +: 8];
      else if (avg) res[l*8 +: 8] = sgn;
      else          res[l*8 +: 8] = mx_nx[l];
    end
  end
  assign out_data_o = res;

  // ---- state --------------------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      cnt_q  <= 2'd0;
      mode_q <= 2'd0;
    end else if (take && !pass) begin
      mode_q <= mode;
      cnt_q  <= last ? 2'd0 : cnt_q + 2'd1;      // cnt_q == 3 -> 0 when last
    end
  end

  for (genvar l = 0; l < N; l++) begin : gen_acc
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mx_q[l]  <= '0;
        sum_q[l] <= '0;
      end else if (take && !pass) begin
        mx_q[l]  <= mx_nx[l];
        sum_q[l] <= sum_nx[l];
      end
    end
  end

  // pragma translate_off
  // the mode must not change inside a group (act_seq tags every row of a group with one mode)
  a_mode_stable: assert property (@(posedge clk_i) disable iff (!rst_ni)
      (in_valid_i && in_ready_o && cnt_q != 2'd0) |-> (in_mode_i == mode_q))
    else $error("pool_unit: mode changed inside a group");
  // pragma translate_on

endmodule
