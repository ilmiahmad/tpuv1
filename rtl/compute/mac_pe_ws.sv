// Weight-stationary MAC processing element (TPU v1 Fig. 4), one of N x N.
//
//   activation a  : left -> right (registered), with valid and swap sideband
//   partial sum   : top  -> bottom (registered): psum_o = psum_i + a_i * w
//   weight shadow : shift chain top -> bottom (w_shift_i), loaded while w_active computes
//   swap          : when a valid row carrying the swap bit reaches this PE, it computes with
//                   (and latches) the shadow weight - the diagonal wavefront of the paper.
// Signed INT8 x INT8 -> INT16 product, INT32 accumulate. One always_ff per register (L18).

// use_dsp: Vivado maps an 8x8 multiply-add to LUTs by default (M3 OOC: 0 DSP, 26.7k LUTs);
// force one DSP48E2 per PE (multiply + 32-bit accumulate of the incoming psum).
(* use_dsp = "yes" *)
module mac_pe_ws (
  input  logic               clk_i,
  input  logic               rst_ni,
  // activation in from the left
  input  logic signed [7:0]  a_i,
  input  logic               a_valid_i,
  input  logic               a_swap_i,
  // partial sum in from above
  input  logic signed [31:0] psum_i,
  // weight shadow chain from above
  input  logic signed [7:0]  w_sh_i,
  input  logic               w_shift_i,
  // to the right
  output logic signed [7:0]  a_o,
  output logic               a_valid_o,
  output logic               a_swap_o,
  // down
  output logic signed [31:0] psum_o,
  output logic signed [7:0]  w_sh_o
);

  logic signed [7:0]  w_shadow_q, w_active_q;
  logic               swap_now;
  logic signed [7:0]  w_eff;
  logic signed [15:0] prod;

  assign swap_now = a_valid_i && a_swap_i;
  assign w_eff    = swap_now ? w_shadow_q : w_active_q;
  assign prod     = a_i * w_eff;                       // signed 8 x 8 -> 16

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)        w_shadow_q <= '0;
    else if (w_shift_i) w_shadow_q <= w_sh_i;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)       w_active_q <= '0;
    else if (swap_now) w_active_q <= w_shadow_q;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      a_valid_o <= 1'b0;
      a_swap_o  <= 1'b0;
    end else begin
      a_valid_o <= a_valid_i;
      a_swap_o  <= swap_now;
    end
  end

  // datapath registers (no reset needed; validity travels in a_valid)
  always_ff @(posedge clk_i) begin
    a_o    <= a_i;
    psum_o <= psum_i + 32'(prod);
  end

  assign w_sh_o = w_shadow_q;

endmodule
