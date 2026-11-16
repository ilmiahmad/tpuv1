// One requantisation lane, bit-exact TFLite int8 double rounding (docs/m4_compute_core.md sec 1):
//   x = acc + bias ; y = RDBPOT(SRDHM(x << left, M0), right) ; out = clamp(y + zp, min, max)
// Pipeline (all registers advance on en_i; stall-able):
//   S1: x << left, right, SRDHM-overflow flag      S2: product (DSP)   S3: product (DSP PREG)
//   S4: ab + nudge                                  S5: truncating /2^31, saturate
//   S6: RoundingDivideByPOT                         S7: + zp, clamp -> int8 (output register)
// The stage-0 entry register (acc + params) lives in activation_unit.

module requant_lane (
  input  logic              clk_i,
  input  logic              en_i,
  // stage-0 values
  input  logic [31:0]       acc_i,
  input  logic [31:0]       m0_i,
  input  logic [7:0]        shift_i,
  input  logic [31:0]       bias_i,
  input  logic [7:0]        zp_i,
  input  logic [7:0]        amin_i,
  input  logic [7:0]        amax_i,
  output logic [7:0]        out_o
);

  // ---------------- S1 ----------------
  logic signed [31:0] x_s0;
  logic signed [7:0]  sh_s0;
  logic [4:0]         left_s0, right_s0;
  logic signed [31:0] xl_s0;
  assign x_s0     = $signed(acc_i) + $signed(bias_i);                 // int32 wrap
  assign sh_s0    = $signed(shift_i);
  assign left_s0  = (sh_s0 > 8'sd0) ? sh_s0[4:0] : 5'd0;
  assign right_s0 = (sh_s0 > 8'sd0) ? 5'd0 : 5'(-sh_s0);              // 0..31 (compiler range)
  assign xl_s0    = x_s0 <<< left_s0;                                 // int32 wrap

  logic signed [31:0] xl_s1, m_s1;
  logic [4:0]         right_s1;
  logic               ovf_s1;
  logic [7:0]         zp_s1, mn_s1, mx_s1;
  always_ff @(posedge clk_i) begin
    if (en_i) begin
      xl_s1    <= xl_s0;
      m_s1     <= $signed(m0_i);
      right_s1 <= right_s0;
      ovf_s1   <= (xl_s0 == 32'sh8000_0000) && (m0_i == 32'h8000_0000);
      zp_s1    <= zp_i;  mn_s1 <= amin_i;  mx_s1 <= amax_i;
    end
  end

  // ---------------- S2 / S3: 32x32 signed product in DSPs ----------------
  (* use_dsp = "yes" *) logic signed [63:0] prod_s2;
  (* use_dsp = "yes" *) logic signed [63:0] prod_s3;
  logic [4:0] right_s2, right_s3;
  logic       ovf_s2, ovf_s3;
  logic [7:0] zp_s2, mn_s2, mx_s2, zp_s3, mn_s3, mx_s3;
  always_ff @(posedge clk_i) begin
    if (en_i) begin
      prod_s2  <= 64'(xl_s1) * 64'(m_s1);
      right_s2 <= right_s1; ovf_s2 <= ovf_s1; zp_s2 <= zp_s1; mn_s2 <= mn_s1; mx_s2 <= mx_s1;
    end
  end
  always_ff @(posedge clk_i) begin
    if (en_i) begin
      prod_s3  <= prod_s2;
      right_s3 <= right_s2; ovf_s3 <= ovf_s2; zp_s3 <= zp_s2; mn_s3 <= mn_s2; mx_s3 <= mx_s2;
    end
  end

  // ---------------- S4: ab + nudge ----------------
  localparam logic signed [63:0] NudgePos = 64'sd1073741824;          //  2^30
  localparam logic signed [63:0] NudgeNeg = -64'sd1073741823;         //  1 - 2^30
  logic signed [63:0] v_s4;
  logic [4:0] right_s4;
  logic       ovf_s4;
  logic [7:0] zp_s4, mn_s4, mx_s4;
  always_ff @(posedge clk_i) begin
    if (en_i) begin
      v_s4     <= prod_s3 + ((prod_s3 >= 64'sd0) ? NudgePos : NudgeNeg);
      right_s4 <= right_s3; ovf_s4 <= ovf_s3; zp_s4 <= zp_s3; mn_s4 <= mn_s3; mx_s4 <= mx_s3;
    end
  end

  // ---------------- S5: C-style truncating division by 2^31, saturation ----------------
  // (ab + nudge)/2^31 of two int32 operands always fits int32 except INT32_MIN*INT32_MIN,
  // which ovf handles: the upper 32 bits are a pure sign extension.
  /* verilator lint_off UNUSEDSIGNAL */
  logic signed [63:0] vq_s4;
  /* verilator lint_on UNUSEDSIGNAL */
  assign vq_s4 = (v_s4 < 64'sd0) ? ((v_s4 + 64'sd2147483647) >>> 31) : (v_s4 >>> 31);
  logic signed [31:0] hi_s5;
  logic [4:0] right_s5;
  logic [7:0] zp_s5, mn_s5, mx_s5;
  always_ff @(posedge clk_i) begin
    if (en_i) begin
      hi_s5    <= ovf_s4 ? 32'sh7FFF_FFFF : vq_s4[31:0];
      right_s5 <= right_s4; zp_s5 <= zp_s4; mn_s5 <= mn_s4; mx_s5 <= mx_s4;
    end
  end

  // ---------------- S6: RoundingDivideByPOT ----------------
  logic [31:0]        mask_s5, rem_s5, thr_s5;
  logic signed [31:0] shr_s5;
  assign mask_s5 = (32'd1 << right_s5) - 32'd1;
  assign rem_s5  = 32'(hi_s5) & mask_s5;
  assign thr_s5  = (mask_s5 >> 1) + ((hi_s5 < 32'sd0) ? 32'd1 : 32'd0);
  assign shr_s5  = hi_s5 >>> right_s5;
  logic signed [31:0] y_s6;
  logic [7:0] zp_s6, mn_s6, mx_s6;
  always_ff @(posedge clk_i) begin
    if (en_i) begin
      y_s6  <= shr_s5 + ((rem_s5 > thr_s5) ? 32'sd1 : 32'sd0);
      zp_s6 <= zp_s5; mn_s6 <= mn_s5; mx_s6 <= mx_s5;
    end
  end

  // ---------------- S7: + zp (int32 wrap), clamp -> int8 ----------------
  logic signed [31:0] z_s6, lo_s6, hi_lim_s6;
  assign z_s6      = y_s6 + 32'($signed(zp_s6));
  assign lo_s6     = 32'($signed(mn_s6));
  assign hi_lim_s6 = 32'($signed(mx_s6));
  always_ff @(posedge clk_i) begin
    if (en_i) begin
      if      (z_s6 < lo_s6)     out_o <= mn_s6;
      else if (z_s6 > hi_lim_s6) out_o <= mx_s6;
      else                       out_o <= z_s6[7:0];
    end
  end

endmodule
