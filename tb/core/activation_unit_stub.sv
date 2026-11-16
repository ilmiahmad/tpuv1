// BEHAVIOURAL STAND-IN for rtl/activation/activation_unit. Same ports, exact
// docs/m4_compute_core.md sec 1/sec 2 arithmetic, 1-cycle latency. Used only until A's RTL exists;
// final verification uses the real module.
module activation_unit #(
  parameter int unsigned N = 16
) (
  input  logic              clk_i, rst_ni,
  input  logic              cfg_we_i,
  input  logic [15:0]       cfg_target_i,
  input  logic [63:0]       cfg_value_i,
  output logic              cfg_err_o,
  input  logic              in_valid_i,
  output logic              in_ready_o,
  input  logic [N*32-1:0]   in_data_i,
  input  logic [1:0]        in_qset_i,
  output logic              out_valid_o,
  input  logic              out_ready_i,
  output logic [N*8-1:0]    out_data_o,
  output logic              busy_o
);
  logic signed [31:0] m0 [4][N], bias [4][N];
  logic signed [7:0]  sh [4][N], zp [4], amin [4], amax [4];

  function automatic logic signed [31:0] srdhm(input logic signed [31:0] a, input logic signed [31:0] b);
    longint ab, nudge;
    if (a == 32'sh8000_0000 && b == 32'sh8000_0000) return 32'sh7fff_ffff;
    ab = longint'(a) * longint'(b);
    nudge = (ab >= 0) ? (64'sd1 << 30) : (64'sd1 - (64'sd1 << 30));
    return 32'((ab + nudge) / (64'sd1 << 31));          // SV '/' truncates toward zero
  endfunction
  function automatic logic signed [31:0] rdbpot(input logic signed [31:0] x, input int e);
    logic signed [31:0] mask, rem, thr;
    mask = (32'sd1 <<< e) - 32'sd1;
    rem  = x & mask;
    thr  = (mask >>> 1) + ((x < 0) ? 32'sd1 : 32'sd0);
    return (x >>> e) + ((rem > thr) ? 32'sd1 : 32'sd0);
  endfunction
  function automatic logic signed [7:0] requant(input logic signed [31:0] acc, input int q, input int c);
    logic signed [31:0] x, y; int left, right; longint z;
    x = acc + bias[q][c];
    left  = (sh[q][c] > 0) ? int'(sh[q][c]) : 0;
    right = (sh[q][c] > 0) ? 0 : -int'(sh[q][c]);
    y = rdbpot(srdhm(x <<< left, m0[q][c]), right);
    z = longint'(y) + longint'(zp[q]);
    if (z < longint'(amin[q])) z = amin[q];
    if (z > longint'(amax[q])) z = amax[q];
    return 8'(z);
  endfunction

  wire [3:0] kind = cfg_target_i[15:12];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) cfg_err_o <= 1'b0;
    else cfg_err_o <= cfg_we_i && !((kind == 4'h1 || kind == 4'h2) ? cfg_target_i[11:6] == 0
                                    : (kind == 4'h3) ? cfg_target_i[11:2] == 0 : 1'b0);
  end
  always_ff @(posedge clk_i) if (cfg_we_i) begin
    if (kind == 4'h1 && cfg_target_i[11:6] == 0) begin
      m0[cfg_target_i[5:4]][cfg_target_i[3:0]] <= cfg_value_i[31:0];
      sh[cfg_target_i[5:4]][cfg_target_i[3:0]] <= cfg_value_i[39:32];
    end
    if (kind == 4'h2 && cfg_target_i[11:6] == 0) bias[cfg_target_i[5:4]][cfg_target_i[3:0]] <= cfg_value_i[31:0];
    if (kind == 4'h3 && cfg_target_i[11:2] == 0) begin
      zp[cfg_target_i[1:0]] <= cfg_value_i[7:0]; amin[cfg_target_i[1:0]] <= cfg_value_i[15:8];
      amax[cfg_target_i[1:0]] <= cfg_value_i[23:16];
    end
  end

  assign in_ready_o = !out_valid_o || out_ready_i;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) out_valid_o <= 1'b0;
    else if (in_ready_o) out_valid_o <= in_valid_i;
  end
  always_ff @(posedge clk_i) if (in_valid_i && in_ready_o)
    for (int c = 0; c < N; c++) out_data_o[8*c +: 8] <= requant(in_data_i[32*c +: 32], int'(in_qset_i), c);
  assign busy_o = out_valid_o;
endmodule
