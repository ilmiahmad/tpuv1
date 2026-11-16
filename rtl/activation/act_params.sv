// Requantisation parameter store (docs/m4_compute_core.md sec 2): 4 sets x N channels of
// {M0, shift, bias} plus per-set {zp_out, act_min, act_max}. Written by SET_CONFIG.
// A write in cycle t is visible to rows accepted by activation_unit in cycle t+1 or later
// (rows latch their parameters at entry).
// cfg_err_o: registered 1-cycle pulse, the cycle after a write to an illegal target.

module act_params #(
  parameter int unsigned N = 16
) (
  input  logic                   clk_i,
  input  logic                   rst_ni,
  input  logic                   cfg_we_i,
  input  logic [15:0]            cfg_target_i,
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [63:0]            cfg_value_i,     // only [39:0] defined by the contract (sec 2)
  /* verilator lint_on UNUSEDSIGNAL */
  output logic                   cfg_err_o,
  // read: selected set
  input  logic [1:0]             rd_q_i,
  output logic [N-1:0][31:0]     m0_o,
  output logic [N-1:0][7:0]      shift_o,
  output logic [N-1:0][31:0]     bias_o,
  output logic [7:0]             zp_o,
  output logic [7:0]             amin_o,
  output logic [7:0]             amax_o
);

  localparam int unsigned NQ = 4;

  logic [NQ-1:0][N-1:0][31:0] m0_q, bias_q;
  logic [NQ-1:0][N-1:0][7:0]  sh_q;
  logic [NQ-1:0][7:0]         zp_q, amin_q, amax_q;

  // decode
  wire [3:0] kind    = cfg_target_i[15:12];
  wire [1:0] q_idx   = cfg_target_i[5:4];
  wire [3:0] c_idx   = cfg_target_i[3:0];
  wire       ch_ok   = (cfg_target_i[11:6] == 6'd0) && (32'(c_idx) < N);
  wire       is_mul  = (kind == 4'h1) && ch_ok;
  wire       is_bias = (kind == 4'h2) && ch_ok;
  wire       is_set  = (kind == 4'h3) && (cfg_target_i[11:2] == 10'd0);
  wire       legal   = is_mul || is_bias || is_set;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) m0_q <= '0;
    else if (cfg_we_i && is_mul) m0_q[q_idx][c_idx] <= cfg_value_i[31:0];
  end
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) sh_q <= '0;
    else if (cfg_we_i && is_mul) sh_q[q_idx][c_idx] <= cfg_value_i[39:32];
  end
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) bias_q <= '0;
    else if (cfg_we_i && is_bias) bias_q[q_idx][c_idx] <= cfg_value_i[31:0];
  end
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      zp_q <= '0; amin_q <= '0; amax_q <= '0;
    end else if (cfg_we_i && is_set) begin
      zp_q[cfg_target_i[1:0]]   <= cfg_value_i[7:0];
      amin_q[cfg_target_i[1:0]] <= cfg_value_i[15:8];
      amax_q[cfg_target_i[1:0]] <= cfg_value_i[23:16];
    end
  end
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) cfg_err_o <= 1'b0;
    else         cfg_err_o <= cfg_we_i && !legal;
  end

  assign m0_o    = m0_q[rd_q_i];
  assign shift_o = sh_q[rd_q_i];
  assign bias_o  = bias_q[rd_q_i];
  assign zp_o    = zp_q[rd_q_i];
  assign amin_o  = amin_q[rd_q_i];
  assign amax_o  = amax_q[rd_q_i];

endmodule
