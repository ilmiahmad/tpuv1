// Output de-skew: column n of the bottom row is delayed by (N-1-n) cycles so a whole output row
// leaves in one cycle. The valid comes from column N-1 (the latest, not delayed).

module output_deskew #(
  parameter int unsigned N = 16
) (
  input  logic                 clk_i,
  input  logic                 rst_ni,
  input  logic [N-1:0][31:0]   psum_i,
  input  logic                 valid_last_i,   // valid of column N-1
  output logic [N-1:0][31:0]   y_o,
  output logic                 y_valid_o
);

  assign y_o[N-1]  = psum_i[N-1];
  assign y_valid_o = valid_last_i;

  for (genvar n = 0; n < N - 1; n++) begin : gen_col
    localparam int unsigned D = N - 1 - n;
    logic [D-1:0][31:0] q;
    always_ff @(posedge clk_i) begin
      q[0] <= psum_i[n];
      for (int i = 1; i < D; i++) q[i] <= q[i-1];
    end
    assign y_o[n] = q[D-1];
  end

  // rst_ni kept on the port for a uniform interface; the delay line carries data only
  /* verilator lint_off UNUSEDSIGNAL */
  logic unused_rst;
  assign unused_rst = rst_ni;
  /* verilator lint_on UNUSEDSIGNAL */

endmodule
