// N x N weight-stationary systolic array. PE(k,n): row k = input lane k, column n = output n.
// Inputs per row k are already skewed by k cycles (data_setup); outputs per column n leave the
// bottom row skewed by n cycles (output_deskew aligns them).

module systolic_array #(
  parameter int unsigned N = 16
) (
  input  logic                     clk_i,
  input  logic                     rst_ni,
  // per row (already skewed)
  input  logic [N-1:0][7:0]        a_row_i,
  input  logic [N-1:0]             valid_row_i,
  input  logic [N-1:0]             swap_row_i,
  // weight word into the top of every column, shifted when w_shift_i
  input  logic [N-1:0][7:0]        w_top_i,
  input  logic                     w_shift_i,
  // bottom row, per column (column n is n cycles later than column 0)
  output logic [N-1:0][31:0]       psum_bot_o,
  output logic [N-1:0]             valid_bot_o
);

  // a/valid/swap travel right: index [k][n] is the input of PE(k,n); [k][N] is unused
  logic [N-1:0][N:0][7:0]  a_w;
  logic [N-1:0][N:0]       v_w;
  logic [N-1:0][N:0]       s_w;
  // psum and weight shadow travel down: [k][n] is the input of PE(k,n); [N][n] is the bottom
  logic [N:0][N-1:0][31:0] p_w;
  /* verilator lint_off UNUSEDSIGNAL */      // w_w[N] = shadow leaving the bottom row (unused)
  logic [N:0][N-1:0][7:0]  w_w;
  /* verilator lint_on UNUSEDSIGNAL */

  for (genvar k = 0; k < N; k++) begin : gen_row_in
    assign a_w[k][0] = a_row_i[k];
    assign v_w[k][0] = valid_row_i[k];
    assign s_w[k][0] = swap_row_i[k];
  end
  for (genvar n = 0; n < N; n++) begin : gen_col_in
    assign p_w[0][n] = '0;
    assign w_w[0][n] = w_top_i[n];
  end

  for (genvar k = 0; k < N; k++) begin : gen_k
    for (genvar n = 0; n < N; n++) begin : gen_n
      mac_pe_ws i_pe (
        .clk_i,
        .rst_ni,
        .a_i       ( a_w[k][n]   ),
        .a_valid_i ( v_w[k][n]   ),
        .a_swap_i  ( s_w[k][n]   ),
        .psum_i    ( p_w[k][n]   ),
        .w_sh_i    ( w_w[k][n]   ),
        .w_shift_i ( w_shift_i   ),
        .a_o       ( a_w[k][n+1] ),
        .a_valid_o ( v_w[k][n+1] ),
        .a_swap_o  ( s_w[k][n+1] ),
        .psum_o    ( p_w[k+1][n] ),
        .w_sh_o    ( w_w[k+1][n] )
      );
    end
  end

  // bottom row: the psum leaving PE(N-1,n) is aligned with that PE's registered valid
  for (genvar n = 0; n < N; n++) begin : gen_out
    assign psum_bot_o[n]  = p_w[N][n];
    assign valid_bot_o[n] = v_w[N-1][n+1];
  end

endmodule
