// TPU-KV260 Matrix Unit (M3): N x N weight-stationary systolic array (TPU v1 sec 2, Fig. 4).
// Contract: docs/m3_matrix_unit.md.
//
//   x row accepted (cycle a) -> input register -> data_setup (lane k +k) -> array
//   -> bottom row (column n at +n) -> output_deskew (column n +N-1-n) -> y row
//   Latency: y_valid_o for a row accepted in cycle a is high in cycle a + L, L = 2N.
//
// Bubbles (x_valid_i low) travel through as invalid rows; y is never back-pressured.
// x_ready_o drops only for a 'new_weights' row whose tile is not fully loaded yet.

// SYNCASYNCNET waived: rst_ni is an async reset everywhere; it only also appears in the
// sim-only assertion 'disable iff' (sampled), which Verilator counts as a synchronous use.
/* verilator lint_off SYNCASYNCNET */
module matrix_unit #(
  parameter int unsigned N = 16
) (
  input  logic             clk_i,
  input  logic             rst_ni,
  // activation rows
  input  logic             x_valid_i,
  output logic             x_ready_o,
  input  logic [N*8-1:0]   x_data_i,
  input  logic             x_new_w_i,
  // weight words (Weight FIFO consumer)
  input  logic             w_valid_i,
  output logic             w_ready_o,
  input  logic [N*8-1:0]   w_data_i,
  // results
  output logic             y_valid_o,
  output logic [N*32-1:0]  y_data_o,
  // status / perf
  output logic             busy_o,
  output logic             weight_stall_o,
  output logic             weight_shift_o
);

  localparam int unsigned L = 2 * N;          // documented latency (cycles)

  logic tile_loaded, w_shift;
  logic x_acc;

  assign x_ready_o      = !(x_valid_i && x_new_w_i && !tile_loaded);
  assign x_acc          = x_valid_i && x_ready_o;
  assign weight_stall_o = x_valid_i && x_new_w_i && !tile_loaded;
  assign weight_shift_o = w_shift;

  weight_loader #(.N(N)) i_wload (
    .clk_i, .rst_ni,
    .w_valid_i     ( w_valid_i ),
    .w_ready_o     ( w_ready_o ),
    .w_shift_o     ( w_shift ),
    .tile_loaded_o ( tile_loaded ),
    .tile_take_i   ( x_acc && x_new_w_i )
  );

  // input register (accepted rows only; otherwise a bubble)
  logic             in_valid_q, in_swap_q;
  logic [N*8-1:0]   in_data_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      in_valid_q <= 1'b0;
      in_swap_q  <= 1'b0;
    end else begin
      in_valid_q <= x_acc;
      in_swap_q  <= x_acc && x_new_w_i;
    end
  end
  always_ff @(posedge clk_i) begin
    if (x_acc) in_data_q <= x_data_i;
  end

  logic [N-1:0][7:0] row_data;
  logic [N-1:0]      row_valid, row_swap;

  data_setup #(.N(N)) i_setup (
    .clk_i, .rst_ni,
    .data_i  ( in_data_q  ),
    .valid_i ( in_valid_q ),
    .swap_i  ( in_swap_q  ),
    .data_o  ( row_data   ),
    .valid_o ( row_valid  ),
    .swap_o  ( row_swap   )
  );

  logic [N-1:0][31:0] psum_bot;
  logic [N-1:0]       valid_bot;

  systolic_array #(.N(N)) i_array (
    .clk_i, .rst_ni,
    .a_row_i     ( row_data  ),
    .valid_row_i ( row_valid ),
    .swap_row_i  ( row_swap  ),
    .w_top_i     ( w_data_i  ),
    .w_shift_i   ( w_shift   ),
    .psum_bot_o  ( psum_bot  ),
    .valid_bot_o ( valid_bot )
  );

  logic [N-1:0][31:0] y_cols;

  output_deskew #(.N(N)) i_deskew (
    .clk_i, .rst_ni,
    .psum_i       ( psum_bot         ),
    .valid_last_i ( valid_bot[N-1]   ),
    .y_o          ( y_cols           ),
    .y_valid_o    ( y_valid_o        )
  );
  assign y_data_o = y_cols;

  // rows in flight (accepted, not yet out)
  localparam int unsigned FW = $clog2(L + 2) + 1;
  logic [FW-1:0] inflight_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) inflight_q <= '0;
    else         inflight_q <= inflight_q + FW'(x_acc) - FW'(y_valid_o);
  end
  assign busy_o = (inflight_q != '0);

  // pragma translate_off
  a_no_underflow: assert property (@(posedge clk_i) disable iff (!rst_ni)
      y_valid_o |-> (inflight_q != '0) || x_acc)
    else $error("matrix_unit: output without an accepted input");
  // pragma translate_on

endmodule
/* verilator lint_on SYNCASYNCNET */
