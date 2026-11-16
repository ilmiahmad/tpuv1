// Weight loader: shifts N words per tile from the Weight FIFO into the array's shadow chain.
//   tile_loaded_o : N words shifted since the last tile was consumed
//   tile_take_i   : a 'new_weights' row was accepted -> the shadow tile is being swapped in;
//                   the shadow chain must not move until the swap has reached PE(N-1,N-1),
//                   2N-2 cycles after the row enters the array (+1 input register) - guard G.
// Word order: tile word t = W[N-1-t] (bottom row first), see docs/m3_matrix_unit.md.

module weight_loader #(
  parameter int unsigned N = 16
) (
  input  logic              clk_i,
  input  logic              rst_ni,
  // Weight FIFO consumer side
  input  logic              w_valid_i,
  output logic              w_ready_o,
  // array shift chain
  output logic              w_shift_o,
  // tile handshake with the activation side
  output logic              tile_loaded_o,
  input  logic              tile_take_i
);

  // Shifts are safe from cycle a+2N-1 on (a = acceptance cycle of the swap row): the guard is
  // loaded at a and counts down; shifting is allowed when it reads 0, i.e. from a+1+G. G=2N-2
  // is the exact minimum; 2N leaves margin at negligible cost.
  localparam int unsigned G  = 2 * N;
  localparam int unsigned CW = $clog2(N + 1);
  localparam int unsigned GW = $clog2(G + 1);

  logic [CW-1:0] count_q;
  logic [GW-1:0] guard_q;
  logic          can_shift;

  assign tile_loaded_o = (count_q == CW'(N));
  assign can_shift     = (guard_q == '0) && !tile_loaded_o;
  assign w_ready_o     = can_shift;
  assign w_shift_o     = w_valid_i && can_shift;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)          count_q <= '0;
    else if (tile_take_i) count_q <= '0;
    else if (w_shift_o)   count_q <= count_q + CW'(1);
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)              guard_q <= '0;
    else if (tile_take_i)     guard_q <= GW'(G);
    else if (guard_q != '0)   guard_q <= guard_q - GW'(1);
  end

  // pragma translate_off
  // a tile can only be taken when it is fully loaded, and never while words are shifting
  a_take_loaded: assert property (@(posedge clk_i) disable iff (!rst_ni)
      tile_take_i |-> tile_loaded_o && !w_shift_o)
    else $error("weight_loader: tile taken before it was loaded");
  // pragma translate_on

endmodule
