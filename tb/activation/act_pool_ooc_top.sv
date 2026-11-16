// Synthesis wrapper only (not a testbench): act_seq + activation_unit connected exactly as in
// tpu_core, so out-of-context Vivado sees the real ACC-read -> requant -> pool -> UB-write paths.
module act_pool_ooc_top #(
  parameter int unsigned N = 16
) (
  input  logic               clk_i,
  input  logic               rst_ni,
  input  logic               cmd_valid_i,
  output logic               cmd_ready_o,
  input  logic [10:0]        cmd_acc_row_i,
  input  logic [15:0]        cmd_ub_row_i,
  input  logic [15:0]        cmd_rows_i,
  input  logic [1:0]         cmd_qset_i,
  input  logic [1:0]         cmd_pool_i,
  input  logic [15:0]        cmd_pool_w_i,
  output logic               idle_o,
  input  logic               acc_rd_gnt_i,
  output logic               acc_rd_en_o,
  output logic [10:0]        acc_rd_addr_o,
  input  logic [N*32-1:0]    acc_rd_data_i,
  input  logic               cfg_we_i,
  input  logic [15:0]        cfg_target_i,
  input  logic [63:0]        cfg_value_i,
  output logic               cfg_err_o,
  output logic               ub_wr_en_o,
  output logic [15:0]        ub_wr_addr_o,
  output logic [N*8-1:0]     ub_wr_data_o,
  output logic               act_active_o
);
  logic au_v, au_r, au_ov, au_or, au_busy;
  logic [N*32-1:0] au_d;
  logic [1:0] au_q;
  logic [N*8-1:0] au_od;

  act_seq #(.N(N)) i_seq (
    .clk_i, .rst_ni,
    .cmd_valid_i, .cmd_ready_o, .cmd_acc_row_i, .cmd_ub_row_i, .cmd_rows_i, .cmd_qset_i,
    .cmd_pool_i, .cmd_pool_w_i, .idle_o,
    .acc_rd_gnt_i, .acc_rd_en_o, .acc_rd_addr_o, .acc_rd_data_i,
    .au_valid_o(au_v), .au_ready_i(au_r), .au_data_o(au_d), .au_qset_o(au_q),
    .au_out_valid_i(au_ov), .au_out_ready_o(au_or), .au_out_data_i(au_od), .au_busy_i(au_busy),
    .ub_wr_en_o, .ub_wr_addr_o, .ub_wr_data_o, .act_active_o
  );

  activation_unit #(.N(N)) i_au (
    .clk_i, .rst_ni,
    .cfg_we_i, .cfg_target_i, .cfg_value_i, .cfg_err_o,
    .in_valid_i(au_v), .in_ready_o(au_r), .in_data_i(au_d), .in_qset_i(au_q),
    .out_valid_o(au_ov), .out_ready_i(au_or), .out_data_o(au_od), .busy_o(au_busy)
  );
endmodule
