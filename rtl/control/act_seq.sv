// ACTIVATE sequencer (docs/m4_compute_core.md sec 5, docs/m6_cnn_contract.md sec 5):
//   ACC rows acc_row.. (ACC read port, only when matmul_seq does not use it; latency 1)
//   -> skid FIFO -> activation_unit (valid/ready) -> [pool_unit] -> UB write port (ub_row..).
// Per-row qset travels with the row; per-row {UB address, pool mode, group-last} travel in a tag
// FIFO so consecutive ACTIVATE commands overlap without draining.
//
// pool == 0: rows are read and written in order, one UB row per ACC row (unchanged since M4).
// pool != 0: `rows` pre-pool rows of a raster of width POOL_W are read in pool order, i.e. for
//   each pair of raster rows (2i, 2i+1) and each j: (2i,2j) (2i,2j+1) (2i+1,2j) (2i+1,2j+1).
//   The ACC read pointer follows that order with incremental deltas only (no multiplier, L20):
//     pos0 -> pos1: +1   pos1 -> pos2: +(PW-1)   pos2 -> pos3: +1
//     pos3 -> next group of the pair: +(1-PW)     pos3 of the last group of a pair: +1
//   (the last pixel (2i+1,PW-1) is raster-adjacent to (2i+2,0)).  The four rows of a group carry
//   the same UB address ub_row + i*(PW/2) + j, which is simply ub_row + group_index, and the
//   pool_unit emits the pooled row in the cycle the fourth row leaves the activation unit.
// UB writes never wait (native write port), so pool_unit's out_ready is tied high.

module act_seq #(
  parameter int unsigned N = 16
) (
  input  logic               clk_i,
  input  logic               rst_ni,
  // command slot
  input  logic               cmd_valid_i,
  output logic               cmd_ready_o,
  input  logic [10:0]        cmd_acc_row_i,
  input  logic [15:0]        cmd_ub_row_i,
  input  logic [15:0]        cmd_rows_i,
  input  logic [1:0]         cmd_qset_i,
  input  logic [1:0]         cmd_pool_i,        // 0 none, 1 max 2x2, 2 avg 2x2 (CNN contract sec 5)
  input  logic [15:0]        cmd_pool_w_i,      // pre-pool raster width POOL_W (pool != 0)
  output logic               idle_o,
  // ACC read port (granted only when the matmul sequencer is not reading)
  input  logic               acc_rd_gnt_i,
  output logic               acc_rd_en_o,
  output logic [10:0]        acc_rd_addr_o,
  input  logic [N*32-1:0]    acc_rd_data_i,
  // activation unit
  output logic               au_valid_o,
  input  logic               au_ready_i,
  output logic [N*32-1:0]    au_data_o,
  output logic [1:0]         au_qset_o,
  input  logic               au_out_valid_i,
  output logic               au_out_ready_o,
  input  logic [N*8-1:0]     au_out_data_i,
  input  logic               au_busy_i,
  // UB write port
  output logic               ub_wr_en_o,
  output logic [15:0]        ub_wr_addr_o,
  output logic [N*8-1:0]     ub_wr_data_o,
  // perf
  output logic               act_active_o
);

  localparam int unsigned FDepth = 4;
  localparam int unsigned TDepth = 32;

  localparam int unsigned MetaW = 21;     // {mode[1:0], last, qset[1:0], ub_addr[15:0]}

  logic        active_q;
  logic [10:0] acc_q;
  logic [15:0] ub_q, left_q;
  logic [1:0]  qset_q;
  logic [1:0]  pool_q;                    // pool mode of the running command
  logic [1:0]  pos_q;                     // position inside the current 2x2 group
  logic [10:0] jcnt_q;                    // groups left in the current raster-row pair, minus 1
  logic [10:0] half_m1_q;                 // PW/2 - 1  (jcnt reload)
  logic [10:0] d_mid_q, d_next_q;         // PW-1, 1-PW   (mod 2048: ACC addresses are 11 bits)
  logic        rd_v_q;                    // ACC read issued last cycle (data valid now)
  logic [MetaW-1:0] rd_meta_q;            // meta of that read

  logic f_full, f_empty, t_full, t_empty;
  logic [$clog2(FDepth)-1:0] f_usage;
  logic [$clog2(TDepth)-1:0] t_usage;
  logic [$clog2(FDepth):0]   f_cnt;
  logic [$clog2(TDepth):0]   t_cnt;
  assign f_cnt = f_full ? ($clog2(FDepth)+1)'(FDepth) : {1'b0, f_usage};
  assign t_cnt = t_full ? ($clog2(TDepth)+1)'(TDepth) : {1'b0, t_usage};

  // credit: a row read now needs a skid slot next cycle and a tag slot when it enters the unit
  wire want  = active_q && (32'(f_cnt) + 32'(rd_v_q) < FDepth) && (32'(t_cnt) + 32'(f_cnt) + 32'(rd_v_q) < TDepth);
  wire issue = want && acc_rd_gnt_i;
  assign cmd_ready_o = !active_q;

  wire pooling = (pool_q != 2'd0);
  wire last_of_group = !pooling || (pos_q == 2'd3);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      active_q <= 1'b0; acc_q <= '0; ub_q <= '0; left_q <= '0; qset_q <= '0;
      pool_q <= '0; pos_q <= '0; jcnt_q <= '0;
      half_m1_q <= '0; d_mid_q <= '0; d_next_q <= '0;
    end else if (cmd_valid_i && cmd_ready_o) begin
      active_q <= 1'b1; acc_q <= cmd_acc_row_i; ub_q <= cmd_ub_row_i; left_q <= cmd_rows_i;
      qset_q <= cmd_qset_i;
      pool_q <= cmd_pool_i;
      pos_q  <= 2'd0;
      half_m1_q <= cmd_pool_w_i[11:1] - 11'd1;                    // PW/2 - 1
      jcnt_q    <= cmd_pool_w_i[11:1] - 11'd1;
      d_mid_q   <= cmd_pool_w_i[10:0] - 11'd1;                    // PW - 1
      d_next_q  <= 11'd1 - cmd_pool_w_i[10:0];                    // 1 - PW (mod 2048)
    end else if (issue) begin
      left_q <= left_q - 16'd1;
      if (left_q == 16'd1) active_q <= 1'b0;
      if (!pooling) begin
        acc_q <= acc_q + 11'd1;
        ub_q  <= ub_q + 16'd1;
      end else begin
        pos_q <= pos_q + 2'd1;                                    // 3 -> 0
        unique case (pos_q)
          2'd0:    acc_q <= acc_q + 11'd1;
          2'd1:    acc_q <= acc_q + d_mid_q;
          2'd2:    acc_q <= acc_q + 11'd1;
          default: begin                                          // pos 3: group done
            ub_q <= ub_q + 16'd1;
            if (jcnt_q == 11'd0) begin
              acc_q  <= acc_q + 11'd1;                            // next raster-row pair
              jcnt_q <= half_m1_q;
            end else begin
              acc_q  <= acc_q + d_next_q;                         // next group in this pair
              jcnt_q <= jcnt_q - 11'd1;
            end
          end
        endcase
      end
    end
  end

  assign acc_rd_en_o   = issue;
  assign acc_rd_addr_o = acc_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) rd_v_q <= 1'b0;
    else         rd_v_q <= issue;
  end
  always_ff @(posedge clk_i) begin
    if (issue) rd_meta_q <= {pool_q, last_of_group, qset_q, ub_q};
  end

  // skid FIFO: {meta, row}
  logic [N*32-1:0]  f_row;
  logic [MetaW-1:0] f_meta;
  wire au_acc = au_valid_o && au_ready_i;

  fifo_v3 #(.FALL_THROUGH(1'b0), .DATA_WIDTH(MetaW + N*32), .DEPTH(FDepth)) i_skid (
    .clk_i, .rst_ni, .flush_i(1'b0), .testmode_i(1'b0),
    .full_o(f_full), .empty_o(f_empty), .usage_o(f_usage),
    .data_i({rd_meta_q, acc_rd_data_i}), .push_i(rd_v_q),
    .data_o({f_meta, f_row}), .pop_i(au_acc)
  );

  assign au_valid_o   = !f_empty;
  assign au_data_o    = f_row;
  assign au_qset_o    = f_meta[17:16];
  assign act_active_o = au_acc;

  // {mode, last, ub_addr} of every row inside the activation unit
  logic [18:0] t_meta;
  logic [1:0]  t_mode;
  logic        t_last;
  logic [15:0] t_ub;
  fifo_v3 #(.FALL_THROUGH(1'b0), .DATA_WIDTH(19), .DEPTH(TDepth)) i_tags (
    .clk_i, .rst_ni, .flush_i(1'b0), .testmode_i(1'b0),
    .full_o(t_full), .empty_o(t_empty), .usage_o(t_usage),
    .data_i({f_meta[20:18], f_meta[15:0]}), .push_i(au_acc),
    .data_o(t_meta), .pop_i(au_out_valid_i && au_out_ready_o)
  );
  assign {t_mode, t_last, t_ub} = t_meta;

  // pooling stage (rows of pool commands), pass-through for pool == 0
  logic            p_in_ready, p_out_valid, p_busy;
  logic [N*8-1:0]  p_out_data;
  pool_unit #(.N(N)) i_pool (
    .clk_i, .rst_ni,
    .in_valid_i  ( au_out_valid_i && (t_mode != 2'd0) ),
    .in_ready_o  ( p_in_ready ),
    .in_data_i   ( au_out_data_i ),
    .in_mode_i   ( t_mode ),
    .out_valid_o ( p_out_valid ),
    .out_ready_i ( 1'b1 ),                              // UB native write never waits
    .out_data_o  ( p_out_data ),
    .busy_o      ( p_busy )
  );

  assign au_out_ready_o = 1'b1;                         // UB native write never waits
  assign ub_wr_en_o     = (t_mode == 2'd0) ? au_out_valid_i : p_out_valid;
  assign ub_wr_addr_o   = t_ub;
  assign ub_wr_data_o   = (t_mode == 2'd0) ? au_out_data_i : p_out_data;

  assign idle_o = !active_q && !rd_v_q && f_empty && t_empty && !au_busy_i && !au_out_valid_i && !p_busy;

  // pragma translate_off
  a_tag_ok:  assert property (@(posedge clk_i) disable iff (!rst_ni) au_out_valid_i |-> !t_empty)
    else $error("act_seq: activation output without a tag");
  a_tag_fit: assert property (@(posedge clk_i) disable iff (!rst_ni) au_acc |-> !t_full)
    else $error("act_seq: tag FIFO overflow");
  a_skid_fit: assert property (@(posedge clk_i) disable iff (!rst_ni) rd_v_q |-> !f_full)
    else $error("act_seq: skid FIFO overflow");
  a_pool_ready: assert property (@(posedge clk_i) disable iff (!rst_ni) p_in_ready)
    else $error("act_seq: pool_unit blocked although the UB write never waits");
  // the tag's group-last flag and pool_unit's own 4-row counter must agree
  a_group_sync: assert property (@(posedge clk_i) disable iff (!rst_ni)
      (au_out_valid_i && t_mode != 2'd0) |-> (p_out_valid == t_last))
    else $error("act_seq: group framing mismatch between tags and pool_unit");
  // pragma translate_on

endmodule
