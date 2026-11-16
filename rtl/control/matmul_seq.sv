// MATMUL sequencer (docs/m4_compute_core.md sec 5):
//   UB rows ub_row..ub_row+B-1 --(UB read port, latency UbLatency)--> skid FIFO --> matrix_unit.x
//   matrix_unit.y (fixed latency, never back-pressured) --> ACC write (acc=0) or
//   ACC read-modify-write (acc=1).
// The RMW reads the ACC row when its y arrives (S0), adds and writes it the next cycle (S1).
// A read that hits the row being written in the same cycle (back-to-back MATMULs with B = 1,
// or B = 2 re-reading a row) takes the in-flight write value instead (forwarding). A write two or
// more cycles earlier is already in the RAM.
//
// CONVOLVE mode (docs/m6_cnn_contract.md): the UB row addresses come from conv_address_gen instead
// of the linear counter. Padded rows issue no UB read; their data ({16 x PAD_VAL}) is substituted
// at the end of the UB read latency, so row order and timing are identical to MATMUL rows. The
// ACC accumulate/forwarding and everything downstream of the skid FIFO is shared.

module matmul_seq #(
  parameter int unsigned N = 16
) (
  input  logic               clk_i,
  input  logic               rst_ni,
  // command slot
  input  logic               cmd_valid_i,
  output logic               cmd_ready_o,
  input  logic [15:0]        cmd_ub_row_i,
  input  logic [10:0]        cmd_acc_row_i,
  input  logic [15:0]        cmd_rows_i,
  input  logic               cmd_acc_i,
  input  logic               cmd_new_w_i,
  // CONVOLVE extras (cmd_conv_i = 0 for MATMUL; cmd_rows_i = B = oh_cnt * OUT_W for CONVOLVE)
  input  logic               cmd_conv_i,
  input  logic [3:0]         cmd_kh_i,
  input  logic [3:0]         cmd_kw_i,
  input  logic [11:0]        cmd_oh0_i,
  input  logic [11:0]        cmd_ohcnt_i,
  input  logic [15:0]        conv_in_w_i,
  input  logic [15:0]        conv_in_h_i,
  input  logic [15:0]        conv_out_w_i,
  input  logic [3:0]         conv_sh_i,
  input  logic [3:0]         conv_sw_i,
  input  logic [7:0]         conv_pad_t_i,
  input  logic [7:0]         conv_pad_l_i,
  input  logic [7:0]         conv_pad_val_i,
  output logic               idle_o,
  // UB read port
  output logic               ub_rd_en_o,
  output logic [15:0]        ub_rd_addr_o,
  input  logic [N*8-1:0]     ub_rd_data_i,
  // matrix unit
  output logic               x_valid_o,
  input  logic               x_ready_i,
  output logic [N*8-1:0]     x_data_o,
  output logic               x_new_w_o,
  input  logic               y_valid_i,
  input  logic [N*32-1:0]    y_data_i,
  // ACC read port (this unit has priority) and write port
  output logic               acc_rd_en_o,
  output logic [10:0]        acc_rd_addr_o,
  input  logic [N*32-1:0]    acc_rd_data_i,
  output logic               acc_wr_en_o,
  output logic [10:0]        acc_wr_addr_o,
  output logic [N*32-1:0]    acc_wr_data_o,
  // perf
  output logic               array_active_o
);

  localparam int unsigned UbLat  = tpu_pkg::UbLatency;
  localparam int unsigned FDepth = 8;          // skid FIFO, > UbLat + 1
  localparam int unsigned TDepth = 64;         // rows inside the MXU (<= 2N + 2)

  typedef struct packed {
    logic        new_w;
    logic        acc;
    logic [10:0] acc_addr;
  } meta_t;

  // ---------------------------------------------------------------------------------------------
  // Command / issue
  // ---------------------------------------------------------------------------------------------
  logic        active_q, first_q, acc_q, conv_q;
  logic [15:0] ub_q, left_q;
  logic [10:0] accr_q;
  logic [7:0]  padval_q;

  logic [$clog2(FDepth):0] f_cnt;
  logic [UbLat-1:0]        pipe_v;
  meta_t                   pipe_m [UbLat];
  logic [$clog2(UbLat+1):0] inflight;

  always_comb begin
    inflight = '0;
    for (int i = 0; i < UbLat; i++) inflight += ($clog2(UbLat+1)+1)'(pipe_v[i]);
  end

  // conv_address_gen row stream (valid only for conv commands)
  logic        cag_v, cag_pad, cag_cmd_ready, cag_idle;
  logic [15:0] cag_addr;

  wire room  = (32'(f_cnt) + 32'(inflight) < FDepth);
  wire issue = active_q && room && (!conv_q || cag_v);
  assign cmd_ready_o = !active_q;

  conv_address_gen i_cag (
    .clk_i, .rst_ni,
    .cmd_valid_i(cmd_valid_i && cmd_ready_o && cmd_conv_i), .cmd_ready_o(cag_cmd_ready),
    .cmd_ub_row_i, .cmd_kh_i, .cmd_kw_i, .cmd_oh0_i, .cmd_ohcnt_i,
    .in_w_i(conv_in_w_i), .in_h_i(conv_in_h_i), .out_w_i(conv_out_w_i),
    .sh_i(conv_sh_i), .sw_i(conv_sw_i), .pad_t_i(conv_pad_t_i), .pad_l_i(conv_pad_l_i),
    .row_valid_o(cag_v), .row_ready_i(active_q && conv_q && room),
    .row_pad_o(cag_pad), .row_addr_o(cag_addr), .idle_o(cag_idle)
  );

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      active_q <= 1'b0; first_q <= 1'b0; acc_q <= 1'b0; ub_q <= '0; left_q <= '0; accr_q <= '0;
      conv_q <= 1'b0; padval_q <= '0;
    end else if (cmd_valid_i && cmd_ready_o) begin
      active_q <= 1'b1; first_q <= cmd_new_w_i; acc_q <= cmd_acc_i;
      ub_q <= cmd_ub_row_i; left_q <= cmd_rows_i; accr_q <= cmd_acc_row_i;
      conv_q <= cmd_conv_i; padval_q <= conv_pad_val_i;
    end else if (issue) begin
      first_q <= 1'b0;
      ub_q    <= ub_q + 16'd1;
      accr_q  <= accr_q + 11'd1;
      left_q  <= left_q - 16'd1;
      if (left_q == 16'd1) active_q <= 1'b0;
    end
  end

  assign ub_rd_en_o   = issue && !(conv_q && cag_pad);
  assign ub_rd_addr_o = conv_q ? cag_addr : ub_q;

  // metadata travels alongside the UB read latency
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) pipe_v <= '0;
    else         pipe_v <= {pipe_v[UbLat-2:0], issue};
  end
  always_ff @(posedge clk_i) begin
    pipe_m[0] <= '{new_w: first_q, acc: acc_q, acc_addr: accr_q};
    for (int i = 1; i < UbLat; i++) pipe_m[i] <= pipe_m[i-1];
  end

  // pad flag and pad value travel with the row through the UB read latency (a later command may
  // change padval_q while an earlier command's rows are still in flight)
  logic [UbLat-1:0] pipe_pad;
  logic [7:0]       pipe_pv [UbLat];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) pipe_pad <= '0;
    else         pipe_pad <= {pipe_pad[UbLat-2:0], conv_q && cag_pad};
  end
  always_ff @(posedge clk_i) begin
    pipe_pv[0] <= padval_q;
    for (int i = 1; i < UbLat; i++) pipe_pv[i] <= pipe_pv[i-1];
  end
  wire [N*8-1:0] rd_data = pipe_pad[UbLat-1] ? {N{pipe_pv[UbLat-1]}} : ub_rd_data_i;

  // ---------------------------------------------------------------------------------------------
  // Skid FIFO -> matrix unit
  // ---------------------------------------------------------------------------------------------
  logic                     f_full, f_empty;
  logic [$clog2(FDepth)-1:0] f_usage;
  logic [N*8-1:0]           f_data;
  meta_t                    f_meta;
  wire x_acc = x_valid_o && x_ready_i;

  fifo_v3 #(.FALL_THROUGH(1'b0), .DATA_WIDTH(N*8 + $bits(meta_t)), .DEPTH(FDepth)) i_skid (
    .clk_i, .rst_ni, .flush_i(1'b0), .testmode_i(1'b0),
    .full_o(f_full), .empty_o(f_empty), .usage_o(f_usage),
    .data_i({pipe_m[UbLat-1], rd_data}), .push_i(pipe_v[UbLat-1]),
    .data_o({f_meta, f_data}), .pop_i(x_acc)
  );
  assign f_cnt = f_full ? ($clog2(FDepth)+1)'(FDepth) : {1'b0, f_usage};

  assign x_valid_o      = !f_empty;
  assign x_data_o       = f_data;
  assign x_new_w_o      = f_meta.new_w;
  assign array_active_o = x_acc;

  // ---------------------------------------------------------------------------------------------
  // Row tags for the y side (MXU keeps order and has fixed latency)
  // ---------------------------------------------------------------------------------------------
  logic t_full, t_empty;
  meta_t t_meta;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [$clog2(TDepth)-1:0] t_usage_unused;
  wire unused_new_w = t_meta.new_w;       // new_w only matters on the x side
  /* verilator lint_on UNUSEDSIGNAL */
  fifo_v3 #(.FALL_THROUGH(1'b0), .DATA_WIDTH($bits(meta_t)), .DEPTH(TDepth)) i_tags (
    .clk_i, .rst_ni, .flush_i(1'b0), .testmode_i(1'b0),
    .full_o(t_full), .empty_o(t_empty), .usage_o(t_usage_unused),
    .data_i(f_meta), .push_i(x_acc),
    .data_o(t_meta), .pop_i(y_valid_i)
  );

  // ---------------------------------------------------------------------------------------------
  // y -> ACC: S0 (read if accumulating), S1 (add, write)
  // ---------------------------------------------------------------------------------------------
  logic             s1_v, s1_acc, s1_fwd;
  logic [10:0]      s1_addr;
  logic [N*32-1:0]  s1_y, last_wdata;
  logic [N*32-1:0]  s1_sum;

  assign acc_rd_en_o   = y_valid_i && t_meta.acc;
  assign acc_rd_addr_o = t_meta.acc_addr;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      s1_v <= 1'b0; s1_acc <= 1'b0; s1_fwd <= 1'b0; s1_addr <= '0;
    end else begin
      s1_v    <= y_valid_i;
      s1_acc  <= t_meta.acc;
      s1_addr <= t_meta.acc_addr;
      s1_fwd  <= y_valid_i && t_meta.acc && s1_v && (s1_addr == t_meta.acc_addr);
    end
  end
  always_ff @(posedge clk_i) begin
    if (y_valid_i) s1_y <= y_data_i;
    if (s1_v)      last_wdata <= acc_wr_data_o;
  end

  always_comb begin
    for (int c = 0; c < N; c++) begin
      logic [31:0] base;
      base = s1_fwd ? last_wdata[32*c +: 32] : acc_rd_data_i[32*c +: 32];
      s1_sum[32*c +: 32] = base + s1_y[32*c +: 32];       // int32 wrap (two's complement)
    end
  end

  assign acc_wr_en_o   = s1_v;
  assign acc_wr_addr_o = s1_addr;
  assign acc_wr_data_o = s1_acc ? s1_sum : s1_y;

  assign idle_o = !active_q && (inflight == '0) && f_empty && t_empty && !s1_v && cag_idle;

  // pragma translate_off
  a_cag_idle_on_cmd: assert property (@(posedge clk_i) disable iff (!rst_ni)
      cmd_valid_i && cmd_ready_o && cmd_conv_i |-> cag_cmd_ready)
    else $error("matmul_seq: conv command accepted while conv_address_gen is busy");
  a_conv_rows_match: assert property (@(posedge clk_i) disable iff (!rst_ni)
      conv_q && !active_q |-> !cag_v)
    else $error("matmul_seq: conv_address_gen produced more rows than B");
  a_tag_ok:  assert property (@(posedge clk_i) disable iff (!rst_ni) y_valid_i |-> !t_empty)
    else $error("matmul_seq: y without a tag");
  a_tag_fit: assert property (@(posedge clk_i) disable iff (!rst_ni) x_acc |-> !t_full)
    else $error("matmul_seq: tag FIFO overflow");
  a_skid_fit: assert property (@(posedge clk_i) disable iff (!rst_ni) pipe_v[UbLat-1] |-> !f_full)
    else $error("matmul_seq: skid FIFO overflow");
  // pragma translate_on

endmodule
