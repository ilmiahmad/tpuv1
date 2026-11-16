// TPU-KV260 compute core, Phase A (docs/m4_compute_core.md sec 5):
//   decoder -> matmul_seq -> matrix_unit (M3) -> ACC (accumulate RMW with forwarding)
//           -> act_seq -> activation_unit -> UB
//           -> SET_CONFIG -> activation_unit parameter sets
// UB/ACC are reached through the fabric's native ports (rtl/top/tpu_fabric.sv).
// ACC read port: matmul_seq (accumulate, cannot stall) has priority over act_seq.

module tpu_core #(
  parameter int unsigned N = 16,
  parameter bit EnableDma = 1'b0,
  parameter int unsigned DmaOutstanding = 8
) (
  input  logic               clk_i,
  input  logic               rst_ni,
  // Instruction FIFO
  input  logic               if_valid_i,
  output logic               if_ready_o,
  input  logic [127:0]       if_data_i,
  // Weight FIFO -> matrix_unit
  input  logic               wf_valid_i,
  output logic               wf_ready_o,
  input  logic [127:0]       wf_data_i,
  // UB native ports
  output logic               ub_rd_en_o,
  output logic [15:0]        ub_rd_addr_o,
  input  logic [N*8-1:0]     ub_rd_data_i,
  output logic               ub_wr_en_o,
  output logic [15:0]        ub_wr_addr_o,
  output logic [N*8-1:0]     ub_wr_data_o,
  output logic [N-1:0]       ub_wr_be_o,
  // ACC native ports
  output logic               acc_rd_en_o,
  output logic [10:0]        acc_rd_addr_o,
  input  logic [N*32-1:0]    acc_rd_data_i,
  output logic               acc_wr_en_o,
  output logic [10:0]        acc_wr_addr_o,
  output logic [N*32-1:0]    acc_wr_data_o,
  output logic [3:0]         acc_wr_be_o,
  // control / status
  input  logic               run_i,
  input  logic               err_clr_i,
  input  logic               perf_clr_i,
  // Optional Phase B DMA descriptor/completion stream to idma_subsystem.
  output logic               dma_req_valid_o,
  input  logic               dma_req_ready_i,
  output logic [63:0]        dma_src_addr_o,
  output logic [63:0]        dma_dst_addr_o,
  output logic [23:0]        dma_length_o,
  output logic [31:0]        dma_reps_o,
  output logic [63:0]        dma_src_stride_o,
  output logic [63:0]        dma_dst_stride_o,
  input  logic               dma_rsp_valid_i,
  output logic               dma_rsp_ready_o,
  input  logic               dma_rsp_error_i,
  input  logic               dma_bus_error_i,
  output logic               dma_error_o,
  output logic               busy_o,
  output logic               halted_o,
  output logic               cfg_err_o,
  output logic [7:0]         cfg_err_op_o,
  output logic               irq_o,
  output logic [31:0]        irq_tag_o,
  output logic [31:0]        debug_tag_o,
  output logic [31:0]        perf_o [8]
);

  // decoder <-> units
  logic        mm_v, mm_r, mm_acc, mm_new_w, mm_idle;
  logic [15:0] mm_ub, mm_rows;
  logic [10:0] mm_accr;
  logic        act_v, act_r, act_idle;
  logic [10:0] act_accr;
  logic [15:0] act_ub, act_rows;
  logic [1:0]  act_q, act_pool;
  logic [15:0] act_pool_w;
  // CONVOLVE (CNN contract sec 6)
  logic        mm_conv;
  logic [3:0]  mm_conv_kh, mm_conv_kw, conv_sh, conv_sw;
  logic [11:0] mm_conv_oh0, mm_conv_ohcnt;
  logic [15:0] conv_in_w, conv_in_h, conv_out_w;
  logic [7:0]  conv_pad_t, conv_pad_l, conv_pad_val;
  logic        cfg_we;
  logic [15:0] cfg_target;
  logic [63:0] cfg_value;
  logic        dec_busy, sync_wait, retire;
  logic        dma_capacity, dma_idle, dma_fault, dma_ub_read_busy, dma_ub_write_busy;
  logic        dec_dma_valid, dec_dma_ready;
  logic [1:0]  dec_dma_kind;

  assign dma_req_valid_o = dec_dma_valid;
  if (EnableDma) begin : gen_dma_tracking
    logic kind_full, kind_empty;
    logic [1:0] kind_head;
    logic ub_read_busy_q, ub_write_busy_q, dma_error_q;
    logic issue, complete;
    assign dma_capacity = !kind_full;
    assign dec_dma_ready = dma_req_ready_i && !kind_full;
    assign issue = dma_req_valid_o && dec_dma_ready;
    assign dma_rsp_ready_o = !kind_empty;
    assign complete = dma_rsp_valid_i && dma_rsp_ready_o;
    assign dma_idle = kind_empty;
    assign dma_ub_read_busy = ub_read_busy_q;
    assign dma_ub_write_busy = ub_write_busy_q;
    // Include the current error response to prevent same-cycle success IRQ.
    assign dma_fault = dma_error_q || dma_bus_error_i || (dma_rsp_valid_i && dma_rsp_error_i);
    assign dma_error_o = dma_error_q;

    cc_fifo #(.Depth(DmaOutstanding), .data_t(logic [1:0])) i_kinds (
      .clk_i, .rst_ni, .clr_i(1'b0), .flush_i(1'b0),
      .full_o(kind_full), .empty_o(kind_empty), .usage_o(),
      .data_i(dec_dma_kind), .push_i(issue), .data_o(kind_head), .pop_i(complete)
    );
    // At most one UB transfer is admitted at a time. Keep its ownership until
    // completion, including when it is queued behind independent weight traffic.
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        ub_read_busy_q <= 1'b0;
        ub_write_busy_q <= 1'b0;
      end else begin
        if (complete && kind_head == 2) ub_read_busy_q <= 1'b0;
        if (complete && kind_head == 0) ub_write_busy_q <= 1'b0;
        if (issue && dec_dma_kind == 2) ub_read_busy_q <= 1'b1;
        if (issue && dec_dma_kind == 0) ub_write_busy_q <= 1'b1;
      end
    end
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) dma_error_q <= 1'b0;
      // A software clear does not flush accepted or already offered requests.
      // The shared bus monitor is cleared by the same software pulse.
      else if (err_clr_i && kind_empty && !dma_req_valid_o) dma_error_q <= 1'b0;
      else if (dma_bus_error_i || (complete && dma_rsp_error_i)) dma_error_q <= 1'b1;
    end

    // pragma translate_off
    initial assert (DmaOutstanding > 0) else $fatal(1, "DmaOutstanding must be positive");
    a_dma_completion_has_kind: assert property (@(posedge clk_i) disable iff (!rst_ni)
      dma_rsp_valid_i |-> !kind_empty);
    a_dma_has_capacity: assert property (@(posedge clk_i) disable iff (!rst_ni)
      dma_req_valid_o && dma_req_ready_i |-> !kind_full);
    a_no_ub_dma_overlap: assert property (@(posedge clk_i) disable iff (!rst_ni)
      !(ub_read_busy_q && ub_write_busy_q));
    a_ub_dma_write_excludes_native: assert property (@(posedge clk_i) disable iff (!rst_ni)
      ub_write_busy_q |-> !(ub_rd_en_o || ub_wr_en_o));
    a_ub_dma_read_excludes_native_write: assert property (@(posedge clk_i) disable iff (!rst_ni)
      ub_read_busy_q |-> !ub_wr_en_o);
    // pragma translate_on
  end else begin : gen_no_dma
    assign dma_capacity = 1'b1;
    assign dec_dma_ready = 1'b0;
    assign dma_rsp_ready_o = 1'b0;
    assign dma_idle = 1'b1;
    assign dma_ub_read_busy = 1'b0;
    assign dma_ub_write_busy = 1'b0;
    assign dma_fault = 1'b0;
    assign dma_error_o = 1'b0;
  end

  decoder #(.EnableDma(EnableDma)) i_dec (
    .clk_i, .rst_ni,
    .if_valid_i, .if_ready_o, .if_data_i,
    .run_i, .err_clr_i,
    .dma_req_valid_o(dec_dma_valid), .dma_req_ready_i(dec_dma_ready), .dma_capacity_i(dma_capacity),
    .dma_src_addr_o, .dma_dst_addr_o, .dma_length_o, .dma_reps_o,
    .dma_src_stride_o, .dma_dst_stride_o, .dma_kind_o(dec_dma_kind),
    .dma_idle_i(dma_idle), .dma_ub_read_busy_i(dma_ub_read_busy),
    .dma_ub_write_busy_i(dma_ub_write_busy), .dma_fault_i(dma_fault),
    .mm_valid_o(mm_v), .mm_ready_i(mm_r), .mm_ub_row_o(mm_ub), .mm_acc_row_o(mm_accr),
    .mm_rows_o(mm_rows), .mm_acc_o(mm_acc), .mm_new_w_o(mm_new_w), .mm_idle_i(mm_idle),
    .mm_conv_o(mm_conv), .mm_conv_kh_o(mm_conv_kh), .mm_conv_kw_o(mm_conv_kw),
    .mm_conv_oh0_o(mm_conv_oh0), .mm_conv_ohcnt_o(mm_conv_ohcnt),
    .conv_in_w_o(conv_in_w), .conv_in_h_o(conv_in_h), .conv_out_w_o(conv_out_w),
    .conv_sh_o(conv_sh), .conv_sw_o(conv_sw), .conv_pad_t_o(conv_pad_t), .conv_pad_l_o(conv_pad_l),
    .conv_pad_val_o(conv_pad_val),
    .act_valid_o(act_v), .act_ready_i(act_r), .act_acc_row_o(act_accr), .act_ub_row_o(act_ub),
    .act_rows_o(act_rows), .act_qset_o(act_q), .act_pool_o(act_pool), .act_pool_w_o(act_pool_w),
    .act_idle_i(act_idle),
    .cfg_we_o(cfg_we), .cfg_target_o(cfg_target), .cfg_value_o(cfg_value),
    .busy_o(dec_busy), .halted_o, .cfg_err_o, .cfg_err_op_o,
    .irq_o, .irq_tag_o, .debug_tag_o,
    .sync_wait_o(sync_wait), .retire_o(retire)
  );

  // matrix unit + its sequencer
  logic             x_v, x_r, x_new_w, y_v, mxu_busy, w_stall, w_shift, arr_active;
  logic [N*8-1:0]   x_d;
  logic [N*32-1:0]  y_d;
  logic             mm_acc_rd_en;
  logic [10:0]      mm_acc_rd_addr;

  matmul_seq #(.N(N)) i_mm (
    .clk_i, .rst_ni,
    .cmd_valid_i(mm_v), .cmd_ready_o(mm_r), .cmd_ub_row_i(mm_ub), .cmd_acc_row_i(mm_accr),
    .cmd_rows_i(mm_rows), .cmd_acc_i(mm_acc), .cmd_new_w_i(mm_new_w),
    .cmd_conv_i(mm_conv), .cmd_kh_i(mm_conv_kh), .cmd_kw_i(mm_conv_kw),
    .cmd_oh0_i(mm_conv_oh0), .cmd_ohcnt_i(mm_conv_ohcnt),
    .conv_in_w_i(conv_in_w), .conv_in_h_i(conv_in_h), .conv_out_w_i(conv_out_w),
    .conv_sh_i(conv_sh), .conv_sw_i(conv_sw), .conv_pad_t_i(conv_pad_t), .conv_pad_l_i(conv_pad_l),
    .conv_pad_val_i(conv_pad_val), .idle_o(mm_idle),
    .ub_rd_en_o, .ub_rd_addr_o, .ub_rd_data_i,
    .x_valid_o(x_v), .x_ready_i(x_r), .x_data_o(x_d), .x_new_w_o(x_new_w),
    .y_valid_i(y_v), .y_data_i(y_d),
    .acc_rd_en_o(mm_acc_rd_en), .acc_rd_addr_o(mm_acc_rd_addr), .acc_rd_data_i,
    .acc_wr_en_o, .acc_wr_addr_o, .acc_wr_data_o,
    .array_active_o(arr_active)
  );
  assign acc_wr_be_o = '1;

  matrix_unit #(.N(N)) i_mxu (
    .clk_i, .rst_ni,
    .x_valid_i(x_v), .x_ready_o(x_r), .x_data_i(x_d), .x_new_w_i(x_new_w),
    .w_valid_i(wf_valid_i), .w_ready_o(wf_ready_o), .w_data_i(wf_data_i[N*8-1:0]),
    .y_valid_o(y_v), .y_data_o(y_d),
    .busy_o(mxu_busy), .weight_stall_o(w_stall), .weight_shift_o(w_shift)
  );

  // activation path
  logic             act_acc_rd_en, au_v, au_r, au_ov, au_or, au_busy, act_active, au_cfg_err;
  logic [10:0]      act_acc_rd_addr;
  logic [N*32-1:0]  au_d;
  logic [1:0]       au_q;
  logic [N*8-1:0]   au_od;

  act_seq #(.N(N)) i_act (
    .clk_i, .rst_ni,
    .cmd_valid_i(act_v), .cmd_ready_o(act_r), .cmd_acc_row_i(act_accr), .cmd_ub_row_i(act_ub),
    .cmd_rows_i(act_rows), .cmd_qset_i(act_q), .cmd_pool_i(act_pool), .cmd_pool_w_i(act_pool_w),
    .idle_o(act_idle),
    .acc_rd_gnt_i(!mm_acc_rd_en), .acc_rd_en_o(act_acc_rd_en), .acc_rd_addr_o(act_acc_rd_addr),
    .acc_rd_data_i,
    .au_valid_o(au_v), .au_ready_i(au_r), .au_data_o(au_d), .au_qset_o(au_q),
    .au_out_valid_i(au_ov), .au_out_ready_o(au_or), .au_out_data_i(au_od), .au_busy_i(au_busy),
    .ub_wr_en_o, .ub_wr_addr_o, .ub_wr_data_o,
    .act_active_o(act_active)
  );
  assign ub_wr_be_o = '1;

  activation_unit #(.N(N)) i_au (
    .clk_i, .rst_ni,
    .cfg_we_i(cfg_we), .cfg_target_i(cfg_target), .cfg_value_i(cfg_value), .cfg_err_o(au_cfg_err),
    .in_valid_i(au_v), .in_ready_o(au_r), .in_data_i(au_d), .in_qset_i(au_q),
    .out_valid_o(au_ov), .out_ready_i(au_or), .out_data_o(au_od), .busy_o(au_busy)
  );

  // ACC read port arbitration: accumulate reads first (y cannot be back-pressured)
  assign acc_rd_en_o   = mm_acc_rd_en || act_acc_rd_en;
  assign acc_rd_addr_o = mm_acc_rd_en ? mm_acc_rd_addr : act_acc_rd_addr;

  assign busy_o = dec_busy || !mm_idle || !act_idle || mxu_busy || !dma_idle;

  // performance counters (TPU v1 Table 3)
  perf_counters i_perf (
    .clk_i, .rst_ni, .clr_i(perf_clr_i),
    .inc_i({retire, act_active, sync_wait, !arr_active && busy_o, w_shift, w_stall, arr_active, 1'b1}),
    .cnt_o(perf_o)
  );

  // pragma translate_off
  a_acc_rd_excl: assert property (@(posedge clk_i) disable iff (!rst_ni) !(mm_acc_rd_en && act_acc_rd_en))
    else $error("tpu_core: ACC read port double grant");
  a_no_au_cfg_err: assert property (@(posedge clk_i) disable iff (!rst_ni) !au_cfg_err)
    else $error("tpu_core: activation_unit rejected a SET_CONFIG the decoder accepted");
  // pragma translate_on

endmodule
