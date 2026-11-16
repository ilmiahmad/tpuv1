// Instruction decoder (Phase A by default; optional Phase B DMA opcodes).
// Pops one 16-byte instruction from the Instruction FIFO, checks it, and dispatches it in order
// to a unit with a free 1-entry command slot (TPU v1 4-stage CISC pipeline):
//   MATMUL / CONVOLVE -> matmul_seq   ACTIVATE (pool 0/1/2) -> act_seq
//   SET_CONFIG kinds 1-3 -> activation_unit (after it drains); kind 4 (conv/pool geometry,
//   docs/m6_cnn_contract.md sec 3) is handled here and never reaches the activation unit
//   SYNC / INTERRUPT_HOST wait for units to drain   DEBUG_TAG, NOP retire   HALT stops fetching
// Any illegal instruction: not executed, cfg_err latched (sticky, with opcode), fetching stops
// until err_clr_i.

module decoder #(
  parameter bit EnableDma = 1'b0
) (
  input  logic          clk_i,
  input  logic          rst_ni,
  // Instruction FIFO
  input  logic          if_valid_i,
  output logic          if_ready_o,
  input  logic [127:0]  if_data_i,
  // control
  input  logic          run_i,
  input  logic          err_clr_i,
  // Phase B DMA command stream. The instruction retires on request acceptance.
  output logic          dma_req_valid_o,
  input  logic          dma_req_ready_i,
  input  logic          dma_capacity_i,
  output logic [63:0]   dma_src_addr_o,
  output logic [63:0]   dma_dst_addr_o,
  output logic [23:0]   dma_length_o,
  output logic [31:0]   dma_reps_o,
  output logic [63:0]   dma_src_stride_o,
  output logic [63:0]   dma_dst_stride_o,
  output logic [1:0]    dma_kind_o,       // 0: DDR->UB, 1: weights, 2: UB->DDR
  input  logic          dma_idle_i,
  input  logic          dma_ub_read_busy_i,
  input  logic          dma_ub_write_busy_i,
  input  logic          dma_fault_i,
  // MATMUL command slot
  output logic          mm_valid_o,
  input  logic          mm_ready_i,
  output logic [15:0]   mm_ub_row_o,
  output logic [10:0]   mm_acc_row_o,
  output logic [15:0]   mm_rows_o,          // MATMUL: B; CONVOLVE: B = oh_cnt * OUT_W
  output logic          mm_acc_o,
  output logic          mm_new_w_o,
  input  logic          mm_idle_i,
  // CONVOLVE extras on the MATMUL slot + static geometry registers (CNN contract sec 6)
  output logic          mm_conv_o,
  output logic [3:0]    mm_conv_kh_o,
  output logic [3:0]    mm_conv_kw_o,
  output logic [11:0]   mm_conv_oh0_o,
  output logic [11:0]   mm_conv_ohcnt_o,
  output logic [15:0]   conv_in_w_o,
  output logic [15:0]   conv_in_h_o,
  output logic [15:0]   conv_out_w_o,
  output logic [3:0]    conv_sh_o,
  output logic [3:0]    conv_sw_o,
  output logic [7:0]    conv_pad_t_o,
  output logic [7:0]    conv_pad_l_o,
  output logic [7:0]    conv_pad_val_o,
  // ACTIVATE command slot
  output logic          act_valid_o,
  input  logic          act_ready_i,
  output logic [10:0]   act_acc_row_o,
  output logic [15:0]   act_ub_row_o,
  output logic [15:0]   act_rows_o,
  output logic [1:0]    act_qset_o,
  output logic [1:0]    act_pool_o,         // 0 none, 1 max 2x2, 2 avg 2x2
  output logic [15:0]   act_pool_w_o,       // POOL_W register (pre-pool raster width)
  input  logic          act_idle_i,
  // SET_CONFIG -> activation_unit parameter port
  output logic          cfg_we_o,
  output logic [15:0]   cfg_target_o,
  output logic [63:0]   cfg_value_o,
  // status
  output logic          busy_o,           // instruction held in the decoder
  output logic          halted_o,
  output logic          cfg_err_o,
  output logic [7:0]    cfg_err_op_o,
  output logic          irq_o,
  output logic [31:0]   irq_tag_o,
  output logic [31:0]   debug_tag_o,
  output logic          sync_wait_o,      // perf: blocked in SYNC
  output logic          retire_o          // perf: one instruction retired
);

  localparam logic [7:0] OP_NOP = 8'h00, OP_RHM = 8'h01, OP_RW = 8'h02, OP_WHM = 8'h03,
                         OP_MATMUL = 8'h10, OP_CONV = 8'h11, OP_ACT = 8'h20, OP_SETCFG = 8'h30,
                         OP_SYNC = 8'h31, OP_IRQ = 8'h32, OP_DBG = 8'h3E, OP_HALT = 8'h3F;

  logic         inst_v;
  logic [127:0] inst;
  logic         halted_q, err_q, run_q;
  logic         dma_offered_q;
  logic         irq_pulse_q;
  wire fault = EnableDma && dma_fault_i;

  wire fetch_en = run_i && !halted_q && !err_q && !fault;
  assign if_ready_o = fetch_en && !inst_v;

  // ---------------------------------------------------------------------------------------------
  // Field decode and legality
  // ---------------------------------------------------------------------------------------------
  wire [7:0] op = inst[7:0];
  wire is_dma = (op == OP_RHM) || (op == OP_RW) || (op == OP_WHM);

  /* verilator lint_off UNUSEDSIGNAL */
  function automatic logic set_target_ok(input logic [15:0] t);   // t[1:0]/t[3:0] = q/c (any)
    case (t[15:12])
      4'h1, 4'h2: return (t[11:6] == '0);                 // q = t[5:4] (0..3), c = t[3:0]
      4'h3:       return (t[11:2] == '0);                 // q = t[1:0]
      4'h4:       return (t[11:2] == '0);                 // 0x4000..0x4003 conv / pool geometry
      default:    return 1'b0;
    endcase
  endfunction
  /* verilator lint_on UNUSEDSIGNAL */

  // range checks (main-thread contract extension): rows must stay inside UB (65536) / ACC (2048)
  wire [16:0] mm_ub_end   = {1'b0, inst[31:16]} + {1'b0, inst[63:48]};
  wire [16:0] mm_acc_end  = {6'd0, inst[42:32]} + {1'b0, inst[63:48]};
  wire [16:0] act_acc_end = {6'd0, inst[26:16]} + {1'b0, inst[63:48]};
  wire [16:0] act_ub_end  = (inst[15:13] != 3'd0) ? ({1'b0, inst[47:32]} + {3'd0, inst[63:50]})
                                                  : ({1'b0, inst[47:32]} + {1'b0, inst[63:48]});
  wire mm_range_ok  = (mm_ub_end  <= 17'd65536) && (mm_acc_end  <= 17'd2048);
  wire act_range_ok = (act_ub_end <= 17'd65536) && (act_acc_end <= 17'd2048);

  // Pipeline DMA legality so neither a 12x24 stride product nor its wide end
  // address comparison lies in the issue path. Widths are explicit (L15/L20).
  logic [1:0] dma_check_q;
  (* use_dsp = "yes" *) logic [23:0] dma_rows_q;
  (* use_dsp = "yes" *) logic [35:0] dma_stride_rows_q;
  logic dma_legal_q;
  wire [11:0] dma_rows_minus_one = inst[100:89] - 12'd1;
  wire [24:0] dma_ub_end = {9'd0, inst[31:16]} + {1'b0, dma_rows_q};
  wire [48:0] dma_ddr_addr = {inst[76:32], 4'd0};
  wire [40:0] dma_ddr_span = {1'b0, dma_stride_rows_q, 4'd0} +
                              {25'd0, inst[88:77], 4'd0};
  wire [49:0] dma_ddr_end = {1'b0, dma_ddr_addr} + {9'd0, dma_ddr_span};
  wire [49:0] weight_ddr_end = {1'b0, dma_ddr_addr} + {26'd0, inst[31:16], 8'd0};

  function automatic logic ddr_span_ok(input logic [48:0] first, input logic [49:0] limit);
    return ((first < 49'h8000_0000) && (limit <= 50'h8000_0000)) ||
           ((first >= 49'h8_0000_0000) && (first < 49'h8_8000_0000) &&
            (limit <= 50'h8_8000_0000));
  endfunction

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      dma_check_q <= '0;
      dma_rows_q <= '0;
      dma_stride_rows_q <= '0;
      dma_legal_q <= 1'b0;
    end else if ((if_valid_i && if_ready_o) || !inst_v) begin
      dma_check_q <= '0;
    end else if (EnableDma && is_dma) begin
      if (dma_check_q == 0) begin
        dma_rows_q <= 24'(inst[88:77]) * 24'(inst[100:89]);
        dma_stride_rows_q <= 36'(dma_rows_minus_one) * 36'(inst[124:101]);
        dma_check_q <= 1;
      end else if (dma_check_q == 1) begin
        if (op == OP_RW)
          dma_legal_q <= (inst[15:8] == 0) && (inst[127:77] == 0) &&
                         (inst[31:16] != 0) && ddr_span_ok(dma_ddr_addr, weight_ddr_end);
        else
          dma_legal_q <= (inst[15:8] == 0) && (inst[127:125] == 0) &&
                         (inst[88:77] != 0) && (inst[100:89] != 0) &&
                         (dma_ub_end <= 25'd65536) && ddr_span_ok(dma_ddr_addr, dma_ddr_end);
        dma_check_q <= 2;
      end
    end
  end
  wire dma_check_pending = EnableDma && is_dma && (dma_check_q != 2);

  // ---------------------------------------------------------------------------------------------
  // CNN geometry registers (SET_CONFIG kind 4) - reset on reset only
  // ---------------------------------------------------------------------------------------------
  wire is_setcfg4   = (op == OP_SETCFG) && (inst[31:28] == 4'h4);
  wire [1:0] cfg4_i = inst[17:16];
  logic [15:0] in_w_q, in_h_q, out_w_q, out_h_q, pool_w_q;
  logic [3:0]  sh_q, sw_q;
  logic [7:0]  pad_t_q, pad_l_q, pad_val_q;

  // ---------------------------------------------------------------------------------------------
  // CONVOLVE / pooled-ACTIVATE legality: registered multi-cycle checks (cp_q), like the DMA checks.
  //   CONVOLVE : cp_q 0 -> products B = oh_cnt*OUT_W and IN_H*IN_W; 1 -> legality; 2 = done
  //   pool ACT : cp_q 0 -> init; 1..12 -> rows mod (2*POOL_W), one bit per cycle; 13 -> legality;
  //              14 = done (a division would put a long path in the issue logic)
  // ---------------------------------------------------------------------------------------------
  wire is_conv     = (op == OP_CONV);
  wire pool_chk    = (op == OP_ACT) && ((inst[15:13] == 3'd1) || (inst[15:13] == 3'd2));
  logic [3:0]  cp_q;
  (* use_dsp = "yes" *) logic [27:0] conv_b_q;
  (* use_dsp = "yes" *) logic [31:0] conv_img_q;
  logic        conv_legal_q, pool_legal_q;
  logic [12:0] pool_rem_q;
  wire cp_done = is_conv ? (cp_q == 4'd2) : (cp_q == 4'd14);
  wire cp_pending = (is_conv || pool_chk) && !cp_done;

  wire conv_geom_ok = (in_w_q != 0) && (in_h_q != 0) && (out_w_q != 0) && (out_h_q != 0) &&
                      (sh_q != 0) && (sw_q != 0) && (pad_t_q <= 8'd15) && (pad_l_q <= 8'd15);
  wire [12:0] conv_oh_end = {1'b0, inst[75:64]} + {1'b0, inst[87:76]};
  wire [27:0] conv_acc_end = {17'd0, inst[42:32]} + conv_b_q;
  wire [32:0] conv_ub_end  = {17'd0, inst[31:16]} + {1'b0, conv_img_q};
  wire        conv_fields_ok = (inst[15:10] == '0) && (inst[47:43] == '0) && (inst[63:56] == '0) &&
                               (inst[127:88] == '0) && (inst[87:76] != '0) &&
                               ({4'd0, conv_oh_end} <= {1'b0, out_h_q});

  wire [12:0] pool_d = {1'b0, pool_w_q[10:0], 1'b0};            // 2*POOL_W (POOL_W <= 1024 checked)
  wire [15:0] pool_rows_f   = inst[63:48];
  wire [3:0]  pool_bit_idx  = 4'd12 - cp_q;                      // cp_q 1..12 -> bit 11 .. 0
  wire        pool_bit      = pool_rows_f[pool_bit_idx];
  wire [12:0] pool_rem_shift = {pool_rem_q[11:0], pool_bit};
  wire pool_w_ok = (pool_w_q >= 16'd2) && !pool_w_q[0] && (pool_w_q <= 16'd1024);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      cp_q <= '0; conv_b_q <= '0; conv_img_q <= '0; conv_legal_q <= 1'b0;
      pool_legal_q <= 1'b0; pool_rem_q <= '0;
    end else if ((if_valid_i && if_ready_o) || !inst_v) begin
      cp_q <= '0;
    end else if (is_conv) begin
      if (cp_q == 4'd0) begin
        conv_b_q   <= 28'(inst[87:76]) * 28'(out_w_q);
        conv_img_q <= 32'(in_h_q) * 32'(in_w_q);
        cp_q <= 4'd1;
      end else if (cp_q == 4'd1) begin
        conv_legal_q <= conv_fields_ok && conv_geom_ok && (conv_acc_end <= 28'd2048) &&
                        (conv_ub_end <= 33'd65536);
        cp_q <= 4'd2;
      end
    end else if (pool_chk && (cp_q != 4'd14)) begin
      cp_q <= cp_q + 4'd1;
      if (cp_q == 4'd0) pool_rem_q <= '0;
      else if (cp_q <= 4'd12)
        pool_rem_q <= (pool_rem_shift >= pool_d) ? (pool_rem_shift - pool_d) : pool_rem_shift;
      else if (cp_q == 4'd13)
        pool_legal_q <= (pool_rem_q == '0) && pool_w_ok;
    end
  end

  logic legal;
  always_comb begin
    case (op)
      OP_NOP, OP_HALT: legal = (inst[127:8] == '0);
      OP_MATMUL:       legal = (inst[15:10] == '0) && (inst[47:43] == '0) && (inst[127:64] == '0)
                               && (inst[63:48] != '0) && mm_range_ok;
      OP_ACT:          legal = (inst[11:8] <= 4'd2) && !inst[12] && (inst[15:13] <= 3'd2)
                               && ((inst[15:13] == '0) || (cp_done && pool_legal_q))
                               && (inst[31:27] == '0) && (inst[63:48] != '0)
                               && (inst[67:64] < 4'd4) && (inst[71:68] == '0)
                               && (inst[127:88] == '0) && act_range_ok;   // [87:72] ignored unless eltwise
      OP_SETCFG:       legal = (inst[15:8] == '0) && (inst[127:96] == '0) && set_target_ok(inst[31:16])
                               && (!is_setcfg4 || ((inst[95:64] == '0) &&
                                   ((cfg4_i != 2'd3) || (inst[63:48] == '0))));
      OP_SYNC:         legal = (inst[15:12] == '0) && (inst[127:16] == '0);
      OP_IRQ, OP_DBG:  legal = (inst[15:8] == '0) && (inst[31:16] == '0) && (inst[127:64] == '0);
      OP_RHM, OP_RW, OP_WHM: legal = EnableDma && (dma_check_q == 2) && dma_legal_q;
      OP_CONV:         legal = cp_done && conv_legal_q;
      default:         legal = 1'b0;                      // unknown opcode
    endcase
  end

  // Coarse UB interlocks protect all UB rows while an accepted DMA uses them.
  // Weight traffic is independent and can overlap matrix execution.
  wire ub_dma_busy = dma_ub_read_busy_i || dma_ub_write_busy_i;
  wire mm_dma_ok = !EnableDma || !dma_ub_write_busy_i;
  wire act_dma_ok = !EnableDma || !ub_dma_busy;
  wire dma_issue_ok = (op == OP_RW) ||
                     ((op == OP_RHM) && mm_idle_i && act_idle_i && !ub_dma_busy) ||
                     ((op == OP_WHM) && act_idle_i && !ub_dma_busy);
  wire [3:0] smask = inst[11:8];
  wire dma_drained = !EnableDma || dma_idle_i;
  wire sync_ok = (!smask[0] || dma_drained) && (!smask[1] || mm_idle_i) &&
                 (!smask[2] || mm_idle_i) && (!smask[3] || act_idle_i);
  wire all_idle = dma_drained && mm_idle_i && act_idle_i;

  // dispatch decision for the held instruction
  logic go;
  always_comb begin
    go = 1'b0;
    if (inst_v && legal && !fault) begin
      case (op)
        OP_RHM, OP_RW, OP_WHM: go = dma_req_valid_o && dma_req_ready_i;
        OP_MATMUL, OP_CONV: go = mm_ready_i && mm_dma_ok;
        OP_ACT:    go = act_ready_i && act_dma_ok;
        OP_SETCFG: go = is_setcfg4 ? ((cfg4_i == 2'd3) ? act_idle_i : mm_idle_i)  // geometry changes
                                   : act_idle_i;          // only between rows / commands
        OP_SYNC:   go = sync_ok;
        OP_IRQ:    go = all_idle;
        default:   go = 1'b1;                             // NOP, DEBUG_TAG, HALT
      endcase
    end
    // Once valid has been sampled while stalled, honor the request handshake
    // even if a later fault stops the program. Cancelling it would violate the
    // transport contract. No subsequent instruction is fetched or dispatched.
    if (dma_offered_q && dma_req_valid_o && dma_req_ready_i) go = 1'b1;
  end
  wire bad = inst_v && !dma_check_pending && !cp_pending && !legal && !fault;

  assign mm_valid_o   = go && ((op == OP_MATMUL) || (op == OP_CONV));
  assign mm_acc_o     = inst[8];
  assign mm_new_w_o   = inst[9];
  assign mm_ub_row_o  = inst[31:16];
  assign mm_acc_row_o = inst[42:32];
  assign mm_rows_o    = is_conv ? conv_b_q[15:0] : inst[63:48];   // conv_b_q <= 2048 when legal
  assign mm_conv_o      = is_conv;
  assign mm_conv_kh_o   = inst[51:48];
  assign mm_conv_kw_o   = inst[55:52];
  assign mm_conv_oh0_o  = inst[75:64];
  assign mm_conv_ohcnt_o = inst[87:76];
  assign conv_in_w_o    = in_w_q;
  assign conv_in_h_o    = in_h_q;
  assign conv_out_w_o   = out_w_q;
  assign conv_sh_o      = sh_q;
  assign conv_sw_o      = sw_q;
  assign conv_pad_t_o   = pad_t_q;
  assign conv_pad_l_o   = pad_l_q;
  assign conv_pad_val_o = pad_val_q;

  assign act_valid_o   = go && (op == OP_ACT);
  assign act_acc_row_o = inst[26:16];
  assign act_ub_row_o  = inst[47:32];
  assign act_rows_o    = inst[63:48];
  assign act_qset_o    = inst[65:64];
  assign act_pool_o    = inst[14:13];                 // only 0/1/2 can be legal
  assign act_pool_w_o  = pool_w_q;

  assign cfg_we_o     = go && (op == OP_SETCFG) && !is_setcfg4;
  assign cfg_target_o = inst[31:16];
  assign cfg_value_o  = inst[95:32];

  assign dma_req_valid_o = EnableDma && inst_v && legal && is_dma &&
                           (dma_offered_q || (!fault && dma_capacity_i && dma_issue_ok));
  always_comb begin
    dma_src_addr_o = '0; dma_dst_addr_o = '0;
    dma_length_o = '0; dma_reps_o = '0;
    dma_src_stride_o = '0; dma_dst_stride_o = '0;
    dma_kind_o = 0;
    case (op)
      OP_RHM: begin
        dma_src_addr_o = {15'd0, inst[76:32], 4'd0};
        dma_dst_addr_o = 64'hA010_0000 + {44'd0, inst[31:16], 4'd0};
        dma_length_o = {8'd0, inst[88:77], 4'd0};
        dma_reps_o = {20'd0, inst[100:89]};
        dma_src_stride_o = {36'd0, inst[124:101], 4'd0};
        dma_dst_stride_o = {48'd0, inst[88:77], 4'd0};
      end
      OP_RW: begin
        dma_kind_o = 1;
        dma_src_addr_o = {15'd0, inst[76:32], 4'd0};
        dma_dst_addr_o = 64'hA030_0000;
        dma_length_o = {inst[31:16], 8'd0};
        dma_reps_o = 1;
      end
      OP_WHM: begin
        dma_kind_o = 2;
        dma_src_addr_o = 64'hA010_0000 + {44'd0, inst[31:16], 4'd0};
        dma_dst_addr_o = {15'd0, inst[76:32], 4'd0};
        dma_length_o = {8'd0, inst[88:77], 4'd0};
        dma_reps_o = {20'd0, inst[100:89]};
        dma_src_stride_o = {48'd0, inst[88:77], 4'd0};
        dma_dst_stride_o = {36'd0, inst[124:101], 4'd0};
      end
      default: ;
    endcase
  end

  // geometry registers (kind 4): one writer; cleared on reset AND on every RUN rising edge so a
  // program never inherits the previous program's geometry (L14); err_clr does not clear them and
  // the requant parameter sets (kinds 1-3, in the activation unit) are not touched
  wire geom_we    = go && is_setcfg4;
  wire run_rising = run_i && !run_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      in_w_q <= '0; in_h_q <= '0; out_w_q <= '0; out_h_q <= '0; pool_w_q <= '0;
      sh_q <= '0; sw_q <= '0; pad_t_q <= '0; pad_l_q <= '0; pad_val_q <= '0;
    end else if (run_rising) begin
      in_w_q <= '0; in_h_q <= '0; out_w_q <= '0; out_h_q <= '0; pool_w_q <= '0;
      sh_q <= '0; sw_q <= '0; pad_t_q <= '0; pad_l_q <= '0; pad_val_q <= '0;
    end else if (geom_we) begin
      case (cfg4_i)
        2'd0: begin in_w_q  <= inst[47:32]; in_h_q <= inst[63:48]; end
        2'd1: begin out_w_q <= inst[47:32]; out_h_q <= inst[63:48]; end
        2'd2: begin sh_q <= inst[35:32]; sw_q <= inst[39:36]; pad_t_q <= inst[47:40];
                    pad_l_q <= inst[55:48]; pad_val_q <= inst[63:56]; end
        2'd3: pool_w_q <= inst[47:32];
      endcase
    end
  end

  assign sync_wait_o = inst_v && legal && (op == OP_SYNC) && !sync_ok && !fault;
  assign retire_o    = go;

  // ---------------------------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      inst_v <= 1'b0;
    end else if (if_valid_i && if_ready_o) begin
      inst_v <= 1'b1;
    end else if (go || bad || (fault && !dma_offered_q)) begin
      inst_v <= 1'b0;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) dma_offered_q <= 1'b0;
    else if (dma_req_valid_o && dma_req_ready_i) dma_offered_q <= 1'b0;
    else if (dma_req_valid_o) dma_offered_q <= 1'b1;
  end
  always_ff @(posedge clk_i) begin
    if (if_valid_i && if_ready_o) inst <= if_data_i;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) run_q <= 1'b0;
    else         run_q <= run_i;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)                          halted_q <= 1'b0;
    else if (run_i && !run_q)             halted_q <= 1'b0;   // restart
    else if (go && op == OP_HALT)         halted_q <= 1'b1;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      err_q <= 1'b0; cfg_err_op_o <= '0;
    end else if (err_clr_i) begin
      err_q <= 1'b0; cfg_err_op_o <= '0;
    end else if (bad && !err_q) begin
      err_q <= 1'b1; cfg_err_op_o <= op;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      irq_pulse_q <= 1'b0; irq_tag_o <= '0; debug_tag_o <= '0;
    end else begin
      irq_pulse_q <= go && (op == OP_IRQ);
      if (go && op == OP_IRQ) irq_tag_o   <= inst[63:32];
      if (go && op == OP_DBG) debug_tag_o <= inst[63:32];
    end
  end

  assign busy_o    = inst_v;
  assign halted_o  = halted_q;
  assign cfg_err_o = err_q;
  assign irq_o = irq_pulse_q && !fault;

  // pragma translate_off
  a_dma_stable: assert property (@(posedge clk_i) disable iff (!rst_ni)
    dma_req_valid_o && !dma_req_ready_i |=> dma_req_valid_o &&
      $stable({dma_src_addr_o, dma_dst_addr_o, dma_length_o, dma_reps_o,
               dma_src_stride_o, dma_dst_stride_o, dma_kind_o}));
  a_no_fault_dispatch: assert property (@(posedge clk_i) disable iff (!rst_ni)
    fault |-> !(mm_valid_o || act_valid_o || cfg_we_o || irq_o));
  // pragma translate_on

endmodule
