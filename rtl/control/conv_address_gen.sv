// Convolution address generator (docs/m6_cnn_contract.md sec 4, sec 6). One (pad, row) per output pixel
// of one CONVOLVE (one kernel tap, one input channel-tile plane, a window of output rows):
//
//   for oh in oh0 .. oh0+oh_cnt-1, for ow in 0 .. out_w-1:
//     y = oh*SH + kh - PAD_T,  x = ow*SW + kw - PAD_L
//     pad = !(0 <= y < IN_H && 0 <= x < IN_W);   row = ub_row + y*IN_W + x   (valid when !pad)
//
// Per-row address path = adders only (+SW per column, +SH*IN_W per output row; L20). The few
// multiplies (oh0*SH, SH*IN_W, y0*IN_W) run once per command in a registered, DSP-hinted startup
// pipeline (5 cycles from the accepted command to the first row). All geometry is captured at
// command accept, so later decoder register writes cannot disturb a running command.
//
// Row addresses are computed modulo 2^16: the contract's legality rule (ub_row + IN_H*IN_W <=
// 65536) bounds every valid address below 2^16, so wrap-around in the unused (padded) rows is
// harmless. Output is a registered ready/valid stream.

module conv_address_gen (
  input  logic        clk_i,
  input  logic        rst_ni,
  // command (captured when cmd_valid_i && cmd_ready_o)
  input  logic        cmd_valid_i,
  output logic        cmd_ready_o,
  input  logic [15:0] cmd_ub_row_i,
  input  logic [3:0]  cmd_kh_i,
  input  logic [3:0]  cmd_kw_i,
  input  logic [11:0] cmd_oh0_i,
  input  logic [11:0] cmd_ohcnt_i,
  // static geometry (sampled at command accept)
  input  logic [15:0] in_w_i,
  input  logic [15:0] in_h_i,
  input  logic [15:0] out_w_i,
  input  logic [3:0]  sh_i,
  input  logic [3:0]  sw_i,
  input  logic [7:0]  pad_t_i,
  input  logic [7:0]  pad_l_i,
  // row stream
  output logic        row_valid_o,
  input  logic        row_ready_i,
  output logic        row_pad_o,
  output logic [15:0] row_addr_o,
  output logic        idle_o
);

  // captured command / geometry
  logic [15:0] ub_row_q, in_w_q, in_h_q, outw_m1_q;
  logic [3:0]  kh_q, kw_q, sh_q, sw_q;
  logic [11:0] oh0_q, cmd_ohcnt_q;
  logic [7:0]  pad_t_q, pad_l_q;

  // startup pipeline
  logic [2:0]  st_q;                       // 0 idle/run, 1..4 startup
  // Only the low 16 bits of the two wide products are used (addresses are modulo 2^16).
  /* verilator lint_off UNUSEDSIGNAL */
  (* use_dsp = "yes" *) logic [15:0] oh_sh_q;           // oh0 * SH
  (* use_dsp = "yes" *) logic [19:0] strow_full_q;      // SH * IN_W
  (* use_dsp = "yes" *) logic signed [35:0] yprod_q;    // y0 * IN_W
  /* verilator lint_on UNUSEDSIGNAL */
  logic signed [17:0] ktap_q;              // kh - PAD_T

  // running state
  logic               run_q, yok_q;
  logic signed [17:0] y_q, x_q, x0_q;
  logic [15:0]        rowptr_q, strow_q, ow_q;
  logic [11:0]        ohleft_q;

  // output register
  logic        out_v_q, out_pad_q;
  logic [15:0] out_addr_q;

  assign cmd_ready_o = (st_q == 3'd0) && !run_q && !out_v_q;
  assign idle_o      = cmd_ready_o;

  // ---------------------------------------------------------------------------------------------
  // Per-row candidate (combinational from registered state; adders and comparators only)
  // ---------------------------------------------------------------------------------------------
  wire signed [17:0] in_w_s = $signed({2'b00, in_w_q});
  wire signed [17:0] in_h_s = $signed({2'b00, in_h_q});
  wire x_ok        = (x_q >= 18'sd0) && (x_q < in_w_s);
  wire pad_c       = !(yok_q && x_ok);
  wire [15:0] addr_c = rowptr_q + x_q[15:0];
  wire last_col    = (ow_q == outw_m1_q);
  wire signed [17:0] y_next = y_q + $signed({14'd0, sh_q});
  wire y_next_ok   = (y_next >= 18'sd0) && (y_next < in_h_s);

  wire take    = !out_v_q || row_ready_i;
  wire advance = run_q && take;

  // geometry / command capture and startup pipeline
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      st_q <= 3'd0;
      ub_row_q <= '0; in_w_q <= '0; in_h_q <= '0; outw_m1_q <= '0;
      kh_q <= '0; kw_q <= '0; sh_q <= '0; sw_q <= '0; oh0_q <= '0; pad_t_q <= '0; pad_l_q <= '0;
      cmd_ohcnt_q <= '0;
    end else if (cmd_valid_i && cmd_ready_o) begin
      st_q <= 3'd1; cmd_ohcnt_q <= cmd_ohcnt_i;
      ub_row_q <= cmd_ub_row_i; in_w_q <= in_w_i; in_h_q <= in_h_i; outw_m1_q <= out_w_i - 16'd1;
      kh_q <= cmd_kh_i; kw_q <= cmd_kw_i; sh_q <= sh_i; sw_q <= sw_i; oh0_q <= cmd_oh0_i;
      pad_t_q <= pad_t_i; pad_l_q <= pad_l_i;
    end else if (st_q != 3'd0) begin
      st_q <= (st_q == 3'd4) ? 3'd0 : st_q + 3'd1;
    end
  end

  always_ff @(posedge clk_i) begin
    if (st_q == 3'd1) begin
      oh_sh_q      <= 16'(oh0_q) * 16'(sh_q);
      strow_full_q <= 20'(sh_q) * 20'(in_w_q);
      ktap_q       <= $signed({14'd0, kh_q}) - $signed({10'd0, pad_t_q});
      x0_q         <= $signed({14'd0, kw_q}) - $signed({10'd0, pad_l_q});
    end
    if (st_q == 3'd2)          y_q <= $signed({2'b00, oh_sh_q}) + ktap_q;
    else if (advance && last_col) y_q <= y_next;       // next output row
    if (st_q == 3'd3) yprod_q <= y_q * in_w_s;
  end

  // run state (single always_ff writer per register)
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      run_q <= 1'b0; yok_q <= 1'b0; x_q <= '0; ow_q <= '0; ohleft_q <= '0;
      rowptr_q <= '0; strow_q <= '0;
    end else if (st_q == 3'd3) begin
      yok_q <= (y_q >= 18'sd0) && (y_q < in_h_s);
      x_q   <= x0_q;
      ow_q  <= '0;
    end else if (st_q == 3'd4) begin
      rowptr_q <= ub_row_q + yprod_q[15:0];
      strow_q  <= strow_full_q[15:0];
      ohleft_q <= cmd_ohcnt_q;
      run_q    <= 1'b1;
    end else if (advance) begin
      if (last_col) begin
        ow_q     <= '0;
        x_q      <= x0_q;
        yok_q    <= y_next_ok;
        rowptr_q <= rowptr_q + strow_q;
        ohleft_q <= ohleft_q - 12'd1;
        if (ohleft_q == 12'd1) run_q <= 1'b0;
      end else begin
        ow_q <= ow_q + 16'd1;
        x_q  <= x_q + $signed({14'd0, sw_q});
      end
    end
  end

  // output register
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      out_v_q <= 1'b0; out_pad_q <= 1'b0; out_addr_q <= '0;
    end else if (advance) begin
      out_v_q <= 1'b1; out_pad_q <= pad_c; out_addr_q <= pad_c ? 16'd0 : addr_c;
    end else if (row_ready_i) begin
      out_v_q <= 1'b0;
    end
  end

  assign row_valid_o = out_v_q;
  assign row_pad_o   = out_pad_q;
  assign row_addr_o  = out_addr_q;

  // pragma translate_off
  a_stable: assert property (@(posedge clk_i) disable iff (!rst_ni)
    row_valid_o && !row_ready_i |=> row_valid_o && $stable({row_pad_o, row_addr_o}))
    else $error("conv_address_gen: output changed while stalled");
  // pragma translate_on

endmodule
