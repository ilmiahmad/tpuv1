// First-word-fall-through FIFO with BRAM storage (L19: no flip-flop arrays for deep FIFOs).
// RAM (latency 1) feeds a 4-entry register output stage so one word per cycle is sustained.
// level_o counts every stored word (RAM + in flight + output stage).

module fifo_ram #(
  parameter int unsigned Depth = 1024,               // RAM words (power of two)
  parameter int unsigned Width = 128,
  localparam int unsigned AW   = $clog2(Depth),
  localparam int unsigned LW   = $clog2(Depth + 8) + 1
) (
  input  logic             clk_i,
  input  logic             rst_ni,
  input  logic             flush_i,
  // push
  input  logic             push_valid_i,
  output logic             push_ready_o,
  input  logic [Width-1:0] push_data_i,
  // pop (FWFT)
  output logic             pop_valid_o,
  input  logic             pop_ready_i,
  output logic [Width-1:0] pop_data_o,
  output logic [LW-1:0]    level_o
);

  localparam int unsigned OutDepth = 4;

  logic [AW-1:0] wptr_q, rptr_q;
  logic [AW:0]   ram_cnt_q;
  logic          inflight_q;
  logic [Width-1:0] ram_rdata;

  logic out_full, out_empty;
  logic [$clog2(OutDepth)-1:0] out_usage_raw;
  logic [$clog2(OutDepth):0]   out_usage;
  assign out_usage = out_full ? ($clog2(OutDepth)+1)'(OutDepth) : {1'b0, out_usage_raw};

  wire push = push_valid_i && push_ready_o;
  wire rd   = (ram_cnt_q != 0) && (32'(out_usage) + 32'(inflight_q) < OutDepth);
  wire pop  = pop_valid_o && pop_ready_i;

  assign push_ready_o = (ram_cnt_q != (AW+1)'(Depth));

  ram_tdp #(
    .Depth(Depth), .Width(Width), .Latency(1), .RamStyle("block")
  ) i_ram (
    .clk_i,
    .a_en_i(push), .a_we_i({(Width/8){push}}), .a_addr_i(wptr_q), .a_wdata_i(push_data_i), .a_rdata_o(),
    .b_en_i(rd),   .b_we_i('0),                 .b_addr_i(rptr_q), .b_wdata_i('0),          .b_rdata_o(ram_rdata)
  );

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wptr_q <= '0; rptr_q <= '0; ram_cnt_q <= '0; inflight_q <= 1'b0;
    end else if (flush_i) begin
      wptr_q <= '0; rptr_q <= '0; ram_cnt_q <= '0; inflight_q <= 1'b0;
    end else begin
      if (push) wptr_q <= wptr_q + 1'b1;
      if (rd)   rptr_q <= rptr_q + 1'b1;
      ram_cnt_q  <= ram_cnt_q + (AW+1)'(push) - (AW+1)'(rd);
      inflight_q <= rd;
    end
  end

  fifo_v3 #(
    .FALL_THROUGH ( 1'b0     ),
    .DATA_WIDTH   ( Width    ),
    .DEPTH        ( OutDepth )
  ) i_out (
    .clk_i,
    .rst_ni,
    .flush_i    ( flush_i       ),
    .testmode_i ( 1'b0          ),
    .full_o     ( out_full      ),
    .empty_o    ( out_empty     ),
    .usage_o    ( out_usage_raw ),
    .data_i     ( ram_rdata     ),
    .push_i     ( inflight_q && !flush_i ),
    .data_o     ( pop_data_o    ),
    .pop_i      ( pop           )
  );

  assign pop_valid_o = !out_empty;
  assign level_o     = LW'(ram_cnt_q) + LW'(inflight_q) + LW'(out_usage);

  // pragma translate_off
  a_no_out_overflow: assert property (@(posedge clk_i) disable iff (!rst_ni)
      inflight_q |-> !out_full) else $error("fifo_ram: output stage overflow");
  // pragma translate_on

endmodule
