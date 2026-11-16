// Stream window: an AXI4 address window in front of a BRAM FIFO (Weight FIFO, Instruction FIFO).
// The address inside the window is ignored, so a DMA burst to incrementing addresses streams in.
//   write beat, all 16 strobes set  -> push (backpressure: the beat waits while the FIFO is full)
//   write beat, partial strobes      -> dropped, SLVERR, sticky err_o
//   read beat, dbg_pop_i = 0         -> returns {level} without side effects (safe to peek)
//   read beat, dbg_pop_i = 1         -> pops the head word (debug loopback FIFO -> DDR);
//                                       reading an empty FIFO returns SLVERR (never stalls, L5)
// Native consumer (weight loader / decoder) pops through pop_*; it is disabled while
// dbg_pop_i = 1 so there is exactly one consumer at a time.

module stream_window #(
  parameter type axi_req_t  = logic,
  parameter type axi_resp_t = logic,
  parameter int unsigned IdWidth = 6,
  parameter int unsigned Depth   = 1024,
  localparam int unsigned Width  = tpu_pkg::DataWidth,
  localparam int unsigned LW     = $clog2(Depth + 8) + 1
) (
  input  logic             clk_i,
  input  logic             rst_ni,
  input  axi_req_t         axi_req_i,
  output axi_resp_t        axi_resp_o,
  // native consumer
  output logic             pop_valid_o,
  input  logic             pop_ready_i,
  output logic [Width-1:0] pop_data_o,
  // control / status
  input  logic             dbg_pop_i,
  input  logic             flush_i,
  input  logic             err_clr_i,
  output logic             err_o,
  output logic [LW-1:0]    level_o
);

  logic             m_req, m_we, m_gnt;
  logic [Width-1:0] m_wdata;
  logic [Width/8-1:0] m_strb;
  logic             m_rvalid, m_err;
  logic [Width-1:0] m_rdata;

  axi_to_detailed_mem #(
    .axi_req_t ( axi_req_t  ), .axi_resp_t ( axi_resp_t ),
    .AddrWidth ( 64 ), .DataWidth ( Width ), .IdWidth ( IdWidth ), .UserWidth ( 1 ),
    .NumBanks  ( 1  ), .BufDepth  ( 1 ), .HideStrb ( 1'b0 ), .OutFifoDepth ( 1 )
  ) i_a2m (
    .clk_i, .rst_ni, .busy_o(),
    .axi_req_i, .axi_resp_o,
    .mem_req_o(m_req), .mem_gnt_i(m_gnt), .mem_addr_o(), .mem_wdata_o(m_wdata),
    .mem_strb_o(m_strb), .mem_atop_o(), .mem_lock_o(), .mem_we_o(m_we), .mem_id_o(),
    .mem_user_o(), .mem_cache_o(), .mem_prot_o(), .mem_qos_o(), .mem_region_o(),
    .mem_rvalid_i(m_rvalid), .mem_rdata_i(m_rdata), .mem_err_i(m_err), .mem_exokay_i(1'b0)
  );

  logic             f_push_ready, f_pop_valid, f_pop_ready;
  logic [Width-1:0] f_pop_data;

  wire full_beat  = &m_strb;
  wire do_push    = m_req &&  m_we && full_beat;
  wire bad_write  = m_req &&  m_we && !full_beat;
  wire do_dbgpop  = m_req && !m_we && dbg_pop_i && f_pop_valid;
  wire bad_pop    = m_req && !m_we && dbg_pop_i && !f_pop_valid;
  wire peek       = m_req && !m_we && !dbg_pop_i;

  // grant: pushes wait for space; everything else completes immediately
  assign m_gnt = do_push ? f_push_ready : 1'b1;

  fifo_ram #(.Depth(Depth), .Width(Width)) i_fifo (
    .clk_i, .rst_ni, .flush_i,
    .push_valid_i ( do_push ),
    .push_ready_o ( f_push_ready ),
    .push_data_i  ( m_wdata ),
    .pop_valid_o  ( f_pop_valid ),
    .pop_ready_i  ( f_pop_ready ),
    .pop_data_o   ( f_pop_data ),
    .level_o
  );

  assign f_pop_ready = dbg_pop_i ? do_dbgpop : pop_ready_i;
  assign pop_valid_o = f_pop_valid && !dbg_pop_i;
  assign pop_data_o  = f_pop_data;

  // one response per granted request, one cycle later
  logic             vld_q, err_q;
  logic [Width-1:0] rdata_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      vld_q <= 1'b0; err_q <= 1'b0; rdata_q <= '0;
    end else begin
      vld_q   <= m_req && m_gnt;
      err_q   <= bad_write || bad_pop;
      rdata_q <= do_dbgpop ? f_pop_data : (peek ? Width'(level_o) : '0);
    end
  end
  assign m_rvalid = vld_q;
  assign m_err    = err_q;
  assign m_rdata  = rdata_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)                     err_o <= 1'b0;
    else if (err_clr_i)              err_o <= 1'b0;
    else if (bad_write || bad_pop)   err_o <= 1'b1;
  end

endmodule
