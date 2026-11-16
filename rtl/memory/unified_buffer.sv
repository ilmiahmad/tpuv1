// Unified Buffer: 1 MiB of 16 x INT8 rows (URAM).
//   RAM port B = native READ  (MXU feeder)                       - latency UbLatency (3)
//   RAM port A = native WRITE (activation write-back), priority  > AXI window (iDMA / HPM0)
// AXI requests wait (mem_gnt = 0) while a native write uses port A (M4 contract sec 4).
// Phase A ownership: while owner_tpu_i is set, AXI writes are dropped, answered with SLVERR
// and flag own_err_o (sticky). Native accesses are never blocked.

module unified_buffer #(
  parameter type axi_req_t  = logic,
  parameter type axi_resp_t = logic,
  parameter int unsigned IdWidth = 6,
  parameter bit AllowCoreDma = 1'b0,
  localparam int unsigned Rows    = tpu_pkg::UbRows,
  localparam int unsigned Width   = tpu_pkg::DataWidth,
  localparam int unsigned RamLat  = 2,                   // URAM + output register (XPM)
  localparam int unsigned Latency = tpu_pkg::UbLatency   // + 1 input register stage
) (
  input  logic                     clk_i,
  input  logic                     rst_ni,
  // AXI window
  input  axi_req_t                 axi_req_i,
  output axi_resp_t                axi_resp_o,
  // native read port
  input  logic                     rd_en_i,
  input  tpu_pkg::ub_addr_t        rd_addr_i,
  output logic [Width-1:0]         rd_data_o,        // valid UbLatency (3) cycles after rd_en_i
  // native write port (priority over AXI)
  input  logic                     wr_en_i,
  input  tpu_pkg::ub_addr_t        wr_addr_i,
  input  logic [Width-1:0]         wr_data_i,
  input  logic [Width/8-1:0]       wr_be_i,
  // control / status
  input  logic                     owner_tpu_i,
  input  logic                     err_clr_i,
  output logic                     own_err_o
);

  logic             m_req, m_we, m_gnt;
  logic [63:0]      m_addr;
  logic [Width-1:0] m_wdata;
  logic [Width/8-1:0] m_strb;
  logic             m_rvalid, m_err;
  logic             m_core;
  logic [Width-1:0] m_rdata;

  axi_to_detailed_mem #(
    .axi_req_t ( axi_req_t  ), .axi_resp_t ( axi_resp_t ),
    .AddrWidth ( 64 ), .DataWidth ( Width ), .IdWidth ( IdWidth ), .UserWidth ( 1 ),
    .NumBanks  ( 1  ), .BufDepth  ( Latency ), .HideStrb ( 1'b0 ), .OutFifoDepth ( 1 )
  ) i_a2m (
    .clk_i, .rst_ni, .busy_o(),
    .axi_req_i, .axi_resp_o,
    .mem_req_o(m_req), .mem_gnt_i(m_gnt), .mem_addr_o(m_addr), .mem_wdata_o(m_wdata),
    .mem_strb_o(m_strb), .mem_atop_o(), .mem_lock_o(), .mem_we_o(m_we), .mem_id_o(),
    .mem_user_o(m_core), .mem_cache_o(), .mem_prot_o(), .mem_qos_o(), .mem_region_o(),
    .mem_rvalid_i(m_rvalid), .mem_rdata_i(m_rdata), .mem_err_i(m_err), .mem_exokay_i(1'b0)
  );

  assign m_gnt = !wr_en_i;                               // native write has priority
  wire   a_axi   = m_req && m_gnt;                       // AXI beat granted this cycle
  wire   blocked = a_axi && m_we && owner_tpu_i && !(AllowCoreDma && m_core);
  wire   a_axi_ok = a_axi && !blocked;

  // Input register stage on both ports: the URAM cascade (16 deep) cannot also absorb the
  // xbar/axi_to_mem logic in one 10 ns cycle (M2 OOC: WNS +0.36 ns, 28 levels) - L20.
  logic                 a_en_q, b_en_q;
  logic [Width/8-1:0]   a_we_q;
  tpu_pkg::ub_addr_t    a_addr_q, b_addr_q;
  logic [Width-1:0]     a_wdata_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      a_en_q <= 1'b0; a_we_q <= '0; b_en_q <= 1'b0;
    end else begin
      a_en_q <= wr_en_i || a_axi_ok;
      a_we_q <= wr_en_i ? wr_be_i : ((a_axi_ok && m_we) ? m_strb : '0);
      b_en_q <= rd_en_i;
    end
  end
  always_ff @(posedge clk_i) begin
    a_addr_q  <= wr_en_i ? wr_addr_i : m_addr[19:4];
    a_wdata_q <= wr_en_i ? wr_data_i : m_wdata;
    b_addr_q  <= rd_addr_i;
  end

  ram_tdp #(
    .Depth(Rows), .Width(Width), .Latency(RamLat), .RamStyle("ultra")
  ) i_ram (
    .clk_i,
    .a_en_i    ( a_en_q    ), .a_we_i ( a_we_q ), .a_addr_i ( a_addr_q ), .a_wdata_i ( a_wdata_q ),
    .a_rdata_o ( m_rdata   ),
    .b_en_i    ( b_en_q    ), .b_we_i ( '0     ), .b_addr_i ( b_addr_q ), .b_wdata_i ( '0        ),
    .b_rdata_o ( rd_data_o )
  );

  // fixed-latency response pipeline (one response per granted AXI request)
  logic [Latency-1:0] vld_q, err_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      vld_q <= '0; err_q <= '0;
    end else begin
      vld_q <= {vld_q[Latency-2:0], a_axi};
      err_q <= {err_q[Latency-2:0], blocked};
    end
  end
  assign m_rvalid = vld_q[Latency-1];
  assign m_err    = err_q[Latency-1];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)        own_err_o <= 1'b0;
    else if (err_clr_i) own_err_o <= 1'b0;
    else if (blocked)   own_err_o <= 1'b1;
  end

endmodule
