// Accumulators: 2048 rows x 16 x INT32 = 128 KiB, as 4 BRAM banks of 2048 x 128 b.
//   RAM port B = native READ  (whole 512-bit row)                - latency 1
//   RAM port A = native WRITE (whole row, per-bank enable), priority > AXI window
// AXI window (debug / raw-output path): byte offset -> row = off[16:6], bank = off[5:4].
// AXI requests wait (mem_gnt = 0) while a native write uses port A (M4 contract sec 4).
// Ownership as in unified_buffer (AXI writes only).

module accumulators #(
  parameter type axi_req_t  = logic,
  parameter type axi_resp_t = logic,
  parameter int unsigned IdWidth = 6,
  localparam int unsigned Rows   = tpu_pkg::AccRows,
  localparam int unsigned Banks  = tpu_pkg::AccBanks,
  localparam int unsigned Width  = tpu_pkg::DataWidth
) (
  input  logic                      clk_i,
  input  logic                      rst_ni,
  input  axi_req_t                  axi_req_i,
  output axi_resp_t                 axi_resp_o,
  // native read port: whole row
  input  logic                      rd_en_i,
  input  tpu_pkg::acc_addr_t        rd_addr_i,
  output tpu_pkg::acc_row_t         rd_data_o,       // valid 1 cycle after rd_en_i
  // native write port: whole row, per-bank enable (priority over AXI)
  input  logic                      wr_en_i,
  input  tpu_pkg::acc_addr_t        wr_addr_i,
  input  tpu_pkg::acc_row_t         wr_data_i,
  input  logic [Banks-1:0]          wr_be_i,         // per 128-bit bank (4 x INT32 each)
  input  logic                      owner_tpu_i,
  input  logic                      err_clr_i,
  output logic                      own_err_o
);

  logic             m_req, m_we, m_gnt;
  logic [63:0]      m_addr;
  logic [Width-1:0] m_wdata, m_rdata;
  logic [Width/8-1:0] m_strb;
  logic             m_rvalid, m_err;

  axi_to_detailed_mem #(
    .axi_req_t ( axi_req_t  ), .axi_resp_t ( axi_resp_t ),
    .AddrWidth ( 64 ), .DataWidth ( Width ), .IdWidth ( IdWidth ), .UserWidth ( 1 ),
    .NumBanks  ( 1  ), .BufDepth  ( 1 ), .HideStrb ( 1'b0 ), .OutFifoDepth ( 1 )
  ) i_a2m (
    .clk_i, .rst_ni, .busy_o(),
    .axi_req_i, .axi_resp_o,
    .mem_req_o(m_req), .mem_gnt_i(m_gnt), .mem_addr_o(m_addr), .mem_wdata_o(m_wdata),
    .mem_strb_o(m_strb), .mem_atop_o(), .mem_lock_o(), .mem_we_o(m_we), .mem_id_o(),
    .mem_user_o(), .mem_cache_o(), .mem_prot_o(), .mem_qos_o(), .mem_region_o(),
    .mem_rvalid_i(m_rvalid), .mem_rdata_i(m_rdata), .mem_err_i(m_err), .mem_exokay_i(1'b0)
  );

  assign m_gnt = !wr_en_i;                               // native write has priority
  wire                 a_axi    = m_req && m_gnt;
  wire                 blocked  = a_axi && m_we && owner_tpu_i;
  wire                 a_axi_ok = a_axi && !blocked;
  wire [1:0]           a_bank   = m_addr[5:4];
  tpu_pkg::acc_addr_t  a_row;
  assign a_row = m_addr[16:6];

  logic [Banks-1:0][Width-1:0] a_rdata;

  for (genvar k = 0; k < Banks; k++) begin : gen_bank
    logic             pa_en;
    logic [Width/8-1:0] pa_we;
    tpu_pkg::acc_addr_t pa_addr;
    logic [Width-1:0] pa_wdata;
    always_comb begin
      if (wr_en_i) begin
        pa_en    = wr_be_i[k];
        pa_we    = {(Width/8){wr_be_i[k]}};
        pa_addr  = wr_addr_i;
        pa_wdata = wr_data_i[k*Width +: Width];
      end else begin
        pa_en    = a_axi_ok && (a_bank == 2'(k));
        pa_we    = m_we ? m_strb : '0;
        pa_addr  = a_row;
        pa_wdata = m_wdata;
      end
    end
    ram_tdp #(
      .Depth(Rows), .Width(Width), .Latency(1), .RamStyle("block")
    ) i_ram (
      .clk_i,
      .a_en_i    ( pa_en    ),
      .a_we_i    ( pa_we    ),
      .a_addr_i  ( pa_addr  ),
      .a_wdata_i ( pa_wdata ),
      .a_rdata_o ( a_rdata[k] ),
      .b_en_i    ( rd_en_i ),
      .b_we_i    ( '0 ),
      .b_addr_i  ( rd_addr_i ),
      .b_wdata_i ( '0 ),
      .b_rdata_o ( rd_data_o[k*Width +: Width] )
    );
  end

  logic [1:0] bank_q;
  logic       vld_q, err_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      vld_q <= 1'b0; err_q <= 1'b0; bank_q <= '0;
    end else begin
      vld_q <= a_axi; err_q <= blocked; bank_q <= a_bank;
    end
  end
  assign m_rvalid = vld_q;
  assign m_err    = err_q;
  assign m_rdata  = a_rdata[bank_q];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)        own_err_o <= 1'b0;
    else if (err_clr_i) own_err_o <= 1'b0;
    else if (blocked)   own_err_o <= 1'b1;
  end

endmodule
