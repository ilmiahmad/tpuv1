// TPU-KV260 data fabric (M2):
//
//   slave ports : [0] iDMA backend   [1] HPM0 data window (CPU debug / instruction pushes)
//   master ports: [0] HP0 -> PS DDR  [1] Unified Buffer  [2] Accumulators
//                 [3] Weight FIFO window  [4] Instruction FIFO window
//   HPM0 -> DDR is disabled (connectivity); unmapped addresses get DECERR (L5).
//
// All protocol logic is PULP (axi_xbar, axi_to_detailed_mem); custom code here is only the
// memories' fixed-latency glue and FIFOs.

module tpu_fabric
  import tpu_pkg::*;
#(
  parameter bit AllowCoreDma = 1'b0
)
(
  input  logic          clk_i,
  input  logic          rst_ni,
  // masters
  input  fs_req_t       dma_req_i,
  output fs_resp_t      dma_rsp_o,
  input  fs_req_t       hpm_req_i,
  output fs_resp_t      hpm_rsp_o,
  // to PS DDR
  output fm_req_t       hp0_req_o,
  input  fm_resp_t      hp0_rsp_i,
  // Unified Buffer native ports: read (RAM port B, latency 3), write (RAM port A, > AXI)
  input  logic                 ub_rd_en_i,
  input  ub_addr_t             ub_rd_addr_i,
  output data_t                ub_rd_data_o,
  input  logic                 ub_wr_en_i,
  input  ub_addr_t             ub_wr_addr_i,
  input  data_t                ub_wr_data_i,
  input  logic [StrbWidth-1:0] ub_wr_be_i,
  // Accumulator native ports: read (port B, latency 1), write (port A, > AXI), whole rows
  input  logic                 acc_rd_en_i,
  input  acc_addr_t            acc_rd_addr_i,
  output acc_row_t             acc_rd_data_o,
  input  logic                 acc_wr_en_i,
  input  acc_addr_t            acc_wr_addr_i,
  input  acc_row_t             acc_wr_data_i,
  input  logic [AccBanks-1:0]  acc_wr_be_i,
  // FIFO consumers
  output logic                wf_valid_o,
  input  logic                wf_ready_i,
  output data_t               wf_data_o,
  output logic                if_valid_o,
  input  logic                if_ready_i,
  output data_t               if_data_o,
  // control
  input  logic                ub_owner_tpu_i,
  input  logic                acc_owner_tpu_i,
  input  logic                wf_dbg_pop_i,
  input  logic                if_dbg_pop_i,
  input  logic                wf_flush_i,
  input  logic                if_flush_i,
  input  logic                err_clr_i,
  // status
  output logic [3:0]          err_o,          // {ififo, wfifo, acc_own, ub_own}
  output logic [15:0]         wf_level_o,
  output logic [15:0]         if_level_o
);

  localparam axi_pkg::xbar_cfg_t XbarCfg = '{
    NoSlvPorts:         32'd2,
    NoMstPorts:         32'(XNUM),
    MaxMstTrans:        32'd8,
    MaxSlvTrans:        32'd8,
    FallThrough:        1'b0,
    LatencyMode:        axi_pkg::CUT_ALL_AX,
    PipelineStages:     32'd0,
    AxiIdWidthSlvPorts: 32'(SlvIdWidth),
    AxiIdUsedSlvPorts:  32'd2,
    UniqueIds:          1'b0,
    AxiAddrWidth:       32'(AddrWidth),
    AxiDataWidth:       32'(DataWidth),
    NoAddrRules:        32'd6
  };

  // [slave port][master port]: HPM0 (1) must not reach DDR (0)
  localparam bit [1:0][XNUM-1:0] Connectivity = '{
    1: 5'b11110,
    0: 5'b11111
  };

  localparam axi_pkg::xbar_rule_64_t [5:0] AddrMap = '{
    '{idx: 32'(XIFIFO), start_addr: IFIFO_BASE,  end_addr: IFIFO_BASE + IFIFO_SIZE},
    '{idx: 32'(XWFIFO), start_addr: WFIFO_BASE,  end_addr: WFIFO_BASE + WFIFO_SIZE},
    '{idx: 32'(XACC),   start_addr: ACC_BASE,    end_addr: ACC_BASE + ACC_SIZE},
    '{idx: 32'(XUB),    start_addr: UB_BASE,     end_addr: UB_BASE + UB_SIZE},
    '{idx: 32'(XHP0),   start_addr: DDR_HI_BASE, end_addr: DDR_HI_END},
    '{idx: 32'(XHP0),   start_addr: DDR_LO_BASE, end_addr: DDR_LO_END}
  };

  fs_req_t  [1:0]      slv_req;
  fs_resp_t [1:0]      slv_rsp;
  fm_req_t  [XNUM-1:0] mst_req;
  fm_resp_t [XNUM-1:0] mst_rsp;

  assign slv_req[0] = dma_req_i;
  assign slv_req[1] = hpm_req_i;
  assign dma_rsp_o  = slv_rsp[0];
  assign hpm_rsp_o  = slv_rsp[1];

  axi_xbar #(
    .Cfg           ( XbarCfg         ),
    .ATOPs         ( 1'b0            ),
    .Connectivity  ( Connectivity    ),
    .slv_aw_chan_t ( fs_aw_chan_t    ), .mst_aw_chan_t ( fm_aw_chan_t ),
    .w_chan_t      ( fs_w_chan_t     ),
    .slv_b_chan_t  ( fs_b_chan_t     ), .mst_b_chan_t  ( fm_b_chan_t  ),
    .slv_ar_chan_t ( fs_ar_chan_t    ), .mst_ar_chan_t ( fm_ar_chan_t ),
    .slv_r_chan_t  ( fs_r_chan_t     ), .mst_r_chan_t  ( fm_r_chan_t  ),
    .slv_req_t     ( fs_req_t        ), .slv_resp_t    ( fs_resp_t    ),
    .mst_req_t     ( fm_req_t        ), .mst_resp_t    ( fm_resp_t    ),
    .rule_t        ( axi_pkg::xbar_rule_64_t )
  ) i_xbar (
    .clk_i, .rst_ni,
    .slv_ports_req_i       ( slv_req ),
    .slv_ports_resp_o      ( slv_rsp ),
    .mst_ports_req_o       ( mst_req ),
    .mst_ports_resp_i      ( mst_rsp ),
    .addr_map_i            ( AddrMap ),
    .en_default_mst_port_i ( '0      ),
    .default_mst_port_i    ( '0      )
  );

  // Register boundary (all 5 channels) between the xbar and every slave: axi_to_detailed_mem
  // turns AXI requests combinationally into RAM controls, which together with the xbar demux
  // was 24-28 logic levels (M2 OOC, L20). PULP axi_cut = spill registers, no protocol change.
  fm_req_t  [XNUM-1:0] cut_req;
  fm_resp_t [XNUM-1:0] cut_rsp;
  for (genvar k = 0; k < XNUM; k++) begin : gen_cut
    axi_cut #(
      .aw_chan_t ( fm_aw_chan_t ), .w_chan_t ( fm_w_chan_t ), .b_chan_t ( fm_b_chan_t ),
      .ar_chan_t ( fm_ar_chan_t ), .r_chan_t ( fm_r_chan_t ),
      .axi_req_t ( fm_req_t     ), .axi_resp_t ( fm_resp_t )
    ) i_cut (
      .clk_i, .rst_ni,
      .slv_req_i  ( mst_req[k] ), .slv_resp_o ( mst_rsp[k] ),
      .mst_req_o  ( cut_req[k] ), .mst_resp_i ( cut_rsp[k] )
    );
  end

  assign hp0_req_o     = cut_req[XHP0];
  assign cut_rsp[XHP0] = hp0_rsp_i;

  logic ub_err, acc_err, wf_err, if_err;
  logic [$clog2(WfifoWords + 8):0] wf_level;
  logic [$clog2(IfifoWords + 8):0] if_level;

  unified_buffer #(
    .axi_req_t(fm_req_t), .axi_resp_t(fm_resp_t), .IdWidth(MstIdWidth), .AllowCoreDma(AllowCoreDma)
  ) i_ub (
    .clk_i, .rst_ni, .axi_req_i(cut_req[XUB]), .axi_resp_o(cut_rsp[XUB]),
    .rd_en_i(ub_rd_en_i), .rd_addr_i(ub_rd_addr_i), .rd_data_o(ub_rd_data_o),
    .wr_en_i(ub_wr_en_i), .wr_addr_i(ub_wr_addr_i), .wr_data_i(ub_wr_data_i), .wr_be_i(ub_wr_be_i),
    .owner_tpu_i(ub_owner_tpu_i), .err_clr_i, .own_err_o(ub_err)
  );

  accumulators #(
    .axi_req_t(fm_req_t), .axi_resp_t(fm_resp_t), .IdWidth(MstIdWidth)
  ) i_acc (
    .clk_i, .rst_ni, .axi_req_i(cut_req[XACC]), .axi_resp_o(cut_rsp[XACC]),
    .rd_en_i(acc_rd_en_i), .rd_addr_i(acc_rd_addr_i), .rd_data_o(acc_rd_data_o),
    .wr_en_i(acc_wr_en_i), .wr_addr_i(acc_wr_addr_i), .wr_data_i(acc_wr_data_i), .wr_be_i(acc_wr_be_i),
    .owner_tpu_i(acc_owner_tpu_i), .err_clr_i, .own_err_o(acc_err)
  );

  stream_window #(
    .axi_req_t(fm_req_t), .axi_resp_t(fm_resp_t), .IdWidth(MstIdWidth), .Depth(WfifoWords)
  ) i_wfifo (
    .clk_i, .rst_ni, .axi_req_i(cut_req[XWFIFO]), .axi_resp_o(cut_rsp[XWFIFO]),
    .pop_valid_o(wf_valid_o), .pop_ready_i(wf_ready_i), .pop_data_o(wf_data_o),
    .dbg_pop_i(wf_dbg_pop_i), .flush_i(wf_flush_i), .err_clr_i, .err_o(wf_err), .level_o(wf_level)
  );

  stream_window #(
    .axi_req_t(fm_req_t), .axi_resp_t(fm_resp_t), .IdWidth(MstIdWidth), .Depth(IfifoWords)
  ) i_ififo (
    .clk_i, .rst_ni, .axi_req_i(cut_req[XIFIFO]), .axi_resp_o(cut_rsp[XIFIFO]),
    .pop_valid_o(if_valid_o), .pop_ready_i(if_ready_i), .pop_data_o(if_data_o),
    .dbg_pop_i(if_dbg_pop_i), .flush_i(if_flush_i), .err_clr_i, .err_o(if_err), .level_o(if_level)
  );

  assign err_o      = {if_err, wf_err, acc_err, ub_err};
  assign wf_level_o = 16'(wf_level);
  assign if_level_o = 16'(if_level);

endmodule
