// TPU-KV260 - PULP iDMA subsystem (M0/M1, optional M5 Phase B requester)
//
//   AXI4-Lite cfg -> axi_lite_to_apb -> idma_reg64_2d -> idma_nd_midend (2D)
//                                            ^                   |
//                          idma_transfer_id_gen <-- nd_rsp       v
//                                                     idma_backend_rw_axi -> axi_rw_join -> AXI4 master
//
// EnableCore: register frontend + core descriptors -> idma_request_arbiter -> ND midend.
// Source tags and buffered ND responses route completions; only host completions retire IDs.
//
// iDMA v0.7.4 ships reg64_2d pre-generated with the apb4-flat CPU IF, so the AXI4-Lite config
// port is bridged with PULP axi_lite_to_apb instead of regenerating with axi4-lite-flat.
// Internal addresses are 64-bit (reg64 frontend); the top level truncates to the 49-bit HP0
// address and asserts the upper bits are zero.
// No custom DMA logic: this file only instantiates and wires PULP IP (ADR-0001).

`include "axi/typedef.svh"
`include "apb/typedef.svh"
`include "idma/typedef.svh"

module idma_subsystem #(
  parameter int unsigned DataWidth     = 32'd128,  // HP0 / HPM0 width (L1)
  parameter int unsigned AddrWidth     = 32'd64,   // reg64 frontend
  parameter int unsigned AxiIdWidth    = 32'd6,    // HP0 slave ID width
  parameter int unsigned UserWidth     = 32'd1,
  parameter int unsigned TFLenWidth    = 32'd24,
  parameter int unsigned NumAxInFlight = 32'd4,
  parameter int unsigned BufferDepth   = 32'd3,
  parameter int unsigned CfgAddrWidth  = 32'd32,
  parameter int unsigned CfgDataWidth  = 32'd32,
  parameter int unsigned IdCounterWidth = 32'd32,  // reduced only in the M1 ID-wrap test
  parameter bit EnableCore = 1'b0,  // Phase B transport; keep the legacy host path by default
  parameter int unsigned CoreOutstanding = 8,
  // AXI4 master types (to the data xbar / HP0)
  parameter type axi_req_t      = logic,
  parameter type axi_rsp_t      = logic,
  parameter type axi_ar_chan_t  = logic,
  parameter type axi_aw_chan_t  = logic,
  // AXI4-Lite cfg slave types
  parameter type axil_req_t     = logic,
  parameter type axil_rsp_t     = logic
) (
  input  logic      clk_i,
  input  logic      rst_ni,
  // configuration (from HPM0 via axi_to_axi_lite / axi_lite_xbar)
  input  axil_req_t cfg_req_i,
  output axil_rsp_t cfg_rsp_o,
  // data master
  output axi_req_t  axi_req_o,
  input  axi_rsp_t  axi_rsp_i,
  // status for irq_glue / tpu_regs
  output logic [31:0] done_id_o,
  output logic        busy_o,
  // sticky error flags (M1): [0] read resp != OKAY, [1] write resp != OKAY,
  // [2] midend/backend reported error (e.g. reps == 0). Cleared by err_clr_i.
  output logic [2:0]  err_o,
  input  logic        err_clr_i,
  // Phase B core descriptor stream. Byte units; fields are held until handshake.
  input  logic core_req_valid_i,
  output logic core_req_ready_o,
  input  logic [AddrWidth-1:0] core_src_addr_i,
  input  logic [AddrWidth-1:0] core_dst_addr_i,
  input  logic [TFLenWidth-1:0] core_length_i,
  input  logic [31:0] core_reps_i,
  input  logic [AddrWidth-1:0] core_src_stride_i,
  input  logic [AddrWidth-1:0] core_dst_stride_i,
  output logic core_rsp_valid_o,
  input  logic core_rsp_ready_i,
  // iDMA response error only; AXI response errors still use the passive err_o monitor.
  output logic core_rsp_error_o
);

  // ------------------------------------------------------------------------------------------
  // Types
  // ------------------------------------------------------------------------------------------
  localparam int unsigned CfgStrbWidth   = CfgDataWidth / 8;
  localparam int unsigned NumDim         = 32'd2;

  typedef logic [AddrWidth-1:0]    addr_t;
  typedef logic [AxiIdWidth-1:0]   id_t;
  typedef logic [TFLenWidth-1:0]   tf_len_t;
  typedef logic [AddrWidth-1:0]    reps_t;      // reg64: 64-bit reps/strides (as upstream tb)
  typedef logic [AddrWidth-1:0]    strides_t;

  typedef logic [CfgAddrWidth-1:0] cfg_addr_t;
  typedef logic [CfgDataWidth-1:0] cfg_data_t;
  typedef logic [CfgStrbWidth-1:0] cfg_strb_t;

  `APB_TYPEDEF_REQ_T(apb_req_t, cfg_addr_t, cfg_data_t, cfg_strb_t)
  `APB_TYPEDEF_RESP_T(apb_rsp_t, cfg_data_t)

  `IDMA_TYPEDEF_FULL_REQ_T(idma_req_t, id_t, addr_t, tf_len_t)
  `IDMA_TYPEDEF_FULL_RSP_T(idma_rsp_t, addr_t)
  `IDMA_TYPEDEF_FULL_ND_REQ_T(idma_nd_req_t, idma_req_t, reps_t, strides_t)

  // backend meta channel types (as in iDMA test/include/tb_idma_mx_common.svh)
  typedef struct packed { axi_ar_chan_t ar_chan; } axi_read_meta_channel_t;
  typedef struct packed { axi_read_meta_channel_t axi; } read_meta_channel_t;
  typedef struct packed { axi_aw_chan_t aw_chan; } axi_write_meta_channel_t;
  typedef struct packed { axi_write_meta_channel_t axi; } write_meta_channel_t;

  typedef struct packed {
    int unsigned idx;
    cfg_addr_t   start_addr;
    cfg_addr_t   end_addr;
  } rule_t;

  // ------------------------------------------------------------------------------------------
  // AXI4-Lite -> APB (single slave: the whole cfg window maps to the reg frontend)
  // ------------------------------------------------------------------------------------------
  apb_req_t apb_req;
  apb_rsp_t apb_rsp;

  localparam rule_t [0:0] CfgMap = '{'{idx: 32'd0, start_addr: '0, end_addr: '1}};

  axi_lite_to_apb #(
    .NoApbSlaves     ( 32'd1        ),
    .NoRules         ( 32'd1        ),
    .AddrWidth       ( CfgAddrWidth ),
    .DataWidth       ( CfgDataWidth ),
    .PipelineRequest ( 1'b0         ),
    .PipelineResponse( 1'b0         ),
    .axi_lite_req_t  ( axil_req_t   ),
    .axi_lite_resp_t ( axil_rsp_t   ),
    .apb_req_t       ( apb_req_t    ),
    .apb_resp_t      ( apb_rsp_t    ),
    .rule_t          ( rule_t       )
  ) i_axil_to_apb (
    .clk_i,
    .rst_ni,
    .axi_lite_req_i  ( cfg_req_i ),
    .axi_lite_resp_o ( cfg_rsp_o ),
    .apb_req_o       ( apb_req   ),
    .apb_resp_i      ( apb_rsp   ),
    .addr_map_i      ( CfgMap    )
  );

  // ------------------------------------------------------------------------------------------
  // Register frontend + transfer-ID tracking
  // ------------------------------------------------------------------------------------------
  idma_nd_req_t          fe_req;
  logic                  fe_req_valid, fe_req_ready;
  logic [IdCounterWidth-1:0] next_id, done_id;
  idma_pkg::idma_busy_t  be_busy;
  logic                  mid_busy, nd_mid_busy, arb_busy;
  logic                  nd_rsp_valid, nd_rsp_ready;
  logic                  host_retire;

  idma_reg64_2d #(
    .NumRegs        ( 32'd1          ),
    .NumStreams     ( 32'd1          ),
    .IdCounterWidth ( IdCounterWidth ),
    .DataWidth      ( DataWidth      ),
    .ComputeOps     ( '0             ),  // plain copies only; MX compute ops refused
    .apb_req_t      ( apb_req_t      ),
    .apb_rsp_t      ( apb_rsp_t      ),
    .dma_req_t      ( idma_nd_req_t  )
  ) i_reg_fe (
    .clk_i,
    .rst_ni,
    .dma_ctrl_req_i ( apb_req      ),
    .dma_ctrl_rsp_o ( apb_rsp      ),
    .dma_req_o      ( fe_req       ),
    .req_valid_o    ( fe_req_valid ),
    .req_ready_i    ( fe_req_ready ),
    .next_id_i      ( next_id      ),
    .stream_idx_o   ( /* single stream */ ),
    .done_id_i      ( done_id      ),
    .busy_i         ( be_busy      ),
    .midend_busy_i  ( mid_busy     )
  );

  idma_transfer_id_gen #(
    .IdWidth ( IdCounterWidth )
  ) i_id_gen (
    .clk_i,
    .rst_ni,
    .issue_i     ( fe_req_valid & fe_req_ready ),
    .retire_i    ( host_retire ),
    .next_o      ( next_id ),
    .completed_o ( done_id )
  );

  // ------------------------------------------------------------------------------------------
  // ND midend (2D)
  idma_rsp_t nd_rsp;
  idma_nd_req_t nd_req;
  logic nd_req_valid, nd_req_ready;
  // ------------------------------------------------------------------------------------------
  idma_req_t be_req;
  logic      be_req_valid, be_req_ready;
  idma_rsp_t be_rsp;
  logic      be_rsp_valid, be_rsp_ready;

  idma_nd_midend #(
    .NumDim        ( NumDim               ),
    .addr_t        ( addr_t               ),
    .idma_req_t    ( idma_req_t           ),
    .idma_rsp_t    ( idma_rsp_t           ),
    .idma_nd_req_t ( idma_nd_req_t        ),
    .RepWidths     ( '{default: 32'd32}   )
  ) i_nd_midend (
    .clk_i,
    .rst_ni,
    .nd_req_i          ( nd_req       ),
    .nd_req_valid_i    ( nd_req_valid ),
    .nd_req_ready_o    ( nd_req_ready ),
    .nd_rsp_o          ( nd_rsp       ),
    .nd_rsp_valid_o    ( nd_rsp_valid ),
    .nd_rsp_ready_i    ( nd_rsp_ready ),
    .burst_req_o       ( be_req       ),
    .burst_req_valid_o ( be_req_valid ),
    .burst_req_ready_i ( be_req_ready ),
    .burst_rsp_i       ( be_rsp       ),
    .burst_rsp_valid_i ( be_rsp_valid ),
    .burst_rsp_ready_o ( be_rsp_ready ),
    .busy_o            ( nd_mid_busy  )
  );

  if (EnableCore) begin : gen_core_requester
    idma_nd_req_t core_req;
    idma_rsp_t core_rsp;
    always_comb begin
      core_req = '0;
      core_req.burst_req.src_addr = core_src_addr_i;
      core_req.burst_req.dst_addr = core_dst_addr_i;
      core_req.burst_req.length = core_length_i;
      core_req.burst_req.user = 1'b1; // trusted core requester; frontend/HPM requests keep user=0
      core_req.burst_req.opt.src_protocol = idma_pkg::AXI;
      core_req.burst_req.opt.dst_protocol = idma_pkg::AXI;
      core_req.burst_req.opt.src.burst = axi_pkg::BURST_INCR;
      core_req.burst_req.opt.dst.burst = axi_pkg::BURST_INCR;
      core_req.burst_req.opt.src.cache = axi_pkg::CACHE_MODIFIABLE;
      core_req.burst_req.opt.dst.cache = axi_pkg::CACHE_MODIFIABLE;
      core_req.d_req[0].reps = reps_t'(core_reps_i);
      core_req.d_req[0].src_strides = core_src_stride_i;
      core_req.d_req[0].dst_strides = core_dst_stride_i;
    end
    idma_request_arbiter #(
      .Outstanding(CoreOutstanding), .req_t(idma_nd_req_t), .rsp_t(idma_rsp_t)
    ) i_requests (
      .clk_i, .rst_ni,
      .host_req_i(fe_req), .host_req_valid_i(fe_req_valid), .host_req_ready_o(fe_req_ready),
      .host_rsp_valid_o(host_retire), .host_rsp_ready_i(1'b1),
      .core_req_i(core_req), .core_req_valid_i, .core_req_ready_o,
      .core_rsp_o(core_rsp), .core_rsp_valid_o, .core_rsp_ready_i,
      .nd_req_o(nd_req), .nd_req_valid_o(nd_req_valid), .nd_req_ready_i(nd_req_ready),
      .nd_rsp_i(nd_rsp), .nd_rsp_valid_i(nd_rsp_valid), .nd_rsp_ready_o(nd_rsp_ready),
      .busy_o(arb_busy)
    );
    assign core_rsp_error_o = core_rsp.error;
  end else begin : gen_host_only
    assign nd_req = fe_req;
    assign nd_req_valid = fe_req_valid;
    assign fe_req_ready = nd_req_ready;
    assign nd_rsp_ready = 1'b1;
    assign host_retire = nd_rsp_valid;
    assign arb_busy = 1'b0;
    assign core_req_ready_o = 1'b0;
    assign core_rsp_valid_o = 1'b0;
    assign core_rsp_error_o = 1'b0;
  end
  assign mid_busy = nd_mid_busy | arb_busy;

  // ------------------------------------------------------------------------------------------
  // Backend rw_axi
  // ------------------------------------------------------------------------------------------
  axi_req_t axi_read_req, axi_write_req;
  axi_rsp_t axi_read_rsp, axi_write_rsp;

  idma_backend_rw_axi #(
    .DataWidth            ( DataWidth            ),
    .AddrWidth            ( AddrWidth            ),
    .UserWidth            ( UserWidth            ),
    .AxiIdWidth           ( AxiIdWidth           ),
    .NumAxInFlight        ( NumAxInFlight        ),
    .BufferDepth          ( BufferDepth          ),
    .TFLenWidth           ( TFLenWidth           ),
    .MemSysDepth          ( 32'd0                ),
    .CombinedShifter      ( 1'b0                 ),
    .EnableCompute        ( 1'b0                 ),
    .RAWCouplingAvail     ( 1'b1                 ),
    .MaskInvalidData      ( 1'b1                 ),
    .HardwareLegalizer    ( 1'b1                 ),
    .RejectZeroTransfers  ( 1'b1                 ),
    .ErrorCap             ( idma_pkg::NO_ERROR_HANDLING ),
    .PrintFifoInfo        ( 1'b0                 ),
    .idma_req_t           ( idma_req_t           ),
    .idma_rsp_t           ( idma_rsp_t           ),
    .idma_eh_req_t        ( idma_pkg::idma_eh_req_t ),
    .idma_busy_t          ( idma_pkg::idma_busy_t ),
    .axi_req_t            ( axi_req_t            ),
    .axi_rsp_t            ( axi_rsp_t            ),
    .read_meta_channel_t  ( read_meta_channel_t  ),
    .write_meta_channel_t ( write_meta_channel_t )
  ) i_backend (
    .clk_i,
    .rst_ni,
    .idma_req_i      ( be_req        ),
    .req_valid_i     ( be_req_valid  ),
    .req_ready_o     ( be_req_ready  ),
    .idma_rsp_o      ( be_rsp        ),
    .rsp_valid_o     ( be_rsp_valid  ),
    .rsp_ready_i     ( be_rsp_ready  ),
    .idma_eh_req_i   ( '0            ),
    .eh_req_valid_i  ( 1'b0          ),
    .eh_req_ready_o  ( /* unused */  ),
    .axi_read_req_o  ( axi_read_req  ),
    .axi_read_rsp_i  ( axi_read_rsp  ),
    .axi_write_req_o ( axi_write_req ),
    .axi_write_rsp_i ( axi_write_rsp ),
    .busy_o          ( be_busy       )
  );

  axi_rw_join #(
    .axi_req_t  ( axi_req_t ),
    .axi_resp_t ( axi_rsp_t )
  ) i_rw_join (
    .clk_i,
    .rst_ni,
    .slv_read_req_i   ( axi_read_req  ),
    .slv_read_resp_o  ( axi_read_rsp  ),
    .slv_write_req_i  ( axi_write_req ),
    .slv_write_resp_o ( axi_write_rsp ),
    .mst_req_o        ( axi_req_o     ),
    .mst_resp_i       ( axi_rsp_i     )
  );

  // ------------------------------------------------------------------------------------------
  // Passive error monitor. With ErrorCap = NO_ERROR_HANDLING the backend ties idma_rsp.error
  // to 0, so a SLVERR/DECERR from the memory system would otherwise be invisible: the transfer
  // "completes" and done_id advances (M1 finding). This only observes responses.
  // ------------------------------------------------------------------------------------------
  logic [2:0] err_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      err_q <= '0;
    end else if (err_clr_i) begin
      err_q <= '0;
    end else begin
      if (axi_rsp_i.r_valid && axi_req_o.r_ready && axi_rsp_i.r.resp != axi_pkg::RESP_OKAY)
        err_q[0] <= 1'b1;
      if (axi_rsp_i.b_valid && axi_req_o.b_ready && axi_rsp_i.b.resp != axi_pkg::RESP_OKAY)
        err_q[1] <= 1'b1;
      if (nd_rsp_valid && nd_rsp_ready && nd_rsp.error)
        err_q[2] <= 1'b1;
    end
  end
  assign err_o = err_q;

  assign done_id_o = 32'(done_id);
  assign busy_o    = mid_busy | (|be_busy);

endmodule
