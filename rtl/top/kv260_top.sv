// TPU-KV260 - board top (M2: data fabric + on-chip memories).
//
//   PS M_AXI_HPM0 (128b) -> axi_iw_converter 16->2 -> axi_demux_simple (by address)
//      < 0xA010_0000 : axi_dw_converter 128->32 -> axi_to_axi_lite -> axi_lite_xbar
//                        0xA000_0000 +4K : tpu_regs (axi_lite_regs)
//                        0xA000_1000 +4K : iDMA reg64_2d cfg
//      >= 0xA010_0000 : tpu_fabric slave port 1 (UB / ACC / WFIFO / IFIFO debug window)
//   idma_subsystem -> tpu_fabric slave port 0 -> {HP0 -> DDR, UB, ACC, WFIFO, IFIFO}
//   tpu_core drives the memories' native ports and consumes the WFIFO/IFIFO.
//
// Address map: rtl/common/tpu_pkg.sv = docs/interfaces.md. One clock (pl_clk0, 100 MHz).

`include "axi/typedef.svh"
`include "axi/assign.svh"

`timescale 1ns/1ps

module kv260_top #(
  parameter logic [31:0] BUILD_ID = 32'h0,  // stamped by the build script (L7)
  parameter bit EnableDma = 1'b1
) ();

  import tpu_m0_pkg::axil_req_t;
  import tpu_m0_pkg::axil_resp_t;
  import tpu_m0_pkg::axil_aw_chan_t;
  import tpu_m0_pkg::axil_w_chan_t;
  import tpu_m0_pkg::axil_b_chan_t;
  import tpu_m0_pkg::axil_ar_chan_t;
  import tpu_m0_pkg::axil_r_chan_t;

  localparam logic [31:0] Version = EnableDma ? 32'h0000_0500 : 32'h0000_0300;  // 0x500: Phase B + CNN (CONVOLVE, pooling)

  // ------------------------------------------------------------------------------------------
  // PS
  // ------------------------------------------------------------------------------------------
  logic clk, ps_rst_n, rst_n;

  // HPM0 flat
  logic [39:0]  hpm_araddr, hpm_awaddr;
  logic [1:0]   hpm_arburst, hpm_awburst, hpm_bresp, hpm_rresp;
  logic [3:0]   hpm_arcache, hpm_awcache, hpm_arqos, hpm_awqos;
  logic [15:0]  hpm_arid, hpm_awid, hpm_bid, hpm_rid, hpm_aruser, hpm_awuser, hpm_wstrb;
  logic [7:0]   hpm_arlen, hpm_awlen;
  logic         hpm_arlock, hpm_awlock;
  logic [2:0]   hpm_arprot, hpm_awprot, hpm_arsize, hpm_awsize;
  logic         hpm_arready, hpm_arvalid, hpm_awready, hpm_awvalid;
  logic         hpm_bready, hpm_bvalid, hpm_rlast, hpm_rready, hpm_rvalid;
  logic [127:0] hpm_rdata, hpm_wdata;
  logic         hpm_wlast, hpm_wready, hpm_wvalid;

  // HP0 flat
  logic [48:0]  hp_araddr, hp_awaddr;
  logic [1:0]   hp_arburst, hp_awburst, hp_bresp, hp_rresp;
  logic [3:0]   hp_arcache, hp_awcache, hp_arqos, hp_awqos;
  logic [5:0]   hp_arid, hp_awid, hp_bid, hp_rid;
  logic [7:0]   hp_arlen, hp_awlen;
  logic         hp_arlock, hp_awlock, hp_aruser, hp_awuser;
  logic [2:0]   hp_arprot, hp_awprot, hp_arsize, hp_awsize;
  logic         hp_arready, hp_arvalid, hp_awready, hp_awvalid;
  logic         hp_bready, hp_bvalid, hp_rlast, hp_rready, hp_rvalid;
  logic [127:0] hp_rdata, hp_wdata;
  logic [15:0]  hp_wstrb;
  logic         hp_wlast, hp_wready, hp_wvalid;

  logic [0:0]   irq;

  kv260_ps_wrapper i_ps (
    .M_AXI_HPM0_araddr (hpm_araddr ), .M_AXI_HPM0_arburst(hpm_arburst), .M_AXI_HPM0_arcache(hpm_arcache),
    .M_AXI_HPM0_arid   (hpm_arid   ), .M_AXI_HPM0_arlen  (hpm_arlen  ), .M_AXI_HPM0_arlock (hpm_arlock ),
    .M_AXI_HPM0_arprot (hpm_arprot ), .M_AXI_HPM0_arqos  (hpm_arqos  ), .M_AXI_HPM0_arready(hpm_arready),
    .M_AXI_HPM0_arsize (hpm_arsize ), .M_AXI_HPM0_aruser (hpm_aruser ), .M_AXI_HPM0_arvalid(hpm_arvalid),
    .M_AXI_HPM0_awaddr (hpm_awaddr ), .M_AXI_HPM0_awburst(hpm_awburst), .M_AXI_HPM0_awcache(hpm_awcache),
    .M_AXI_HPM0_awid   (hpm_awid   ), .M_AXI_HPM0_awlen  (hpm_awlen  ), .M_AXI_HPM0_awlock (hpm_awlock ),
    .M_AXI_HPM0_awprot (hpm_awprot ), .M_AXI_HPM0_awqos  (hpm_awqos  ), .M_AXI_HPM0_awready(hpm_awready),
    .M_AXI_HPM0_awsize (hpm_awsize ), .M_AXI_HPM0_awuser (hpm_awuser ), .M_AXI_HPM0_awvalid(hpm_awvalid),
    .M_AXI_HPM0_bid    (hpm_bid    ), .M_AXI_HPM0_bready (hpm_bready ), .M_AXI_HPM0_bresp  (hpm_bresp  ),
    .M_AXI_HPM0_bvalid (hpm_bvalid ), .M_AXI_HPM0_rdata  (hpm_rdata  ), .M_AXI_HPM0_rid    (hpm_rid    ),
    .M_AXI_HPM0_rlast  (hpm_rlast  ), .M_AXI_HPM0_rready (hpm_rready ), .M_AXI_HPM0_rresp  (hpm_rresp  ),
    .M_AXI_HPM0_rvalid (hpm_rvalid ), .M_AXI_HPM0_wdata  (hpm_wdata  ), .M_AXI_HPM0_wlast  (hpm_wlast  ),
    .M_AXI_HPM0_wready (hpm_wready ), .M_AXI_HPM0_wstrb  (hpm_wstrb  ), .M_AXI_HPM0_wvalid (hpm_wvalid ),

    .S_AXI_HP0_araddr  (hp_araddr  ), .S_AXI_HP0_arburst (hp_arburst ), .S_AXI_HP0_arcache (hp_arcache ),
    .S_AXI_HP0_arid    (hp_arid    ), .S_AXI_HP0_arlen   (hp_arlen   ), .S_AXI_HP0_arlock  (hp_arlock  ),
    .S_AXI_HP0_arprot  (hp_arprot  ), .S_AXI_HP0_arqos   (hp_arqos   ), .S_AXI_HP0_arready (hp_arready ),
    .S_AXI_HP0_arsize  (hp_arsize  ), .S_AXI_HP0_aruser  (hp_aruser  ), .S_AXI_HP0_arvalid (hp_arvalid ),
    .S_AXI_HP0_awaddr  (hp_awaddr  ), .S_AXI_HP0_awburst (hp_awburst ), .S_AXI_HP0_awcache (hp_awcache ),
    .S_AXI_HP0_awid    (hp_awid    ), .S_AXI_HP0_awlen   (hp_awlen   ), .S_AXI_HP0_awlock  (hp_awlock  ),
    .S_AXI_HP0_awprot  (hp_awprot  ), .S_AXI_HP0_awqos   (hp_awqos   ), .S_AXI_HP0_awready (hp_awready ),
    .S_AXI_HP0_awsize  (hp_awsize  ), .S_AXI_HP0_awuser  (hp_awuser  ), .S_AXI_HP0_awvalid (hp_awvalid ),
    .S_AXI_HP0_bid     (hp_bid     ), .S_AXI_HP0_bready  (hp_bready  ), .S_AXI_HP0_bresp   (hp_bresp   ),
    .S_AXI_HP0_bvalid  (hp_bvalid  ), .S_AXI_HP0_rdata   (hp_rdata   ), .S_AXI_HP0_rid     (hp_rid     ),
    .S_AXI_HP0_rlast   (hp_rlast   ), .S_AXI_HP0_rready  (hp_rready  ), .S_AXI_HP0_rresp   (hp_rresp   ),
    .S_AXI_HP0_rvalid  (hp_rvalid  ), .S_AXI_HP0_wdata   (hp_wdata   ), .S_AXI_HP0_wlast   (hp_wlast   ),
    .S_AXI_HP0_wready  (hp_wready  ), .S_AXI_HP0_wstrb   (hp_wstrb   ), .S_AXI_HP0_wvalid  (hp_wvalid  ),

    .pl_clk0    ( clk      ),
    .pl_resetn0 ( ps_rst_n ),
    .pl_ps_irq0 ( irq      )
  );

  // reset: async assert, sync deassert
  logic [1:0] rst_sync_q;
  always_ff @(posedge clk or negedge ps_rst_n) begin
    if (!ps_rst_n) rst_sync_q <= 2'b00;
    else           rst_sync_q <= {rst_sync_q[0], 1'b1};
  end
  assign rst_n = rst_sync_q[1];

  // ------------------------------------------------------------------------------------------
  // HPM0 side types (32-bit addr: the PL window is 0xA000_0000; 16-bit IDs)
  // ------------------------------------------------------------------------------------------
  typedef logic [31:0]  h_addr_t;
  typedef logic [15:0]  h_id_t;
  typedef logic [0:0]   h_user_t;
  typedef logic [127:0] h_data_t;
  typedef logic [15:0]  h_strb_t;
  typedef logic [31:0]  n_data_t;
  typedef logic [3:0]   n_strb_t;
  `AXI_TYPEDEF_AW_CHAN_T(h_aw_t, h_addr_t, h_id_t, h_user_t)
  `AXI_TYPEDEF_AR_CHAN_T(h_ar_t, h_addr_t, h_id_t, h_user_t)
  `AXI_TYPEDEF_B_CHAN_T (h_b_t, h_id_t, h_user_t)
  `AXI_TYPEDEF_W_CHAN_T (hw_w_t, h_data_t, h_strb_t, h_user_t)   // wide (128b)
  `AXI_TYPEDEF_R_CHAN_T (hw_r_t, h_data_t, h_id_t, h_user_t)
  `AXI_TYPEDEF_W_CHAN_T (hn_w_t, n_data_t, n_strb_t, h_user_t)   // narrow (32b)
  `AXI_TYPEDEF_R_CHAN_T (hn_r_t, n_data_t, h_id_t, h_user_t)
  `AXI_TYPEDEF_REQ_T (hw_req_t,  h_aw_t, hw_w_t, h_ar_t)
  `AXI_TYPEDEF_RESP_T(hw_resp_t, h_b_t,  hw_r_t)
  // after ID narrowing (16 -> 2 bits): PULP ID-tracking logic is sized 2^IdWidth
  typedef logic [1:0] s_id_t;
  `AXI_TYPEDEF_AW_CHAN_T(s_aw_t, h_addr_t, s_id_t, h_user_t)
  `AXI_TYPEDEF_AR_CHAN_T(s_ar_t, h_addr_t, s_id_t, h_user_t)
  `AXI_TYPEDEF_B_CHAN_T (s_b_t, s_id_t, h_user_t)
  `AXI_TYPEDEF_R_CHAN_T (sw_r_t, h_data_t, s_id_t, h_user_t)
  `AXI_TYPEDEF_R_CHAN_T (sn_r_t, n_data_t, s_id_t, h_user_t)
  `AXI_TYPEDEF_REQ_T (sw_req_t,  s_aw_t, hw_w_t, s_ar_t)
  `AXI_TYPEDEF_RESP_T(sw_resp_t, s_b_t,  sw_r_t)
  `AXI_TYPEDEF_REQ_T (hn_req_t,  s_aw_t, hn_w_t, s_ar_t)
  `AXI_TYPEDEF_RESP_T(hn_resp_t, s_b_t,  sn_r_t)

  hw_req_t  hpm_req;
  hw_resp_t hpm_rsp;

  always_comb begin
    hpm_req = '0;
    hpm_req.aw.id     = hpm_awid;    hpm_req.aw.addr  = hpm_awaddr[31:0];
    hpm_req.aw.len    = hpm_awlen;   hpm_req.aw.size  = hpm_awsize;
    hpm_req.aw.burst  = hpm_awburst; hpm_req.aw.lock  = hpm_awlock;
    hpm_req.aw.cache  = hpm_awcache; hpm_req.aw.prot  = hpm_awprot;
    hpm_req.aw.qos    = hpm_awqos;
    hpm_req.aw_valid  = hpm_awvalid;
    hpm_req.w.data    = hpm_wdata;   hpm_req.w.strb   = hpm_wstrb;
    hpm_req.w.last    = hpm_wlast;
    hpm_req.w_valid   = hpm_wvalid;
    hpm_req.b_ready   = hpm_bready;
    hpm_req.ar.id     = hpm_arid;    hpm_req.ar.addr  = hpm_araddr[31:0];
    hpm_req.ar.len    = hpm_arlen;   hpm_req.ar.size  = hpm_arsize;
    hpm_req.ar.burst  = hpm_arburst; hpm_req.ar.lock  = hpm_arlock;
    hpm_req.ar.cache  = hpm_arcache; hpm_req.ar.prot  = hpm_arprot;
    hpm_req.ar.qos    = hpm_arqos;
    hpm_req.ar_valid  = hpm_arvalid;
    hpm_req.r_ready   = hpm_rready;
  end
  assign hpm_awready = hpm_rsp.aw_ready;
  assign hpm_wready  = hpm_rsp.w_ready;
  assign hpm_bid     = hpm_rsp.b.id;
  assign hpm_bresp   = hpm_rsp.b.resp;
  assign hpm_bvalid  = hpm_rsp.b_valid;
  assign hpm_arready = hpm_rsp.ar_ready;
  assign hpm_rid     = hpm_rsp.r.id;
  assign hpm_rdata   = hpm_rsp.r.data;
  assign hpm_rresp   = hpm_rsp.r.resp;
  assign hpm_rlast   = hpm_rsp.r.last;
  assign hpm_rvalid  = hpm_rsp.r_valid;

  // ID width 16 -> 2 (remap; the PS issues few distinct IDs to this window)
  sw_req_t  hpms_req;
  sw_resp_t hpms_rsp;

  axi_iw_converter #(
    .AxiSlvPortIdWidth      ( 16        ),
    .AxiMstPortIdWidth      ( 2         ),
    .AxiSlvPortMaxUniqIds   ( 4         ),
    .AxiSlvPortMaxTxnsPerId ( 2         ),
    .AxiSlvPortMaxTxns      ( 4         ),
    .AxiMstPortMaxUniqIds   ( 4         ),
    .AxiMstPortMaxTxnsPerId ( 2         ),
    .AxiAddrWidth           ( 32        ),
    .AxiDataWidth           ( 128       ),
    .AxiUserWidth           ( 1         ),
    .slv_req_t              ( hw_req_t  ),
    .slv_resp_t             ( hw_resp_t ),
    .mst_req_t              ( sw_req_t  ),
    .mst_resp_t             ( sw_resp_t )
  ) i_hpm_iw (
    .clk_i      ( clk      ),
    .rst_ni     ( rst_n    ),
    .slv_req_i  ( hpm_req  ),
    .slv_resp_o ( hpm_rsp  ),
    .mst_req_o  ( hpms_req ),
    .mst_resp_i ( hpms_rsp )
  );

  // split by address: config (AXI-Lite) region vs. data fabric window
  sw_req_t  [1:0] hsplit_req;
  sw_resp_t [1:0] hsplit_rsp;
  wire hpm_aw_data = (hpms_req.aw.addr >= 32'(tpu_pkg::CFG_END));
  wire hpm_ar_data = (hpms_req.ar.addr >= 32'(tpu_pkg::CFG_END));

  axi_demux_simple #(
    .AxiIdWidth  ( 2         ),
    .AtopSupport ( 1'b0      ),
    .axi_req_t   ( sw_req_t  ),
    .axi_resp_t  ( sw_resp_t ),
    .NoMstPorts  ( 2         ),
    .MaxTrans    ( 4         ),
    .AxiLookBits ( 2         ),
    .UniqueIds   ( 1'b0      )
  ) i_hpm_split (
    .clk_i           ( clk         ),
    .rst_ni          ( rst_n       ),
    .slv_req_i       ( hpms_req    ),
    .slv_aw_select_i ( hpm_aw_data ),
    .slv_ar_select_i ( hpm_ar_data ),
    .slv_resp_o      ( hpms_rsp    ),
    .mst_reqs_o      ( hsplit_req  ),
    .mst_resps_i     ( hsplit_rsp  )
  );

  // data window: widen to the fabric's slave-port types (64-bit addr, 5-bit IDs).
  // Intentional width change (L15 waiver): addr/ID zero-extended on requests; response IDs are
  // the same zero-extended 2-bit IDs, so truncating them back is lossless.
  /* verilator lint_off WIDTHEXPAND */
  /* verilator lint_off WIDTHTRUNC */
  tpu_pkg::fs_req_t  hpm_data_req;
  tpu_pkg::fs_resp_t hpm_data_rsp;
  always_comb begin
    hpm_data_req = '0;
    `AXI_SET_REQ_STRUCT(hpm_data_req, hsplit_req[1])
  end
  always_comb begin
    hsplit_rsp[1] = '0;
    `AXI_SET_RESP_STRUCT(hsplit_rsp[1], hpm_data_rsp)
  end
  /* verilator lint_on WIDTHTRUNC */
  /* verilator lint_on WIDTHEXPAND */

  // 128 -> 32 data width (config region)
  hn_req_t  hpmn_req;
  hn_resp_t hpmn_rsp;

  axi_dw_converter #(
    .AxiMaxReads         ( 1        ),
    .AxiSlvPortDataWidth ( 128      ),
    .AxiMstPortDataWidth ( 32       ),
    .AxiAddrWidth        ( 32       ),
    .AxiIdWidth          ( 2        ),
    .aw_chan_t           ( s_aw_t   ),
    .mst_w_chan_t        ( hn_w_t   ),
    .slv_w_chan_t        ( hw_w_t   ),
    .b_chan_t            ( s_b_t    ),
    .ar_chan_t           ( s_ar_t   ),
    .mst_r_chan_t        ( sn_r_t   ),
    .slv_r_chan_t        ( sw_r_t   ),
    .axi_mst_req_t       ( hn_req_t ),
    .axi_mst_resp_t      ( hn_resp_t),
    .axi_slv_req_t       ( sw_req_t ),
    .axi_slv_resp_t      ( sw_resp_t)
  ) i_hpm_dw (
    .clk_i      ( clk      ),
    .rst_ni     ( rst_n    ),
    .slv_req_i  ( hsplit_req[0] ),
    .slv_resp_o ( hsplit_rsp[0] ),
    .mst_req_o  ( hpmn_req ),
    .mst_resp_i ( hpmn_rsp )
  );

  // AXI4 -> AXI4-Lite
  axil_req_t  lite_req;
  axil_resp_t lite_rsp;

  axi_to_axi_lite #(
    .AxiAddrWidth    ( 32          ),
    .AxiDataWidth    ( 32          ),
    .AxiIdWidth      ( 2           ),
    .AxiUserWidth    ( 1           ),
    .AxiMaxWriteTxns ( 2           ),
    .AxiMaxReadTxns  ( 2           ),
    .FullBW          ( 1'b0        ),
    .FallThrough     ( 1'b0        ),
    .full_req_t      ( hn_req_t    ),
    .full_resp_t     ( hn_resp_t   ),
    .lite_req_t      ( axil_req_t  ),
    .lite_resp_t     ( axil_resp_t )
  ) i_to_lite (
    .clk_i      ( clk      ),
    .rst_ni     ( rst_n    ),
    .slv_req_i  ( hpmn_req ),
    .slv_resp_o ( hpmn_rsp ),
    .mst_req_o  ( lite_req ),
    .mst_resp_i ( lite_rsp )
  );

  // AXI4-Lite xbar: 1 slave port, 2 master ports, DECERR elsewhere
  localparam axi_pkg::xbar_cfg_t LiteXbarCfg = '{
    NoSlvPorts:         32'd1,
    NoMstPorts:         32'd2,
    MaxMstTrans:        32'd2,
    MaxSlvTrans:        32'd2,
    FallThrough:        1'b0,
    LatencyMode:        axi_pkg::CUT_ALL_PORTS,
    PipelineStages:     32'd0,
    AxiIdWidthSlvPorts: 32'd0,
    AxiIdUsedSlvPorts:  32'd0,
    UniqueIds:          1'b0,
    AxiAddrWidth:       32'd32,
    AxiDataWidth:       32'd32,
    NoAddrRules:        32'd2
  };
  localparam axi_pkg::xbar_rule_32_t [1:0] LiteMap = '{
    '{idx: 32'd1, start_addr: 32'hA000_1000, end_addr: 32'hA000_2000},  // iDMA cfg
    '{idx: 32'd0, start_addr: 32'hA000_0000, end_addr: 32'hA000_1000}   // tpu regs
  };

  axil_req_t  [1:0] lx_req;
  axil_resp_t [1:0] lx_rsp;

  axi_lite_xbar #(
    .Cfg        ( LiteXbarCfg            ),
    .aw_chan_t  ( axil_aw_chan_t         ),
    .w_chan_t   ( axil_w_chan_t          ),
    .b_chan_t   ( axil_b_chan_t          ),
    .ar_chan_t  ( axil_ar_chan_t         ),
    .r_chan_t   ( axil_r_chan_t          ),
    .axi_req_t  ( axil_req_t             ),
    .axi_resp_t ( axil_resp_t            ),
    .rule_t     ( axi_pkg::xbar_rule_32_t)
  ) i_lite_xbar (
    .clk_i                 ( clk        ),
    .rst_ni                ( rst_n      ),
    .slv_ports_req_i       ( lite_req   ),
    .slv_ports_resp_o      ( lite_rsp   ),
    .mst_ports_req_o       ( lx_req     ),
    .mst_ports_resp_i      ( lx_rsp     ),
    .addr_map_i            ( LiteMap    ),
    .en_default_mst_port_i ( 1'b0       ),
    .default_mst_port_i    ( '0         )
  );

  // ------------------------------------------------------------------------------------------
  // Registers (PULP axi_lite_regs): 72 bytes (tpu_regs v0.3)
  //   0x00 VERSION (RO)  0x04 SCRATCH (RW)  0x08 BUILD_ID (RO)  0x0C IDMA_DONE_ID (RO, live)
  //   0x10 STATUS  (RO, live: [0] idma busy, [1] HP0 upper-address error (sticky),
  //                 [2] iDMA read resp err, [3] iDMA write resp err, [4] iDMA midend err,
  //                 [5] UB ownership err, [6] ACC ownership err, [7] WFIFO err, [8] IFIFO err;
  //                 [8:1] sticky until CTRL[0])
  //   0x14 CTRL    (RW levels: [0] err_clr (fabric + core cfg_err), [1] UB owner=TPU,
  //                 [2] ACC owner=TPU, [3] WFIFO debug-pop, [4] IFIFO debug-pop,
  //                 [5] WFIFO flush, [6] IFIFO flush, [7] core run, [8] core IRQ clear (level),
  //                 [9] perf-counter clear)
  //   0x18 FIFO_LEVEL (RO: [15:0] WFIFO words, [31:16] IFIFO words)
  //   0x1C CORE_STATUS (RO: [0] busy, [1] halted, [2] cfg_err, [3] irq pending, [15:8] cfg_err opcode)
  //   0x20 IRQ_TAG (RO)   0x24..0x40 PERF0..7 (RO)   0x44 DEBUG_TAG (RO)
  // ------------------------------------------------------------------------------------------
  localparam int unsigned RegBytes = 72;
  // AxiReadOnly has one bit per byte: SCRATCH bytes 4..7 and CTRL bytes 20..23 are writable.
  localparam logic [RegBytes-1:0] RegRO = ~((72'hF << 4) | (72'hF << 20));

  logic [31:0] idma_done_id;
  logic        idma_busy;
  logic        addr_hi_err_q;
  logic [2:0]  idma_err;
  logic [3:0]  fab_err;
  logic [15:0] wf_level, if_level;
  logic        core_busy, core_halted, core_cfg_err, core_irq_pulse, core_irq_q;
  logic        core_dma_error;
  logic [7:0]  core_cfg_err_op;
  logic [31:0] core_irq_tag, core_dbg_tag;
  logic [31:0] core_perf [8];
  logic [RegBytes-1:0][7:0] reg_q;

  logic [RegBytes-1:0][7:0] reg_d;
  logic [RegBytes-1:0]      reg_load;

  always_comb begin
    reg_d    = '0;
    reg_load = '0;
    reg_d[3:0]   = Version;      reg_load[3:0]   = '1;
    reg_d[11:8]  = BUILD_ID;     reg_load[11:8]  = '1;
    reg_d[15:12] = idma_done_id; reg_load[15:12] = '1;
    reg_d[19:16] = {23'd0, fab_err, idma_err, addr_hi_err_q, idma_busy}; reg_load[19:16] = '1;
    reg_d[27:24] = {if_level, wf_level};                                  reg_load[27:24] = '1;
    reg_d[31:28] = {16'd0, core_cfg_err_op, 3'd0, core_dma_error, core_irq_q, core_cfg_err, core_halted, core_busy};
    reg_load[31:28] = '1;
    reg_d[35:32] = core_irq_tag; reg_load[35:32] = '1;
    for (int i = 0; i < 8; i++) begin
      reg_d[36+4*i +: 4] = core_perf[i]; reg_load[36+4*i +: 4] = '1;
    end
    reg_d[71:68] = core_dbg_tag; reg_load[71:68] = '1;
  end

  axi_lite_regs #(
    .RegNumBytes  ( RegBytes    ),
    .AxiAddrWidth ( 32          ),
    .AxiDataWidth ( 32          ),
    .AxiReadOnly  ( RegRO       ),
    .RegRstVal    ( '0          ),
    .req_lite_t   ( axil_req_t  ),
    .resp_lite_t  ( axil_resp_t )
  ) i_regs (
    .clk_i       ( clk       ),
    .rst_ni      ( rst_n     ),
    .axi_req_i   ( lx_req[0] ),
    .axi_resp_o  ( lx_rsp[0] ),
    .wr_active_o ( ),
    .rd_active_o ( ),
    .reg_d_i     ( reg_d     ),
    .reg_load_i  ( reg_load  ),
    .reg_q_o     ( reg_q     )
  );

  // ------------------------------------------------------------------------------------------
  // iDMA -> HP0
  // ------------------------------------------------------------------------------------------
  tpu_pkg::fs_req_t  dma_req;
  tpu_pkg::fs_resp_t dma_rsp;
  tpu_pkg::fm_req_t  hp0_req;
  tpu_pkg::fm_resp_t hp0_rsp;
  logic dma_core_valid, dma_core_ready, dma_core_rsp_valid, dma_core_rsp_ready, dma_core_rsp_error;
  logic [63:0] dma_core_src, dma_core_dst, dma_core_src_stride, dma_core_dst_stride;
  logic [23:0] dma_core_length;
  logic [31:0] dma_core_reps;

  idma_subsystem #(
    .DataWidth     ( tpu_pkg::DataWidth   ),
    .AddrWidth     ( tpu_pkg::AddrWidth   ),
    .AxiIdWidth    ( tpu_pkg::SlvIdWidth  ),
    .UserWidth     ( tpu_pkg::UserWidth   ),
    .EnableCore    ( EnableDma ),
    .axi_req_t     ( tpu_pkg::fs_req_t    ),
    .axi_rsp_t     ( tpu_pkg::fs_resp_t   ),
    .axi_ar_chan_t ( tpu_pkg::fs_ar_chan_t ),
    .axi_aw_chan_t ( tpu_pkg::fs_aw_chan_t ),
    .axil_req_t    ( axil_req_t    ),
    .axil_rsp_t    ( axil_resp_t   )
  ) i_idma (
    .clk_i     ( clk          ),
    .rst_ni    ( rst_n        ),
    .cfg_req_i ( lx_req[1]    ),
    .cfg_rsp_o ( lx_rsp[1]    ),
    .axi_req_o ( dma_req      ),
    .axi_rsp_i ( dma_rsp      ),
    .done_id_o ( idma_done_id ),
    .busy_o    ( idma_busy    ),
    .err_o     ( idma_err     ),
    .err_clr_i ( reg_q[20][0] ),
    .core_req_valid_i(dma_core_valid), .core_req_ready_o(dma_core_ready),
    .core_src_addr_i(dma_core_src), .core_dst_addr_i(dma_core_dst),
    .core_length_i(dma_core_length), .core_reps_i(dma_core_reps),
    .core_src_stride_i(dma_core_src_stride), .core_dst_stride_i(dma_core_dst_stride),
    .core_rsp_valid_o(dma_core_rsp_valid), .core_rsp_ready_i(dma_core_rsp_ready),
    .core_rsp_error_o(dma_core_rsp_error)
  );

  // ------------------------------------------------------------------------------------------
  // Data fabric + memories + compute core (M4). The core drives the native ports and
  // consumes both FIFOs; AXI writes to UB/ACC are rejected while owned by the TPU.
  // ------------------------------------------------------------------------------------------
  tpu_pkg::ub_addr_t  core_ub_raddr, core_ub_waddr;
  tpu_pkg::data_t     core_ub_rdata, core_ub_wdata;
  tpu_pkg::strb_t     core_ub_wbe;
  tpu_pkg::acc_addr_t core_acc_raddr, core_acc_waddr;
  tpu_pkg::acc_row_t  core_acc_rdata, core_acc_wdata;
  logic [3:0]         core_acc_wbe;
  logic               core_ub_re, core_ub_we, core_acc_re, core_acc_we;
  logic               wf_valid, wf_ready, if_valid, if_ready;
  tpu_pkg::data_t     wf_data, if_data;

  tpu_fabric #(.AllowCoreDma(EnableDma)) i_fabric (
    .clk_i           ( clk ),
    .rst_ni          ( rst_n ),
    .dma_req_i       ( dma_req ),
    .dma_rsp_o       ( dma_rsp ),
    .hpm_req_i       ( hpm_data_req ),
    .hpm_rsp_o       ( hpm_data_rsp ),
    .hp0_req_o       ( hp0_req ),
    .hp0_rsp_i       ( hp0_rsp ),
    .ub_rd_en_i      ( core_ub_re ),   .ub_rd_addr_i ( core_ub_raddr ), .ub_rd_data_o ( core_ub_rdata ),
    .ub_wr_en_i      ( core_ub_we ),   .ub_wr_addr_i ( core_ub_waddr ), .ub_wr_data_i ( core_ub_wdata ),
    .ub_wr_be_i      ( core_ub_wbe ),
    .acc_rd_en_i     ( core_acc_re ),  .acc_rd_addr_i ( core_acc_raddr ), .acc_rd_data_o ( core_acc_rdata ),
    .acc_wr_en_i     ( core_acc_we ),  .acc_wr_addr_i ( core_acc_waddr ), .acc_wr_data_i ( core_acc_wdata ),
    .acc_wr_be_i     ( core_acc_wbe ),
    .wf_valid_o      ( wf_valid ), .wf_ready_i ( wf_ready ), .wf_data_o ( wf_data ),
    .if_valid_o      ( if_valid ), .if_ready_i ( if_ready ), .if_data_o ( if_data ),
    .ub_owner_tpu_i  ( reg_q[20][1] ),
    .acc_owner_tpu_i ( reg_q[20][2] ),
    .wf_dbg_pop_i    ( reg_q[20][3] ),
    .if_dbg_pop_i    ( reg_q[20][4] ),
    .wf_flush_i      ( reg_q[20][5] ),
    .if_flush_i      ( reg_q[20][6] ),
    .err_clr_i       ( reg_q[20][0] ),
    .err_o           ( fab_err  ),
    .wf_level_o      ( wf_level ),
    .if_level_o      ( if_level )
  );

  // HP0 is 49-bit: flag (sticky) any request with nonzero upper address bits.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) addr_hi_err_q <= 1'b0;
    else if ((hp0_req.aw_valid && |hp0_req.aw.addr[63:49]) ||
             (hp0_req.ar_valid && |hp0_req.ar.addr[63:49]))
      addr_hi_err_q <= 1'b1;
  end

  assign hp_awid    = hp0_req.aw.id;    assign hp_awaddr  = hp0_req.aw.addr[48:0];
  assign hp_awlen   = hp0_req.aw.len;   assign hp_awsize  = hp0_req.aw.size;
  assign hp_awburst = hp0_req.aw.burst; assign hp_awlock  = hp0_req.aw.lock;
  assign hp_awcache = hp0_req.aw.cache; assign hp_awprot  = hp0_req.aw.prot;
  assign hp_awqos   = hp0_req.aw.qos;   assign hp_awuser  = 1'b0;
  assign hp_awvalid = hp0_req.aw_valid;
  assign hp_wdata   = hp0_req.w.data;   assign hp_wstrb   = hp0_req.w.strb;
  assign hp_wlast   = hp0_req.w.last;   assign hp_wvalid  = hp0_req.w_valid;
  assign hp_bready  = hp0_req.b_ready;
  assign hp_arid    = hp0_req.ar.id;    assign hp_araddr  = hp0_req.ar.addr[48:0];
  assign hp_arlen   = hp0_req.ar.len;   assign hp_arsize  = hp0_req.ar.size;
  assign hp_arburst = hp0_req.ar.burst; assign hp_arlock  = hp0_req.ar.lock;
  assign hp_arcache = hp0_req.ar.cache; assign hp_arprot  = hp0_req.ar.prot;
  assign hp_arqos   = hp0_req.ar.qos;   assign hp_aruser  = 1'b0;
  assign hp_arvalid = hp0_req.ar_valid;
  assign hp_rready  = hp0_req.r_ready;

  always_comb begin
    hp0_rsp          = '0;
    hp0_rsp.aw_ready = hp_awready;
    hp0_rsp.w_ready  = hp_wready;
    hp0_rsp.b.id     = hp_bid;   hp0_rsp.b.resp = hp_bresp; hp0_rsp.b_valid = hp_bvalid;
    hp0_rsp.ar_ready = hp_arready;
    hp0_rsp.r.id     = hp_rid;   hp0_rsp.r.data = hp_rdata; hp0_rsp.r.resp  = hp_rresp;
    hp0_rsp.r.last   = hp_rlast; hp0_rsp.r_valid = hp_rvalid;
  end

  // ------------------------------------------------------------------------------------------
  // Compute core (M4): decoder + matmul/activate sequencers + matrix unit + activation unit.
  // run = CTRL[7] (0->1 edge restarts after HALT); cfg_err clears with CTRL[0]; perf CTRL[9].
  // ------------------------------------------------------------------------------------------
  tpu_core #(.N(16), .EnableDma(EnableDma)) i_core (
    .clk_i           ( clk ),
    .rst_ni          ( rst_n ),
    .if_valid_i      ( if_valid ), .if_ready_o ( if_ready ), .if_data_i ( if_data ),
    .wf_valid_i      ( wf_valid ), .wf_ready_o ( wf_ready ), .wf_data_i ( wf_data ),
    .ub_rd_en_o      ( core_ub_re ),   .ub_rd_addr_o ( core_ub_raddr ), .ub_rd_data_i ( core_ub_rdata ),
    .ub_wr_en_o      ( core_ub_we ),   .ub_wr_addr_o ( core_ub_waddr ), .ub_wr_data_o ( core_ub_wdata ),
    .ub_wr_be_o      ( core_ub_wbe ),
    .acc_rd_en_o     ( core_acc_re ),  .acc_rd_addr_o ( core_acc_raddr ), .acc_rd_data_i ( core_acc_rdata ),
    .acc_wr_en_o     ( core_acc_we ),  .acc_wr_addr_o ( core_acc_waddr ), .acc_wr_data_o ( core_acc_wdata ),
    .acc_wr_be_o     ( core_acc_wbe ),
    .run_i           ( reg_q[20][7] ),
    .err_clr_i       ( reg_q[20][0] ),
    .perf_clr_i      ( reg_q[21][1] ),
    .busy_o          ( core_busy ),
    .halted_o        ( core_halted ),
    .cfg_err_o       ( core_cfg_err ),
    .cfg_err_op_o    ( core_cfg_err_op ),
    .irq_o           ( core_irq_pulse ),
    .irq_tag_o       ( core_irq_tag ),
    .debug_tag_o     ( core_dbg_tag ),
    .perf_o          ( core_perf ),
    .dma_req_valid_o(dma_core_valid), .dma_req_ready_i(dma_core_ready),
    .dma_src_addr_o(dma_core_src), .dma_dst_addr_o(dma_core_dst),
    .dma_length_o(dma_core_length), .dma_reps_o(dma_core_reps),
    .dma_src_stride_o(dma_core_src_stride), .dma_dst_stride_o(dma_core_dst_stride),
    .dma_rsp_valid_i(dma_core_rsp_valid), .dma_rsp_ready_o(dma_core_rsp_ready),
    .dma_rsp_error_i(dma_core_rsp_error),
    .dma_bus_error_i((|idma_err) || (|fab_err) || addr_hi_err_q), .dma_error_o(core_dma_error)
  );

  // Sticky IRQ flag on pl_ps_irq0[0]; cleared while CTRL[8] is 1.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)               core_irq_q <= 1'b0;
    else if (reg_q[21][0])    core_irq_q <= 1'b0;
    else if (core_irq_pulse)  core_irq_q <= 1'b1;
  end
  assign irq = core_irq_q;

endmodule
