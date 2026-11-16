// TPU-KV260 - M0 board-gate top.
//
//   PS M_AXI_HPM0 (128b) -> axi_dw_converter 128->32 -> axi_to_axi_lite -> axi_lite_xbar
//        0xA000_0000 +4K : axi_lite_regs  (VERSION / SCRATCH / BUILD_ID / DONE_ID / STATUS)
//        0xA000_1000 +4K : iDMA reg64_2d cfg (via idma_subsystem's axi_lite_to_apb)
//        anything else   : DECERR (L5: every access gets a response)
//   idma_subsystem AXI4 master (128b, 64b addr) -> PS S_AXI_HP0 (128b, 49b addr) -> DDR
//
// Only PULP IP + wiring; one clock (pl_clk0, 100 MHz). The register window is the base-DT
// generic-uio "fabric" node (0xA000_0000, 64 KiB, IRQ SPI 121 on pl_ps_irq0[0]).

`include "axi/typedef.svh"
`include "axi/assign.svh"

`timescale 1ns/1ps

module m0_board_top #(
  parameter logic [31:0] BUILD_ID = 32'h0  // stamped by the build script (L7)
) ();

  import tpu_m0_pkg::*;

  localparam logic [31:0] Version = 32'h0000_0110;  // M1: iDMA error flags, v0.1.1

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

  // 128 -> 32 data width
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
    .slv_req_i  ( hpms_req ),
    .slv_resp_o ( hpms_rsp ),
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
  // Registers (PULP axi_lite_regs): 32 bytes
  //   0x00 VERSION (RO)  0x04 SCRATCH (RW)  0x08 BUILD_ID (RO)  0x0C IDMA_DONE_ID (RO, live)
  //   0x10 STATUS  (RO, live: [0] idma busy, [1] HP0 upper-address error (sticky),
  //                 [2] AXI read resp error, [3] AXI write resp error, [4] midend error;
  //                 [4:2] sticky until CTRL[0])
  //   0x14 CTRL    (RW: [0] err_clr, level - write 1 then 0)
  //   0x18..0x1C reserved (RO 0)
  // ------------------------------------------------------------------------------------------
  localparam int unsigned RegBytes = 32;
  localparam logic [RegBytes-1:0] RegRO = 32'hFF0F_FF0F;  // SCRATCH (4..7), CTRL (20..23) writable

  logic [31:0] idma_done_id;
  logic        idma_busy;
  logic        addr_hi_err_q;
  logic [2:0]  idma_err;
  logic [RegBytes-1:0][7:0] reg_q;

  logic [RegBytes-1:0][7:0] reg_d;
  logic [RegBytes-1:0]      reg_load;

  always_comb begin
    reg_d    = '0;
    reg_load = '0;
    reg_d[3:0]   = Version;      reg_load[3:0]   = '1;
    reg_d[11:8]  = BUILD_ID;     reg_load[11:8]  = '1;
    reg_d[15:12] = idma_done_id; reg_load[15:12] = '1;
    reg_d[19:16] = {27'd0, idma_err, addr_hi_err_q, idma_busy}; reg_load[19:16] = '1;
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
  axi_req_t  dma_req;
  axi_resp_t dma_rsp;

  idma_subsystem #(
    .DataWidth     ( DataWidth     ),
    .AddrWidth     ( AddrWidth     ),
    .AxiIdWidth    ( AxiIdWidth    ),
    .UserWidth     ( UserWidth     ),
    .axi_req_t     ( axi_req_t     ),
    .axi_rsp_t     ( axi_resp_t    ),
    .axi_ar_chan_t ( axi_ar_chan_t ),
    .axi_aw_chan_t ( axi_aw_chan_t ),
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
    .core_req_valid_i(1'b0), .core_req_ready_o(),
    .core_src_addr_i('0), .core_dst_addr_i('0), .core_length_i('0), .core_reps_i('0),
    .core_src_stride_i('0), .core_dst_stride_i('0),
    .core_rsp_valid_o(), .core_rsp_ready_i(1'b1), .core_rsp_error_o()
  );

  // HP0 is 49-bit: flag (sticky) any request with nonzero upper address bits.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) addr_hi_err_q <= 1'b0;
    else if ((dma_req.aw_valid && |dma_req.aw.addr[63:49]) ||
             (dma_req.ar_valid && |dma_req.ar.addr[63:49]))
      addr_hi_err_q <= 1'b1;
  end

  assign hp_awid    = dma_req.aw.id;    assign hp_awaddr  = dma_req.aw.addr[48:0];
  assign hp_awlen   = dma_req.aw.len;   assign hp_awsize  = dma_req.aw.size;
  assign hp_awburst = dma_req.aw.burst; assign hp_awlock  = dma_req.aw.lock;
  assign hp_awcache = dma_req.aw.cache; assign hp_awprot  = dma_req.aw.prot;
  assign hp_awqos   = dma_req.aw.qos;   assign hp_awuser  = 1'b0;
  assign hp_awvalid = dma_req.aw_valid;
  assign hp_wdata   = dma_req.w.data;   assign hp_wstrb   = dma_req.w.strb;
  assign hp_wlast   = dma_req.w.last;   assign hp_wvalid  = dma_req.w_valid;
  assign hp_bready  = dma_req.b_ready;
  assign hp_arid    = dma_req.ar.id;    assign hp_araddr  = dma_req.ar.addr[48:0];
  assign hp_arlen   = dma_req.ar.len;   assign hp_arsize  = dma_req.ar.size;
  assign hp_arburst = dma_req.ar.burst; assign hp_arlock  = dma_req.ar.lock;
  assign hp_arcache = dma_req.ar.cache; assign hp_arprot  = dma_req.ar.prot;
  assign hp_arqos   = dma_req.ar.qos;   assign hp_aruser  = 1'b0;
  assign hp_arvalid = dma_req.ar_valid;
  assign hp_rready  = dma_req.r_ready;

  always_comb begin
    dma_rsp          = '0;
    dma_rsp.aw_ready = hp_awready;
    dma_rsp.w_ready  = hp_wready;
    dma_rsp.b.id     = hp_bid;   dma_rsp.b.resp = hp_bresp; dma_rsp.b_valid = hp_bvalid;
    dma_rsp.ar_ready = hp_arready;
    dma_rsp.r.id     = hp_rid;   dma_rsp.r.data = hp_rdata; dma_rsp.r.resp  = hp_rresp;
    dma_rsp.r.last   = hp_rlast; dma_rsp.r_valid = hp_rvalid;
  end

  // M0: no interrupt source yet (irq_glue arrives with tpu_regs in M2)
  assign irq = 1'b0;

endmodule
