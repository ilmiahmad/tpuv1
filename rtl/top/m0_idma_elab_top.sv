// M0 elaboration top: concrete KV260 types around idma_subsystem, so the pinned PULP IP
// can be elaborated/linted in Verilator, Vivado and Xcelium. Not a board top.

`include "axi/typedef.svh"

module m0_idma_elab_top
  import tpu_m0_pkg::*;
(
  input  logic       clk_i,
  input  logic       rst_ni,
  input  axil_req_t  cfg_req_i,
  output axil_resp_t cfg_rsp_o,
  output axi_req_t   axi_req_o,
  input  axi_resp_t  axi_rsp_i,
  output logic [31:0] done_id_o,
  output logic        busy_o
);

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
    .clk_i, .rst_ni,
    .cfg_req_i, .cfg_rsp_o,
    .axi_req_o, .axi_rsp_i,
    .done_id_o, .busy_o, .err_o(), .err_clr_i(1'b0),
    .core_req_valid_i(1'b0), .core_req_ready_o(),
    .core_src_addr_i('0), .core_dst_addr_i('0), .core_length_i('0), .core_reps_i('0),
    .core_src_stride_i('0), .core_dst_stride_i('0),
    .core_rsp_valid_o(), .core_rsp_ready_i(1'b1), .core_rsp_error_o()
  );

endmodule
