// Concrete AXI / AXI4-Lite types for the KV260 data path (128-bit, 64-bit internal addr).
`include "axi/typedef.svh"

package tpu_m0_pkg;
  localparam int unsigned DataWidth  = 128;
  localparam int unsigned AddrWidth  = 64;
  localparam int unsigned AxiIdWidth = 6;
  localparam int unsigned UserWidth  = 1;

  typedef logic [AddrWidth-1:0]    addr_t;
  typedef logic [DataWidth-1:0]    data_t;
  typedef logic [DataWidth/8-1:0]  strb_t;
  typedef logic [AxiIdWidth-1:0]   id_t;
  typedef logic [UserWidth-1:0]    user_t;
  `AXI_TYPEDEF_ALL(axi, addr_t, id_t, data_t, strb_t, user_t)

  typedef logic [31:0] cfg_addr_t;
  typedef logic [31:0] cfg_data_t;
  typedef logic [3:0]  cfg_strb_t;
  `AXI_LITE_TYPEDEF_ALL(axil, cfg_addr_t, cfg_data_t, cfg_strb_t)
endpackage
