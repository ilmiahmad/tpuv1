// TPU-KV260 - shared parameters, address map and data-fabric AXI types (frozen in M2;
// docs/interfaces.md is the human-readable copy - change both together).
`include "axi/typedef.svh"

package tpu_pkg;

  // ------------------------------------------------------------------------------------------
  // Data fabric: 128-bit, 64-bit addresses (iDMA reg64), IDs 5 b on xbar slave ports,
  // 6 b on xbar master ports (= HP0 slave ID width; the xbar prepends 1 bit for 2 slave ports).
  // ------------------------------------------------------------------------------------------
  localparam int unsigned DataWidth   = 128;
  localparam int unsigned StrbWidth   = DataWidth / 8;
  localparam int unsigned AddrWidth   = 64;
  localparam int unsigned SlvIdWidth  = 5;
  localparam int unsigned MstIdWidth  = 6;
  localparam int unsigned UserWidth   = 1;

  typedef logic [AddrWidth-1:0]  addr_t;
  typedef logic [DataWidth-1:0]  data_t;
  typedef logic [StrbWidth-1:0]  strb_t;
  typedef logic [SlvIdWidth-1:0] sid_t;
  typedef logic [MstIdWidth-1:0] mid_t;
  typedef logic [UserWidth-1:0]  user_t;

  // xbar slave-port side (masters: iDMA, HPM0 data window)
  `AXI_TYPEDEF_ALL(fs, addr_t, sid_t, data_t, strb_t, user_t)
  // xbar master-port side (slaves: HP0, UB, ACC, WFIFO, IFIFO)
  `AXI_TYPEDEF_ALL(fm, addr_t, mid_t, data_t, strb_t, user_t)

  // ------------------------------------------------------------------------------------------
  // Address map - identical in the iDMA and HPM0 views
  // ------------------------------------------------------------------------------------------
  localparam logic [63:0] REGS_BASE   = 64'h0000_0000_A000_0000;  // tpu_regs (AXI-Lite)  4 KiB
  localparam logic [63:0] IDMA_BASE   = 64'h0000_0000_A000_1000;  // iDMA cfg (AXI-Lite)  4 KiB
  localparam logic [63:0] CFG_END     = 64'h0000_0000_A010_0000;  // end of the AXI-Lite region
  localparam logic [63:0] UB_BASE     = 64'h0000_0000_A010_0000;  // Unified Buffer       1 MiB
  localparam logic [63:0] UB_SIZE     = 64'h0000_0000_0010_0000;
  localparam logic [63:0] ACC_BASE    = 64'h0000_0000_A020_0000;  // Accumulators       128 KiB
  localparam logic [63:0] ACC_SIZE    = 64'h0000_0000_0002_0000;
  localparam logic [63:0] WFIFO_BASE  = 64'h0000_0000_A030_0000;  // Weight FIFO window   1 MiB
  localparam logic [63:0] WFIFO_SIZE  = 64'h0000_0000_0010_0000;
  localparam logic [63:0] IFIFO_BASE  = 64'h0000_0000_A040_0000;  // Instr FIFO window   64 KiB
  localparam logic [63:0] IFIFO_SIZE  = 64'h0000_0000_0001_0000;
  localparam logic [63:0] PL_END      = 64'h0000_0000_A080_0000;  // HPM0 BD segment: 8 MiB
  localparam logic [63:0] DDR_LO_BASE = 64'h0000_0000_0000_0000;  // KV260 4 GiB DDR
  localparam logic [63:0] DDR_LO_END  = 64'h0000_0000_8000_0000;
  localparam logic [63:0] DDR_HI_BASE = 64'h0000_0008_0000_0000;
  localparam logic [63:0] DDR_HI_END  = 64'h0000_0008_8000_0000;

  // xbar master-port indices
  localparam int unsigned XHP0 = 0, XUB = 1, XACC = 2, XWFIFO = 3, XIFIFO = 4, XNUM = 5;

  // ------------------------------------------------------------------------------------------
  // Memory geometry
  // ------------------------------------------------------------------------------------------
  localparam int unsigned ArrayDim    = 16;                    // 16x16 systolic array
  localparam int unsigned UbRows      = 65536;                 // 1 MiB of 16 x INT8 rows
  localparam int unsigned AccRows     = 2048;                  // 2048 x 16 x INT32
  localparam int unsigned AccBanks    = 4;                     // 4 x 128 b = one 512 b row
  localparam int unsigned WfifoWords  = 1024;                  // 64 tiles x 16 rows x 128 b
  localparam int unsigned IfifoWords  = 2048;                  // M5: preload whole Phase B MLP programs
  // UB read latency (native port and AXI side): input register + URAM (2, XPM)  - M2 timing fix
  localparam int unsigned UbLatency   = 3;

  typedef logic [$clog2(UbRows)-1:0]  ub_addr_t;
  typedef logic [$clog2(AccRows)-1:0] acc_addr_t;
  typedef logic [AccBanks*DataWidth-1:0] acc_row_t;            // 16 x INT32

endpackage
