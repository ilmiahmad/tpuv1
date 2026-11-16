// Signal bundle between the UVM environment and the DUT (idma_subsystem + tpu_fabric).
interface tpu_uvm_if (input logic clk);
  import tpu_pkg::*;

  logic rst_n;
  // AXI4-Lite config (iDMA register frontend)
  tpu_m0_pkg::axil_req_t  cfg_req;
  tpu_m0_pkg::axil_resp_t cfg_rsp;
  // HPM0 data window (CPU debug)
  fs_req_t  hpm_req;
  fs_resp_t hpm_rsp;
  // native memory ports (TPU side)
  logic ub_rd_en, ub_wr_en;   logic [StrbWidth-1:0] ub_wr_be;
  ub_addr_t ub_rd_addr, ub_wr_addr;  data_t ub_wr_data, ub_rd_data;
  logic acc_rd_en, acc_wr_en; logic [AccBanks-1:0] acc_wr_be;
  acc_addr_t acc_rd_addr, acc_wr_addr; acc_row_t acc_wr_data, acc_rd_data;
  // FIFO consumers
  logic  wf_valid, wf_ready, if_valid, if_ready;
  data_t wf_data, if_data;
  // control / status
  logic        ub_own, acc_own, wf_dbg, if_dbg, wf_flush, if_flush, err_clr;
  logic [3:0]  fab_err;
  logic [15:0] wf_level, if_level;
  logic [31:0] done_id;
  logic        dma_busy;
  logic [2:0]  dma_err;

  // monitors sample with #1step (pre-edge values = the values of the handshake)
  clocking mon_cb @(posedge clk);
    default input #1step;
    input wf_valid, wf_ready, wf_data, if_valid, if_ready, if_data, wf_level, if_level;
  endclocking
endinterface
