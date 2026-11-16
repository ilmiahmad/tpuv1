// UVM top: idma_subsystem + tpu_fabric, HP0 through a random-stall delayer into axi_sim_mem.
`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_uvm_top;
  import uvm_pkg::*;
  import tpu_pkg::*;
  import tpu_uvm_pkg::*;

  logic clk = 1'b0;
  always #5ns clk = ~clk;

  tpu_uvm_if vif (clk);

  fs_req_t  dma_req;
  fs_resp_t dma_rsp;
  fm_req_t  hp0_req, mem_req;
  fm_resp_t hp0_rsp, mem_rsp;

  initial begin
    vif.rst_n = 1'b0;
    {vif.ub_own, vif.acc_own, vif.wf_dbg, vif.if_dbg, vif.wf_flush, vif.if_flush, vif.err_clr} = '0;
    repeat (10) @(posedge clk);
    #2ns vif.rst_n = 1'b1;
  end

  idma_subsystem #(
    .DataWidth(DataWidth), .AddrWidth(AddrWidth), .AxiIdWidth(SlvIdWidth), .UserWidth(UserWidth),
    .axi_req_t(fs_req_t), .axi_rsp_t(fs_resp_t),
    .axi_ar_chan_t(fs_ar_chan_t), .axi_aw_chan_t(fs_aw_chan_t),
    .axil_req_t(tpu_m0_pkg::axil_req_t), .axil_rsp_t(tpu_m0_pkg::axil_resp_t)
  ) i_idma (
    .clk_i(clk), .rst_ni(vif.rst_n), .cfg_req_i(vif.cfg_req), .cfg_rsp_o(vif.cfg_rsp),
    .axi_req_o(dma_req), .axi_rsp_i(dma_rsp), .done_id_o(vif.done_id), .busy_o(vif.dma_busy),
    .err_o(vif.dma_err), .err_clr_i(vif.err_clr),
    .core_req_valid_i(1'b0), .core_req_ready_o(),
    .core_src_addr_i('0), .core_dst_addr_i('0), .core_length_i('0), .core_reps_i('0),
    .core_src_stride_i('0), .core_dst_stride_i('0),
    .core_rsp_valid_o(), .core_rsp_ready_i(1'b1), .core_rsp_error_o()
  );

  tpu_fabric i_fab (
    .clk_i(clk), .rst_ni(vif.rst_n),
    .dma_req_i(dma_req), .dma_rsp_o(dma_rsp), .hpm_req_i(vif.hpm_req), .hpm_rsp_o(vif.hpm_rsp),
    .hp0_req_o(hp0_req), .hp0_rsp_i(hp0_rsp),
    .ub_rd_en_i(vif.ub_rd_en), .ub_rd_addr_i(vif.ub_rd_addr), .ub_rd_data_o(vif.ub_rd_data),
    .ub_wr_en_i(vif.ub_wr_en), .ub_wr_addr_i(vif.ub_wr_addr), .ub_wr_data_i(vif.ub_wr_data), .ub_wr_be_i(vif.ub_wr_be),
    .acc_rd_en_i(vif.acc_rd_en), .acc_rd_addr_i(vif.acc_rd_addr), .acc_rd_data_o(vif.acc_rd_data),
    .acc_wr_en_i(vif.acc_wr_en), .acc_wr_addr_i(vif.acc_wr_addr), .acc_wr_data_i(vif.acc_wr_data), .acc_wr_be_i(vif.acc_wr_be),
    .wf_valid_o(vif.wf_valid), .wf_ready_i(vif.wf_ready), .wf_data_o(vif.wf_data),
    .if_valid_o(vif.if_valid), .if_ready_i(vif.if_ready), .if_data_o(vif.if_data),
    .ub_owner_tpu_i(vif.ub_own), .acc_owner_tpu_i(vif.acc_own), .wf_dbg_pop_i(vif.wf_dbg), .if_dbg_pop_i(vif.if_dbg),
    .wf_flush_i(vif.wf_flush), .if_flush_i(vif.if_flush), .err_clr_i(vif.err_clr),
    .err_o(vif.fab_err), .wf_level_o(vif.wf_level), .if_level_o(vif.if_level)
  );

  axi_delayer #(
    .aw_chan_t(fm_aw_chan_t), .w_chan_t(fm_w_chan_t), .b_chan_t(fm_b_chan_t),
    .ar_chan_t(fm_ar_chan_t), .r_chan_t(fm_r_chan_t), .axi_req_t(fm_req_t), .axi_resp_t(fm_resp_t),
    .StallRandomInput(1'b1), .StallRandomOutput(1'b1), .FixedDelayInput(0), .FixedDelayOutput(0)
  ) i_delay (
    .clk_i(clk), .rst_ni(vif.rst_n), .slv_req_i(hp0_req), .slv_resp_o(hp0_rsp),
    .mst_req_o(mem_req), .mst_resp_i(mem_rsp)
  );

  axi_sim_mem #(
    .AddrWidth(AddrWidth), .DataWidth(DataWidth), .IdWidth(MstIdWidth), .UserWidth(UserWidth),
    .axi_req_t(fm_req_t), .axi_rsp_t(fm_resp_t),
    .WarnUninitialized(1'b0), .ClearErrOnAccess(1'b1), .ApplDelay(2ns), .AcqDelay(8ns)
  ) i_ddr (
    .clk_i(clk), .rst_ni(vif.rst_n), .axi_req_i(mem_req), .axi_rsp_o(mem_rsp),
    .mon_r_last_o(), .mon_r_beat_count_o(), .mon_r_user_o(), .mon_r_id_o(),
    .mon_r_data_o(), .mon_r_addr_o(), .mon_r_valid_o(),
    .mon_w_last_o(), .mon_w_beat_count_o(), .mon_w_user_o(), .mon_w_id_o(),
    .mon_w_data_o(), .mon_w_addr_o(), .mon_w_valid_o()
  );

  // HP0 rules (L2) and fabric safety, checked in every UVM test
  a_l2: assert property (@(posedge clk) disable iff (!vif.rst_n)
      (hp0_req.ar_valid && hp0_req.ar.len != 0) |-> hp0_req.ar.size == 3'd4);

  // DDR backdoor for the environment
  class ddr_bd extends ddr_backdoor;
    function new(string name = "ddr_bd"); super.new(name); endfunction
    virtual function void write(longint unsigned a, byte unsigned d); i_ddr.mem[a] = d; endfunction
    virtual function int read(longint unsigned a); return i_ddr.mem.exists(a) ? int'(i_ddr.mem[a]) : -1; endfunction
    virtual function void set_rerr(longint unsigned a, axi_pkg::resp_t r); i_ddr.rerr[a] = r; endfunction
  endclass

  initial begin
    ddr_bd bd = new();
    uvm_config_db#(virtual tpu_uvm_if)::set(null, "*", "vif", vif);
    uvm_config_db#(ddr_backdoor)::set(null, "*", "ddr", bd);
    run_test();
  end
endmodule
