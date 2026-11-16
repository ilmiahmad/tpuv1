// Simulation PS model: 32-bit CPU accesses on the 128-bit HPM0 port.
`timescale 1ns/1ps
module kv260_ps_wrapper (
  output logic [39:0]M_AXI_HPM0_araddr,
  output logic [1:0]M_AXI_HPM0_arburst,
  output logic [3:0]M_AXI_HPM0_arcache,
  output logic [15:0]M_AXI_HPM0_arid,
  output logic [7:0]M_AXI_HPM0_arlen,
  output logic M_AXI_HPM0_arlock,
  output logic [2:0]M_AXI_HPM0_arprot,
  output logic [3:0]M_AXI_HPM0_arqos,
  input logic M_AXI_HPM0_arready,
  output logic [2:0]M_AXI_HPM0_arsize,
  output logic [15:0]M_AXI_HPM0_aruser,
  output logic M_AXI_HPM0_arvalid,
  output logic [39:0]M_AXI_HPM0_awaddr,
  output logic [1:0]M_AXI_HPM0_awburst,
  output logic [3:0]M_AXI_HPM0_awcache,
  output logic [15:0]M_AXI_HPM0_awid,
  output logic [7:0]M_AXI_HPM0_awlen,
  output logic M_AXI_HPM0_awlock,
  output logic [2:0]M_AXI_HPM0_awprot,
  output logic [3:0]M_AXI_HPM0_awqos,
  input logic M_AXI_HPM0_awready,
  output logic [2:0]M_AXI_HPM0_awsize,
  output logic [15:0]M_AXI_HPM0_awuser,
  output logic M_AXI_HPM0_awvalid,
  input logic [15:0]M_AXI_HPM0_bid,
  output logic M_AXI_HPM0_bready,
  input logic [1:0]M_AXI_HPM0_bresp,
  input logic M_AXI_HPM0_bvalid,
  input logic [127:0]M_AXI_HPM0_rdata,
  input logic [15:0]M_AXI_HPM0_rid,
  input logic M_AXI_HPM0_rlast,
  output logic M_AXI_HPM0_rready,
  input logic [1:0]M_AXI_HPM0_rresp,
  input logic M_AXI_HPM0_rvalid,
  output logic [127:0]M_AXI_HPM0_wdata,
  output logic M_AXI_HPM0_wlast,
  input logic M_AXI_HPM0_wready,
  output logic [15:0]M_AXI_HPM0_wstrb,
  output logic M_AXI_HPM0_wvalid,
  input logic [48:0]S_AXI_HP0_araddr,
  input logic [1:0]S_AXI_HP0_arburst,
  input logic [3:0]S_AXI_HP0_arcache,
  input logic [5:0]S_AXI_HP0_arid,
  input logic [7:0]S_AXI_HP0_arlen,
  input logic S_AXI_HP0_arlock,
  input logic [2:0]S_AXI_HP0_arprot,
  input logic [3:0]S_AXI_HP0_arqos,
  output logic S_AXI_HP0_arready,
  input logic [2:0]S_AXI_HP0_arsize,
  input logic S_AXI_HP0_aruser,
  input logic S_AXI_HP0_arvalid,
  input logic [48:0]S_AXI_HP0_awaddr,
  input logic [1:0]S_AXI_HP0_awburst,
  input logic [3:0]S_AXI_HP0_awcache,
  input logic [5:0]S_AXI_HP0_awid,
  input logic [7:0]S_AXI_HP0_awlen,
  input logic S_AXI_HP0_awlock,
  input logic [2:0]S_AXI_HP0_awprot,
  input logic [3:0]S_AXI_HP0_awqos,
  output logic S_AXI_HP0_awready,
  input logic [2:0]S_AXI_HP0_awsize,
  input logic S_AXI_HP0_awuser,
  input logic S_AXI_HP0_awvalid,
  output logic [5:0]S_AXI_HP0_bid,
  input logic S_AXI_HP0_bready,
  output logic [1:0]S_AXI_HP0_bresp,
  output logic S_AXI_HP0_bvalid,
  output logic [127:0]S_AXI_HP0_rdata,
  output logic [5:0]S_AXI_HP0_rid,
  output logic S_AXI_HP0_rlast,
  input logic S_AXI_HP0_rready,
  output logic [1:0]S_AXI_HP0_rresp,
  output logic S_AXI_HP0_rvalid,
  input logic [127:0]S_AXI_HP0_wdata,
  input logic S_AXI_HP0_wlast,
  output logic S_AXI_HP0_wready,
  input logic [15:0]S_AXI_HP0_wstrb,
  input logic S_AXI_HP0_wvalid,
  output logic pl_clk0,
  input logic [0:0]pl_ps_irq0,
  output logic pl_resetn0
);
  initial begin
    M_AXI_HPM0_araddr = '0;
    M_AXI_HPM0_arburst = '0;
    M_AXI_HPM0_arcache = '0;
    M_AXI_HPM0_arid = '0;
    M_AXI_HPM0_arlen = '0;
    M_AXI_HPM0_arlock = '0;
    M_AXI_HPM0_arprot = '0;
    M_AXI_HPM0_arqos = '0;
    M_AXI_HPM0_arsize = '0;
    M_AXI_HPM0_aruser = '0;
    M_AXI_HPM0_arvalid = '0;
    M_AXI_HPM0_awaddr = '0;
    M_AXI_HPM0_awburst = '0;
    M_AXI_HPM0_awcache = '0;
    M_AXI_HPM0_awid = '0;
    M_AXI_HPM0_awlen = '0;
    M_AXI_HPM0_awlock = '0;
    M_AXI_HPM0_awprot = '0;
    M_AXI_HPM0_awqos = '0;
    M_AXI_HPM0_awsize = '0;
    M_AXI_HPM0_awuser = '0;
    M_AXI_HPM0_awvalid = '0;
    M_AXI_HPM0_bready = '0;
    M_AXI_HPM0_rready = '0;
    M_AXI_HPM0_wdata = '0;
    M_AXI_HPM0_wlast = '0;
    M_AXI_HPM0_wstrb = '0;
    M_AXI_HPM0_wvalid = '0;
`ifndef TPU_PHASE_B_SIM
    S_AXI_HP0_arready = '0;
    S_AXI_HP0_awready = '0;
    S_AXI_HP0_bid = '0;
    S_AXI_HP0_bresp = '0;
    S_AXI_HP0_bvalid = '0;
    S_AXI_HP0_rdata = '0;
    S_AXI_HP0_rid = '0;
    S_AXI_HP0_rlast = '0;
    S_AXI_HP0_rresp = '0;
    S_AXI_HP0_rvalid = '0;
    S_AXI_HP0_wready = '0;
`endif
    pl_clk0 = '0;
    pl_resetn0 = '0;
  end
  always #5 pl_clk0 = ~pl_clk0;
  task automatic write32(input logic [31:0] addr, data,
                         input logic [3:0] strobe, input logic [1:0] expected);
    logic [15:0] saved_id;
    logic [1:0] saved_resp;
    @(negedge pl_clk0);
    M_AXI_HPM0_awaddr = {8'b0, addr};
    M_AXI_HPM0_awid = 16'h9234;
    M_AXI_HPM0_awsize = 2;
    M_AXI_HPM0_awburst = 1;
    M_AXI_HPM0_wdata = 128'(data) << (8*addr[3:0]);
    M_AXI_HPM0_wstrb = 16'(strobe) << addr[3:0];
    M_AXI_HPM0_wlast = 1;
    fork
      begin
        M_AXI_HPM0_awvalid = 1;
        do @(posedge pl_clk0); while (!M_AXI_HPM0_awready);
        @(negedge pl_clk0); M_AXI_HPM0_awvalid = 0;
      end
      begin
        repeat (3) @(negedge pl_clk0);
        M_AXI_HPM0_wvalid = 1;
        do @(posedge pl_clk0); while (!M_AXI_HPM0_wready);
        @(negedge pl_clk0); M_AXI_HPM0_wvalid = 0;
      end
    join
    do @(posedge pl_clk0); while (!M_AXI_HPM0_bvalid);
    saved_id = M_AXI_HPM0_bid; saved_resp = M_AXI_HPM0_bresp;
    repeat (4) begin
      @(posedge pl_clk0);
      assert (M_AXI_HPM0_bvalid && M_AXI_HPM0_bid == saved_id && M_AXI_HPM0_bresp == saved_resp)
        else $fatal(1, "B response changed under backpressure");
    end
    assert (saved_id == 16'h9234 && saved_resp == expected)
      else $fatal(1, "WRITE addr=%h response=%h expected=%h id=%h", addr, saved_resp, expected, saved_id);
    @(negedge pl_clk0); M_AXI_HPM0_bready = 1;
    @(posedge pl_clk0);
    @(negedge pl_clk0); M_AXI_HPM0_bready = 0;
  endtask

  task automatic read32(input logic [31:0] addr, output logic [31:0] data);
    logic [127:0] held;
    @(negedge pl_clk0);
    M_AXI_HPM0_araddr = {8'b0, addr}; M_AXI_HPM0_arid = 16'h4321;
    M_AXI_HPM0_arsize = 2; M_AXI_HPM0_arburst = 1; M_AXI_HPM0_arvalid = 1;
    do @(posedge pl_clk0); while (!M_AXI_HPM0_arready);
    @(negedge pl_clk0); M_AXI_HPM0_arvalid = 0;
    do @(posedge pl_clk0); while (!M_AXI_HPM0_rvalid);
    held = M_AXI_HPM0_rdata;
    repeat (4) begin
      @(posedge pl_clk0);
      assert (M_AXI_HPM0_rvalid && M_AXI_HPM0_rdata == held &&
              M_AXI_HPM0_rid == 16'h4321 && M_AXI_HPM0_rresp == 0 && M_AXI_HPM0_rlast)
        else $fatal(1, "R response mismatch/backpressure violation addr=%h", addr);
    end
    data = held[8*addr[3:0] +: 32];
    @(negedge pl_clk0); M_AXI_HPM0_rready = 1;
    @(posedge pl_clk0);
    @(negedge pl_clk0); M_AXI_HPM0_rready = 0;
  endtask

`ifdef TPU_PHASE_B_SIM
  `include "kv260_ps_phase_b.svh"
`else
  logic [31:0] got;
  initial begin
    repeat (8) @(negedge pl_clk0);
    pl_resetn0 = 1;
    repeat (8) @(negedge pl_clk0);
    for (int run = 0; run < 2; run++) begin
      read32(32'hA0000000, got);
      assert (got == 32'h300) else $fatal(1, "VERSION %h", got);
      read32(32'hA0000008, got);
      assert (got == 32'h12345678) else $fatal(1, "BUILD_ID %h", got);
      write32(32'hA0000004, 32'h5aa51234, 4'hf, 0);
      read32(32'hA0000004, got);
      assert (got == 32'h5aa51234) else $fatal(1, "SCRATCH %h", got);
      for (int lane = 0; lane < 4; lane++)
        write32(32'hA0000004, 32'h87654321, 4'(1 << lane), 0);
      read32(32'hA0000004, got);
      assert (got == 32'h87654321) else $fatal(1, "SCRATCH strobes %h", got);
      write32(32'hA0000014, 32'h300, 4'hf, 0);
      read32(32'hA0000014, got);
      assert (got == 32'h300) else $fatal(1, "CTRL bits8/9 %h", got);
      read32(32'hA0000024, got);
      assert (got == 0) else $fatal(1, "PERF clear did not reach core %h", got);
      write32(32'hA0000014, 0, 4'hf, 0);
      read32(32'hA0000014, got);
      assert (got == 0) else $fatal(1, "CTRL clear %h", got);
      for (int off = 0; off < 72; off += 4)
        if (off != 4 && off != 20)
          write32(32'hA0000000 + 32'(off), '1, 4'hf, 2);
      read32(32'hA0000000, got);
      assert (got == 32'h300) else $fatal(1, "VERSION corrupted %h", got);
      read32(32'hA0000008, got);
      assert (got == 32'h12345678) else $fatal(1, "BUILD_ID corrupted %h", got);
    end
    $display("M4_BOARD_REGS_PASS");
    $finish;
  end
  initial begin
    #100000;
    $fatal(1, "Board register path timed out");
  end
`endif
endmodule

`ifdef TPU_PHASE_B_SIM
module tb_m5_full_system;
  kv260_top #(.BUILD_ID(32'h12345678)) dut ();
endmodule
`else
module tb_m4_board_regs;
  kv260_top #(.BUILD_ID(32'h12345678), .EnableDma(1'b0)) dut ();
endmodule
`endif
