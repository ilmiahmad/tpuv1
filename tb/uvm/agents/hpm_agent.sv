// HPM0 data-window master agent (single 128-bit beats, as the CPU debug path).

class hpm_item extends uvm_sequence_item;
  rand bit                 write;
  rand bit [63:0]          addr;
  rand tpu_pkg::data_t     data;
  rand tpu_pkg::strb_t     strb;
  axi_pkg::resp_t          resp;
  `uvm_object_utils_begin(hpm_item)
    `uvm_field_int(write, UVM_ALL_ON)
    `uvm_field_int(addr,  UVM_ALL_ON | UVM_HEX)
    `uvm_field_int(data,  UVM_ALL_ON | UVM_HEX)
    `uvm_field_int(strb,  UVM_ALL_ON | UVM_HEX)
  `uvm_object_utils_end
  function new(string name = "hpm_item"); super.new(name); endfunction
endclass

class hpm_driver extends uvm_driver #(hpm_item);
  `uvm_component_utils(hpm_driver)
  virtual tpu_uvm_if vif;
  localparam time TA = 2ns;
  function new(string name, uvm_component parent); super.new(name, parent); endfunction
  function void build_phase(uvm_phase phase);
    if (!uvm_config_db#(virtual tpu_uvm_if)::get(this, "", "vif", vif)) `uvm_fatal("NOVIF", "hpm_driver")
  endfunction
  task run_phase(uvm_phase phase);
    vif.hpm_req = '0;
    forever begin
      seq_item_port.get_next_item(req);
      if (req.write) begin
        vif.hpm_req.aw = '0; vif.hpm_req.aw.addr = req.addr; vif.hpm_req.aw.size = 3'd4;
        vif.hpm_req.aw.burst = axi_pkg::BURST_INCR; vif.hpm_req.aw_valid = 1'b1;
        vif.hpm_req.w = '0; vif.hpm_req.w.data = req.data; vif.hpm_req.w.strb = req.strb;
        vif.hpm_req.w.last = 1'b1; vif.hpm_req.w_valid = 1'b1; vif.hpm_req.b_ready = 1'b1;
        fork
          begin do @(posedge vif.clk); while (!vif.hpm_rsp.aw_ready); #TA vif.hpm_req.aw_valid = 1'b0; end
          begin do @(posedge vif.clk); while (!vif.hpm_rsp.w_ready);  #TA vif.hpm_req.w_valid  = 1'b0; end
        join
        while (!vif.hpm_rsp.b_valid) @(posedge vif.clk);
        req.resp = vif.hpm_rsp.b.resp;
        @(posedge vif.clk); #TA vif.hpm_req.b_ready = 1'b0;
      end else begin
        vif.hpm_req.ar = '0; vif.hpm_req.ar.addr = req.addr; vif.hpm_req.ar.size = 3'd4;
        vif.hpm_req.ar.burst = axi_pkg::BURST_INCR; vif.hpm_req.ar_valid = 1'b1; vif.hpm_req.r_ready = 1'b1;
        do @(posedge vif.clk); while (!vif.hpm_rsp.ar_ready);
        #TA vif.hpm_req.ar_valid = 1'b0;
        while (!vif.hpm_rsp.r_valid) @(posedge vif.clk);
        req.data = vif.hpm_rsp.r.data; req.resp = vif.hpm_rsp.r.resp;
        @(posedge vif.clk); #TA vif.hpm_req.r_ready = 1'b0;
      end
      seq_item_port.item_done();
    end
  endtask
endclass

class hpm_agent extends uvm_agent;
  `uvm_component_utils(hpm_agent)
  uvm_sequencer #(hpm_item) sqr;
  hpm_driver                drv;
  function new(string name, uvm_component parent); super.new(name, parent); endfunction
  function void build_phase(uvm_phase phase);
    sqr = uvm_sequencer#(hpm_item)::type_id::create("sqr", this);
    drv = hpm_driver::type_id::create("drv", this);
  endfunction
  function void connect_phase(uvm_phase phase); drv.seq_item_port.connect(sqr.seq_item_export); endfunction
endclass
