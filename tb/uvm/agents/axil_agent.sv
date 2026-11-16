// AXI4-Lite master agent (iDMA config port) + RAL adapter.

class axil_item extends uvm_sequence_item;
  rand bit          write;
  rand bit [31:0]   addr;
  rand bit [31:0]   data;
  axi_pkg::resp_t   resp;
  `uvm_object_utils_begin(axil_item)
    `uvm_field_int(write, UVM_ALL_ON)
    `uvm_field_int(addr,  UVM_ALL_ON | UVM_HEX)
    `uvm_field_int(data,  UVM_ALL_ON | UVM_HEX)
  `uvm_object_utils_end
  function new(string name = "axil_item"); super.new(name); endfunction
endclass

class axil_driver extends uvm_driver #(axil_item);
  `uvm_component_utils(axil_driver)
  virtual tpu_uvm_if vif;
  localparam time TA = 2ns;
  function new(string name, uvm_component parent); super.new(name, parent); endfunction

  function void build_phase(uvm_phase phase);
    if (!uvm_config_db#(virtual tpu_uvm_if)::get(this, "", "vif", vif))
      `uvm_fatal("NOVIF", "axil_driver: no vif")
  endfunction

  task run_phase(uvm_phase phase);
    vif.cfg_req = '0;
    forever begin
      seq_item_port.get_next_item(req);
      if (req.write) do_write(req); else do_read(req);
      seq_item_port.item_done();
    end
  endtask

  task do_write(axil_item t);
    vif.cfg_req.aw.addr = t.addr; vif.cfg_req.aw_valid = 1'b1;
    vif.cfg_req.w.data  = t.data; vif.cfg_req.w.strb   = '1; vif.cfg_req.w_valid = 1'b1;
    vif.cfg_req.b_ready = 1'b1;
    fork
      begin do @(posedge vif.clk); while (!vif.cfg_rsp.aw_ready); #TA vif.cfg_req.aw_valid = 1'b0; end
      begin do @(posedge vif.clk); while (!vif.cfg_rsp.w_ready);  #TA vif.cfg_req.w_valid  = 1'b0; end
    join
    while (!vif.cfg_rsp.b_valid) @(posedge vif.clk);
    t.resp = vif.cfg_rsp.b.resp;
    @(posedge vif.clk); #TA vif.cfg_req.b_ready = 1'b0;
  endtask

  task do_read(axil_item t);
    vif.cfg_req.ar.addr = t.addr; vif.cfg_req.ar_valid = 1'b1; vif.cfg_req.r_ready = 1'b1;
    do @(posedge vif.clk); while (!vif.cfg_rsp.ar_ready);
    #TA vif.cfg_req.ar_valid = 1'b0;
    while (!vif.cfg_rsp.r_valid) @(posedge vif.clk);
    t.data = vif.cfg_rsp.r.data; t.resp = vif.cfg_rsp.r.resp;
    @(posedge vif.clk); #TA vif.cfg_req.r_ready = 1'b0;
  endtask
endclass

class axil_agent extends uvm_agent;
  `uvm_component_utils(axil_agent)
  uvm_sequencer #(axil_item) sqr;
  axil_driver                drv;
  function new(string name, uvm_component parent); super.new(name, parent); endfunction
  function void build_phase(uvm_phase phase);
    sqr = uvm_sequencer#(axil_item)::type_id::create("sqr", this);
    drv = axil_driver::type_id::create("drv", this);
  endfunction
  function void connect_phase(uvm_phase phase);
    drv.seq_item_port.connect(sqr.seq_item_export);
  endfunction
endclass

class axil_reg_adapter extends uvm_reg_adapter;
  `uvm_object_utils(axil_reg_adapter)
  function new(string name = "axil_reg_adapter"); super.new(name); endfunction
  virtual function uvm_sequence_item reg2bus(const ref uvm_reg_bus_op rw);
    axil_item t = axil_item::type_id::create("t");
    t.write = (rw.kind == UVM_WRITE);
    t.addr  = rw.addr[31:0];
    t.data  = rw.data[31:0];
    return t;
  endfunction
  virtual function void bus2reg(uvm_sequence_item bus_item, ref uvm_reg_bus_op rw);
    axil_item t;
    if (!$cast(t, bus_item)) `uvm_fatal("ADAPT", "not an axil_item")
    rw.kind   = t.write ? UVM_WRITE : UVM_READ;
    rw.addr   = t.addr;
    rw.data   = t.data;
    rw.status = (t.resp == axi_pkg::RESP_OKAY) ? UVM_IS_OK : UVM_NOT_OK;
  endfunction
endclass
