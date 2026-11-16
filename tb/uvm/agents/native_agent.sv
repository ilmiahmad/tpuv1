// Native (TPU-side) memory port agent: UB (128 b, latency tpu_pkg::UbLatency) and ACC (512 b row, latency 1).

typedef enum { NAT_UB, NAT_ACC } nat_mem_e;

class nat_item extends uvm_sequence_item;
  rand nat_mem_e        mem;
  rand bit              write;
  rand int unsigned     row;
  rand tpu_pkg::acc_row_t data;   // UB uses the low 128 bits
  rand bit [15:0]       be;       // UB byte enables / ACC bank enables [3:0]
  `uvm_object_utils_begin(nat_item)
    `uvm_field_enum(nat_mem_e, mem, UVM_ALL_ON)
    `uvm_field_int(write, UVM_ALL_ON)
    `uvm_field_int(row,   UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "nat_item"); super.new(name); endfunction
endclass

class nat_driver extends uvm_driver #(nat_item);
  `uvm_component_utils(nat_driver)
  virtual tpu_uvm_if vif;
  localparam time TA = 2ns;
  function new(string name, uvm_component parent); super.new(name, parent); endfunction
  function void build_phase(uvm_phase phase);
    if (!uvm_config_db#(virtual tpu_uvm_if)::get(this, "", "vif", vif)) `uvm_fatal("NOVIF", "nat_driver")
  endfunction
  task run_phase(uvm_phase phase);
    vif.ub_rd_en = 0; vif.ub_rd_addr = '0; vif.ub_wr_en = 0; vif.ub_wr_be = '0; vif.ub_wr_addr = '0; vif.ub_wr_data = '0;
    vif.acc_rd_en = 0; vif.acc_rd_addr = '0; vif.acc_wr_en = 0; vif.acc_wr_be = '0; vif.acc_wr_addr = '0; vif.acc_wr_data = '0;
    forever begin
      seq_item_port.get_next_item(req);
      @(posedge vif.clk); #TA;
      if (req.mem == NAT_UB) begin
        if (req.write) begin
          vif.ub_wr_en = 1; vif.ub_wr_addr = req.row; vif.ub_wr_be = req.be; vif.ub_wr_data = req.data[127:0];
          @(posedge vif.clk); #TA vif.ub_wr_en = 0; vif.ub_wr_be = '0;
        end else begin
          vif.ub_rd_en = 1; vif.ub_rd_addr = req.row;
          @(posedge vif.clk); #TA vif.ub_rd_en = 0;
          repeat (tpu_pkg::UbLatency - 1) @(posedge vif.clk);
          #1 req.data = '0; req.data[127:0] = vif.ub_rd_data;
        end
      end else begin
        if (req.write) begin
          vif.acc_wr_en = 1; vif.acc_wr_addr = req.row; vif.acc_wr_be = req.be[3:0]; vif.acc_wr_data = req.data;
          @(posedge vif.clk); #TA vif.acc_wr_en = 0; vif.acc_wr_be = '0;
        end else begin
          vif.acc_rd_en = 1; vif.acc_rd_addr = req.row;
          @(posedge vif.clk); #1 req.data = vif.acc_rd_data;
          #(TA-1) vif.acc_rd_en = 0;
        end
      end
      seq_item_port.item_done();
    end
  endtask
endclass

class nat_agent extends uvm_agent;
  `uvm_component_utils(nat_agent)
  uvm_sequencer #(nat_item) sqr;
  nat_driver                drv;
  function new(string name, uvm_component parent); super.new(name, parent); endfunction
  function void build_phase(uvm_phase phase);
    sqr = uvm_sequencer#(nat_item)::type_id::create("sqr", this);
    drv = nat_driver::type_id::create("drv", this);
  endfunction
  function void connect_phase(uvm_phase phase); drv.seq_item_port.connect(sqr.seq_item_export); endfunction
endclass
