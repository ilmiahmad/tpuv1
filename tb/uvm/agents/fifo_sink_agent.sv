// FIFO consumer agent (Weight FIFO / Instruction FIFO native pop port): drives ready with a
// configurable probability and publishes every popped word.

class fifo_word extends uvm_sequence_item;
  tpu_pkg::data_t data;
  `uvm_object_utils_begin(fifo_word)
    `uvm_field_int(data, UVM_ALL_ON | UVM_HEX)
  `uvm_object_utils_end
  function new(string name = "fifo_word"); super.new(name); endfunction
endclass

class fifo_sink_agent extends uvm_agent;
  `uvm_component_utils(fifo_sink_agent)
  virtual tpu_uvm_if vif;
  bit                is_wf = 1;           // 1 = Weight FIFO, 0 = Instruction FIFO
  int unsigned       ready_pct = 70;      // 0 = never pop (backpressure tests)
  int unsigned       popped;
  uvm_analysis_port #(fifo_word) ap;
  localparam time TA = 2ns;

  function new(string name, uvm_component parent); super.new(name, parent); endfunction
  function void build_phase(uvm_phase phase);
    ap = new("ap", this);
    if (!uvm_config_db#(virtual tpu_uvm_if)::get(this, "", "vif", vif)) `uvm_fatal("NOVIF", "fifo_sink")
  endfunction

  task run_phase(uvm_phase phase);
    if (is_wf) vif.wf_ready = 0; else vif.if_ready = 0;
    fork
      forever begin   // ready driver
        @(posedge vif.clk); #TA;
        if (is_wf) vif.wf_ready = ($urandom_range(0, 99) < ready_pct);
        else       vif.if_ready = ($urandom_range(0, 99) < ready_pct);
      end
      forever begin   // monitor
        @(vif.mon_cb);
        if (is_wf ? (vif.mon_cb.wf_valid && vif.mon_cb.wf_ready) : (vif.mon_cb.if_valid && vif.mon_cb.if_ready)) begin
          fifo_word w = fifo_word::type_id::create("w");
          w.data = is_wf ? vif.mon_cb.wf_data : vif.mon_cb.if_data;
          popped++;
          ap.write(w);
        end
      end
    join
  endtask
endclass
