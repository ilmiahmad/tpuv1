// Environment + virtual sequencer.

class tpu_vsqr extends uvm_sequencer;
  `uvm_component_utils(tpu_vsqr)
  uvm_sequencer #(axil_item) axil_sqr;
  uvm_sequencer #(hpm_item)  hpm_sqr;
  uvm_sequencer #(nat_item)  nat_sqr;
  idma_reg_block             regs;
  tpu_ref_model              model;
  tpu_scoreboard             sb;
  tpu_coverage               cov;
  ddr_backdoor               ddr;
  virtual tpu_uvm_if         vif;
  fifo_sink_agent            wf_sink;
  function new(string name, uvm_component parent); super.new(name, parent); endfunction
endclass

class tpu_env extends uvm_env;
  `uvm_component_utils(tpu_env)
  axil_agent       axil;
  hpm_agent        hpm;
  nat_agent        nat;
  fifo_sink_agent  wf_sink, if_sink;
  tpu_scoreboard   sb;
  tpu_coverage     cov;
  tpu_vsqr         vsqr;
  idma_reg_block   regs;
  axil_reg_adapter adapter;
  tpu_ref_model    model;

  function new(string name, uvm_component parent); super.new(name, parent); endfunction

  function void build_phase(uvm_phase phase);
    axil    = axil_agent::type_id::create("axil", this);
    hpm     = hpm_agent::type_id::create("hpm", this);
    nat     = nat_agent::type_id::create("nat", this);
    wf_sink = fifo_sink_agent::type_id::create("wf_sink", this);
    if_sink = fifo_sink_agent::type_id::create("if_sink", this);
    if_sink.is_wf = 0; if_sink.ready_pct = 0;   // IFIFO consumer is the decoder (M4); idle here
    sb      = tpu_scoreboard::type_id::create("sb", this);
    cov     = tpu_coverage::type_id::create("cov", this);
    vsqr    = tpu_vsqr::type_id::create("vsqr", this);
    model   = tpu_ref_model::type_id::create("model");
    regs    = idma_reg_block::type_id::create("regs");
    regs.build();
    regs.default_map.set_auto_predict(1);
    adapter = axil_reg_adapter::type_id::create("adapter");
    if (!uvm_config_db#(virtual tpu_uvm_if)::get(this, "", "vif", vsqr.vif)) `uvm_fatal("NOVIF", "env")
    if (!uvm_config_db#(ddr_backdoor)::get(this, "", "ddr", vsqr.ddr)) `uvm_fatal("NODDR", "env")
  endfunction

  function void connect_phase(uvm_phase phase);
    regs.default_map.set_sequencer(axil.sqr, adapter);
    wf_sink.ap.connect(sb.wf_imp);
    sb.model = model; sb.ddr = vsqr.ddr;
    vsqr.axil_sqr = axil.sqr; vsqr.hpm_sqr = hpm.sqr; vsqr.nat_sqr = nat.sqr;
    vsqr.regs = regs; vsqr.model = model; vsqr.sb = sb; vsqr.cov = cov; vsqr.wf_sink = wf_sink;
  endfunction
endclass
