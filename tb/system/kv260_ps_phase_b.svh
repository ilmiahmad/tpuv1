// Included inside the PS simulation wrapper. Exercise the real kv260_top through
// CPU-width HPM accesses; HP0 connects to the PULP model with random AXI stalls.
  import tpu_m0_pkg::*;
  axi_req_t hp_req, mem_req;
  axi_resp_t hp_rsp, mem_rsp;
  always_comb begin
    hp_req = '0;
    hp_req.ar.addr = {15'd0,S_AXI_HP0_araddr}; hp_req.ar.id=S_AXI_HP0_arid;
    hp_req.ar.len=S_AXI_HP0_arlen; hp_req.ar.size=S_AXI_HP0_arsize;
    hp_req.ar.burst=S_AXI_HP0_arburst; hp_req.ar.cache=S_AXI_HP0_arcache;
    hp_req.ar.lock=S_AXI_HP0_arlock; hp_req.ar.prot=S_AXI_HP0_arprot;
    hp_req.ar.qos=S_AXI_HP0_arqos; hp_req.ar_valid=S_AXI_HP0_arvalid;
    hp_req.aw.addr={15'd0,S_AXI_HP0_awaddr}; hp_req.aw.id=S_AXI_HP0_awid;
    hp_req.aw.len=S_AXI_HP0_awlen; hp_req.aw.size=S_AXI_HP0_awsize;
    hp_req.aw.burst=S_AXI_HP0_awburst; hp_req.aw.cache=S_AXI_HP0_awcache;
    hp_req.aw.lock=S_AXI_HP0_awlock; hp_req.aw.prot=S_AXI_HP0_awprot;
    hp_req.aw.qos=S_AXI_HP0_awqos; hp_req.aw_valid=S_AXI_HP0_awvalid;
    hp_req.w.data=S_AXI_HP0_wdata; hp_req.w.strb=S_AXI_HP0_wstrb;
    hp_req.w.last=S_AXI_HP0_wlast; hp_req.w_valid=S_AXI_HP0_wvalid;
    hp_req.r_ready=S_AXI_HP0_rready; hp_req.b_ready=S_AXI_HP0_bready;
    S_AXI_HP0_arready=hp_rsp.ar_ready; S_AXI_HP0_awready=hp_rsp.aw_ready;
    S_AXI_HP0_wready=hp_rsp.w_ready; S_AXI_HP0_bvalid=hp_rsp.b_valid;
    S_AXI_HP0_bresp=hp_rsp.b.resp; S_AXI_HP0_bid=hp_rsp.b.id;
    S_AXI_HP0_rvalid=hp_rsp.r_valid; S_AXI_HP0_rdata=hp_rsp.r.data;
    S_AXI_HP0_rresp=hp_rsp.r.resp; S_AXI_HP0_rid=hp_rsp.r.id; S_AXI_HP0_rlast=hp_rsp.r.last;
  end
  axi_delayer #(
    .aw_chan_t(axi_aw_chan_t), .w_chan_t(axi_w_chan_t), .b_chan_t(axi_b_chan_t),
    .ar_chan_t(axi_ar_chan_t), .r_chan_t(axi_r_chan_t),
    .axi_req_t(axi_req_t), .axi_resp_t(axi_resp_t),
    .StallRandomInput(1), .StallRandomOutput(1)
  ) i_delay (
    .clk_i(pl_clk0), .rst_ni(pl_resetn0), .slv_req_i(hp_req), .slv_resp_o(hp_rsp),
    .mst_req_o(mem_req), .mst_resp_i(mem_rsp)
  );
  axi_sim_mem #(
    .AddrWidth(64), .DataWidth(128), .IdWidth(6), .UserWidth(1),
    .axi_req_t(axi_req_t), .axi_rsp_t(axi_resp_t), .WarnUninitialized(0),
    .ClearErrOnAccess(1), .ApplDelay(2ns), .AcqDelay(8ns)
  ) i_ddr (
    .clk_i(pl_clk0), .rst_ni(pl_resetn0), .axi_req_i(mem_req), .axi_rsp_o(mem_rsp),
    .mon_r_last_o(), .mon_r_beat_count_o(), .mon_r_user_o(), .mon_r_id_o(),
    .mon_r_data_o(), .mon_r_addr_o(), .mon_r_valid_o(),
    .mon_w_last_o(), .mon_w_beat_count_o(), .mon_w_user_o(), .mon_w_id_o(),
    .mon_w_data_o(), .mon_w_addr_o(), .mon_w_valid_o()
  );
  localparam logic [63:0] INPUT=64'h8_1000_0000, WEIGHTS=64'h8_2000_0000,
    OUTPUT=64'h8_3000_0000, PROGRAM=64'h8_4000_0000, SCRATCH=64'h8_5000_0000;
  localparam logic [31:0] CTRL=32'ha0000014, STATUS=32'ha0000010,
    CSTATUS=32'ha000001c, TAG=32'ha0000020, CFG=32'ha0001000;
  logic [127:0] program_image[0:2047], inputs[0:65535], weights[0:65535], outputs[0:65535];
  logic [511:0] acc_expected[0:2047];
  int program_words, input_rows, weight_words, output_rows, acc_rows, expected_dma;
  int expected_cfg, expected_op;
  logic [31:0] expected_tag;
  int accepted_dma=0, irq_count=0, runs=0, fault_runs=0, ownership_runs=0;
  bit autonomous_active=0;
  string case_root, only_case;

  `define M5_DUT $root.tb_m5_full_system.dut
  always @(posedge pl_clk0) begin
    if (pl_resetn0 && `M5_DUT.dma_core_valid && `M5_DUT.dma_core_ready) accepted_dma++;
    if (pl_resetn0 && autonomous_active && `M5_DUT.i_idma.fe_req_valid && `M5_DUT.i_idma.fe_req_ready)
      $fatal(1,"Linux issued DMA after RUN");
    if (pl_resetn0 && `M5_DUT.core_irq_pulse) begin
      assert (!`M5_DUT.idma_busy && !`M5_DUT.core_dma_error) else $fatal(1,"IRQ before DMA drain or on error");
      for (int r=0;r<output_rows;r++) for(int b=0;b<16;b++)
        assert(i_ddr.mem[OUTPUT+64'(r)*16+64'(b)] === outputs[r][8*b+:8])
          else $fatal(1,"output not committed at IRQ, row=%0d byte=%0d",r,b);
      irq_count++;
    end
  end
  task automatic wr(input logic [31:0] a,d); write32(a,d,4'hf,0); endtask
  task automatic wr64(input logic [31:0] a,input logic [63:0] d);
    wr(a,d[31:0]); wr(a+4,d[63:32]);
  endtask
  task automatic host_dma(input logic [63:0] s,d,input int unsigned n);
    logic [31:0] tid,done_id;
    int timeout_;
    if(n==0) return;
    wr64(CFG+'hd8,s); wr64(CFG+'hd0,d); wr64(CFG+'he0,64'(n));
    wr64(CFG+'h108,0); wr64(CFG+'h100,0); wr64(CFG+'h110,1); wr(CFG,0);
    read32(CFG+'h44,tid); timeout_=0;
    do begin
      read32(CFG+'h84,done_id); timeout_++;
      if(timeout_>20000) $fatal(1,"host DMA timeout");
    end while(32'(done_id-tid)>=32'h80000000);
  endtask
  task automatic load_meta(input string directory);
    int fd,n;
    string line,k,v;
    program_words=0; input_rows=0; weight_words=0; output_rows=0; acc_rows=0;
    expected_cfg=0; expected_dma=0; expected_op=0; expected_tag=0;
    fd=$fopen({directory,"/meta.txt"},"r");
    if(fd==0) $fatal(1,"missing case %s",directory);
    while($fgets(line,fd)!=0) begin
      n=$sscanf(line,"%s %s",k,v);
      if(n==2) case(k)
        "prog_instrs": program_words=v.atoi();
        "ub_init_rows": input_rows=v.atoi();
        "weight_words": weight_words=v.atoi();
        "ddr_out_rows": output_rows=v.atoi();
        "acc_rows": acc_rows=v.atoi();
        "dma_count": expected_dma=v.atoi();
        "cfg_err": expected_cfg=v.atoi();
        "cfg_err_op": expected_op=v.atohex();
        "irq_tag": expected_tag=32'(v.atohex());
        default: ;
      endcase
    end
    $fclose(fd);
    assert(program_words>0 && program_words<=2048) else $fatal(1,"invalid program length");
    $readmemh({directory,"/prog.hex"},program_image,0,program_words-1);
    if(input_rows>0) $readmemh({directory,"/ub_init.hex"},inputs,0,input_rows-1);
    if(weight_words>0) $readmemh({directory,"/wmem.hex"},weights,0,weight_words-1);
    if(output_rows>0) $readmemh({directory,"/ddr_expect.hex"},outputs,0,output_rows-1);
    if(acc_rows>0) $readmemh({directory,"/acc_expect.hex"},acc_expected,0,acc_rows-1);
  endtask
  task automatic wait_drained(input string label_);
    logic [31:0] cs,st;
    int t;
    t=0;
    do begin
      read32(CSTATUS,cs); read32(STATUS,st); t++;
      if(t>20000) $fatal(1,"drain timeout %s cs=%h status=%h",label_,cs,st);
    end while(cs[0] || st[0]);
    assert(`M5_DUT.i_core.gen_dma_tracking.kind_empty && !`M5_DUT.dma_core_valid)
      else $fatal(1,"accepted/offered core request not drained %s",label_);
  endtask
  task automatic clear_state;
    logic [31:0] cs,st,levels;
    wr(CTRL,32'h361); wr(CTRL,0);
    read32(CSTATUS,cs); read32(STATUS,st); read32(32'ha0000018,levels);
    // HALT persists until the next RUN edge and is normal between cases.
    assert((cs & 32'hffff_fffd)==0 && st==0 && levels==0)
      else $fatal(1,"clear/flush did not recover cs=%h status=%h levels=%h",cs,st,levels);
  endtask
  task automatic prepare_case(input string name);
    logic [31:0] st;
    load_meta({case_root,"/",name});
    clear_state();
    for(int r=0;r<input_rows;r++) for(int b=0;b<16;b++) i_ddr.mem[INPUT+64'(r)*16+64'(b)]=inputs[r][8*b+:8];
    for(int r=0;r<weight_words;r++) for(int b=0;b<16;b++) i_ddr.mem[WEIGHTS+64'(r)*16+64'(b)]=weights[r][8*b+:8];
    for(int r=0;r<program_words;r++) for(int b=0;b<16;b++) i_ddr.mem[PROGRAM+64'(r)*16+64'(b)]=program_image[r][8*b+:8];
    for(int off=-16;off<output_rows*16+16;off++) i_ddr.mem[OUTPUT+64'(off)]=8'hee;
    for(int off=0;off<acc_rows*64;off++) i_ddr.mem[SCRATCH+64'(off)]=0;
    host_dma(SCRATCH,tpu_pkg::ACC_BASE,acc_rows*64);
    host_dma(PROGRAM,tpu_pkg::IFIFO_BASE,program_words*16);
    read32(STATUS,st); assert(st==0) else $fatal(1,"setup status %h",st);
  endtask
  task automatic run_case(input string name);
    logic [31:0] cs,st,tag;
    int before_dma,before_irq,t;
    prepare_case(name);
    before_dma=accepted_dma; before_irq=irq_count;
    autonomous_active=1;
    wr(CTRL,32'h86); t=0;
    do begin
      read32(CSTATUS,cs); t++;
      if(cs[4]) $fatal(1,"DMA fault in %s",name);
      if(t>50000) $fatal(1,"core timeout %s cs=%h dma=%0d",name,cs,accepted_dma-before_dma);
    end while(expected_cfg!=0 ? !cs[2] : (cs & 32'ha)!=32'ha);
    autonomous_active=0; wr(CTRL,6);
    assert(accepted_dma-before_dma==expected_dma) else $fatal(1,"DMA instruction count %s got%0d exp%0d",name,accepted_dma-before_dma,expected_dma);
    if(expected_cfg!=0) begin
      assert(cs[2] && int'(cs[15:8])==expected_op && irq_count==before_irq)
        else $fatal(1,"wrong cfg_err/IRQ %s cs=%h",name,cs);
    end else begin
      read32(TAG,tag);
      assert(tag==expected_tag && irq_count==before_irq+1) else $fatal(1,"bad completion tag %s",name);
      for(int r=0;r<output_rows;r++) for(int b=0;b<16;b++)
        assert(i_ddr.mem[OUTPUT+64'(r)*16+64'(b)]===outputs[r][8*b+:8]) else $fatal(1,"DDR mismatch %s row%0d",name,r);
      for(int b=0;b<16;b++)
        assert(i_ddr.mem[OUTPUT-16+64'(b)]==8'hee && i_ddr.mem[OUTPUT+64'(output_rows)*16+64'(b)]==8'hee)
          else $fatal(1,"DDR output guard %s",name);
      host_dma(tpu_pkg::ACC_BASE,SCRATCH,acc_rows*64);
      for(int r=0;r<acc_rows;r++) for(int b=0;b<64;b++)
        assert(i_ddr.mem[SCRATCH+64'(r)*64+64'(b)]===acc_expected[r][8*b+:8]) else $fatal(1,"ACC mismatch %s row%0d byte%0d",name,r,b);
    end
    wr(CTRL,0); read32(STATUS,st);
    assert(st==0) else $fatal(1,"final fabric error %s st=%h",name,st);
    runs++;
    $display("[PASS] %-32s prog=%0d dma=%0d cfg_err=%0d",name,program_words,expected_dma,expected_cfg);
  endtask
  task automatic test_owned_ub_guard;
    localparam logic [63:0] OLD_IMAGE=SCRATCH+64'h20000,
      NEW_IMAGE=SCRATCH+64'h21000, GOT_IMAGE=SCRATCH+64'h22000;
    logic [31:0] st,cs;
    int before_dma,before_irq;
    clear_state();
    before_dma=accepted_dma; before_irq=irq_count;
    for(int b=0;b<64;b++) begin
      i_ddr.mem[OLD_IMAGE+64'(b)]=8'(b*5+'h11);
      i_ddr.mem[NEW_IMAGE+64'(b)]=~8'(b*5+'h11);
      i_ddr.mem[GOT_IMAGE+64'(b)]=8'hee;
    end
    host_dma(OLD_IMAGE,tpu_pkg::UB_BASE,64);
    wr(CTRL,6);                       // TPU owns UB; Linux still has source tag zero.
    host_dma(NEW_IMAGE,tpu_pkg::UB_BASE,64);
    read32(STATUS,st); read32(CSTATUS,cs);
    assert(st[5] && st[3] && cs[4])
      else $fatal(1,"owned UB host write was not rejected/flagged status=%h cs=%h",st,cs);
    host_dma(tpu_pkg::UB_BASE,GOT_IMAGE,64);
    for(int b=0;b<64;b++)
      assert(i_ddr.mem[GOT_IMAGE+64'(b)]===8'(b*5+'h11))
        else $fatal(1,"owned UB bytes changed after rejected tag0 write byte=%0d",b);
    assert(accepted_dma==before_dma && irq_count==before_irq)
      else $fatal(1,"inactive core issued work/IRQ during ownership guard test");
    wr(CTRL,0); wait_drained("owned UB rejection"); clear_state();
    run_case("fc1");                  // Core tag one must write the same owned UB successfully.
    ownership_runs++;
    $display("[PASS] owned UB rejects Linux tag0, preserves bytes, core tag1 recovers");
  endtask
  task automatic test_dma_fault(input bit read_fault);
    logic [31:0] cs,st;
    int before_dma,before_irq,t;
    prepare_case("fc1");
    if(read_fault) i_ddr.rerr[INPUT]=axi_pkg::RESP_SLVERR;
    else i_ddr.werr[OUTPUT]=axi_pkg::RESP_DECERR;
    before_dma=accepted_dma; before_irq=irq_count;
    autonomous_active=1;
    wr(CTRL,32'h86); t=0;
    do begin
      read32(CSTATUS,cs); t++;
      assert(irq_count==before_irq && !cs[3])
        else $fatal(1,"success IRQ on injected DMA %s error",read_fault ? "read" : "write");
      if(t>20000) $fatal(1,"injected DMA %s error did not stop core cs=%h",read_fault ? "read" : "write",cs);
    end while(!cs[4]);
    assert(!cs[2]) else $fatal(1,"DMA bus error became instruction cfg_err cs=%h",cs);
    wr(CTRL,6);                       // Stop RUN, retain ownership until all accepted work drains.
    wait_drained(read_fault ? "RHM SLVERR" : "WHM DECERR");
    read32(STATUS,st); read32(CSTATUS,cs);
    assert(cs[4] && !cs[3] && irq_count==before_irq &&
           (read_fault ? st[2] : st[3]))
      else $fatal(1,"injected DMA error missing or IRQ escaped status=%h cs=%h",st,cs);
    assert(accepted_dma>before_dma && accepted_dma-before_dma<=expected_dma)
      else $fatal(1,"fault did not drain accepted descriptors");
    autonomous_active=0;
    // Remove any residual injection before validating a complete fresh job.
    i_ddr.rerr[INPUT]=axi_pkg::RESP_OKAY;
    i_ddr.werr[OUTPUT]=axi_pkg::RESP_OKAY;
    clear_state();
    assert(irq_count==before_irq)
      else $fatal(1,"clearing DMA fault executed a stale held IRQ");
    run_case("fc1");
    fault_runs++;
    $display("[PASS] DMA %s error stops IRQ, drains, clears and fresh fc1 recovers",
      read_fault ? "read SLVERR" : "write DECERR");
  endtask
  task automatic run_manifest();
    int fd;
    string name,line;
    fd=$fopen({case_root,"/cases.txt"},"r");
    if(fd==0) $fatal(1,"generate Phase B cases first");
    while($fgets(line,fd)!=0) begin
      if($sscanf(line,"%s",name)==1) begin
        // Manifest entries are paths; take their final component.
        for(int i=name.len()-1;i>=0;i--) if(name.getc(i)==8'h2f) begin name=name.substr(i+1,name.len()-1); break; end
        run_case(name);
      end
    end
    $fclose(fd);
  endtask
  initial begin : phase_b_tests
    int fd;
    string name,line;
    logic [31:0] v;
    bit directed_faults;
    case_root="build/m5/cases"; only_case="";
    if($value$plusargs("CASES_DIR=%s",case_root));
    if($value$plusargs("ONLY=%s",only_case));
    directed_faults=(only_case=="") || $test$plusargs("FAULTS");
    repeat(8) @(negedge pl_clk0); pl_resetn0=1;
    repeat(8) @(negedge pl_clk0);
    read32(32'ha0000000,v); assert(v==32'h500) else $fatal(1,"version %h",v);  // 0x500 = Phase B + CNN
    read32(32'ha0000008,v); assert(v==32'h12345678) else $fatal(1,"build %h",v);
    if(only_case!="") run_case(only_case);
    else begin
      run_manifest();
      // Back-to-back repeat without reset (L14): the four MLPs by default; +REPEAT_LIST repeats
      // every case of the given manifest instead (used for the CNN directory).
      if($test$plusargs("REPEAT_LIST")) run_manifest();
      else begin run_case("fc1"); run_case("mlp3"); run_case("mlp_b1"); run_case("mlp_wide"); end
    end
    if(directed_faults && !$test$plusargs("NO_DIRECTED")) begin
      test_owned_ub_guard();
      test_dma_fault(1'b1); test_dma_fault(1'b0);
    end
    $display("M5_FULL_SYSTEM_PASS runs=%0d fault_recoveries=%0d ownership_checks=%0d",runs,fault_runs,ownership_runs);
    $finish;
  end
  initial begin #20ms; $fatal(1,"full-system watchdog"); end
  `undef M5_DUT
