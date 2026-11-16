// TPU-KV260 UVM package (Xcelium, UVM 1.2).
`include "uvm_macros.svh"

package tpu_uvm_pkg;
  import uvm_pkg::*;

  `include "agents/axil_agent.sv"
  `include "agents/hpm_agent.sv"
  `include "agents/native_agent.sv"
  `include "agents/fifo_sink_agent.sv"
  `include "env/idma_ral.sv"
  `include "env/dma_xfer.sv"
  `include "env/tpu_model_sb.sv"
  `include "env/tpu_env.sv"
  `include "sequences/tpu_vseqs.sv"
  `include "tests/tpu_tests.sv"
endpackage
