# M0: elaborate (synth_design -rtl) the pinned PULP iDMA subsystem for the KV260 part.
#   vivado -mode batch -source scripts/m0_vivado_elab.tcl
set_part xck26-sfvc784-2LV-c
source build/m0/vivado_sources.tcl
set_param general.maxThreads 1
synth_design -rtl -top m0_idma_elab_top -part xck26-sfvc784-2LV-c -name m0_rtl
puts "M0_VIVADO_ELAB_OK"
