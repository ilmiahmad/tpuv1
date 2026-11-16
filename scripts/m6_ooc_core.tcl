# CNN: OOC synth + place/route of tpu_core (N=16, EnableDma=1, conv + pool) for xck26 @ 100 MHz.
#   vivado -mode batch -source scripts/m6_ooc_core.tcl     (run from the repo root)
set root [file normalize [file dirname [info script]]/..]
set out  $root/build/m6/a/ooc
file mkdir $out
set_param general.maxThreads 2
create_project -in_memory -part xck26-sfvc784-2LV-c
set ROOT $root
source $out/vivado_sources.tcl
read_xdc -mode out_of_context $root/soc/vivado/constraints/ooc_100mhz.xdc   ;# before synth (L21)
synth_design -verilog_define TPU_USE_XPM -generic EnableDma=1 -top tpu_core \
  -part xck26-sfvc784-2LV-c -mode out_of_context -flatten_hierarchy rebuilt
report_utilization -file $out/util_synth.rpt
opt_design
place_design
route_design
report_utilization -file $out/util_routed.rpt
report_utilization -hierarchical -hierarchical_depth 2 -file $out/util_hier.rpt
report_timing_summary -max_paths 10 -file $out/timing.rpt
report_timing -max_paths 3 -file $out/timing_paths.rpt
set dsp  [llength [get_cells -hier -filter {REF_NAME == DSP48E2}]]
set bram [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ BLOCKRAM.BRAM.*}]]
set ff   [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.*}]]
set wns  [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "M6_CORE_OOC dsp=$dsp bram=$bram ff=$ff wns=$wns"
