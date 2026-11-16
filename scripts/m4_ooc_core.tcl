# M4: OOC synth + place/route of tpu_core (N=16, real activation_unit) for xck26 @ 100 MHz.
set root [file normalize [file dirname [info script]]/..]
set out  $root/build/m4/core/ooc
file mkdir $out
set_param general.maxThreads 2
create_project -in_memory -part xck26-sfvc784-2LV-c
set ROOT $root
source $root/build/m4/core/vivado_sources.tcl
read_xdc -mode out_of_context $root/soc/vivado/constraints/ooc_100mhz.xdc
synth_design -verilog_define TPU_USE_XPM -top tpu_core -part xck26-sfvc784-2LV-c -mode out_of_context -flatten_hierarchy rebuilt
report_utilization -file $out/util_synth.rpt
opt_design
place_design
route_design
report_utilization -file $out/util_routed.rpt
report_utilization -hierarchical -hierarchical_depth 2 -file $out/util_hier.rpt
report_timing_summary -max_paths 10 -file $out/timing.rpt
set dsp  [llength [get_cells -hier -filter {REF_NAME == DSP48E2}]]
set bram [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ BLOCKRAM.BRAM.*}]]
set lut  [llength [get_cells -hier -filter {PRIMITIVE_GROUP == LUT}]]
set ff   [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.*}]]
set wns  [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "M4_CORE_OOC dsp=$dsp bram=$bram lut=$lut ff=$ff wns=$wns"
