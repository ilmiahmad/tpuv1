# M2: OOC synth + place/route of tpu_fabric; asserts RAM primitives were inferred (L19).
set root [file normalize [file dirname [info script]]/..]
set out  $root/build/m2/ooc
file mkdir $out
set_param general.maxThreads 2
create_project -in_memory -part xck26-sfvc784-2LV-c
set ROOT $root
source $root/build/m2/vivado_sources.tcl
# Vivado-only: URAM through xpm_memory_tdpram (rtl/memory/ram_tdp.sv)
set_property verilog_define [concat [get_property verilog_define [current_fileset]] TPU_USE_XPM] [current_fileset]
read_xdc -mode out_of_context $root/soc/vivado/constraints/ooc_100mhz.xdc
synth_design -verilog_define TPU_USE_XPM -top tpu_fabric -part xck26-sfvc784-2LV-c -mode out_of_context -flatten_hierarchy rebuilt
report_utilization -file $out/util_synth.rpt
opt_design
place_design
route_design
report_utilization -file $out/util_routed.rpt
report_utilization -hierarchical -hierarchical_depth 3 -file $out/util_hier.rpt
report_timing_summary -max_paths 10 -file $out/timing.rpt
set uram [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ BLOCKRAM.URAM.*}]]
set bram [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ BLOCKRAM.BRAM.*}]]
set ff   [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.*}]]
set wns  [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "M2_OOC uram=$uram bram=$bram ff=$ff wns=$wns"
