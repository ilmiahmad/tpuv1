# M4 OOC synth + P&R of activation_unit (N=16) on xck26 @ 100 MHz.
set root [file normalize [file dirname [info script]]/..]
set out  $root/build/m4/act/ooc
file mkdir $out
set_param general.maxThreads 2
create_project -in_memory -part xck26-sfvc784-2LV-c
read_verilog -sv [list $root/rtl/activation/act_params.sv $root/rtl/activation/requant_lane.sv $root/rtl/activation/activation_unit.sv]
read_xdc -mode out_of_context $root/soc/vivado/constraints/ooc_100mhz.xdc
synth_design -top activation_unit -part xck26-sfvc784-2LV-c -mode out_of_context -flatten_hierarchy rebuilt
report_utilization -file $out/util_synth.rpt
opt_design
place_design
phys_opt_design
route_design
report_utilization -file $out/util_routed.rpt
report_timing_summary -max_paths 10 -file $out/timing.rpt
set dsp [llength [get_cells -hier -filter {REF_NAME == DSP48E2}]]
set lut [llength [get_cells -hier -filter {PRIMITIVE_GROUP == CLB && PRIMITIVE_SUBGROUP == LUT}]]
set ff  [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.*}]]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "M4_ACT_OOC dsp=$dsp lut=$lut ff=$ff wns=$wns"
