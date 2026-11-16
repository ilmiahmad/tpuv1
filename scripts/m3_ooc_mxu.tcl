# M3: OOC synth + place/route of matrix_unit (N=16) on the KV260 part at 100 MHz.
# XDC read BEFORE synth_design (L21). Reports DSP / LUT / FF / WNS.
set root [file normalize [file dirname [info script]]/..]
set out  $root/build/m3/ooc
file mkdir $out
set_param general.maxThreads 2
create_project -in_memory -part xck26-sfvc784-2LV-c
foreach f {mac_pe_ws systolic_array data_setup output_deskew weight_loader matrix_unit} {
  read_verilog -sv $root/rtl/compute/$f.sv
}
read_xdc -mode out_of_context $root/soc/vivado/constraints/ooc_100mhz.xdc
synth_design -top matrix_unit -part xck26-sfvc784-2LV-c -mode out_of_context \
             -flatten_hierarchy rebuilt -generic N=16
report_utilization -file $out/util_synth.rpt
opt_design
place_design
phys_opt_design
route_design
report_utilization -file $out/util_routed.rpt
report_timing_summary -max_paths 10 -file $out/timing.rpt
set dsp  [llength [get_cells -hier -filter {REF_NAME == DSP48E2}]]   ;# macro count (sub-cells would give 9x)
set lut  [llength [get_cells -hier -filter {PRIMITIVE_GROUP == CLB && PRIMITIVE_TYPE =~ CLB.LUT.*}]]
set ff   [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.*}]]
set wns  [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "M3_OOC dsp=$dsp lut=$lut ff=$ff wns=$wns"
