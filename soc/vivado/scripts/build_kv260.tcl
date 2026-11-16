# KV260 bitstream: kv260_ps BD (PS only) + kv260_top (iDMA + fabric + memories).
#   vivado -mode batch -source soc/vivado/scripts/build_kv260.tcl
# Lessons: -jobs 1 / maxThreads 2 (L22), timing + clock-network reports (L20/L21), BUILD_ID (L7).
set root  [file normalize [file dirname [info script]]/../../..]
set out   $root/build/kv260
set build_id [clock seconds]
file mkdir $out
set_param general.maxThreads 2

create_project -force kv260 $out/prj -part xck26-sfvc784-2LV-c
set bp [lindex [get_board_parts -quiet *kv260_som*] end]
if {$bp ne ""} { set_property board_part $bp [current_project] }

source $root/soc/vivado/bd/kv260_ps.tcl
generate_target all [get_files kv260_ps.bd]
add_files -norecurse [make_wrapper -files [get_files kv260_ps.bd] -top]

set ROOT $root
source $root/build/kv260_sources.tcl
# Vivado-only: URAM through xpm_memory_tdpram (rtl/memory/ram_tdp.sv)
set_property verilog_define [concat [get_property verilog_define [current_fileset]] TPU_USE_XPM] [current_fileset]
set_property top kv260_top [current_fileset]
set_property generic "BUILD_ID=32'd$build_id" [current_fileset]

launch_runs synth_1 -jobs 1
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { error "synth_1 failed" }
launch_runs impl_1 -to_step write_bitstream -jobs 1
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "impl_1 failed" }

open_run impl_1
report_timing_summary -max_paths 10 -file $out/timing.rpt
report_utilization -file $out/util.rpt
report_clock_networks -file $out/clock_networks.rpt
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set bit [glob $out/prj/kv260.runs/impl_1/*.bit]
file copy -force $bit $out/tpu_kv260.bit
set fh [open $out/build_id.txt w]; puts $fh [format "0x%08X" $build_id]; close $fh
set uram [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ BLOCKRAM.URAM.*}]]
set bram [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ BLOCKRAM.BRAM.*}]]
puts "KV260_BUILD_DONE build_id=[format 0x%08X $build_id] wns=$wns uram=$uram bram=$bram"
