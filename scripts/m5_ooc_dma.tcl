# Enabled Phase B iDMA transport: real frontend/midend/backend, eight source tags.
# Generate build/m5_dma/sources.tcl using bender script vivado -t rtl -t m0_elab -t m5_elab.
set root [file normalize [file dirname [info script]]/..]
set out $root/build/m5_dma/ooc
file mkdir $out
set_param general.maxThreads 2
create_project -in_memory -part xck26-sfvc784-2LV-c
source $root/build/m5_dma/sources.tcl
set xdc [open $out/clock.xdc w]
puts $xdc {create_clock -name clk -period 10.000 [get_ports clk_i]}
close $xdc
read_xdc $out/clock.xdc
synth_design -top m5_idma_elab_top -mode out_of_context -part xck26-sfvc784-2LV-c
opt_design
report_utilization -file $out/util.rpt
report_timing_summary -max_paths 10 -file $out/timing.rpt
report_clock_networks -file $out/clocks.rpt
write_checkpoint -force $out/dma.dcp
set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
puts "M5_DMA_OOC_DONE wns=$wns"
if {$wns < 0.5} { error "M5 DMA transport misses 100 MHz margin target: WNS=$wns" }
