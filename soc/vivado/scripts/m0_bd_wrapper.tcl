# Generate the kv260_ps BD + HDL wrapper only (to inspect port names).
set root [file normalize [file dirname [info script]]/../../..]
create_project -force m0_bd $root/build/m0_board/bd_prj -part xck26-sfvc784-2LV-c
set bp [lindex [get_board_parts -quiet *kv260_som*] end]
if {$bp ne ""} { set_property board_part $bp [current_project]; puts "BOARD_PART $bp" }
source $root/soc/vivado/bd/kv260_ps.tcl
set w [make_wrapper -files [get_files kv260_ps.bd] -top]
puts "WRAPPER $w"
foreach a [get_bd_addr_segs] { catch { puts "SEG [get_property NAME $a] [get_property OFFSET $a] [get_property RANGE $a]" } }
