# kv260_ps — PS-only block design (ADR-0001). Everything else lives in hand-written SV.
# PS port widths/clock here MUST match what the board actually runs (L1): HPM0 128-bit,
# HP0 128-bit, pl_clk0 100 MHz. On the board these are set by the dtbo (afi config-afi,
# clocking0 assigned-clock-rates) — the PSU__* values below only shape the PL interface.

create_bd_design kv260_ps

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
if {[llength [get_board_parts -quiet *kv260_som*]] > 0} {
  apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
}
set_property -dict [list \
  CONFIG.PSU__USE__M_AXI_GP0          {1}   \
  CONFIG.PSU__MAXIGP0__DATA_WIDTH     {128} \
  CONFIG.PSU__USE__M_AXI_GP1          {0}   \
  CONFIG.PSU__USE__M_AXI_GP2          {0}   \
  CONFIG.PSU__USE__S_AXI_GP2          {1}   \
  CONFIG.PSU__SAXIGP2__DATA_WIDTH     {128} \
  CONFIG.PSU__FPGA_PL0_ENABLE         {1}   \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
  CONFIG.PSU__USE__IRQ0               {1}   \
] $ps

# External interfaces with stable names for the SV top.
make_bd_intf_pins_external -name M_AXI_HPM0 [get_bd_intf_pins $ps/M_AXI_HPM0_FPD]
make_bd_intf_pins_external -name S_AXI_HP0  [get_bd_intf_pins $ps/S_AXI_HP0_FPD]
make_bd_pins_external      -name pl_clk0    [get_bd_pins $ps/pl_clk0]
make_bd_pins_external      -name pl_resetn0 [get_bd_pins $ps/pl_resetn0]
make_bd_pins_external      -name pl_ps_irq0 [get_bd_pins $ps/pl_ps_irq0]

# Both AXI ports are clocked by pl_clk0 (single clock domain, no CDC).
connect_bd_net [get_bd_pins $ps/pl_clk0] [get_bd_pins $ps/maxihpm0_fpd_aclk]
connect_bd_net [get_bd_pins $ps/pl_clk0] [get_bd_pins $ps/saxihp0_fpd_aclk]
set_property CONFIG.ASSOCIATED_BUSIF {M_AXI_HPM0:S_AXI_HP0} [get_bd_ports pl_clk0]

# Address map: 0xA000_0000 + 8 MiB (tpu_pkg::PL_END). The first 64 KiB is also the base-DT
# generic-uio "fabric" node (/dev/uio4); the memory windows above it are reached via /dev/mem.
assign_bd_address
set hpm_seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces $ps/Data] -filter {NAME =~ *M_AXI_HPM0*}]
set_property offset 0xA0000000 $hpm_seg
set_property range  8M         $hpm_seg

validate_bd_design
save_bd_design
