# Out-of-context block checks: 100 MHz on clk_i (read BEFORE synth_design, L21)
create_clock -name clk -period 10.000 [get_ports clk_i]
