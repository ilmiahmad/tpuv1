#!/bin/csh -f
# M4 activation_unit regression on the Xcelium server (Xcelium 24.09, csh). Stall-pattern sweep.
mkdir -p build/m4/act/xrun
xrun -sv -64bit -timescale 1ns/1ps -elaborate -access +r -xmlibdirname build/m4/act/xrun/x.d \
     -l build/m4/act/xrun/elab.log -top tb_activation_unit \
     rtl/activation/act_params.sv rtl/activation/requant_lane.sv rtl/activation/activation_unit.sv \
     tb/activation/tb_activation_unit.sv >& /dev/null
set seeds = (1 2 3 4 5 6)
set gaps  = (30 0 80 10 50 95)
set stls  = (30 0 10 80 50 95)
set i = 1
while ($i <= 6)
  xrun -R -xmlibdirname build/m4/act/xrun/x.d -l build/m4/act/xrun/sim_s$seeds[$i].log \
       +SEED=$seeds[$i] +IN_GAP_PCT=$gaps[$i] +OUT_STALL_PCT=$stls[$i] >& /dev/null
  grep -H -e M4_ACT_ -e accepted= build/m4/act/xrun/sim_s$seeds[$i].log
  @ i++
end
