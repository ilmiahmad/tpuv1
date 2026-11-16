#!/bin/csh -f
# M1 seed sweep (random stalls) + one no-stall run. Reuses compiled snapshot dirs.
mkdir -p build/m1
foreach s (2 3 4 5 6 7 8 9)
  xrun -sv -64bit -timescale 1ns/1ps -access +r -xmlibdirname build/m1/x_main.d \
       -l build/m1/xrun_main_s${s}.log -top tb_m1_idma -f build/m0/xrun_sim.f +SEED=${s} +NRAND=400 >& /dev/null
end
xrun -sv -64bit -timescale 1ns/1ps -access +r -xmlibdirname build/m1/x_nostall.d \
     -l build/m1/xrun_nostall.log -top tb_m1_idma -defparam tb_m1_idma.STALL=0 \
     -f build/m0/xrun_sim.f +SEED=11 +NRAND=400 >& /dev/null
