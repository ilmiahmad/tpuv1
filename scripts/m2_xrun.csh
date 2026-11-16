#!/bin/csh -f
# M2 fabric/memory regression on the Xcelium server (Xcelium 24.09, csh).  csh scripts/m2_xrun.csh [SEED]
set seed = 1
if ($#argv >= 1) set seed = $1
mkdir -p build/m2
xrun -sv -64bit -timescale 1ns/1ps -access +r -xmlibdirname build/m2/x.d \
     -l build/m2/xrun_s${seed}.log -top tb_m2_fabric -f build/m2/xrun_sim.f +SEED=${seed}
