#!/bin/csh -f
# M1 iDMA regression on the Xcelium server (Xcelium 24.09, csh).
#   csh scripts/m1_xrun.csh [SEED] [NRAND]
set seed = 1
set nrand = 300
if ($#argv >= 1) set seed = $1
if ($#argv >= 2) set nrand = $2
mkdir -p build/m1
# main regression (random stalls, 32-bit IDs)
xrun -sv -64bit -timescale 1ns/1ps -access +r -xmlibdirname build/m1/x_main.d \
     -l build/m1/xrun_main_s${seed}.log -top tb_m1_idma -f build/m0/xrun_sim.f \
     +SEED=${seed} +NRAND=${nrand}
# ID-wrap variant (4-bit IDs)
xrun -sv -64bit -timescale 1ns/1ps -access +r -xmlibdirname build/m1/x_wrap.d \
     -l build/m1/xrun_wrap.log -top tb_m1_idma -defparam tb_m1_idma.ID_W=4 -f build/m0/xrun_sim.f
