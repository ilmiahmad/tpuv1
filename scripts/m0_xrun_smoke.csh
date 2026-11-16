#!/bin/csh -f
# M0 iDMA smoke simulation on the Xcelium server (Xcelium 24.09, csh).
mkdir -p build/m0/xcelium_sim
xrun -sv -64bit -timescale 1ns/1ps -access +r \
     -xmlibdirname build/m0/xcelium_sim/xcelium.d \
     -l build/m0/xrun_smoke.log \
     -top tb_m0_idma_smoke \
     -f build/m0/xrun_sim.f
