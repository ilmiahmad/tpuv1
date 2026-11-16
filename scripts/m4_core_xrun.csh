#!/bin/csh -f
# M4 tpu_core regression on the Xcelium server (Xcelium 24.09, csh): all golden cases in build/m4/cases.
mkdir -p build/m4/core/xrun
xrun -sv -64bit -timescale 1ns/1ps -access +r -xmlibdirname build/m4/core/xrun/x.d \
     -l build/m4/core/xrun/core.log -top tb_tpu_core -f build/m4/core/xrun.f \
     +CASES=build/m4/cases/cases.txt
