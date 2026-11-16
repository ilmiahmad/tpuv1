#!/bin/csh -f
# CONVOLVE / decoder regression on Xcelium 24.09 (the Xcelium server, csh). Run from the repo root after
# scripts/m6_xrun_prepare.sh and an rsync. Usage: csh scripts/m6_xrun.csh unit|core|full [args]
#   unit : tb_conv_address_gen on build/m6/addr/cases.txt
#   core : tb_tpu_core on +CASES=<cases.txt>            (Phase A cases, EnableDma = 0)
#   full : tb_m5_full_system on +CASES_DIR=<dir> [extra plusargs, e.g. +REPEAT_LIST +NO_DIRECTED]
#          (Phase B, kv260_top + behavioural PS)
set mode = $1
mkdir -p build/m6/a/xr
if ($mode == unit) then
  xrun -sv -64bit -timescale 1ns/1ps -access +r -xmlibdirname build/m6/a/xr/unit.d \
    -l build/m6/a/xr/unit.log -top tb_conv_address_gen \
    rtl/control/conv_address_gen.sv tb/control/tb_conv_address_gen.sv +CASES=build/m6/addr/cases.txt
else if ($mode == core) then
  xrun -sv -64bit -timescale 1ns/1ps -access +r -xmlibdirname build/m6/a/xr/core.d \
    -l build/m6/a/xr/core_$argv[3].log -top tb_tpu_core -f build/m6/a/xr/core.f \
    tb/core/tb_tpu_core.sv +CASES=$argv[2]
else if ($mode == full) then
  xrun -sv -64bit -timescale 1ns/1ps -access +r -define TPU_PHASE_B_SIM -incdir tb/system \
    -xmlibdirname build/m6/a/xr/full.d -l build/m6/a/xr/full_$argv[3].log -top tb_m5_full_system \
    -f build/m6/a/xr/full.f +CASES_DIR=$argv[2] $argv[4-]
endif
exit $status
