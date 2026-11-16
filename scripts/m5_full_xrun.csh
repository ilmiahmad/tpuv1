#!/bin/csh -f
# Phase B full-system regression (kv260_top + behavioural PS) on Xcelium 24.09, csh.
# Run from the repo root on the Xcelium server after scripts/m5_full_xrun_prepare.sh + rsync.
#   csh scripts/m5_full_xrun.csh [SEED] [extra plusargs, e.g. +CASES_DIR=build/m6/cases_all]
set seed = 1
set extra = ""
if ($#argv >= 1) set seed = $1
if ($#argv >= 2) set extra = "$argv[2-]"
mkdir -p build/m5/full/xcelium
xrun -sv -64bit -timescale 1ns/1ps -access +r -svseed $seed \
  -define TPU_PHASE_B_SIM -incdir tb/system \
  -xmlibdirname build/m5/full/xcelium/x.d -l build/m5/full/xcelium/full_s${seed}.log \
  -top tb_m5_full_system -f build/m5/full/xrun.f $extra
exit $status
