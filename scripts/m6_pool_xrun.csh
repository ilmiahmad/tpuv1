#!/bin/csh -f
# Pooling regression on the Xcelium server (Xcelium 24.09, csh): pool_unit vs the golden-model vectors, then the
# act_seq + activation_unit + pool_unit testbench. Run from the repo root of the m6_c workspace.
#   csh scripts/m6_pool_xrun.csh [SEED]
set seed = 1
if ($#argv >= 1) set seed = $1
mkdir -p build/m6/pool_c/xrun
xrun -sv -64bit -timescale 1ns/1ps -access +r -svseed $seed \
  -xmlibdirname build/m6/pool_c/xrun/unit.d -l build/m6/pool_c/xrun/pool_unit_s${seed}.log \
  -top tb_pool_unit rtl/activation/pool_unit.sv tb/activation/tb_pool_unit.sv \
  +POOLDIR=build/m6/pool +SEED=$seed
if ($status != 0) exit 1
xrun -sv -64bit -timescale 1ns/1ps -access +r -svseed $seed \
  -xmlibdirname build/m6/pool_c/xrun/seq.d -l build/m6/pool_c/xrun/act_seq_s${seed}.log \
  -top tb_act_seq_pool -f build/m6/pool_c/xrun_deps.f \
  rtl/activation/act_params.sv rtl/activation/requant_lane.sv rtl/activation/activation_unit.sv \
  rtl/activation/pool_unit.sv rtl/control/act_seq.sv tb/activation/tb_act_seq_pool.sv \
  +SEED=$seed +REPS=2
exit $status
