#!/bin/csh -f
# Run from repository root on the Xcelium server after generating build/m5_dma/xrun.f.
set seed = 1
if ($#argv >= 1) set seed = $1
mkdir -p build/m5_dma/xcelium
xrun -sv -64bit -timescale 1ns/1ps -access +r -svseed $seed \
  -xmlibdirname build/m5_dma/xcelium/dma.d -l build/m5_dma/xcelium/dma_s${seed}.log \
  -top tb_m5_dma -f build/m5_dma/xrun.f +SEED=$seed
if ($status != 0) exit 1
xrun -sv -64bit -timescale 1ns/1ps -access +r \
  -xmlibdirname build/m5_dma/xcelium/arbiter.d -l build/m5_dma/xcelium/arbiter.log \
  -top tb_m5_arbiter -f build/m5_dma/xrun.f
exit $status
