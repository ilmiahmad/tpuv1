#!/bin/csh -f
# M0: Xcelium 24.09 elaboration of the pinned PULP iDMA subsystem (run on the Xcelium server, csh).
#   cd ~/tpuv1_kv260 && csh scripts/m0_xrun_elab.csh
mkdir -p build/m0/xcelium
xrun -elaborate -sv -64bit -timescale 1ns/1ps \
     -xmlibdirname build/m0/xcelium/xcelium.d \
     -l build/m0/xrun_elab.log \
     -top m0_idma_elab_top \
     -f build/m0/xrun.f
if ($status == 0) then
  echo "M0_XCELIUM_ELAB_OK"
else
  echo "M0_XCELIUM_ELAB_FAIL"
endif
