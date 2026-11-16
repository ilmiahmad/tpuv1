#!/bin/csh -f
# M3 matrix_unit regression on the Xcelium server (Xcelium 24.09, csh): every golden case, N=4/8/16.
#   csh scripts/m3_xrun.csh [SEED]
set seed = 1
if ($#argv >= 1) set seed = $1
mkdir -p build/m3/xrun
foreach n (4 8 16)
  xrun -sv -64bit -timescale 1ns/1ps -elaborate -xmlibdirname build/m3/xrun/x_N${n}.d \
       -l build/m3/xrun/elab_N${n}.log -defparam tb_matrix_unit.N=${n} -top tb_matrix_unit \
       -f build/m3/rtl_only.f tb/compute/tb_matrix_unit.sv >& /dev/null
end
foreach d (`cat build/m3/vectors/cases.txt`)
  set c = `basename $d`
  set n = `awk '$1=="N"{print $2}' $d/meta.txt`
  xrun -R -xmlibdirname build/m3/xrun/x_N${n}.d -l build/m3/xrun/sim_${c}_s${seed}.log \
       +VEC=$d +SEED=${seed} >& /dev/null
  echo "$c `grep -h -o 'M3_MXU_[A-Z]*' build/m3/xrun/sim_${c}_s${seed}.log` `grep -h 'checked=' build/m3/xrun/sim_${c}_s${seed}.log`"
end
