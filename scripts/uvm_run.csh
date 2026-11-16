#!/bin/csh -f
# UVM regression on the Xcelium server (Xcelium 24.09, UVM 1.2), csh.
#   csh scripts/uvm_run.csh [TEST] [SEED] [NXFER]
set test  = tpu_dma_random_test
set seed  = 1
set nxfer = 200
set verb = UVM_LOW
set extra = ""
set tag = ""
if ($#argv >= 1) set test  = $1
if ($#argv >= 2) set seed  = $2
if ($#argv >= 3) set nxfer = $3
if ($#argv >= 4) set verb = $4
if ($#argv >= 5) set extra = "$5"
if ($#argv >= 6) set tag = $6
mkdir -p build/uvm
xrun -sv -64bit -uvm -uvmhome CDNS-1.2 -timescale 1ns/1ps -access +r \
     -xmlibdirname build/uvm/x.d -l build/uvm/${test}_s${seed}${tag}.log \
     -coverage functional -covworkdir build/uvm/cov -covtest ${test}_s${seed}${tag} -covoverwrite \
     -f build/m2/xrun_sim.f -incdir tb/uvm tb/uvm/tpu_uvm_if.sv tb/uvm/tpu_uvm_pkg.sv tb/uvm/tb_uvm_top.sv \
     -top tb_uvm_top +UVM_TESTNAME=${test} +NXFER=${nxfer} ${extra} -svseed ${seed} +UVM_VERBOSITY=${verb}
