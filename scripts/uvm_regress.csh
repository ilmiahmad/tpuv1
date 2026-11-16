#!/bin/csh -f
# UVM regression: all tests, several seeds (the Xcelium server, csh). Summary: build/uvm/summary.txt
mkdir -p build/uvm
rm -f build/uvm/summary.txt
foreach s (1 2 3)
  csh scripts/uvm_run.csh tpu_dma_random_test $s 200 >& /dev/null
end
csh scripts/uvm_run.csh tpu_fifo_backpressure_test 1 >& /dev/null
csh scripts/uvm_run.csh tpu_ownership_err_test 1 >& /dev/null
foreach f (`ls build/uvm/tpu_*_s?.log`)
  set r = `grep -h -o "TPU_UVM_[A-Z]*[^ ]*" $f`
  echo "$f $r" >> build/uvm/summary.txt
end
