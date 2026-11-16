#!/bin/sh
# Unbind and unload the TPU driver, give the node back to uio_pdrv_genirq.
# ALWAYS run before "xmutil unloadapp": a driver bound to an unloaded PL can hang the CPU (L5).
DEV=a0000000.fabric
BUS=/sys/bus/platform
[ -e $BUS/drivers/tpu_kv260/$DEV ] && echo $DEV > $BUS/drivers/tpu_kv260/unbind
[ -e $BUS/devices/$DEV ] && echo > $BUS/devices/$DEV/driver_override
lsmod | grep -q '^tpu_kv260' && rmmod tpu_kv260
[ -e $BUS/devices/$DEV ] && echo $DEV > $BUS/drivers_probe
echo unloaded
