#!/bin/sh
# Load the TPU driver and bind it to the PL fabric node (replaces uio_pdrv_genirq).
# Preconditions: the tpu_kv260 PL image is loaded (xmutil loadapp tpu_kv260). Run as root.
set -e
DEV=a0000000.fabric
BUS=/sys/bus/platform
HERE="$(cd "$(dirname "$0")" && pwd)"
KO="${KO:-$HERE/kernel/tpu_kv260.ko}"

[ "$(cat /sys/class/fpga_manager/fpga0/state)" = operating ] || { echo "PL is not loaded (xmutil loadapp tpu_kv260)"; exit 1; }
[ -e $BUS/devices/$DEV ] || { echo "platform device $DEV not found"; exit 1; }
lsmod | grep -q '^tpu_kv260' || insmod "$KO" "$@"
if [ -e $BUS/devices/$DEV/driver ]; then
    cur=$(basename "$(readlink -f $BUS/devices/$DEV/driver)")
    [ "$cur" = tpu_kv260 ] && { echo "already bound"; exit 0; }
    echo $DEV > $BUS/devices/$DEV/driver/unbind
fi
echo tpu_kv260 > $BUS/devices/$DEV/driver_override
echo $DEV > $BUS/drivers_probe
[ -e /dev/tpu0 ] && echo "/dev/tpu0 ready" || { echo "bind failed, see dmesg"; exit 1; }
