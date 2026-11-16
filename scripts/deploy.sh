#!/usr/bin/env bash
# Deploy build/kv260 (kv260_top) to the KV260 as xmutil app "tpu_kv260" (L6) and run a board test. Other apps are left untouched; restore e.g. with
#   ssh kv260 'sudo xmutil unloadapp; sudo xmutil loadapp ne_vslam_pja'
# Usage: scripts/deploy.sh [board] [test script under software/linux/idma_runtime] [extra test args]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/build/kv260"
board="${1:-kv260}"
test="${2:-m2_board_test.py}"
test_args="${3:-}"
app=tpu_kv260
ref_app=ne_vslam_pja   # known-good overlay: fclk0 100 MHz, afi config-afi (HPM0 128-bit)

[[ -f "$out/tpu_kv260.bit" && -f "$out/build_id.txt" ]] || { echo "build first" >&2; exit 1; }
build_id="$(cat "$out/build_id.txt")"
set +u; source /opt/AMD/2025.2/Vivado/settings64.sh >/dev/null 2>&1 || true; set -u
(cd "$out" && bootgen -image "$root/soc/vivado/scripts/bit_bin.bif" -arch zynqmp -process_bitstream bin -w >/dev/null)
scp -q "$out/tpu_kv260.bit.bin" "$board:/tmp/$app.bit.bin"
scp -q "$root/software/linux/idma_runtime/$test" "$board:/tmp/$test"
if [[ "$test" = m4_board_test.py ]]; then
    [[ -d "$root/build/m4/cases" ]] || { echo "generate M4 cases first" >&2; exit 1; }
    scp -qr "$root/build/m4/cases" "$board:/tmp/"
fi
if [[ "$test" = m5_board_test.py ]]; then
    [[ -d "$root/build/m5/cases" ]] || { echo "generate Phase B cases first" >&2; exit 1; }
    tar -C "$root/build/m5" -czf "$root/build/m5/cases.tar.gz" cases
    scp -q "$root/build/m5/cases.tar.gz" "$board:/tmp/tpu_m5_cases.tar.gz"
    scp -q "$root/software/linux/idma_runtime/m4_board_test.py" "$board:/tmp/m4_board_test.py"
fi

ssh "$board" bash -s <<REMOTE
set -euo pipefail
# A reboot into a new kernel can leave the out-of-tree buffer driver unavailable.
# Fail before reconfiguring PL if the matching module has not been installed.
if [ ! -e /dev/udmabuf0 ]; then
    sudo modprobe u-dma-buf udmabuf0=16777216
fi
test -r /sys/class/u-dma-buf/udmabuf0/phys_addr
if [ "$test" = m5_board_test.py ]; then
    mkdir -p /tmp/tpu_m5
    tar -xzf /tmp/tpu_m5_cases.tar.gz -C /tmp/tpu_m5
fi
src=/lib/firmware/xilinx/$ref_app
dst=/lib/firmware/xilinx/$app
sudo mkdir -p \$dst
sudo cp /tmp/$app.bit.bin \$dst/$app.bit.bin
sudo cp \$src/shell.json \$dst/
dtc -q -I dtb -O dts \$src/$ref_app.dtbo -o /tmp/$app.dts
sed -i 's|$ref_app/$ref_app.bit.bin|$app/$app.bit.bin|' /tmp/$app.dts
grep -q '$app/$app.bit.bin' /tmp/$app.dts
dtc -q -@ -I dts -O dtb /tmp/$app.dts -o /tmp/$app.dtbo
sudo cp /tmp/$app.dtbo \$dst/$app.dtbo
sudo xmutil unloadapp >/dev/null 2>&1 || true
sudo xmutil loadapp $app
state=\$(cat /sys/class/fpga_manager/fpga0/state)
echo "fpga state: \$state"
[ "\$state" = operating ] || exit 2
cd /tmp
sudo python3 -u /tmp/$test --build-id $build_id $test_args
REMOTE
