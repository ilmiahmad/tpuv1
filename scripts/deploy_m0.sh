#!/usr/bin/env bash
# Deploy the M0 bitstream to the KV260 as its own xmutil app "tpu_kv260" (L6), then run the
# on-board gate test. Other apps are left untouched; restore e.g. with
#   ssh kv260 'sudo xmutil unloadapp; sudo xmutil loadapp ne_vslam_pja'
# Usage: scripts/deploy_m0.sh [board]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/build/m0_board"
board="${1:-kv260}"
app=tpu_kv260
ref_app=ne_vslam_pja   # known-good overlay: fclk0 100 MHz, afi config-afi (HPM0 128-bit)

[[ -f "$out/tpu_kv260.bit" && -f "$out/build_id.txt" ]] || { echo "build first" >&2; exit 1; }
build_id="$(cat "$out/build_id.txt")"
set +u; source /opt/AMD/2025.2/Vivado/settings64.sh >/dev/null 2>&1 || true; set -u
(cd "$out" && bootgen -image "$root/soc/vivado/scripts/bit_bin.bif" -arch zynqmp -process_bitstream bin -w >/dev/null)
scp -q "$out/tpu_kv260.bit.bin" "$board:/tmp/$app.bit.bin"
scp -q "$root/software/linux/idma_runtime/m0_board_test.py" "$board:/tmp/m0_board_test.py"

ssh "$board" bash -s <<REMOTE
set -euo pipefail
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
sudo python3 /tmp/m0_board_test.py --build-id $build_id
REMOTE
