#!/usr/bin/env bash
# Exercise the real board top, including HPM ID/data conversion and AXI-Lite register permissions.
set -euo pipefail
ulimit -c 0
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
out=build/m4_board_regs
mkdir -p "$out"
bender script verilator -t rtl -t kv260 -t m0_elab -t tpu -t m3 -t m4 > "$out/sources.f"
verilator --binary --timing --assert -j 2 -Wno-fatal --top-module tb_m4_board_regs \
  -Mdir "$out/obj" -f "$out/sources.f" tb/system/kv260_ps_sim.sv > "$out/build.log" 2>&1
"$out/obj/Vtb_m4_board_regs"
