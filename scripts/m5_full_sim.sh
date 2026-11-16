#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
out=build/m5/full
mkdir -p "$out"
python3 model/mlp/compile_phase_b.py
bender script verilator -t rtl -t kv260 -t m0_elab -t tpu -t m3 -t m4 > "$out/sources.f"
axi_src="$(bender path axi)/src"
verilator --binary --timing --assert -j 2 -Wno-fatal --top-module tb_m5_full_system \
  +define+TPU_PHASE_B_SIM +incdir+tb/system -Mdir "$out/obj" -f "$out/sources.f" \
  "$axi_src/axi_delayer.sv" "$axi_src/axi_sim_mem.sv" tb/system/kv260_ps_sim.sv > "$out/build.log" 2>&1
stdbuf -oL "$out/obj/Vtb_m5_full_system" "$@" | tee "$out/run.log"
