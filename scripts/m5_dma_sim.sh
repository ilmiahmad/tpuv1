#!/usr/bin/env bash
# Real iDMA ND arbitration regression (optional seed arguments).
set -euo pipefail
ulimit -c 0
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
out=build/m5_dma
scripts/m5_dma_prepare.sh
axi_src="$(bender path axi)/src"
verilator --binary --timing --assert -j 2 -Wno-fatal --top-module tb_m5_dma \
  -Mdir "$out/obj" -f "$out/sources.f" \
  "$axi_src/axi_delayer.sv" "$axi_src/axi_sim_mem.sv" tb/idma_standalone/tb_m5_dma.sv \
  > "$out/build.log" 2>&1
if [[ $# = 0 ]]; then set -- 1 7 29; fi
for seed in "$@"; do
  stdbuf -oL "$out/obj/Vtb_m5_dma" "+SEED=$seed" "+verilator+seed+$seed" | tee "$out/seed_$seed.log"
done
verilator --binary --timing --assert -j 2 -Wno-fatal --top-module tb_m5_arbiter \
  -Mdir "$out/arbiter_obj" -f "$out/sources.f" tb/idma_standalone/tb_m5_arbiter.sv \
  > "$out/arbiter_build.log" 2>&1
"$out/arbiter_obj/Vtb_m5_arbiter" | tee "$out/arbiter.log"
