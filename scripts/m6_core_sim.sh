#!/usr/bin/env bash
# Verilator build + run of tb_tpu_core (tpu_core + tpu_fabric + real memories) on case lists.
#   scripts/m6_core_sim.sh <cases.txt> [more plusargs]     (default EnableDma=0 build, Phase A)
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
out=build/m6/a/core
mkdir -p "$out"
bender script verilator -t rtl -t tpu -t m0_elab -t m3 -t m4 > "$out/sources.f"
if [[ ! -x "$out/obj/Vtb_tpu_core" || tb/core/tb_tpu_core.sv -nt "$out/obj/Vtb_tpu_core" || \
      rtl/control/decoder.sv -nt "$out/obj/Vtb_tpu_core" || rtl/control/matmul_seq.sv -nt "$out/obj/Vtb_tpu_core" || \
      rtl/control/conv_address_gen.sv -nt "$out/obj/Vtb_tpu_core" || rtl/control/act_seq.sv -nt "$out/obj/Vtb_tpu_core" || \
      rtl/top/tpu_core.sv -nt "$out/obj/Vtb_tpu_core" || rtl/activation/pool_unit.sv -nt "$out/obj/Vtb_tpu_core" ]]; then
  verilator --binary --timing --assert -j 2 -Wno-fatal -Wno-lint -Wno-style --top-module tb_tpu_core \
    -Mdir "$out/obj" -f "$out/sources.f" tb/core/tb_tpu_core.sv > "$out/build.log" 2>&1
fi
cases="$1"; shift
"$out/obj/Vtb_tpu_core" +CASES="$cases" "$@"
