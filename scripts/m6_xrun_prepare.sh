#!/usr/bin/env bash
# Portable Xcelium file lists for the CNN conv/decoder verification (relative paths).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
mkdir -p build/m6/a/xr
mk() {   # mk <out> <bender targets...>
  local out="$1"; shift
  bender script flist-plus "$@" > "$out"
  python3 - "$out" "$root" <<'PY'
import sys; from pathlib import Path
p=Path(sys.argv[1]); p.write_text(p.read_text().replace(sys.argv[2]+'/','./'))
PY
}
mk build/m6/a/xr/core.f -t rtl -t tpu -t m0_elab -t m3 -t m4
mk build/m6/a/xr/full.f -t rtl -t kv260 -t m0_elab -t tpu -t m3 -t m4
axi="$(bender path axi)"; axi="${axi#"$root"/}"
printf '%s\n%s\ntb/system/kv260_ps_sim.sv\n' "$axi/src/axi_delayer.sv" "$axi/src/axi_sim_mem.sv" >> build/m6/a/xr/full.f
