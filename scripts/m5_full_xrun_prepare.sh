#!/usr/bin/env bash
# Prepare the portable Xcelium file list for the Phase B full-system test (tb_m5_full_system).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
mkdir -p build/m5/full
python3 model/mlp/compile_phase_b.py >/dev/null
bender script flist-plus -t rtl -t kv260 -t m0_elab -t tpu -t m3 -t m4 > build/m5/full/xrun.f
python3 - "$(bender path axi)" <<'PY'
from pathlib import Path
import sys
root = Path.cwd()
manifest = root / 'build/m5/full/xrun.f'
text = manifest.read_text().replace(str(root) + '/', './')
axi = Path(sys.argv[1]).resolve().relative_to(root)
files = [axi/'src/axi_delayer.sv', axi/'src/axi_sim_mem.sv', Path('tb/system/kv260_ps_sim.sv')]
manifest.write_text(text + '\n' + '\n'.join(map(str, files)) + '\n')
PY
