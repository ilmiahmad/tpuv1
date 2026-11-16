#!/usr/bin/env bash
# Prepare portable simulation lists and the local Vivado source list.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
mkdir -p build/m5_dma
bender script verilator -t rtl -t m0_elab > build/m5_dma/sources.f
bender script flist-plus -t rtl -t m0_elab > build/m5_dma/xrun.f
bender script vivado -t rtl -t m0_elab -t m5_elab > build/m5_dma/sources.tcl
python3 - "$(bender path axi)" <<'PY'
from pathlib import Path
import sys
root = Path.cwd()
manifest = root / 'build/m5_dma/xrun.f'
text = manifest.read_text().replace(str(root) + '/', './')
axi = Path(sys.argv[1]).resolve().relative_to(root)
files = [axi/'src/axi_delayer.sv', axi/'src/axi_sim_mem.sv',
         Path('tb/idma_standalone/tb_m5_dma.sv'), Path('tb/idma_standalone/tb_m5_arbiter.sv')]
manifest.write_text(text + '\n' + '\n'.join(map(str, files)) + '\n')
PY
