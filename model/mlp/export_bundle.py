#!/usr/bin/env python3
"""Export a quantized MLP as a deployable *bundle* for the Linux driver runtime.

A bundle is everything the user-space runtime (software/runtime/tpu_mlp.py) needs to run one
network at one fixed batch size through /dev/tpu0 ("compile once, run many"):

  model.json     {name, dims, acts, batch, in_features, out_features, in_scale, in_zp, out_scale,
                  out_zp, in_rows, out_rows, input_bytes, weights_bytes, output_bytes, ...}
  program.bin    autonomous Phase B program (READ_HOST_MEM / READ_WEIGHTS / compute /
                 WRITE_HOST_MEM / INTERRUPT_HOST / HALT), 16 bytes per instruction, little
                 endian, UNrelocated (DDR address fields hold nominal values; the kernel patches
                 them from relocs.json)
  relocs.json    [{index, buffer: "input"|"weights"|"output", offset}]   (relocs.txt: same, one
                 "index buffer offset" line each, for the C example)
  weights.bin    Weight Memory image in consumption order (16-byte words; tile order: layer,
                 output tile outer, input tile inner; inside a tile word t = W_tile[15-t])
  sample_in.bin  input-buffer image (UB layout, zero beyond the input rows), ready to copy
  sample_out.bin expected output-buffer bytes for sample_in (TFLite-exact integer pipeline)
  meta.txt       key/value sizes for the C example

Layouts (docs/m4_compute_core.md §6): input row kt*B + b = x[b][16kt:16kt+16] (k-tile major);
output row nt*B + b = y[b][16nt:16nt+16] (n-tile major); lane l of a row = byte l.

The program/weights/sample of the four standard cases are byte-identical to build/m5/cases.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "model/isa"))
sys.path.insert(0, str(ROOT / "model/quantization"))
import asm                                   # noqa: E402
import sim as isasim                         # noqa: E402
import compile_mlp as cm                     # noqa: E402
import compile_phase_b as pb                 # noqa: E402

BUNDLE_VERSION = 1
BASES = {"input": pb.INPUT, "weights": pb.WEIGHTS, "output": pb.OUTPUT}


def pad16(n):
    return (n + 15) // 16 * 16


def quantize_input(net, x):
    """float [B, K] -> int64 [B, pad16(K)] int8 values; padded lanes = input zero point."""
    x = np.asarray(x, dtype=np.float64)
    xq = np.full((x.shape[0], pad16(x.shape[1])), net.in_zp, np.int64)
    xq[:, :x.shape[1]] = cm.quantize(x, net.in_scale, net.in_zp)
    return xq


def _rows_bytes(rows):
    return b"".join(isasim.row_to_int([int(v) for v in r]).to_bytes(16, "little") for r in rows)


def build_bundle(net, name, B, xq, out_dir):
    """Write the bundle for `net` at batch B with sample input xq (int64 [B, pad16(K)])."""
    dims = net.dims
    if xq.shape != (B, pad16(dims[0])):
        raise ValueError(f"sample input must be [{B}, {pad16(dims[0])}], got {xq.shape}")
    yq = net.int_forward(xq)                                       # golden #1 (numpy)
    prog, out_first, ub_end, _acc_rows = cm.emit_program(net, B)
    ww = cm.weight_words(net)
    ub = [[0] * 16 for _ in range(ub_end)]
    for kt in range(pad16(dims[0]) // 16):
        for b in range(B):
            ub[kt * B + b] = [int(v) for v in xq[b, 16 * kt:16 * kt + 16]]
    exp = [[int(v) for v in yq[b, 16 * nt:16 * nt + 16]]
           for nt in range(yq.shape[1] // 16) for b in range(B)]
    program, reloc, consumed = pb.autonomous(prog, len(ub), out_first, len(exp))
    if consumed != len(ww) * 16:
        raise AssertionError("weight tiles consumed != weight image")
    if len(program) > pb.IFIFO_WORDS:
        raise ValueError(f"{name}: program has {len(program)} instructions > {pb.IFIFO_WORDS} (IFIFO)")
    in_bytes, w_bytes, exp_bytes = _rows_bytes(ub), b"".join(w.to_bytes(16, "little") for w in ww), _rows_bytes(exp)
    # golden #2: instruction simulator executing the autonomous program on DDR images
    mem = {BASES["input"] + i: v for i, v in enumerate(in_bytes)}
    mem.update({BASES["weights"] + i: v for i, v in enumerate(w_bytes)})
    m = isasim.CoreSim(enable_dma=True, ddr=mem).run(program)
    got = bytes(m.ddr[BASES["output"] + i] for i in range(len(exp_bytes)))
    if got != exp_bytes or not (m.halted and m.irq_count == 1 and not m.cfg_err):
        raise AssertionError(f"{name}: simulator disagrees with the integer pipeline")
    # relocation list: check every entry against the instruction's nominal address
    relocs = []
    for r in reloc:
        op, f, _, _ = asm.decode(program[r["index"]])
        if op not in (asm.READ_HOST_MEM, asm.READ_WEIGHTS, asm.WRITE_HOST_MEM) or \
                f["ddr_addr4"] * 16 != BASES[r["region"]] + r["offset"]:
            raise AssertionError(f"{name}: relocation {r} inconsistent with the program")
        relocs.append(dict(index=r["index"], buffer=r["region"], offset=r["offset"]))
    dma_idx = {i for i, w in enumerate(program) if (w & 255) in (1, 2, 3)}
    if dma_idx != {r["index"] for r in relocs}:
        raise AssertionError(f"{name}: a DMA instruction has no relocation entry")

    L = net.layers[-1]
    model = dict(
        bundle_version=BUNDLE_VERSION, name=name, dims=list(dims), acts=list(net.acts), batch=B,
        in_features=dims[0], out_features=dims[-1],
        in_scale=float(net.in_scale), in_zp=int(net.in_zp),
        out_scale=float(L["s_out"]), out_zp=int(L["zp_out"]),
        in_rows=pad16(dims[0]) // 16 * B, out_rows=pad16(dims[-1]) // 16 * B,
        input_bytes=len(in_bytes), weights_bytes=len(w_bytes), output_bytes=len(exp_bytes),
        program_instrs=len(program), n_relocs=len(relocs), out_first_row=out_first,
        layout="input row = kt*batch + b; output row = nt*batch + b; lane l = byte l")
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    (d / "program.bin").write_bytes(b"".join(asm.to_bytes(w) for w in program))
    (d / "relocs.json").write_text(json.dumps(relocs, indent=1) + "\n")
    (d / "relocs.txt").write_text("".join(f"{r['index']} {r['buffer']} {r['offset']}\n" for r in relocs))
    (d / "weights.bin").write_bytes(w_bytes)
    (d / "sample_in.bin").write_bytes(in_bytes)
    (d / "sample_out.bin").write_bytes(exp_bytes)
    (d / "meta.txt").write_text("".join(f"{k} {model[k]}\n" for k in (
        "batch", "program_instrs", "n_relocs", "input_bytes", "weights_bytes", "output_bytes")))
    return model, yq


# ---- the four standard cases (same networks / seeds / inputs as compile_mlp.main) -------------
CASES = (("fc1", [16, 16], ["none"], 16, 11, None),
         ("mlp3", [64, 128, 64, 16], ["relu", "relu", "none"], 32, 33, None),
         ("mlp_b1", [64, 128, 64, 16], ["relu", "relu", "none"], 1, 33, "mlp3"),
         ("mlp_wide", [256, 256], ["relu6"], 100, 55, None))


def export_cases(out_root):
    out_root = Path(out_root)
    nets, report = {}, []
    for name, dims, acts, B, seed, share in CASES:
        net = nets[share] if share else cm.Net(dims, acts, seed)
        nets[name] = net
        x = np.random.default_rng(seed + 1).uniform(-1, 1, (B, dims[0]))
        model, _ = build_bundle(net, name, B, quantize_input(net, x), out_root / name)
        same = _compare_m5(out_root / name, name)
        report.append((name, model, same))
    return report


def _compare_m5(bundle, name):
    """Byte-compare against build/m5/cases/<name> when it exists."""
    case = ROOT / "build/m5/cases" / name
    if not (case / "prog.bin").exists():
        return None
    hex_bytes = lambda p: b"".join(int(l, 16).to_bytes(16, "little") for l in p.read_text().split())
    pairs = ((bundle / "program.bin", (case / "prog.bin").read_bytes()),
             (bundle / "weights.bin", hex_bytes(case / "wmem.hex")),
             (bundle / "sample_in.bin", hex_bytes(case / "ub_init.hex")),
             (bundle / "sample_out.bin", hex_bytes(case / "ddr_expect.hex")))
    return all(p.read_bytes() == ref for p, ref in pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "build/bundles"))
    a = ap.parse_args()
    for name, m, same in export_cases(a.out):
        tag = {None: "no m5 case to compare", True: "== build/m5/cases", False: "DIFFERS from build/m5/cases"}[same]
        print(f"{name:9s} batch={m['batch']:3d} instrs={m['program_instrs']:4d} relocs={m['n_relocs']:3d} "
              f"in={m['input_bytes']:6d}B weights={m['weights_bytes']:6d}B out={m['output_bytes']:5d}B  [{tag}]")
        if same is False:
            sys.exit(1)


if __name__ == "__main__":
    main()
