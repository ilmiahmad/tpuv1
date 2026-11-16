#!/usr/bin/env python3
"""Host-only validation of bundles: run them through tpu.py / tpu_mlp.py on the MOCK driver backed
by the golden simulator (no hardware).  Proves the bundle format, the relocation logic, the
packing and the output layout end to end.

    python3 software/runtime/run_bundle_golden.py [bundle_dir ...]      (default: build/bundles/*)
    python3 software/runtime/run_bundle_golden.py --digits              (also the digits test set)
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import tpu                       # noqa: E402
import tpu_mlp                   # noqa: E402
import tpu_mock                  # noqa: E402

try:
    import numpy as np
except ImportError:              # pragma: no cover
    np = None


def run_bundle(kernel, path):
    with tpu.Device(backend=kernel.backend()) as dev, tpu_mlp.TpuMlp(dev, path) as net:
        res, exact = net.run_sample()
        return net.model, res, exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", nargs="*")
    ap.add_argument("--digits", action="store_true", help="also run the digits test set (needs numpy)")
    a = ap.parse_args()
    paths = [Path(p) for p in a.bundles] or sorted(p.parent for p in (ROOT / "build/bundles").glob("*/model.json"))
    if not paths:
        print("no bundles found; run model/mlp/export_bundle.py and model/mlp/train_demo.py", file=sys.stderr)
        return 2
    kernel = tpu_mock.MockKernel()
    bad = 0
    for p in paths:
        model, res, exact = run_bundle(kernel, p)
        print(f"[{'PASS' if exact else 'FAIL'}] {model['name']:12s} batch={model['batch']:3d} {res}")
        bad += not exact
    if a.digits and np is not None:
        d = ROOT / "build/bundles"
        xq, y, gold = (np.load(d / f"digits_test_{n}.npy") for n in ("xq", "y", "golden"))
        with tpu.Device(backend=kernel.backend()) as dev, tpu_mlp.TpuMlp(dev, d / "digits_b32") as net:
            got = net.infer_q_many(xq[:96])
            exact = bool((got == gold[:96]).all())
            acc = float((got.argmax(1) == y[:96]).mean())
            print(f"[{'PASS' if exact else 'FAIL'}] digits_b32 first 96 test samples bit-exact vs golden; accuracy {acc:.3f}")
            bad += not exact
    print("BUNDLE_GOLDEN_PASS" if not bad else f"BUNDLE_GOLDEN_FAIL ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
