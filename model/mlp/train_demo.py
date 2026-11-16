#!/usr/bin/env python3
"""Train a small MLP classifier with numpy, quantize it (TFLite int8) and export TPU bundles.

Dataset (synthetic, no downloads; fixed seeds): 10 classes of 8x8 "glyphs". Each class is a random
binary prototype (pairwise Hamming distance >= 16 of 64 pixels); samples are the prototype shifted
by up to +/-1 pixel in x and y, with brightness jitter, ~10% pixel flips and Gaussian noise (sigma 0.25), so the
classes overlap enough to need a non-linear classifier.

Network 64 -> 128 -> 64 -> 10, ReLU, trained with momentum SGD on cross-entropy.  The float weights
go through Net.from_arrays (calibration on training samples), then bundles for batch sizes
1/8/32/128 are exported to build/bundles/digits_b<N>/ together with the test set:
  digits_test_x.npy (float64 [1000,64])    digits_test_y.npy (int64 [1000])
  digits_test_xq.npy (int8 [1000,64] quantized inputs)
  digits_test_golden.npy (int8 [1000,10] TFLite-exact integer-pipeline outputs)
  digits_report.json  (float vs INT8 accuracy and logit differences)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import compile_mlp as cm                     # noqa: E402
import export_bundle as eb                   # noqa: E402

DIMS = [64, 128, 64, 10]
ACTS = ["relu", "relu", "none"]
BATCHES = (1, 8, 32, 128)


# ---------------------------------------------------------------------------------------------
def make_prototypes(rng, n=10, min_dist=16):
    while True:
        p = (rng.random((n, 8, 8)) < 0.4).astype(np.float64)
        flat = p.reshape(n, 64)
        d = (flat[:, None, :] != flat[None, :, :]).sum(-1) + np.eye(n) * 99
        if d.min() >= min_dist:
            return p


def shift(img, dx, dy):
    out = np.zeros_like(img)
    ys, yd = (slice(0, 8 - dy), slice(dy, 8)) if dy >= 0 else (slice(-dy, 8), slice(0, 8 + dy))
    xs, xd = (slice(0, 8 - dx), slice(dx, 8)) if dx >= 0 else (slice(-dx, 8), slice(0, 8 + dx))
    out[yd, xd] = img[ys, xs]
    return out


def make_dataset(rng, protos, n):
    y = rng.integers(0, len(protos), n)
    x = np.zeros((n, 64))
    for i, c in enumerate(y):
        img = shift(protos[c], int(rng.integers(-1, 2)), int(rng.integers(-1, 2)))
        img = img * rng.uniform(0.5, 1.0)
        flip = rng.random((8, 8)) < 0.10
        img = np.where(flip, 1.0 - img, img) + rng.normal(0, 0.25, (8, 8))
        x[i] = np.clip(img, 0.0, 1.0).reshape(64)
    return x, y


# ---------------------------------------------------------------------------------------------
def forward(Ws, bs, x):
    acts = [x]
    h = x
    for l, (W, b) in enumerate(zip(Ws, bs)):
        h = h @ W + b
        if l < len(Ws) - 1:
            h = np.maximum(h, 0)
        acts.append(h)
    return acts


def train(x, y, epochs, seed):
    rng = np.random.default_rng(seed)
    Ws = [rng.normal(0, np.sqrt(2.0 / DIMS[i]), (DIMS[i], DIMS[i + 1])) for i in range(3)]
    bs = [np.zeros(DIMS[i + 1]) for i in range(3)]
    vW = [np.zeros_like(W) for W in Ws]
    vb = [np.zeros_like(b) for b in bs]
    n = len(x)
    onehot = np.eye(10)[y]
    for ep in range(epochs):
        lr = 0.05 * (0.5 ** (ep // 12))
        perm = rng.permutation(n)
        for s in range(0, n, 64):
            idx = perm[s:s + 64]
            acts = forward(Ws, bs, x[idx])
            z = acts[-1] - acts[-1].max(1, keepdims=True)
            p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
            g = (p - onehot[idx]) / len(idx)
            for l in range(2, -1, -1):
                gW = acts[l].T @ g + 1e-4 * Ws[l]
                gb = g.sum(0)
                if l:
                    g = (g @ Ws[l].T) * (acts[l] > 0)
                vW[l] = 0.9 * vW[l] - lr * gW
                vb[l] = 0.9 * vb[l] - lr * gb
                Ws[l] += vW[l]
                bs[l] += vb[l]
    return Ws, bs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "build/bundles"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--test", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2024)
    protos = make_prototypes(rng)
    xtr, ytr = make_dataset(rng, protos, a.train)
    xte, yte = make_dataset(rng, protos, a.test)
    Ws, bs = train(xtr, ytr, a.epochs, seed=7)

    net = cm.Net.from_arrays(Ws, bs, ACTS, xtr[:512])
    float_logits = forward(Ws, bs, xte)[-1]
    xq = eb.quantize_input(net, xte)                                   # [n, 64] int64 (K already 16-aligned)
    yq = net.int_forward(xq)[:, :10]                                   # golden int8 logits [n, 10]
    L = net.layers[-1]
    deq = (yq - L["zp_out"]) * L["s_out"]
    acc_f = float((float_logits.argmax(1) == yte).mean())
    acc_q = float((yq.argmax(1) == yte).mean())
    agree = float((float_logits.argmax(1) == yq.argmax(1)).mean())
    diff = np.abs(deq - float_logits)
    rng_logits = float(float_logits.max() - float_logits.min())
    report = dict(float_accuracy=acc_f, int8_accuracy=acc_q, argmax_agreement=agree,
                  max_abs_logit_diff=float(diff.max()), mean_abs_logit_diff=float(diff.mean()),
                  logit_range=rng_logits, max_diff_over_range=float(diff.max() / rng_logits),
                  train_accuracy=float((forward(Ws, bs, xtr)[-1].argmax(1) == ytr).mean()),
                  n_train=a.train, n_test=a.test, in_scale=float(net.in_scale), in_zp=int(net.in_zp),
                  out_scale=float(L["s_out"]), out_zp=int(L["zp_out"]))
    np.save(out / "digits_test_x.npy", xte)
    np.save(out / "digits_test_y.npy", yte.astype(np.int64))
    np.save(out / "digits_test_xq.npy", xq[:, :64].astype(np.int8))
    np.save(out / "digits_test_golden.npy", yq.astype(np.int8))
    (out / "digits_report.json").write_text(json.dumps(report, indent=2) + "\n")
    for B in BATCHES:
        model, _ = eb.build_bundle(net, f"digits_b{B}", B, xq[:B], out / f"digits_b{B}")
        print(f"digits_b{B:<3d} instrs={model['program_instrs']:4d} relocs={model['n_relocs']:3d} "
              f"in={model['input_bytes']:6d}B weights={model['weights_bytes']:6d}B out={model['output_bytes']:5d}B")
    print(f"float accuracy {acc_f * 100:.2f}%   INT8 (golden) accuracy {acc_q * 100:.2f}%   "
          f"argmax agreement {agree * 100:.2f}%")
    print(f"logit |float - int8|: max {diff.max():.4f} mean {diff.mean():.4f} (logit range {rng_logits:.2f}, "
          f"max/range {diff.max() / rng_logits * 100:.2f}%)")


if __name__ == "__main__":
    main()
