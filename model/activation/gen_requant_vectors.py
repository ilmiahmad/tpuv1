#!/usr/bin/env python3
"""Requantization test vectors for tb_activation_unit (docs/m4_compute_core.md sections 1-3).

Output: build/m4/requant/ (git-ignored), N = 16, two phases (a parameter re-load between them
checks "a parameter write takes effect for rows that enter after it").

Files per phase k (0, 1), all $readmemh-compatible, one value per line, MSB first, no 0x:
  params_p<k>.hex  one SET_CONFIG write per line, 128 bits = {target[63:0], value[63:0]}:
                   32 hex digits, the first 16 = target (zero-extended 16-bit target),
                   the last 16 = value. Replay them in order through cfg_we/cfg_target/cfg_value
                   BEFORE streaming in_p<k>.hex. All targets are valid (no cfg_err expected).
  in_p<k>.hex      one ACC row per line, N x INT32: channel c = bits [32c+31:32c] (two's compl.)
  qset_p<k>.hex    one hex digit per line: qset (0..3) of the same row
  out_p<k>.hex     expected UB row per line, N x INT8: channel c = bits [8c+7:8c]
Also:
  bad_params.hex   SET_CONFIG writes that MUST pulse cfg_err_o and change nothing
                   (same 128-bit format). Replay after phase 1, then re-stream in_p1.hex and
                   expect out_p1.hex again.
  meta.txt         'key value' lines: N, phases, rows_p0, rows_p1, params_p0, params_p1, bad_params

Parameter encoding (contract section 2):
  target 0x1000|(q<<4)|c : value[31:0] = M0 (int32), value[39:32] = shift (int8, two's compl.)
  target 0x2000|(q<<4)|c : value[31:0] = bias (int32)
  target 0x3000|q        : value[7:0] = zp_out, [15:8] = act_min, [23:16] = act_max (int8)
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quantization"))
from tflite_int8 import requant, INT32_MIN, INT32_MAX   # noqa: E402

N = 16
NSETS = 4


def u(x, bits):
    return x & ((1 << bits) - 1)


def hexw(v, bits):
    return f"{u(v, bits):0{bits // 4}x}"


def pack(vals, w):
    acc = 0
    for i, v in enumerate(vals):
        acc |= u(v, w) << (w * i)
    return acc


def rand_params(rnd, phase):
    sets = []
    for q in range(NSETS):
        m0, sh, bias = [], [], []
        for c in range(N):
            kind = rnd.random()
            if kind < 0.05:
                m0.append(0)
            elif kind < 0.10:
                m0.append(INT32_MAX)
            elif kind < 0.15:
                m0.append(1 << 30)
            else:
                m0.append(rnd.randint(1 << 30, INT32_MAX))
            r = rnd.random()
            if r < 0.08:
                sh.append(-31)
            elif r < 0.16:
                sh.append(7)
            elif r < 0.24:
                sh.append(0)
            else:
                sh.append(rnd.randint(-31, 7))
            b = rnd.random()
            if b < 0.05:
                bias.append(INT32_MAX)            # forces acc+bias wrap for positive acc
            elif b < 0.10:
                bias.append(INT32_MIN)
            elif b < 0.5:
                bias.append(rnd.randint(-(1 << 20), 1 << 20))
            else:
                bias.append(rnd.randint(INT32_MIN, INT32_MAX))
        zp = rnd.randint(-128, 127)
        lo = rnd.randint(-128, 127)
        hi = rnd.randint(lo, 127)
        if q == 0:
            lo, hi = -128, 127                  # set 0: no clamp beyond int8
        if q == 1:
            lo, hi = max(-128, zp), 127         # set 1: ReLU-style
        sets.append(dict(m0=m0, shift=sh, bias=bias, zp_out=zp, act_min=lo, act_max=hi))
    return sets


def param_writes(sets):
    w = []
    for q, p in enumerate(sets):
        for c in range(N):
            w.append((0x1000 | (q << 4) | c, u(p["m0"][c], 32) | (u(p["shift"][c], 8) << 32)))
            w.append((0x2000 | (q << 4) | c, u(p["bias"][c], 32)))
        w.append((0x3000 | q, u(p["zp_out"], 8) | (u(p["act_min"], 8) << 8) | (u(p["act_max"], 8) << 16)))
    return w


def rand_acc(rnd):
    r = rnd.random()
    if r < 0.03:
        return INT32_MIN
    if r < 0.06:
        return INT32_MAX
    if r < 0.10:
        return rnd.choice([0, 1, -1, 2, -2, (1 << 30), -(1 << 30)])
    if r < 0.55:
        return rnd.randint(-(1 << 18), 1 << 18)   # realistic GEMM magnitudes (|y| <= 262144)
    return rnd.randint(INT32_MIN, INT32_MAX)


def gen_rows(rnd, sets, n):
    ins, qs, outs = [], [], []
    for _ in range(n):
        q = rnd.randrange(NSETS)
        row = [rand_acc(rnd) for _ in range(N)]
        p = sets[q]
        out = [requant(a, p["bias"][c], p["m0"][c], p["shift"][c], p["zp_out"], p["act_min"], p["act_max"])
               for c, a in enumerate(row)]
        ins.append(row); qs.append(q); outs.append(out)
    return ins, qs, outs


BAD_TARGETS = [0x1040, 0x1050, 0x10F3, 0x2040, 0x2077, 0x3004, 0x3007, 0x0000, 0x4000, 0x1100,
               0x2100, 0x3010, 0xFFFF, 0x0FFF, 0x5003]
# 0x1040..: q = 4..; 0x1100/0x2100: bit 8 set (q field would be >= 16); 0x3010: bits above q set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/m4/requant")
    ap.add_argument("--rows", type=int, nargs=2, default=[2000, 1000])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rnd = random.Random(4242)
    meta = {"N": N, "phases": 2}
    for k in range(2):
        sets = rand_params(rnd, k)
        writes = param_writes(sets)
        if k == 1:
            writes = writes[::-1]                # different write order, same final state
        ins, qs, outs = gen_rows(rnd, sets, a.rows[k])
        with open(f"{a.out}/params_p{k}.hex", "w") as f:
            for t, v in writes:
                f.write(f"{t:016x}{v:016x}\n")
        with open(f"{a.out}/in_p{k}.hex", "w") as f:
            for r in ins:
                f.write(hexw(pack(r, 32), N * 32) + "\n")
        with open(f"{a.out}/qset_p{k}.hex", "w") as f:
            for q in qs:
                f.write(f"{q:x}\n")
        with open(f"{a.out}/out_p{k}.hex", "w") as f:
            for r in outs:
                f.write(hexw(pack(r, 8), N * 8) + "\n")
        meta[f"rows_p{k}"] = len(ins)
        meta[f"params_p{k}"] = len(writes)
    with open(f"{a.out}/bad_params.hex", "w") as f:
        for i, t in enumerate(BAD_TARGETS):
            f.write(f"{t:016x}{(0x5A5A5A5A00000000 + i):016x}\n")
    meta["bad_params"] = len(BAD_TARGETS)
    with open(f"{a.out}/meta.txt", "w") as f:
        for kk, vv in meta.items():
            f.write(f"{kk} {vv}\n")
    with open(f"{a.out}/README.txt", "w") as f:
        f.write(__doc__)
    print(f"requant vectors -> {a.out}: " + ", ".join(f"{k}={v}" for k, v in meta.items()))


if __name__ == "__main__":
    main()
