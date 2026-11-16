#!/usr/bin/env python3
"""Integer reference for CONVOLVE (implicit im2col), 2x2 pooling and a TFLite-style int8 conv layer
(docs/m6_cnn_contract.md sections 1, 4, 5, 7).

Pure functions here are used by (a) the unit-vector generator (address generator / pool unit),
(b) model/isa/sim.py (instruction-level execution) and (c) model/cnn/compile_cnn.py.

Layout (contract section 1): a feature map H x W x C, C padded to CT*16, is CT planes of H*W
UB rows: row(ct, y, x) = base + ct*H*W + y*W + x; lane l = channel 16*ct + l.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "quantization"))
from tflite_int8 import requant, quantize_multiplier, act_range    # noqa: E402

N = 16


# ---------------------------------------------------------------------------------------------
# CONVOLVE address sequence (contract section 4)
# ---------------------------------------------------------------------------------------------
def conv_rows(in_w, in_h, out_w, out_h, sh, sw, pad_t, pad_l, kh, kw, oh0, oh_cnt, ub_row):
    """-> list of (pad: bool, row: int) in generation order; row = 0 for padded taps."""
    out = []
    for oh in range(oh0, oh0 + oh_cnt):
        for ow in range(out_w):
            y = oh * sh + kh - pad_t
            x = ow * sw + kw - pad_l
            if 0 <= y < in_h and 0 <= x < in_w:
                out.append((False, ub_row + y * in_w + x))
            else:
                out.append((True, 0))
    return out


def conv_legal(g, oh0, oh_cnt, acc_row, ub_row):
    """Contract section 4 legality. g: dict in_w in_h out_w out_h sh sw pad_t pad_l.
    -> None if legal, else a reason string."""
    if min(g["in_w"], g["in_h"], g["out_w"], g["out_h"], g["sh"], g["sw"]) == 0:
        return "unconfigured/zero geometry"
    if g["pad_t"] > 15 or g["pad_l"] > 15:
        return "pad > 15"
    if oh_cnt == 0 or oh0 + oh_cnt > g["out_h"]:
        return "oh window"
    b = oh_cnt * g["out_w"]
    if b > 2048 or acc_row + b > 2048:
        return "ACC range"
    if ub_row + g["in_h"] * g["in_w"] > 65536:
        return "UB range"
    return None


# ---------------------------------------------------------------------------------------------
# Pooling (contract section 5)
# ---------------------------------------------------------------------------------------------
def pool_lane_max(v):
    return max(v)


def pool_lane_avg(v):
    """TFLite int8 AveragePool, count = 4: round half away from zero, trunc division of s+-2 by 4."""
    s = sum(v)
    return (s + 2) >> 2 if s >= 0 else -(((-s) + 2) >> 2)


def pool_group(rows4, mode):
    """rows4: four int8 rows (lists of 16) in pool order -> one pooled row."""
    f = pool_lane_max if mode == "max" else pool_lane_avg
    return [f([rows4[k][c] for k in range(4)]) for c in range(N)]


def pool_raster(rows, pool_w, mode):
    """rows: list of requantized int8 rows (pre-pool raster, PW wide, PH = len/PW tall) ->
    pooled rows in order i outer, j inner."""
    assert pool_w >= 2 and pool_w % 2 == 0 and len(rows) % (2 * pool_w) == 0
    ph = len(rows) // pool_w
    out = []
    for i in range(ph // 2):
        for j in range(pool_w // 2):
            g = [rows[(2 * i) * pool_w + 2 * j], rows[(2 * i) * pool_w + 2 * j + 1],
                 rows[(2 * i + 1) * pool_w + 2 * j], rows[(2 * i + 1) * pool_w + 2 * j + 1]]
            out.append(pool_group(g, mode))
    return out


def pool_group_order(rows, pool_w):
    """Raster rows -> the same rows regrouped in pool order (4 per output)."""
    ph = len(rows) // pool_w
    out = []
    for i in range(ph // 2):
        for j in range(pool_w // 2):
            out += [rows[(2 * i) * pool_w + 2 * j], rows[(2 * i) * pool_w + 2 * j + 1],
                    rows[(2 * i + 1) * pool_w + 2 * j], rows[(2 * i + 1) * pool_w + 2 * j + 1]]
    return out


# ---------------------------------------------------------------------------------------------
# Integer conv layer (raw sum with folded bias) and the textbook TFLite formulation
# ---------------------------------------------------------------------------------------------
def conv_acc_folded(xq, wq, stride, pad, zp_in):
    """What the hardware computes: raw sum x_q*w_q over all taps, padded taps = zp_in.
    xq: H x W x Cin int; wq: KH x KW x Cin x Cout int. Returns OH x OW x Cout int64 (no bias)."""
    h, w, cin = xq.shape
    kh, kw, _, cout = wq.shape
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    import numpy as np
    acc = np.zeros((oh, ow, cout), np.int64)
    for y in range(oh):
        for x in range(ow):
            for dy in range(kh):
                for dx in range(kw):
                    yy, xx = y * stride + dy - pad, x * stride + dx - pad
                    px = xq[yy, xx, :] if (0 <= yy < h and 0 <= xx < w) else np.full(cin, zp_in)
                    acc[y, x, :] += px.astype(np.int64) @ wq[dy, dx].astype(np.int64)
    return acc


def conv_acc_textbook(xq, wq, bq, stride, pad, zp_in):
    """TFLite reference: sum (x_q - zp_in) * w_q + bias, padded taps skipped."""
    import numpy as np
    h, w, cin = xq.shape
    kh, kw, _, cout = wq.shape
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    acc = np.zeros((oh, ow, cout), np.int64) + bq.astype(np.int64)
    for y in range(oh):
        for x in range(ow):
            for dy in range(kh):
                for dx in range(kw):
                    yy, xx = y * stride + dy - pad, x * stride + dx - pad
                    if 0 <= yy < h and 0 <= xx < w:
                        acc[y, x, :] += (xq[yy, xx, :].astype(np.int64) - zp_in) @ wq[dy, dx].astype(np.int64)
    return acc


if __name__ == "__main__":
    # smoke test of the pure helpers
    r = conv_rows(8, 8, 8, 8, 1, 1, 1, 1, 0, 0, 0, 8, 100)
    assert r[0] == (True, 0) and r[9] == (False, 100) and len(r) == 64
    assert pool_lane_avg([127] * 4) == 127 and pool_lane_avg([-128] * 4) == -128
    assert pool_lane_avg([1, 1, 1, 0]) == 1 and pool_lane_avg([-1, -1, -1, 0]) == -1
    assert pool_lane_avg([1, 0, 0, 0]) == 0 and pool_lane_avg([-1, 0, 0, 0]) == 0
    assert pool_lane_avg([2, 0, 0, 0]) == 1 and pool_lane_avg([-2, 0, 0, 0]) == -1
    print("conv_ref helpers: PASS")
