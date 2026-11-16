#!/usr/bin/env python3
"""CNN compiler + TFLite-style integer reference (docs/m6_cnn_contract.md sections 1-7).

Writes Phase A cases to build/m6/cases/<name>/ with exactly the same file set as compile_mlp.py
(prog.hex/.bin/.lst, wmem.hex, ub_init.hex, ub_expect.hex, acc_expect.hex, meta.txt, cases.txt).
Phase B (autonomous) versions are made from them by `model/mlp/compile_phase_b.py --cnn`.

Networks: conv (k x k, stride, symmetric pad) + optional ReLU/ReLU6 + optional 2x2 max/avg pool,
then optional FC. Batch = 1 image. Quantization follows TFLite int8: asymmetric activations,
per-output-channel symmetric weights (zp 0, [-127,127]), int32 bias, QuantizeMultiplier,
CalculateActivationRangeQuantized. The hardware computes the raw sum x_q*w_q over ALL taps with
padded taps = zp_in, so the input zero-point term -zp_in*sum(w_q) is folded into the bias for
every tap (asserted equal to the textbook "skip padded taps" pipeline).

Layout (contract section 1): planar by channel tile, row = base + ct*H*W + y*W + x.
Weight tiles: nt outer, then taps (kh, kw) and input channel tile ct innermost; tile word t =
W_tile[15-t]; W_tile[k][n] = W[kh][kw][16ct+k][16nt+n]. Per tap, all output-row windows are issued
back to back, with new_weights = 1 only on the first window (one tile per tap).
FC after conv: input rows are the planar feature map (kt = ct*P + p, P pixels), B = 1.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "quantization"))
sys.path.insert(0, os.path.join(HERE, "..", "isa"))
sys.path.insert(0, os.path.join(HERE, "..", "mlp"))
sys.path.insert(0, HERE)
import asm                                                              # noqa: E402
import sim as isasim                                                    # noqa: E402
import conv_ref as cr                                                   # noqa: E402
import compile_mlp as mlp                                               # noqa: E402
from tflite_int8 import quantize_multiplier, act_range, requant         # noqa: E402

N = 16
ACC_HALF = 1024
U = asm
pad16 = mlp.pad16
quant_act = mlp.quant_act
quantize = mlp.quantize


# ---------------------------------------------------------------------------------------------
# float network
# ---------------------------------------------------------------------------------------------
def fconv(x, W, b, stride, pad):
    h, w, _ = x.shape
    kh, kw, _, cout = W.shape
    oh, ow = (h + 2 * pad - kh) // stride + 1, (w + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((pad, pad), (pad, pad), (0, 0)))
    y = np.zeros((oh, ow, cout))
    for dy in range(kh):
        for dx in range(kw):
            y += xp[dy:dy + stride * oh:stride, dx:dx + stride * ow:stride, :] @ W[dy, dx]
    return y + b


def fpool(x, mode):
    h, w, c = x.shape
    t = x.reshape(h // 2, 2, w // 2, 2, c)
    return t.max(axis=(1, 3)) if mode == "max" else t.mean(axis=(1, 3))


def fact(x, act):
    return np.maximum(x, 0) if act == "relu" else (np.clip(x, 0, 6) if act == "relu6" else x)


def np_pool_int(x, mode):
    """Independent integer pooling (numpy reshape), TFLite int8: avg = round half away from zero."""
    h, w, c = x.shape
    t = x.reshape(h // 2, 2, w // 2, 2, c).astype(np.int64)
    if mode == "max":
        return t.max(axis=(1, 3))
    s = t.sum(axis=(1, 3))
    return np.where(s >= 0, (s + 2) // 4, -((-s + 2) // 4))


# ---------------------------------------------------------------------------------------------
class Net:
    """spec: dict(h, w, c, lo, hi, layers=[dict(kind='conv'|'fc', ...)]); builds float weights, the
    calibration, and the TFLite int8 parameters of every layer."""

    def __init__(self, spec, seed, bcal=6):
        rng = np.random.default_rng(seed)
        self.spec, self.layers = spec, []
        h, w, c = spec["h"], spec["w"], spec["c"]
        self.in_scale, self.in_zp = quant_act(None, spec["lo"], spec["hi"])
        self.xcal = [rng.uniform(spec["lo"], spec["hi"], (h, w, c)) for _ in range(bcal)]
        feats = list(self.xcal)
        s_in, zp_in = self.in_scale, self.in_zp
        for li, ls in enumerate(spec["layers"]):
            L = dict(ls)
            if L["kind"] == "conv":
                k, cout = L["k"], L["cout"]
                L["cin"], L["ih"], L["iw"] = c, h, w
                L["Wf"] = rng.normal(0, 1.0 / np.sqrt(k * k * c), (k, k, c, cout))
                L["bf"] = rng.normal(0, 0.1, cout)
                outs = [fact(fconv(f, L["Wf"], L["bf"], L["stride"], L["pad"]), L["act"]) for f in feats]
                L["oh"], L["ow"] = outs[0].shape[0], outs[0].shape[1]
                if L.get("pool"):
                    assert L["oh"] % 2 == 0 and L["ow"] % 2 == 0, "pool needs even output dims"
                    feats = [fpool(o, L["pool"]) for o in outs]
                    h, w = L["oh"] // 2, L["ow"] // 2
                else:
                    feats = outs
                    h, w = L["oh"], L["ow"]
                c = cout
                wmat = L["Wf"]                                   # K x K x cin x cout
                lo = min(o.min() for o in outs); hi = max(o.max() for o in outs)
                red_axes = (0, 1, 2)
            else:  # fc on the flattened HWC feature map
                nf = h * w * c
                L["nfeat"], L["fh"], L["fw"], L["fc_c"] = nf, h, w, c
                L["Wf"] = rng.normal(0, 1.0 / np.sqrt(nf), (nf, L["nout"]))
                L["bf"] = rng.normal(0, 0.1, L["nout"])
                outs = [fact(f.reshape(-1) @ L["Wf"] + L["bf"], L["act"]) for f in feats]
                feats = outs
                lo = min(o.min() for o in outs); hi = max(o.max() for o in outs)
                c = L["nout"]
                red_axes = (0,)
            s_out, zp_out = quant_act(None, float(lo), float(hi))
            s_w = np.maximum(np.abs(L["Wf"]).max(axis=red_axes), 1e-8) / 127.0
            wq = np.clip(np.round(L["Wf"] / s_w), -127, 127).astype(np.int64)
            bq = np.round(L["bf"] / (s_in * s_w)).astype(np.int64)
            m0, sh = zip(*[quantize_multiplier(float(s_in * s_w[ch] / s_out)) for ch in range(len(s_w))])
            amin, amax = act_range(L["act"], s_out, zp_out)
            L.update(wq=wq, bq=bq, bhw=bq - zp_in * wq.sum(axis=red_axes), m0=list(m0), shift=list(sh),
                     s_in=s_in, zp_in=zp_in, s_out=s_out, zp_out=zp_out, act_min=amin, act_max=amax)
            self.layers.append(L)
            s_in, zp_in = s_out, zp_out          # pooling keeps scale / zero point (TFLite)

    def float_forward(self, x):
        for L in self.layers:
            if L["kind"] == "conv":
                x = fact(fconv(x, L["Wf"], L["bf"], L["stride"], L["pad"]), L["act"])
                if L.get("pool"):
                    x = fpool(x, L["pool"])
            else:
                x = fact(x.reshape(-1) @ L["Wf"] + L["bf"], L["act"])
        return x

    # ---- golden #1: numpy integer pipeline (independent of sim.py / conv_rows / pool_raster) ----
    def int_forward(self, xq):
        """xq: H x W x pad16(c) int8 image (padded lanes arbitrary). Returns the final tensor
        (H x W x padded channels for conv, padded vector for fc) and per-layer outputs."""
        x = xq.astype(np.int64)
        outs = []
        for L in self.layers:
            if L["kind"] == "conv":
                cin, cout = L["cin"], L["cout"]
                cpi, cpo = pad16(cin), pad16(cout)
                wpad = np.zeros((L["k"], L["k"], cpi, cpo), np.int64)
                wpad[:, :, :cin, :cout] = L["wq"]
                acc = cr.conv_acc_folded(x[:, :, :cpi], wpad, L["stride"], L["pad"], L["zp_in"])
                # textbook TFLite (skip padded taps, subtract zp, add the original bias) must agree
                tb = cr.conv_acc_textbook(x[:, :, :cin], L["wq"], L["bq"], L["stride"], L["pad"], L["zp_in"])
                assert np.array_equal(acc[:, :, :cout] + L["bhw"], tb), "folded bias != textbook conv"
                assert np.abs(acc).max() < 2 ** 31
                y = np.zeros_like(acc)
                for ch in range(cpo):
                    bias, m0, sh = (int(L["bhw"][ch]), L["m0"][ch], L["shift"][ch]) if ch < cout else (0, 0, 0)
                    for yy in range(acc.shape[0]):
                        for xx in range(acc.shape[1]):
                            y[yy, xx, ch] = requant(int(acc[yy, xx, ch]), bias, m0, sh, L["zp_out"],
                                                    L["act_min"], L["act_max"])
                if L.get("pool"):
                    y = np_pool_int(y, L["pool"])
                x = y
            else:
                feats = x[:, :, :L["fc_c"]].reshape(-1)
                nout = L["nout"]
                acc = feats @ L["wq"]
                out = np.zeros(pad16(nout), np.int64)
                for ch in range(pad16(nout)):
                    bias, m0, sh = (int(L["bhw"][ch]), L["m0"][ch], L["shift"][ch]) if ch < nout else (0, 0, 0)
                    a = int(acc[ch]) if ch < nout else 0
                    out[ch] = requant(a, bias, m0, sh, L["zp_out"], L["act_min"], L["act_max"])
                x = out
            outs.append(x)
        return x, outs


# ---------------------------------------------------------------------------------------------
# weights and program
# ---------------------------------------------------------------------------------------------
def conv_windows(L):
    ws = L.get("windows") or [(0, L["oh"])]
    assert sum(c for _, c in ws) == L["oh"] and ws[0][0] == 0
    assert all(ws[i][0] + ws[i][1] == ws[i + 1][0] for i in range(len(ws) - 1))
    assert sum(c * L["ow"] for _, c in ws) <= ACC_HALF, "output tile does not fit an ACC half"
    if L.get("pool"):
        assert all(o % 2 == 0 and c % 2 == 0 for o, c in ws), "pool windows must be even-aligned"
    return ws


def tile_words(tile):
    """16 x 16 W_tile[k][n] -> words, word t = W_tile[15-t] (byte n = column n)."""
    return [isasim.row_to_int([int(v) for v in tile[15 - t]]) for t in range(16)]


def weight_words(net):
    words = []
    for L in net.layers:
        if L["kind"] == "conv":
            cpi, cpo = pad16(L["cin"]), pad16(L["cout"])
            W = np.zeros((L["k"], L["k"], cpi, cpo), np.int64)
            W[:, :, :L["cin"], :L["cout"]] = L["wq"]
            for nt in range(cpo // 16):
                for kh in range(L["k"]):
                    for kw in range(L["k"]):
                        for ct in range(cpi // 16):
                            words += tile_words(W[kh, kw, 16 * ct:16 * ct + 16, 16 * nt:16 * nt + 16])
        else:
            fh, fw, c = L["fh"], L["fw"], L["fc_c"]
            ct_n, P = pad16(c) // 16, fh * fw
            Kp = ct_n * P * 16                              # planar K: kt = ct*P + p, lane l
            Wp = np.zeros((Kp, pad16(L["nout"])), np.int64)
            for ct in range(ct_n):
                for p in range(P):
                    for l in range(16):
                        ch = 16 * ct + l
                        if ch < c:
                            Wp[(ct * P + p) * 16 + l, :L["nout"]] = L["wq"][p * c + ch, :]
            for nt in range(pad16(L["nout"]) // 16):
                for kt in range(ct_n * P):
                    words += tile_words(Wp[16 * kt:16 * kt + 16, 16 * nt:16 * nt + 16])
    return words


def layout(net):
    """UB allocation: planar feature maps. Returns (per-layer dicts with in_base/out_base, ub_end)."""
    spec = net.spec
    ct_in = pad16(spec["c"]) // 16
    base = 0
    cur = dict(base=0, h=spec["h"], w=spec["w"], ct=ct_in)
    nxt = ct_in * spec["h"] * spec["w"]
    infos = []
    for L in net.layers:
        if L["kind"] == "conv":
            ph, pw = (L["oh"] // 2, L["ow"] // 2) if L.get("pool") else (L["oh"], L["ow"])
            nt = pad16(L["cout"]) // 16
            infos.append(dict(in_base=cur["base"], out_base=nxt, ph=ph, pw=pw, nt=nt, ct_in=cur["ct"]))
            cur = dict(base=nxt, h=ph, w=pw, ct=nt)
            nxt += nt * ph * pw
        else:
            nt = pad16(L["nout"]) // 16
            infos.append(dict(in_base=cur["base"], out_base=nxt, ph=1, pw=1, nt=nt, ct_in=cur["ct"]))
            cur = dict(base=nxt, h=1, w=1, ct=nt)
            nxt += nt
    return infos, nxt


def emit_program(net, tag=0xD0E0_0006):
    infos, ub_end = layout(net)
    prog = []
    tile_idx = 0
    qset_busy, region_busy, act_pending = {}, {0: False, 1: False}, False
    FUNC = {"none": 0, "relu": 1, "relu6": 2}

    def drain():
        nonlocal qset_busy, region_busy, act_pending
        prog.append(U.sync(U.U_ACT))
        qset_busy, region_busy, act_pending = {}, {0: False, 1: False}, False

    def requant_cfg(L, nt, q):
        n_real = L["cout"] if L["kind"] == "conv" else L["nout"]
        for c in range(16):
            ch = 16 * nt + c
            m0, sh, bias = (L["m0"][ch], L["shift"][ch], int(L["bhw"][ch])) if ch < n_real else (0, 0, 0)
            prog.append(U.set_config(0x1000 | (q << 4) | c, (m0 & 0xFFFFFFFF) | ((sh & 0xFF) << 32)))
            prog.append(U.set_config(0x2000 | (q << 4) | c, bias & 0xFFFFFFFF))
        prog.append(U.set_config(0x3000 | q, (L["zp_out"] & 0xFF) | ((L["act_min"] & 0xFF) << 8)
                                 | ((L["act_max"] & 0xFF) << 16)))

    for L, I in zip(net.layers, infos):
        if act_pending:
            drain()                                    # UB RAW (next layer reads what ACTIVATE wrote)
        if L["kind"] == "conv":
            ws = conv_windows(L)
            prog += [U.set_conv_in(L["iw"], L["ih"]), U.set_conv_out(L["ow"], L["oh"]),
                     U.set_conv_misc(L["stride"], L["stride"], L["pad"], L["pad"], L["zp_in"])]
            if L.get("pool"):
                prog.append(U.set_pool_w(L["ow"]))
            offs, run = [], 0
            for _, cnt in ws:
                offs.append(run); run += cnt * L["ow"]
            for nt in range(I["nt"]):
                q, half = tile_idx % 4, tile_idx % 2
                if qset_busy.get(q) or region_busy[half]:
                    drain()
                requant_cfg(L, nt, q)
                a0 = half * ACC_HALF
                tap = 0
                for kh in range(L["k"]):
                    for kw in range(L["k"]):
                        for ct in range(I["ct_in"]):
                            for wi, (oh0, cnt) in enumerate(ws):
                                prog.append(U.convolve(I["in_base"] + ct * L["ih"] * L["iw"], a0 + offs[wi],
                                                       kh, kw, oh0, cnt, acc=1 if tap > 0 else 0,
                                                       new_weights=1 if wi == 0 else 0))
                            tap += 1
                prog.append(U.sync(U.U_MATRIX))
                for wi, (oh0, cnt) in enumerate(ws):
                    if L.get("pool"):
                        row = I["out_base"] + nt * I["ph"] * I["pw"] + (oh0 // 2) * I["pw"]
                    else:
                        row = I["out_base"] + nt * L["oh"] * L["ow"] + oh0 * L["ow"]
                    prog.append(U.activate(a0 + offs[wi], row, cnt * L["ow"], q, func=FUNC[L["act"]],
                                           pool={None: 0, "max": 1, "avg": 2}[L.get("pool")]))
                qset_busy[q], region_busy[half], act_pending = True, True, True
                tile_idx += 1
        else:
            for nt in range(I["nt"]):
                q, half = tile_idx % 4, tile_idx % 2
                if qset_busy.get(q) or region_busy[half]:
                    drain()
                requant_cfg(L, nt, q)
                a0 = half * ACC_HALF
                for kt in range(I["ct_in"] * L["fh"] * L["fw"]):
                    prog.append(U.matmul(I["in_base"] + kt, a0, 1, acc=1 if kt > 0 else 0, new_weights=1))
                prog.append(U.sync(U.U_MATRIX))
                prog.append(U.activate(a0, I["out_base"] + nt, 1, q, func=FUNC[L["act"]]))
                qset_busy[q], region_busy[half], act_pending = True, True, True
                tile_idx += 1
    prog += [U.sync(U.U_ALL), U.interrupt_host(tag), U.halt()]
    return prog, infos, ub_end


def input_rows(net, xq, ub_end):
    """UB image: input planes (lane l of plane ct = channel 16ct+l), zeros elsewhere."""
    h, w = net.spec["h"], net.spec["w"]
    ub = [[0] * N for _ in range(ub_end)]
    for ct in range(pad16(net.spec["c"]) // 16):
        for y in range(h):
            for x in range(w):
                ub[ct * h * w + y * w + x] = [int(v) for v in xq[y, x, 16 * ct:16 * ct + 16]]
    return ub


def expected_rows(net, infos, final):
    """Final tensor -> UB rows of the output region (planar)."""
    I, L = infos[-1], net.layers[-1]
    rows = []
    if L["kind"] == "conv":
        for nt in range(I["nt"]):
            for y in range(I["ph"]):
                for x in range(I["pw"]):
                    rows.append([int(v) for v in final[y, x, 16 * nt:16 * nt + 16]])
    else:
        for nt in range(I["nt"]):
            rows.append([int(v) for v in final[16 * nt:16 * nt + 16]])
    return rows


# ---------------------------------------------------------------------------------------------
def build_cnn_case(outdir, name, spec, seed):
    net = Net(spec, seed)
    rng = np.random.default_rng(seed + 1)
    x = rng.uniform(spec["lo"], spec["hi"], (spec["h"], spec["w"], spec["c"]))
    cp = pad16(spec["c"])
    xq = np.full((spec["h"], spec["w"], cp), net.in_zp, np.int64)     # padded lanes: zp (weights are 0)
    xq[:, :, :spec["c"]] = quantize(x, net.in_scale, net.in_zp)
    final, _ = net.int_forward(xq)                                    # golden #1 (numpy)
    prog, infos, ub_end = emit_program(net)
    ww = weight_words(net)
    ub = input_rows(net, xq, ub_end)
    exp = expected_rows(net, infos, final)
    s = isasim.CoreSim(ub, ww, check_hazards=True).run(prog)          # golden #2 (instruction simulator)
    out_first = infos[-1]["out_base"]
    got = [s.ub.get(out_first + i) for i in range(len(exp))]
    assert got == exp, f"{name}: sim.py disagrees with the integer pipeline"
    assert s.wpos == len(ww) and s.irq_count == 1 and not s.cfg_err, (name, s.wpos, len(ww), s.cfg_err_why if s.cfg_err else "")
    # float sanity
    yf = net.float_forward(x)
    L = net.layers[-1]
    if L["kind"] == "conv":
        nco = L["cout"]
        deq = (final[:, :, :nco] - L["zp_out"]) * L["s_out"]
    else:
        deq = (final[:L["nout"]] - L["zp_out"]) * L["s_out"]
    err = float(np.abs(deq - yf).max() / max(1e-9, np.abs(yf).max()))
    acc_rows = max(s.acc) + 1
    layers = "|".join(
        (f"conv{l['k']}x{l['k']}s{l['stride']}p{l['pad']}-{l['cin']}to{l['cout']}" + (f"-{l['pool']}pool" if l.get("pool") else ""))
        if l["kind"] == "conv" else f"fc{l['nfeat']}to{l['nout']}" for l in net.layers)
    d = mlp.write_case(outdir, name, prog, ww, ub, out_first, exp, acc_rows, s.acc,
                       dict(kind="cnn", layers=layers, in_zp=net.in_zp, out_cols=pad16(L.get("cout", L.get("nout"))),
                            irq_expected=1, irq_tag=f"{s.irq_tag:08x}", cfg_err=0, cfg_err_op="00",
                            float_rel_err=f"{err:.4f}"))
    return dict(name=name, instrs=len(prog), weights=len(ww), ub_rows=len(ub), out_rows=len(exp),
                zp_in=net.in_zp, err=err, path=d)


def C(cout, k=3, stride=1, pad=1, act="relu", pool=None, windows=None):
    return dict(kind="conv", cout=cout, k=k, stride=stride, pad=pad, act=act, pool=pool, windows=windows)


def F(nout, act="none"):
    return dict(kind="fc", nout=nout, act=act)


# name -> (spec, seed). lo/hi choose the input zero point: (0,1) -> zp -128, (-1,0) -> zp 127
CASES = {
    "conv3x3_s1p1":     (dict(h=8, w=8, c=16, lo=-0.5, hi=1.0, layers=[C(16)]), 101),
    "conv3x3_s2p1":     (dict(h=8, w=8, c=16, lo=-0.7, hi=0.4, layers=[C(16, stride=2, act="none")]), 102),
    "conv3x3_valid":    (dict(h=8, w=8, c=16, lo=-0.2, hi=0.9, layers=[C(16, pad=0, act="relu6")]), 103),
    "conv3x3_ct2_co32": (dict(h=6, w=6, c=32, lo=-0.4, hi=0.8, layers=[C(32)]), 104),
    "conv_nonsquare":   (dict(h=6, w=10, c=16, lo=-0.6, hi=0.3, layers=[C(16, act="none")]), 105),
    "conv1x1":          (dict(h=8, w=8, c=16, lo=-0.3, hi=0.7, layers=[C(16, k=1, pad=0)]), 106),
    "conv5x5_p2":       (dict(h=8, w=8, c=16, lo=-0.8, hi=0.5, layers=[C(16, k=5, pad=2)]), 107),
    "conv_split":       (dict(h=16, w=16, c=16, lo=0.0, hi=1.0,
                              layers=[C(16, windows=[(0, 4), (4, 4), (8, 4), (12, 4)])]), 108),
    "conv_pool_max":    (dict(h=8, w=8, c=16, lo=-0.5, hi=1.0, layers=[C(16, pool="max")]), 109),
    "conv_pool_avg":    (dict(h=8, w=8, c=16, lo=-1.0, hi=0.0, layers=[C(16, pool="avg")]), 110),
    "cnn_tiny":         (dict(h=12, w=12, c=3, lo=-0.5, hi=1.0,
                              layers=[C(16, pool="max"), C(32, pool="max"), F(10)]), 111),
    "cnn_tiny_avg":     (dict(h=12, w=12, c=3, lo=-1.0, hi=0.5,
                              layers=[C(16, pool="avg"), C(32, pool="avg"), F(10)]), 112),
}


# ---------------------------------------------------------------------------------------------
# cfgerr cases: valid fc1-style prefix, optional valid setup, ONE illegal instruction, marker
# ACTIVATE that must NOT run, INTERRUPT_HOST, HALT (same structure as compile_mlp.cfgerr_cases)
# ---------------------------------------------------------------------------------------------
def cnn_cfgerr_cases(outdir):
    net = mlp.Net([16, 16], ["none"], seed=900)
    B = 16
    rng = np.random.default_rng(901)
    xq = quantize(rng.uniform(-1, 1, (B, 16)), net.in_scale, net.in_zp)
    L = net.layers[0]
    prefix = []
    for c in range(16):
        prefix.append(U.set_config(0x1000 | c, (L["m0"][c] & 0xFFFFFFFF) | ((L["shift"][c] & 0xFF) << 32)))
        prefix.append(U.set_config(0x2000 | c, int(L["bhw"][c]) & 0xFFFFFFFF))
    prefix.append(U.set_config(0x3000, (L["zp_out"] & 0xFF) | ((L["act_min"] & 0xFF) << 8) | ((L["act_max"] & 0xFF) << 16)))
    prefix += [U.matmul(0, 0, B, 0, 1), U.sync(U.U_MATRIX), U.activate(0, 16, B, 0), U.sync(U.U_ALL)]
    marker = [U.activate(0, 32, B, 0), U.interrupt_host(0xBAD0_0006), U.halt()]
    ww = mlp.weight_words(net)

    def geom(in_w=8, in_h=8, out_w=8, out_h=8, sh=1, sw=1, pt=1, pl=1, pv=-3):
        return [U.set_conv_in(in_w, in_h), U.set_conv_out(out_w, out_h), U.set_conv_misc(sh, sw, pt, pl, pv)]

    def cv(ub_row=0, acc_row=0, kh=1, kw=1, oh0=0, cnt=8):
        return U.convolve(ub_row, acc_row, kh, kw, oh0, cnt)

    good = cv()
    pool_ok = [U.set_pool_w(4)]

    def act(**kw):
        d = dict(acc_row=0, ub_row=48, rows=8, qset=0, pool=1)
        d.update(kw)
        return U.encode(U.ACTIVATE, **d)

    cases = {
        # CONVOLVE legality (contract section 4)
        "cfgerr_conv_unconfigured": ([], cv(cnt=1)),
        "cfgerr_conv_in_w0":    (geom(in_w=0), good),
        "cfgerr_conv_in_h0":    (geom(in_h=0), good),
        "cfgerr_conv_out_w0":   (geom(out_w=0), good),
        "cfgerr_conv_out_h0":   (geom(out_h=0), good),
        "cfgerr_conv_sh0":      (geom(sh=0), good),
        "cfgerr_conv_sw0":      (geom(sw=0), good),
        "cfgerr_conv_pad_t16":  (geom(pt=16), good),
        "cfgerr_conv_pad_l16":  (geom(pl=16), good),
        "cfgerr_conv_ohcnt0":   (geom(), cv(cnt=0)),
        "cfgerr_conv_oh_range": (geom(), cv(oh0=5, cnt=4)),
        "cfgerr_conv_oh0_out":  (geom(), cv(oh0=8, cnt=1)),
        "cfgerr_conv_b_range":  (geom(out_w=64, out_h=64), cv(cnt=33)),
        "cfgerr_conv_acc_range": (geom(), cv(acc_row=2047, cnt=1)),
        "cfgerr_conv_ub_range": (geom(), cv(ub_row=65500)),
        "cfgerr_conv_rsv_flag": (geom(), good | (1 << 10)),
        "cfgerr_conv_rsv_gap1": (geom(), good | (1 << 44)),
        "cfgerr_conv_rsv_gap2": (geom(), good | (1 << 60)),
        "cfgerr_conv_rsv_hi":   (geom(), good | (1 << 100)),
        # SET_CONFIG kind 4 (section 3)
        "cfgerr_cfg4_4004":     ([], U.set_config(0x4004, 1)),
        "cfgerr_cfg4_4fff":     ([], U.set_config(0x4FFF, 1)),
        "cfgerr_cfg4_in_hi":    ([], U.set_config(0x4000, (1 << 32) | 8)),
        "cfgerr_cfg4_out_hi":   ([], U.set_config(0x4001, (1 << 40) | 8)),
        "cfgerr_cfg4_misc_hi":  ([], U.set_config(0x4002, (1 << 33) | 0x11)),
        "cfgerr_cfg4_pool_hi":  ([], U.set_config(0x4003, (1 << 16) | 4)),
        # ACTIVATE pooling (section 5)
        "cfgerr_pool_3":        (pool_ok, act(pool=3)),
        "cfgerr_pool_7":        (pool_ok, act(pool=7)),
        "cfgerr_pool_unconfigured": ([], act()),
        "cfgerr_pool_odd_w":    ([U.set_pool_w(3)], act(rows=12)),
        "cfgerr_pool_w1":       ([U.set_pool_w(1)], act(rows=8)),
        "cfgerr_pool_rows_mod": (pool_ok, act(rows=12)),
        "cfgerr_pool_acc_range": (pool_ok, act(acc_row=2040, rows=16)),
        "cfgerr_pool_ub_range": (pool_ok, act(ub_row=65535, rows=8)),
        "cfgerr_pool_out16":    (pool_ok, act(out16=1)),
        "cfgerr_pool_eltwise":  (pool_ok, act(eltwise=1)),
        "cfgerr_pool_qset4":    (pool_ok, act(qset=4)),
        "cfgerr_pool_func3":    (pool_ok, act(func=3)),
        "cfgerr_pool_avg_rows0": (pool_ok, act(pool=2, rows=0)),
    }
    out = []
    ref = None
    for name, (setup, bad) in cases.items():
        if isinstance(setup, int):
            setup = [setup]
        prog = prefix + list(setup) + [bad] + marker
        ub = [[0] * N for _ in range(48)]
        for b in range(B):
            ub[b] = [int(v) for v in xq[b]]
        s = isasim.CoreSim(ub, ww).run(prog)
        assert s.cfg_err and s.cfg_err_op == (bad & 0xFF), (name, s.cfg_err_why if s.cfg_err else "no cfg_err")
        assert s.instr_count == len(prefix) + len(setup) and s.irq_count == 0, name
        ref = net.int_forward(np.array(xq))
        exp = [[int(v) for v in ref[b]] for b in range(B)] + [[0] * N for _ in range(B)]
        got = [s.ub.get(16 + i) for i in range(2 * B)]
        assert got == exp, name
        mlp.write_case(outdir, name, prog, ww, ub, 16, exp, B, s.acc,
                       dict(kind="cfgerr", layers="16-16", out_cols=16, irq_expected=0, irq_tag="00000000",
                            cfg_err=1, cfg_err_op=f"{bad & 0xFF:02x}", cfg_err_index=len(prefix) + len(setup),
                            float_rel_err="na"))
        out.append((name, len(prog)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/m6/cases")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    results = []
    for name, (spec, seed) in CASES.items():
        results.append(build_cnn_case(a.out, name, spec, seed))
    errs = cnn_cfgerr_cases(a.out)
    names = [r["name"] for r in results] + [n for n, _ in errs]
    with open(os.path.join(a.out, "cases.txt"), "w") as fh:
        for n in names:
            fh.write(os.path.join(a.out, n) + "\n")
    for r in results:
        print(f"{r['name']:18s} instrs={r['instrs']:5d} weight_words={r['weights']:5d} ub_rows={r['ub_rows']:5d} "
              f"out_rows={r['out_rows']:4d} zp_in={r['zp_in']:5d} float_rel_err={r['err']:.4f}")
    print(f"{len(errs)} cfg_err cases; cases.txt -> {a.out}/cases.txt")


if __name__ == "__main__":
    main()
