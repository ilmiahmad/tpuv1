#!/usr/bin/env python3
"""MLP compiler + TFLite-exact integer reference (docs/m4_compute_core.md §6, §8).

For each case writes build/m4/cases/<name>/:
  prog.hex      program, one 128-bit instruction per line (32 hex digits, MSB first), ends with
                INTERRUPT_HOST + HALT (also prog.bin, 16 B little-endian per instruction)
  wmem.hex      Weight Memory words in FIFO order (128 bits/line; byte n = column n;
                tile order: layer, nt outer, kt inner; inside a tile word t = W_tile[15-t])
  ub_init.hex   UB rows from row 0 (128 bits/line, lane k = bits [8k+7:8k]); covers every row
                the program touches (zeros where not input) so the hardware UB is defined
  ub_expect.hex expected UB rows of the output region (first row = meta out_first_row)
  acc_expect.hex final ACC rows 0..acc_rows-1 (512 bits/line, channel c = bits [32c+31:32c])
  prog.lst      disassembly
  meta.txt      key/value: N B layers prog_instrs weight_words ub_init_rows out_first_row
                out_rows out_cols acc_rows irq_expected irq_tag cfg_err cfg_err_op
Expected outputs come from the numpy integer pipeline here AND from model/isa/sim.py executing
the emitted program; they must agree (checked at generation time).

Quantization follows TFLite int8 FullyConnected: asymmetric int8 activations, per-channel
symmetric int8 weights (zp 0, [-127,127]), int32 bias = round(b / (s_in*s_w)), effective
multiplier s_in*s_w/s_out via QuantizeMultiplier, activation range via
CalculateActivationRangeQuantized. The hardware computes the raw sum x_q*w_q, so the input
zero-point term -zp_in*sum_k w_q[k][c] is folded into the hardware bias.

Layout (§6): UB activations k-tile major: row base + kt*B + b = X[b][16kt:16kt+16].
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "quantization"))
sys.path.insert(0, os.path.join(HERE, "..", "isa"))
import asm                                                             # noqa: E402
import sim as isasim                                                   # noqa: E402
from tflite_int8 import quantize_multiplier, act_range, requant         # noqa: E402

N = 16
ACC_HALF = 1024                       # ACC double buffer: tiles alternate rows [0,B) / [1024,1024+B)
U = asm


def pad16(n):
    return (n + 15) // 16 * 16


def quant_act(x, lo, hi):
    """TFLite-style asymmetric int8 params for a float range (always includes 0)."""
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    scale = (hi - lo) / 255.0 if hi > lo else 1.0
    zp = int(np.clip(np.round(-128 - lo / scale), -128, 127))
    return scale, zp


def quantize(x, scale, zp):
    return np.clip(np.round(x / scale) + zp, -128, 127).astype(np.int64)


# ---------------------------------------------------------------------------------------------
class Net:
    """Float MLP + its TFLite int8 quantization."""

    def __init__(self, dims, acts, seed, Bcal=64):
        rng = np.random.default_rng(seed)
        self.dims, self.acts = dims, acts
        self.Wf = [rng.normal(0, 1.0 / np.sqrt(dims[i]), (dims[i], dims[i + 1])) for i in range(len(dims) - 1)]
        self.bf = [rng.normal(0, 0.1, dims[i + 1]) for i in range(len(dims) - 1)]
        # calibration: activation ranges on a random batch (fixed)
        xc = rng.uniform(-1, 1, (Bcal, dims[0]))
        self._calibrate(xc)

    @classmethod
    def from_arrays(cls, Ws, bs, acts, calib_x):
        """Quantize a TRAINED float MLP (Ws[l]: K x N, bs[l]: N, acts[l] in none/relu/relu6)
        with the same TFLite int8 recipe as the random constructor; calib_x: M x K float
        calibration batch (activation ranges)."""
        Ws = [np.asarray(W, dtype=np.float64) for W in Ws]
        bs = [np.asarray(b, dtype=np.float64) for b in bs]
        if not Ws or len(Ws) != len(bs) or len(Ws) != len(acts):
            raise ValueError("Ws, bs and acts must have the same non-zero length")
        for l, (W, b) in enumerate(zip(Ws, bs)):
            if W.ndim != 2 or b.shape != (W.shape[1],):
                raise ValueError(f"layer {l}: bad shapes W{W.shape} b{b.shape}")
            if l and W.shape[0] != Ws[l - 1].shape[1]:
                raise ValueError(f"layer {l}: input width {W.shape[0]} != previous output {Ws[l - 1].shape[1]}")
        if any(a not in ("none", "relu", "relu6") for a in acts):
            raise ValueError("acts must be none/relu/relu6")
        xc = np.asarray(calib_x, dtype=np.float64)
        if xc.ndim != 2 or xc.shape[1] != Ws[0].shape[0] or xc.shape[0] < 1:
            raise ValueError(f"calib_x must be M x {Ws[0].shape[0]}")
        net = cls.__new__(cls)
        net.dims = [Ws[0].shape[0]] + [W.shape[1] for W in Ws]
        net.acts = list(acts)
        net.Wf, net.bf = Ws, bs
        net._calibrate(xc)
        return net

    def _calibrate(self, xc):
        acts = self.acts
        self.in_scale, self.in_zp = quant_act(xc, xc.min(), xc.max())
        self.layers = []
        s_in, zp_in, h = self.in_scale, self.in_zp, xc
        for l, (W, b, act) in enumerate(zip(self.Wf, self.bf, acts)):
            h = h @ W + b
            if act == "relu":
                h = np.maximum(h, 0)
            elif act == "relu6":
                h = np.clip(h, 0, 6)
            s_out, zp_out = quant_act(h, h.min(), h.max())
            s_w = np.maximum(np.abs(W).max(axis=0), 1e-8) / 127.0
            wq = np.clip(np.round(W / s_w), -127, 127).astype(np.int64)
            bq = np.round(b / (s_in * s_w)).astype(np.int64)
            m0, sh = zip(*[quantize_multiplier(float(s_in * s_w[c] / s_out)) for c in range(W.shape[1])])
            lo, hi = act_range(act, s_out, zp_out)
            # hardware bias: fold the input zero-point term (raw x_q * w_q in the array)
            bhw = bq - zp_in * wq.sum(axis=0)
            self.layers.append(dict(wq=wq, bq=bq, bhw=bhw, m0=list(m0), shift=list(sh),
                                    s_in=s_in, zp_in=zp_in, s_out=s_out, zp_out=zp_out,
                                    act_min=lo, act_max=hi))
            s_in, zp_in = s_out, zp_out

    def int_forward(self, xq):
        """TFLite-exact integer pipeline on padded tensors. xq: B x pad16(dims[0]) int8."""
        h = xq.astype(np.int64)
        for L in self.layers:
            K, Nout = L["wq"].shape
            Kp, Np = pad16(K), pad16(Nout)
            W = np.zeros((Kp, Np), np.int64); W[:K, :Nout] = L["wq"]
            acc = h[:, :Kp] @ W                                   # raw sum, int64 (fits int32)
            assert np.abs(acc).max() < 2**31
            out = np.zeros_like(acc)
            for c in range(Np):
                if c < Nout:
                    bias, m0, sh = int(L["bhw"][c]), L["m0"][c], L["shift"][c]
                else:
                    bias, m0, sh = 0, 0, 0                         # padded channel -> zp clamp
                for b in range(acc.shape[0]):
                    out[b, c] = requant(int(acc[b, c]), bias, m0, sh, L["zp_out"], L["act_min"], L["act_max"])
            h = out
        return h

    def float_forward(self, x):
        h = x
        for W, b, act in zip(self.Wf, self.bf, self.acts):
            h = h @ W + b
            h = np.maximum(h, 0) if act == "relu" else (np.clip(h, 0, 6) if act == "relu6" else h)
        return h


# ---------------------------------------------------------------------------------------------
def weight_words(net):
    """FIFO order: layer, nt outer, kt inner; word t = W_tile[15-t]."""
    words = []
    for L in net.layers:
        K, Nout = L["wq"].shape
        W = np.zeros((pad16(K), pad16(Nout)), np.int64); W[:K, :Nout] = L["wq"]
        for nt in range(pad16(Nout) // 16):
            for kt in range(pad16(K) // 16):
                tile = W[16 * kt:16 * kt + 16, 16 * nt:16 * nt + 16]
                for t in range(16):
                    words.append(isasim.row_to_int([int(v) for v in tile[15 - t]]))
    return words


def emit_program(net, B, in_base=0, tag=0xD0E0_0001):
    """Returns (prog words, out_first_row, ub rows used, acc rows used)."""
    assert 1 <= B <= ACC_HALF
    prog = []
    base = in_base
    nxt = in_base + (pad16(net.dims[0]) // 16) * B
    tile_idx = 0
    qset_busy = {}            # qset -> True if an unsynced ACTIVATE may still use it
    region_busy = {0: False, 1: False}   # ACC half -> unsynced ACTIVATE reads it
    act_pending = False
    for l, L in enumerate(net.layers):
        K, Nout = L["wq"].shape
        KT, NT = pad16(K) // 16, pad16(Nout) // 16
        if l > 0 and act_pending:
            prog.append(U.sync(U.U_ACT))          # RAW: next layer reads UB rows ACTIVATE wrote
            qset_busy = {}; region_busy = {0: False, 1: False}; act_pending = False
        out_base = nxt
        for nt in range(NT):
            q = tile_idx % 4
            half = tile_idx % 2
            if qset_busy.get(q) or region_busy[half]:
                prog.append(U.sync(U.U_ACT))      # WAR on the qset / ACC half
                qset_busy = {}; region_busy = {0: False, 1: False}; act_pending = False
            for c in range(16):
                ch = 16 * nt + c
                if ch < Nout:
                    m0, sh, bias = L["m0"][ch], L["shift"][ch], int(L["bhw"][ch])
                else:
                    m0, sh, bias = 0, 0, 0
                prog.append(U.set_config(0x1000 | (q << 4) | c, (m0 & 0xFFFFFFFF) | ((sh & 0xFF) << 32)))
                prog.append(U.set_config(0x2000 | (q << 4) | c, bias & 0xFFFFFFFF))
            prog.append(U.set_config(0x3000 | q, (L["zp_out"] & 0xFF) | ((L["act_min"] & 0xFF) << 8)
                                     | ((L["act_max"] & 0xFF) << 16)))
            a = half * ACC_HALF
            for kt in range(KT):
                prog.append(U.matmul(base + kt * B, a, B, acc=1 if kt > 0 else 0, new_weights=1))
            prog.append(U.sync(U.U_MATRIX))       # RAW: ACTIVATE reads the ACC rows just written
            prog.append(U.activate(a, out_base + nt * B, B, q,
                                   func={"none": 0, "relu": 1, "relu6": 2}[net.acts[l]]))
            qset_busy[q] = True; region_busy[half] = True; act_pending = True
            tile_idx += 1
        base = out_base
        nxt = out_base + NT * B
    prog.append(U.sync(U.U_ALL))
    prog.append(U.interrupt_host(tag))
    prog.append(U.halt())
    acc_rows = ACC_HALF + B if tile_idx > 1 else B
    return prog, base, nxt, acc_rows


def write_case(outdir, name, prog, wwords, ub_init_rows, out_first, exp_rows, acc_rows, acc_img, meta_extra):
    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)
    asm.write_hex(os.path.join(d, "prog.hex"), prog)
    asm.write_bin(os.path.join(d, "prog.bin"), prog)
    with open(os.path.join(d, "prog.lst"), "w") as fh:
        for i, w in enumerate(prog):
            fh.write(f"{i:5d}  {w:032x}  {asm.disasm(w)}\n")
    asm.write_hex(os.path.join(d, "wmem.hex"), wwords)
    asm.write_hex(os.path.join(d, "ub_init.hex"), [isasim.row_to_int(r) for r in ub_init_rows])
    asm.write_hex(os.path.join(d, "ub_expect.hex"), [isasim.row_to_int(r) for r in exp_rows])
    with open(os.path.join(d, "acc_expect.hex"), "w") as fh:
        for r in range(acc_rows):
            row = acc_img.get(r)
            fh.write(f"{isasim.row_to_int(row, 32) if row else 0:0128x}\n")
    meta = dict(N=N, prog_instrs=len(prog), weight_words=len(wwords), ub_init_rows=len(ub_init_rows),
                out_first_row=out_first, out_rows=len(exp_rows), acc_rows=acc_rows)
    meta.update(meta_extra)
    with open(os.path.join(d, "meta.txt"), "w") as fh:
        for k, v in meta.items():
            fh.write(f"{k} {v}\n")
    return d


def build_mlp_case(outdir, name, dims, acts, B, seed, net=None):
    net = net or Net(dims, acts, seed)
    rng = np.random.default_rng(seed + 1)
    x = rng.uniform(-1, 1, (B, dims[0]))
    xq = np.full((B, pad16(dims[0])), net.in_zp, np.int64)     # padded lanes: zp (any value works)
    xq[:, :dims[0]] = quantize(x, net.in_scale, net.in_zp)
    yq = net.int_forward(xq)                                   # golden #1
    prog, out_first, ub_end, acc_rows = emit_program(net, B)
    ww = weight_words(net)
    # UB image: input k-tile major, zeros elsewhere up to every touched row
    ub = [[0] * N for _ in range(ub_end)]
    for kt in range(pad16(dims[0]) // 16):
        for b in range(B):
            ub[kt * B + b] = [int(v) for v in xq[b, 16 * kt:16 * kt + 16]]
    exp = []
    for nt in range(yq.shape[1] // 16):
        for b in range(B):
            exp.append([int(v) for v in yq[b, 16 * nt:16 * nt + 16]])
    # golden #2: instruction simulator on the emitted program (with hazard checking)
    s = isasim.CoreSim(ub, ww, check_hazards=True).run(prog)
    got = [s.ub.get(out_first + i) for i in range(len(exp))]
    assert got == exp, f"{name}: sim.py disagrees with the integer pipeline"
    assert s.wpos == len(ww) and s.irq_count == 1 and not s.cfg_err
    # float sanity (quantization quality, informational)
    yf = net.float_forward(x)
    L = net.layers[-1]
    deq = (yq[:, :dims[-1]] - L["zp_out"]) * L["s_out"]
    err = float(np.abs(deq - yf).max() / max(1e-9, np.abs(yf).max()))
    d = write_case(outdir, name, prog, ww, ub, out_first, exp, acc_rows, s.acc,
                   dict(B=B, layers="-".join(map(str, dims)), out_cols=pad16(dims[-1]),
                        irq_expected=1, irq_tag=f"{s.irq_tag:08x}", cfg_err=0, cfg_err_op="00",
                        float_rel_err=f"{err:.4f}"))
    return d, net, len(prog), len(ww), len(ub), err


# ---------------------------------------------------------------------------------------------
def cfgerr_cases(outdir):
    """Valid fc1-style prefix, then one invalid instruction, then a marker ACTIVATE that must
    NOT execute, INTERRUPT_HOST, HALT. Expected: prefix output only, marker rows still 0."""
    net = Net([16, 16], ["none"], seed=900)
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
    marker = [U.activate(0, 32, B, 0), U.interrupt_host(0xBAD0_0001), U.halt()]
    ww = weight_words(net)
    m = U.matmul(0, 0, 1)
    bad = {
        "cfgerr_unknown_op":   U.encode_raw(0x05),
        "cfgerr_unknown_op7f": U.encode_raw(0x7F),
        "cfgerr_dma_read":     U.encode(U.READ_HOST_MEM, ub_row=0, ddr_addr4=0x100, row_len=1, reps=1),
        "cfgerr_dma_weights":  U.encode(U.READ_WEIGHTS, tiles=1, ddr_addr4=0x100),
        "cfgerr_dma_write":    U.encode(U.WRITE_HOST_MEM, ub_row=0, ddr_addr4=0x100, row_len=1, reps=1),
        # M6: CONVOLVE is real; with the geometry registers never written it is still cfg_err
        "cfgerr_convolve":     U.encode(U.CONVOLVE, ub_row=0, acc_row=0, oh_cnt=1),
        "cfgerr_act_sigmoid":  U.encode(U.ACTIVATE, acc_row=0, ub_row=48, rows=1, func=3),
        "cfgerr_act_tanh":     U.encode(U.ACTIVATE, acc_row=0, ub_row=48, rows=1, func=4),
        "cfgerr_act_func15":   U.encode(U.ACTIVATE, acc_row=0, ub_row=48, rows=1, func=15),
        "cfgerr_act_out16":    U.encode(U.ACTIVATE, acc_row=0, ub_row=48, rows=1, out16=1),
        "cfgerr_act_pool":     U.encode(U.ACTIVATE, acc_row=0, ub_row=48, rows=1, pool=1),
        "cfgerr_act_eltwise":  U.encode(U.ACTIVATE, acc_row=0, ub_row=48, rows=1, eltwise=1),
        "cfgerr_act_qset4":    U.encode(U.ACTIVATE, acc_row=0, ub_row=48, rows=1, qset=4),
        # M6: kind 4 targets 0x4000-0x4003 are conv/pool geometry registers; 0x4004 is still invalid
        "cfgerr_cfg_target":   U.set_config(0x4004, 1),
        "cfgerr_cfg_q4":       U.set_config(0x1040, 1),
        "cfgerr_cfg_zp_q4":    U.set_config(0x3004, 1),
        "cfgerr_mm_rsv_hi":    m | (1 << 64),
        "cfgerr_mm_rsv_flag":  m | (1 << 10),
        "cfgerr_mm_rsv_gap":   m | (1 << 43),
        "cfgerr_sync_rsv":     U.sync(U.U_ALL) | (1 << 12),
        "cfgerr_nop_rsv":      U.encode(U.NOP) | (1 << 100),
        "cfgerr_halt_rsv":     U.halt() | (1 << 8),
        # coordinator decisions: zero counts and range overflow are cfg_err, no wrapping
        "cfgerr_mm_b0":        U.matmul(0, 0, 0),
        "cfgerr_act_rows0":    U.encode(U.ACTIVATE, acc_row=0, ub_row=48, rows=0),
        "cfgerr_mm_ub_range":  U.matmul(65535, 0, 2),
        "cfgerr_mm_acc_range": U.matmul(0, 2047, 2),
        "cfgerr_act_acc_range": U.encode(U.ACTIVATE, acc_row=2047, ub_row=48, rows=2),
        "cfgerr_act_ub_range": U.encode(U.ACTIVATE, acc_row=0, ub_row=65535, rows=2),
    }
    out = []
    for name, w in bad.items():
        prog = prefix + [w] + marker
        ub = [[0] * N for _ in range(48)]
        for b in range(B):
            ub[b] = [int(v) for v in xq[b]]
        s = isasim.CoreSim(ub, ww).run(prog)
        assert s.cfg_err and s.cfg_err_op == (w & 0xFF), name
        assert s.irq_count == 0
        # expected: rows 16..31 = prefix output, rows 32..47 = untouched zeros (marker blocked)
        ref = net.int_forward(np.array(xq))
        exp = [[int(v) for v in ref[b]] for b in range(B)] + [[0] * N for _ in range(B)]
        got = [s.ub.get(16 + i) for i in range(2 * B)]
        assert got == exp, name
        write_case(outdir, name, prog, ww, ub, 16, exp, B, s.acc,
                   dict(B=B, layers="16-16", out_cols=16, irq_expected=0, irq_tag="00000000",
                        cfg_err=1, cfg_err_op=f"{w & 0xFF:02x}", cfg_err_index=len(prefix),
                        float_rel_err="na"))
        out.append((name, len(prog)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/m4/cases")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cases = []
    d, _, ni, nw, nu, e = build_mlp_case(a.out, "fc1", [16, 16], ["none"], 16, seed=11)
    cases.append(("fc1", ni, nw, nu, e))
    d, net3, ni, nw, nu, e = build_mlp_case(a.out, "mlp3", [64, 128, 64, 16], ["relu", "relu", "none"], 32, seed=33)
    cases.append(("mlp3", ni, nw, nu, e))
    d, _, ni, nw, nu, e = build_mlp_case(a.out, "mlp_b1", [64, 128, 64, 16], ["relu", "relu", "none"], 1, seed=33, net=net3)
    cases.append(("mlp_b1", ni, nw, nu, e))
    d, _, ni, nw, nu, e = build_mlp_case(a.out, "mlp_wide", [256, 256], ["relu6"], 100, seed=55)
    cases.append(("mlp_wide", ni, nw, nu, e))
    errs = cfgerr_cases(a.out)
    names = [c[0] for c in cases] + [n for n, _ in errs]
    with open(os.path.join(a.out, "cases.txt"), "w") as fh:
        for n in names:
            fh.write(os.path.join(a.out, n) + "\n")
    for n, ni, nw, nu, e in cases:
        print(f"{n:10s} instrs={ni:4d} weight_words={nw:5d} ub_rows={nu:5d} float_rel_err={e:.4f}")
    print(f"{len(errs)} cfg_err cases; cases.txt -> {a.out}/cases.txt")


if __name__ == "__main__":
    main()
