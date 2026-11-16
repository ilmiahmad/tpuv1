#!/usr/bin/env python3
"""Instruction-level golden simulator of the Phase A TPU core (docs/m4_compute_core.md §5).

Executes a program (list of 128-bit words) against UB / ACC / Weight-FIFO / parameter state
with the exact §1 requantization, the M3 weight-tile order and the §5 cfg_err rules.

Also a HAZARD CHECKER: the hardware overlaps MATMUL and ACTIVATE (decoupled units) and has
no RAW scoreboard in Phase A, so a program is only correct if SYNC is placed wherever a unit
touches rows another still-unsynced unit writes/reads. Violations raise HazardError.

CLI: python3 model/isa/sim.py <case_dir>    (prog.hex, wmem.hex, ub_init.hex, meta.txt)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "quantization"))
sys.path.insert(0, os.path.join(HERE, "..", "cnn"))
import asm                                                     # noqa: E402
from tflite_int8 import requant, wrap32, wrap8                 # noqa: E402
import conv_ref as cr                                          # noqa: E402  (CONVOLVE / pooling, M6 contract)

N = 16
UB_ROWS, ACC_ROWS, NSETS = 65536, 2048, 4


def ddr_range_valid(address, length, reps=1, stride=0):
    end = address + (reps - 1) * stride + length
    return length > 0 and reps > 0 and any(lo <= address < end <= hi for lo, hi in
        ((0, 0x80000000), (0x800000000, 0x880000000)))


class HazardError(Exception):
    pass


def s8(b):
    return b - 256 if b >= 128 else b


class CoreSim:
    def __init__(self, ub_init=None, wwords=None, check_hazards=True, enable_dma=False, ddr=None):
        self.ub = {}                                  # row -> list of 16 int8
        for r, row in enumerate(ub_init or []):
            self.ub[r] = list(row)
        self.acc = {}                                 # row -> list of 16 int32
        self.wq = list(wwords or [])                  # weight words (128-bit ints), FIFO order
        self.wpos = 0
        self.W = [[0] * N for _ in range(N)]          # active tile W[k][n] (reset: zeros)
        self.params = [dict(m0=[0] * N, shift=[0] * N, bias=[0] * N, zp_out=0, act_min=0, act_max=0)
                       for _ in range(NSETS)]
        self.cfg_err, self.cfg_err_op = False, None
        self.halted = False
        self.irq_count, self.irq_tag, self.debug_tag = 0, None, None
        self.instr_count = 0
        self.check = check_hazards
        self.enable_dma = enable_dma
        self.ddr = dict(ddr or {})
        self.dma_count = 0
        # conv / pool geometry registers (SET_CONFIG kind 4); reset to 0, not cleared by cfg_err
        self.conv = dict(in_w=0, in_h=0, out_w=0, out_h=0, sh=0, sw=0, pad_t=0, pad_l=0, pad_val=0)
        self.pool_w = 0
        # hazard tracking: rows touched since the owning unit was last synced
        self.mat_ub_rd, self.mat_acc_wr = set(), set()
        self.act_acc_rd, self.act_ub_wr, self.act_qsets = set(), set(), set()
        self.conv_busy, self.pool_busy = False, False      # unsynced CONVOLVE / pooled ACTIVATE

    # ---------------------------------------------------------------- cfg_err rules (§5)
    def _invalid(self, word):
        op, f, rsv, known = asm.decode(word)
        if not known:
            return f"unknown opcode {op:#x}"
        if op in (asm.READ_HOST_MEM, asm.READ_WEIGHTS, asm.WRITE_HOST_MEM):
            if not self.enable_dma:
                return "DMA opcode in Phase A"
            if rsv:
                return "reserved DMA bits set"
            address = f["ddr_addr4"] * 16
            if op == asm.READ_WEIGHTS:
                return None if ddr_range_valid(address, f["tiles"] * 256) else "weight DMA range/count"
            if f["row_len"] == 0 or f["reps"] == 0:
                return "zero DMA row length/repetitions"
            if f["ub_row"] + f["row_len"] * f["reps"] > UB_ROWS:
                return "DMA UB range overflow"
            if not ddr_range_valid(address, f["row_len"] * 16, f["reps"], f["ddr_stride"] * 16):
                return "DMA DDR range overflow or non-DDR address"
            return None
        if rsv:
            return "reserved bits set"
        if op == asm.CONVOLVE:
            why = cr.conv_legal(self.conv, f["oh0"], f["oh_cnt"], f["acc_row"], f["ub_row"])
            if why:
                return f"CONVOLVE {why}"
        if op == asm.MATMUL:
            if f["B"] == 0:
                return "MATMUL B == 0"
            if f["ub_row"] + f["B"] > UB_ROWS:
                return "MATMUL ub range overflow"
            if f["acc_row"] + f["B"] > ACC_ROWS:
                return "MATMUL acc range overflow"
        if op == asm.ACTIVATE:
            if f["rows"] == 0:
                return "ACTIVATE rows == 0"
            if f["acc_row"] + f["rows"] > ACC_ROWS:
                return "ACTIVATE acc range overflow"
            if f["pool"] > 2:
                return f"pool {f['pool']}"
            out_rows = f["rows"] // 4 if f["pool"] else f["rows"]
            if f["ub_row"] + out_rows > UB_ROWS:
                return "ACTIVATE ub range overflow"
            if f["func"] > 2:
                return f"func {f['func']}"
            if f["out16"] or f["eltwise"]:
                return "out16/eltwise (M6)"
            if f["pool"]:
                pw = self.pool_w
                if pw < 2 or pw % 2:
                    return "pool: POOL_W unconfigured or odd"
                if f["rows"] % (2 * pw):
                    return "pool: rows not a multiple of 2*POOL_W"
            if f["qset"] >= NSETS:
                return f"qset {f['qset']}"
        if op == asm.SET_CONFIG:
            if not self._target_ok(f["target"]):
                return f"SET_CONFIG target {f['target']:#06x}"
            if f["target"] >> 12 == 4:                       # value bits outside the defined fields
                if f["value"] >> (16 if f["target"] == asm.POOL_W else 32):
                    return "SET_CONFIG kind 4: reserved value bits"
        return None

    @staticmethod
    def _target_ok(t):
        kind, rest = t >> 12, t & 0xFFF
        if kind in (1, 2):
            return (rest >> 4) < NSETS                       # q = bits [11:4], c = [3:0]
        if kind == 3:
            return rest < NSETS                              # q = bits [11:0]
        if kind == 4:
            return t in (asm.CONV_IN, asm.CONV_OUT, asm.CONV_MISC, asm.POOL_W)
        return False

    # ---------------------------------------------------------------- hazards
    def _hz(self, cond, msg):
        if self.check and cond:
            raise HazardError(f"instr {self.instr_count}: {msg}")

    def _sync(self, mask):
        if mask & asm.U_MATRIX:
            self.mat_ub_rd.clear(); self.mat_acc_wr.clear(); self.conv_busy = False
        if mask & asm.U_ACT:
            self.act_acc_rd.clear(); self.act_ub_wr.clear(); self.act_qsets.clear(); self.pool_busy = False

    # ---------------------------------------------------------------- execution
    def step(self, word):
        if self.halted or self.cfg_err:
            return False
        bad = self._invalid(word)
        if bad:
            self.cfg_err, self.cfg_err_op, self.cfg_err_why = True, word & 0xFF, bad
            return False
        self.instr_count += 1
        op, f, _, _ = asm.decode(word)
        if op in (asm.READ_HOST_MEM, asm.READ_WEIGHTS, asm.WRITE_HOST_MEM):
            self._dma(op, f)
        elif op == asm.MATMUL:
            self._matmul(**f)
        elif op == asm.CONVOLVE:
            self._convolve(**f)
        elif op == asm.ACTIVATE:
            self._activate(f["acc_row"], f["ub_row"], f["rows"], f["qset"], f["pool"])
        elif op == asm.SET_CONFIG:
            self._set_config(f["target"], f["value"])
        elif op == asm.SYNC:
            self._sync(f["mask"])
        elif op == asm.INTERRUPT_HOST:
            self._sync(asm.U_ALL)                            # waits for every unit to drain
            self.irq_count += 1; self.irq_tag = f["tag"]
        elif op == asm.DEBUG_TAG:
            self.debug_tag = f["tag"]
        elif op == asm.HALT:
            self.halted = True
        return not self.halted

    def _dma(self, op, f):
        self.dma_count += 1
        address = f["ddr_addr4"] * 16
        if op == asm.READ_WEIGHTS:
            for off in range(0, f["tiles"] * 256, 16):
                self.wq.append(int.from_bytes(bytes(self.ddr[address + off + b] for b in range(16)), "little"))
            return
        # Match the coarse hardware UB interlocks, which drain conflicting units.
        self._sync(asm.U_ACT | (asm.U_MATRIX if op == asm.READ_HOST_MEM else 0))
        for rep in range(f["reps"]):
            for row in range(f["row_len"]):
                urow = f["ub_row"] + rep * f["row_len"] + row
                daddr = address + rep * f["ddr_stride"] * 16 + row * 16
                if op == asm.READ_HOST_MEM:
                    self.ub[urow] = [s8(self.ddr[daddr + b]) for b in range(16)]
                else:
                    self.ddr.update((daddr + b, value & 255) for b, value in enumerate(self.ub[urow]))

    def run(self, prog, max_instr=10**6):
        for w in prog:
            if not self.step(w) or self.instr_count > max_instr:
                break
        return self

    def _next_tile(self):
        if self.wpos + N > len(self.wq):
            raise RuntimeError("Weight FIFO underflow (program needs more weight words)")
        words = self.wq[self.wpos:self.wpos + N]
        self.wpos += N
        # word t = W_tile[N-1-t]; byte n of a word = column n
        for t, w in enumerate(words):
            k = N - 1 - t
            self.W[k] = [s8((w >> (8 * n)) & 0xFF) for n in range(N)]

    def _mac(self, xs, acc_row, acc, what):
        """xs: list of 16-lane int8 input rows -> ACC[acc_row + b] (+)= xs[b] . W."""
        for b, x in enumerate(xs):
            y = [sum(x[k] * self.W[k][n] for k in range(N)) for n in range(N)]
            r = acc_row + b
            if acc:
                old = self.acc.get(r)
                if old is None:
                    raise RuntimeError(f"{what} accumulates into uninitialised ACC row {r}")
                y = [wrap32(o + v) for o, v in zip(old, y)]
            self.acc[r] = [wrap32(v) for v in y]

    def _matmul(self, ub_row, acc_row, B, acc, new_weights):
        rows_in = {ub_row + b for b in range(B)}
        rows_out = {acc_row + b for b in range(B)}
        self._hz(rows_in & self.act_ub_wr, "MATMUL reads UB rows an unsynced ACTIVATE writes (RAW)")
        self._hz(rows_out & self.act_acc_rd, "MATMUL writes ACC rows an unsynced ACTIVATE reads (WAR)")
        if new_weights:
            self._next_tile()
        xs = []
        for b in range(B):
            x = self.ub.get(ub_row + b)
            if x is None:
                raise RuntimeError(f"MATMUL reads uninitialised UB row {ub_row + b}")
            xs.append(x)
        self._mac(xs, acc_row, acc, "MATMUL")
        self.mat_ub_rd |= rows_in
        self.mat_acc_wr |= rows_out

    def _convolve(self, ub_row, acc_row, kh, kw, oh0, oh_cnt, acc, new_weights):
        """Contract section 4: implicit im2col; padded taps supply {16 x PAD_VAL}."""
        g = self.conv
        taps = cr.conv_rows(g["in_w"], g["in_h"], g["out_w"], g["out_h"], g["sh"], g["sw"],
                            g["pad_t"], g["pad_l"], kh, kw, oh0, oh_cnt, ub_row)
        rows_in = {r for pad, r in taps if not pad}
        rows_out = {acc_row + b for b in range(len(taps))}
        self._hz(rows_in & self.act_ub_wr, "CONVOLVE reads UB rows an unsynced ACTIVATE writes (RAW)")
        self._hz(rows_out & self.act_acc_rd, "CONVOLVE writes ACC rows an unsynced ACTIVATE reads (WAR)")
        if new_weights:
            self._next_tile()
        pad_row = [wrap8(g["pad_val"])] * N
        xs = []
        for pad, r in taps:
            x = pad_row if pad else self.ub.get(r)
            if x is None:
                raise RuntimeError(f"CONVOLVE reads uninitialised UB row {r}")
            xs.append(x)
        self._mac(xs, acc_row, acc, "CONVOLVE")
        self.mat_ub_rd |= rows_in
        self.mat_acc_wr |= rows_out
        self.conv_busy = True

    def _activate(self, acc_row, ub_row, rows, q, pool=0):
        out_rows = rows // 4 if pool else rows
        rows_in = {acc_row + r for r in range(rows)}
        rows_out = {ub_row + r for r in range(out_rows)}
        self._hz(rows_in & self.mat_acc_wr, "ACTIVATE reads ACC rows an unsynced MATMUL writes (RAW)")
        self._hz(rows_out & self.mat_ub_rd, "ACTIVATE writes UB rows an unsynced MATMUL reads (WAR)")
        p = self.params[q]
        vals = []
        for r in range(rows):
            a = self.acc.get(acc_row + r)
            if a is None:
                raise RuntimeError(f"ACTIVATE reads uninitialised ACC row {acc_row + r}")
            vals.append([
                requant(a[c], p["bias"][c], p["m0"][c], p["shift"][c], p["zp_out"], p["act_min"], p["act_max"])
                for c in range(N)])
        if pool:
            vals = cr.pool_raster(vals, self.pool_w, "max" if pool == 1 else "avg")
            self.pool_busy = True
        for r, row in enumerate(vals):
            self.ub[ub_row + r] = row
        self.act_acc_rd |= rows_in
        self.act_ub_wr |= rows_out
        self.act_qsets.add(q)

    def _set_config(self, t, v):
        kind = t >> 12
        if kind == 4:
            self._hz(t != asm.POOL_W and self.conv_busy, "SET_CONFIG conv geometry while an unsynced CONVOLVE runs")
            self._hz(t == asm.POOL_W and self.pool_busy, "SET_CONFIG POOL_W while an unsynced pooled ACTIVATE runs")
            g = self.conv
            if t == asm.CONV_IN:
                g["in_w"], g["in_h"] = v & 0xFFFF, (v >> 16) & 0xFFFF
            elif t == asm.CONV_OUT:
                g["out_w"], g["out_h"] = v & 0xFFFF, (v >> 16) & 0xFFFF
            elif t == asm.CONV_MISC:
                g["sh"], g["sw"] = v & 15, (v >> 4) & 15
                g["pad_t"], g["pad_l"], g["pad_val"] = (v >> 8) & 255, (v >> 16) & 255, wrap8(v >> 24)
            else:
                self.pool_w = v & 0xFFFF
            return
        q = ((t & 0xFFF) >> 4) if kind in (1, 2) else (t & 0xFFF)
        self._hz(q in self.act_qsets, f"SET_CONFIG qset {q} while an unsynced ACTIVATE uses it")
        p = self.params[q]
        if kind == 1:
            c = t & 0xF
            p["m0"][c] = wrap32(v & 0xFFFFFFFF)
            p["shift"][c] = wrap8((v >> 32) & 0xFF)
        elif kind == 2:
            p["bias"][t & 0xF] = wrap32(v & 0xFFFFFFFF)
        else:
            p["zp_out"] = wrap8(v & 0xFF)
            p["act_min"] = wrap8((v >> 8) & 0xFF)
            p["act_max"] = wrap8((v >> 16) & 0xFF)


# ---------------------------------------------------------------------------------------------
# file helpers (shared hex formats)
# ---------------------------------------------------------------------------------------------
def row_to_int(row, w=8):
    v = 0
    for i, x in enumerate(row):
        v |= (x & ((1 << w) - 1)) << (w * i)
    return v


def int_to_row(v, w=8, n=N):
    m = (1 << w) - 1
    out = []
    for i in range(n):
        x = (v >> (w * i)) & m
        out.append(x - (1 << w) if x >> (w - 1) else x)
    return out


def read_meta(path):
    meta = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                k, v = line.split(None, 1)
                meta[k] = v.strip()
    return meta


def run_case(case_dir, check_hazards=True):
    prog = asm.read_hex(os.path.join(case_dir, "prog.hex"))
    ww = asm.read_hex(os.path.join(case_dir, "wmem.hex")) if os.path.getsize(os.path.join(case_dir, "wmem.hex")) else []
    ub = [int_to_row(v) for v in asm.read_hex(os.path.join(case_dir, "ub_init.hex"))]
    sim = CoreSim(ub, ww, check_hazards).run(prog)
    return sim


def check_case(case_dir):
    meta = read_meta(os.path.join(case_dir, "meta.txt"))
    sim = run_case(case_dir)
    exp = [int_to_row(v) for v in asm.read_hex(os.path.join(case_dir, "ub_expect.hex"))]
    first = int(meta["out_first_row"])
    got = [sim.ub.get(first + i) for i in range(len(exp))]
    errs = []
    if got != exp:
        bad = sum(1 for g, e in zip(got, exp) if g != e)
        errs.append(f"{bad}/{len(exp)} output rows differ")
    if int(meta["cfg_err"]) != int(sim.cfg_err):
        errs.append(f"cfg_err {sim.cfg_err} != meta {meta['cfg_err']}")
    if sim.cfg_err and int(meta["cfg_err_op"], 16) != sim.cfg_err_op:
        errs.append(f"cfg_err_op {sim.cfg_err_op:#x} != meta {meta['cfg_err_op']}")
    if int(meta["irq_expected"]) != sim.irq_count:
        errs.append(f"irq {sim.irq_count} != meta {meta['irq_expected']}")
    if not sim.cfg_err and sim.wpos != int(meta["weight_words"]):
        errs.append(f"consumed {sim.wpos} weight words, meta says {meta['weight_words']}")
    return errs, sim


if __name__ == "__main__":
    for d in sys.argv[1:]:
        e, s = check_case(d)
        print(f"{d}: {'PASS' if not e else 'FAIL ' + '; '.join(e)}  (instr={s.instr_count}, "
              f"wwords={s.wpos}, irq={s.irq_count}, cfg_err={s.cfg_err})")
