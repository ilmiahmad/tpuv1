#!/usr/bin/env python3
"""Self-tests for the CNN model (run: python3 model/cnn/test_cnn.py; pytest-compatible).

Independent cross-checks wherever possible: address sequences via a padded index map (numpy gather)
instead of conv_ref's loop; pooling via numpy reshape and a math.floor round-half-away formula.
"""
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "isa"))
sys.path.insert(0, os.path.join(HERE, "..", "quantization"))
import asm                      # noqa: E402
import sim                      # noqa: E402
import conv_ref as cr           # noqa: E402
import compile_cnn as cc        # noqa: E402

N = 16


def indep_rows(in_w, in_h, out_w, out_h, sh, sw, pt, pl, kh, kw, oh0, cnt, ub_row):
    """Index-map gather: pad the (y, x) -> row map with -1 and take a strided slice."""
    idx = ub_row + np.arange(in_h * in_w).reshape(in_h, in_w)
    B = 512                                                       # border larger than any tap offset
    big = np.full((in_h + 2 * B, in_w + 2 * B), -1, np.int64)
    big[B:B + in_h, B:B + in_w] = idx
    out = []
    for oh in range(oh0, oh0 + cnt):
        for ow in range(out_w):
            v = big[B + oh * sh + kh - pt, B + ow * sw + kw - pl]
            out.append((v < 0, 0 if v < 0 else int(v)))
    return out


def test_addresses_vs_independent():
    rnd = random.Random(5)
    for _ in range(300):
        g = (rnd.randint(1, 30), rnd.randint(1, 30), rnd.randint(1, 30), rnd.randint(1, 30),
             rnd.randint(1, 4), rnd.randint(1, 4), rnd.randint(0, 15), rnd.randint(0, 15))
        in_h, in_w, out_h, out_w, sh, sw, pt, pl = g
        kh, kw = rnd.randint(0, 15), rnd.randint(0, 15)
        oh0 = rnd.randint(0, out_h - 1)
        cnt = rnd.randint(1, out_h - oh0)
        ub = rnd.randint(0, 65536 - in_h * in_w)
        a = cr.conv_rows(in_w, in_h, out_w, out_h, sh, sw, pt, pl, kh, kw, oh0, cnt, ub)
        b = indep_rows(in_w, in_h, out_w, out_h, sh, sw, pt, pl, kh, kw, oh0, cnt, ub)
        assert a == b, (g, kh, kw, oh0, cnt)


def test_unit_vector_files():
    lst = os.path.join(ROOT, "build", "m6", "addr", "cases.txt")
    assert os.path.exists(lst), "run model/cnn/gen_unit_vectors.py"
    n = 0
    for d in open(lst).read().split():
        cfg = dict((k, int(v)) for k, v in (l.split() for l in open(os.path.join(ROOT, d, "cfg.txt"))))
        rows = [int(l, 16) for l in open(os.path.join(ROOT, d, "rows.hex"))]
        exp = indep_rows(cfg["in_w"], cfg["in_h"], cfg["out_w"], cfg["out_h"], cfg["sh"], cfg["sw"],
                         cfg["pad_t"], cfg["pad_l"], cfg["kh"], cfg["kw"], cfg["oh0"], cfg["oh_cnt"], cfg["ub_row"])
        assert rows == [(int(p) << 16) | r for p, r in exp], d
        assert len(rows) == cfg["oh_cnt"] * cfg["out_w"] <= 2048
        n += 1
    lst = os.path.join(ROOT, "build", "m6", "pool", "cases.txt")
    for d in open(lst).read().split():
        meta = sim.read_meta(os.path.join(ROOT, d, "meta.txt"))
        words = [int(l, 16) for l in open(os.path.join(ROOT, d, "in.hex"))]
        out = [int(l, 16) for l in open(os.path.join(ROOT, d, "out.hex"))]
        assert len(words) == 4 * int(meta["groups"]) and len(out) == int(meta["groups"])
        for g in range(int(meta["groups"])):
            rows = np.array([sim.int_to_row(w) for w in words[4 * g:4 * g + 4]], np.int64)   # 4 x 16
            got = np.array(sim.int_to_row(out[g]), np.int64)
            if meta["mode"] == "max":
                exp = rows.max(axis=0)
            else:
                s = rows.sum(axis=0)
                exp = np.array([int(math.copysign(math.floor(abs(int(v)) / 4 + 0.5), v)) for v in s])
            assert (got == exp).all(), (d, g)
    print(f"  checked {n} address cases + pool cases against independent formulas")


def test_avg_pool_all_sums():
    """Every achievable 4-value sum: round half away from zero == the textbook formula."""
    for s_ in range(-512, 509):
        v = [max(-128, min(127, s_ // 4))] * 3
        v.append(s_ - sum(v))
        if -128 <= v[3] <= 127:
            assert cr.pool_lane_avg(v) == int(math.copysign(math.floor(abs(s_) / 4 + 0.5), s_)), s_
    assert cr.pool_lane_avg([127] * 4) == 127 and cr.pool_lane_avg([-128] * 4) == -128


def test_pool_numpy_vs_group():
    rnd = np.random.default_rng(3)
    for mode in ("max", "avg"):
        for _ in range(20):
            ph, pw = int(rnd.choice([2, 4, 6])), int(rnd.choice([2, 4, 8]))
            r = rnd.integers(-128, 128, (ph * pw, N))
            a = np.array(cr.pool_raster([list(map(int, x)) for x in r], pw, mode))
            img = r.reshape(ph, pw, N)
            b = cc.np_pool_int(img, mode).reshape(-1, N)
            assert (a == b).all(), (mode, ph, pw)


def _prog(words):
    return list(words)


def _bad(sim_obj, word):
    return sim_obj._invalid(word)


def _geom_sim(**kw):
    s = sim.CoreSim()
    g = dict(in_w=8, in_h=8, out_w=8, out_h=8, sh=1, sw=1, pad_t=1, pad_l=1, pad_val=-3)
    g.update(kw)
    s.conv.update(g)
    return s


def test_conv_legality_rules():
    ok = asm.convolve(0, 0, 1, 1, 0, 8)
    assert _bad(_geom_sim(), ok) is None
    assert _bad(sim.CoreSim(), ok)                                        # unconfigured geometry
    for key in ("in_w", "in_h", "out_w", "out_h", "sh", "sw"):
        assert _bad(_geom_sim(**{key: 0}), ok), key
    assert _bad(_geom_sim(pad_t=16), ok) and _bad(_geom_sim(pad_l=16), ok)
    assert _bad(_geom_sim(pad_t=15, pad_l=15), ok) is None
    assert _bad(_geom_sim(), asm.convolve(0, 0, 1, 1, 0, 0))              # oh_cnt == 0
    assert _bad(_geom_sim(), asm.convolve(0, 0, 1, 1, 5, 4))              # oh0 + cnt > out_h
    assert _bad(_geom_sim(), asm.convolve(0, 0, 1, 1, 8, 1))              # oh0 == out_h
    assert _bad(_geom_sim(), asm.convolve(0, 0, 1, 1, 7, 1)) is None      # exact fit of the window
    big = _geom_sim(out_w=64, out_h=64)
    assert _bad(big, asm.convolve(0, 0, 1, 1, 0, 33))                     # B = 2112 > 2048
    assert _bad(big, asm.convolve(0, 0, 1, 1, 0, 32)) is None             # B = 2048 exactly
    assert _bad(_geom_sim(), asm.convolve(0, 2047, 1, 1, 0, 1))           # acc_row + B > 2048
    assert _bad(_geom_sim(), asm.convolve(0, 2040, 1, 1, 0, 1)) is None   # acc_row + B == 2048
    assert _bad(_geom_sim(), asm.convolve(65500, 0, 1, 1, 0, 1))          # ub_row + H*W > 65536
    assert _bad(_geom_sim(), asm.convolve(65536 - 64, 0, 1, 1, 0, 1)) is None   # exact fit
    for bit in (10, 15, 43, 47, 56, 63, 88, 127):
        assert _bad(_geom_sim(), ok | (1 << bit)), bit


def test_kind4_config_rules():
    s = sim.CoreSim()
    for t in (0x4000, 0x4001, 0x4002, 0x4003):
        assert s._invalid(asm.set_config(t, 5)) is None, hex(t)
    for t in (0x4004, 0x4010, 0x4FFF, 0x4100):
        assert s._invalid(asm.set_config(t, 5)), hex(t)
    for t in (0x4000, 0x4001, 0x4002):
        assert s._invalid(asm.set_config(t, 1 << 32)), hex(t)
        assert s._invalid(asm.set_config(t, 0xFFFFFFFF)) is None
    assert s._invalid(asm.set_config(0x4003, 1 << 16)) and s._invalid(asm.set_config(0x4003, 0xFFFF)) is None
    # kinds 1-3 unchanged
    assert s._invalid(asm.set_config(0x3003, 1)) is None and s._invalid(asm.set_config(0x3004, 1))
    # register decode
    s = sim.CoreSim()
    s.step(asm.set_conv_in(9, 7)); s.step(asm.set_conv_out(5, 3))
    s.step(asm.set_conv_misc(2, 3, 4, 5, -128)); s.step(asm.set_pool_w(6))
    assert s.conv == dict(in_w=9, in_h=7, out_w=5, out_h=3, sh=2, sw=3, pad_t=4, pad_l=5, pad_val=-128)
    assert s.pool_w == 6


def test_pool_legality_rules():
    def s_pw(pw):
        s = sim.CoreSim(); s.pool_w = pw
        return s
    a = lambda **kw: asm.encode(asm.ACTIVATE, **{**dict(acc_row=0, ub_row=48, rows=8, qset=0, pool=1), **kw})
    assert s_pw(4)._invalid(a()) is None and s_pw(4)._invalid(a(pool=2)) is None
    for p in (3, 4, 5, 6, 7):
        assert s_pw(4)._invalid(a(pool=p)), p
    assert s_pw(0)._invalid(a()) and s_pw(1)._invalid(a(rows=4)) and s_pw(3)._invalid(a(rows=12))
    assert s_pw(4)._invalid(a(rows=12)) and s_pw(4)._invalid(a(rows=4))   # rows % 8 != 0
    assert s_pw(4)._invalid(a(rows=0))
    assert s_pw(4)._invalid(a(acc_row=2040, rows=16))                     # 2056 > 2048
    assert s_pw(4)._invalid(a(acc_row=2040, rows=8)) is None              # 2048 exact
    assert s_pw(4)._invalid(a(ub_row=65535, rows=8))                      # 2 output rows past the end
    assert s_pw(2)._invalid(a(ub_row=65535, rows=4)) is None              # 1 output row: exact fit
    assert s_pw(4)._invalid(a(out16=1)) and s_pw(4)._invalid(a(eltwise=1)) and s_pw(4)._invalid(a(qset=4))
    assert s_pw(4)._invalid(a(func=3)) and s_pw(4)._invalid(a(func=15))
    assert s_pw(0)._invalid(a(pool=0, rows=1)) is None                    # no pool: unchanged rules


def _tile(rnd):
    W = [[rnd.randint(-128, 127) for _ in range(N)] for _ in range(N)]
    return W, [sim.row_to_int(W[15 - t]) for t in range(16)]


def test_convolve_executes_exactly():
    """One tap of a 3x3 conv checked against a direct numpy per-pixel computation incl. padding."""
    rnd = random.Random(8)
    W, words = _tile(rnd)
    H = Wd = 6
    img = [[[rnd.randint(-128, 127) for _ in range(N)] for _ in range(Wd)] for _ in range(H)]
    ub = [img[y][x] for y in range(H) for x in range(Wd)]
    pv = -41
    prog = [asm.set_conv_in(Wd, H), asm.set_conv_out(3, 3), asm.set_conv_misc(2, 2, 1, 1, pv),
            asm.convolve(0, 10, 2, 0, 0, 3, 0, 1), asm.sync(), asm.halt()]
    s = sim.CoreSim(ub, words).run(prog)
    for oh in range(3):
        for ow in range(3):
            y, x = oh * 2 + 2 - 1, ow * 2 + 0 - 1
            row = img[y][x] if 0 <= y < H and 0 <= x < Wd else [pv] * N
            exp = [sum(row[k] * W[k][n] for k in range(N)) for n in range(N)]
            assert s.acc[10 + oh * 3 + ow] == exp, (oh, ow)
    # accumulate, with the SAME tile reused (new_weights = 0) for a second window
    prog = [asm.set_conv_in(Wd, H), asm.set_conv_out(3, 3), asm.set_conv_misc(1, 1, 1, 1, 0),
            asm.convolve(0, 0, 1, 1, 0, 1, 0, 1), asm.convolve(0, 3, 1, 1, 1, 2, 0, 0),
            asm.convolve(0, 0, 0, 0, 0, 1, 1, 1), asm.sync(), asm.halt()]
    s = sim.CoreSim(ub, words + [0] * 16).run(prog)
    assert s.wpos == 32 and not s.cfg_err


def test_hazards_conv_pool():
    rnd = random.Random(9)
    _, words = _tile(rnd)
    ub = [[1] * N for _ in range(64)]
    cfg = [asm.set_conv_in(8, 8), asm.set_conv_out(8, 8), asm.set_conv_misc(1, 1, 1, 1, 0)]
    q = [asm.set_config(0x1000 | c, (1 << 31) - 1) for c in range(16)] + [asm.set_config(0x3000, 0x7F8000)]
    conv = asm.convolve(0, 0, 1, 1, 0, 1, 0, 1)

    def run(extra):
        return sim.CoreSim(ub + [[0] * N] * 64, words).run(cfg + q + extra)
    # ACTIVATE of rows a CONVOLVE wrote, no SYNC -> RAW
    try:
        run([conv, asm.activate(0, 100, 8, 0)]); raise AssertionError("conv->activate RAW not detected")
    except sim.HazardError:
        pass
    run([conv, asm.sync(asm.U_MATRIX), asm.activate(0, 100, 8, 0)])        # fine with SYNC
    # CONVOLVE reads UB rows an unsynced ACTIVATE wrote -> RAW; writes ACC rows an ACTIVATE reads -> WAR
    ok = [conv, asm.sync(asm.U_MATRIX), asm.activate(0, 8, 8, 0)]
    try:
        run(ok + [asm.convolve(8, 64, 1, 1, 0, 1, 0, 0)]); raise AssertionError("activate->conv RAW not detected")
    except sim.HazardError:
        pass
    try:
        run(ok + [asm.convolve(0, 0, 1, 1, 0, 1, 0, 0)]); raise AssertionError("activate->conv WAR not detected")
    except sim.HazardError:
        pass
    # geometry rewrite while an unsynced CONVOLVE runs; POOL_W rewrite while a pooled ACTIVATE runs
    try:
        run([conv, asm.set_conv_in(4, 4)]); raise AssertionError("geometry write during CONVOLVE not detected")
    except sim.HazardError:
        pass
    run([conv, asm.sync(asm.U_MATRIX), asm.set_conv_in(4, 4)])
    pool = [asm.set_pool_w(4), conv, asm.sync(asm.U_MATRIX), asm.activate(0, 100, 8, 0, pool=1)]
    try:
        run(pool + [asm.set_pool_w(2)]); raise AssertionError("POOL_W write during pooled ACTIVATE not detected")
    except sim.HazardError:
        pass
    run(pool + [asm.sync(asm.U_ACT), asm.set_pool_w(2)])


def test_folded_equals_textbook():
    rnd = np.random.default_rng(1)
    for stride, pad, k in ((1, 1, 3), (2, 1, 3), (1, 0, 3), (1, 2, 5), (3, 0, 1)):
        zp = int(rnd.integers(-128, 128))
        x = rnd.integers(-128, 128, (9, 7, 4)).astype(np.int64)
        w = rnd.integers(-127, 128, (k, k, 4, 5)).astype(np.int64)
        b = rnd.integers(-5000, 5000, 5).astype(np.int64)
        folded = cr.conv_acc_folded(x, w, stride, pad, zp) + (b - zp * w.sum(axis=(0, 1, 2)))
        assert np.array_equal(folded, cr.conv_acc_textbook(x, w, b, stride, pad, zp))


def test_generated_cases():
    for sub in ("cases", "cases_b"):
        lst = os.path.join(ROOT, "build", "m6", sub, "cases.txt")
        assert os.path.exists(lst), f"run model/cnn/compile_cnn.py and compile_phase_b.py --cnn ({sub})"
        n = 0
        for d in open(lst).read().split():
            if sub == "cases":
                errs, _ = sim.check_case(os.path.join(ROOT, d))
                assert not errs, (d, errs)
            n += 1
        print(f"  {sub}: {n} cases")


def test_zero_points_and_pad_values_covered():
    zps = {}
    for d in open(os.path.join(ROOT, "build", "m6", "cases", "cases.txt")).read().split():
        meta = sim.read_meta(os.path.join(ROOT, d, "meta.txt"))
        if meta.get("kind") == "cnn":
            zps[d.split("/")[-1]] = int(meta["in_zp"])
    assert min(zps.values()) == -128 and max(zps.values()) == 127, zps
    assert all(v != 0 for v in zps.values()), zps


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: PASS")
    print("CNN_MODEL_PASS")
