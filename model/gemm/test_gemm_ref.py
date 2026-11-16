#!/usr/bin/env python3
"""Self-tests for the M3 golden model. Run: python3 model/gemm/test_gemm_ref.py  (or pytest)."""
import os, random, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemm_ref as g          # noqa: E402
import gen_vectors as gv      # noqa: E402

try:
    import numpy as np
except ImportError:           # pure-Python fallback
    np = None


def independent_gemm(x, w):
    if np is not None:
        return (np.array(x, dtype=np.int64) @ np.array(w, dtype=np.int64)).tolist()
    n = len(w)
    return [[sum(int(r[k]) * int(w[k][c]) for k in range(n)) for c in range(n)] for r in x]


def test_pack_roundtrip_and_twos_complement():
    r = random.Random(1)
    for n in (4, 8, 16):
        for _ in range(200):
            row = [r.randint(-128, 127) for _ in range(n)]
            h = g.x_row_hex(row)
            assert len(h) == 2 * n
            assert g.unpack_lanes(int(h, 16), n, 8) == row
            y = [r.randint(-(1 << 31), (1 << 31) - 1) for _ in range(n)]
            hy = g.y_row_hex(y)
            assert len(hy) == 8 * n and g.unpack_lanes(int(hy, 16), n, 32) == y
    # explicit two's complement + lane placement (lane 0 = least significant byte)
    assert g.x_row_hex([-1, 0, 0, -128]) == "800000ff"
    assert g.y_row_hex([-1, 1, 0, -2]) == "fffffffe" "00000000" "00000001" "ffffffff"


def test_tile_word_order():
    r = random.Random(2)
    for n in (4, 8, 16):
        w = [[r.randint(-128, 127) for _ in range(n)] for _ in range(n)]
        words = g.tile_fifo_words(w)
        assert words[0] == w[n - 1] and words[n - 1] == w[0]       # bottom row first
        assert g.tile_from_fifo_words(words) == w
        hexed = [g.w_word_hex(wd) for wd in words]                  # byte c -> column c
        assert all(g.unpack_lanes(int(h, 16), n, 8) == wd for h, wd in zip(hexed, words))


def test_extremes():
    for n in (4, 8, 16):
        assert g.gemm([[-128] * n], [[-128] * n for _ in range(n)]) == [[16384 * n] * n]
        assert g.gemm([[127] * n], [[-128] * n for _ in range(n)]) == [[-16256 * n] * n]


def test_gemm_vs_independent():
    r = random.Random(3)
    for n in (4, 8, 16):
        w = [[r.randint(-128, 127) for _ in range(n)] for _ in range(n)]
        x = [[r.randint(-128, 127) for _ in range(n)] for _ in range(50)]
        assert g.gemm(x, w) == independent_gemm(x, w)


def test_generated_files_consistent():
    with tempfile.TemporaryDirectory() as tmp:
        gv.main(["--out", tmp, "--n", "4", "8"])
        for name in gv.CASES:
            for n in (4, 8):
                d = os.path.join(tmp, f"{name}_N{n}")
                meta = dict(l.split() for l in open(os.path.join(d, "meta.txt")))
                xs = open(os.path.join(d, "x.hex")).read().split()
                xn = open(os.path.join(d, "xnew.hex")).read().split()
                ws = open(os.path.join(d, "w.hex")).read().split()
                ys = open(os.path.join(d, "y.hex")).read().split()
                assert int(meta["N"]) == n and int(meta["rows"]) == len(xs) == len(xn) == len(ys)
                assert int(meta["words"]) == len(ws) == n * int(meta["blocks"]) == n * xn.count("1")
                assert xn[0] == "1"
                # recompute y from the files alone (tile order + packing), independently
                tiles = [g.tile_from_fifo_words([g.unpack_lanes(int(h, 16), n, 8) for h in ws[t*n:(t+1)*n]])
                         for t in range(len(ws) // n)]
                ti = -1
                for xh, f, yh in zip(xs, xn, ys):
                    ti += f == "1"
                    exp = independent_gemm([g.unpack_lanes(int(xh, 16), n, 8)], tiles[ti])[0]
                    assert g.unpack_lanes(int(yh, 16), n, 32) == exp
        assert len(open(os.path.join(tmp, "cases.txt")).read().split()) == 2 * len(gv.CASES)


# ---------------------------------------------------------------------------------------------
# Cycle-level weight-stationary model (contract section 2), used to validate the dataflow text.
# ---------------------------------------------------------------------------------------------
def ws_cycle_sim(n, blocks, bubble_pct=0, wgap_pct=0, seed=0):
    """blocks: list of (tile, rows). Returns (outputs, stats). Models:
       input skew (row k delayed k), PE regs (a/valid/swap right, psum down), shadow chain fed at
       row 0 with FIFO words (bottom row first), swap travelling with the first row of a block,
       loader guard (no shift until 2N-2 cycles after the swap row entered row 0), output deskew."""
    r = random.Random(seed)
    words = [wd for t, _ in blocks for wd in g.tile_fifo_words(t)]
    rows = [(row, i == 0) for _, rs in blocks for i, row in enumerate(rs)]
    total_rows = len(rows)
    Z = (0, False, False)
    a = [[Z] * n for _ in range(n)]                 # (val, valid, swap) registered, to the right
    ps = [[(0, False)] * n for _ in range(n)]        # (psum, valid) registered, downward
    shadow = [[0] * n for _ in range(n)]
    active = [[0] * n for _ in range(n)]
    skew = [[Z] * k for k in range(n)]               # row k delay line, length k
    deskew = [[(0, False)] * (n - 1 - c) for c in range(n)]
    out, wi, ri = [], 0, 0
    shifted, tile_loaded, guard = 0, False, 0
    stats = dict(weight_stall=0, weight_shift=0, cycles=0)
    while len(out) < total_rows:
        stats["cycles"] += 1
        assert stats["cycles"] < 100000, "simulation did not finish"
        # ---- loader (decided from current state) ----
        can_shift = (not tile_loaded) and guard == 0 and wi < len(words) and r.randrange(100) >= wgap_pct
        # ---- input acceptance ----
        inj = None
        if ri < total_rows and r.randrange(100) >= bubble_pct:
            row, new_w = rows[ri]
            if new_w and not tile_loaded:
                stats["weight_stall"] += 1
            else:
                inj = (row, new_w); ri += 1
                if new_w:
                    tile_loaded, guard = False, 2 * n - 1
        # ---- skew: entries entering each row this cycle ----
        left = []
        for k in range(n):
            ent = (inj[0][k], True, inj[1]) if inj else Z
            if k == 0:
                left.append(ent)
            else:
                left.append(skew[k][0]); skew[k] = skew[k][1:] + [ent]
        # ---- PEs (synchronous: read old regs) ----
        na = [[Z] * n for _ in range(n)]
        nps = [[(0, False)] * n for _ in range(n)]
        for k in range(n):
            for c in range(n):
                val, vld, sw = left[k] if c == 0 else a[k][c - 1]
                pin = 0 if k == 0 else ps[k - 1][c][0]
                if vld and sw:
                    active[k][c] = shadow[k][c]
                na[k][c] = (val, vld, sw)
                nps[k][c] = (pin + val * active[k][c] if vld else 0, vld)
        # ---- shadow shift (top row gets the new word) ----
        if can_shift:
            assert guard == 0
            for k in range(n - 1, 0, -1):
                shadow[k] = list(shadow[k - 1])
            shadow[0] = list(words[wi]); wi += 1; shifted += 1
            stats["weight_shift"] += 1
            if shifted == n:
                shifted, tile_loaded = 0, True
        if guard:
            guard -= 1
        # ---- bottom row -> deskew -> output ----
        aligned = []
        for c in range(n):
            ent = ps[n - 1][c]
            if n - 1 - c == 0:
                aligned.append(ent)
            else:
                aligned.append(deskew[c][0]); deskew[c] = deskew[c][1:] + [ent]
        if aligned[0][1]:
            assert all(e[1] for e in aligned), "deskew misaligned"
            out.append([e[0] for e in aligned])
        a, ps = na, nps
    return out, stats


def test_cycle_model_matches_contract():
    n = 4
    for name in ("multi_tile", "b1_stream", "extremes"):
        blocks = gv.case_blocks(name, n, seed=4242)
        exp = [y for t, rs in blocks for y in g.gemm(rs, t)]
        for bub, gap, seed in ((0, 0, 1), (30, 40, 2), (60, 70, 3)):
            got, st = ws_cycle_sim(n, blocks, bub, gap, seed)
            assert got == exp, f"{name} bubbles={bub} wgaps={gap}: cycle model != gemm_ref"
            assert st["weight_shift"] == n * len(blocks)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"[PASS] {t.__name__}")
    blocks = gv.case_blocks("multi_tile", 4, seed=4242)
    _, st = ws_cycle_sim(4, blocks, 30, 40, 2)
    print(f"cycle model (N=4 multi_tile, 30% bubbles, 40% w-gaps): {st}")
    print(f"ALL {len(tests)} TESTS PASSED")
