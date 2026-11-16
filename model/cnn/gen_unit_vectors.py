#!/usr/bin/env python3
"""Unit-test vectors for the CONVOLVE address generator and the pool unit (contract section 7).

  build/m6/addr/<name>/cfg.txt    key value lines (decimal, pad_val signed):
                                  in_w in_h out_w out_h sh sw pad_t pad_l pad_val kh kw oh0 oh_cnt ub_row
  build/m6/addr/<name>/rows.hex   one 5-hex-digit word per generated row: {pad[16], row[15:0]}
                                  (row = 0 for a padded tap); B = oh_cnt*out_w lines, in generation order
  build/m6/addr/cases.txt         one case directory (relative to the repo root) per line

  build/m6/pool/<name>/in.hex     4*G lines of 32 hex digits (lane c = bits [8c+7:8c], int8): G groups of
                                  four requantized rows in POOL ORDER (2i,2j),(2i,2j+1),(2i+1,2j),(2i+1,2j+1)
  build/m6/pool/<name>/out.hex    G lines: the pooled rows, group order = output order (i outer, j inner)
  build/m6/pool/<name>/meta.txt   mode max|avg, groups G; raster-form cases also: pool_w pool_h and raster.hex
  build/m6/pool/cases.txt
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import conv_ref as cr                                              # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
N = 16


def out_dim(i, k, s, p):
    return (i + 2 * p - k) // s + 1


def write_addr_case(root, name, g, kh, kw, oh0, oh_cnt, ub_row, pad_val):
    assert cr.conv_legal(g, oh0, oh_cnt, 0, ub_row) is None, (name, cr.conv_legal(g, oh0, oh_cnt, 0, ub_row))
    rows = cr.conv_rows(g["in_w"], g["in_h"], g["out_w"], g["out_h"], g["sh"], g["sw"],
                        g["pad_t"], g["pad_l"], kh, kw, oh0, oh_cnt, ub_row)
    d = os.path.join(root, "addr", name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cfg.txt"), "w") as fh:
        for k, v in (("in_w", g["in_w"]), ("in_h", g["in_h"]), ("out_w", g["out_w"]), ("out_h", g["out_h"]),
                     ("sh", g["sh"]), ("sw", g["sw"]), ("pad_t", g["pad_t"]), ("pad_l", g["pad_l"]),
                     ("pad_val", pad_val), ("kh", kh), ("kw", kw), ("oh0", oh0), ("oh_cnt", oh_cnt),
                     ("ub_row", ub_row)):
            fh.write(f"{k} {v}\n")
    with open(os.path.join(d, "rows.hex"), "w") as fh:
        for pad, row in rows:
            fh.write(f"{(int(pad) << 16) | row:05x}\n")
    return len(rows)


def gen_addr(root):
    rnd = random.Random(2026)
    names, nrows = [], 0

    def add(geo, g, taps, windows, ub_row=0, pad_val=0):
        nonlocal nrows
        for kh, kw in taps:
            for oh0, cnt in windows:
                nm = f"{geo}_kh{kh}_kw{kw}_oh{oh0}n{cnt}"
                nrows += write_addr_case(root, nm, g, kh, kw, oh0, cnt, ub_row, pad_val)
                names.append(f"build/m6/addr/{nm}")

    def geom(in_h, in_w, k_h, k_w, s_h, s_w, p_t, p_l, out_h=None, out_w=None):
        return dict(in_w=in_w, in_h=in_h, sh=s_h, sw=s_w, pad_t=p_t, pad_l=p_l,
                    out_h=out_h if out_h is not None else out_dim(in_h, k_h, s_h, p_t),
                    out_w=out_w if out_w is not None else out_dim(in_w, k_w, s_w, p_l))

    def taps(kh, kw):
        return [(a, b) for a in range(kh) for b in range(kw)]

    g = geom(8, 8, 3, 3, 1, 1, 1, 1);  add("s1p1_8x8", g, taps(3, 3), [(0, g["out_h"])], 0, -7)
    g = geom(8, 8, 3, 3, 2, 2, 1, 1);  add("s2p1_8x8", g, taps(3, 3), [(0, g["out_h"])], 40, 127)
    g = geom(8, 8, 3, 3, 1, 1, 0, 0);  add("valid_8x8", g, taps(3, 3), [(0, g["out_h"])], 0, 0)
    g = geom(6, 10, 3, 3, 1, 1, 1, 1); add("nonsq_6x10", g, taps(3, 3), [(0, g["out_h"])], 9, -128)
    g = geom(8, 8, 1, 1, 1, 1, 0, 0);  add("k1_8x8", g, taps(1, 1), [(0, 8)], 0, 3)
    g = geom(8, 8, 5, 5, 1, 1, 2, 2);  add("k5p2_8x8", g, taps(5, 5), [(0, 8)], 0, -100)
    g = geom(16, 16, 3, 3, 1, 1, 1, 1)
    add("split_16x16", g, taps(3, 3), [(0, 4), (4, 4), (8, 4), (12, 4)], 256, -5)
    g = geom(12, 12, 3, 3, 3, 3, 0, 0); add("s3_12x12", g, taps(3, 3), [(0, 4)], 0, 11)
    g = geom(10, 10, 3, 3, 2, 2, 3, 3); add("p3s2_10x10", g, taps(3, 3), [(0, 7), (2, 3)], 1000, -128)
    # asymmetric pad / output sizes that disagree with the textbook formula (generator is parametric)
    g = geom(5, 5, 3, 3, 1, 1, 1, 2, out_h=9, out_w=5); add("asym_5x5", g, taps(3, 3), [(0, 9), (3, 5)], 0, 1)
    g = geom(7, 7, 3, 3, 1, 1, 2, 0, out_h=7, out_w=7); add("asym_7x7", g, taps(3, 3), [(0, 7)], 0, 2)
    # all-padding rows: 1x1 image with large pad
    g = geom(1, 1, 7, 7, 1, 1, 3, 3, out_h=7, out_w=7);  add("img1x1_p3", g, [(0, 0), (3, 3), (6, 6), (0, 6)], [(0, 7), (3, 2)], 5, 9)
    # exact fit at the top of the UB and a window starting late
    g = geom(8, 8, 3, 3, 1, 1, 1, 1);  add("topfit_8x8", g, [(0, 0), (1, 1), (2, 2)], [(0, 8)], 65536 - 64, 0)
    # maximum window: B = 2048 (64 columns x 32 rows), big image near the top of the UB
    g = geom(64, 64, 3, 3, 1, 1, 1, 1)
    add("big_64x64", g, [(0, 0), (1, 1), (2, 2)], [(0, 32), (32, 32)], 60000, 77)
    g = geom(20, 20, 4, 4, 3, 3, 2, 2); add("k4s3_20x20", g, [(0, 0), (1, 2), (3, 3)], [(0, 7), (2, 4)], 700, -1)
    # strides / pads sweep: s in 1..3 x pad 0..3, 3x3 kernel, 11x9 image, corner taps
    for s in (1, 2, 3):
        for p in (0, 1, 2, 3):
            g = geom(9, 11, 3, 3, s, s, p, p)
            add(f"sweep_s{s}p{p}", g, [(0, 0), (0, 2), (2, 0), (2, 2), (1, 1)], [(0, g["out_h"])], 0, 4)
    # random geometries, legal by construction (fixed seed)
    for i in range(40):
        in_h, in_w = rnd.randint(1, 40), rnd.randint(1, 40)
        sh, sw = rnd.choice([1, 1, 1, 2, 2, 3, 4, 15]), rnd.choice([1, 1, 1, 2, 2, 3, 4, 15])
        p_t, p_l = rnd.randint(0, 15), rnd.randint(0, 15)
        out_h, out_w = rnd.randint(1, 40), rnd.randint(1, 40)
        g = dict(in_w=in_w, in_h=in_h, sh=sh, sw=sw, pad_t=p_t, pad_l=p_l, out_h=out_h, out_w=out_w)
        oh0 = rnd.randint(0, out_h - 1)
        cnt = rnd.randint(1, min(out_h - oh0, 2048 // out_w))
        kh, kw = rnd.randint(0, 15), rnd.randint(0, 15)
        ub = rnd.randint(0, 65536 - in_h * in_w)
        add(f"rand{i:02d}", g, [(kh, kw)], [(oh0, cnt)], ub, rnd.randint(-128, 127))
    with open(os.path.join(root, "addr", "cases.txt"), "w") as fh:
        fh.write("\n".join(names) + "\n")
    return len(names), nrows


# ---------------------------------------------------------------------------------------------
def row_int(row):
    v = 0
    for c, x in enumerate(row):
        v |= (x & 255) << (8 * c)
    return v


def write_pool_case(root, name, mode, groups, extra=None, raster=None):
    d = os.path.join(root, "pool", name)
    os.makedirs(d, exist_ok=True)
    out = [cr.pool_group(g, mode) for g in groups]
    with open(os.path.join(d, "in.hex"), "w") as fh:
        for g in groups:
            for r in g:
                fh.write(f"{row_int(r):032x}\n")
    with open(os.path.join(d, "out.hex"), "w") as fh:
        for r in out:
            fh.write(f"{row_int(r):032x}\n")
    with open(os.path.join(d, "meta.txt"), "w") as fh:
        fh.write(f"mode {mode}\ngroups {len(groups)}\n")
        for k, v in (extra or {}).items():
            fh.write(f"{k} {v}\n")
    if raster is not None:
        with open(os.path.join(d, "raster.hex"), "w") as fh:
            for r in raster:
                fh.write(f"{row_int(r):032x}\n")
    return len(groups)


def gen_pool(root):
    rnd = random.Random(7)
    names, total = [], 0

    def rrow(lo=-128, hi=127):
        return [rnd.randint(lo, hi) for _ in range(N)]

    def add(name, mode, groups, **kw):
        nonlocal total
        total += write_pool_case(root, name, mode, groups, **kw)
        names.append(f"build/m6/pool/{name}")

    for mode in ("max", "avg"):
        add(f"{mode}_rand", mode, [[rrow() for _ in range(4)] for _ in range(300)])
        add(f"{mode}_small_range", mode, [[rrow(-3, 3) for _ in range(4)] for _ in range(200)])
        add(f"{mode}_all_min", mode, [[[-128] * N for _ in range(4)] for _ in range(3)])
        add(f"{mode}_all_max", mode, [[[127] * N for _ in range(4)] for _ in range(3)])
        add(f"{mode}_mixed_extreme", mode,
            [[[(-128 if (c + k) % 2 else 127) for c in range(N)] for k in range(4)] for _ in range(4)]
            + [[[(-128 if k < 2 else 127) for c in range(N)] for k in range(4)]]
            + [[[(127 if k == j else -128) for c in range(N)] for k in range(4)] for j in range(4)])
        add(f"{mode}_ties", mode, [[[rnd.choice([-5, 5]) for _ in range(N)]] * 4 for _ in range(50)]
            + [[rrow(-2, 2)] * 4 for _ in range(20)])
        add(f"{mode}_negative", mode, [[rrow(-128, -1) for _ in range(4)] for _ in range(100)])
    # avg: every possible 4-value sum -512..508 exercised on every lane (round half away from zero)
    sums = list(range(-512, 509))
    groups = []
    for base in range(0, len(sums), N):
        chunk = sums[base:base + N]
        chunk += [0] * (N - len(chunk))
        g = [[0] * N for _ in range(4)]
        for c, s in enumerate(chunk):
            # constructive random split of s into four int8 values (always feasible for |s| in range)
            rem, v = s, []
            for k in range(3):
                left = 3 - k                      # values still to choose after this one
                lo = max(-128, rem - 127 * left)
                hi = min(127, rem + 128 * left)
                x = rnd.randint(lo, hi)
                v.append(x); rem -= x
            assert -128 <= rem <= 127
            v.append(rem)
            for k in range(4):
                g[k][c] = v[k]
        groups.append(g)
    add("avg_sum_sweep", "avg", groups)
    # raster form (also usable for act_seq read-order tests)
    for mode, pw, ph in (("max", 4, 4), ("avg", 6, 4), ("max", 8, 2), ("avg", 2, 8)):
        raster = [rrow() for _ in range(pw * ph)]
        ordered = cr.pool_group_order(raster, pw)
        groups = [ordered[i:i + 4] for i in range(0, len(ordered), 4)]
        pooled = cr.pool_raster(raster, pw, mode)
        assert pooled == [cr.pool_group(g, mode) for g in groups]
        add(f"raster_{mode}_{pw}x{ph}", mode, groups, extra=dict(pool_w=pw, pool_h=ph), raster=raster)
    with open(os.path.join(root, "pool", "cases.txt"), "w") as fh:
        fh.write("\n".join(names) + "\n")
    return len(names), total


def main():
    root = os.path.join(ROOT, "build", "m6")
    os.makedirs(os.path.join(root, "addr"), exist_ok=True)
    os.makedirs(os.path.join(root, "pool"), exist_ok=True)
    n, rows = gen_addr(root)
    print(f"addr: {n} cases, {rows} generated rows -> build/m6/addr/cases.txt")
    n, groups = gen_pool(root)
    print(f"pool: {n} cases, {groups} groups -> build/m6/pool/cases.txt")


if __name__ == "__main__":
    main()
