#!/usr/bin/env python3
"""Generate M3 matrix-unit vectors (docs/m3_matrix_unit.md section 4).

  python3 model/gemm/gen_vectors.py [--out build/m3/vectors] [--n 4 8 16]

Per case dir <out>/<case>_N<N>/: meta.txt, x.hex, xnew.hex, w.hex, y.hex.
<out>/cases.txt lists the case dirs relative to the repo root (one per line).
"""
import argparse, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemm_ref as g  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CASES = ["single", "extremes", "multi_tile", "b1_stream", "random_long"]


def rand_mat(r, rows, cols):
    return [[r.randint(-128, 127) for _ in range(cols)] for _ in range(rows)]


def case_blocks(name, n, seed):
    """Return list of (tile W[k][n], x rows) — each block starts with a new tile."""
    r = random.Random(seed)
    if name == "single":
        return [(rand_mat(r, n, n), rand_mat(r, 16, n))]
    if name == "extremes":
        neg = [[-128] * n for _ in range(n)]
        pos = [[127] * n for _ in range(n)]
        mix_w = [[-128 if (k + c) % 2 else 127 for c in range(n)] for k in range(n)]
        mix_x = [[127 if k % 2 else -128 for k in range(n)], [-128 if k % 2 else 127 for k in range(n)],
                 [-128] * n, [127] * n, [0] * n]
        return [(neg, [[-128] * n] * 4 + [[127] * n] * 2),
                (pos, [[127] * n] * 4 + [[-128] * n] * 2),
                (mix_w, mix_x),
                (neg, [[-128] * n] * 3)]
    if name == "multi_tile":
        return [(rand_mat(r, n, n), rand_mat(r, b, n)) for b in (1, 2, 7, 16, 33, 64)]
    if name == "b1_stream":
        return [(rand_mat(r, n, n), rand_mat(r, 1, n)) for _ in range(20)]
    if name == "random_long":
        sizes = [r.randint(1, 50) for _ in range(40)]
        while sum(sizes) < 1000:                      # contract: 1000+ rows
            sizes[r.randrange(40)] += r.randint(1, 50)
        return [(rand_mat(r, n, n), rand_mat(r, b, n)) for b in sizes]
    raise ValueError(name)


def write_case(out_dir, name, n, seed):
    blocks = case_blocks(name, n, seed)
    d = os.path.join(out_dir, f"{name}_N{n}")
    os.makedirs(d, exist_ok=True)
    xs, xnew, ws, ys = [], [], [], []
    for w, rows in blocks:
        ws += [g.w_word_hex(word) for word in g.tile_fifo_words(w)]
        for i, (row, yrow) in enumerate(zip(rows, g.gemm(rows, w))):
            xs.append(g.x_row_hex(row)); xnew.append("1" if i == 0 else "0"); ys.append(g.y_row_hex(yrow))
    for fn, lines in (("x.hex", xs), ("xnew.hex", xnew), ("w.hex", ws), ("y.hex", ys)):
        with open(os.path.join(d, fn), "w") as f:
            f.write("\n".join(lines) + "\n")
    with open(os.path.join(d, "meta.txt"), "w") as f:
        f.write(f"N {n}\nrows {len(xs)}\nwords {len(ws)}\nblocks {len(blocks)}\n")
    return d, len(xs), len(ws), len(blocks)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "build", "m3", "vectors"))
    ap.add_argument("--n", type=int, nargs="+", default=[4, 8, 16])
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    listed = []
    for n in a.n:
        for ci, name in enumerate(CASES):
            d, rows, words, blocks = write_case(out, name, n, seed=1000 * n + ci)
            listed.append(os.path.relpath(d, REPO))
            print(f"{os.path.relpath(d, REPO)}: rows={rows} words={words} blocks={blocks}")
    with open(os.path.join(out, "cases.txt"), "w") as f:
        f.write("\n".join(listed) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
