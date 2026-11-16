"""M3 golden model: INT8 x INT8 -> INT32 GEMM and the hex packing of docs/m3_matrix_unit.md.

Contract (sections 1, 2, 4):
  y[b][n] = sum_k x[b][k] * W[k][n]; x, W signed INT8; y signed INT32 (no saturation).
  x row  : lane k at bits [8k+7:8k], two's complement.
  w word : byte n -> column n; tile word t = W[N-1-t][*] (bottom row first).
  y row  : column n at bits [32n+31:32n], two's complement.
  hex    : one value per line, MSB first, no 0x, zero-padded to the full width.
Pure standard library.
"""

INT8_MIN, INT8_MAX = -128, 127


def check_int8(v):
    if not INT8_MIN <= v <= INT8_MAX:
        raise ValueError(f"not int8: {v}")
    return v


def gemm(x_rows, w):
    """x_rows: list of rows (len N) of int8; w: N x N int8 (w[k][n]). Returns int rows (exact)."""
    n = len(w)
    out = []
    for row in x_rows:
        out.append([sum(row[k] * w[k][c] for k in range(n)) for c in range(n)])
    return out


# ---------------- packing ----------------
def pack_lanes(vals, bits):
    """Pack signed lanes (lane i at bits [bits*i+bits-1 : bits*i]) into an unsigned int."""
    mask = (1 << bits) - 1
    word = 0
    for i, v in enumerate(vals):
        word |= (v & mask) << (bits * i)
    return word


def unpack_lanes(word, n, bits):
    mask, sign = (1 << bits) - 1, 1 << (bits - 1)
    out = []
    for i in range(n):
        v = (word >> (bits * i)) & mask
        out.append(v - (1 << bits) if v & sign else v)
    return out


def to_hex(word, bits):
    return format(word, "0%dx" % ((bits + 3) // 4))


def x_row_hex(row):
    return to_hex(pack_lanes([check_int8(v) for v in row], 8), 8 * len(row))


def w_word_hex(wrow):
    return to_hex(pack_lanes([check_int8(v) for v in wrow], 8), 8 * len(wrow))


def y_row_hex(row):
    for v in row:
        if not -(1 << 31) <= v < (1 << 31):
            raise ValueError(f"not int32: {v}")
    return to_hex(pack_lanes(row, 32), 32 * len(row))


def tile_fifo_words(w):
    """Serialise one N x N tile in Weight FIFO order: word t = W[N-1-t]."""
    n = len(w)
    return [list(w[n - 1 - t]) for t in range(n)]


def tile_from_fifo_words(words):
    n = len(words)
    return [list(words[n - 1 - k]) for k in range(n)]
