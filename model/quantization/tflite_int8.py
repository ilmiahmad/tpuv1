#!/usr/bin/env python3
"""TFLite int8 requantization, bit-exact (docs/m4_compute_core.md section 1).

Double-rounding (gemmlowp) MultiplyByQuantizedMultiplier, as in TFLite reference_ops when
TFLITE_SINGLE_ROUNDING is not defined (the default). Pure Python integers, emulating int32 /
int64 C semantics explicitly.
"""
import math

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def wrap32(x):
    """Two's-complement int32 wrap."""
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x & 0x80000000 else x


def wrap8(x):
    x &= 0xFF
    return x - 256 if x & 0x80 else x


def c_div(a, b):
    """C/C++ integer division: truncates toward zero."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def srdhm(a, b):
    """SaturatingRoundingDoublingHighMul(int32 a, int32 b) -> int32."""
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    ab = a * b                                  # exact; fits int64
    nudge = (1 << 30) if ab >= 0 else (1 - (1 << 30))
    return wrap32(c_div(ab + nudge, 1 << 31))


def rdbpot(x, e):
    """RoundingDivideByPOT(int32 x, e in 0..31)."""
    assert 0 <= e <= 31
    mask = (1 << e) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> e) + (1 if remainder > threshold else 0)   # Python >> is arithmetic


def mbqm(x, m0, shift):
    """MultiplyByQuantizedMultiplier(int32 x, int32 M0, int shift), shift > 0 = left."""
    left = shift if shift > 0 else 0
    right = 0 if shift > 0 else -shift
    return rdbpot(srdhm(wrap32(x << left), m0), right)


def requant(acc, bias, m0, shift, zp_out, act_min, act_max):
    """One channel: acc (int32) -> int8, contract section 1."""
    x = wrap32(acc + bias)
    y = mbqm(x, m0, shift)
    z = wrap32(y + zp_out)
    return max(act_min, min(act_max, z))


def requant_row(acc_row, qparams):
    """acc_row: list of N int32. qparams: dict with lists bias/m0/shift and scalars."""
    return [requant(a, qparams["bias"][c], qparams["m0"][c], qparams["shift"][c],
                    qparams["zp_out"], qparams["act_min"], qparams["act_max"])
            for c, a in enumerate(acc_row)]


def quantize_multiplier(real):
    """TFLite QuantizeMultiplier(double) -> (M0 int32, shift int). M0 in [2^30, 2^31-1] or 0."""
    if real == 0.0:
        return 0, 0
    q, shift = math.frexp(real)                 # real = q * 2^shift, q in [0.5, 1)
    v = q * (1 << 31)
    q_fixed = int(math.copysign(math.floor(abs(v) + 0.5), v))   # std::round: half away from 0
    assert q_fixed <= (1 << 31)
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    if shift > 30:
        shift, q_fixed = 30, (1 << 31) - 1
    return q_fixed, shift


def act_range(kind, scale, zp):
    """TFLite CalculateActivationRangeQuantized for int8 output."""
    qmin, qmax = -128, 127
    if kind == "none":
        return qmin, qmax
    if kind == "relu":
        return max(qmin, zp), qmax
    if kind == "relu6":
        return max(qmin, zp), min(qmax, zp + int(round(6.0 / scale)))
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------------------------
def _ref_srdhm_alt(a, b):
    # independent formulation via Python fractions semantics
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    ab = a * b
    num = ab + ((1 << 30) if ab >= 0 else 1 - (1 << 30))
    q = int(num / (1 << 31)) if abs(num) < (1 << 52) else c_div(num, 1 << 31)
    return wrap32(q)


def test_corners():
    assert srdhm(INT32_MIN, INT32_MIN) == INT32_MAX
    assert srdhm(INT32_MIN, INT32_MAX) == -INT32_MAX       # -2^31*(2^31-1)*2/2^32 rounded
    assert srdhm(INT32_MAX, INT32_MAX) == INT32_MAX - 1
    assert srdhm(0, 12345) == 0
    # truncating division matters for negatives: ab = -(2^30) -> (-(2^30) + 1 - 2^30)/2^31
    assert srdhm(-(1 << 15), 1 << 15) == c_div(-(1 << 30) + 1 - (1 << 30), 1 << 31) == 0
    assert srdhm(-3, 1 << 30) == -1      # ab=-3*2^30; +(1-2^30) -> -(4*2^30)+1 ; /2^31 -> -1 (trunc)
    # RoundingDivideByPOT ties: positive ties round up, negative ties round away from zero
    assert rdbpot(5, 1) == 3 and rdbpot(-5, 1) == -3
    assert rdbpot(6, 2) == 2 and rdbpot(-6, 2) == -2          # 1.5 -> 2, -1.5 -> -2
    assert rdbpot(2, 2) == 1 and rdbpot(-2, 2) == -1          # 0.5 -> 1, -0.5 -> -1
    assert rdbpot(1, 2) == 0 and rdbpot(-1, 2) == 0
    assert rdbpot(INT32_MIN, 31) == -1 and rdbpot(INT32_MAX, 31) == 1
    assert rdbpot(12345, 0) == 12345
    # shift extremes
    assert mbqm(1000, 1 << 30, -31) == 0
    assert mbqm(1 << 24, 1 << 30, 7) == wrap32(srdhm(wrap32((1 << 24) << 7), 1 << 30))
    assert mbqm(100, (1 << 31) - 1, 0) == 100
    assert mbqm(-100, (1 << 31) - 1, 0) == -100
    # clamp
    assert requant(10**6, 0, (1 << 31) - 1, 0, 0, -128, 127) == 127
    assert requant(-10**6, 0, (1 << 31) - 1, 0, 0, -128, 127) == -128


def test_random_against_alt():
    import random
    rnd = random.Random(7)
    for _ in range(200000):
        a = rnd.randint(INT32_MIN, INT32_MAX)
        b = rnd.choice([rnd.randint(1 << 30, INT32_MAX), rnd.randint(INT32_MIN, INT32_MAX)])
        assert srdhm(a, b) == _ref_srdhm_alt(a, b)
    for _ in range(100000):
        x = rnd.randint(INT32_MIN, INT32_MAX)
        e = rnd.randint(0, 31)
        # independent: round-half-away-from-zero of x / 2^e (gemmlowp RoundingDivideByPOT)
        q, r = divmod(abs(x), 1 << e)
        exp = q + (1 if 2 * r >= (1 << e) and e > 0 else 0)
        exp = exp if x >= 0 else -exp
        assert rdbpot(x, e) == exp, (x, e)


def test_quantize_multiplier():
    for real in (0.5, 0.25, 0.75, 1.0, 1e-5, 0.0123, 3.9):
        m0, sh = quantize_multiplier(real)
        assert (1 << 30) <= m0 < (1 << 31)
        assert abs(m0 * 2.0 ** (sh - 31) - real) / real < 1e-9


if __name__ == "__main__":
    test_corners()
    test_random_against_alt()
    test_quantize_multiplier()
    print("tflite_int8 self-tests: PASS")
