#!/usr/bin/env python3
"""Self-tests for sim.py (run: python3 model/isa/test_sim.py; pytest-compatible)."""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import asm                    # noqa: E402
import sim                    # noqa: E402

N = 16


def _tile_words(W):
    """W[k][n] int8 -> FIFO words, word t = W[15-t]."""
    return [sim.row_to_int(W[15 - t]) for t in range(16)]


def _prog_one_layer(sync_before_act=True):
    p = [asm.set_config(0x1000 | c, ((1 << 31) - 1)) for c in range(16)]     # M0 ~ 1, shift 0
    p += [asm.set_config(0x3000, 0x7F8000)]                                   # zp 0, [-128,127]
    p += [asm.matmul(0, 0, 4, 0, 1)]
    if sync_before_act:
        p += [asm.sync(asm.U_MATRIX)]
    p += [asm.activate(0, 8, 4, 0), asm.sync(), asm.interrupt_host(1), asm.halt()]
    return p


def test_gemm_and_tile_order():
    rnd = random.Random(1)
    W = [[rnd.randint(-128, 127) for _ in range(N)] for _ in range(N)]
    X = [[rnd.randint(-128, 127) for _ in range(N)] for _ in range(4)]
    s = sim.CoreSim(X + [[0] * N] * 8, _tile_words(W)).run(_prog_one_layer())
    for b in range(4):
        exp = [sum(X[b][k] * W[k][n] for k in range(N)) for n in range(N)]
        assert s.acc[b] == exp
        assert s.ub[8 + b] == [max(-128, min(127, v)) for v in exp]   # M0 = 2^31-1 ~ x1
    assert s.irq_count == 1 and s.halted and s.wpos == 16


def test_hazard_missing_sync_matrix():
    try:
        sim.CoreSim([[1] * N] * 12, [0] * 16).run(_prog_one_layer(sync_before_act=False))
    except sim.HazardError:
        return
    raise AssertionError("missing SYNC(matrix) not detected")


def test_hazard_layer_raw_and_qset():
    p = _prog_one_layer()[:-4] + [asm.activate(0, 8, 4, 0)]          # ACTIVATE writes UB 8..11
    p += [asm.matmul(8, 100, 4, 0, 0)]                               # next layer reads 8..11: RAW
    try:
        sim.CoreSim([[1] * N] * 12, [0] * 16).run(p)
        raise AssertionError("UB RAW across layers not detected")
    except sim.HazardError:
        pass
    p = _prog_one_layer()[:-3] + [asm.set_config(0x3000, 0)]         # rewrite qset 0 while in use
    try:
        sim.CoreSim([[1] * N] * 12, [0] * 16).run(p)
        raise AssertionError("SET_CONFIG on a busy qset not detected")
    except sim.HazardError:
        pass


def test_cfg_err_stops_and_blocks():
    p = _prog_one_layer()[:-3] + [asm.encode_raw(0x05), asm.activate(0, 20, 4, 0), asm.interrupt_host(2)]
    s = sim.CoreSim([[1] * N] * 24, [0] * 16).run(p)
    assert s.cfg_err and s.cfg_err_op == 0x05 and s.irq_count == 0 and s.ub[20] == [1] * N  # marker blocked


def _expect_cfg_err(word, op):
    p = [asm.sync(), word, asm.interrupt_host(7), asm.halt()]
    s = sim.CoreSim([[1] * N] * 4, [0] * 16).run(p)
    assert s.cfg_err and s.cfg_err_op == op and s.irq_count == 0, asm.disasm(word)


def test_count_and_range_cfg_err():
    # B == 0 / rows == 0
    _expect_cfg_err(asm.matmul(0, 0, 0), asm.MATMUL)
    _expect_cfg_err(asm.encode(asm.ACTIVATE, rows=0), asm.ACTIVATE)
    # range overflow, no wrapping
    _expect_cfg_err(asm.matmul(65535, 0, 2), asm.MATMUL)
    _expect_cfg_err(asm.matmul(0, 2047, 2), asm.MATMUL)
    _expect_cfg_err(asm.encode(asm.ACTIVATE, acc_row=2047, rows=2, ub_row=48), asm.ACTIVATE)
    _expect_cfg_err(asm.encode(asm.ACTIVATE, acc_row=0, rows=2, ub_row=65535), asm.ACTIVATE)
    # exact fits are legal (last row of each memory)
    for w in (asm.matmul(65535, 2047, 1), asm.matmul(0, 0, 2048)):
        s = sim.CoreSim([[1] * N] * 65536 if w == asm.matmul(65535, 2047, 1) else [[1] * N] * 2048,
                        [0] * 16).run([w])
        assert not s.cfg_err, asm.disasm(w)
    # ub_operand_row is ignored when eltwise == 0 (not a cfg_err)
    assert sim.CoreSim()._invalid(asm.encode(asm.ACTIVATE, ub_operand_row=123, rows=1)) is None


def test_weight_underflow():
    try:
        sim.CoreSim([[1] * N] * 12, []).run(_prog_one_layer())
        raise AssertionError("weight underflow not detected")
    except RuntimeError:
        pass


def test_generated_cases():
    lst = os.path.join(HERE, "..", "..", "build", "m4", "cases", "cases.txt")
    if not os.path.exists(lst):
        print("  (no build/m4/cases yet; run model/mlp/compile_mlp.py)")
        return
    root = os.path.join(HERE, "..", "..")
    for d in open(lst).read().split():
        errs, _ = sim.check_case(os.path.join(root, d))
        assert not errs, (d, errs)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: PASS")
