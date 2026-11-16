#!/usr/bin/env python3
"""Host-side tests for the user-space driver stack (no hardware): ABI layout vs the C header,
instruction encoders vs model/isa/asm.py, bundle packing/layout, numpy and pure-Python paths, the
mock driver semantics, and the full on-board test program run against the mock.

    python3 software/runtime/tests/test_host.py          (pytest also works)
"""
import errno
import json
import random
import subprocess
import struct
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent
ROOT = RUNTIME.parents[1]
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(ROOT / "model/isa"))
sys.path.insert(0, str(ROOT / "model/mlp"))
sys.path.insert(0, str(ROOT / "model/quantization"))
import tpu                                    # noqa: E402
import tpu_isa as isa                         # noqa: E402
import tpu_mlp                                # noqa: E402
import tpu_mock                               # noqa: E402
import asm                                    # noqa: E402

try:
    import numpy as np
except ImportError:                           # pragma: no cover
    np = None

BUNDLES = ROOT / "build/bundles"


def ensure_bundles():
    if not (BUNDLES / "digits_b32/model.json").exists() or not (BUNDLES / "mlp3/model.json").exists():
        for script in ("export_bundle.py", "train_demo.py"):
            subprocess.run([sys.executable, str(ROOT / "model/mlp" / script)], check=True, capture_output=True)


def new_dev():
    return tpu.Device(backend=tpu_mock.MockKernel().backend())


# ------------------------------------------------------------------------------------------------
def test_ioctl_numbers_and_sizes():
    assert (tpu.INFO_SIZE, tpu.ALLOC_SIZE, tpu.RELOC_SIZE, tpu.RUN_SIZE) == (112, 32, 16, 128)
    assert hex(tpu.TPU_IOC_GET_INFO) == "0x80705400"
    assert hex(tpu.TPU_IOC_ALLOC) == "0xc0205401"
    assert hex(tpu.TPU_IOC_FREE) == "0x40045402"
    assert hex(tpu.TPU_IOC_RUN) == "0xc0805403"
    assert hex(tpu.TPU_IOC_RECOVER) == "0x5404"


def test_c_abi_selfcheck():
    try:
        bad = tpu.c_selfcheck()
    except RuntimeError as e:                 # no gcc: nothing to compare against
        print(f"  (skipped: {e})")
        return
    assert bad == [], bad


def test_decode_run_layout():
    vals = (0, 0, 3, 2, 0, 0, tpu.RUN_CFG_ERR, 0x10, 0x4, 0x8, 0xABCD, *range(10, 18), 123456, 0, 0, 0, 0)
    raw = bytearray(struct.pack(tpu.RUN_FMT, *vals))
    r = tpu.Device._decode_run(raw)
    assert (r.status, r.cfg_err_op, r.core_status, r.hw_status, r.irq_tag, r.wall_ns) == (1, 0x10, 4, 8, 0xABCD, 123456)
    assert list(r.perf.items()) == list(zip(tpu.PERF_NAMES, range(10, 18)))
    assert abs(r.wall_us - 123.456) < 1e-9 and not r.ok and r.status_name == "CFG_ERR"


def test_isa_encoders_match_assembler():
    rnd = random.Random(1)
    for _ in range(300):
        ub, acc, n = rnd.randrange(65536), rnd.randrange(2048), rnd.randrange(65536)
        a, nw = rnd.randrange(2), rnd.randrange(2)
        assert isa.matmul(ub, acc, n, a, nw) == asm.matmul(ub, acc, n, a, nw)
        q, f = rnd.randrange(4), rnd.randrange(3)
        assert isa.activate(acc, ub, n, q, f) == asm.activate(acc, ub, n, q, f)
        tgt, val = rnd.randrange(65536), rnd.getrandbits(64)
        assert isa.set_config(tgt, val) == asm.set_config(tgt, val)
        m, tag = rnd.randrange(16), rnd.getrandbits(32)
        assert isa.sync(m) == asm.sync(m) and isa.interrupt_host(tag) == asm.interrupt_host(tag)
        addr, rl, reps = rnd.randrange(1 << 45) * 16, rnd.randrange(1, 4096), rnd.randrange(1, 4096)
        st = rnd.choice([None, rnd.randrange(1 << 24) * 16])
        assert isa.read_host_mem(ub, addr, rl, reps, st) == asm.read_host_mem(ub, addr, rl, reps, st)
        assert isa.write_host_mem(ub, addr, rl, reps, st) == asm.write_host_mem(ub, addr, rl, reps, st)
        tiles = rnd.randrange(65536)
        assert isa.read_weights(addr, tiles) == asm.read_weights(addr, tiles)
    assert isa.halt() == asm.halt() and isa.nop() == 0
    assert isa.to_bytes([isa.halt()]) == asm.to_bytes(asm.halt())


# ---- mock driver semantics ------------------------------------------------------------------------
def test_mock_buffers_and_limits():
    k = tpu_mock.MockKernel()
    d1, d2 = tpu.Device(backend=k.backend()), tpu.Device(backend=k.backend())
    b = d1.alloc(5000)
    assert b.alloc_size == 8192 and b.handle > 0 and b.mmap_offset == b.handle << 12
    b.write(b"hello")
    assert b.read(5) == b"hello"
    assert d2.info().total_buf_bytes == 8192
    for bad in (0,):
        try:
            d1.alloc(bad)
            assert False
        except OSError as e:
            assert e.errno == errno.EINVAL
    try:
        d1.alloc(k.max_buf + 4096)
        assert False
    except OSError as e:
        assert e.errno == errno.ENOSPC
    try:
        d2.free_handle(b.handle)              # not the owner
        assert False
    except OSError as e:
        assert e.errno == errno.ENOENT
    d1.close()
    assert d2.info().total_buf_bytes == 0     # freed with the descriptor
    d2.close()


def test_mock_validation_rules():
    with new_dev() as dev:
        buf = dev.alloc(4096)
        dma = isa.to_bytes([isa.read_host_mem(0, 0x810000000, 1), isa.halt()])
        cases = [
            (lambda: dev.run(dma, []), errno.EINVAL),                                   # DMA without reloc
            (lambda: dev.run(dma, [(1, buf, 0)]), errno.EINVAL),                        # reloc on HALT
            (lambda: dev.run(dma, [(7, buf, 0)]), errno.EINVAL),                        # index out of range
            (lambda: dev.run(dma, [(0, buf, 8)]), errno.EINVAL),                        # misaligned
            (lambda: dev.run(dma, [(0, buf, 8192)]), errno.EINVAL),                     # beyond
            (lambda: dev.run(dma, [(0, 0x7777, 0)]), errno.ENOENT),                     # unknown handle
        ]
        for fn, want in cases:
            try:
                fn()
                assert False, "accepted"
            except OSError as e:
                assert e.errno == want, (e.errno, want)
        assert dev.info().jobs_ok == 0 and dev.info().jobs_failed == 0
        ok = dev.run(isa.to_bytes([isa.sync(), isa.interrupt_host(9), isa.halt()]))
        assert ok.ok and ok.irq_tag == 9


# ---- bundle packing / layout / paths ---------------------------------------------------------------
def _rand_q(rnd, b, k):
    return [[rnd.randrange(-128, 128) for _ in range(k)] for _ in range(b)]


def test_bundles_golden_all():
    ensure_bundles()
    names = sorted(p.parent.name for p in BUNDLES.glob("*/model.json"))
    assert {"fc1", "mlp3", "mlp_b1", "mlp_wide", "digits_b1", "digits_b8", "digits_b32", "digits_b128"} <= set(names)
    k = tpu_mock.MockKernel()
    for n in names:
        with tpu.Device(backend=k.backend()) as dev, tpu_mlp.TpuMlp(dev, BUNDLES / n) as net:
            res, exact = net.run_sample()
            assert exact, f"{n}: {res}"


def test_infer_matches_golden_and_layout(use_numpy=True):
    ensure_bundles()
    saved = tpu_mlp.np
    tpu_mlp.np = saved if use_numpy else None
    try:
        xq = np.load(BUNDLES / "digits_test_xq.npy")
        gold = np.load(BUNDLES / "digits_test_golden.npy")
        with new_dev() as dev, tpu_mlp.TpuMlp(dev, BUNDLES / "digits_b8") as net:
            B = net.B
            for n in (8, 5, 1):                                 # full batch, padded batch, single sample
                x = xq[:n].tolist() if not use_numpy else xq[:n]
                y = net.infer_q(x)
                got = np.asarray(y)
                assert got.shape == (n, 10) and (got == gold[:n]).all(), f"batch {n}"
                # independent check of the input layout: row kt*B + b, lane l  <->  x[b][16kt+l] (pad = zp)
                raw = net.in_buf.read(net.model["in_rows"] * 16)
                zp = net.model["in_zp"] & 0xFF
                for kt in range(4):
                    for b in range(B):
                        for lane in range(16):
                            want = (int(xq[b, 16 * kt + lane]) & 0xFF) if b < n else zp
                            assert raw[(kt * B + b) * 16 + lane] == want, (n, kt, b, lane)
    finally:
        tpu_mlp.np = saved


def test_infer_pure_python_path():
    test_infer_matches_golden_and_layout(use_numpy=False)


def test_numpy_and_pure_paths_agree_incl_float():
    ensure_bundles()
    xf = np.load(BUNDLES / "digits_test_x.npy")[:8]
    with new_dev() as dev, tpu_mlp.TpuMlp(dev, BUNDLES / "digits_b8") as net:
        y_np = net.infer(xf)
        q_np = net.quantize(xf)
        saved = tpu_mlp.np
        tpu_mlp.np = None
        try:
            q_py = net.quantize(xf.tolist())
            y_py = net.infer(xf.tolist())
        finally:
            tpu_mlp.np = saved
        assert (np.asarray(q_py) == q_np).all()
        assert np.array_equal(np.asarray(y_py), y_np), "float path differs between numpy and pure Python"


def test_batch_and_input_rules():
    ensure_bundles()
    with new_dev() as dev, tpu_mlp.TpuMlp(dev, BUNDLES / "digits_b8") as net:
        rnd = random.Random(3)
        for bad in (_rand_q(rnd, 9, 64), _rand_q(rnd, 2, 63), [], [[200] + [0] * 63]):
            try:
                net.infer_q(bad)
                assert False, "accepted invalid input"
            except ValueError:
                pass
        assert net.infer_q(_rand_q(rnd, 3, 64)).shape == (3, 10)


def test_k_not_multiple_of_16_and_many():
    """Network with K=20 inputs and N=10 outputs: exercises K padding both paths + infer_q_many."""
    import compile_mlp as cm
    import export_bundle as eb
    net = cm.Net([20, 24, 10], ["relu", "none"], seed=5)
    rnd = np.random.default_rng(9)
    x = rnd.uniform(-1, 1, (5, 20))
    xq_pad = eb.quantize_input(net, x)
    with tempfile.TemporaryDirectory() as d:
        eb.build_bundle(net, "k20", 5, xq_pad, Path(d) / "k20")
        gold = net.int_forward(xq_pad)[:, :10]
        for use_numpy in (True, False):
            saved = tpu_mlp.np
            tpu_mlp.np = saved if use_numpy else None
            try:
                with new_dev() as dev, tpu_mlp.TpuMlp(dev, Path(d) / "k20") as tm:
                    xin = xq_pad[:, :20].astype(np.int64)
                    got = tm.infer_q(xin if use_numpy else xin.tolist())
                    assert (np.asarray(got) == gold).all(), f"numpy={use_numpy}"
                    many = tm.infer_q_many(np.concatenate([xin, xin]) if use_numpy else xin.tolist() * 2)
                    assert (np.asarray(many) == np.concatenate([gold, gold])).all()
            finally:
                tpu_mlp.np = saved


def test_failed_job_raises():
    ensure_bundles()
    with new_dev() as dev, tpu_mlp.TpuMlp(dev, BUNDLES / "fc1") as net:
        net.bundle.program = net.bundle.program[:-16]            # drop HALT: no completion
        try:
            net.infer_q(_rand_q(random.Random(1), 2, 16))
            assert False
        except tpu_mlp.TpuRunError as e:
            assert e.result.status == tpu.RUN_TIMEOUT


# ---- the on-board test program against the mock ----------------------------------------------------
def test_board_suite_on_mock():
    ensure_bundles()
    import board_test_driver as bt
    rc = bt.main(["--mock", "--bundles", str(BUNDLES), "--only", "basic,negatives,recovery,timeout,loop",
                  "--loops", "10"])
    assert rc == 0


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    bad = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception:
            bad += 1
            print(f"[FAIL] {name}")
            traceback.print_exc()
    print(f"{len(tests)} tests, {bad} failed")
    print("TPU_HOST_TESTS_PASS" if not bad else f"TPU_HOST_TESTS_FAIL ({bad})")
    return 1 if bad else 0


sys.path.insert(0, str(HERE))
if __name__ == "__main__":
    sys.exit(main())
