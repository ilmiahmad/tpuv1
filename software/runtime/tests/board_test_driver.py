#!/usr/bin/env python3
"""On-board test of the TPU Linux driver (/dev/tpu0) through tpu.py / tpu_mlp.py.

    python3 board_test_driver.py --bundles ~/bundles [--only basic,negatives] [--loops 2000]
    python3 board_test_driver.py --mock --bundles build/bundles    # same checks on the host mock

Tests (--only takes a comma list): basic negatives bundles recovery timeout loop concurrent signals
One [PASS]/[FAIL] line per check, final TPU_DRIVER_TEST_PASS or TPU_DRIVER_TEST_FAIL.
Exit codes: 0 pass, 1 failures, 2 no device / no bundles.
"""
import argparse
import errno
import json
import multiprocessing
import os
import signal
import struct
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tpu                       # noqa: E402
import tpu_isa as isa            # noqa: E402
import tpu_mlp                   # noqa: E402

try:
    import numpy as np
except ImportError:              # pragma: no cover
    np = None

ALL_TESTS = ("basic", "negatives", "bundles", "recovery", "timeout", "loop", "concurrent", "signals")
BUNDLES = ("fc1", "mlp3", "mlp_b1", "mlp_wide", "digits_b1", "digits_b8", "digits_b32", "digits_b128")
NOMINAL = 0x810000000                      # any DDR-high address; the kernel replaces it


class Ctx:
    def __init__(self, args):
        self.path, self.mock = args.device, args.mock
        self.bundles = Path(args.bundles) if args.bundles else None
        self.loops, self.procs = args.loops, args.procs
        self.digits_limit = args.digits_limit
        self.kernel = None
        if self.mock:
            import tpu_mock
            self.kernel = tpu_mock.MockKernel()
            self.loops = min(self.loops, 60)
            self.digits_limit = self.digits_limit or 96
        self.fails = 0
        self.checks = 0

    def open(self):
        return tpu.Device(backend=self.kernel.backend()) if self.mock else tpu.Device(self.path)

    def check(self, name, cond, detail=""):
        self.checks += 1
        if cond:
            print(f"[PASS] {name}" + (f" ({detail})" if detail else ""), flush=True)
        else:
            self.fails += 1
            print(f"[FAIL] {name}" + (f": {detail}" if detail else ""), flush=True)
        return cond

    def bundle(self, name):
        if self.bundles and (self.bundles / name / "model.json").exists():
            return self.bundles / name
        return None


def counters(dev):
    i = dev.info()
    return (i.jobs_ok, i.jobs_failed, i.irq_count)


def expect_errno(ctx, dev, name, errnos, fn):
    """fn must raise OSError with errno in `errnos` and must not touch the job counters."""
    before = counters(dev)
    try:
        fn()
        ctx.check(name, False, "no error raised")
        return
    except OSError as e:
        ok = e.errno in errnos
        ctx.check(name, ok and counters(dev) == before,
                  "" if ok else f"errno {e.errno} ({os.strerror(e.errno) if e.errno else e}), want {sorted(errnos)}")
        if ok and counters(dev) != before:
            ctx.check(name + " (no hardware effect)", False, "job counters changed")


def tiny_ok_program(tag=0x7E57):
    return isa.to_bytes([isa.sync(isa.U_ALL), isa.interrupt_host(tag), isa.halt()])


def run_tiny_ok(ctx, dev, name):
    res = dev.run(tiny_ok_program(0x1234))
    return ctx.check(name, res.ok and res.irq_tag == 0x1234, str(res))


# =============================================================================================
def t_basic(ctx):
    with ctx.open() as dev:
        i = dev.info()
        ctx.check("info: ABI/limits/identity",
                  i.abi_version == tpu.ABI_VERSION and 0x400 <= i.hw_version < 0x600 and i.build_id != 0
                  and i.max_instr == 2048 and i.ub_rows == 65536 and i.acc_rows == 2048 and i.max_buf_bytes > 0
                  and i.total_buf_limit >= i.max_buf_bytes, str(i))
        ctx.check("info: device idle", i.state == tpu.STATE_IDLE, i.state_name)
        # alloc / mmap / write / read / free
        b = dev.alloc(4096)
        ctx.check("alloc 4096: handle/size", b.handle > 0 and b.alloc_size == 4096 and b.mmap_offset % 4096 == 0)
        pat = bytes((i * 7 + 3) & 255 for i in range(4096))
        b.write(pat)
        ctx.check("mmap write/read pattern", b.read() == pat)
        if np is not None:
            v = b.np("uint8")
            ctx.check("numpy view", bytes(v[:16]) == pat[:16])
            del v
        b2 = dev.alloc(5000)
        ctx.check("alloc 5000 rounds to pages", b2.alloc_size == 8192, f"alloc_size={b2.alloc_size}")
        ctx.check("fresh buffer is zero filled", dev.alloc(4096).read() == bytes(4096))
        b2.free()
        expect_errno(ctx, dev, "alloc(0) -> EINVAL", {errno.EINVAL}, lambda: dev.alloc(0))
        expect_errno(ctx, dev, "alloc(> max_buf) -> ENOSPC", {errno.ENOSPC}, lambda: dev.alloc(i.max_buf_bytes + 4096))
        raw = bytearray(struct.pack(tpu.ALLOC_FMT, 4096, 1, 0, 0, 0))                 # flags != 0
        expect_errno(ctx, dev, "alloc flags != 0 -> EINVAL", {errno.EINVAL},
                     lambda: dev._ioctl(tpu.TPU_IOC_ALLOC, raw))
        expect_errno(ctx, dev, "free unknown handle -> ENOENT", {errno.ENOENT}, lambda: dev.free_handle(0x7FFF0001))
        h = b.handle
        b.free()
        expect_errno(ctx, dev, "double free -> ENOENT", {errno.ENOENT}, lambda: dev.free_handle(h))
        try:
            dev._be.mmap(4096, (h << 12))
            ctx.check("mmap of a freed handle fails", False, "mapping succeeded")
        except OSError:
            ctx.check("mmap of a freed handle fails", True)
    # buffers are released when the descriptor closes
    with ctx.open() as d2:
        base = d2.info().total_buf_bytes
        with ctx.open() as d1:
            sizes = [d1.alloc(s).alloc_size for s in (4096, 65536, 100000)]
            ctx.check("driver-wide byte accounting", d2.info().total_buf_bytes == base + sum(sizes),
                      f"{d2.info().total_buf_bytes} vs {base + sum(sizes)}")
        end = time.time() + 2
        while d2.info().total_buf_bytes != base and time.time() < end:
            time.sleep(0.01)
        ctx.check("buffers freed when the fd is closed", d2.info().total_buf_bytes == base,
                  f"{d2.info().total_buf_bytes} vs {base}")


def t_negatives(ctx):
    with ctx.open() as dev:
        bi, bw, bo = dev.alloc(4096), dev.alloc(4096), dev.alloc(4096)
        size = bi.alloc_size
        rhm = lambda rows=1, addr=NOMINAL: isa.read_host_mem(0, addr, rows)
        dma_prog = isa.to_bytes([rhm(), isa.sync(isa.U_DMA), isa.interrupt_host(1), isa.halt()])
        ok_prog = tiny_ok_program()
        # keep every buffer whose address is handed to the kernel alive in a named variable
        ok_buf, dma_buf = bytearray(ok_prog), bytearray(dma_prog)
        big_rel = bytearray(2049 * 16)

        expect_errno(ctx, dev, "zero instructions -> EINVAL", {errno.EINVAL},
                     lambda: dev.run_raw(tpu._addr_of(ok_buf), 0, 0, 0))
        big = bytearray(2049 * 16)
        expect_errno(ctx, dev, "2049 instructions -> EINVAL", {errno.EINVAL},
                     lambda: dev.run_raw(tpu._addr_of(big), 0, 2049, 0))
        expect_errno(ctx, dev, "DMA instruction without relocation -> EINVAL", {errno.EINVAL},
                     lambda: dev.run(dma_prog, []))
        expect_errno(ctx, dev, "relocation on a non-DMA instruction -> EINVAL", {errno.EINVAL},
                     lambda: dev.run(ok_prog, [(0, bi, 0)]))
        expect_errno(ctx, dev, "reloc instr_index out of range -> EINVAL", {errno.EINVAL},
                     lambda: dev.run(dma_prog, [(0, bi, 0), (9, bo, 0)]))
        expect_errno(ctx, dev, "unknown buffer handle", {errno.ENOENT, errno.EINVAL},
                     lambda: dev.run(dma_prog, [(0, 0x7FFF0001, 0)]))
        expect_errno(ctx, dev, "offset beyond the buffer -> EINVAL", {errno.EINVAL},
                     lambda: dev.run(dma_prog, [(0, bi, size + 4096)]))
        expect_errno(ctx, dev, "span beyond the buffer -> EINVAL", {errno.EINVAL},
                     lambda: dev.run(isa.to_bytes([rhm(rows=2), isa.halt()]), [(0, bi, size - 16)]))
        expect_errno(ctx, dev, "misaligned offset -> EINVAL", {errno.EINVAL},
                     lambda: dev.run(dma_prog, [(0, bi, 8)]))
        expect_errno(ctx, dev, "relocations with n_relocs > 2048 -> EINVAL", {errno.EINVAL},
                     lambda: dev.run_raw(tpu._addr_of(dma_buf), tpu._addr_of(big_rel),
                                         len(dma_prog) // 16, 2049))
        expect_errno(ctx, dev, "flags != 0 -> EINVAL", {errno.EINVAL},
                     lambda: dev.run_raw(tpu._addr_of(ok_buf), 0, 3, 0, 0, 1))
        expect_errno(ctx, dev, "timeout_ms > 60000 -> EINVAL", {errno.EINVAL},
                     lambda: dev.run_raw(tpu._addr_of(ok_buf), 0, 3, 0, 60001))
        expect_errno(ctx, dev, "NULL program pointer -> EFAULT", {errno.EFAULT},
                     lambda: dev.run_raw(0, 0, 3, 0))
        expect_errno(ctx, dev, "bad program pointer -> EFAULT", {errno.EFAULT},
                     lambda: dev.run_raw(0x10, 0, 3, 0))
        expect_errno(ctx, dev, "NULL relocs pointer with n_relocs=1 -> EFAULT", {errno.EFAULT},
                     lambda: dev.run_raw(tpu._addr_of(dma_buf), 0, len(dma_prog) // 16, 1))
        expect_errno(ctx, dev, "bad relocs pointer -> EFAULT", {errno.EFAULT},
                     lambda: dev.run_raw(tpu._addr_of(dma_buf), 0x1000, len(dma_prog) // 16, 1))
        run_tiny_ok(ctx, dev, "device healthy after the rejected programs")
        # unaligned user pointers are legal (the kernel copies bytes): they must simply work
        prog_odd = bytearray(1) + bytearray(ok_prog)
        res = dev.run_raw(tpu._addr_of(prog_odd) + 1, 0, 3, 0)
        ctx.check("unaligned program pointer works", res.ok and res.irq_tag == 0x7E57, str(res))
        rel_odd = bytearray(1) + tpu.Device._pack_relocs([(0, bi, 0)])
        res = dev.run_raw(tpu._addr_of(dma_buf), tpu._addr_of(rel_odd) + 1, len(dma_prog) // 16, 1)
        ctx.check("unaligned relocation pointer works", res.ok, str(res))


def load_bundle_net(ctx, dev, name):
    p = ctx.bundle(name)
    return tpu_mlp.TpuMlp(dev, p) if p else None


def t_bundles(ctx):
    names = [n for n in BUNDLES if ctx.bundle(n)]
    if not names:
        ctx.check("bundles present", False, f"none of {BUNDLES} found under {ctx.bundles}")
        return
    with ctx.open() as dev:
        for name in names:
            with load_bundle_net(ctx, dev, name) as net:
                m = net.model
                res, exact = net.run_sample()
                ctx.check(f"{name}: bit-exact vs golden", exact and res.perf["retired"] == m["program_instrs"],
                          f"{res}" if not exact else f"{res.wall_us:.0f} us kernel, retired={res.perf['retired']}")
                reps = [net.run_sample() for _ in range(3)]
                ctx.check(f"{name}: 3 back-to-back repeats bit-exact", all(e for _, e in reps),
                          "kernel us " + "/".join(f"{r.wall_us:.0f}" for r, _ in reps))
        # digits accuracy on the whole test set (numpy needed for the .npy files)
        d = ctx.bundles / "digits_b32" if ctx.bundles else None
        if np is not None and d is not None and (ctx.bundles / "digits_test_xq.npy").exists() and ctx.bundle("digits_b32"):
            xq, y, gold = (np.load(ctx.bundles / f"digits_test_{n}.npy") for n in ("xq", "y", "golden"))
            n = ctx.digits_limit or len(xq)
            report = json.loads((ctx.bundles / "digits_report.json").read_text())
            with load_bundle_net(ctx, dev, "digits_b32") as net:
                got = net.infer_q_many(xq[:n])
                acc = float((got.argmax(1) == y[:n]).mean())
                gacc = float((gold[:n].argmax(1) == y[:n]).mean())
                ctx.check(f"digits: {n} test samples bit-exact vs golden", bool((got == gold[:n]).all()))
                ctx.check("digits: TPU accuracy == golden INT8 accuracy", acc == gacc,
                          f"TPU {acc:.4f} golden {gacc:.4f} (float {report['float_accuracy']:.4f} on the full set)")
                xf = np.load(ctx.bundles / "digits_test_x.npy")[:n]
                yf = net.infer(xf[:32])
                ctx.check("digits: infer(float) matches dequantized golden",
                          bool(np.allclose(yf, net.dequantize(gold[:32]))))
        else:
            print("[SKIP] digits accuracy: numpy or digits files missing")


def dma_2d_case():
    rows = 12
    inp = bytes(((r * 17 + b * 3) & 255) for r in range(rows) for b in range(16))
    exp = bytearray(b"\xee" * (rows * 16))
    for rep in range(3):
        for row in range(2):
            o = (rep * 5 + row) * 16
            exp[o:o + 16] = inp[o:o + 16]
    prog = [isa.read_host_mem(65530, NOMINAL, 2, 3, 80), isa.sync(isa.U_DMA),
            isa.write_host_mem(65530, NOMINAL + 0x10000000, 2, 3, 80), isa.interrupt_host(0xD0E00002), isa.halt()]
    return isa.to_bytes(prog), inp, bytes(exp)


CFGERR_CASES = (       # name, word, relocated DMA instruction?, expected cfg_err_op
    ("unknown op 0x05", 0x05, False, 0x05),
    ("unknown op 0x7f", 0x7F, False, 0x7F),
    ("RHM zero row_len", isa.read_host_mem(0, NOMINAL, 0), True, 0x01),
    ("RHM reserved bit", isa.read_host_mem(0, NOMINAL, 1) | (1 << 127), True, 0x01),
    ("READ_WEIGHTS zero tiles", isa.read_weights(NOMINAL, 0), True, 0x02),
    ("WHM zero reps", isa.write_host_mem(0, NOMINAL, 1, 0), True, 0x03),
    ("MATMUL B=0", isa.matmul(0, 0, 0), False, 0x10),
    ("MATMUL acc range", isa.matmul(0, 2047, 2), False, 0x10),
    ("ACTIVATE sigmoid (func 3)", isa.activate(0, 0, 1, 0, func=3), False, 0x20),
    ("ACTIVATE rows=0", isa.activate(0, 0, 0, 0), False, 0x20),
    ("SET_CONFIG target 0x4004", isa.set_config(0x4004, 1), False, 0x30),
    ("SYNC reserved bit", isa.sync(isa.U_ALL) | (1 << 12), False, 0x31),
    ("HALT reserved bit", isa.halt() | (1 << 8), False, 0x3F),
)


def t_recovery(ctx):
    with ctx.open() as dev:
        prog, inp, exp = dma_2d_case()
        bi, bo = dev.alloc(len(inp)), dev.alloc(len(exp))
        bi.write(inp)
        bo.fill(0xEE)
        res = dev.run(prog, [(0, bi, 0), (2, bo, 0)])
        ctx.check("dma_2d (3 strided reps in/out)", res.ok and bo.read(len(exp)) == exp and res.irq_tag == 0xD0E00002,
                  str(res))
        good = ctx.bundle("fc1")
        net = tpu_mlp.TpuMlp(dev, good) if good else None
        for name, word, reloc, op in CFGERR_CASES:
            p = isa.to_bytes([word, isa.interrupt_host(0xBAD), isa.halt()])
            before = counters(dev)
            res = dev.run(p, [(0, bi, 0)] if reloc else [])
            ctx.check(f"cfg_err: {name}", res.status == tpu.RUN_CFG_ERR and res.cfg_err_op == op,
                      f"{res}" if res.status != tpu.RUN_CFG_ERR or res.cfg_err_op != op else f"op 0x{res.cfg_err_op:02x}")
            run_tiny_ok(ctx, dev, f"  recovery: tiny job after '{name}'")
            if net is not None:
                ctx.check(f"  recovery: fc1 bit-exact after '{name}'", net.run_sample()[1])
            after = counters(dev)
            ctx.check(f"  failed-job counter advanced for '{name}'", after[1] == before[1] + 1,
                      f"jobs_failed {before[1]} -> {after[1]}")
        if net:
            net.close()
        i = dev.info()
        ctx.check("device not wedged after all error programs", i.state != tpu.STATE_WEDGED, i.state_name)


def t_timeout(ctx):
    with ctx.open() as dev:
        prog = isa.to_bytes([isa.nop()])                           # never HALTs / interrupts
        t = time.perf_counter()
        res = dev.run(prog, [], timeout_ms=100)
        dt = (time.perf_counter() - t) * 1000
        ctx.check("program without HALT -> TIMEOUT", res.status == tpu.RUN_TIMEOUT, str(res))
        if not ctx.mock:
            ctx.check("timeout fires near 100 ms", 90 <= dt <= 400, f"{dt:.0f} ms")
        run_tiny_ok(ctx, dev, "tiny job after the timeout")
        net = load_bundle_net(ctx, dev, "mlp3") or load_bundle_net(ctx, dev, "fc1")
        if net:
            ctx.check("bundle bit-exact after the timeout", net.run_sample()[1])
            net.close()
        ctx.check("device not wedged after timeout", dev.info().state != tpu.STATE_WEDGED)


def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p / 100))]


def t_loop(ctx):
    with ctx.open() as dev:
        net = load_bundle_net(ctx, dev, "fc1")
        if net is None:
            ctx.check("loop: fc1 bundle present", False)
            return
        before = dev.info().jobs_ok
        n = ctx.loops
        kern, host, bad = [], [], 0
        expect = net.bundle.sample_out()
        net.in_buf.write(net.bundle.sample_in())
        t0 = time.perf_counter()
        for _ in range(n):
            t = time.perf_counter()
            res = dev.run(net.bundle.program, net._relocs, 0)
            host.append((time.perf_counter() - t) * 1e6)
            kern.append(res.wall_us)
            if not res.ok or net.out_buf.read() != expect:
                bad += 1
        total = time.perf_counter() - t0
        ctx.check(f"{n} back-to-back fc1 runs all bit-exact", bad == 0, f"{bad} bad")
        ctx.check("job counter advanced by exactly n", dev.info().jobs_ok == before + n,
                  f"{dev.info().jobs_ok} vs {before + n}")
        print(f"[INFO] fc1 x{n}: {n / total:.0f} jobs/s; kernel wall us p50/p90/p99/max = "
              f"{pct(kern, 50):.0f}/{pct(kern, 90):.0f}/{pct(kern, 99):.0f}/{max(kern):.0f}; host ioctl us p50/p99 = "
              f"{pct(host, 50):.0f}/{pct(host, 99):.0f}", flush=True)
        net.close()


def _worker(args):
    path, bundle, iters, mock = args
    with tpu.Device(path) as dev, tpu_mlp.TpuMlp(dev, bundle) as net:
        bad = 0
        for _ in range(iters):
            _, exact = net.run_sample()
            bad += not exact
        return bad


def t_concurrent(ctx):
    if ctx.mock:
        print("[SKIP] concurrent: needs the real driver (mock state is per process)")
        return
    b = ctx.bundle("mlp3")
    if b is None:
        ctx.check("concurrent: mlp3 bundle present", False)
        return
    iters = 40
    with ctx.open() as dev:
        before = dev.info().jobs_ok
        mp = multiprocessing.get_context("fork")
        with mp.Pool(ctx.procs) as pool:
            bad = pool.map(_worker, [(ctx.path, b, iters, False)] * ctx.procs)
        ctx.check(f"{ctx.procs} processes x {iters} mlp3 jobs concurrently, all bit-exact", sum(bad) == 0, f"bad={bad}")
        ctx.check("job counter == processes x iterations", dev.info().jobs_ok == before + ctx.procs * iters,
                  f"{dev.info().jobs_ok - before} vs {ctx.procs * iters}")


def t_signals(ctx):
    got = {"n": 0}

    def handler(signum, frame):
        got["n"] += 1

    sigs = (signal.SIGALRM, signal.SIGUSR1, signal.SIGINT)
    old = {s: signal.signal(s, handler) for s in sigs}
    stop = threading.Event()
    main_ident = threading.main_thread().ident

    def sender():
        k = 0
        while not stop.is_set():
            signal.pthread_kill(main_ident, (signal.SIGUSR1, signal.SIGINT)[k & 1])
            k += 1
            time.sleep(0.0002)

    try:
        with ctx.open() as dev:
            net = load_bundle_net(ctx, dev, "mlp_wide")
            if net is None:
                ctx.check("signals: mlp_wide bundle present", False)
                return
            net.in_buf.write(net.bundle.sample_in())
            n = 20 if ctx.mock else 300
            t = threading.Thread(target=sender, daemon=True)
            signal.setitimer(signal.ITIMER_REAL, 0.00031, 0.00031)
            t.start()
            ok = intr = other = 0
            expect = net.bundle.sample_out()
            for _ in range(n):
                try:
                    net.out_buf.fill(0xEE)
                    res = dev.run(net.bundle.program, net._relocs, 0)
                    if res.ok and net.out_buf.read() == expect:
                        ok += 1
                    else:
                        other += 1
                except OSError as e:
                    if e.errno == errno.EINTR:
                        intr += 1
                    else:
                        other += 1
            signal.setitimer(signal.ITIMER_REAL, 0, 0)
            stop.set()
            t.join()
            ctx.check(f"mlp_wide x{n} under signal storm: only OK or EINTR", other == 0 and ok + intr == n,
                      f"ok={ok} eintr={intr} other={other} signals handled={got['n']}")
            _, exact = net.run_sample()
            ctx.check("bit-exact job after the signal storm", exact)
            ctx.check("device not wedged after the signal storm", dev.info().state != tpu.STATE_WEDGED)
            net.close()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0, 0)
        stop.set()
        for s, h in old.items():
            signal.signal(s, h)


TESTS = dict(basic=t_basic, negatives=t_negatives, bundles=t_bundles, recovery=t_recovery, timeout=t_timeout,
             loop=t_loop, concurrent=t_concurrent, signals=t_signals)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="/dev/tpu0")
    ap.add_argument("--bundles", default=None, help="directory containing the bundle directories")
    ap.add_argument("--only", default=",".join(ALL_TESTS))
    ap.add_argument("--loops", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--digits-limit", type=int, default=None, help="limit the digits test samples")
    ap.add_argument("--mock", action="store_true", help="run against the host mock (golden simulator)")
    a = ap.parse_args(argv)
    if a.bundles is None:
        for cand in (Path.cwd() / "bundles", Path.home() / "bundles", HERE.parents[2] / "build" / "bundles"):
            if cand.exists():
                a.bundles = str(cand)
                break
    ctx = Ctx(a)
    if not a.mock and not os.path.exists(a.device):
        print(f"{a.device} not found: load the driver first (software/driver/tpu-load.sh, "
              f"docs/driver_user_guide.md)", file=sys.stderr)
        return 2
    names = [n for n in a.only.split(",") if n]
    unknown = [n for n in names if n not in TESTS]
    if unknown:
        print(f"unknown test(s): {unknown}; choose from {ALL_TESTS}", file=sys.stderr)
        return 2
    t0 = time.time()
    for n in names:
        print(f"--- {n} ---", flush=True)
        try:
            TESTS[n](ctx)
        except Exception as e:                                  # a crashed test is a failed test
            import traceback
            traceback.print_exc()
            ctx.check(f"{n}: raised {type(e).__name__}", False, str(e))
    print(f"{ctx.checks} checks, {ctx.fails} failed, {time.time() - t0:.1f} s")
    print("TPU_DRIVER_TEST_PASS" if ctx.fails == 0 else f"TPU_DRIVER_TEST_FAIL ({ctx.fails})")
    return 0 if ctx.fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
