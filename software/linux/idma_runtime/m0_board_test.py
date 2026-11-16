#!/usr/bin/env python3
"""TPU-KV260 M0 board gate (runs on the KV260 as root).

Checks, in order, and stops at the first hard failure:
  1. fpga0 is operating and the PL window answers (L5: never touch an unloaded PL).
  2. VERSION @0x0, SCRATCH @0x4 read/write (catches HPM0 lane/stride errors, L1),
     BUILD_ID @0x8 equals the value stamped at build time (catches stale bitstreams, L7).
  3. PULP iDMA over S_AXI_HP0 into DDR (u-dma-buf, CACHED mapping + explicit
     sync_for_device / sync_for_cpu; an O_SYNC mapping is Device memory on arm64 and any
     unaligned memcpy into it raises SIGBUS — found on the board in M0):
     1D aligned, 1D misaligned 4097 B (crosses 4 KiB), 2D strided, each twice (L14),
     zero-length (issues an ID, retires without traffic), and a 4 MiB copy for bandwidth.

All register accesses are single aligned 32-bit loads/stores through a memoryview: reading
next_id has a side effect (launch), so it must never be read byte-wise or speculatively.
"""
import argparse, mmap, os, sys, time

UIO        = "/dev/uio4"            # base-DT generic-uio "fabric": 0xA000_0000, 64 KiB, SPI 121
UIO_NAME   = "fabric"
UDMABUF    = "/dev/udmabuf0"
REGS       = 0x0000                 # axi_lite_regs
IDMA       = 0x1000                 # idma_reg64_2d cfg
R_VERSION, R_SCRATCH, R_BUILD_ID, R_DONE_MIRROR, R_STATUS, R_CTRL = 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14
VERSION = 0x00000110   # v0.1.1: STATUS[4:2] iDMA error flags, CTRL[0] clear
D_CONF, D_STATUS0, D_NEXT_ID, D_DONE_ID = 0x000, 0x004, 0x044, 0x084
D_DST, D_SRC, D_LEN = 0x0D0, 0x0D8, 0x0E0
D_DSTRIDE, D_SSTRIDE, D_REPS = 0x100, 0x108, 0x110
CONF_2D = 0x400

fails = 0
def ok(msg):  print(f"[PASS] {msg}")
def bad(msg):
    global fails; fails += 1; print(f"[FAIL] {msg}")

def sysfs(path):
    with open(path) as f: return f.read().strip()

class Win:
    def __init__(self, mm): self.w = memoryview(mm).cast("I")
    def rd(self, off): return self.w[off >> 2]
    def wr(self, off, v): self.w[off >> 2] = v & 0xFFFFFFFF
    def wr64(self, off, v): self.wr(off, v); self.wr(off + 4, v >> 32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-id", required=True)
    a = ap.parse_args()
    want_id = int(a.build_id, 0)

    # -- 1. PL loaded + correct UIO node ------------------------------------------------------
    state = sysfs("/sys/class/fpga_manager/fpga0/state")
    if state != "operating": print(f"fpga0 state={state}; refusing to touch the PL"); return 2
    uio = os.path.basename(UIO)
    if sysfs(f"/sys/class/uio/{uio}/name") != UIO_NAME or \
       int(sysfs(f"/sys/class/uio/{uio}/maps/map0/addr"), 16) != 0xA0000000:
        print(f"{UIO} is not the 0xA000_0000 fabric node"); return 2
    fd = os.open(UIO, os.O_RDWR | os.O_SYNC)
    mm = mmap.mmap(fd, 0x10000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=0)
    w = Win(mm)

    # -- 2. identity registers ----------------------------------------------------------------
    ver = w.rd(R_VERSION)
    (ok if ver == VERSION else bad)(f"VERSION = 0x{ver:08x} (expect 0x{VERSION:08x})")
    lane_ok = True
    for pat in (0xA5A5A5A5, 0x5A5A5A5A, 0x12345678, 0xFFFFFFFF, 0x00000000):
        w.wr(R_SCRATCH, pat)
        got = w.rd(R_SCRATCH)
        if got != pat: lane_ok = False; bad(f"SCRATCH wrote 0x{pat:08x} read 0x{got:08x}")
    if lane_ok: ok("SCRATCH @0x4 read/write (no lane/stride error, L1)")
    bid = w.rd(R_BUILD_ID)
    if bid != want_id:
        bad(f"BUILD_ID 0x{bid:08x} != expected 0x{want_id:08x} (stale bitstream? L7)")
        return 1
    ok(f"BUILD_ID = 0x{bid:08x} matches the build")
    if fails: return 1
    w.wr(R_CTRL, 1); w.wr(R_CTRL, 0)                     # clear sticky iDMA error flags
    st = w.rd(R_STATUS)
    (ok if (st >> 2) & 7 == 0 else bad)(f"iDMA error flags clear after CTRL[0] (STATUS=0x{st:x})")

    # -- 3. iDMA over HP0 ---------------------------------------------------------------------
    phys = int(sysfs("/sys/class/u-dma-buf/udmabuf0/phys_addr"), 16)
    size = int(sysfs("/sys/class/u-dma-buf/udmabuf0/size"))
    bfd = os.open(UDMABUF, os.O_RDWR)                    # cached; HP0 is non-coherent -> sync
    buf = mmap.mmap(bfd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    print(f"u-dma-buf phys=0x{phys:x} size={size >> 20} MiB "
          f"dma_coherent={sysfs('/sys/class/u-dma-buf/udmabuf0/dma_coherent')}")

    def sync(for_device, off=0, length=size):
        base = "/sys/class/u-dma-buf/udmabuf0/"
        for k, v in (("sync_offset", off), ("sync_size", length), ("sync_direction", 0)):
            with open(base + k, "w") as f: f.write(str(v))
        with open(base + ("sync_for_device" if for_device else "sync_for_cpu"), "w") as f:
            f.write("1")

    def copy(name, so, do, length, reps=1, ss=0, ds=0, seed=0, check=True):
        rows = []
        for r in range(reps):
            data = bytes(((seed + 7 * r + 13 * i + (i >> 8)) & 0xFF) for i in range(length))
            buf[so + r * ss: so + r * ss + length] = data
            buf[do + r * ds - 1: do + r * ds + length + 1] = b"\xa5" + b"\xee" * length + b"\x5a"
            rows.append(data)
        span_lo = max(0, min(so, do - 1))
        span_hi = max(so + (reps - 1) * ss, do + (reps - 1) * ds) + length + 1
        sync(True, span_lo & ~0xFFF, ((span_hi + 0xFFF) & ~0xFFF) - (span_lo & ~0xFFF))
        w.wr64(IDMA + D_SRC, phys + so)
        w.wr64(IDMA + D_DST, phys + do)
        w.wr64(IDMA + D_LEN, length)
        w.wr64(IDMA + D_SSTRIDE, ss)
        w.wr64(IDMA + D_DSTRIDE, ds)
        w.wr64(IDMA + D_REPS, reps)
        w.wr(IDMA + D_CONF, CONF_2D if reps > 1 else 0)
        t0 = time.perf_counter()
        tid = w.rd(IDMA + D_NEXT_ID)                      # launch: read exactly once
        if tid == 0: bad(f"{name}: next_id=0 (rejected)"); return None
        deadline = t0 + 2.0
        while ((w.rd(IDMA + D_DONE_ID) - tid) & 0xFFFFFFFF) >= 0x80000000:
            if time.perf_counter() > deadline:
                bad(f"{name}: timeout id={tid} done={w.rd(IDMA + D_DONE_ID)} "
                    f"status0=0x{w.rd(IDMA + D_STATUS0):x} hi_err={w.rd(R_STATUS) >> 1 & 1}")
                return None
        dt = time.perf_counter() - t0
        sync(False, span_lo & ~0xFFF, ((span_hi + 0xFFF) & ~0xFFF) - (span_lo & ~0xFFF))
        if check:
            for r, data in enumerate(rows):
                got = bytes(buf[do + r * ds: do + r * ds + length])
                if got != data:
                    nbad = sum(1 for x, y in zip(got, data) if x != y)
                    bad(f"{name}: row {r}: {nbad}/{length} bytes wrong"); return None
                if buf[do + r * ds - 1] != 0xA5 or buf[do + r * ds + length] != 0x5A:
                    bad(f"{name}: row {r}: guard byte overwritten"); return None
            ok(f"{name}: id={tid} {length * reps} B in {dt * 1e3:.2f} ms")
        return dt

    for run in range(2):                                      # L14: back-to-back
        copy(f"1D aligned 256 B (run {run})",           0x000000, 0x100000, 256, seed=0x11 + run)
        copy(f"1D misaligned 4097 B (run {run})",       0x001003, 0x200011, 4097, seed=0x22 + run)
        copy(f"2D 8x100 B, strides 256/128 (run {run})", 0x300005, 0x400000, 100, 8, 256, 128,
             seed=0x33 + run)

    # zero-length: v0.7.4 issues an ID and retires it without bus traffic (sim-verified)
    w.wr64(IDMA + D_LEN, 0); w.wr(IDMA + D_CONF, 0)
    zid = w.rd(IDMA + D_NEXT_ID)
    t0 = time.perf_counter()
    while ((w.rd(IDMA + D_DONE_ID) - zid) & 0xFFFFFFFF) >= 0x80000000 and time.perf_counter() - t0 < 1:
        pass
    done = w.rd(IDMA + D_DONE_ID)
    (ok if zid != 0 and ((done - zid) & 0xFFFFFFFF) < 0x80000000 else bad)(
        f"zero-length: id={zid} retired (done_id={done})")
    st = w.rd(R_STATUS)
    (ok if (st >> 2) & 7 == 0b100 else bad)(
        f"zero-length sets only the midend error flag (STATUS=0x{st:x}, expect [4:2]=100)")
    w.wr(R_CTRL, 1); w.wr(R_CTRL, 0)

    # bandwidth: 4 MiB DDR -> DDR (read + write both over HP0); no byte check (Python is slow)
    n = 4 << 20
    dt = copy("4 MiB DDR->DDR", 0x000000, 0x800000, n, seed=0, check=False) if size >= 2 * n + 0x10 else None
    if dt:
        ok(f"4 MiB copy: {dt * 1e3:.2f} ms -> {n / dt / 1e6:.0f} MB/s (read+write each)")
        sample = [0, 1, n // 2 + 3, n - 1]
        if any(buf[0x800000 + i] != buf[i] for i in sample): bad("4 MiB copy: sample mismatch")

    st = w.rd(R_STATUS)
    (ok if (st >> 1) & 0xF == 0 else bad)(
        f"STATUS=0x{st:x}: no HP0 upper-address error, no AXI read/write error, no midend error")
    print("M0_BOARD_GATE_PASS" if fails == 0 else f"M0_BOARD_GATE_FAIL ({fails})")
    return 0 if fails == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
