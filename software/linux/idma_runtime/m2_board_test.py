#!/usr/bin/env python3
"""TPU-KV260 M2 board test (runs on the KV260 as root): data fabric + on-chip memories.

Register window 0xA000_0000 (+64 KiB) via /dev/uio4; memory windows (UB 0xA010_0000, ACC
0xA020_0000) via /dev/mem for CPU debug reads (aligned 32-bit loads only; push windows are never
touched by the CPU — 32-bit stores are partial beats and are rejected by design).
Data moves only through iDMA (DDR <-> on-chip), with the u-dma-buf cached + synced (L8a).
"""
import argparse, mmap, os, random, sys, time

VERSION = 0x00000200
UIO, UIO_NAME = "/dev/uio4", "fabric"
R_VERSION, R_SCRATCH, R_BUILD_ID, R_DONE, R_STATUS, R_CTRL, R_LEVEL = 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18
IDMA = 0x1000
D_CONF, D_NEXT_ID, D_DONE_ID, D_DST, D_SRC, D_LEN = 0x000, 0x044, 0x084, 0x0D0, 0x0D8, 0x0E0
D_DSTRIDE, D_SSTRIDE, D_REPS = 0x100, 0x108, 0x110
UB_BASE, UB_SIZE = 0xA010_0000, 1 << 20
ACC_BASE, ACC_SIZE = 0xA020_0000, 128 << 10
WFIFO_BASE, IFIFO_BASE = 0xA030_0000, 0xA040_0000
C_ERRCLR, C_UBOWN, C_ACCOWN, C_WFPOP, C_IFPOP, C_WFFLUSH, C_IFFLUSH = (1 << i for i in range(7))
ST_NAMES = ["busy", "hp0_hi", "dma_rd_err", "dma_wr_err", "dma_mid_err",
            "ub_own_err", "acc_own_err", "wfifo_err", "ififo_err"]

fails = 0
def ok(m): print(f"[PASS] {m}")
def bad(m):
    global fails; fails += 1; print(f"[FAIL] {m}")
def sysfs(p):
    with open(p) as f: return f.read().strip()

class Win:
    def __init__(self, mm): self.w = memoryview(mm).cast("I")
    def rd(self, o): return self.w[o >> 2]
    def wr(self, o, v): self.w[o >> 2] = v & 0xFFFFFFFF
    def wr64(self, o, v): self.wr(o, v); self.wr(o + 4, v >> 32)

def status_str(st):
    return ",".join(n for i, n in enumerate(ST_NAMES) if (st >> i) & 1) or "clean"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--build-id", required=True)
    a = ap.parse_args()
    rnd = random.Random(1)

    if sysfs("/sys/class/fpga_manager/fpga0/state") != "operating": print("PL not loaded"); return 2
    if sysfs("/sys/class/uio/uio4/name") != UIO_NAME: print("uio4 is not the fabric node"); return 2
    fd = os.open(UIO, os.O_RDWR | os.O_SYNC)
    w = Win(mmap.mmap(fd, 0x10000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE))
    mfd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    ubw = Win(mmap.mmap(mfd, UB_SIZE, mmap.MAP_SHARED, mmap.PROT_READ, offset=UB_BASE))
    accw = Win(mmap.mmap(mfd, ACC_SIZE, mmap.MAP_SHARED, mmap.PROT_READ, offset=ACC_BASE))

    # ---- identity ---------------------------------------------------------------------------
    ver = w.rd(R_VERSION)
    (ok if ver == VERSION else bad)(f"VERSION 0x{ver:08x} (expect 0x{VERSION:08x})")
    w.wr(R_SCRATCH, 0x5A5AA5A5)
    (ok if w.rd(R_SCRATCH) == 0x5A5AA5A5 else bad)("SCRATCH read/write")
    bid = w.rd(R_BUILD_ID)
    if bid != int(a.build_id, 0): bad(f"BUILD_ID 0x{bid:08x} stale (L7)"); return 1
    ok(f"BUILD_ID 0x{bid:08x}")
    w.wr(R_CTRL, C_ERRCLR | C_WFFLUSH | C_IFFLUSH); w.wr(R_CTRL, 0)
    st, lv = w.rd(R_STATUS), w.rd(R_LEVEL)
    (ok if st == 0 and lv == 0 else bad)(f"clean start: STATUS={status_str(st)} levels=0x{lv:08x}")

    # ---- DMA plumbing -----------------------------------------------------------------------
    phys = int(sysfs("/sys/class/u-dma-buf/udmabuf0/phys_addr"), 16)
    size = int(sysfs("/sys/class/u-dma-buf/udmabuf0/size"))
    bfd = os.open("/dev/udmabuf0", os.O_RDWR)
    buf = mmap.mmap(bfd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    def sync(dev):
        b = "/sys/class/u-dma-buf/udmabuf0/"
        for k, v in (("sync_offset", 0), ("sync_size", size), ("sync_direction", 0)):
            with open(b + k, "w") as f: f.write(str(v))
        with open(b + ("sync_for_device" if dev else "sync_for_cpu"), "w") as f: f.write("1")

    def dma(src, dst, length, reps=1, ss=0, ds=0, timeout=2.0):
        w.wr64(IDMA + D_SRC, src); w.wr64(IDMA + D_DST, dst); w.wr64(IDMA + D_LEN, length)
        w.wr64(IDMA + D_SSTRIDE, ss); w.wr64(IDMA + D_DSTRIDE, ds); w.wr64(IDMA + D_REPS, reps)
        w.wr(IDMA + D_CONF, 0x400 if reps > 1 else 0)
        t0 = time.perf_counter()
        tid = w.rd(IDMA + D_NEXT_ID)
        while ((w.rd(IDMA + D_DONE_ID) - tid) & 0xFFFFFFFF) >= 0x80000000:
            if time.perf_counter() - t0 > timeout:
                raise RuntimeError(f"DMA id {tid} timeout, STATUS={status_str(w.rd(R_STATUS))}")
        return time.perf_counter() - t0

    def rand_bytes(n): return rnd.randbytes(n)
    D = lambda off: phys + off                                # DDR address inside u-dma-buf

    # ---- UB: DDR -> UB -> DDR ----------------------------------------------------------------
    for run in range(2):
        for t in range(6):
            n = rnd.randint(1, 8000)
            uoff = rnd.randrange(0, UB_SIZE - n)
            so, do = 0x10000 + rnd.randrange(16), 0x200000 + rnd.randrange(16)
            data = rand_bytes(n)
            buf[so:so + n] = data; buf[do:do + n] = b"\xee" * n
            sync(True)
            dma(D(so), UB_BASE + uoff, n); dma(UB_BASE + uoff, D(do), n)
            sync(False)
            if bytes(buf[do:do + n]) != data: bad(f"UB run{run} #{t}: {n} B @UB+0x{uoff:x} mismatch")
        ok(f"UB run{run}: 6 random DDR->UB->DDR round trips (misaligned, up to 8000 B)")
    # 2D through the UB
    rows, rl, ss, us = 8, 100, 512, 128
    src = bytearray(rows * ss)
    for r in range(rows): src[r * ss: r * ss + rl] = rand_bytes(rl)
    buf[0x10000:0x10000 + len(src)] = src; buf[0x300000:0x300000 + rows * ss] = b"\xee" * (rows * ss)
    sync(True)
    dma(D(0x10000), UB_BASE + 0x40000, rl, rows, ss, us)
    dma(UB_BASE + 0x40000, D(0x300000), rl, rows, us, ss)
    sync(False)
    good = all(buf[0x300000 + r * ss: 0x300000 + r * ss + rl] == src[r * ss: r * ss + rl] for r in range(rows))
    (ok if good else bad)("UB 2D strided round trip (8 x 100 B)")

    # ---- CPU debug view of the UB (HPM0 -> fabric, narrow 32-bit reads) ----------------------
    blk = rand_bytes(4096)
    buf[0x20000:0x20000 + 4096] = blk; sync(True)
    dma(D(0x20000), UB_BASE + 0x8_0000, 4096)
    words = [ubw.rd(0x8_0000 + 4 * i) for i in range(1024)]
    got = b"".join(x.to_bytes(4, "little") for x in words)
    (ok if got == blk else bad)("CPU debug read of UB (1024 x 32-bit via HPM0) matches iDMA-written data")

    # ---- full 1 MiB UB and full 128 KiB ACC, with bandwidth ---------------------------------
    big = rand_bytes(UB_SIZE)
    buf[0x400000:0x400000 + UB_SIZE] = big; buf[0x600000:0x600000 + UB_SIZE] = b"\x00" * UB_SIZE
    sync(True)
    t_in = dma(D(0x400000), UB_BASE, UB_SIZE); t_out = dma(UB_BASE, D(0x600000), UB_SIZE)
    sync(False)
    (ok if bytes(buf[0x600000:0x600000 + UB_SIZE]) == big else bad)(
        f"full UB 1 MiB: DDR->UB {UB_SIZE / t_in / 1e6:.0f} MB/s, UB->DDR {UB_SIZE / t_out / 1e6:.0f} MB/s")
    acc = rand_bytes(ACC_SIZE)
    buf[0x800000:0x800000 + ACC_SIZE] = acc; buf[0x900000:0x900000 + ACC_SIZE] = b"\x00" * ACC_SIZE
    sync(True)
    dma(D(0x800000), ACC_BASE, ACC_SIZE); dma(ACC_BASE, D(0x900000), ACC_SIZE)
    sync(False)
    (ok if bytes(buf[0x900000:0x900000 + ACC_SIZE]) == acc else bad)("full ACC 128 KiB DDR->ACC->DDR")
    words = [accw.rd(0x40 * 100 + 4 * i) for i in range(16)]           # row 100 = 4 banks
    got = b"".join(x.to_bytes(4, "little") for x in words)
    (ok if got == acc[0x40 * 100: 0x40 * 101] else bad)("CPU debug read of ACC row 100 (bank layout)")

    # ---- Weight FIFO / Instruction FIFO: push, level, debug-pop loopback ----------------------
    for name, base, nwords, popbit, shift in (("WFIFO", WFIFO_BASE, 256, C_WFPOP, 0),
                                              ("IFIFO", IFIFO_BASE, 128, C_IFPOP, 16)):
        data = rand_bytes(nwords * 16)
        buf[0x30000:0x30000 + len(data)] = data; buf[0x340000:0x340000 + len(data)] = b"\xee" * len(data)
        sync(True)
        dma(D(0x30000), base, len(data))
        lvl = (w.rd(R_LEVEL) >> shift) & 0xFFFF
        (ok if lvl == nwords else bad)(f"{name}: {nwords} words pushed by iDMA, FIFO_LEVEL={lvl}")
        w.wr(R_CTRL, popbit)
        dma(base, D(0x340000), len(data))
        w.wr(R_CTRL, 0)
        sync(False)
        lvl = (w.rd(R_LEVEL) >> shift) & 0xFFFF
        (ok if bytes(buf[0x340000:0x340000 + len(data)]) == data and lvl == 0 else bad)(
            f"{name}: debug-pop loopback {name}->DDR in order, level back to {lvl}")
    # flush
    buf[0x30000:0x30000 + 256] = rand_bytes(256); sync(True)
    dma(D(0x30000), WFIFO_BASE, 256)
    w.wr(R_CTRL, C_WFFLUSH); w.wr(R_CTRL, 0)
    (ok if w.rd(R_LEVEL) & 0xFFFF == 0 else bad)("WFIFO flush empties the FIFO")

    # ---- ownership: TPU-owned UB rejects iDMA writes -----------------------------------------
    before = [ubw.rd(0x8_0000 + 4 * i) for i in range(64)]
    buf[0x50000:0x50000 + 256] = b"\x77" * 256; sync(True)
    w.wr(R_CTRL, C_UBOWN)
    dma(D(0x50000), UB_BASE + 0x8_0000, 256)
    st = w.rd(R_STATUS)
    w.wr(R_CTRL, 0)
    after = [ubw.rd(0x8_0000 + 4 * i) for i in range(64)]
    want = (1 << 3) | (1 << 5)                                     # dma_wr_err + ub_own_err
    (ok if (st & want) == want and before == after else bad)(
        f"TPU-owned UB: iDMA write blocked, STATUS={status_str(st)}, UB unchanged={before == after}")
    w.wr(R_CTRL, C_ERRCLR); w.wr(R_CTRL, 0)

    st = w.rd(R_STATUS)
    (ok if st == 0 else bad)(f"final STATUS={status_str(st)}")
    print("M2_BOARD_PASS" if fails == 0 else f"M2_BOARD_FAIL ({fails})")
    return 0 if fails == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
