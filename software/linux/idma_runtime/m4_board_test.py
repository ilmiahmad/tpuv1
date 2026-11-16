#!/usr/bin/env python3
"""TPU-KV260 M4 board test (runs on the KV260 as root): first MLPs on the real core.

Register window 0xA000_0000 via /dev/uio4 (tpu_regs v0.3). Program, weights and inputs move
DDR -> on-chip through iDMA; the program/weight streams are interleaved in chunks no larger
than each FIFO's free space (finding: a single sequential push deadlocks
once the program exceeds the 512-entry IFIFO).

Usage: m4_board_test.py --build-id 0x... [--cases fc1,mlp3,mlp_b1,mlp_wide] [--cases-dir DIR]
"""
import argparse, glob, mmap, os, sys, time

VERSION = 0x00000300
UIO_NAME = "fabric"
R_VERSION, R_SCRATCH, R_BUILD_ID, R_DONE, R_STATUS, R_CTRL, R_LEVEL = 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18
R_CSTATUS, R_IRQ_TAG, R_PERF0, R_DBG = 0x1C, 0x20, 0x24, 0x44
IDMA = 0x1000
D_CONF, D_NEXT_ID, D_DONE_ID, D_DST, D_SRC, D_LEN = 0x000, 0x044, 0x084, 0x0D0, 0x0D8, 0x0E0
D_DSTRIDE, D_SSTRIDE, D_REPS = 0x100, 0x108, 0x110
UB_BASE, ACC_BASE = 0xA010_0000, 0xA020_0000
WFIFO_BASE, IFIFO_BASE = 0xA030_0000, 0xA040_0000
WFIFO_WORDS, IFIFO_WORDS = 1024, 512
C_ERRCLR, C_UBOWN, C_ACCOWN, C_WFPOP, C_IFPOP, C_WFFLUSH, C_IFFLUSH = (1 << i for i in range(7))
C_RUN, C_IRQCLR, C_PERFCLR = 1 << 7, 1 << 8, 1 << 9
PERF_NAMES = ["total", "array_active", "weight_stall", "weight_shift",
              "non_matrix", "raw_stall", "act_active", "instr_count"]
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

def load_hex(path):
    out = bytearray()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out += int(line, 16).to_bytes(16, "little")
    return bytes(out)

def load_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            k, _, v = line.partition(" ")
            meta[k.strip()] = v.strip()
    return meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-id", required=True)
    ap.add_argument("--cases", default="fc1,mlp3,mlp_b1,mlp_wide")
    ap.add_argument("--cases-dir", default="cases")
    ap.add_argument("--trace", action="store_true", help="log each DMA before issuing it")
    a = ap.parse_args()

    # Check host prerequisites before touching PL registers (L5, L8).
    if not os.path.exists("/sys/class/u-dma-buf/udmabuf0/phys_addr"):
        print(f"Missing udmabuf0 for kernel {os.uname().release}; load its matching u-dma-buf module")
        return 2
    if sysfs("/sys/class/fpga_manager/fpga0/state") != "operating": print("PL not loaded"); return 2
    nodes = [p for p in glob.glob("/sys/class/uio/uio*")
             if sysfs(p + "/name") == UIO_NAME
             and int(sysfs(p + "/maps/map0/addr"), 0) == 0xA0000000]
    if len(nodes) != 1: print("Expected one fabric UIO at 0xA0000000"); return 2
    uio = "/dev/" + os.path.basename(nodes[0])
    print(f"[INFO] opening {uio}", flush=True)
    fd = os.open(uio, os.O_RDWR | os.O_SYNC)
    w = Win(mmap.mmap(fd, 0x10000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE))

    # ---- identity ---------------------------------------------------------------------------
    ver = w.rd(R_VERSION)
    (ok if ver == VERSION else bad)(f"VERSION 0x{ver:08x} (expect 0x{VERSION:08x})")
    if ver != VERSION: return 1
    bid = w.rd(R_BUILD_ID)
    if bid != int(a.build_id, 0): bad(f"BUILD_ID 0x{bid:08x} stale (L7)"); return 1
    ok(f"BUILD_ID 0x{bid:08x}")
    print("[INFO] checking SCRATCH write/read", flush=True)
    w.wr(R_SCRATCH, 0x5A5AA5A5)
    if w.rd(R_SCRATCH) != 0x5A5AA5A5:
        bad("SCRATCH write/read"); return 1
    ok("SCRATCH write/read")
    print("[INFO] clearing CTRL, FIFOs, IRQ and counters", flush=True)
    clear = C_ERRCLR | C_WFFLUSH | C_IFFLUSH | C_IRQCLR | C_PERFCLR
    w.wr(R_CTRL, clear)
    if w.rd(R_CTRL) != clear:
        bad("CTRL write/read"); return 1
    w.wr(R_CTRL, 0)
    st, lv, cs = w.rd(R_STATUS), w.rd(R_LEVEL), w.rd(R_CSTATUS)
    # HALT persists until the next RUN edge; it is normal on a repeated invocation.
    (ok if st == 0 and lv == 0 and (cs & ~2) == 0 else bad)(
        f"clean start: STATUS={status_str(st)} levels=0x{lv:08x} CSTATUS=0x{cs:08x}")

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

    def dma(src, dst, length, timeout=5.0):
        if length == 0: return 0.0
        if a.trace: print(f"[DMA] 0x{src:x} -> 0x{dst:x}, {length} bytes", flush=True)
        w.wr64(IDMA + D_SRC, src); w.wr64(IDMA + D_DST, dst); w.wr64(IDMA + D_LEN, length)
        w.wr64(IDMA + D_SSTRIDE, 0); w.wr64(IDMA + D_DSTRIDE, 0); w.wr64(IDMA + D_REPS, 1)
        w.wr(IDMA + D_CONF, 0)
        t0 = time.perf_counter()
        tid = w.rd(IDMA + D_NEXT_ID)
        while ((w.rd(IDMA + D_DONE_ID) - tid) & 0xFFFFFFFF) >= 0x80000000:
            if time.perf_counter() - t0 > timeout:
                raise RuntimeError(f"DMA id {tid} timeout, STATUS={status_str(w.rd(R_STATUS))}")
        st = w.rd(R_STATUS)
        if st & ~1: raise RuntimeError(f"DMA id {tid}: STATUS={status_str(st)}")
        return time.perf_counter() - t0

    D = lambda off: phys + off
    OFF_UB, OFF_PROG, OFF_WMEM, OFF_OUT = 0x100000, 0x400000, 0x600000, 0x800000

    def levels():
        lv = w.rd(R_LEVEL)
        return lv >> 16, lv & 0xFFFF          # ififo, wfifo

    # ---- per-case: load, run, compare ---------------------------------------------------------
    for case in a.cases.split(","):
        print(f"[INFO] starting {case}", flush=True)
        cdir = os.path.join(a.cases_dir, case)
        meta = load_meta(os.path.join(cdir, "meta.txt"))
        prog  = load_hex(os.path.join(cdir, "prog.hex"))
        wmem  = load_hex(os.path.join(cdir, "wmem.hex"))
        ub_init = load_hex(os.path.join(cdir, "ub_init.hex"))
        ub_exp  = load_hex(os.path.join(cdir, "ub_expect.hex"))
        out_off = int(meta["out_first_row"]) * 16
        acc_exp = None
        ap_ = os.path.join(cdir, "acc_expect.hex")
        if os.path.exists(ap_):
            acc_exp = bytearray()
            with open(ap_) as f:
                for line in f:
                    line = line.strip()
                    if line: acc_exp += int(line, 16).to_bytes(64, "little")

        # Clear events and flush FIFOs between completed cases.
        w.wr(R_CTRL, C_ERRCLR | C_WFFLUSH | C_IFFLUSH | C_IRQCLR | C_PERFCLR); w.wr(R_CTRL, 0)

        # UB init image
        buf[OFF_UB:OFF_UB + len(ub_init)] = ub_init; sync(True)
        dma(D(OFF_UB), UB_BASE, len(ub_init))
        # Match the simulation fixture, including unused ACC rows compared below.
        acc_bytes = int(meta["acc_rows"]) * 64
        if acc_bytes:
            buf[OFF_OUT:OFF_OUT + acc_bytes] = bytes(acc_bytes); sync(True)
            dma(D(OFF_OUT), ACC_BASE, acc_bytes)

        # Start the core, then interleave FIFO pushes while it drains them.
        buf[OFF_PROG:OFF_PROG + len(prog)] = prog
        buf[OFF_WMEM:OFF_WMEM + len(wmem)] = wmem; sync(True)
        w.wr(R_CTRL, C_UBOWN | C_ACCOWN)
        pi = wi = 0
        w.wr(R_CTRL, C_UBOWN | C_ACCOWN | C_RUN)
        print(f"[INFO] {case}: RUN, feeding program and weights", flush=True)
        t0 = time.perf_counter()
        done = False
        while not done:
            il, wl = levels()
            if pi < len(prog) and il < IFIFO_WORDS:
                n = min(len(prog) - pi, (IFIFO_WORDS - il) * 16)
                dma(D(OFF_PROG + pi), IFIFO_BASE, n); pi += n
            if wi < len(wmem) and wl < WFIFO_WORDS:
                n = min(len(wmem) - wi, (WFIFO_WORDS - wl) * 16)
                dma(D(OFF_WMEM + wi), WFIFO_BASE, n); wi += n
            cs = w.rd(R_CSTATUS)
            if cs & 0x4:   # cfg_err
                if not int(meta["cfg_err"]):
                    bad(f"{case}: unexpected cfg_err opcode 0x{(cs >> 8) & 0xFF:02x}")
                done = True
            elif (cs & 0xA) == 0xA:  # IRQ pending and trailing HALT consumed
                done = True
            elif time.perf_counter() - t0 > 30:
                bad(f"{case}: timeout (pi={pi}/{len(prog)} wi={wi}/{len(wmem)} CSTATUS=0x{cs:08x})")
                done = True
        w.wr(R_CTRL, C_UBOWN | C_ACCOWN)   # run = 0

        tag = w.rd(R_IRQ_TAG)
        exp_err = int(meta["cfg_err"])
        if not exp_err:
            (ok if (w.rd(R_CSTATUS) & 0xC) == 0x8 else bad)(f"{case}: completed with irq, no cfg_err")
            (ok if tag == int(meta["irq_tag"], 16) else bad)(
                f"{case}: IRQ_TAG 0x{tag:08x} (expect {meta['irq_tag']})")
            # UB output region
            buf[OFF_OUT:OFF_OUT + len(ub_exp)] = b"\xee" * len(ub_exp); sync(True)
            dma(UB_BASE + out_off, D(OFF_OUT), len(ub_exp)); sync(False)
            (ok if bytes(buf[OFF_OUT:OFF_OUT + len(ub_exp)]) == ub_exp else bad)(
                f"{case}: UB output rows {meta['out_first_row']}..+{meta['out_rows']} bit-exact")
            # ACC image
            if acc_exp is not None:
                buf[OFF_OUT:OFF_OUT + len(acc_exp)] = b"\xee" * len(acc_exp); sync(True)
                dma(ACC_BASE, D(OFF_OUT), len(acc_exp)); sync(False)
                (ok if bytes(buf[OFF_OUT:OFF_OUT + len(acc_exp)]) == bytes(acc_exp) else bad)(
                    f"{case}: ACC rows 0..{meta['acc_rows']} bit-exact")
        else:
            cs = w.rd(R_CSTATUS)
            (ok if (cs & 0x4) and ((cs >> 8) & 0xFF) == int(meta["cfg_err_op"], 16) else bad)(
                f"{case}: cfg_err raised, opcode 0x{(cs >> 8) & 0xFF:02x}")
        perf = [w.rd(R_PERF0 + 4 * i) for i in range(8)]
        print(f"[INFO] {case}: " + " ".join(f"{n}={v}" for n, v in zip(PERF_NAMES, perf)))
        w.wr(R_CTRL, 0)

    st = w.rd(R_STATUS)
    (ok if st == 0 else bad)(f"final STATUS={status_str(st)}")
    print("M4_BOARD_PASS" if fails == 0 else f"M4_BOARD_FAIL ({fails})")
    return 0 if fails == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
