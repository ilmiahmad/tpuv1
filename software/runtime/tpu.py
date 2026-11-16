#!/usr/bin/env python3
"""Python binding for the TPU-KV260 Linux driver (/dev/tpu0, ABI v1).

Stdlib only (numpy is optional and only used for zero-copy views).  The ABI is
software/driver/include/uapi/tpu_kv260.h; the struct layouts and ioctl numbers below must match it
byte for byte (tests/test_host.py compiles the header on the host and compares; so does
`python3 tpu.py --selfcheck`, which also works on the board where gcc exists).

    with tpu.Device() as dev:
        print(dev.info())
        inp, wts, out = dev.alloc(4096), dev.alloc(65536), dev.alloc(4096)
        res = dev.run(program_bytes, [(idx, inp, 0), (idx2, wts, 0), (idx3, out, 0)])
        assert res.ok
"""
import ctypes
import errno
import mmap
import os
import struct
import sys
import time
from dataclasses import dataclass, field

try:                                    # optional
    import numpy as np
except ImportError:                     # pragma: no cover
    np = None

ABI_VERSION = 1
MAX_INSTR = 2048
MAX_RELOCS = 2048
NUM_PERF = 8
PERF_NAMES = ("total", "array_active", "w_stall", "w_shift", "non_matrix", "sync", "act", "retired")

STATE_IDLE, STATE_BUSY, STATE_WEDGED = 0, 1, 2
STATE_NAMES = {STATE_IDLE: "idle", STATE_BUSY: "busy", STATE_WEDGED: "wedged"}

RUN_OK, RUN_CFG_ERR, RUN_DMA_FAULT, RUN_TIMEOUT, RUN_PUSH_FAIL = range(5)
STATUS_NAMES = {RUN_OK: "OK", RUN_CFG_ERR: "CFG_ERR", RUN_DMA_FAULT: "DMA_FAULT",
                RUN_TIMEOUT: "TIMEOUT", RUN_PUSH_FAIL: "PUSH_FAIL"}

# ---- struct layouts (little endian, explicit padding; sizes asserted below) -------------------
INFO_FMT = "<8I 10Q"                    # tpu_info: 8 x u32, 6 counters/limits + reserved[4]
ALLOC_FMT = "<QIIQQ"                    # tpu_alloc
RELOC_FMT = "<IIQ"                      # tpu_reloc
RUN_FMT = "<QQ4I 5I 8I 4x Q 4Q"         # tpu_run: ptrs, n/timeout/flags | outputs | perf | pad | wall_ns | rsv
INFO_SIZE, ALLOC_SIZE, RELOC_SIZE, RUN_SIZE = (struct.calcsize(f) for f in (INFO_FMT, ALLOC_FMT, RELOC_FMT, RUN_FMT))
assert (INFO_SIZE, ALLOC_SIZE, RELOC_SIZE, RUN_SIZE) == (112, 32, 16, 128)

# ---- ioctl numbers (asm-generic: aarch64 and x86; 2 dir bits, 14 size bits, 8 type, 8 nr) ------
_IOC_NONE, _IOC_WRITE, _IOC_READ = 0, 1, 2


def _ioc(direction, typ, nr, size):
    return (direction << 30) | (size << 16) | (ord(typ) << 8) | nr


TPU_IOC_GET_INFO = _ioc(_IOC_READ, "T", 0, INFO_SIZE)
TPU_IOC_ALLOC = _ioc(_IOC_READ | _IOC_WRITE, "T", 1, ALLOC_SIZE)
TPU_IOC_FREE = _ioc(_IOC_WRITE, "T", 2, 4)
TPU_IOC_RUN = _ioc(_IOC_READ | _IOC_WRITE, "T", 3, RUN_SIZE)
TPU_IOC_RECOVER = _ioc(_IOC_NONE, "T", 4, 0)


@dataclass
class Info:
    abi_version: int
    hw_version: int
    build_id: int
    state: int
    max_instr: int
    ub_rows: int
    acc_rows: int
    irq_mode: int
    max_buf_bytes: int
    total_buf_limit: int
    total_buf_bytes: int
    jobs_ok: int
    jobs_failed: int
    irq_count: int

    @property
    def state_name(self):
        return STATE_NAMES.get(self.state, f"unknown({self.state})")

    def __str__(self):
        return (f"TPU abi={self.abi_version} hw_version=0x{self.hw_version:08x} build_id=0x{self.build_id:08x} "
                f"state={self.state_name} irq={'on' if self.irq_mode else 'poll'} max_instr={self.max_instr} "
                f"ub_rows={self.ub_rows} acc_rows={self.acc_rows} buf={self.total_buf_bytes}/{self.total_buf_limit}B "
                f"(max {self.max_buf_bytes}B) jobs ok/failed={self.jobs_ok}/{self.jobs_failed} irqs={self.irq_count}")


@dataclass
class RunResult:
    status: int
    cfg_err_op: int
    core_status: int
    hw_status: int
    irq_tag: int
    perf: dict = field(default_factory=dict)
    wall_ns: int = 0

    @property
    def ok(self):
        return self.status == RUN_OK

    @property
    def status_name(self):
        return STATUS_NAMES.get(self.status, f"unknown({self.status})")

    @property
    def wall_us(self):
        return self.wall_ns / 1000.0

    def __str__(self):
        base = f"{self.status_name} wall={self.wall_us:.1f}us tag=0x{self.irq_tag:08x}"
        if self.status == RUN_CFG_ERR:
            base += f" cfg_err_op=0x{self.cfg_err_op:02x}"
        return base + " perf=" + ",".join(f"{k}={v}" for k, v in self.perf.items())


class TpuError(OSError):
    """ioctl/mmap failure; .errno carries the kernel errno."""


def _addr_of(obj):
    """Address of a bytes/bytearray/memoryview/mmap payload (kept alive by the caller)."""
    if isinstance(obj, bytes):
        return ctypes.cast(ctypes.c_char_p(obj), ctypes.c_void_p).value
    mv = memoryview(obj)
    if mv.readonly:
        raise TypeError("need a writable buffer or bytes")
    return ctypes.addressof((ctypes.c_char * mv.nbytes).from_buffer(mv))


class Buffer:
    """DMA buffer owned by a Device. `mem` supports slicing and the buffer protocol."""

    def __init__(self, dev, handle, size, alloc_size, mmap_offset, mem):
        self._dev, self.handle, self.size = dev, handle, size
        self.alloc_size, self.mmap_offset, self.mem = alloc_size, mmap_offset, mem
        self._closed = False

    def view(self, offset=0, length=None):
        length = self.size - offset if length is None else length
        return memoryview(self.mem)[offset:offset + length]

    def np(self, dtype="uint8", offset=0, count=-1):
        if np is None:
            raise RuntimeError("numpy not available")
        return np.frombuffer(self.mem, dtype=dtype, count=count, offset=offset)

    def write(self, data, offset=0):
        if offset < 0 or offset + len(data) > self.alloc_size:
            raise ValueError("write beyond the buffer")
        self.mem[offset:offset + len(data)] = bytes(data) if not isinstance(data, (bytes, bytearray)) else data

    def read(self, length=None, offset=0):
        length = self.size - offset if length is None else length
        return bytes(self.mem[offset:offset + length])

    def fill(self, byte, offset=0, length=None):
        length = self.size - offset if length is None else length
        self.mem[offset:offset + length] = bytes([byte]) * length

    def free(self):
        if not self._closed:
            self._closed = True
            self._dev._free_buffer(self)

    close = free

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.free()


class _FileBackend:
    """Real /dev/tpu0 access: fcntl.ioctl + mmap."""

    def __init__(self, path):
        self.fd = os.open(path, os.O_RDWR | getattr(os, "O_CLOEXEC", 0))

    def ioctl(self, request, arg):
        import fcntl
        return fcntl.ioctl(self.fd, request, arg, True)

    def mmap(self, length, offset):
        return mmap.mmap(self.fd, length, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=offset)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class Device:
    """One open file descriptor on /dev/tpu0 (buffers are per-descriptor and freed on close)."""

    def __init__(self, path="/dev/tpu0", backend=None):
        self.path = path
        self._be = backend if backend is not None else _FileBackend(path)
        self._buffers = {}

    # -- plumbing ------------------------------------------------------------------------------
    def _ioctl(self, request, arg):
        try:
            return self._be.ioctl(request, arg)
        except OSError as e:
            raise TpuError(e.errno, os.strerror(e.errno) if e.errno else str(e)) from None

    def close(self):
        for b in list(self._buffers.values()):
            try:
                b.free()
            except OSError:
                pass
        if self._be is not None:
            self._be.close()
            self._be = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- API -----------------------------------------------------------------------------------
    def info(self):
        buf = bytearray(INFO_SIZE)
        self._ioctl(TPU_IOC_GET_INFO, buf)
        v = struct.unpack(INFO_FMT, buf)
        return Info(*v[:8], *v[8:14])

    def alloc(self, size):
        """Allocate and mmap a DMA buffer of at least `size` bytes (zero filled by the kernel)."""
        arg = bytearray(struct.pack(ALLOC_FMT, size, 0, 0, 0, 0))
        self._ioctl(TPU_IOC_ALLOC, arg)
        _, _, handle, off, asize = struct.unpack(ALLOC_FMT, arg)
        try:
            mem = self._be.mmap(asize, off)
        except OSError as e:
            try:
                self._ioctl(TPU_IOC_FREE, bytearray(struct.pack("<I", handle)))
            finally:
                raise TpuError(e.errno, "mmap failed") from None
        buf = Buffer(self, handle, size, asize, off, mem)
        self._buffers[handle] = buf
        return buf

    def free_handle(self, handle):
        """Raw free by handle (negative tests)."""
        self._ioctl(TPU_IOC_FREE, bytearray(struct.pack("<I", handle)))

    def _free_buffer(self, buf):
        self._buffers.pop(buf.handle, None)
        try:
            buf.mem.close()
        except BufferError:
            pass                                    # an exported view still exists; process exit unmaps
        self._ioctl(TPU_IOC_FREE, bytearray(struct.pack("<I", buf.handle)))

    @staticmethod
    def _pack_relocs(relocs):
        out = bytearray()
        for index, buf, offset in relocs:
            out += struct.pack(RELOC_FMT, index, buf.handle if isinstance(buf, Buffer) else int(buf), offset)
        return out

    def run_raw(self, program_ptr, relocs_ptr, n_instr, n_relocs, timeout_ms=0, flags=0):
        """Low-level RUN with raw user pointers (used by the negative tests)."""
        arg = bytearray(RUN_SIZE)
        struct.pack_into("<QQ4I", arg, 0, program_ptr, relocs_ptr, n_instr, n_relocs, timeout_ms, flags)
        self._ioctl(TPU_IOC_RUN, arg)
        return self._decode_run(arg)

    @staticmethod
    def _decode_run(arg):
        v = struct.unpack(RUN_FMT, arg)
        perf = dict(zip(PERF_NAMES, v[11:19]))
        return RunResult(status=v[6], cfg_err_op=v[7], core_status=v[8], hw_status=v[9], irq_tag=v[10],
                         perf=perf, wall_ns=v[19])

    def run(self, program, relocs=(), timeout_ms=0):
        """Run one program. `program`: bytes-like, n*16 bytes. `relocs`: iterable of
        (instr_index, Buffer-or-handle, byte_offset). Returns RunResult (raises TpuError for
        rejected arguments, EINTR, or a wedged device)."""
        prog = bytes(program) if not isinstance(program, (bytes, bytearray)) else program
        if len(prog) % 16:
            raise ValueError("program length must be a multiple of 16 bytes")
        rel = self._pack_relocs(relocs)
        prog_b = bytearray(prog)                                   # stable, writable copy
        pptr = _addr_of(prog_b) if prog_b else 0
        rptr = _addr_of(rel) if rel else 0
        res = self.run_raw(pptr, rptr, len(prog_b) // 16, len(rel) // RELOC_SIZE, timeout_ms, 0)
        return res

    def recover(self):
        self._ioctl(TPU_IOC_RECOVER, 0)


# ---- self check against the C header -----------------------------------------------------------
_CHECK_C = r"""
#include <stdio.h>
#include <stddef.h>
#include <linux/ioctl.h>
#include "tpu_kv260.h"
int main(void) {
  printf("INFO %zu ALLOC %zu RELOC %zu RUN %zu\n", sizeof(struct tpu_info), sizeof(struct tpu_alloc),
         sizeof(struct tpu_reloc), sizeof(struct tpu_run));
  printf("run_off status %zu wall_ns %zu perf %zu program %zu\n", offsetof(struct tpu_run, status),
         offsetof(struct tpu_run, wall_ns), offsetof(struct tpu_run, perf), offsetof(struct tpu_run, program));
  printf("info_off max_buf_bytes %zu jobs_ok %zu irq_count %zu\n", offsetof(struct tpu_info, max_buf_bytes),
         offsetof(struct tpu_info, jobs_ok), offsetof(struct tpu_info, irq_count));
  printf("IOC %lu %lu %lu %lu %lu\n", (unsigned long)TPU_IOC_GET_INFO, (unsigned long)TPU_IOC_ALLOC,
         (unsigned long)TPU_IOC_FREE, (unsigned long)TPU_IOC_RUN, (unsigned long)TPU_IOC_RECOVER);
  return 0;
}
"""


def c_selfcheck(header_dir=None, cc="gcc"):
    """Compile the uapi header and compare sizes/offsets/ioctl numbers; returns a list of mismatches
    (empty = all fine). Raises RuntimeError if no compiler is available."""
    import shutil
    import subprocess
    import tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    header_dir = header_dir or os.path.join(here, "..", "driver", "include", "uapi")
    if shutil.which(cc) is None:
        raise RuntimeError(f"{cc} not found")
    with tempfile.TemporaryDirectory() as d:
        src, exe = os.path.join(d, "chk.c"), os.path.join(d, "chk")
        open(src, "w").write(_CHECK_C)
        subprocess.run([cc, "-I", header_dir, "-o", exe, src], check=True, capture_output=True)
        out = subprocess.run([exe], check=True, capture_output=True, text=True).stdout.split("\n")
    want = {
        "INFO": INFO_SIZE, "ALLOC": ALLOC_SIZE, "RELOC": RELOC_SIZE, "RUN": RUN_SIZE}
    bad = []
    t = out[0].split()
    got = {t[i]: int(t[i + 1]) for i in range(0, len(t), 2)}
    bad += [f"sizeof {k}: C {got[k]} py {want[k]}" for k in want if got[k] != want[k]]
    t = out[1].split()[1:]
    got = {t[i]: int(t[i + 1]) for i in range(0, len(t), 2)}
    py = {"program": 0,
          "status": struct.calcsize("<QQ4I"),
          "perf": struct.calcsize("<QQ4I 5I"),
          "wall_ns": struct.calcsize("<QQ4I 5I 8I 4x")}
    bad += [f"run offset {k}: C {got[k]} py {py[k]}" for k in py if got[k] != py[k]]
    t = out[2].split()[1:]
    got = {t[i]: int(t[i + 1]) for i in range(0, len(t), 2)}
    py = {"max_buf_bytes": struct.calcsize("<8I"), "jobs_ok": struct.calcsize("<8I 3Q"),
          "irq_count": struct.calcsize("<8I 5Q")}
    bad += [f"info offset {k}: C {got[k]} py {py[k]}" for k in py if got[k] != py[k]]
    nums = [int(x) for x in out[3].split()[1:]]
    pyn = [TPU_IOC_GET_INFO, TPU_IOC_ALLOC, TPU_IOC_FREE, TPU_IOC_RUN, TPU_IOC_RECOVER]
    bad += [f"ioctl {i}: C {a:#x} py {b:#x}" for i, (a, b) in enumerate(zip(nums, pyn)) if a != b]
    return bad


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selfcheck" in argv:
        bad = c_selfcheck()
        print("TPU_PY_ABI_SELFCHECK_" + ("FAIL: " + "; ".join(bad) if bad else "PASS"))
        return 1 if bad else 0
    path = argv[0] if argv else "/dev/tpu0"
    try:
        with Device(path) as dev:
            print(dev.info())
    except FileNotFoundError:
        print(f"{path}: no such device (is the driver loaded? see docs/driver_user_guide.md)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
