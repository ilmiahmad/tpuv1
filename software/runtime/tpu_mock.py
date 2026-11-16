#!/usr/bin/env python3
"""Host-side mock of /dev/tpu0: an executable description of the driver ABI, backed by the golden
instruction-level simulator (model/isa/sim.py).  Used by tests/test_host.py and
run_bundle_golden.py to exercise tpu.py / tpu_mlp.py without hardware.

It implements the semantics of software/driver/include/uapi/tpu_kv260.h:
  * per-descriptor buffers (page rounded, zero filled), driver-wide limits, handles > 0,
    mmap_offset = handle << 12, freed when the descriptor closes;
  * RUN validation before any hardware effect: flags, instruction/reloc counts, timeout,
    relocations only on DMA instructions (0x01/0x02/0x03), known handles, 16-byte aligned
    offsets, span inside the buffer, every DMA instruction relocated exactly once;
  * address patching and execution on the golden simulator against the buffers' contents.
"""
import ctypes
import errno
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "model", "isa"))
import tpu                                   # noqa: E402
import sim as isasim                         # noqa: E402

PAGE = 4096
DMA_OPS = (0x01, 0x02, 0x03)
FAKE_BASE = 0x8_1000_0000                    # inside the DDR-high window
FAKE_STRIDE = 0x100_0000                     # 16 MiB per handle


class _Mem(bytearray):
    def close(self):
        pass


class MockKernel:
    def __init__(self, max_buf=8 << 20, total_limit=64 << 20):
        self.max_buf, self.total_limit = max_buf, total_limit
        self.total = 0
        self.next_handle = 1
        self.bufs = {}                       # handle -> (owner, _Mem, size)
        self.slot_of = {}                    # handle -> address slot (reused after free, like a real allocator)
        self.jobs_ok = self.jobs_failed = self.irq_count = 0
        self.hw_version, self.build_id = 0x400, 0x6AF93498
        self.state = tpu.STATE_IDLE

    def backend(self):
        return MockBackend(self)


class MockBackend:
    def __init__(self, kernel):
        self.k = kernel
        self.owned = set()

    def close(self):
        for h in list(self.owned):
            self._free(h)

    # -- mmap ----------------------------------------------------------------------------------
    def mmap(self, length, offset):
        h = offset >> 12
        if h not in self.owned or offset & (PAGE - 1):
            raise OSError(errno.EINVAL, "bad mmap offset")
        mem = self.k.bufs[h][1]
        if length > len(mem):
            raise OSError(errno.EINVAL, "mmap beyond the buffer")
        return mem

    def _free(self, h):
        if h not in self.owned:
            raise OSError(errno.ENOENT, "unknown handle")
        self.owned.discard(h)
        _, mem, _ = self.k.bufs.pop(h)
        self.k.slot_of.pop(h, None)
        self.k.total -= len(mem)

    # -- ioctl ---------------------------------------------------------------------------------
    def ioctl(self, request, arg):
        if request == tpu.TPU_IOC_GET_INFO:
            return self._info(arg)
        if request == tpu.TPU_IOC_ALLOC:
            return self._alloc(arg)
        if request == tpu.TPU_IOC_FREE:
            return self._free(struct.unpack("<I", arg)[0])
        if request == tpu.TPU_IOC_RUN:
            return self._run(arg)
        if request == tpu.TPU_IOC_RECOVER:
            return 0
        raise OSError(errno.ENOTTY, "bad ioctl")

    def _info(self, arg):
        k = self.k
        arg[:] = struct.pack(tpu.INFO_FMT, tpu.ABI_VERSION, k.hw_version, k.build_id, k.state, tpu.MAX_INSTR,
                             65536, 2048, 1, k.max_buf, k.total_limit, k.total, k.jobs_ok, k.jobs_failed,
                             k.irq_count, 0, 0, 0, 0)

    def _alloc(self, arg):
        size, flags, _, _, _ = struct.unpack(tpu.ALLOC_FMT, arg)
        k = self.k
        if size == 0 or flags != 0:
            raise OSError(errno.EINVAL, "bad alloc")
        asize = (size + PAGE - 1) & ~(PAGE - 1)
        if asize > k.max_buf or k.total + asize > k.total_limit:
            raise OSError(errno.ENOSPC, "buffer limit")
        h = k.next_handle
        k.next_handle += 1
        k.bufs[h] = (self, _Mem(asize), asize)
        used = set(k.slot_of.values())
        k.slot_of[h] = next(i for i in range(len(used) + 1) if i not in used)
        k.total += asize
        self.owned.add(h)
        arg[:] = struct.pack(tpu.ALLOC_FMT, size, flags, h, h << 12, asize)

    def _run(self, arg):
        p, r, n_instr, n_rel, timeout_ms, flags = struct.unpack_from("<QQ4I", arg, 0)
        k = self.k
        if flags != 0 or not (1 <= n_instr <= tpu.MAX_INSTR) or n_rel > tpu.MAX_RELOCS or timeout_ms > 60000:
            raise OSError(errno.EINVAL, "bad run arguments")
        if p < 0x10000 or (n_rel and r < 0x10000):          # NULL page and below mmap_min_addr: EFAULT
            raise OSError(errno.EFAULT, "bad user pointer")
        prog = bytearray(ctypes.string_at(p, n_instr * 16))
        rels = [struct.unpack_from(tpu.RELOC_FMT, ctypes.string_at(r, n_rel * 16), 16 * i) for i in range(n_rel)]
        seen = set()
        for idx, h, off in rels:
            if idx >= n_instr:
                raise OSError(errno.EINVAL, "reloc index out of range")
            if idx in seen:
                raise OSError(errno.EINVAL, "duplicate relocation")
            seen.add(idx)
            word = int.from_bytes(prog[idx * 16:idx * 16 + 16], "little")
            if (word & 255) not in DMA_OPS:
                raise OSError(errno.EINVAL, "relocation on a non-DMA instruction")
            if h not in self.owned:
                raise OSError(errno.ENOENT, "unknown handle")
            size = len(k.bufs[h][1])
            if off % 16:
                raise OSError(errno.EINVAL, "misaligned offset")
            op = word & 255
            if op == 2:
                span = ((word >> 16) & 0xFFFF) * 256
            else:
                reps, length, stride = (word >> 89) & 0xFFF, ((word >> 77) & 0xFFF) * 16, (word >> 101) & 0xFFFFFF
                span = ((reps - 1) * stride * 16 + length) if reps and length else 0
            if off > size or off + span > size:
                raise OSError(errno.EINVAL, "relocation exceeds the buffer")
            addr = FAKE_BASE + k.slot_of[h] * FAKE_STRIDE + off
            word = (word & ~(((1 << 45) - 1) << 32)) | ((addr >> 4) << 32)
            prog[idx * 16:idx * 16 + 16] = word.to_bytes(16, "little")
        for i in range(n_instr):
            if prog[i * 16] in DMA_OPS and i not in seen:
                raise OSError(errno.EINVAL, "DMA instruction without relocation")
        # execute on the golden simulator
        ddr = {}
        used = {h for _, h, _ in rels}
        for h in used:
            base = FAKE_BASE + k.slot_of[h] * FAKE_STRIDE
            ddr.update({base + i: b for i, b in enumerate(k.bufs[h][1])})
        words = [int.from_bytes(prog[i * 16:i * 16 + 16], "little") for i in range(n_instr)]
        m = isasim.CoreSim(enable_dma=True, ddr=ddr).run(words)
        for h in used:
            base = FAKE_BASE + k.slot_of[h] * FAKE_STRIDE
            mem = k.bufs[h][1]
            mem[:] = bytes(m.ddr.get(base + i, 0) for i in range(len(mem)))
        if m.cfg_err:
            status = tpu.RUN_CFG_ERR
        elif m.halted and m.irq_count >= 1:
            status = tpu.RUN_OK
        else:
            status = tpu.RUN_TIMEOUT
        if status == tpu.RUN_OK:
            k.jobs_ok += 1
            k.irq_count += 1
        else:
            k.jobs_failed += 1
        perf = [0] * 8
        perf[7] = m.instr_count
        tag = m.irq_tag or 0
        arg[:] = struct.pack(tpu.RUN_FMT, p, r, n_instr, n_rel, timeout_ms, flags, status,
                             (m.cfg_err_op or 0) if m.cfg_err else 0, 0x1A if status == 0 else 0x4, 0,
                             tag, *perf, 1000 * n_instr, 0, 0, 0, 0)
