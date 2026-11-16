#!/usr/bin/env python3
"""MLP inference on the TPU through the Linux driver ("compile once, run many").

    import tpu, tpu_mlp
    with tpu.Device() as dev:
        net = tpu_mlp.TpuMlp(dev, "build/bundles/digits_b32")     # weights + program loaded once
        y_float = net.infer(x_float)        # [b<=32, 64] float -> [b, 10] float logits
        y_int8  = net.infer_q(x_int8)       # [b<=32, 64] int8  -> [b, 10] int8
        print(net.last)                     # timing / hardware counters of the last call

A bundle (model/mlp/export_bundle.py) fixes the batch size B of the compiled program.  Smaller
batches are padded with rows holding the input zero point (a real zero); larger batches raise.
Layouts: input UB row kt*B + b = x[b][16kt:16kt+16]; output row nt*B + b = y[b][16nt:16nt+16].
Works with or without numpy (without it the packing is pure Python and much slower).
"""
import json
import os
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:                     # pragma: no cover
    np = None

import tpu


class TpuRunError(RuntimeError):
    def __init__(self, result):
        super().__init__(f"TPU job failed: {result}")
        self.result = result


class Bundle:
    """Parsed bundle directory (see model/mlp/export_bundle.py)."""

    def __init__(self, path):
        self.path = Path(path)
        self.model = json.loads((self.path / "model.json").read_text())
        if self.model.get("bundle_version") != 1:
            raise ValueError(f"unsupported bundle_version {self.model.get('bundle_version')}")
        self.program = (self.path / "program.bin").read_bytes()
        self.weights = (self.path / "weights.bin").read_bytes()
        self.relocs = [(r["index"], r["buffer"], r["offset"])
                       for r in json.loads((self.path / "relocs.json").read_text())]
        m = self.model
        if len(self.program) != m["program_instrs"] * 16 or len(self.weights) != m["weights_bytes"]:
            raise ValueError("bundle sizes disagree with model.json")
        if len(self.relocs) != m["n_relocs"]:
            raise ValueError("relocation count disagrees with model.json")

    def sample_in(self):
        return (self.path / "sample_in.bin").read_bytes()

    def sample_out(self):
        return (self.path / "sample_out.bin").read_bytes()


def _pad16(n):
    return (n + 15) // 16 * 16


class TpuMlp:
    def __init__(self, dev, bundle, timeout_ms=0):
        self.dev = dev
        self.bundle = bundle if isinstance(bundle, Bundle) else Bundle(bundle)
        self.model = m = self.bundle.model
        self.B, self.K, self.N = m["batch"], m["in_features"], m["out_features"]
        self.kt, self.nt = _pad16(self.K) // 16, _pad16(self.N) // 16
        self.timeout_ms = timeout_ms
        self.in_buf = dev.alloc(m["input_bytes"])
        self.w_buf = dev.alloc(m["weights_bytes"])
        self.out_buf = dev.alloc(m["output_bytes"])
        self.in_buf.fill(0)                              # rows beyond the input stay zero (UB init)
        self.w_buf.write(self.bundle.weights)            # Weight Memory: written once
        bufs = {"input": self.in_buf, "weights": self.w_buf, "output": self.out_buf}
        self._relocs = [(i, bufs[b], off) for i, b, off in self.bundle.relocs]
        self.last = {}
        self.totals = dict(runs=0, kernel_us=0.0, host_us=0.0, samples=0)

    def close(self):
        for b in (self.in_buf, self.w_buf, self.out_buf):
            b.free()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- packing -------------------------------------------------------------------------------
    def _check_q(self, x_q):
        if np is not None:
            a = np.asarray(x_q)
            if a.ndim != 2 or a.shape[1] != self.K:
                raise ValueError(f"expected shape [b, {self.K}], got {a.shape}")
            if a.shape[0] > self.B or a.shape[0] < 1:
                raise ValueError(f"batch {a.shape[0]} outside 1..{self.B} (compiled batch)")
            if a.dtype.kind not in "iu" or int(a.min()) < -128 or int(a.max()) > 127:
                raise ValueError("input must be integers in [-128, 127]")
            return a
        rows = [list(r) for r in x_q]
        if not rows or len(rows) > self.B or any(len(r) != self.K for r in rows):
            raise ValueError(f"expected [1..{self.B}, {self.K}] input")
        if any((not isinstance(v, int)) or v < -128 or v > 127 for r in rows for v in r):
            raise ValueError("input must be integers in [-128, 127]")
        return rows

    def _pack(self, x_q):
        zp = self.model["in_zp"]
        if np is not None:
            a = x_q
            full = np.full((self.B, self.kt * 16), zp, dtype=np.int8)
            full[:a.shape[0], :self.K] = a
            return np.ascontiguousarray(full.reshape(self.B, self.kt, 16).transpose(1, 0, 2)).tobytes()
        rows = x_q
        out = bytearray(self.kt * self.B * 16)
        zpb = zp & 0xFF
        for b in range(self.B):
            r = rows[b] if b < len(rows) else None
            for kt in range(self.kt):
                base = (kt * self.B + b) * 16
                for lane in range(16):
                    k = kt * 16 + lane
                    out[base + lane] = (r[k] & 0xFF) if (r is not None and k < self.K) else zpb
        return bytes(out)

    def _unpack(self, raw, b):
        if np is not None:
            a = np.frombuffer(raw, dtype=np.int8).reshape(self.nt, self.B, 16)
            return np.ascontiguousarray(a.transpose(1, 0, 2).reshape(self.B, self.nt * 16)[:b, :self.N])
        out = []
        for bi in range(b):
            row = []
            for n in range(self.N):
                nt, lane = divmod(n, 16)
                v = raw[(nt * self.B + bi) * 16 + lane]
                row.append(v - 256 if v > 127 else v)
            out.append(row)
        return out

    # -- inference -----------------------------------------------------------------------------
    def _run(self):
        t = time.perf_counter()
        res = self.dev.run(self.bundle.program, self._relocs, self.timeout_ms)
        run_us = (time.perf_counter() - t) * 1e6
        if not res.ok:
            raise TpuRunError(res)
        return res, run_us

    def infer_q(self, x_q):
        """int8 [b, K] -> int8 [b, N] (numpy array if numpy is available, else lists)."""
        t0 = time.perf_counter()
        x = self._check_q(x_q)
        b = len(x)
        packed = self._pack(x)
        self.in_buf.write(packed)
        t1 = time.perf_counter()
        res, run_us = self._run()
        t2 = time.perf_counter()
        y = self._unpack(self.out_buf.read(self.model["out_rows"] * 16), b)
        t3 = time.perf_counter()
        self.last = dict(pack_us=(t1 - t0) * 1e6, run_us=run_us, unpack_us=(t3 - t2) * 1e6,
                         host_us=(t3 - t0) * 1e6, kernel_us=res.wall_us, perf=res.perf, status=res.status_name)
        self.totals["runs"] += 1
        self.totals["kernel_us"] += res.wall_us
        self.totals["host_us"] += self.last["host_us"]
        self.totals["samples"] += b
        return y

    def quantize(self, x):
        """float [b, K] -> int8 values with the bundle's input scale / zero point."""
        s, zp = self.model["in_scale"], self.model["in_zp"]
        if np is not None:
            a = np.asarray(x, dtype=np.float64)
            return np.clip(np.round(a / s) + zp, -128, 127).astype(np.int8)
        return [[max(-128, min(127, round(v / s) + zp)) for v in r] for r in x]

    def dequantize(self, y_q):
        s, zp = self.model["out_scale"], self.model["out_zp"]
        if np is not None:
            return (np.asarray(y_q, dtype=np.float64) - zp) * s
        return [[(v - zp) * s for v in r] for r in y_q]

    def infer(self, x):
        """float [b, K] -> float [b, N] (quantize, run, dequantize)."""
        return self.dequantize(self.infer_q(self.quantize(x)))

    def infer_q_many(self, x_q):
        """Run any number of samples in chunks of the compiled batch (last chunk padded)."""
        n = len(x_q)
        outs = [self.infer_q(x_q[i:i + self.B]) for i in range(0, n, self.B)]
        return np.concatenate(outs) if np is not None else [r for o in outs for r in o]

    def run_sample(self):
        """Run the bundle's own sample; returns (RunResult, bit_exact). The output buffer is
        pre-filled with 0xEE, so a byte the program failed to write shows up as a mismatch."""
        self.in_buf.write(self.bundle.sample_in())
        self.out_buf.fill(0xEE)
        res = self.dev.run(self.bundle.program, self._relocs, self.timeout_ms)
        got = self.out_buf.read(self.model["output_bytes"])
        return res, (res.ok and got == self.bundle.sample_out())
