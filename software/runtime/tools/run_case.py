#!/usr/bin/env python3
"""Run compiled Phase B case directories on the TPU through /dev/tpu0 and compare bit-exactly.

    python3 run_case.py CASES_ROOT_OR_CASE_DIR... [--cases a,b,c] [--repeat 3] [--list]

A case directory (model/mlp/compile_phase_b.py, model/cnn/compile_cnn.py) holds prog.bin,
relocations.json ([{index, region, offset}], region = input|weights|output), ub_init.hex (input
rows), wmem.hex (weight tiles), ddr_expect.hex (expected output rows) and meta.txt.  A directory
with cases.txt is treated as a list of cases.  Cases with `cfg_err 1` in meta.txt must end in
CFG_ERR with the listed opcode (and the next case must still run: recovery).

This is the generic tool for programs that are not MLP bundles (CNN, DMA, error cases); MLP
bundles go through tpu_mlp.py.  Exit code 0 = all pass, 1 = failure, 2 = no device/cases.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tpu  # noqa: E402


def read_hex(path):
    """One 128-bit word per line, MSB first -> bytes in memory order (little endian)."""
    text = Path(path).read_text().split()
    return b"".join(int(w, 16).to_bytes(16, "little") for w in text)


def read_meta(path):
    meta = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            k, _, v = line.partition(" ")
            meta[k] = v.strip()
    return meta


def case_dirs(args):
    out = []
    for a in args:
        p = Path(a)
        if (p / "cases.txt").exists():
            for line in (p / "cases.txt").read_text().split():
                q = Path(line)
                out.append(q if q.is_absolute() or q.exists() else p / q.name)
        elif (p / "prog.bin").exists():
            out.append(p)
    return out


def run_case(dev, case, repeat):
    meta = read_meta(case / "meta.txt")
    prog = (case / "prog.bin").read_bytes()
    relocs = json.loads((case / "relocations.json").read_text())
    inputs = read_hex(case / "ub_init.hex") if (case / "ub_init.hex").stat().st_size else b""
    weights = read_hex(case / "wmem.hex") if (case / "wmem.hex").stat().st_size else b""
    expect = read_hex(case / "ddr_expect.hex") if (case / "ddr_expect.hex").stat().st_size else b""
    want_err = int(meta.get("cfg_err", "0")) != 0
    bufs = {}
    try:
        for name, data in (("input", inputs), ("weights", weights), ("output", expect)):
            bufs[name] = dev.alloc(max(len(data), 16))
        bufs["input"].write(inputs)
        bufs["weights"].write(weights)
        rel = [(r["index"], bufs[r["region"]], r["offset"]) for r in relocs]
        results = []
        for rep in range(repeat):
            bufs["output"].fill(0xEE, 0, max(len(expect), 16))
            try:
                res = dev.run(prog, rel, timeout_ms=5000)
            except tpu.TpuError as e:
                # Illegal raw DMA addresses (cfgerr_dma_*, no relocations) can never reach the
                # hardware through the driver: EINVAL from the kernel validation is the correct result.
                relocated = {r["index"] for r in relocs}
                raw_dma = [i for i in range(len(prog) // 16) if prog[i * 16] in (1, 2, 3) and i not in relocated]
                if e.errno == 22 and want_err and raw_dma:
                    results.append((True, "rejected by kernel validation (raw DMA address), by design", None))
                    continue
                raise
            got = bufs["output"].read(len(expect)) if expect else b""
            if want_err:
                good = res.status == tpu.RUN_CFG_ERR and res.cfg_err_op == int(meta["cfg_err_op"], 16)
                why = f"status={res.status_name} op={res.cfg_err_op:#x} want CFG_ERR op {meta['cfg_err_op']}"
            else:
                good = res.ok and got == expect
                why = f"status={res.status_name}" + (f" cfg_err_op={res.cfg_err_op:#x}" if res.status == tpu.RUN_CFG_ERR else "") + ("" if got == expect else
                      f" OUTPUT MISMATCH first diff @{next(i for i, (a, b) in enumerate(zip(got, expect)) if a != b)}")
            results.append((good, why, res))
        return results
    finally:
        for b in bufs.values():
            b.free()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--cases", default="", help="comma list of case names to run")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--device", default="/dev/tpu0")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--mock", action="store_true", help="run on the host mock (golden simulator)")
    args = ap.parse_args()
    cases = case_dirs(args.paths)
    if args.cases:
        keep = set(args.cases.split(","))
        cases = [c for c in cases if c.name in keep]
    if not cases:
        print("no cases found")
        return 2
    if args.list:
        print("\n".join(c.name for c in cases))
        return 0
    try:
        if args.mock:
            import tpu_mock
            dev = tpu.Device(backend=tpu_mock.MockKernel(max_buf=64 << 20, total_limit=256 << 20).backend())
        else:
            dev = tpu.Device(args.device)
    except OSError as e:
        print(f"cannot open {args.device}: {e}")
        return 2
    fails = 0
    with dev:
        info = dev.info()
        print(f"# {info}")
        for case in cases:
            results = run_case(dev, case, args.repeat)
            good = all(r[0] for r in results)
            fails += not good
            res = results[-1][2]
            if res is None:
                line = f"[{'PASS' if good else 'FAIL'}] {case.name:<24} x{len(results)} kernel rejected the program (EINVAL), as designed"
            else:
                perf = res.perf
                line = (f"[{'PASS' if good else 'FAIL'}] {case.name:<24} x{len(results)} "
                        f"wall={res.wall_us:7.1f}us cycles={perf['total']:6d} array={perf['array_active']:6d} "
                        f"wstall={perf['w_stall']:5d} retired={perf['retired']:4d}")
            if not good:
                line += "  " + "; ".join(r[1] for r in results if not r[0])[:160]
            print(line, flush=True)
        after = dev.info()
        print(f"# jobs ok/failed {after.jobs_ok}/{after.jobs_failed}  state {after.state_name}")
    print("TPU_CASES_PASS" if not fails else f"TPU_CASES_FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
