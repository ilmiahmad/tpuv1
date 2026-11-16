#!/usr/bin/env python3
"""Turn the four M4 MLPs into autonomous DDR -> TPU -> DDR programs.

The host relocates DDR addresses and loads the complete program before RUN.
Every input, weight tile and output transfer is then issued by the decoder.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'model/isa'))
import asm
import sim

INPUT, WEIGHTS, OUTPUT = 0x810000000, 0x820000000, 0x830000000
IFIFO_WORDS = 2048


def image(words, base):
    return {base + i*16 + b: v for i, word in enumerate(words)
            for b, v in enumerate(word.to_bytes(16, 'little'))}


def write_meta(path, values):
    path.write_text(''.join(f'{k} {v}\n' for k, v in values.items()))


def save(path, prog, reloc, inputs, weights, expected, meta, ub_expected=None):
    path.mkdir(parents=True, exist_ok=True)
    if len(prog) > IFIFO_WORDS:
        raise ValueError(f'{path.name}: program exceeds {IFIFO_WORDS}-entry IFIFO')
    asm.write_hex(path/'prog.hex', prog)
    asm.write_bin(path/'prog.bin', prog)
    (path/'prog.lst').write_text(''.join(f'{i:4d} {asm.disasm(w)}\n' for i, w in enumerate(prog)))
    (path/'relocations.json').write_text(json.dumps(reloc, indent=2)+'\n')
    asm.write_hex(path/'ub_init.hex', inputs)
    asm.write_hex(path/'wmem.hex', weights)
    asm.write_hex(path/'ddr_expect.hex', expected)
    if ub_expected is not None:
        asm.write_hex(path/'ub_expect.hex', ub_expected)
    meta.update(prog_instrs=len(prog), weight_words=len(weights), ub_init_rows=len(inputs),
                ddr_out_rows=len(expected), ddr_input=f'{INPUT:x}', ddr_weights=f'{WEIGHTS:x}',
                ddr_output=f'{OUTPUT:x}')
    write_meta(path/'meta.txt', meta)


def autonomous(prog, init_rows, out_first, out_rows):
    out, reloc = [], []
    def append(word, region=None, offset=0):
        if region is not None:
            reloc.append(dict(index=len(out), region=region, offset=offset))
        out.append(word)
    for row in range(0, init_rows, 4095):
        append(asm.read_host_mem(row, INPUT+row*16, min(4095, init_rows-row)), 'input', row*16)
    append(asm.sync(asm.U_DMA))
    weight_offset = 0
    for word in prog[:-2]:   # keep final SYNC; replace IRQ/HALT after output DMA
        op, fields, _, _ = asm.decode(word)
        if op in (asm.MATMUL, asm.CONVOLVE) and fields['new_weights']:   # M6: CONVOLVE consumes tiles too
            append(asm.read_weights(WEIGHTS+weight_offset, 1), 'weights', weight_offset)
            weight_offset += 256
        append(word)
    for row in range(0, out_rows, 4095):
        append(asm.write_host_mem(out_first+row, OUTPUT+row*16, min(4095, out_rows-row)), 'output', row*16)
    append(asm.sync(asm.U_DMA))
    out.extend(prog[-2:])
    return out, reloc, weight_offset


def build_mlp(src, dst):
    meta = sim.read_meta(src/'meta.txt')
    inputs, weights = asm.read_hex(src/'ub_init.hex'), asm.read_hex(src/'wmem.hex')
    expected = asm.read_hex(src/'ub_expect.hex')
    program, reloc, consumed = autonomous(asm.read_hex(src/'prog.hex'), len(inputs),
                                         int(meta['out_first_row']), int(meta['out_rows']))
    assert consumed == len(weights)*16
    memory = image(inputs, INPUT) | image(weights, WEIGHTS)
    model = sim.CoreSim(enable_dma=True, ddr=memory).run(program)
    got = bytes(model.ddr[OUTPUT+i] for i in range(len(expected)*16))
    assert got == b''.join(w.to_bytes(16, 'little') for w in expected), src.name
    assert model.halted and model.irq_count == 1 and not model.cfg_err
    meta.update(dma_count=model.dma_count, phase='B')
    save(dst, program, reloc, inputs, weights, expected, meta, expected)
    shutil.copyfile(src/'acc_expect.hex', dst/'acc_expect.hex')
    print(f'{dst.name}: {len(program)} instructions, {model.dma_count} DMA commands, model bit-exact')


def build_2d(dst):
    inputs = [int.from_bytes(bytes((r*17+b*3)&255 for b in range(16)), 'little') for r in range(12)]
    expected = [int.from_bytes(b'\xee'*16, 'little')]*12
    ub = []
    for rep in range(3):
        for row in range(2):
            expected[rep*5+row] = inputs[rep*5+row]
            ub.append(inputs[rep*5+row])
    p = [asm.read_host_mem(65530, INPUT, 2, 3, 80), asm.sync(asm.U_DMA),
         asm.write_host_mem(65530, OUTPUT, 2, 3, 80), asm.interrupt_host(0xd0e00002), asm.halt()]
    # No explicit SYNC follows WHM: INTERRUPT_HOST itself must drain DMA.
    mem = image(inputs, INPUT) | image([int.from_bytes(b'\xee'*16,'little')]*12, OUTPUT)
    m = sim.CoreSim(enable_dma=True, ddr=mem).run(p)
    assert bytes(m.ddr[OUTPUT+i] for i in range(192)) == b''.join(w.to_bytes(16,'little') for w in expected)
    save(dst, p, [dict(index=0,region='input',offset=0),dict(index=2,region='output',offset=0)],
         inputs, [], expected, dict(out_first_row=65530,out_rows=6,acc_rows=0,irq_expected=1,
         irq_tag='d0e00002',cfg_err=0,cfg_err_op='00',dma_count=2,phase='B'), ub)
    (dst/'acc_expect.hex').write_text('')


def build_errors(out):
    rhm = asm.read_host_mem
    whm = asm.write_host_mem
    rw = asm.read_weights
    cases = dict(
        dma_zero_rows=rhm(0,INPUT,0), dma_zero_reps=rhm(0,INPUT,1,0),
        dma_ub_end=rhm(65535,INPUT,2), dma_ub_product=rhm(0,INPUT,4095,4095),
        dma_pl_source=rhm(0,0xa0100000,1), dma_ddr_hole=rhm(0,0x80000000,1),
        dma_low_cross=rhm(0,0x7ffffff0,2), dma_high_cross=rhm(0,0x87ffffff0,2),
        dma_stride_cross=rhm(0,0x7ffffff0,1,2,32),
        dma_reserved=rhm(0,INPUT,1)|(1<<127), dma_flags=rhm(0,INPUT,1)|(1<<8),
        write_zero_rows=whm(0,OUTPUT,0), write_zero_reps=whm(0,OUTPUT,1,0),
        write_ub_end=whm(65535,OUTPUT,2), write_pl_destination=whm(0,0xa0100000,1),
        write_stride_cross=whm(0,0x87ffffff0,1,2,32),
        weights_zero=rw(WEIGHTS,0), weights_pl_source=rw(0xa0300000,1),
        weights_low_cross=rw(0x7ffffff0,1), weights_reserved=rw(WEIGHTS,1)|(1<<77),
        weights_flags=rw(WEIGHTS,1)|(1<<8), weights_unmapped=rw(1<<48,1))
    for name, word in cases.items():
        p = [word, asm.interrupt_host(0xbad), asm.halt()]
        m = sim.CoreSim(enable_dma=True).run(p)
        assert m.cfg_err and m.cfg_err_op == word&255 and m.dma_count == 0
        dst = out/('cfgerr_'+name)
        save(dst,p,[],[],[],[],dict(out_first_row=0,out_rows=0,acc_rows=0,irq_expected=0,
             irq_tag='00000000',cfg_err=1,cfg_err_op=f'{word&255:02x}',dma_count=0,phase='B'),[])
        (dst/'acc_expect.hex').write_text('')
    return ['cfgerr_'+n for n in cases]


def build_cnn(source, out):
    """M6: autonomous versions of the Phase A CNN cases (legal programs only; the cfgerr_* legality
    cases are decoder-level and identical in both phases, so they stay Phase A)."""
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for line in (source/'cases.txt').read_text().split():
        folder = ROOT/line if not Path(line).is_absolute() else Path(line)
        if int(sim.read_meta(folder/'meta.txt')['cfg_err']): continue
        build_mlp(folder, out/folder.name)
        names.append(folder.name)
    (out/'cases.txt').write_text(''.join(str(out/n)+'\n' for n in names))
    print(f'{len(names)} autonomous CNN cases -> {out}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default=str(ROOT/'build/m5/cases'))
    parser.add_argument('--phase-a-cases',default=str(ROOT/'build/m4/cases'))
    parser.add_argument('--cnn',action='store_true',help='build the Phase B CNN cases only (writes --out)')
    parser.add_argument('--cnn-src',default=str(ROOT/'build/m6/cases'))
    args=parser.parse_args()
    if args.cnn:
        build_cnn(Path(args.cnn_src), Path(args.out))
        return
    out, source=Path(args.out),Path(args.phase_a_cases)
    if not (source/'mlp_wide/prog.hex').exists():
        subprocess.run([sys.executable,str(ROOT/'model/mlp/compile_mlp.py'),'--out',str(source)],check=True)
    out.mkdir(parents=True,exist_ok=True)
    names=['fc1','mlp3','mlp_b1','mlp_wide']
    for name in names: build_mlp(source/name,out/name)
    build_2d(out/'dma_2d'); names.append('dma_2d')
    names += build_errors(out)
    (out/'cases.txt').write_text(''.join(str(out/n)+'\n' for n in names))
    print(f'{len(names)} autonomous/legality cases -> {out}')


if __name__=='__main__': main()
