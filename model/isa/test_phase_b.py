#!/usr/bin/env python3
"""Validate autonomous programs and the exact board runtime relocation path."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'model/isa'))
sys.path.insert(0,str(ROOT/'software/linux/idma_runtime'))
import asm
import sim
from m5_board_test import relocate_program, OFF


def main(cases=None):
    cases=Path(cases) if cases else ROOT/'build/m5/cases'
    passed=0
    for folder in sorted(p for p in cases.iterdir() if (p/'meta.txt').exists()):
        meta=sim.read_meta(folder/'meta.txt')
        data={key:asm.read_hex(folder/file) for key,file in
              [('input','ub_init.hex'),('weights','wmem.hex'),('output','ddr_expect.hex')]}
        original=asm.read_hex(folder/'prog.hex')
        reloc=json.loads((folder/'relocations.json').read_text())
        for phys in (0x39c00000,0x810000000):
            raw=b''.join(w.to_bytes(16,'little') for w in original)
            patched=relocate_program(raw,reloc,phys,16<<20,{k:len(v)*16 for k,v in data.items()})
            program=[int.from_bytes(patched[i:i+16],'little') for i in range(0,len(patched),16)]
            memory={}
            for region in ('input','weights'):
                memory.update((phys+OFF[region]+i*16+b,value) for i,w in enumerate(data[region])
                              for b,value in enumerate(w.to_bytes(16,'little')))
            memory.update((phys+OFF['output']+i,0xee) for i in range(len(data['output'])*16))
            model=sim.CoreSim(enable_dma=True,ddr=memory).run(program)
            assert model.cfg_err==bool(int(meta['cfg_err'])),folder
            assert model.dma_count==int(meta['dma_count']),folder
            if model.cfg_err:
                assert model.cfg_err_op==int(meta['cfg_err_op'],16) and model.irq_count==0,folder
            else:
                got=bytes(model.ddr[phys+OFF['output']+i] for i in range(len(data['output'])*16))
                expected=b''.join(w.to_bytes(16,'little') for w in data['output'])
                assert got==expected and model.irq_tag==int(meta['irq_tag'],16) and model.halted,folder
        passed+=1
    # Runtime rejects relocated transfers extending outside their supplied images.
    word=asm.read_host_mem(0,0x1000,2)
    try:
        relocate_program(word.to_bytes(16,'little'),[dict(index=0,region='input',offset=0)],
                         0x39c00000,16<<20,dict(input=16,weights=0,output=0))
    except ValueError: pass
    else: raise AssertionError('runtime accepted an out-of-image DMA')
    print(f'PHASE_B_MODEL_PASS cases={passed} low/high DDR relocation, runtime bounds')


if __name__=='__main__': main(sys.argv[1] if len(sys.argv)>1 else None)
