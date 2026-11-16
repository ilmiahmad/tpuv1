#!/usr/bin/env python3
"""Autonomous Phase B board gate: no Linux DMA launch between RUN and IRQ/HALT."""
import argparse
import glob
import json
import mmap
import os
from pathlib import Path
import sys
import time

from m4_board_test import Win, load_hex, load_meta, sysfs

VERSION=0x400   # Phase B; the CNN-capable image reports 0x500 (same register map)
CTRL,STATUS,CSTATUS,IRQ_TAG,LEVEL,PERF=0x14,0x10,0x1c,0x20,0x18,0x24
IDMA=0x1000
SRC,DST,LENGTH,NEXT,DONE=0xd8,0xd0,0xe0,0x44,0x84
UB,ACC,IFIFO=0xa0100000,0xa0200000,0xa0400000
OFF=dict(input=0x100000,program=0x400000,weights=0x600000,output=0x800000,scratch=0xa00000)
CLEAR=0x361


def relocate_program(program, relocations, phys, size, region_sizes):
    """Patch the frozen ISA address fields, checking every access against its buffer."""
    result=bytearray(program)
    opcodes={'input':1,'weights':2,'output':3}
    for relocation in relocations:
        index,region,offset=relocation['index'],relocation['region'],relocation['offset']
        if region not in opcodes or not 0<=index<len(result)//16 or offset<0:
            raise ValueError('invalid program relocation')
        word=int.from_bytes(result[index*16:index*16+16],'little')
        if word&255!=opcodes[region]: raise ValueError('relocation opcode/region mismatch')
        if region=='weights': span=((word>>16)&65535)*256
        else:
            rows=(word>>89)&4095; length=((word>>77)&4095)*16
            if rows==0 or length==0: raise ValueError('zero relocated DMA dimensions')
            span=(rows-1)*((word>>101)&0xffffff)*16+length
        address=phys+OFF[region]+offset
        if span==0 or offset+span>region_sizes[region] or OFF[region]+offset+span>size:
            raise ValueError(f'{region} relocation exceeds image/buffer')
        if address&15 or address+span>1<<49: raise ValueError('DMA address alignment/width')
        word=(word&~(((1<<45)-1)<<32))|((address>>4)<<32)
        result[index*16:index*16+16]=word.to_bytes(16,'little')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--build-id',required=True)
    p.add_argument('--cases-dir',default='/tmp/tpu_m5/cases')
    p.add_argument('--cases',default='fc1,mlp3,mlp_b1,mlp_wide,dma_2d')
    p.add_argument('--repeat',type=int,default=2)
    p.add_argument('--trace',action='store_true')
    a=p.parse_args()
    if a.repeat<1: p.error('--repeat must be positive')
    if sysfs('/sys/class/fpga_manager/fpga0/state')!='operating':
        raise RuntimeError('PL is not operating')
    nodes=[n for n in glob.glob('/sys/class/uio/uio*') if sysfs(n+'/name')=='fabric'
           and int(sysfs(n+'/maps/map0/addr'),0)==0xa0000000]
    if len(nodes)!=1: raise RuntimeError('expected one fabric UIO at 0xa0000000')
    dma_sys='/sys/class/u-dma-buf/udmabuf0/'
    phys=int(sysfs(dma_sys+'phys_addr'),0)
    size=int(sysfs(dma_sys+'size'))
    fd=os.open('/dev/'+os.path.basename(nodes[0]),os.O_RDWR|os.O_SYNC)
    w=Win(mmap.mmap(fd,0x10000,mmap.MAP_SHARED,mmap.PROT_READ|mmap.PROT_WRITE))
    if w.rd(0) not in (VERSION,0x500) or w.rd(8)!=int(a.build_id,0):
        raise RuntimeError(f'identity mismatch: VERSION={w.rd(0):08x} BUILD_ID={w.rd(8):08x}')
    print(f'[PASS] VERSION=0x{w.rd(0):08x} BUILD_ID=0x{w.rd(8):08x}',flush=True)
    w.wr(4,0x5a5aa5a5)
    if w.rd(4)!=0x5a5aa5a5: raise RuntimeError('SCRATCH readback')
    bfd=os.open('/dev/udmabuf0',os.O_RDWR)
    buf=mmap.mmap(bfd,size,mmap.MAP_SHARED,mmap.PROT_READ|mmap.PROT_WRITE)
    active=False
    def sync(device):
        for name,value in [('sync_offset',0),('sync_size',size),('sync_direction',0)]:
            Path(dma_sys+name).write_text(str(value))
        Path(dma_sys+('sync_for_device' if device else 'sync_for_cpu')).write_text('1')
    def host_dma(src,dst,length,check=True):
        if active: raise RuntimeError('Linux DMA attempted during autonomous run')
        if length==0: return
        if a.trace: print(f'[HOST DMA] {src:#x}->{dst:#x} {length} bytes',flush=True)
        w.wr64(IDMA+SRC,src); w.wr64(IDMA+DST,dst); w.wr64(IDMA+LENGTH,length)
        w.wr64(IDMA+0x108,0); w.wr64(IDMA+0x100,0); w.wr64(IDMA+0x110,1); w.wr(IDMA,0)
        tid=w.rd(IDMA+NEXT); deadline=time.monotonic()+5
        while ((w.rd(IDMA+DONE)-tid)&0xffffffff)>=0x80000000:
            if time.monotonic()>deadline: raise RuntimeError('Linux setup/debug DMA timeout')
        if check and w.rd(STATUS)&~1: raise RuntimeError(f'DMA status {w.rd(STATUS):08x}')
    def clear():
        if w.rd(STATUS)&1: raise RuntimeError('refusing to flush while DMA busy (L30)')
        w.wr(CTRL,CLEAR); w.wr(CTRL,0)
        if w.rd(STATUS)!=0 or w.rd(CSTATUS)&0x1d or w.rd(LEVEL)!=0:
            raise RuntimeError(f'unclean state CS={w.rd(CSTATUS):08x} ST={w.rd(STATUS):08x}')

    clear()
    # Protected host writes must still fail; core RHM below must work while owned.
    start=OFF['input']; scratch=OFF['scratch']
    buf[start:start+16]=b'\x5a'*16; sync(True)
    host_dma(phys+start,UB,16)
    w.wr(CTRL,2)
    buf[start:start+16]=b'\xa5'*16; sync(True)
    host_dma(phys+start,UB,16,check=False)
    if w.rd(STATUS)&0x28!=0x28: raise RuntimeError('owned UB did not reject host write')
    host_dma(UB,phys+scratch,16,check=False); sync(False)
    if buf[scratch:scratch+16]!=b'\x5a'*16: raise RuntimeError('rejected host write changed UB')
    clear(); print('[PASS] Linux ownership protection and error recovery',flush=True)

    root=Path(a.cases_dir)
    cases=sorted(n.name for n in root.iterdir() if (n/'meta.txt').exists()) if a.cases=='all' else a.cases.split(',')
    runs=0
    for repetition in range(a.repeat):
        for case in cases:
            clear()
            folder=root/case; meta=load_meta(folder/'meta.txt')
            program=bytearray(load_hex(folder/'prog.hex'))
            inputs=load_hex(folder/'ub_init.hex'); weights=load_hex(folder/'wmem.hex')
            expected=load_hex(folder/'ddr_expect.hex')
            acc_expected=b''.join(int(line,16).to_bytes(64,'little') for line in (folder/'acc_expect.hex').read_text().split())
            if len(program)>2048*16: raise RuntimeError('program exceeds IFIFO capacity')
            for region,data in [('input',inputs),('weights',weights),('program',program),('output',expected),('scratch',acc_expected)]:
                if OFF[region]+len(data)+16>size or (region!='scratch' and len(data)>0x200000-32):
                    raise RuntimeError(f'{region} image exceeds reserved buffer range')
            program=relocate_program(program,json.loads((folder/'relocations.json').read_text()),
                                     phys,size,dict(input=len(inputs),weights=len(weights),output=len(expected)))
            buf[OFF['input']:OFF['input']+len(inputs)]=inputs
            buf[OFF['weights']:OFF['weights']+len(weights)]=weights
            buf[OFF['program']:OFF['program']+len(program)]=program
            buf[OFF['output']-16:OFF['output']+len(expected)+16]=b'\xee'*(len(expected)+32)
            buf[scratch:scratch+len(acc_expected)]=bytes(len(acc_expected)); sync(True)
            host_dma(phys+scratch,ACC,len(acc_expected))
            host_dma(phys+OFF['program'],IFIFO,len(program))
            level=w.rd(LEVEL)
            if level>>16!=len(program)//16: raise RuntimeError('program was not preloaded completely')
            host_id=w.rd(IDMA+DONE)
            w.wr(CTRL,0x206)  # TPU owns UB/ACC; hold counters clear until RUN.
            print(f'[RUN] {case} repeat={repetition+1}: {len(program)//16} instructions preloaded',flush=True)
            active=True; t_run=time.perf_counter(); w.wr(CTRL,0x86)
            deadline=time.monotonic()+30
            while True:
                cs=w.rd(CSTATUS)
                if cs&0x10: raise RuntimeError(f'{case}: DMA fault, ST={w.rd(STATUS):08x}')
                if cs&4 or cs&0xa==0xa: break
                if time.monotonic()>deadline: raise RuntimeError(f'{case}: timeout CS={cs:08x} LEVEL={w.rd(LEVEL):08x}')
            perf=[w.rd(PERF+4*i) for i in range(8)]; t_done=time.perf_counter()  # counters keep running after completion
            active=False; w.wr(CTRL,6)
            if w.rd(IDMA+DONE)!=host_id: raise RuntimeError('Linux DMA ID changed during autonomous execution')
            error=int(meta['cfg_err'])
            if error:
                if cs&4==0 or (cs>>8)&255!=int(meta['cfg_err_op'],16) or cs&8:
                    raise RuntimeError(f'{case}: incorrect rejection CS={cs:08x}')
            else:
                if cs&0x1c!=8 or w.rd(IRQ_TAG)!=int(meta['irq_tag'],16):
                    raise RuntimeError(f'{case}: bad completion CS={cs:08x}')
                sync(False)
                got=buf[OFF['output']:OFF['output']+len(expected)]
                if got!=expected:
                    first=next(i for i,(g,e) in enumerate(zip(got,expected)) if g!=e)
                    raise RuntimeError(f'{case}: DDR output mismatch at byte {first}')
                if buf[OFF['output']-16:OFF['output']]!=b'\xee'*16 or buf[OFF['output']+len(expected):OFF['output']+len(expected)+16]!=b'\xee'*16:
                    raise RuntimeError(f'{case}: output guard overwritten')
                if w.rd(PERF+7*4)!=len(program)//16:
                    raise RuntimeError(f'{case}: retired instruction count mismatch')
                host_dma(ACC,phys+scratch,len(acc_expected)); sync(False)
                if buf[scratch:scratch+len(acc_expected)]!=acc_expected:
                    raise RuntimeError(f'{case}: accumulator image mismatch')
            if w.rd(STATUS)!=0: raise RuntimeError(f'{case}: final STATUS={w.rd(STATUS):08x}')
            runs+=1
            print(f'[PASS] {case}: '+('expected cfg_err' if error else 'DDR/ACC bit-exact, autonomous DMA, IRQ/HALT'),flush=True)
            if not error:
                print('[PERF] %s: cycles=%d array_active=%d w_stall=%d w_shift=%d non_matrix=%d sync=%d act=%d retired=%d wall=%.0fus'%((case,)+tuple(perf)+((t_done-t_run)*1e6,)),flush=True)
            w.wr(CTRL,0)
    print(f'M5_BOARD_PASS runs={runs}',flush=True)
    return 0


if __name__=='__main__':
    try: sys.exit(main())
    except (RuntimeError,OSError,ValueError) as error:
        print(f'M5_BOARD_FAIL: {error}',file=sys.stderr,flush=True)
        sys.exit(1)
