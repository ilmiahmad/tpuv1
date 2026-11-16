#!/usr/bin/env python3
"""Minimal TPU instruction encoders (docs/isa.md) for user-space tools and the board tests.

Self-contained on purpose (the board does not have the model/ tree).  tests/test_host.py checks
every encoder against model/isa/asm.py.  Words are 128-bit ints; `to_bytes` gives the 16-byte
little-endian form the driver expects.
"""

OP_NOP, OP_RHM, OP_RW, OP_WHM = 0x00, 0x01, 0x02, 0x03
OP_MATMUL, OP_ACTIVATE = 0x10, 0x20
OP_SET_CONFIG, OP_SYNC, OP_IRQ, OP_HALT = 0x30, 0x31, 0x32, 0x3F
U_DMA, U_WEIGHT, U_MATRIX, U_ACT, U_ALL = 1, 2, 4, 8, 0xF


def _a16(address):
    if address < 0 or address >= 1 << 49 or address & 15:
        raise ValueError(f"address/stride {address:#x} must fit 49 bits and be 16-byte aligned")
    return address >> 4


def nop():
    return OP_NOP


def halt():
    return OP_HALT


def sync(mask=U_ALL):
    return OP_SYNC | (mask << 8)


def interrupt_host(tag):
    return OP_IRQ | (tag << 32)


def matmul(ub_row, acc_row, rows, acc=0, new_weights=0):
    return OP_MATMUL | (acc << 8) | (new_weights << 9) | (ub_row << 16) | (acc_row << 32) | (rows << 48)


def activate(acc_row, ub_row, rows, qset, func=0):
    return OP_ACTIVATE | (func << 8) | (acc_row << 16) | (ub_row << 32) | (rows << 48) | (qset << 64)


def set_config(target, value):
    return OP_SET_CONFIG | (target << 16) | ((value & ((1 << 64) - 1)) << 32)


def _dma(op, ub_row, ddr_addr, row_len, reps, ddr_stride):
    stride = row_len * 16 if ddr_stride is None else ddr_stride
    return op | (ub_row << 16) | (_a16(ddr_addr) << 32) | (row_len << 77) | (reps << 89) | (_a16(stride) << 101)


def read_host_mem(ub_row, ddr_addr, row_len, reps=1, ddr_stride=None):
    return _dma(OP_RHM, ub_row, ddr_addr, row_len, reps, ddr_stride)


def write_host_mem(ub_row, ddr_addr, row_len, reps=1, ddr_stride=None):
    return _dma(OP_WHM, ub_row, ddr_addr, row_len, reps, ddr_stride)


def read_weights(ddr_addr, tiles):
    return OP_RW | (tiles << 16) | (_a16(ddr_addr) << 32)


def to_bytes(words):
    return b"".join((w & ((1 << 128) - 1)).to_bytes(16, "little") for w in words)
