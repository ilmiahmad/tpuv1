#!/usr/bin/env python3
"""TPU-KV260 ISA v1.0 assembler / disassembler (docs/isa.md).

An instruction is a 128-bit integer; byte 0 (bits [7:0]) is the opcode. On the bus and in files
it is 16 bytes little-endian. Hex files hold one instruction per line, 32 hex digits MSB first
($readmemh of a [127:0] word gives exactly the instruction bits).
"""

NOP, READ_HOST_MEM, READ_WEIGHTS, WRITE_HOST_MEM = 0x00, 0x01, 0x02, 0x03
MATMUL, CONVOLVE, ACTIVATE = 0x10, 0x11, 0x20
SET_CONFIG, SYNC, INTERRUPT_HOST, DEBUG_TAG, HALT = 0x30, 0x31, 0x32, 0x3E, 0x3F

NAMES = {NOP: "NOP", READ_HOST_MEM: "READ_HOST_MEM", READ_WEIGHTS: "READ_WEIGHTS",
         WRITE_HOST_MEM: "WRITE_HOST_MEM", MATMUL: "MATMUL", CONVOLVE: "CONVOLVE",
         ACTIVATE: "ACTIVATE", SET_CONFIG: "SET_CONFIG", SYNC: "SYNC",
         INTERRUPT_HOST: "INTERRUPT_HOST", DEBUG_TAG: "DEBUG_TAG", HALT: "HALT"}

# SYNC unit mask bits
U_DMA, U_WEIGHT, U_MATRIX, U_ACT = 1, 2, 4, 8
U_ALL = 0xF

# field layout per opcode: name -> (lsb, width). Everything not listed is reserved (= 0).
FIELDS = {
    NOP: {},
    READ_HOST_MEM:  {"ub_row": (16, 16), "ddr_addr4": (32, 45), "row_len": (77, 12),
                     "reps": (89, 12), "ddr_stride": (101, 24)},
    WRITE_HOST_MEM: {"ub_row": (16, 16), "ddr_addr4": (32, 45), "row_len": (77, 12),
                     "reps": (89, 12), "ddr_stride": (101, 24)},
    READ_WEIGHTS:   {"tiles": (16, 16), "ddr_addr4": (32, 45)},
    MATMUL:   {"acc": (8, 1), "new_weights": (9, 1), "ub_row": (16, 16), "acc_row": (32, 11), "B": (48, 16)},
    # docs/m6_cnn_contract.md section 2 (ADR-0004): one kernel tap x one input plane x an output-row window
    CONVOLVE: {"acc": (8, 1), "new_weights": (9, 1), "ub_row": (16, 16), "acc_row": (32, 11),
               "kh": (48, 4), "kw": (52, 4), "oh0": (64, 12), "oh_cnt": (76, 12)},
    ACTIVATE: {"func": (8, 4), "out16": (12, 1), "pool": (13, 3), "acc_row": (16, 11),
               "ub_row": (32, 16), "rows": (48, 16), "qset": (64, 4), "eltwise": (68, 4),
               "ub_operand_row": (72, 16)},
    SET_CONFIG: {"target": (16, 16), "value": (32, 64)},
    SYNC: {"mask": (8, 4)},
    INTERRUPT_HOST: {"tag": (32, 32)},
    DEBUG_TAG: {"tag": (32, 32)},
    HALT: {},
}


def field_mask(op):
    m = 0xFF
    for lsb, w in FIELDS.get(op, {}).values():
        m |= ((1 << w) - 1) << lsb
    return m


def encode(op, **kw):
    if op not in FIELDS:
        raise ValueError(f"unknown opcode {op:#x}")
    w = op
    for k, v in kw.items():
        lsb, width = FIELDS[op][k]
        if v < 0 or v >= (1 << width):
            raise ValueError(f"{NAMES[op]}.{k}={v} does not fit {width} bits")
        w |= v << lsb
    return w


def encode_raw(word):
    """Escape hatch for negative tests: any 128-bit pattern."""
    return word & ((1 << 128) - 1)


def decode(word):
    """-> (opcode, dict of fields, reserved_bits_set: bool, known: bool)"""
    op = word & 0xFF
    if op not in FIELDS:
        return op, {}, False, False
    f = {k: (word >> lsb) & ((1 << w) - 1) for k, (lsb, w) in FIELDS[op].items()}
    reserved = (word & ~field_mask(op) & ((1 << 128) - 1)) != 0
    return op, f, reserved, True


def disasm(word):
    op, f, rsv, known = decode(word)
    if not known:
        return f"UNKNOWN(op={op:#04x}) {word:032x}"
    args = " ".join(f"{k}={v:#x}" if k in ("value", "tag", "target", "ddr_addr4") else f"{k}={v}"
                    for k, v in f.items())
    return f"{NAMES[op]} {args}" + ("  ; RESERVED BITS SET" if rsv else "")


def to_bytes(word):
    return word.to_bytes(16, "little")


def write_hex(path, words):
    with open(path, "w") as fh:
        for w in words:
            fh.write(f"{w:032x}\n")


def write_bin(path, words):
    with open(path, "wb") as fh:
        for w in words:
            fh.write(to_bytes(w))


def read_hex(path):
    with open(path) as fh:
        return [int(l, 16) for l in fh if l.strip()]


# --- convenience constructors ------------------------------------------------------------------
def matmul(ub_row, acc_row, B, acc=0, new_weights=0):
    return encode(MATMUL, ub_row=ub_row, acc_row=acc_row, B=B, acc=acc, new_weights=new_weights)


def activate(acc_row, ub_row, rows, qset, func=0, pool=0):
    return encode(ACTIVATE, acc_row=acc_row, ub_row=ub_row, rows=rows, qset=qset, func=func, pool=pool)


def convolve(ub_row, acc_row, kh, kw, oh0, oh_cnt, acc=0, new_weights=0):
    return encode(CONVOLVE, ub_row=ub_row, acc_row=acc_row, kh=kh, kw=kw, oh0=oh0, oh_cnt=oh_cnt,
                  acc=acc, new_weights=new_weights)


# SET_CONFIG kind 4 (conv / pool geometry, contract section 3)
CONV_IN, CONV_OUT, CONV_MISC, POOL_W = 0x4000, 0x4001, 0x4002, 0x4003


def set_conv_in(in_w, in_h):
    return set_config(CONV_IN, in_w | (in_h << 16))


def set_conv_out(out_w, out_h):
    return set_config(CONV_OUT, out_w | (out_h << 16))


def set_conv_misc(sh, sw, pad_t, pad_l, pad_val):
    return set_config(CONV_MISC, sh | (sw << 4) | (pad_t << 8) | (pad_l << 16) | ((pad_val & 0xFF) << 24))


def set_pool_w(pool_w):
    return set_config(POOL_W, pool_w)


def set_config(target, value):
    return encode(SET_CONFIG, target=target, value=value & ((1 << 64) - 1))


def sync(mask=U_ALL):
    return encode(SYNC, mask=mask)


def interrupt_host(tag):
    return encode(INTERRUPT_HOST, tag=tag)


def halt():
    return encode(HALT)


def _aligned16(address):
    if address < 0 or address >= 1 << 49 or address & 15:
        raise ValueError(f"address/stride {address:#x} must fit 49 bits and be 16-byte aligned")
    return address >> 4


def read_host_mem(ub_row, ddr_addr, row_len, reps=1, ddr_stride=None):
    return encode(READ_HOST_MEM, ub_row=ub_row, ddr_addr4=_aligned16(ddr_addr),
                  row_len=row_len, reps=reps,
                  ddr_stride=_aligned16(row_len * 16 if ddr_stride is None else ddr_stride))


def write_host_mem(ub_row, ddr_addr, row_len, reps=1, ddr_stride=None):
    return encode(WRITE_HOST_MEM, ub_row=ub_row, ddr_addr4=_aligned16(ddr_addr),
                  row_len=row_len, reps=reps,
                  ddr_stride=_aligned16(row_len * 16 if ddr_stride is None else ddr_stride))


def read_weights(ddr_addr, tiles):
    return encode(READ_WEIGHTS, ddr_addr4=_aligned16(ddr_addr), tiles=tiles)


# --- self-test -------------------------------------------------------------------------------
def _selftest():
    import random
    rnd = random.Random(3)
    for op, flds in FIELDS.items():
        for _ in range(200):
            kw = {k: rnd.randrange(1 << w) for k, (_, w) in flds.items()}
            w = encode(op, **kw)
            op2, f2, rsv, known = decode(w)
            assert known and op2 == op and not rsv and f2 == kw, (op, kw, f2)
            assert int.from_bytes(to_bytes(w), "little") == w
            assert to_bytes(w)[0] == op
    # reserved-bit detection
    assert decode(matmul(1, 2, 3) | (1 << 64))[2]
    assert decode(matmul(1, 2, 3) | (1 << 10))[2]
    assert decode(matmul(1, 2, 3) | (1 << 43))[2]          # gap [47:43]
    assert decode(activate(1, 2, 3, 0) | (1 << 27))[2]     # gap [31:27]
    assert decode(halt() | (1 << 8))[2]
    assert not decode(encode_raw(0x05))[3]
    # CONVOLVE field positions and reserved gaps (contract section 2)
    c = convolve(0xABCD, 0x7FF, 15, 9, 0xFFF, 0x123, 1, 1)
    assert c == (0x123 << 76) | (0xFFF << 64) | (9 << 52) | (15 << 48) | (0x7FF << 32) | (0xABCD << 16) | (3 << 8) | 0x11
    for bit in (10, 15, 43, 47, 56, 63, 88, 127):
        assert decode(c | (1 << bit))[2], bit
    assert not decode(c)[2]
    # ACTIVATE pool field
    assert activate(1, 2, 8, 0, pool=2) >> 13 & 7 == 2
    # kind-4 SET_CONFIG constructors
    assert set_conv_misc(2, 3, 1, 4, -128) == (((0x80 << 24) | (4 << 16) | (1 << 8) | (3 << 4) | 2) << 32) | (0x4002 << 16) | 0x30
    # field positions against isa.md tables (spot checks)
    assert matmul(0xABCD, 0x7FF, 0x1234, 1, 1) == (0x1234 << 48) | (0x7FF << 32) | (0xABCD << 16) | (3 << 8) | 0x10
    assert set_config(0x3002, 0x11223344) == (0x11223344 << 32) | (0x3002 << 16) | 0x30
    assert encode(READ_HOST_MEM, ddr_stride=1) == (1 << 101) | 0x01
    print("asm self-test: PASS")


if __name__ == "__main__":
    _selftest()
