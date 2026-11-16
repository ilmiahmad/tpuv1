/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * TPU-KV260 user-space ABI (/dev/tpu0).  ABI version 1.
 *
 * Model: the process allocates DMA buffers from the driver (ioctl + mmap), fills an input
 * buffer, a weight image ("Weight Memory") and an output buffer, then submits ONE program
 * (<= 2048 16-byte TPU instructions, docs/isa.md) with a relocation list.  The kernel
 * validates the program, patches the DDR address of every DMA instruction from the
 * (buffer handle, offset) pair, preloads the instruction FIFO, runs the TPU to completion
 * (interrupt driven) and returns status + hardware performance counters.  Because there is
 * no IOMMU, EVERY DMA instruction must carry a relocation entry and is bounds-checked
 * against its buffer; a program can never reach memory the caller did not allocate here.
 *
 * Buffers are DMA-coherent (non-cacheable for the CPU): no cache maintenance is needed, and
 * large streaming reads from user space are slower than from normal memory.
 */
#ifndef _UAPI_LINUX_TPU_KV260_H
#define _UAPI_LINUX_TPU_KV260_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define TPU_ABI_VERSION     1
#define TPU_MAX_INSTR       2048    /* instruction FIFO depth (words of 16 bytes) */
#define TPU_MAX_RELOCS      2048
#define TPU_NUM_PERF        8

/* struct tpu_info.state */
#define TPU_STATE_IDLE      0
#define TPU_STATE_BUSY      1
#define TPU_STATE_WEDGED    2       /* a job did not drain: unbind, reload the PL, rebind */

struct tpu_info {
	__u32 abi_version;          /* TPU_ABI_VERSION */
	__u32 hw_version;           /* VERSION register (0x00000400 = Phase B) */
	__u32 build_id;             /* BUILD_ID register (bitstream stamp) */
	__u32 state;                /* TPU_STATE_* */
	__u32 max_instr;            /* TPU_MAX_INSTR */
	__u32 ub_rows;              /* 65536 */
	__u32 acc_rows;             /* 2048 */
	__u32 irq_mode;             /* 1 = interrupt, 0 = polled */
	__u64 max_buf_bytes;        /* per-buffer limit */
	__u64 total_buf_limit;      /* driver-wide limit */
	__u64 total_buf_bytes;      /* currently allocated, driver-wide */
	__u64 jobs_ok;
	__u64 jobs_failed;
	__u64 irq_count;
	__u64 reserved[4];
};

struct tpu_alloc {
	__u64 size;                 /* in: bytes (rounded up to the page size) */
	__u32 flags;                /* in: must be 0 */
	__u32 handle;               /* out: buffer handle (> 0) */
	__u64 mmap_offset;          /* out: pass as the mmap() offset */
	__u64 alloc_size;           /* out: rounded size; mmap() length must be <= this */
};

/* One DDR address inside the program to patch.  The instruction must be READ_HOST_MEM (0x01),
 * READ_WEIGHTS (0x02) or WRITE_HOST_MEM (0x03); offset must be 16-byte aligned. */
struct tpu_reloc {
	__u32 instr_index;          /* index of the 16-byte instruction in the program */
	__u32 handle;               /* buffer handle */
	__u64 offset;               /* byte offset inside the buffer */
};

/* struct tpu_run.status (hardware outcome; the ioctl itself returns 0 when the job ran) */
#define TPU_RUN_OK          0
#define TPU_RUN_CFG_ERR     1       /* illegal instruction (see cfg_err_op) */
#define TPU_RUN_DMA_FAULT   2       /* DMA/bus response error */
#define TPU_RUN_TIMEOUT     3       /* job exceeded timeout_ms (driver drained and recovered) */
#define TPU_RUN_PUSH_FAIL   4       /* host DMA of the program into the instruction FIFO failed */

struct tpu_run {
	/* in */
	__u64 program;              /* user pointer: n_instr * 16 bytes, little endian */
	__u64 relocs;               /* user pointer: n_relocs * struct tpu_reloc */
	__u32 n_instr;              /* 1 .. TPU_MAX_INSTR */
	__u32 n_relocs;             /* 0 .. TPU_MAX_RELOCS */
	__u32 timeout_ms;           /* 0 = default (2000 ms), max 60000 */
	__u32 flags;                /* must be 0 */
	/* out */
	__u32 status;               /* TPU_RUN_* */
	__u32 cfg_err_op;           /* opcode of the offending instruction when CFG_ERR */
	__u32 core_status;          /* CORE_STATUS snapshot at completion */
	__u32 hw_status;            /* STATUS snapshot (sticky error bits) */
	__u32 irq_tag;              /* INTERRUPT_HOST tag */
	__u32 perf[TPU_NUM_PERF];   /* total, array_active, w_stall, w_shift, non_matrix, sync,
				     * act_active, retired (docs/m4_compute_core.md §7) */
	__u64 wall_ns;              /* RUN -> completion, measured in the kernel */
	__u64 reserved[4];
};

#define TPU_IOC_MAGIC       'T'
#define TPU_IOC_GET_INFO    _IOR(TPU_IOC_MAGIC, 0, struct tpu_info)
#define TPU_IOC_ALLOC       _IOWR(TPU_IOC_MAGIC, 1, struct tpu_alloc)
#define TPU_IOC_FREE        _IOW(TPU_IOC_MAGIC, 2, __u32)
#define TPU_IOC_RUN         _IOWR(TPU_IOC_MAGIC, 3, struct tpu_run)
#define TPU_IOC_RECOVER     _IO(TPU_IOC_MAGIC, 4)   /* drain, clear errors, flush FIFOs */

/*
 * errno summary: EINVAL (bad argument / program or relocation rejected), EFAULT, ENOMEM,
 * ENOSPC (buffer limit), ENOENT (unknown handle), EINTR (interrupted while waiting; the job
 * is recovered first), EIO (device wedged), ENODEV.
 */
#endif /* _UAPI_LINUX_TPU_KV260_H */
