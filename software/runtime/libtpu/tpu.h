/* SPDX-License-Identifier: MIT */
/*
 * libtpu: thin C wrapper over the TPU-KV260 Linux driver (/dev/tpu0, ABI v1).
 *
 * Conventions: functions return 0 on success and -1 with errno set on failure (tpu_open returns
 * NULL with errno set).  tpu_run() returns 0 whenever the job reached the hardware and the driver
 * recovered; the hardware outcome is in result->status (TPU_RUN_*).  EINTR from tpu_run() means a
 * signal interrupted the wait: the driver has already recovered the job, simply retry.
 * See software/driver/include/uapi/tpu_kv260.h for the ABI and docs/driver_user_guide.md.
 */
#ifndef LIBTPU_H
#define LIBTPU_H

#include <stddef.h>
#include <stdint.h>

#include "tpu_kv260.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tpu_dev tpu_dev_t;

struct tpu_buf {
	uint32_t handle;      /* driver handle (use in struct tpu_reloc) */
	void *ptr;            /* mmap'ed, zero filled by the kernel, DMA coherent */
	size_t size;          /* requested size */
	size_t alloc_size;    /* rounded size (mapped length) */
};

/* Open /dev/tpu0 (path == NULL) or another node. */
tpu_dev_t *tpu_open(const char *path);
void tpu_close(tpu_dev_t *dev);                 /* frees every buffer still allocated */

int tpu_get_info(tpu_dev_t *dev, struct tpu_info *info);
int tpu_alloc(tpu_dev_t *dev, size_t size, struct tpu_buf *buf);
int tpu_free(tpu_dev_t *dev, struct tpu_buf *buf);

/* program: n_instr * 16 bytes, little endian. relocs may be NULL when n_relocs == 0. */
int tpu_run(tpu_dev_t *dev, const void *program, unsigned n_instr,
	    const struct tpu_reloc *relocs, unsigned n_relocs, unsigned timeout_ms,
	    struct tpu_run *result);
int tpu_recover(tpu_dev_t *dev);

const char *tpu_status_str(uint32_t status);    /* TPU_RUN_* */
const char *tpu_state_str(uint32_t state);      /* TPU_STATE_* */
extern const char *const tpu_perf_names[TPU_NUM_PERF];

#ifdef __cplusplus
}
#endif
#endif /* LIBTPU_H */
