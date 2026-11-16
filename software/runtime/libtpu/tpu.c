// SPDX-License-Identifier: MIT
#include "tpu.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

struct tpu_dev {
	int fd;
};

const char *const tpu_perf_names[TPU_NUM_PERF] = {
	"total", "array_active", "w_stall", "w_shift", "non_matrix", "sync", "act", "retired",
};

/* ioctl that is restarted on EINTR (everything except RUN, whose EINTR carries meaning). */
static int ioctl_retry(int fd, unsigned long req, void *arg)
{
	int rc;

	do {
		rc = ioctl(fd, req, arg);
	} while (rc < 0 && errno == EINTR);
	return rc;
}

tpu_dev_t *tpu_open(const char *path)
{
	tpu_dev_t *dev = calloc(1, sizeof(*dev));

	if (!dev)
		return NULL;
	dev->fd = open(path ? path : "/dev/tpu0", O_RDWR | O_CLOEXEC);
	if (dev->fd < 0) {
		int e = errno;

		free(dev);
		errno = e;
		return NULL;
	}
	return dev;
}

void tpu_close(tpu_dev_t *dev)
{
	if (!dev)
		return;
	close(dev->fd);               /* the kernel frees this descriptor's buffers */
	free(dev);
}

int tpu_get_info(tpu_dev_t *dev, struct tpu_info *info)
{
	if (!dev || !info) {
		errno = EINVAL;
		return -1;
	}
	memset(info, 0, sizeof(*info));
	return ioctl_retry(dev->fd, TPU_IOC_GET_INFO, info);
}

int tpu_alloc(tpu_dev_t *dev, size_t size, struct tpu_buf *buf)
{
	struct tpu_alloc a;
	void *p;

	if (!dev || !buf || size == 0) {
		errno = EINVAL;
		return -1;
	}
	memset(&a, 0, sizeof(a));
	a.size = size;
	if (ioctl_retry(dev->fd, TPU_IOC_ALLOC, &a) < 0)
		return -1;
	p = mmap(NULL, a.alloc_size, PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd, (off_t)a.mmap_offset);
	if (p == MAP_FAILED) {
		int e = errno;
		__u32 h = a.handle;

		ioctl_retry(dev->fd, TPU_IOC_FREE, &h);
		errno = e;
		return -1;
	}
	buf->handle = a.handle;
	buf->ptr = p;
	buf->size = size;
	buf->alloc_size = a.alloc_size;
	return 0;
}

int tpu_free(tpu_dev_t *dev, struct tpu_buf *buf)
{
	__u32 h;

	if (!dev || !buf || !buf->ptr) {
		errno = EINVAL;
		return -1;
	}
	munmap(buf->ptr, buf->alloc_size);
	h = buf->handle;
	buf->ptr = NULL;
	buf->handle = 0;
	return ioctl_retry(dev->fd, TPU_IOC_FREE, &h);
}

int tpu_run(tpu_dev_t *dev, const void *program, unsigned n_instr,
	    const struct tpu_reloc *relocs, unsigned n_relocs, unsigned timeout_ms,
	    struct tpu_run *result)
{
	struct tpu_run r;

	if (!dev || !program || !result || (n_relocs && !relocs)) {
		errno = EINVAL;
		return -1;
	}
	memset(&r, 0, sizeof(r));
	r.program = (__u64)(uintptr_t)program;
	r.relocs = (__u64)(uintptr_t)relocs;
	r.n_instr = n_instr;
	r.n_relocs = n_relocs;
	r.timeout_ms = timeout_ms;
	if (ioctl(dev->fd, TPU_IOC_RUN, &r) < 0)     /* EINTR is reported to the caller */
		return -1;
	*result = r;
	return 0;
}

int tpu_recover(tpu_dev_t *dev)
{
	if (!dev) {
		errno = EINVAL;
		return -1;
	}
	return ioctl_retry(dev->fd, TPU_IOC_RECOVER, NULL);
}

const char *tpu_status_str(uint32_t status)
{
	switch (status) {
	case TPU_RUN_OK: return "OK";
	case TPU_RUN_CFG_ERR: return "CFG_ERR";
	case TPU_RUN_DMA_FAULT: return "DMA_FAULT";
	case TPU_RUN_TIMEOUT: return "TIMEOUT";
	case TPU_RUN_PUSH_FAIL: return "PUSH_FAIL";
	default: return "UNKNOWN";
	}
}

const char *tpu_state_str(uint32_t state)
{
	switch (state) {
	case TPU_STATE_IDLE: return "idle";
	case TPU_STATE_BUSY: return "busy";
	case TPU_STATE_WEDGED: return "wedged";
	default: return "unknown";
	}
}
