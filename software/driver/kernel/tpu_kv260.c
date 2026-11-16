// SPDX-License-Identifier: GPL-2.0
/*
 * TPU-KV260 platform driver (/dev/tpu0).  See docs/driver.md and uapi/tpu_kv260.h.
 *
 * Binds to the base-DT "a0000000.fabric" node (reg 0xA0000000/64 KiB, SPI 89) through
 * driver_override (software/driver/tpu-load.sh), replacing uio_pdrv_genirq.  The only
 * hardware blocks it programs are tpu_regs (+0x000) and the PULP iDMA register frontend
 * (+0x1000, used to preload the instruction FIFO).
 *
 * Safety model (no IOMMU on the PL master):
 *  - every DMA instruction of a submitted program needs a relocation entry and is bounds-
 *    checked against a buffer allocated through this driver; the kernel writes the address;
 *  - weight tiles may never be consumed before they were requested (static check), because a
 *    core stuck waiting for weights cannot be recovered without reloading the PL;
 *  - the PL is never reset while AXI traffic is in flight (lessons L5/L30); a job that does
 *    not drain marks the device WEDGED.
 */
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/io.h>
#include <linux/interrupt.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/dma-mapping.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/spinlock.h>
#include <linux/completion.h>
#include <linux/ktime.h>
#include <linux/delay.h>
#include <linux/idr.h>
#include <linux/kref.h>
#include <linux/bitmap.h>
#include <asm/unaligned.h>

#include "tpu_kv260.h"

#define DRV_NAME            "tpu_kv260"

/* tpu_regs (kv260_top.sv, "tpu_regs v0.3/0.4") */
#define R_VERSION           0x00
#define R_BUILD_ID          0x08
#define R_STATUS            0x10
#define R_CTRL              0x14
#define R_LEVEL             0x18
#define R_CSTATUS           0x1c
#define R_IRQ_TAG           0x20
#define R_PERF(i)           (0x24 + 4 * (i))

#define CTRL_ERRCLR         BIT(0)
#define CTRL_UB_OWN         BIT(1)
#define CTRL_ACC_OWN        BIT(2)
#define CTRL_WF_FLUSH       BIT(5)
#define CTRL_IF_FLUSH       BIT(6)
#define CTRL_RUN            BIT(7)
#define CTRL_IRQ_CLR        BIT(8)
#define CTRL_PERF_CLR       BIT(9)

#define CS_BUSY             BIT(0)
#define CS_HALTED           BIT(1)
#define CS_CFG_ERR          BIT(2)
#define CS_IRQ              BIT(3)
#define CS_DMA_ERR          BIT(4)
#define CS_ERR_OP(cs)       (((cs) >> 8) & 0xff)

#define ST_ERR_MASK         0x1feu          /* STATUS[8:1] sticky error bits */

/* iDMA reg64_2d frontend at +0x1000 */
#define IDMA                0x1000
#define D_CONF              (IDMA + 0x000)
#define D_NEXT_ID           (IDMA + 0x044)
#define D_DONE_ID           (IDMA + 0x084)
#define D_DST               (IDMA + 0x0d0)
#define D_SRC               (IDMA + 0x0d8)
#define D_LEN               (IDMA + 0x0e0)
#define D_DSTRIDE           (IDMA + 0x100)
#define D_SSTRIDE           (IDMA + 0x108)
#define D_REPS              (IDMA + 0x110)

#define IFIFO_PUSH_ADDR     0xa0400000ull   /* docs/interfaces.md */

#define OP_RHM              0x01
#define OP_RW               0x02
#define OP_WHM              0x03
#define OP_MATMUL           0x10
#define OP_CONVOLVE         0x11

#define STAGE_BYTES         (TPU_MAX_INSTR * 16)
#define DEFAULT_TIMEOUT_MS  2000
#define MAX_TIMEOUT_MS      60000

static bool use_irq = true;
module_param(use_irq, bool, 0444);
MODULE_PARM_DESC(use_irq, "Interrupt driven completion (default) or polling");

static unsigned long max_buf_mb = 64;
module_param(max_buf_mb, ulong, 0444);
MODULE_PARM_DESC(max_buf_mb, "Largest single DMA buffer in MiB");

static unsigned long total_buf_mb = 256;
module_param(total_buf_mb, ulong, 0444);
MODULE_PARM_DESC(total_buf_mb, "Driver-wide DMA buffer limit in MiB");

struct tpu_dev {
	struct device       *dev;
	void __iomem        *regs;
	int                  irq;
	bool                 irq_on;
	struct miscdevice    misc;

	struct mutex         job_lock;      /* one job at a time; state changes */
	spinlock_t           ctrl_lock;     /* CTRL shadow (also taken in the IRQ handler) */
	u32                  ctrl;
	int                  state;         /* TPU_STATE_* */

	struct completion    done;
	bool                 job_running;   /* IRQ handler may complete() */
	bool                 snap_valid;
	u32                  snap_cs, snap_tag, snap_perf[TPU_NUM_PERF];
	ktime_t              snap_time;

	void                *stage;         /* program staging buffer (DMA-coherent) */
	dma_addr_t           stage_dma;

	atomic64_t           total_bytes;
	atomic64_t           jobs_ok, jobs_failed, irq_count, irq_missed;
	u32                  last_perf[TPU_NUM_PERF];
	u32                  hw_version, build_id;
};

struct tpu_file {
	struct tpu_dev      *t;
	struct mutex         lock;          /* protects bufs */
	struct idr           bufs;
};

struct tpu_buf {
	struct kref          ref;
	struct tpu_dev      *t;
	u32                  handle;
	size_t               size;
	void                *cpu;
	dma_addr_t           dma;
};

/* ------------------------------------------------------------------------------------------ */
/* register helpers                                                                          */
/* ------------------------------------------------------------------------------------------ */
static inline u32 rd(struct tpu_dev *t, u32 off)
{
	return readl(t->regs + off);
}

static inline void wr(struct tpu_dev *t, u32 off, u32 v)
{
	writel(v, t->regs + off);
}

static void ctrl_set(struct tpu_dev *t, u32 v)
{
	unsigned long fl;

	spin_lock_irqsave(&t->ctrl_lock, fl);
	t->ctrl = v;
	wr(t, R_CTRL, v);
	spin_unlock_irqrestore(&t->ctrl_lock, fl);
}

/* Level-sensitive control bits: set, make sure the write reached the PL (a read is not
 * posted), then restore the shadow value. */
static void ctrl_pulse(struct tpu_dev *t, u32 bits)
{
	unsigned long fl;

	spin_lock_irqsave(&t->ctrl_lock, fl);
	wr(t, R_CTRL, t->ctrl | bits);
	(void)rd(t, R_VERSION);
	wr(t, R_CTRL, t->ctrl);
	(void)rd(t, R_VERSION);
	spin_unlock_irqrestore(&t->ctrl_lock, fl);
}

static int wait_idle(struct tpu_dev *t, unsigned int timeout_ms)
{
	unsigned long end = jiffies + msecs_to_jiffies(timeout_ms);

	while (rd(t, R_CSTATUS) & CS_BUSY) {
		if (time_after(jiffies, end))
			return -ETIMEDOUT;
		usleep_range(50, 100);
	}
	return 0;
}

/* Drain after a stop request: RUN low, owners kept, wait for busy to clear, clear sticky
 * errors / FIFOs / IRQ / counters, owners off.  Never resets the PL. */
static int tpu_drain_locked(struct tpu_dev *t)
{
	int ret;

	ctrl_set(t, CTRL_UB_OWN | CTRL_ACC_OWN);            /* run off */
	ret = wait_idle(t, 500);
	if (ret) {
		t->state = TPU_STATE_WEDGED;
		dev_err(t->dev, "core did not drain (CORE_STATUS=%08x): device wedged\n",
			rd(t, R_CSTATUS));
		return -EIO;
	}
	ctrl_pulse(t, CTRL_ERRCLR | CTRL_WF_FLUSH | CTRL_IF_FLUSH | CTRL_IRQ_CLR | CTRL_PERF_CLR);
	ctrl_set(t, 0);
	if ((rd(t, R_STATUS) & ST_ERR_MASK) || (rd(t, R_CSTATUS) & (CS_BUSY | CS_CFG_ERR | CS_DMA_ERR))) {
		t->state = TPU_STATE_WEDGED;
		dev_err(t->dev, "errors persist after recovery (STATUS=%08x CORE_STATUS=%08x)\n",
			rd(t, R_STATUS), rd(t, R_CSTATUS));
		return -EIO;
	}
	t->state = TPU_STATE_IDLE;
	return 0;
}

/* ------------------------------------------------------------------------------------------ */
/* interrupt                                                                                 */
/* ------------------------------------------------------------------------------------------ */
static irqreturn_t tpu_irq(int irq, void *data)
{
	struct tpu_dev *t = data;
	unsigned long fl;
	u32 cs = rd(t, R_CSTATUS);
	int i;

	if (!(cs & CS_IRQ))
		return IRQ_NONE;

	/* latch the counters now: they keep running after the interrupt */
	t->snap_cs = cs;
	t->snap_tag = rd(t, R_IRQ_TAG);
	for (i = 0; i < TPU_NUM_PERF; i++)
		t->snap_perf[i] = rd(t, R_PERF(i));
	t->snap_time = ktime_get();

	/* clear the sticky level: CTRL[8] high, flush the write, restore */
	spin_lock_irqsave(&t->ctrl_lock, fl);
	wr(t, R_CTRL, t->ctrl | CTRL_IRQ_CLR);
	(void)rd(t, R_VERSION);
	wr(t, R_CTRL, t->ctrl);
	(void)rd(t, R_CSTATUS);
	spin_unlock_irqrestore(&t->ctrl_lock, fl);

	atomic64_inc(&t->irq_count);
	if (t->job_running) {
		t->snap_valid = true;
		complete(&t->done);
	}
	return IRQ_HANDLED;
}

/* ------------------------------------------------------------------------------------------ */
/* buffers                                                                                   */
/* ------------------------------------------------------------------------------------------ */
static void tpu_buf_release(struct kref *ref)
{
	struct tpu_buf *b = container_of(ref, struct tpu_buf, ref);

	dma_free_coherent(b->t->dev, b->size, b->cpu, b->dma);
	atomic64_sub(b->size, &b->t->total_bytes);
	kfree(b);
}

static void tpu_buf_put(struct tpu_buf *b)
{
	kref_put(&b->ref, tpu_buf_release);
}

static void tpu_vma_open(struct vm_area_struct *vma)
{
	struct tpu_buf *b = vma->vm_private_data;

	kref_get(&b->ref);
}

static void tpu_vma_close(struct vm_area_struct *vma)
{
	tpu_buf_put(vma->vm_private_data);
}

static const struct vm_operations_struct tpu_vm_ops = {
	.open  = tpu_vma_open,
	.close = tpu_vma_close,
};

static int tpu_ioctl_alloc(struct tpu_file *f, struct tpu_alloc __user *uarg)
{
	struct tpu_dev *t = f->t;
	struct tpu_alloc a;
	struct tpu_buf *b;
	size_t size;
	int id, ret;

	if (copy_from_user(&a, uarg, sizeof(a)))
		return -EFAULT;
	if (a.flags || !a.size)
		return -EINVAL;
	/* both limits are "buffer limit" errors (ABI: ENOSPC) */
	if (a.size > (max_buf_mb << 20))
		return -ENOSPC;
	size = PAGE_ALIGN(a.size);
	if (atomic64_read(&t->total_bytes) + size > (total_buf_mb << 20))
		return -ENOSPC;

	b = kzalloc(sizeof(*b), GFP_KERNEL);
	if (!b)
		return -ENOMEM;
	kref_init(&b->ref);
	b->t = t;
	b->size = size;
	b->cpu = dma_alloc_coherent(t->dev, size, &b->dma, GFP_KERNEL | __GFP_NOWARN);
	if (!b->cpu) {
		kfree(b);
		return -ENOMEM;
	}
	atomic64_add(size, &t->total_bytes);

	mutex_lock(&f->lock);
	id = idr_alloc(&f->bufs, b, 1, 0, GFP_KERNEL);
	mutex_unlock(&f->lock);
	if (id < 0) {
		tpu_buf_put(b);
		return id;
	}
	b->handle = id;

	a.handle = id;
	a.mmap_offset = (u64)id << PAGE_SHIFT;
	a.alloc_size = size;
	ret = copy_to_user(uarg, &a, sizeof(a)) ? -EFAULT : 0;
	if (ret) {
		mutex_lock(&f->lock);
		idr_remove(&f->bufs, id);
		mutex_unlock(&f->lock);
		tpu_buf_put(b);
	}
	return ret;
}

static int tpu_ioctl_free(struct tpu_file *f, u32 __user *uarg)
{
	struct tpu_buf *b;
	u32 handle;

	if (get_user(handle, uarg))
		return -EFAULT;
	mutex_lock(&f->lock);
	b = idr_remove(&f->bufs, handle);
	mutex_unlock(&f->lock);
	if (!b)
		return -ENOENT;
	tpu_buf_put(b);             /* memory lives on while mapped or in use by a job */
	return 0;
}

static int tpu_mmap(struct file *file, struct vm_area_struct *vma)
{
	struct tpu_file *f = file->private_data;
	struct tpu_buf *b;
	unsigned long len = vma->vm_end - vma->vm_start;
	int ret;

	if (!(vma->vm_flags & VM_SHARED))
		return -EINVAL;
	mutex_lock(&f->lock);
	b = idr_find(&f->bufs, vma->vm_pgoff);
	if (b)
		kref_get(&b->ref);
	mutex_unlock(&f->lock);
	if (!b)
		return -ENOENT;
	if (len > b->size) {
		tpu_buf_put(b);
		return -EINVAL;
	}
	vma->vm_pgoff = 0;
	ret = dma_mmap_coherent(f->t->dev, vma, b->cpu, b->dma, b->size);
	if (ret) {
		tpu_buf_put(b);
		return ret;
	}
	vma->vm_private_data = b;   /* reference owned by the mapping */
	vma->vm_ops = &tpu_vm_ops;
	return 0;
}

/* ------------------------------------------------------------------------------------------ */
/* program validation and relocation                                                         */
/* ------------------------------------------------------------------------------------------ */
static u64 get_field(const u64 w[2], unsigned int pos, unsigned int width)
{
	u64 v = 0;
	unsigned int i;

	for (i = 0; i < width; i++) {
		unsigned int b = pos + i;

		v |= ((w[b >> 6] >> (b & 63)) & 1ull) << i;
	}
	return v;
}

static void set_field(u64 w[2], unsigned int pos, unsigned int width, u64 val)
{
	unsigned int i;

	for (i = 0; i < width; i++) {
		unsigned int b = pos + i;
		u64 m = 1ull << (b & 63);

		w[b >> 6] = (w[b >> 6] & ~m) | (((val >> i) & 1ull) ? m : 0);
	}
}

/* Bytes touched in DDR by one DMA instruction (docs/isa.md). */
static u64 dma_span(const u64 w[2], u8 op)
{
	if (op == OP_RW)
		return get_field(w, 16, 16) * 256ull;
	{
		u64 reps = get_field(w, 89, 12);
		u64 len  = get_field(w, 77, 12) * 16ull;
		u64 strd = get_field(w, 101, 24) * 16ull;

		if (!reps || !len)
			return 0;       /* the decoder rejects it (cfg_err); no DMA is issued */
		return (reps - 1) * strd + len;
	}
}

/*
 * Validate @prog (n instructions), apply the relocations in place and take a reference on
 * every buffer used (returned in @refs, @nrefs entries).  Returns 0 or -EINVAL / -ENOENT.
 */
static int tpu_validate(struct tpu_file *f, u8 *prog, unsigned int n,
			const struct tpu_reloc *rel, unsigned int nrel,
			struct tpu_buf **refs)
{
	DECLARE_BITMAP(seen, TPU_MAX_INSTR);
	u64 pushed = 0, consumed = 0;
	unsigned int i, taken = 0;
	int ret = 0;

	bitmap_zero(seen, TPU_MAX_INSTR);

	for (i = 0; i < nrel; i++) {
		const struct tpu_reloc *r = &rel[i];
		struct tpu_buf *b;
		u64 w[2], span, addr;
		u8 op;

		if (r->instr_index >= n || test_and_set_bit(r->instr_index, seen)) {
			ret = -EINVAL;
			goto out;
		}
		w[0] = get_unaligned_le64(prog + r->instr_index * 16);
		w[1] = get_unaligned_le64(prog + r->instr_index * 16 + 8);
		op = w[0] & 0xff;
		if (op != OP_RHM && op != OP_RW && op != OP_WHM) {
			ret = -EINVAL;
			goto out;
		}
		mutex_lock(&f->lock);
		b = idr_find(&f->bufs, r->handle);
		if (b)
			kref_get(&b->ref);
		mutex_unlock(&f->lock);
		if (!b) {
			ret = -ENOENT;
			goto out;
		}
		refs[taken++] = b;

		span = dma_span(w, op);
		if ((r->offset & 15) || r->offset > b->size || span > b->size - r->offset) {
			ret = -EINVAL;
			goto out;
		}
		addr = (u64)b->dma + r->offset;
		if ((addr & 15) || addr + span > (1ull << 49)) {
			ret = -EINVAL;
			goto out;
		}
		set_field(w, 32, 45, addr >> 4);
		put_unaligned_le64(w[0], prog + r->instr_index * 16);
		put_unaligned_le64(w[1], prog + r->instr_index * 16 + 8);
	}

	/* every DMA instruction must have been relocated; weights must never underflow */
	for (i = 0; i < n; i++) {
		u64 w[2];
		u8 op = prog[i * 16];

		w[0] = get_unaligned_le64(prog + i * 16);
		w[1] = get_unaligned_le64(prog + i * 16 + 8);
		if ((op == OP_RHM || op == OP_RW || op == OP_WHM) && !test_bit(i, seen)) {
			ret = -EINVAL;
			goto out;
		}
		if (op == OP_RW)
			pushed += get_field(w, 16, 16);
		if ((op == OP_MATMUL || op == OP_CONVOLVE) && (w[0] & BIT(9))) {
			if (++consumed > pushed) {
				ret = -EINVAL;      /* the matrix unit would wait forever */
				goto out;
			}
		}
	}
	return taken;

out:
	while (taken)
		tpu_buf_put(refs[--taken]);
	return ret;
}

/* ------------------------------------------------------------------------------------------ */
/* job execution                                                                             */
/* ------------------------------------------------------------------------------------------ */
static void idma_wr64(struct tpu_dev *t, u32 off, u64 v)
{
	wr(t, off, lower_32_bits(v));
	wr(t, off + 4, upper_32_bits(v));
}

/* Host copy of the staged program into the instruction FIFO through the iDMA frontend. */
static int tpu_push_program(struct tpu_dev *t, unsigned int n)
{
	unsigned long end = jiffies + msecs_to_jiffies(100);
	u32 id, st;

	idma_wr64(t, D_SRC, t->stage_dma);
	idma_wr64(t, D_DST, IFIFO_PUSH_ADDR);
	idma_wr64(t, D_LEN, (u64)n * 16);
	idma_wr64(t, D_SSTRIDE, 0);
	idma_wr64(t, D_DSTRIDE, 0);
	idma_wr64(t, D_REPS, 1);
	wr(t, D_CONF, 0);
	id = rd(t, D_NEXT_ID);          /* side effect: launches the transfer (exactly one read) */
	if (!id)
		return -EIO;
	while ((s32)(rd(t, D_DONE_ID) - id) < 0) {
		if (time_after(jiffies, end))
			return -ETIMEDOUT;
		cpu_relax();
	}
	st = rd(t, R_STATUS);
	if (st & 0x1c)                  /* iDMA read/write/midend error */
		return -EIO;
	if ((rd(t, R_LEVEL) >> 16) != n)
		return -EIO;
	return 0;
}

enum { OUT_DONE, OUT_FAULT, OUT_TIMEOUT };

static int tpu_wait_job(struct tpu_dev *t, unsigned int timeout_ms, int *outcome)
{
	unsigned long deadline = jiffies + msecs_to_jiffies(timeout_ms);
	unsigned long slice = max_t(unsigned long, 1, msecs_to_jiffies(2));

	for (;;) {
		u32 cs;

		if (t->irq_on) {
			long rc = wait_for_completion_interruptible_timeout(&t->done, slice);

			if (rc > 0) {
				*outcome = OUT_DONE;
				return 0;
			}
			if (rc < 0)
				return -EINTR;
		} else {
			if (signal_pending(current))
				return -EINTR;
			usleep_range(20, 50);
		}
		cs = rd(t, R_CSTATUS);
		if (cs & (CS_CFG_ERR | CS_DMA_ERR)) {
			*outcome = OUT_FAULT;
			return 0;
		}
		if (cs & CS_IRQ) {          /* interrupt pending but not delivered */
			if (t->irq_on)
				atomic64_inc(&t->irq_missed);
			*outcome = OUT_DONE;
			return 0;
		}
		if (time_after(jiffies, deadline)) {
			*outcome = OUT_TIMEOUT;
			return 0;
		}
	}
}

static int tpu_run_locked(struct tpu_dev *t, struct tpu_run *r, unsigned int n)
{
	ktime_t t0, t1;
	int outcome = OUT_DONE, ret, i;
	u32 cs, hw;
	unsigned long end;

	r->status = TPU_RUN_OK;

	/* clean slate (the previous job already drained; this also covers a first use) */
	if (rd(t, R_CSTATUS) & CS_BUSY) {
		ret = tpu_drain_locked(t);
		if (ret)
			return ret;
	}
	ctrl_set(t, 0);
	ctrl_pulse(t, CTRL_ERRCLR | CTRL_WF_FLUSH | CTRL_IF_FLUSH | CTRL_IRQ_CLR | CTRL_PERF_CLR);

	ret = tpu_push_program(t, n);
	if (ret) {
		r->status = TPU_RUN_PUSH_FAIL;
		atomic64_inc(&t->jobs_failed);
		r->core_status = rd(t, R_CSTATUS);
		r->hw_status = rd(t, R_STATUS);
		return tpu_drain_locked(t);
	}

	t->state = TPU_STATE_BUSY;
	reinit_completion(&t->done);
	t->snap_valid = false;
	ctrl_set(t, CTRL_UB_OWN | CTRL_ACC_OWN | CTRL_PERF_CLR);   /* counters held clear */
	t->job_running = true;
	t0 = ktime_get();
	ctrl_set(t, CTRL_UB_OWN | CTRL_ACC_OWN | CTRL_RUN);

	ret = tpu_wait_job(t, r->timeout_ms, &outcome);
	t->job_running = false;
	if (ret == -EINTR) {
		tpu_drain_locked(t);        /* abort: stop, drain, clear; a wedge is reported next time */
		return -EINTR;
	}

	if (outcome == OUT_DONE) {          /* wait for HALT (or an idle core) */
		end = jiffies + msecs_to_jiffies(20);
		while ((cs = rd(t, R_CSTATUS)) & CS_BUSY && !(cs & CS_HALTED) && !(cs & (CS_CFG_ERR | CS_DMA_ERR))) {
			if (time_after(jiffies, end))
				break;
			usleep_range(20, 50);
		}
	}
	t1 = t->snap_valid ? t->snap_time : ktime_get();

	cs = rd(t, R_CSTATUS);
	hw = rd(t, R_STATUS);
	r->core_status = cs;
	r->hw_status = hw;
	r->irq_tag = t->snap_valid ? t->snap_tag : rd(t, R_IRQ_TAG);
	for (i = 0; i < TPU_NUM_PERF; i++)
		r->perf[i] = t->snap_valid ? t->snap_perf[i] : rd(t, R_PERF(i));
	r->perf[7] = rd(t, R_PERF(7));          /* retired: complete only after HALT */
	r->wall_ns = ktime_to_ns(ktime_sub(t1, t0));
	r->cfg_err_op = (cs & CS_CFG_ERR) ? CS_ERR_OP(cs) : 0;

	if (outcome == OUT_TIMEOUT)
		r->status = TPU_RUN_TIMEOUT;
	else if ((cs & CS_DMA_ERR) || (hw & ST_ERR_MASK))
		r->status = TPU_RUN_DMA_FAULT;
	else if (cs & CS_CFG_ERR)
		r->status = TPU_RUN_CFG_ERR;
	else if (!(cs & CS_HALTED) && (cs & CS_BUSY))
		r->status = TPU_RUN_TIMEOUT;

	memcpy(t->last_perf, r->perf, sizeof(t->last_perf));
	if (r->status == TPU_RUN_OK)
		atomic64_inc(&t->jobs_ok);
	else
		atomic64_inc(&t->jobs_failed);

	return tpu_drain_locked(t);
}

static int tpu_ioctl_run(struct tpu_file *f, struct tpu_run __user *uarg)
{
	struct tpu_dev *t = f->t;
	struct tpu_run r;
	struct tpu_reloc *rel = NULL;
	struct tpu_buf **refs = NULL;
	u8 *prog = NULL;
	int ret, nref = 0;

	if (copy_from_user(&r, uarg, sizeof(r)))
		return -EFAULT;
	if (r.flags || !r.n_instr || r.n_instr > TPU_MAX_INSTR || r.n_relocs > TPU_MAX_RELOCS)
		return -EINVAL;
	if (!r.timeout_ms)
		r.timeout_ms = DEFAULT_TIMEOUT_MS;
	if (r.timeout_ms > MAX_TIMEOUT_MS)
		return -EINVAL;

	prog = kmalloc(r.n_instr * 16, GFP_KERNEL);
	refs = kcalloc(r.n_relocs ? r.n_relocs : 1, sizeof(*refs), GFP_KERNEL);
	if (r.n_relocs)
		rel = kmalloc_array(r.n_relocs, sizeof(*rel), GFP_KERNEL);
	if (!prog || !refs || (r.n_relocs && !rel)) {
		ret = -ENOMEM;
		goto out;
	}
	if (copy_from_user(prog, u64_to_user_ptr(r.program), r.n_instr * 16) ||
	    (r.n_relocs && copy_from_user(rel, u64_to_user_ptr(r.relocs), r.n_relocs * sizeof(*rel)))) {
		ret = -EFAULT;
		goto out;
	}

	ret = tpu_validate(f, prog, r.n_instr, rel, r.n_relocs, refs);
	if (ret < 0)
		goto out;
	nref = ret;

	ret = mutex_lock_interruptible(&t->job_lock);
	if (ret)
		goto out_refs;
	if (t->state == TPU_STATE_WEDGED) {
		ret = -EIO;
	} else {
		memcpy(t->stage, prog, r.n_instr * 16);
		ret = tpu_run_locked(t, &r, r.n_instr);
	}
	mutex_unlock(&t->job_lock);

	if (!ret && copy_to_user(uarg, &r, sizeof(r)))
		ret = -EFAULT;
out_refs:
	while (nref)
		tpu_buf_put(refs[--nref]);
out:
	kfree(rel);
	kfree(refs);
	kfree(prog);
	return ret;
}

static void fill_info(struct tpu_dev *t, struct tpu_info *i)
{
	memset(i, 0, sizeof(*i));
	i->abi_version = TPU_ABI_VERSION;
	i->hw_version = t->hw_version;
	i->build_id = t->build_id;
	i->state = READ_ONCE(t->state);
	i->max_instr = TPU_MAX_INSTR;
	i->ub_rows = 65536;
	i->acc_rows = 2048;
	i->irq_mode = t->irq_on;
	i->max_buf_bytes = max_buf_mb << 20;
	i->total_buf_limit = total_buf_mb << 20;
	i->total_buf_bytes = atomic64_read(&t->total_bytes);
	i->jobs_ok = atomic64_read(&t->jobs_ok);
	i->jobs_failed = atomic64_read(&t->jobs_failed);
	i->irq_count = atomic64_read(&t->irq_count);
}

static long tpu_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	struct tpu_file *f = file->private_data;
	struct tpu_dev *t = f->t;
	void __user *uarg = (void __user *)arg;
	struct tpu_info info;
	int ret;

	switch (cmd) {
	case TPU_IOC_GET_INFO:
		fill_info(t, &info);
		return copy_to_user(uarg, &info, sizeof(info)) ? -EFAULT : 0;
	case TPU_IOC_ALLOC:
		return tpu_ioctl_alloc(f, uarg);
	case TPU_IOC_FREE:
		return tpu_ioctl_free(f, uarg);
	case TPU_IOC_RUN:
		return tpu_ioctl_run(f, uarg);
	case TPU_IOC_RECOVER:
		ret = mutex_lock_interruptible(&t->job_lock);
		if (ret)
			return ret;
		ret = (t->state == TPU_STATE_WEDGED && rd(t, R_CSTATUS) & CS_BUSY) ? -EIO : tpu_drain_locked(t);
		mutex_unlock(&t->job_lock);
		return ret;
	default:
		return -ENOTTY;
	}
}

static int tpu_open(struct inode *inode, struct file *file)
{
	struct tpu_dev *t = container_of(file->private_data, struct tpu_dev, misc);
	struct tpu_file *f = kzalloc(sizeof(*f), GFP_KERNEL);

	if (!f)
		return -ENOMEM;
	f->t = t;
	mutex_init(&f->lock);
	idr_init(&f->bufs);
	file->private_data = f;
	return 0;
}

static int tpu_release(struct inode *inode, struct file *file)
{
	struct tpu_file *f = file->private_data;
	struct tpu_buf *b;
	int id;

	idr_for_each_entry(&f->bufs, b, id)
		tpu_buf_put(b);
	idr_destroy(&f->bufs);
	kfree(f);
	return 0;
}

static const struct file_operations tpu_fops = {
	.owner          = THIS_MODULE,
	.open           = tpu_open,
	.release        = tpu_release,
	.unlocked_ioctl = tpu_ioctl,
	.compat_ioctl   = compat_ptr_ioctl,
	.mmap           = tpu_mmap,
	.llseek         = no_llseek,
};

/* ------------------------------------------------------------------------------------------ */
/* sysfs                                                                                     */
/* ------------------------------------------------------------------------------------------ */
#define TPU_ATTR(name, fmt, val)                                                         \
static ssize_t name##_show(struct device *d, struct device_attribute *a, char *buf)     \
{                                                                                        \
	struct tpu_dev *t = dev_get_drvdata(d);                                          \
	return sysfs_emit(buf, fmt "\n", val);                                           \
}                                                                                        \
static DEVICE_ATTR_RO(name)

TPU_ATTR(version, "0x%08x", t->hw_version);
TPU_ATTR(build_id, "0x%08x", t->build_id);
TPU_ATTR(state, "%s", t->state == TPU_STATE_IDLE ? "idle" : t->state == TPU_STATE_BUSY ? "busy" : "wedged");
TPU_ATTR(jobs_ok, "%llu", (unsigned long long)atomic64_read(&t->jobs_ok));
TPU_ATTR(jobs_failed, "%llu", (unsigned long long)atomic64_read(&t->jobs_failed));
TPU_ATTR(irq_count, "%llu", (unsigned long long)atomic64_read(&t->irq_count));
TPU_ATTR(irq_missed, "%llu", (unsigned long long)atomic64_read(&t->irq_missed));
TPU_ATTR(buf_bytes, "%lld", (long long)atomic64_read(&t->total_bytes));

static ssize_t last_perf_show(struct device *d, struct device_attribute *a, char *buf)
{
	struct tpu_dev *t = dev_get_drvdata(d);

	return sysfs_emit(buf, "total=%u array_active=%u w_stall=%u w_shift=%u non_matrix=%u sync=%u act=%u retired=%u\n",
			  t->last_perf[0], t->last_perf[1], t->last_perf[2], t->last_perf[3],
			  t->last_perf[4], t->last_perf[5], t->last_perf[6], t->last_perf[7]);
}
static DEVICE_ATTR_RO(last_perf);

static struct attribute *tpu_attrs[] = {
	&dev_attr_version.attr, &dev_attr_build_id.attr, &dev_attr_state.attr,
	&dev_attr_jobs_ok.attr, &dev_attr_jobs_failed.attr, &dev_attr_irq_count.attr,
	&dev_attr_irq_missed.attr, &dev_attr_buf_bytes.attr, &dev_attr_last_perf.attr, NULL,
};
ATTRIBUTE_GROUPS(tpu);

/* ------------------------------------------------------------------------------------------ */
/* platform driver                                                                           */
/* ------------------------------------------------------------------------------------------ */
static int tpu_probe(struct platform_device *pdev)
{
	struct device *dev = &pdev->dev;
	struct tpu_dev *t;
	int ret;

	t = devm_kzalloc(dev, sizeof(*t), GFP_KERNEL);
	if (!t)
		return -ENOMEM;
	t->dev = dev;
	mutex_init(&t->job_lock);
	spin_lock_init(&t->ctrl_lock);
	init_completion(&t->done);
	atomic64_set(&t->total_bytes, 0);
	platform_set_drvdata(pdev, t);

	t->regs = devm_platform_ioremap_resource(pdev, 0);
	if (IS_ERR(t->regs))
		return PTR_ERR(t->regs);

	ret = dma_set_mask_and_coherent(dev, DMA_BIT_MASK(49));
	if (ret)
		return ret;

	/* The PL must be loaded: reads of an unprogrammed PL window can stall the CPU (L5). */
	t->hw_version = rd(t, R_VERSION);
	t->build_id = rd(t, R_BUILD_ID);
	if ((t->hw_version >> 8) != 4 && (t->hw_version >> 8) != 5) {
		dev_err(dev, "unsupported/missing TPU (VERSION=0x%08x); load the tpu_kv260 app first\n",
			t->hw_version);
		return -ENODEV;
	}
	dev_info(dev, "TPU VERSION=0x%08x BUILD_ID=0x%08x\n", t->hw_version, t->build_id);

	t->stage = dmam_alloc_coherent(dev, STAGE_BYTES, &t->stage_dma, GFP_KERNEL);
	if (!t->stage)
		return -ENOMEM;

	/* start from a known state */
	ctrl_set(t, 0);
	if (rd(t, R_CSTATUS) & CS_BUSY) {
		dev_warn(dev, "core busy at probe, draining\n");
		ret = tpu_drain_locked(t);
		if (ret)
			return ret;
	}
	ctrl_pulse(t, CTRL_ERRCLR | CTRL_WF_FLUSH | CTRL_IF_FLUSH | CTRL_IRQ_CLR | CTRL_PERF_CLR);

	t->irq = use_irq ? platform_get_irq(pdev, 0) : -ENXIO;
	if (t->irq > 0) {
		ret = devm_request_irq(dev, t->irq, tpu_irq, 0, DRV_NAME, t);
		if (ret)
			return ret;
		t->irq_on = true;
	} else {
		dev_info(dev, "polling mode (no interrupt)\n");
	}

	t->misc.minor = MISC_DYNAMIC_MINOR;
	t->misc.name = "tpu0";
	t->misc.fops = &tpu_fops;
	t->misc.mode = 0660;
	t->misc.parent = dev;
	ret = misc_register(&t->misc);
	if (ret)
		return ret;
	ret = sysfs_create_groups(&dev->kobj, tpu_groups);
	if (ret) {
		misc_deregister(&t->misc);
		return ret;
	}
	dev_info(dev, "/dev/tpu0 ready (%s, irq %d)\n", t->irq_on ? "interrupt" : "polled", t->irq);
	return 0;
}

static int tpu_remove(struct platform_device *pdev)
{
	struct tpu_dev *t = platform_get_drvdata(pdev);

	sysfs_remove_groups(&pdev->dev.kobj, tpu_groups);
	misc_deregister(&t->misc);
	mutex_lock(&t->job_lock);
	if (t->state != TPU_STATE_WEDGED)
		ctrl_set(t, 0);
	mutex_unlock(&t->job_lock);
	return 0;
}

/* No of_match_table on purpose: bind explicitly with driver_override (tpu-load.sh) so the
 * driver can never grab an unrelated "generic-uio" node. */
static struct platform_driver tpu_driver = {
	.probe  = tpu_probe,
	.remove = tpu_remove,
	.driver = { .name = DRV_NAME },
};
module_platform_driver(tpu_driver);

MODULE_DESCRIPTION("TPU-KV260 accelerator driver");
MODULE_LICENSE("GPL");
MODULE_VERSION("1.0");
