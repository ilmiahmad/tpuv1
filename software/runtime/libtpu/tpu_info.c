// SPDX-License-Identifier: MIT
/* tpu_info: print version, identity, state, limits and job counters of /dev/tpu0. */
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "tpu.h"

int main(int argc, char **argv)
{
	const char *path = argc > 1 ? argv[1] : NULL;
	struct tpu_info i;
	tpu_dev_t *dev = tpu_open(path);

	if (!dev) {
		fprintf(stderr, "tpu_info: cannot open %s: %s%s\n", path ? path : "/dev/tpu0", strerror(errno),
			errno == ENOENT ? " (is the driver loaded? see docs/driver_user_guide.md)" : "");
		return errno == ENOENT ? 2 : 1;
	}
	if (tpu_get_info(dev, &i) < 0) {
		fprintf(stderr, "tpu_info: TPU_IOC_GET_INFO: %s\n", strerror(errno));
		tpu_close(dev);
		return 1;
	}
	printf("abi version      %u\n", i.abi_version);
	printf("hw version       0x%08x\n", i.hw_version);
	printf("build id         0x%08x\n", i.build_id);
	printf("state            %s\n", tpu_state_str(i.state));
	printf("completion       %s\n", i.irq_mode ? "interrupt" : "polled");
	printf("max instructions %u\n", i.max_instr);
	printf("UB rows          %u\n", i.ub_rows);
	printf("ACC rows         %u\n", i.acc_rows);
	printf("buffer limits    %" PRIu64 " per buffer, %" PRIu64 " total (%" PRIu64 " in use)\n",
	       (uint64_t)i.max_buf_bytes, (uint64_t)i.total_buf_limit, (uint64_t)i.total_buf_bytes);
	printf("jobs             %" PRIu64 " ok, %" PRIu64 " failed, %" PRIu64 " interrupts\n",
	       (uint64_t)i.jobs_ok, (uint64_t)i.jobs_failed, (uint64_t)i.irq_count);
	tpu_close(dev);
	return 0;
}
