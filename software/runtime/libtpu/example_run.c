// SPDX-License-Identifier: MIT
/*
 * example_run: run an exported MLP bundle on the TPU and compare with the golden output.
 *
 *   ./example_run <bundle_dir> [repeat]
 *
 * Reads program.bin, relocs.txt, weights.bin, sample_in.bin, sample_out.bin and meta.txt (written by
 * model/mlp/export_bundle.py), allocates three DMA buffers, runs the program and prints the
 * hardware counters.  Exit status 0 only if the output is bit-exact.
 */
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tpu.h"

static void *slurp(const char *dir, const char *name, size_t *len)
{
	char path[1024];
	FILE *f;
	void *buf;
	long n;

	snprintf(path, sizeof(path), "%s/%s", dir, name);
	f = fopen(path, "rb");
	if (!f) {
		fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
		exit(2);
	}
	fseek(f, 0, SEEK_END);
	n = ftell(f);
	fseek(f, 0, SEEK_SET);
	buf = malloc(n ? (size_t)n : 1);
	if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
		fprintf(stderr, "cannot read %s\n", path);
		exit(2);
	}
	fclose(f);
	*len = (size_t)n;
	return buf;
}

int main(int argc, char **argv)
{
	size_t prog_len, w_len, in_len, out_len, rel_text_len;
	void *prog, *weights, *input, *golden;
	char *rel_text, *line, *save = NULL;
	struct tpu_reloc *relocs = NULL;
	unsigned n_relocs = 0, cap = 0;
	struct tpu_buf in_buf, w_buf, out_buf;
	struct tpu_buf *by_name;
	struct tpu_run res;
	tpu_dev_t *dev;
	int repeat = argc > 2 ? atoi(argv[2]) : 1, bad = 0, r, i;

	if (argc < 2) {
		fprintf(stderr, "usage: %s <bundle_dir> [repeat]\n", argv[0]);
		return 2;
	}
	prog = slurp(argv[1], "program.bin", &prog_len);
	weights = slurp(argv[1], "weights.bin", &w_len);
	input = slurp(argv[1], "sample_in.bin", &in_len);
	golden = slurp(argv[1], "sample_out.bin", &out_len);
	rel_text = slurp(argv[1], "relocs.txt", &rel_text_len);
	rel_text = realloc(rel_text, rel_text_len + 1);
	rel_text[rel_text_len] = 0;

	dev = tpu_open(NULL);
	if (!dev) {
		fprintf(stderr, "cannot open /dev/tpu0: %s\n", strerror(errno));
		return 2;
	}
	if (tpu_alloc(dev, in_len, &in_buf) || tpu_alloc(dev, w_len, &w_buf) || tpu_alloc(dev, out_len, &out_buf)) {
		perror("tpu_alloc");
		return 1;
	}
	memcpy(in_buf.ptr, input, in_len);
	memcpy(w_buf.ptr, weights, w_len);

	/* relocs.txt: "<instr_index> <input|weights|output> <byte_offset>" per line */
	for (line = strtok_r(rel_text, "\n", &save); line; line = strtok_r(NULL, "\n", &save)) {
		unsigned idx;
		char which[16];
		unsigned long long off;

		if (sscanf(line, "%u %15s %llu", &idx, which, &off) != 3)
			continue;
		by_name = !strcmp(which, "input") ? &in_buf : !strcmp(which, "weights") ? &w_buf : &out_buf;
		if (n_relocs == cap) {
			cap = cap ? cap * 2 : 64;
			relocs = realloc(relocs, cap * sizeof(*relocs));
		}
		relocs[n_relocs].instr_index = idx;
		relocs[n_relocs].handle = by_name->handle;
		relocs[n_relocs].offset = off;
		n_relocs++;
	}

	for (r = 0; r < repeat; r++) {
		memset(out_buf.ptr, 0xEE, out_len);            /* unwritten output bytes would show */
		if (tpu_run(dev, prog, (unsigned)(prog_len / 16), relocs, n_relocs, 0, &res) < 0) {
			perror("tpu_run");
			return 1;
		}
		bad += res.status != TPU_RUN_OK || memcmp(out_buf.ptr, golden, out_len) != 0;
		printf("run %d: %s, kernel wall %.1f us, tag 0x%08x\n", r, tpu_status_str(res.status),
		       (double)res.wall_ns / 1000.0, res.irq_tag);
	}
	for (i = 0; i < TPU_NUM_PERF; i++)
		printf("  %-12s %u\n", tpu_perf_names[i], res.perf[i]);
	printf(bad ? "FAIL: %d of %d runs not bit-exact\n" : "PASS: %d runs bit-exact\n", bad ? bad : repeat, repeat);
	tpu_free(dev, &out_buf);
	tpu_free(dev, &w_buf);
	tpu_free(dev, &in_buf);
	tpu_close(dev);
	free(relocs);
	return bad ? 1 : 0;
}
