# libtpu + tools. Native build on the board (aarch64, gcc 11) or on the host.
#   make            build libtpu.a, tpu_info, example_run
#   make check      compile only (-fsyntax-only), no linking
UAPI ?= ../../driver/include/uapi
CC ?= gcc
CFLAGS ?= -O2
CFLAGS += -std=gnu11 -Wall -Wextra -Werror -Wshadow -Wconversion -Wno-sign-conversion -I. -I$(UAPI)

all: libtpu.a tpu_info example_run

tpu.o: tpu.c tpu.h $(UAPI)/tpu_kv260.h
	$(CC) $(CFLAGS) -c -o $@ $<

libtpu.a: tpu.o
	$(AR) rcs $@ $^

tpu_info: tpu_info.c libtpu.a
	$(CC) $(CFLAGS) -o $@ tpu_info.c libtpu.a

example_run: example_run.c libtpu.a
	$(CC) $(CFLAGS) -o $@ example_run.c libtpu.a

check:
	$(CC) $(CFLAGS) -fsyntax-only tpu.c tpu_info.c example_run.c

clean:
	rm -f *.o libtpu.a tpu_info example_run

.PHONY: all check clean
