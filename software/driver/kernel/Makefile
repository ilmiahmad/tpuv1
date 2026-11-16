# Out-of-tree build, native on the KV260:   make   (needs /lib/modules/$(uname -r)/build)
obj-m := tpu_kv260.o
ccflags-y := -I$(src)/../include/uapi -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare

KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
