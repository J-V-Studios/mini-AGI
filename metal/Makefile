CC = gcc
AS = as
PY = python3

CFLAGS = -std=c11 -ffreestanding -fno-stack-protector -fno-pic -fno-asynchronous-unwind-tables \
	-mno-red-zone -msse2 -mfpmath=sse -O2 -Wall -Wextra -Wno-unused-function \
	-I.
LDFLAGS = -nostdlib -static -Wl,-T,link.ld -Wl,--build-id=none -lgcc

HOST_CFLAGS = -DHOST -std=c11 -O2 -Wall -Wextra -Wno-unused-function

OBJS = entry.o kernel.o model.o math.o mem.o plasticity.o

.PHONY: all test boot clean

all: minagi.img

test: host_test
	./host_test

host_test: math.c mem.c plasticity.c model.c host_test.c model.h math.h mem.h plasticity.h
	$(CC) $(HOST_CFLAGS) -o $@ math.c mem.c plasticity.c model.c host_test.c -lm

minagi.img: boot.bin stage2.bin kernel.elf mkimage.py
	$(PY) mkimage.py boot.bin stage2.bin kernel.elf $@

kernel.elf: $(OBJS) link.ld
	$(CC) -nostdlib -static -Wl,-T,link.ld -Wl,--build-id=none -o $@ $(OBJS) -lgcc

entry.o: entry.S
	$(AS) --64 -o $@ $<

kernel.o: kernel.c model.h
	$(CC) $(CFLAGS) -c -o $@ kernel.c

model.o: model.c model.h math.h mem.h plasticity.h
	$(CC) $(CFLAGS) -c -o $@ model.c

math.o: math.c math.h
	$(CC) $(CFLAGS) -c -o $@ math.c

mem.o: mem.c mem.h
	$(CC) $(CFLAGS) -c -o $@ mem.c

plasticity.o: plasticity.c plasticity.h math.h
	$(CC) $(CFLAGS) -c -o $@ plasticity.c

boot.bin: boot.S
	$(AS) --32 -o boot.o boot.S
	ld -m elf_i386 -Ttext=0x7C00 --oformat binary -o $@ boot.o

stage2.bin: stage2.S
	$(AS) --64 -o stage2.o stage2.S
	ld -m elf_x86_64 -Ttext=0x7E00 --oformat binary -o $@ stage2.o

boot: minagi.img
	qemu-system-x86_64 -m 256M -drive format=raw,file=minagi.img,if=ide \
		-serial stdio -display none -no-reboot

clean:
	rm -f *.o *.elf *.bin *.img host_test
