#!/usr/bin/env python3
"""Pack the boot sector, stage2 and the kernel ELF into one bootable disk."""

import struct
import sys
from pathlib import Path


def elf_image(path):
    data = Path(path).read_bytes()
    if data[:4] != b"\x7fELF":
        raise SystemExit(f"{path} is not an ELF")
    e_phoff = struct.unpack_from("<Q", data, 32)[0]
    e_phentsize = struct.unpack_from("<H", data, 54)[0]
    e_phnum = struct.unpack_from("<H", data, 56)[0]
    loads = []
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type, _, p_offset, p_vaddr, _, p_filesz, _, _ = struct.unpack_from(
            "<IIQQQQQQ", data, off)
        if p_type == 1 and p_filesz:
            loads.append((p_vaddr, p_offset, p_filesz))
    if not loads:
        raise SystemExit(f"{path} has no loadable segment")
    base = min(v for v, _, _ in loads)
    end = max(v + s for v, _, s in loads)
    blob = bytearray(end - base)
    for vaddr, off, size in loads:
        blob[vaddr - base:vaddr - base + size] = data[off:off + size]
    return bytes(blob), base


def patch(buf, magic, payload):
    at = buf.find(magic)
    if at < 0:
        raise SystemExit(f"magic {magic!r} not found")
    buf[at + len(magic):at + len(magic) + len(payload)] = payload


def main():
    boot = bytearray(Path(sys.argv[1]).read_bytes())
    stage2 = bytearray(Path(sys.argv[2]).read_bytes())
    kernel, base = elf_image(sys.argv[3])
    out = Path(sys.argv[4])
    if base != 0x100000:
        raise SystemExit(f"kernel linked at {base:#x}, expected 0x100000")
    if len(boot) != 512:
        raise SystemExit(f"boot sector is {len(boot)} bytes, expected 512")
    if len(kernel) > 0x60000:
        raise SystemExit(f"kernel image is {len(kernel)} bytes, staging area is 384KB")

    stage2_sectors = (len(stage2) + 511) // 512
    stage2 = stage2.ljust(stage2_sectors * 512, b"\x00")
    kernel_sectors = (len(kernel) + 511) // 512
    kernel = kernel.ljust(kernel_sectors * 512, b"\x00")
    kernel_lba = 1 + stage2_sectors

    patch(boot, b"MAGI", struct.pack("<IIH", kernel_sectors, kernel_lba, stage2_sectors))
    patch(stage2, b"MAG2", struct.pack("<I", len(kernel)))

    # Room past the kernel for the weight store at LBA 2048.
    weight_end = 2048 + 4096
    disk = bytearray(weight_end * 512)
    disk[0:512] = boot
    disk[512:512 + len(stage2)] = stage2
    off = kernel_lba * 512
    disk[off:off + len(kernel)] = kernel
    out.write_bytes(disk)
    print(f"image {out}  kernel {len(kernel)} bytes at LBA {kernel_lba}  "
          f"linked {base:#x}")


if __name__ == "__main__":
    main()
