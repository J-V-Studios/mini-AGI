# mini-AGI, with nothing underneath

This directory is the learning machine from `minagi/` brought down to the
metal. A BIOS loads it, the CPU enters long mode, and from there the only
code running is this image. There is no host operating system and no C
library. The weights, the gradients and Adam's moments live in physical
memory. A run is written back to the same disk the machine booted from.

The arithmetic is the one the Python runs:

- vocabulary of the 256 byte values plus the nine structural markers
- RMSNorm, rotary positions, causal attention, SwiGLU
- two prelude blocks, then one recurrent block up to four times
- a shared pool of experts, top-2 of the four the text has admitted
- PonderNet halting, greedy decoding with the adaptation trace
- AdamW, the trunk at a tenth of the expert rate, grad clip 1
- growth by recombining hidden units, refused unless the brakes agree
- pruning of an expert nothing has admitted for its survival window
- the held-out governor on the learning rate

The widths are the fields in `config.yaml`, set to what fits beside the
optimiser state in the machine's own RAM: width 32, 4 heads, expert hidden
32, 8 experts of which 4 may be resident, context 16 characters a lesson.
A host build of the same C checks that a fresh model starts near `log(265)`
nats, that the gradients match a central difference, and that a few dozen
steps on the built-in corpus bring the loss down.

What did not come down here is the part that talks to other machines:
downloading corpora, the plots, and the browser UI. The keyboard is the
corpus you type, and the screen is the sample log.

## Build

```bash
cd metal
make test          # the same C, on the host, against its own gradients
make               # minagi.img, a bootable disk
make boot          # qemu-system-x86_64, serial on the terminal
```

`make test` needs gcc. `make boot` needs qemu. The image is also a raw disk
you can hand to anything that boots an IDE drive:

```bash
qemu-system-x86_64 -m 256M -drive format=raw,file=minagi.img,if=ide \
    -serial stdio -display none
```

On the serial line the boot letters are the stages: `B` the sector, `2` the
second stage, `K` the kernel read in, `S` long mode, `E` the entry. Then the
machine reads its built-in corpus, prints the loss, continues the prompt
`the cat `, and writes the weights at LBA 2048. After `ready` the keyboard
and the serial port take a line:

| line | |
|---|---|
| `train 8` | eight more lessons |
| `held` | loss on the withheld lines |
| `grow` | the brakes, and why they refused |
| `grow!` | recombine one expert anyway |
| `prune` / `prune now` | the survival rule, or drop the last addition |
| `stat` | step, loss, depth, experts, parameter count |
| `save` / `load` | the weight image on the boot disk |
| anything else | a prompt; the reply is greedy |

The screen is the VGA text buffer at `0xB8000`. Serial is COM1.
