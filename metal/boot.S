/* BIOS boot sector. Loads stage2 and the kernel, then jumps to stage2.
   The kernel is staged at 0x20000 because real mode cannot see 1MB. */
.code16
.globl _start
_start:
    cli
    xorw %ax, %ax
    movw %ax, %ds
    movw %ax, %es
    movw %ax, %ss
    movw $0x7C00, %sp
    sti
    movb %dl, drive

    movw $0x3FB, %dx
    movb $0x80, %al
    outb %al, %dx
    movw $0x3F8, %dx
    movb $1, %al
    outb %al, %dx
    movw $0x3F9, %dx
    xorb %al, %al
    outb %al, %dx
    movw $0x3FB, %dx
    movb $0x03, %al
    outb %al, %dx
    movw $0x3FA, %dx
    movb $0xC7, %al
    outb %al, %dx
    movw $0x3FC, %dx
    movb $0x0B, %al
    outb %al, %dx
    movb $'B', %bl
    call putc

    inb $0x92, %al
    orb $2, %al
    outb %al, $0x92

    /* stage2 at 0x7E00, LBA 1 */
    xorl %ecx, %ecx
    movl $1, %eax
    movw stage2_sectors, %cx
    movw $0, %di
    movw $0x7E00, %bx
    call read_chunks
    jc fail
    movb $'2', %bl
    call putc

    /* kernel staged at 0x20000 */
    movl kernel_lba, %eax
    movl kernel_sectors, %ecx
    movw $0x2000, %di
    xorw %bx, %bx
    call read_chunks
    jc fail
    movb $'K', %bl
    call putc
    ljmp $0, $0x7E00

fail:
    movb $'E', %bl
    call putc
    hlt
    jmp fail

/* eax = LBA, ecx = sectors, di = segment, bx = offset.
   One sector at a time. 0x20 paragraphs is 512 bytes. */
read_chunks:
    movw %di, %es
1:
    testl %ecx, %ecx
    jz 3f
    pushl %eax
    pushl %ecx
    movw $1, dap_count
    movw %bx, dap_off
    movw %es, dap_seg
    movl %eax, dap_lba
    movl $0, dap_lba_hi
    movb drive, %dl
    movw $dap, %si
    movb $0x42, %ah
    int $0x13
    jc 4f
    popl %ecx
    popl %eax
    addl $1, %eax
    subl $1, %ecx
    movw %es, %di
    addw $0x20, %di
    movw %di, %es
    jmp 1b
3:
    clc
    ret
4:
    popl %ecx
    popl %eax
    stc
    ret

putc:
    pushw %dx
    movw $0x3FD, %dx
1:
    inb %dx, %al
    testb $0x20, %al
    jz 1b
    movw $0x3F8, %dx
    movb %bl, %al
    outb %al, %dx
    popw %dx
    ret

drive:
    .byte 0

.align 4
dap:
    .byte 0x10, 0
dap_count:
    .word 0
dap_off:
    .word 0
dap_seg:
    .word 0
dap_lba:
    .long 0
dap_lba_hi:
    .long 0

.ascii "MAGI"
kernel_sectors:
    .long 0
kernel_lba:
    .long 0
stage2_sectors:
    .word 0

.org 510
.word 0xAA55
