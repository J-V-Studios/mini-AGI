/* Enters long mode and copies the staged kernel to 1MB. */
.code16
.globl stage2
stage2:
    cli
    lgdt gdtr
    movl %cr0, %eax
    orl $1, %eax
    movl %eax, %cr0
    ljmpl $0x08, $pm32

.code32
pm32:
    movw $0x10, %ax
    movw %ax, %ds
    movw %ax, %es
    movw %ax, %ss
    movl $0x90000, %esp

    movl $0x1000, %edi
    movl $0xC00, %ecx
    xorl %eax, %eax
    rep stosl

    movl $0x2003, 0x1000
    movl $0x3003, 0x2000
    /* PAE entries are 8 bytes. The high half stays 0: the machine
       is mapped below 4GB. */
    movl $0x83, %eax
    xorl %edx, %edx
    movl $0x3000, %edi
    movl $128, %ecx
1:
    movl %eax, (%edi)
    movl %edx, 4(%edi)
    addl $0x200000, %eax
    adcl $0, %edx
    addl $8, %edi
    decl %ecx
    jnz 1b

    movl $0x1000, %eax
    movl %eax, %cr3
    movl %cr4, %eax
    orl $0x20, %eax
    movl %eax, %cr4
    movl $0xC0000080, %ecx
    rdmsr
    orl $0x100, %eax
    wrmsr
    movl %cr0, %eax
    orl $0x80000000, %eax
    movl %eax, %cr0
    ljmpl $0x18, $pm64

.code64
pm64:
    movw $0x20, %ax
    movw %ax, %ds
    movw %ax, %es
    movw %ax, %ss
    movw $0x3F8, %dx
    movb $'S', %al
    outb %al, %dx

    movl kernel_bytes(%rip), %ecx
    movq $0x20000, %rsi
    movq $0x100000, %rdi
    rep movsb

    movabs $0x100000, %rax
    jmpq *%rax

.align 8
gdt:
    .quad 0
    .quad 0x00CF9A000000FFFF
    .quad 0x00CF92000000FFFF
    .quad 0x00AF9A000000FFFF
    .quad 0x00AF92000000FFFF
gdt_end:

gdtr:
    .word gdt_end - gdt - 1
    .long gdt

.ascii "MAG2"
kernel_bytes:
    .long 0
