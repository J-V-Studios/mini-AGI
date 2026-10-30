.section .text.entry
.code64
.global entry64
entry64:
    cli
    cld
    clts
    leaq stack_top(%rip), %rsp
    andq $-16, %rsp
    movq %cr0, %rax
    andq $~4, %rax
    orq $2, %rax
    movq %rax, %cr0
    movq %cr4, %rax
    orq $0x600, %rax
    movq %rax, %cr4
    fninit

    leaq __bss_start(%rip), %rdi
    leaq __bss_end(%rip), %rcx
    subq %rdi, %rcx
    xorq %rax, %rax
    rep stosb

    movw $0x3F8, %dx
    movb $'E', %al
    outb %al, %dx

    xorq %rbp, %rbp
    call kmain
1:
    hlt
    jmp 1b

.global __bss_start
.global __bss_end
.global stack_top
