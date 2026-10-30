ENTRY(entry64)

SECTIONS {
    . = 1M;

    .text : ALIGN(16) {
        *(.text.entry)
        *(.text .text.*)
    }

    .rodata : ALIGN(16) {
        *(.rodata .rodata.*)
    }

    .data : ALIGN(16) {
        *(.data .data.*)
    }

    .bss : ALIGN(16) {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(16);
        stack_bottom = .;
        . += 65536;
        stack_top = .;
        . = ALIGN(16);
        __bss_end = .;
    }
}
