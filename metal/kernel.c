#include "model.h"

#include <stdint.h>

static inline void outb(uint16_t port, uint8_t v) {
    __asm__ volatile("outb %0, %1" : : "a"(v), "Nd"(port));
}
static inline uint8_t inb(uint16_t port) {
    uint8_t v;
    __asm__ volatile("inb %1, %0" : "=a"(v) : "Nd"(port));
    return v;
}
static inline void outw(uint16_t port, uint16_t v) {
    __asm__ volatile("outw %0, %1" : : "a"(v), "Nd"(port));
}
static inline uint16_t inw(uint16_t port) {
    uint16_t v;
    __asm__ volatile("inw %1, %0" : "=a"(v) : "Nd"(port));
    return v;
}

enum { VGA = 0xB8000, COLS = 80, ROWS = 25, WEIGHT_LBA = 2048 };

static int col, row;
static uint8_t attr = 0x07;

static void serial_putc(char c) {
    int i;
    for (i = 0; i < 100000; i++) {
        if (inb(0x3FD) & 0x20) break;
    }
    outb(0x3F8, (uint8_t)c);
}

static void vga_putc(char c) {
    volatile uint16_t *fb = (volatile uint16_t *)VGA;
    int i;
    if (c == '\n') {
        col = 0;
        row++;
    } else {
        fb[row * COLS + col] = (uint16_t)((attr << 8) | (uint8_t)c);
        if (++col >= COLS) {
            col = 0;
            row++;
        }
    }
    if (row >= ROWS) {
        for (i = 0; i < (ROWS - 1) * COLS; i++) fb[i] = fb[i + COLS];
        for (; i < ROWS * COLS; i++) fb[i] = (uint16_t)(attr << 8 | ' ');
        row = ROWS - 1;
    }
}

static void putc(char c) {
    if (c == '\n') serial_putc('\r');
    serial_putc(c);
    vga_putc(c);
}

static void puts(const char *s) {
    while (*s) putc(*s++);
}

static void put_i(int v) {
    char tmp[16];
    int k = 0;
    if (v < 0) {
        putc('-');
        v = -v;
    }
    if (v == 0) tmp[k++] = '0';
    while (v && k < 16) {
        tmp[k++] = (char)('0' + (v % 10));
        v /= 10;
    }
    while (k) putc(tmp[--k]);
}

static void put_f(float v, int prec) {
    int ip, i, dig;
    if (!(v == v)) {
        puts("nan");
        return;
    }
    if (v < 0.f) {
        putc('-');
        v = -v;
    }
    ip = (int)v;
    put_i(ip);
    putc('.');
    v -= (float)ip;
    for (i = 0; i < prec; i++) {
        v *= 10.f;
        dig = (int)v;
        if (dig < 0) dig = 0;
        if (dig > 9) dig = 9;
        putc((char)('0' + dig));
        v -= (float)dig;
    }
}

static void banner(void) {
    volatile uint16_t *fb = (volatile uint16_t *)VGA;
    const char *s = " mini-AGI   bare metal   the weights live in this machine ";
    int i;
    attr = 0x1F;
    for (i = 0; i < COLS; i++) {
        char c = (s[i] && i < 60) ? s[i] : ' ';
        fb[i] = (uint16_t)(0x1F00 | (uint8_t)c);
    }
    row = 1;
    col = 0;
    attr = 0x07;
}

static int ata_wait_drq(void) {
    int i;
    for (i = 0; i < 2000000; i++) {
        uint8_t s = inb(0x1F7);
        if (s & 0x01) return -1;
        if (s & 0x20) return -1;
        if (!(s & 0x80) && (s & 0x08)) return 0;
    }
    return -2;
}

static int ata_wait_ready(void) {
    int i;
    for (i = 0; i < 2000000; i++) {
        uint8_t s = inb(0x1F7);
        if (!(s & 0x80) && (s & 0x40)) return 0;
    }
    return -2;
}

static int ata_pio(uint32_t lba, void *buf, int write) {
    uint16_t *p = buf;
    int i;
    if (ata_wait_ready()) return -1;
    outb(0x1F6, (uint8_t)(0xE0 | ((lba >> 24) & 0x0F)));
    outb(0x1F2, 1);
    outb(0x1F3, (uint8_t)lba);
    outb(0x1F4, (uint8_t)(lba >> 8));
    outb(0x1F5, (uint8_t)(lba >> 16));
    outb(0x1F7, write ? 0x30 : 0x20);
    if (ata_wait_drq()) return -1;
    if (write) {
        for (i = 0; i < 256; i++) outw(0x1F0, p[i]);
    } else {
        for (i = 0; i < 256; i++) p[i] = inw(0x1F0);
    }
    return 0;
}

static int store_model(Model *m, int write) {
    int n = model_sector_count(m);
    int i;
    uint8_t sec[512];
    if (n <= 0 || n > 3000) return -1;
    for (i = 0; i < n; i++) {
        if (write) {
            model_export_sector(m, (unsigned)i, sec);
            if (ata_pio(WEIGHT_LBA + (uint32_t)i, sec, 1)) return -1;
        } else {
            if (ata_pio(WEIGHT_LBA + (uint32_t)i, sec, 0)) return -1;
            if (!model_import_sector(m, (unsigned)i, sec)) return -1;
        }
    }
    return n;
}

static const char scancode[128] = {
    0, 27, '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', '\b',
    '\t', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\n', 0,
    'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '\'', '`', 0, '\\',
    'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', 0, '*', 0, ' '
};

static int key_poll(void) {
    if (inb(0x64) & 1) {
        uint8_t sc = inb(0x60);
        if (sc & 0x80) return 0;
        if (sc < 128) return scancode[sc];
    }
    if (inb(0x3FD) & 1) return inb(0x3F8);
    return 0;
}

static int starts(const char *s, const char *p) {
    while (*p) {
        if (*s != *p) return 0;
        s++;
        p++;
    }
    return 1;
}

static int parse_n(const char *s, int fallback) {
    int n = 0, any = 0;
    while (*s == ' ') s++;
    while (*s >= '0' && *s <= '9') {
        n = n * 10 + (*s - '0');
        s++;
        any = 1;
    }
    return any ? n : fallback;
}

static void do_line(Model *m, char *line) {
    char why[80];
    char stat[160];
    char text[80];
    int ids[64];
    int out[48];
    int n, g, i;
    while (*line == ' ') line++;
    if (!*line) return;
    if (starts(line, "train") || (line[0] == 't' && (line[1] == 0 || line[1] == ' '))) {
        const char *p = line;
        int k;
        while (*p && *p != ' ') p++;
        k = parse_n(p, 1);
        if (k < 1) k = 1;
        if (k > 64) k = 64;
        for (i = 0; i < k; i++) {
            float L = model_lesson(m);
            puts("  lesson ");
            put_i(i + 1);
            puts("  loss ");
            put_f(L, 4);
            putc('\n');
        }
        return;
    }
    if (starts(line, "held")) {
        float h = model_heldout(m);
        puts("held-out ");
        put_f(h, 4);
        putc('\n');
        return;
    }
    if (starts(line, "grow")) {
        int force = starts(line, "grow force") || starts(line, "grow!");
        model_grow(m, force, why, 80);
        puts(why);
        putc('\n');
        return;
    }
    if (starts(line, "prune")) {
        int r = model_prune(m, starts(line, "prune now"));
        puts("pruned ");
        put_i(r);
        putc('\n');
        return;
    }
    if (starts(line, "stat")) {
        model_stat(m, stat, 160);
        puts(stat);
        putc('\n');
        return;
    }
    if (starts(line, "save")) {
        int r = store_model(m, 1);
        if (r < 0) puts("disk did not take the weights\n");
        else {
            puts("wrote ");
            put_i(r);
            puts(" sectors at LBA 2048\n");
        }
        return;
    }
    if (starts(line, "load")) {
        int r = store_model(m, 0);
        if (r < 0) puts("disk had no weights\n");
        else puts("weights restored\n");
        return;
    }
    if (starts(line, "help")) {
        puts("train [n]  held  grow  grow!  prune  prune now  stat  save  load\n");
        puts("any other line is a prompt. the reply is greedy.\n");
        return;
    }
    n = tok_encode(line, ids, 64);
    g = model_generate(m, ids, n, out, 48);
    puts(line);
    tok_decode(out, g, text, 80);
    puts(text);
    putc('\n');
}

void kmain(void) {
    Model *m;
    int i;
    int prompt[8];
    int gen[64];
    char text[80];
    char stat[160];
    char why[80];
    char line[80];
    int len = 0;
    float first = 0.f, loss = 0.f;

    banner();
    puts("no operating system underneath.\n");
    puts("d=32  heads=4  prelude=2  recur<=4  experts=8  resident=4  topk=2  vocab=265\n");
    m = model_create(1);
    model_stat(m, stat, 160);
    puts(stat);
    putc('\n');
    puts("reading the built-in corpus\n");
    for (i = 0; i < 16; i++) {
        loss = model_lesson(m);
        if (i == 0) first = loss;
        puts("  ");
        put_i(i + 1);
        puts("  ");
        put_f(loss, 4);
        putc('\n');
    }
    puts("loss ");
    put_f(first, 4);
    puts(" -> ");
    put_f(loss, 4);
    putc('\n');

    prompt[0] = 't';
    prompt[1] = 'h';
    prompt[2] = 'e';
    prompt[3] = ' ';
    prompt[4] = 'c';
    prompt[5] = 'a';
    prompt[6] = 't';
    prompt[7] = ' ';
    model_generate(m, prompt, 8, gen, 40);
    puts("the cat ");
    tok_decode(gen, 40, text, 80);
    puts(text);
    putc('\n');

    model_grow(m, 0, why, 80);
    puts(why);
    putc('\n');

    puts("saving weights to the disk... ");
    {
        int r = store_model(m, 1);
        if (r < 0) puts("the disk did not answer\n");
        else {
            put_i(r);
            puts(" sectors\n");
        }
    }
    puts("ready. type a line.\n");
    puts("> ");

    for (;;) {
        int c = key_poll();
        if (!c) {
            __asm__ volatile("pause");
            continue;
        }
        if (c == '\b' || c == 127) {
            if (len) {
                len--;
                putc('\b');
                putc(' ');
                putc('\b');
            }
            continue;
        }
        if (c == '\n' || c == '\r') {
            putc('\n');
            line[len] = 0;
            do_line(m, line);
            len = 0;
            puts("> ");
            continue;
        }
        if (c >= 32 && c < 127 && len + 1 < 78) {
            line[len++] = (char)c;
            putc((char)c);
        }
    }
}
