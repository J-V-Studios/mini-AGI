#include "mem.h"

#include <stdint.h>

#ifdef HOST
static unsigned char raw[16 * 1024 * 1024];
static unsigned char *hp;
static unsigned char *base;
#else
extern char __bss_end[];
static unsigned char *hp;
static unsigned char *base;
#endif

void mem_init(void) {
#ifdef HOST
    hp = raw;
#else
    uintptr_t p = ((uintptr_t)__bss_end + 255u) & ~(uintptr_t)255u;
    if (p < 0x400000u) p = 0x400000u;
    hp = (unsigned char *)p;
#endif
    base = hp;
}

void mem_reset(void) {
    hp = base;
}

void *kalloc(size_t n) {
    unsigned char *p;
    size_t i;
    n = (n + 63u) & ~(size_t)63u;
    p = hp;
    hp += n;
    for (i = 0; i < n; i++) p[i] = 0;
    return p;
}

#ifndef HOST
void *memcpy(void *dst, const void *src, size_t n) {
    unsigned char *d = dst;
    const unsigned char *s = src;
    while (n--) *d++ = *s++;
    return dst;
}

void *memset(void *dst, int c, size_t n) {
    unsigned char *d = dst;
    while (n--) *d++ = (unsigned char)c;
    return dst;
}

void *memmove(void *dst, const void *src, size_t n) {
    unsigned char *d = dst;
    const unsigned char *s = src;
    if (d == s || n == 0) return dst;
    if (d < s) {
        while (n--) *d++ = *s++;
    } else {
        d += n;
        s += n;
        while (n--) *--d = *--s;
    }
    return dst;
}
#endif
