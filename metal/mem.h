#ifndef METAL_MEM_H
#define METAL_MEM_H

#include <stddef.h>

void mem_init(void);
void mem_reset(void);
void *kalloc(size_t n);

void *memcpy(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);
void *memmove(void *dst, const void *src, size_t n);

#endif
