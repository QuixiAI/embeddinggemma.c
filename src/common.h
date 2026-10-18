/* embedding-inference: shared basics. C11, no external dependencies. */
#ifndef EI_COMMON_H
#define EI_COMMON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* All supported targets are little-endian; the GGUF format is little-endian. */

typedef uint16_t ei_fp16;

/* Scalar IEEE 754 half -> float. Handles subnormals, inf, nan. */
static inline float ei_fp16_to_fp32(ei_fp16 h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign; /* +/- 0 */
        } else {
            /* subnormal: normalize */
            int e = -1;
            uint32_t m = man;
            do { m <<= 1; e++; } while (!(m & 0x400u));
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (man << 13); /* inf / nan */
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

#ifdef _WIN32
__attribute__((format(gnu_printf, 1, 2), noreturn))
#else
__attribute__((format(printf, 1, 2), noreturn))
#endif
static inline void ei_die(const char *fmt, ...);

#include <stdarg.h>
static inline void ei_die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("embedding-inference: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(1);
}

static inline void *ei_xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) ei_die("out of memory");
    return p;
}

static inline void *ei_xcalloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p) ei_die("out of memory");
    return p;
}

static inline void *ei_xrealloc(void *p, size_t n) {
    void *r = realloc(p, n ? n : 1);
    if (!r) ei_die("out of memory");
    return r;
}

/* Aligned allocation for SIMD weight and activation panels (alignment a power
 * of two, at least sizeof(void *)); free with ei_aligned_free. C11
 * aligned_alloc wants a size that is a multiple of the alignment, so the size
 * is rounded up; the UCRT has no aligned_alloc, so Windows uses _aligned_malloc. */
#ifdef _WIN32
#include <malloc.h>
static inline void *ei_aligned_alloc(size_t align, size_t n) {
    void *p = _aligned_malloc(n ? n : 1, align);
    if (!p) ei_die("out of memory");
    return p;
}
static inline void ei_aligned_free(void *p) { _aligned_free(p); }
#else
static inline void *ei_aligned_alloc(size_t align, size_t n) {
    size_t rounded = (n + align - 1) / align * align;
    void *p = aligned_alloc(align, rounded ? rounded : align);
    if (!p) ei_die("out of memory");
    return p;
}
static inline void ei_aligned_free(void *p) { free(p); }
#endif

#endif /* EI_COMMON_H */
