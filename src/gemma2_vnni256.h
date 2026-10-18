/* Repacked 256-bit AVX-VNNI Q8_0 x Q8_0 GEMM (CPUs with AVX-VNNI
 * but without AVX-512, e.g. Intel Alder Lake / Raptor Lake hybrid parts). */
#ifndef EI_GEMMA2_VNNI256_H
#define EI_GEMMA2_VNNI256_H
#include "quants.h"
#include "parallel.h"
#include <stddef.h>
#include <stdint.h>

/* 1 when the kernel can run here and EI_GEMMA2_CPU_VNNI256 != 0. */
int g2v_enabled(void);
/* Repacked weights for `key` (cached; NULL if m % 8 or k % 32 or the cache is full). */
const void *g2v_weight_of(const void *key, const ei_block_q8_0 *blocks, int m, int k, ei_thread_pool *pool);
/* y[t][8g .. 8g+7] for tokens [t0, t0 + nt), nt in 1..G2V_TILE_TOKENS; activations as
 * produced by g2x_acts (u8 = q ^ 0x80 rows of k bytes, fp32 block scales). */
enum { G2V_TILE_TOKENS = 4 };
void g2v_tile(const void *w, int g, const uint8_t *xu, const float *xs, int t0, int nt, int k, float *y,
              int m);
#endif
