/* cpu-max fork: packed fp32 GEMM for the float-weight projections of the
 * EmbeddingGemma 2 CPU graphs (BF16 per-layer-input projection, F32 vision
 * patch embedding and audio input projection). x86 AVX2+FMA only. */
#ifndef EI_GEMMA2_SGEMM_X86_H
#define EI_GEMMA2_SGEMM_X86_H
#include "parallel.h"
#include <stddef.h>
#include <stdint.h>

/* 1 when the kernel can run here and EI_GEMMA2_CPU_SGEMM != 0. */
int g2s_enabled(void);
/* Weights [m][k] of the given ei_tensor_type (F32, F16 or BF16), expanded once
 * (exactly) into 16-column fp32 panels and cached by `key`. NULL if m % 16. */
const void *g2s_weight_of(const void *key, const void *data, int type, int m, int k, ei_thread_pool *pool);
/* y[t][m0 .. m0 + 16*np) for tokens [t0, t1), panels [p0, p0 + np). */
void g2s_block(const void *w, const float *x, int t0, int t1, int p0, int np, float *y);
int g2s_m(const void *w);
int g2s_k(const void *w);
#endif
