/* x86-64 kernels for the EmbeddingGemma 2 fast CPU graph (runtime-dispatched). */
#ifndef EI_GEMMA2_FAST_X86_H
#define EI_GEMMA2_FAST_X86_H
#include "quants.h"
#include <stddef.h>
#include <stdint.h>

enum { G2X_NONE = 0, G2X_AVX2 = 1, G2X_AVXVNNI = 2, G2X_VNNI512 = 3 };
/* Highest usable tier on this CPU (EI_GEMMA2_CPU_X86 can force a lower one). */
int g2x_level(void);
/* AVX-512 VNNI repacked GEMM selected (EI_GEMMA2_CPU_X86=vnni256 disables it). */
int g2x_gemm512_enabled(void);

#if defined(__x86_64__) || defined(_M_X64)
void g2x_q8_tile(const ei_block_q8_0 *w, const ei_block_q8_0 *x, float *y, int kb, int m, int nt);
void g2x_quantize_row_q8_0(const float *x, ei_block_q8_0 *out, int n);
void g2x_dot4x4(const float *const a[4], const float *const b[4], int k, float out[4][4]);
float g2x_sum_squares(const float *x, int d);
void g2x_bf16_row(const uint16_t *src, float *dst, int k);
void g2x_gelu_mul_row(float *g, const float *u, int n);
/* Repacked weights (cached by key) and activations for g2x_gemm512_tile. */
const void *g2x_weight_of(const void *key, const ei_block_q8_0 *blocks, int m, int k);
void g2x_acts(const ei_block_q8_0 *xq, int t0, int t1, int k, uint8_t *xu, float *xs);
void g2x_gemm512_tile(const void *w, int r, const uint8_t *xu, const float *xs, int t, int nt,
                      int k, float *y, int m);
void g2x_pv(const float *v, size_t vstride, int nk, const float *p, size_t pstride, int nq, int d,
            float *out, size_t ostride);
#endif
#endif
