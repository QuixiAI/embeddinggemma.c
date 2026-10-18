/* x86-64 runtime CPU dispatch for the AVX2/FMA/F16C CPU kernels.
 *
 * Portable x86-64 builds use the compiler baseline (SSE2). The AVX2 kernels
 * live in kernels_avx2.c, where each function carries
 * __attribute__((target("avx2,fma,f16c"))), so no other code is compiled with
 * AVX instructions. Callers reach them only through ei_cpu_avx2_active(). */
#ifndef EI_KERNELS_AVX2_H
#define EI_KERNELS_AVX2_H

#include "quants.h"

#include <stdatomic.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define EI_X86_DISPATCH 1

/* -1 until first use, then 0 (baseline) or 1 (AVX2 + FMA + F16C). */
extern _Atomic int ei_cpu_isa_state;
int ei_cpu_resolve_isa(void);

/* True when the CPU and OS support AVX2, FMA and F16C, unless EI_CPU_ISA=baseline
 * (or a test) forced the SSE2 route. A relaxed load: cheap per kernel call. */
static inline bool ei_cpu_avx2_active(void) {
    int state = atomic_load_explicit(&ei_cpu_isa_state, memory_order_relaxed);
    if (state < 0) state = ei_cpu_resolve_isa();
    return state == 1;
}

float ei_avx2_sum_squares(const float *x, int32_t n);
void ei_avx2_norm_scale(const float *x, const float *w, int32_t n,
                        float scale, float *out);
void ei_avx2_norm_residual(float *residual, const float *x, const float *w,
                           int32_t n, float scale);
void ei_avx2_vec_scale_inplace(float *x, float scale, int32_t n);
float ei_avx2_dot_f32(const float *a, const float *b, int32_t n);
void ei_avx2_axpy(float *dst, const float *src, float scale, int32_t n);

void ei_avx2_dequantize_row_q8_0(const ei_block_q8_0 *blocks, uint64_t row_blocks,
                                 float scale, float *out);
void ei_avx2_quantize_row_q8_0(const float *x, ei_block_q8_0 *out, int32_t nb);
float ei_avx2_vec_dot_q4_0_q8_0(const ei_block_q4_0 *x, const ei_block_q8_0 *y,
                                int32_t nb);
void ei_avx2_vec_dot_q4_0_q8_0_batch4(const ei_block_q4_0 *weights,
                                      const ei_block_q8_0 *inputs,
                                      int32_t input_stride, int32_t count,
                                      int32_t nb, float out[4]);
void ei_avx2_vec_dot_q4_0_q8_0_dual(const ei_block_q4_0 *x0, const ei_block_q4_0 *x1,
                                    const ei_block_q8_0 *y, int32_t nb,
                                    float *out0, float *out1);
void ei_avx2_vec_dot_q4_0_q8_0_triple(const ei_block_q4_0 *x0, const ei_block_q4_0 *x1,
                                      const ei_block_q4_0 *x2, const ei_block_q8_0 *y,
                                      int32_t nb, float *out0, float *out1,
                                      float *out2);

/* Runs `call` (which must return) when the AVX2 route is active. */
#define EI_DISPATCH_AVX2(call) do { if (ei_cpu_avx2_active()) { call; } } while (0)
#else
#define EI_DISPATCH_AVX2(call) do { } while (0)
#endif

/* Select the x86 kernel family: "auto" (default), "baseline" ("sse2"), or
 * "avx2". Returns false for an unknown name or AVX2 on a CPU without it.
 * Non-x86 builds accept only "auto" and "baseline". On x86-64 the EI_CPU_ISA
 * environment variable sets the initial choice. Not for use while kernels run
 * on other threads. */
bool ei_cpu_set_isa(const char *isa);
/* True when this process can run the AVX2 route (x86-64 with AVX2, FMA and F16C). */
bool ei_cpu_avx2_supported(void);

#endif /* EI_KERNELS_AVX2_H */
