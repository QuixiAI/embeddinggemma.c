/* AVX2/FMA/F16C CPU kernels and x86-64 runtime ISA selection.
 *
 * This is the only translation unit with AVX code. Every kernel below carries
 * EI_AVX2_TARGET rather than relying on -mavx2, so the Makefile keeps one
 * baseline CFLAGS for every file and nothing outside these functions (headers,
 * static inline helpers, initializers) is compiled for AVX2. Callers in
 * kernels.c and quants.c reach these only after ei_cpu_avx2_active(). */
#include "kernels_avx2.h"

#include <math.h>

#if defined(EI_X86_DISPATCH)
#include <cpuid.h>
#include <immintrin.h>

_Atomic int ei_cpu_isa_state = -1;

/* AVX2, FMA and F16C in CPUID, and YMM state enabled by the OS (XCR0). */
static bool ei_cpu_detect_avx2(void) {
    unsigned eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
    const unsigned fma = 1u << 12, osxsave = 1u << 27, avx = 1u << 28, f16c = 1u << 29;
    if ((ecx & (fma | osxsave | avx | f16c)) != (fma | osxsave | avx | f16c)) return false;
    unsigned xcr0_lo, xcr0_hi;
    __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    if ((xcr0_lo & 6u) != 6u) return false; /* XMM and YMM state */
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    return (ebx & (1u << 5)) != 0; /* AVX2 */
}

bool ei_cpu_avx2_supported(void) {
    return ei_cpu_detect_avx2();
}

static int ei_cpu_isa_from_name(const char *isa) {
    if (!isa || !*isa || strcmp(isa, "auto") == 0) return ei_cpu_detect_avx2() ? 1 : 0;
    if (strcmp(isa, "baseline") == 0 || strcmp(isa, "sse2") == 0) return 0;
    if (strcmp(isa, "avx2") == 0) return ei_cpu_detect_avx2() ? 1 : -1;
    return -1;
}

int ei_cpu_resolve_isa(void) {
    const char *env = getenv("EI_CPU_ISA");
    int state = ei_cpu_isa_from_name(env);
    if (state < 0) {
        ei_die("EI_CPU_ISA=%s is not supported here (use auto, baseline, or avx2 "
               "on an AVX2+FMA+F16C CPU)", env);
    }
    atomic_store_explicit(&ei_cpu_isa_state, state, memory_order_relaxed);
    return state;
}

bool ei_cpu_set_isa(const char *isa) {
    int state = ei_cpu_isa_from_name(isa);
    if (state < 0) return false;
    atomic_store_explicit(&ei_cpu_isa_state, state, memory_order_relaxed);
    return true;
}

#define EI_AVX2_TARGET __attribute__((target("avx2,fma,f16c")))

/* Hardware half -> float; exact, so it matches ei_fp16_to_fp32 bit for bit. */
EI_AVX2_TARGET static inline float ei_avx2_fp16(ei_fp16 h) {
    return _cvtsh_ss(h);
}

EI_AVX2_TARGET static inline float ei_avx2_hsum_f32x8(__m256 v) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
    return _mm_cvtss_f32(sum);
}

EI_AVX2_TARGET float ei_avx2_sum_squares(const float *x, int32_t n) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    int32_t i = 0;
    for (; i + 31 < n; i += 32) {
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 x2 = _mm256_loadu_ps(x + i + 16);
        __m256 x3 = _mm256_loadu_ps(x + i + 24);
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(x0, x0));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(x1, x1));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(x2, x2));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(x3, x3));
    }
    float sum = ei_avx2_hsum_f32x8(_mm256_add_ps(_mm256_add_ps(sum0, sum1),
                                                 _mm256_add_ps(sum2, sum3)));
    for (; i < n; i++) sum += x[i] * x[i];
    return sum;
}

EI_AVX2_TARGET void ei_avx2_norm_scale(const float *x, const float *w, int32_t n,
                                       float scale, float *out) {
    const __m256 scale8 = _mm256_set1_ps(scale);
    int32_t i = 0;
    for (; i + 15 < n; i += 16) {
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i),
                                                               _mm256_loadu_ps(w + i)), scale8));
        _mm256_storeu_ps(out + i + 8, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i + 8),
                                                                   _mm256_loadu_ps(w + i + 8)), scale8));
    }
    for (; i < n; i++) out[i] = x[i] * scale * w[i];
}

EI_AVX2_TARGET void ei_avx2_norm_residual(float *residual, const float *x, const float *w,
                                          int32_t n, float scale) {
    const __m256 scale8 = _mm256_set1_ps(scale);
    int32_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 y = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i),
                                                _mm256_loadu_ps(w + i)), scale8);
        _mm256_storeu_ps(residual + i, _mm256_add_ps(_mm256_loadu_ps(residual + i), y));
    }
    for (; i < n; i++) residual[i] += x[i] * scale * w[i];
}

EI_AVX2_TARGET void ei_avx2_vec_scale_inplace(float *x, float scale, int32_t n) {
    const __m256 scale8 = _mm256_set1_ps(scale);
    int32_t i = 0;
    for (; i + 7 < n; i += 8) _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), scale8));
    for (; i < n; i++) x[i] *= scale;
}

EI_AVX2_TARGET float ei_avx2_dot_f32(const float *a, const float *b, int32_t n) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    int32_t i = 0;
    for (; i + 15 < n; i += 16) {
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8)));
    }
    float sum = ei_avx2_hsum_f32x8(_mm256_add_ps(sum0, sum1));
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

EI_AVX2_TARGET void ei_avx2_axpy(float *dst, const float *src, float scale, int32_t n) {
    const __m256 scale8 = _mm256_set1_ps(scale);
    int32_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 add = _mm256_mul_ps(_mm256_loadu_ps(src + i), scale8);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i), add));
    }
    for (; i < n; i++) dst[i] += scale * src[i];
}

EI_AVX2_TARGET void ei_avx2_dequantize_row_q8_0(const ei_block_q8_0 *blocks,
                                                uint64_t row_blocks, float scale,
                                                float *out) {
    for (uint64_t b = 0; b < row_blocks; b++) {
        const __m256 d8 = _mm256_set1_ps(ei_avx2_fp16(blocks[b].d) * scale);
        for (int j = 0; j < EI_QK; j += 8) {
            const __m128i q8 = _mm_loadl_epi64((const __m128i *)(blocks[b].qs + j));
            const __m256i q32 = _mm256_cvtepi8_epi32(q8);
            _mm256_storeu_ps(out + b * EI_QK + (uint64_t)j,
                             _mm256_mul_ps(_mm256_cvtepi32_ps(q32), d8));
        }
    }
}

EI_AVX2_TARGET void ei_avx2_quantize_row_q8_0(const float *x, ei_block_q8_0 *out,
                                              int32_t nb) {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int32_t b = 0; b < nb; b++) {
        const float *xb = x + b * EI_QK;
        __m256 maxv = _mm256_setzero_ps();
        for (int j = 0; j < EI_QK; j += 8) {
            maxv = _mm256_max_ps(maxv, _mm256_and_ps(_mm256_loadu_ps(xb + j), abs_mask));
        }
        __m128 max4 = _mm_max_ps(_mm256_castps256_ps128(maxv), _mm256_extractf128_ps(maxv, 1));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_shuffle_ps(max4, max4, 1));
        const float amax = _mm_cvtss_f32(max4);
        const float d = amax / 127.0f;
        out[b].d = ei_fp32_to_fp16(d);
        if (amax == 0.0f) {
            memset(out[b].qs, 0, sizeof out[b].qs);
            continue;
        }
        const __m256 scale = _mm256_set1_ps(1.0f / d);
        __m256i quants[4];
        for (int j = 0; j < 4; j++) {
            __m256 values = _mm256_mul_ps(_mm256_loadu_ps(xb + j * 8), scale);
            __m256 signed_half = _mm256_or_ps(_mm256_and_ps(values, sign), half);
            quants[j] = _mm256_cvttps_epi32(_mm256_add_ps(values, signed_half));
        }
        __m256i q01 = _mm256_packs_epi32(quants[0], quants[1]);
        __m256i q23 = _mm256_packs_epi32(quants[2], quants[3]);
        __m256i q8 = _mm256_packs_epi16(q01, q23);
        _mm256_storeu_si256((__m256i *)out[b].qs, _mm256_permutevar8x32_epi32(q8, order));
    }
}

/* Signed 4-bit weights (low nibbles in lanes 0-15, high nibbles in 16-31) for
 * one Q4_0 block, matching the Q8_0 activation layout. */
EI_AVX2_TARGET static inline __m256i ei_avx2_unpack_q4_0(const ei_block_q4_0 *block) {
    const __m128i packed = _mm_loadu_si128((const __m128i *)block->qs);
    const __m256i q4 = _mm256_inserti128_si256(
        _mm256_castsi128_si256(packed), _mm_srli_epi16(packed, 4), 1);
    return _mm256_sub_epi8(_mm256_and_si256(q4, _mm256_set1_epi8(0x0F)),
                           _mm256_set1_epi8(8));
}

/* Eight int32 partial sums of the 32 int8 products, as floats. */
EI_AVX2_TARGET static inline __m256 ei_avx2_dot_i8x32(__m256i x, __m256i y) {
    const __m256i dot16 = _mm256_maddubs_epi16(_mm256_sign_epi8(x, x),
                                               _mm256_sign_epi8(y, x));
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(dot16, _mm256_set1_epi16(1)));
}

EI_AVX2_TARGET float ei_avx2_vec_dot_q4_0_q8_0(const ei_block_q4_0 *x,
                                               const ei_block_q8_0 *y, int32_t nb) {
    __m256 sum = _mm256_setzero_ps();
    for (int32_t b = 0; b < nb; b++) {
        const __m256i yq = _mm256_loadu_si256((const __m256i *)y[b].qs);
        const __m256 dot = ei_avx2_dot_i8x32(ei_avx2_unpack_q4_0(x + b), yq);
        const __m256 scale = _mm256_set1_ps(ei_avx2_fp16(x[b].d) *
                                            ei_avx2_fp16(y[b].d));
        sum = _mm256_fmadd_ps(dot, scale, sum);
    }
    return ei_avx2_hsum_f32x8(sum);
}

EI_AVX2_TARGET void ei_avx2_vec_dot_q4_0_q8_0_batch4(const ei_block_q4_0 *weights,
                                                     const ei_block_q8_0 *inputs,
                                                     int32_t input_stride, int32_t count,
                                                     int32_t nb, float out[4]) {
    __m256 sums[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
    };
    for (int32_t block = 0; block < nb; block++) {
        const __m256i weight_values = ei_avx2_unpack_q4_0(weights + block);
        const float weight_scale = ei_avx2_fp16(weights[block].d);
        for (int32_t input = 0; input < count; input++) {
            const ei_block_q8_0 *activation = inputs + input * input_stride + block;
            const __m256i values = _mm256_loadu_si256((const __m256i *)activation->qs);
            const __m256 scale = _mm256_set1_ps(
                weight_scale * ei_avx2_fp16(activation->d));
            sums[input] = _mm256_fmadd_ps(ei_avx2_dot_i8x32(weight_values, values),
                                          scale, sums[input]);
        }
    }
    for (int32_t input = 0; input < count; input++) out[input] = ei_avx2_hsum_f32x8(sums[input]);
}

EI_AVX2_TARGET void ei_avx2_vec_dot_q4_0_q8_0_dual(const ei_block_q4_0 *x0,
                                                   const ei_block_q4_0 *x1,
                                                   const ei_block_q8_0 *y, int32_t nb,
                                                   float *out0, float *out1) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    for (int32_t b = 0; b < nb; b++) {
        const __m256i yq = _mm256_loadu_si256((const __m256i *)y[b].qs);
        const float y_scale = ei_avx2_fp16(y[b].d);
        sum0 = _mm256_fmadd_ps(ei_avx2_dot_i8x32(ei_avx2_unpack_q4_0(x0 + b), yq),
                               _mm256_set1_ps(ei_avx2_fp16(x0[b].d) * y_scale), sum0);
        sum1 = _mm256_fmadd_ps(ei_avx2_dot_i8x32(ei_avx2_unpack_q4_0(x1 + b), yq),
                               _mm256_set1_ps(ei_avx2_fp16(x1[b].d) * y_scale), sum1);
    }
    *out0 = ei_avx2_hsum_f32x8(sum0);
    *out1 = ei_avx2_hsum_f32x8(sum1);
}

EI_AVX2_TARGET void ei_avx2_vec_dot_q4_0_q8_0_triple(const ei_block_q4_0 *x0,
                                                     const ei_block_q4_0 *x1,
                                                     const ei_block_q4_0 *x2,
                                                     const ei_block_q8_0 *y, int32_t nb,
                                                     float *out0, float *out1,
                                                     float *out2) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    for (int32_t b = 0; b < nb; b++) {
        const __m256i yq = _mm256_loadu_si256((const __m256i *)y[b].qs);
        const float y_scale = ei_avx2_fp16(y[b].d);
        sum0 = _mm256_fmadd_ps(ei_avx2_dot_i8x32(ei_avx2_unpack_q4_0(x0 + b), yq),
                               _mm256_set1_ps(ei_avx2_fp16(x0[b].d) * y_scale), sum0);
        sum1 = _mm256_fmadd_ps(ei_avx2_dot_i8x32(ei_avx2_unpack_q4_0(x1 + b), yq),
                               _mm256_set1_ps(ei_avx2_fp16(x1[b].d) * y_scale), sum1);
        sum2 = _mm256_fmadd_ps(ei_avx2_dot_i8x32(ei_avx2_unpack_q4_0(x2 + b), yq),
                               _mm256_set1_ps(ei_avx2_fp16(x2[b].d) * y_scale), sum2);
    }
    *out0 = ei_avx2_hsum_f32x8(sum0);
    *out1 = ei_avx2_hsum_f32x8(sum1);
    *out2 = ei_avx2_hsum_f32x8(sum2);
}

#else /* !EI_X86_DISPATCH: NEON and other targets keep compile-time kernels. */

bool ei_cpu_avx2_supported(void) {
    return false;
}

bool ei_cpu_set_isa(const char *isa) {
    return !isa || !*isa || strcmp(isa, "auto") == 0 || strcmp(isa, "baseline") == 0;
}

#endif
