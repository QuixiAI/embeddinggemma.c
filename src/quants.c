#include "quants.h"

#include <math.h>

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#if defined(__AVX2__) || defined(__SSSE3__)
#include <immintrin.h>
#endif

ei_fp16 ei_fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof x);

    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mant = x & 0x007FFFFFu;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;

    if (exp <= 0) {
        if (exp < -10) return (ei_fp16)sign;
        mant |= 0x00800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = sign | (mant >> shift);
        if ((mant >> (shift - 1u)) & 1u) half++;
        return (ei_fp16)half;
    }
    if (exp >= 31) {
        if (mant == 0) return (ei_fp16)(sign | 0x7C00u);
        return (ei_fp16)(sign | 0x7C00u | (mant >> 13) | 1u);
    }

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x00001000u) half++;
    return (ei_fp16)half;
}

const char *ei_quants_kernel_variant(void) {
#if defined(__ARM_NEON) && defined(__aarch64__)
#if defined(__ARM_FEATURE_DOTPROD)
    return "cpu-neon-dotprod";
#else
    return "cpu-neon";
#endif
#elif defined(__AVX2__)
    return "cpu-avx2";
#elif defined(__SSSE3__)
    return "cpu-ssse3";
#else
    return "cpu-scalar";
#endif
}

void ei_dequantize_row_q8_0_scaled(const ei_tensor *t, int32_t row, float scale, float *out) {
    if (t->type != EI_T_Q8_0 || t->ne[0] % EI_QK != 0) {
        ei_die("bad q8_0 row dequant tensor");
    }
    if (row < 0 || (uint64_t)row >= t->ne[1]) {
        ei_die("q8_0 row %d out of range", row);
    }

    uint64_t row_blocks = t->ne[0] / EI_QK;
    const ei_block_q8_0 *blocks =
        (const ei_block_q8_0 *)((const uint8_t *)t->data + (uint64_t)row * ei_tensor_row_bytes(t));
    for (uint64_t b = 0; b < row_blocks; b++) {
        float d = ei_fp16_to_fp32(blocks[b].d) * scale;
#if defined(__ARM_NEON) && defined(__aarch64__)
        const float32x4_t d4 = vdupq_n_f32(d);
        for (int j = 0; j < EI_QK; j += 8) {
            const int16x8_t q16 = vmovl_s8(vld1_s8(blocks[b].qs + j));
            const float32x4_t q0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
            const float32x4_t q1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16)));
            vst1q_f32(out + b * EI_QK + (uint64_t)j, vmulq_f32(q0, d4));
            vst1q_f32(out + b * EI_QK + (uint64_t)j + 4, vmulq_f32(q1, d4));
        }
#elif defined(__AVX2__)
        const __m256 d8 = _mm256_set1_ps(d);
        for (int j = 0; j < EI_QK; j += 8) {
            const __m128i q8 = _mm_loadl_epi64((const __m128i *)(blocks[b].qs + j));
            const __m256i q32 = _mm256_cvtepi8_epi32(q8);
            _mm256_storeu_ps(out + b * EI_QK + (uint64_t)j,
                             _mm256_mul_ps(_mm256_cvtepi32_ps(q32), d8));
        }
#else
        for (int j = 0; j < EI_QK; j++) {
            out[b * EI_QK + (uint64_t)j] = d * (float)blocks[b].qs[j];
        }
#endif
    }
}

void ei_dequantize_row_q8_0(const ei_tensor *t, int32_t row, float *out) {
    ei_dequantize_row_q8_0_scaled(t, row, 1.0f, out);
}

void ei_quantize_row_q8_0(const float *x, ei_block_q8_0 *out, int32_t n) {
    if (n % EI_QK != 0) ei_die("q8_0 quantize length %d is not a multiple of 32", n);

    int32_t nb = n / EI_QK;
    for (int32_t b = 0; b < nb; b++) {
        const float *xb = x + b * EI_QK;
#if defined(__ARM_NEON) && defined(__aarch64__)
        float32x4_t max0 = vdupq_n_f32(0.0f);
        float32x4_t max1 = vdupq_n_f32(0.0f);
        for (int j = 0; j < EI_QK; j += 8) {
            max0 = vmaxq_f32(max0, vabsq_f32(vld1q_f32(xb + j)));
            max1 = vmaxq_f32(max1, vabsq_f32(vld1q_f32(xb + j + 4)));
        }
        float amax = vmaxvq_f32(vmaxq_f32(max0, max1));
#elif defined(__AVX2__)
        const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        __m256 maxv = _mm256_setzero_ps();
        for (int j = 0; j < EI_QK; j += 8) {
            maxv = _mm256_max_ps(maxv, _mm256_and_ps(_mm256_loadu_ps(xb + j), abs_mask));
        }
        __m128 max4 = _mm_max_ps(_mm256_castps256_ps128(maxv), _mm256_extractf128_ps(maxv, 1));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_shuffle_ps(max4, max4, 1));
        float amax = _mm_cvtss_f32(max4);
#else
        float amax = 0.0f;
        for (int j = 0; j < EI_QK; j++) {
            float a = fabsf(xb[j]);
            if (a > amax) amax = a;
        }
#endif
        float d = amax / 127.0f;
        out[b].d = ei_fp32_to_fp16(d);
        if (amax == 0.0f) {
            memset(out[b].qs, 0, sizeof out[b].qs);
            continue;
        }
        float id = 1.0f / d;
#if defined(__ARM_NEON) && defined(__aarch64__)
        const float32x4_t scale = vdupq_n_f32(id);
        const int32x4_t q0 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xb), scale));
        const int32x4_t q1 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xb + 4), scale));
        const int32x4_t q2 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xb + 8), scale));
        const int32x4_t q3 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xb + 12), scale));
        const int32x4_t q4 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xb + 16), scale));
        const int32x4_t q5 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xb + 20), scale));
        const int32x4_t q6 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xb + 24), scale));
        const int32x4_t q7 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xb + 28), scale));
        const int16x8_t q01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        const int16x8_t q23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
        const int16x8_t q45 = vcombine_s16(vqmovn_s32(q4), vqmovn_s32(q5));
        const int16x8_t q67 = vcombine_s16(vqmovn_s32(q6), vqmovn_s32(q7));
        vst1q_s8(out[b].qs, vcombine_s8(vqmovn_s16(q01), vqmovn_s16(q23)));
        vst1q_s8(out[b].qs + 16, vcombine_s8(vqmovn_s16(q45), vqmovn_s16(q67)));
#elif defined(__AVX2__)
        const __m256 scale = _mm256_set1_ps(id);
        const __m256 half = _mm256_set1_ps(0.5f);
        const __m256 sign = _mm256_set1_ps(-0.0f);
        __m256 values[4];
        __m256i quants[4];
        for (int j = 0; j < 4; j++) {
            values[j] = _mm256_mul_ps(_mm256_loadu_ps(xb + j * 8), scale);
            __m256 signed_half = _mm256_or_ps(_mm256_and_ps(values[j], sign), half);
            quants[j] = _mm256_cvttps_epi32(_mm256_add_ps(values[j], signed_half));
        }
        __m256i q01 = _mm256_packs_epi32(quants[0], quants[1]);
        __m256i q23 = _mm256_packs_epi32(quants[2], quants[3]);
        __m256i q8 = _mm256_packs_epi16(q01, q23);
        const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        _mm256_storeu_si256((__m256i *)out[b].qs,
                            _mm256_permutevar8x32_epi32(q8, order));
#else
        for (int j = 0; j < EI_QK; j++) {
            int q = (int)roundf(xb[j] * id);
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            out[b].qs[j] = (int8_t)q;
        }
#endif
    }
}

#if defined(__ARM_NEON) && defined(__aarch64__)
static inline int32x4_t ei_neon_dot_i8x16(int8x16_t a, int8x16_t b) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(vdupq_n_s32(0), a, b);
#else
    int16x8_t p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
#endif
}

static float ei_vec_dot_q4_0_q8_0_neon(const ei_block_q4_0 *x,
                                       const ei_block_q8_0 *y,
                                       int32_t nb) {
    const uint8x16_t low_mask = vdupq_n_u8(0x0Fu);
    const int8x16_t off = vdupq_n_s8(8);
    float32x4_t sum = vdupq_n_f32(0.0f);

    for (int32_t b = 0; b < nb; b++) {
        const uint8x16_t q4 = vld1q_u8(x[b].qs);
        const int8x16_t x0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q4, low_mask)), off);
        const int8x16_t x1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q4, 4)), off);
        const int8x16_t y0 = vld1q_s8(y[b].qs);
        const int8x16_t y1 = vld1q_s8(y[b].qs + 16);
        int32x4_t dot = vaddq_s32(ei_neon_dot_i8x16(x0, y0),
                                  ei_neon_dot_i8x16(x1, y1));
        float scale = ei_fp16_to_fp32(x[b].d) * ei_fp16_to_fp32(y[b].d);
        sum = vfmaq_n_f32(sum, vcvtq_f32_s32(dot), scale);
    }

    return vaddvq_f32(sum);
}
#endif

#if defined(__AVX2__) || defined(__SSSE3__)
static inline float ei_hsum_f32x4(__m128 v) {
    v = _mm_add_ps(v, _mm_movehl_ps(v, v));
    v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
    return _mm_cvtss_f32(v);
}

#if !defined(__AVX2__)
static inline __m128i ei_mul_sum_i8_pairs_sse(__m128i x, __m128i y) {
    const __m128i ax = _mm_sign_epi8(x, x);
    const __m128i sy = _mm_sign_epi8(y, x);
    const __m128i dot16 = _mm_maddubs_epi16(ax, sy);
    return _mm_madd_epi16(dot16, _mm_set1_epi16(1));
}
#endif

#if defined(__AVX2__)
static inline float ei_hsum_f32x8(__m256 v) {
    return ei_hsum_f32x4(_mm_add_ps(_mm256_castps256_ps128(v),
                                    _mm256_extractf128_ps(v, 1)));
}

static float ei_vec_dot_q4_0_q8_0_avx2(const ei_block_q4_0 *x,
                                       const ei_block_q8_0 *y,
                                       int32_t nb) {
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i off = _mm256_set1_epi8(8);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 sum = _mm256_setzero_ps();

    for (int32_t b = 0; b < nb; b++) {
        const __m128i q4_128 = _mm_loadu_si128((const __m128i *)x[b].qs);
        const __m256i q4 = _mm256_inserti128_si256(
            _mm256_castsi128_si256(q4_128), _mm_srli_epi16(q4_128, 4), 1);
        const __m256i xq = _mm256_sub_epi8(_mm256_and_si256(q4, low_mask), off);
        const __m256i yq = _mm256_loadu_si256((const __m256i *)y[b].qs);
        const __m256i ax = _mm256_sign_epi8(xq, xq);
        const __m256i sy = _mm256_sign_epi8(yq, xq);
        const __m256i dot16 = _mm256_maddubs_epi16(ax, sy);
        const __m256 dot = _mm256_cvtepi32_ps(_mm256_madd_epi16(dot16, ones));
        const __m256 scale = _mm256_set1_ps(ei_fp16_to_fp32(x[b].d) *
                                             ei_fp16_to_fp32(y[b].d));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(dot, scale));
    }

    return ei_hsum_f32x8(sum);
}
#endif

#if !defined(__AVX2__)
static float ei_vec_dot_q4_0_q8_0_ssse3(const ei_block_q4_0 *x,
                                        const ei_block_q8_0 *y,
                                        int32_t nb) {
    const __m128i low_mask = _mm_set1_epi8(0x0F);
    const __m128i off = _mm_set1_epi8(8);
    __m128 sum = _mm_setzero_ps();

    for (int32_t b = 0; b < nb; b++) {
        const __m128i q4 = _mm_loadu_si128((const __m128i *)x[b].qs);
        const __m128i x0 = _mm_sub_epi8(_mm_and_si128(q4, low_mask), off);
        const __m128i x1 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q4, 4), low_mask), off);
        const __m128i y0 = _mm_loadu_si128((const __m128i *)y[b].qs);
        const __m128i y1 = _mm_loadu_si128((const __m128i *)(y[b].qs + 16));
        __m128i sumv = _mm_add_epi32(ei_mul_sum_i8_pairs_sse(x0, y0),
                                     ei_mul_sum_i8_pairs_sse(x1, y1));
        __m128 scale = _mm_set1_ps(ei_fp16_to_fp32(x[b].d) *
                                    ei_fp16_to_fp32(y[b].d));
        sum = _mm_add_ps(sum, _mm_mul_ps(_mm_cvtepi32_ps(sumv), scale));
    }

    return ei_hsum_f32x4(sum);
}
#endif
#endif

float ei_vec_dot_q4_0_q8_0(const ei_block_q4_0 *x, const ei_block_q8_0 *y, int32_t n) {
    if (n % EI_QK != 0) ei_die("q4_0 dot length %d is not a multiple of 32", n);
    int32_t nb = n / EI_QK;
#if defined(__ARM_NEON) && defined(__aarch64__)
    return ei_vec_dot_q4_0_q8_0_neon(x, y, nb);
#elif defined(__AVX2__)
    return ei_vec_dot_q4_0_q8_0_avx2(x, y, nb);
#elif defined(__SSSE3__)
    return ei_vec_dot_q4_0_q8_0_ssse3(x, y, nb);
#else
    float sumf = 0.0f;
    for (int32_t b = 0; b < nb; b++) {
        int sumi = 0;
        for (int j = 0; j < 16; j++) {
            int v0 = (int)(x[b].qs[j] & 0x0Fu) - 8;
            int v1 = (int)(x[b].qs[j] >> 4) - 8;
            sumi += v0 * (int)y[b].qs[j];
            sumi += v1 * (int)y[b].qs[j + 16];
        }
        sumf += (float)sumi * ei_fp16_to_fp32(x[b].d) * ei_fp16_to_fp32(y[b].d);
    }
    return sumf;
#endif
}

static void ei_vec_dot_q4_0_q8_0_batch4(const ei_block_q4_0 *weights,
                                        const ei_block_q8_0 *inputs,
                                        int32_t input_stride, int32_t count,
                                        int32_t n, float out[4]) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    int32_t nb = n / EI_QK;
    const uint8x16_t low_mask = vdupq_n_u8(0x0Fu);
    const int8x16_t off = vdupq_n_s8(8);
    float32x4_t sums[4] = {
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
    };
    for (int32_t block = 0; block < nb; block++) {
        const uint8x16_t packed = vld1q_u8(weights[block].qs);
        const int8x16_t low = vsubq_s8(
            vreinterpretq_s8_u8(vandq_u8(packed, low_mask)), off);
        const int8x16_t high = vsubq_s8(
            vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), off);
        const float weight_scale = ei_fp16_to_fp32(weights[block].d);
        for (int32_t input = 0; input < count; input++) {
            const ei_block_q8_0 *activation = inputs + input * input_stride + block;
            const int32x4_t dot = vaddq_s32(
                ei_neon_dot_i8x16(low, vld1q_s8(activation->qs)),
                ei_neon_dot_i8x16(high, vld1q_s8(activation->qs + 16)));
            sums[input] = vfmaq_n_f32(
                sums[input], vcvtq_f32_s32(dot),
                weight_scale * ei_fp16_to_fp32(activation->d));
        }
    }
    for (int32_t input = 0; input < count; input++) out[input] = vaddvq_f32(sums[input]);
#elif defined(__AVX2__)
    int32_t nb = n / EI_QK;
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i off = _mm256_set1_epi8(8);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 sums[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
    };
    for (int32_t block = 0; block < nb; block++) {
        const __m128i packed = _mm_loadu_si128((const __m128i *)weights[block].qs);
        const __m256i q4 = _mm256_inserti128_si256(
            _mm256_castsi128_si256(packed), _mm_srli_epi16(packed, 4), 1);
        const __m256i weight_values = _mm256_sub_epi8(
            _mm256_and_si256(q4, low_mask), off);
        const float weight_scale = ei_fp16_to_fp32(weights[block].d);
        for (int32_t input = 0; input < count; input++) {
            const ei_block_q8_0 *activation = inputs + input * input_stride + block;
            const __m256i values = _mm256_loadu_si256(
                (const __m256i *)activation->qs);
            const __m256i dot16 = _mm256_maddubs_epi16(
                _mm256_sign_epi8(weight_values, weight_values),
                _mm256_sign_epi8(values, weight_values));
            const __m256 dot = _mm256_cvtepi32_ps(
                _mm256_madd_epi16(dot16, ones));
            const __m256 scale = _mm256_set1_ps(
                weight_scale * ei_fp16_to_fp32(activation->d));
            sums[input] = _mm256_add_ps(sums[input], _mm256_mul_ps(dot, scale));
        }
    }
    for (int32_t input = 0; input < count; input++) out[input] = ei_hsum_f32x8(sums[input]);
#else
    for (int32_t input = 0; input < count; input++) {
        out[input] = ei_vec_dot_q4_0_q8_0(
            weights, inputs + input * input_stride, n);
    }
#endif
}

void ei_vec_dot_q4_0_q8_0_dual(const ei_block_q4_0 *x0, const ei_block_q4_0 *x1,
                               const ei_block_q8_0 *y, int32_t n,
                               float *out0, float *out1) {
    if (n % EI_QK != 0) ei_die("dual q4_0 dot length %d is not a multiple of 32", n);
#if defined(__ARM_NEON) && defined(__aarch64__)
    int32_t nb = n / EI_QK;
    const uint8x16_t low_mask = vdupq_n_u8(0x0Fu);
    const int8x16_t off = vdupq_n_s8(8);
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    for (int32_t b = 0; b < nb; b++) {
        const int8x16_t y0 = vld1q_s8(y[b].qs);
        const int8x16_t y1 = vld1q_s8(y[b].qs + 16);
        const uint8x16_t packed0 = vld1q_u8(x0[b].qs);
        const uint8x16_t packed1 = vld1q_u8(x1[b].qs);
        const int8x16_t x00 = vsubq_s8(
            vreinterpretq_s8_u8(vandq_u8(packed0, low_mask)), off);
        const int8x16_t x01 = vsubq_s8(
            vreinterpretq_s8_u8(vshrq_n_u8(packed0, 4)), off);
        const int8x16_t x10 = vsubq_s8(
            vreinterpretq_s8_u8(vandq_u8(packed1, low_mask)), off);
        const int8x16_t x11 = vsubq_s8(
            vreinterpretq_s8_u8(vshrq_n_u8(packed1, 4)), off);
        const int32x4_t dot0 = vaddq_s32(ei_neon_dot_i8x16(x00, y0),
                                         ei_neon_dot_i8x16(x01, y1));
        const int32x4_t dot1 = vaddq_s32(ei_neon_dot_i8x16(x10, y0),
                                         ei_neon_dot_i8x16(x11, y1));
        const float y_scale = ei_fp16_to_fp32(y[b].d);
        sum0 = vfmaq_n_f32(sum0, vcvtq_f32_s32(dot0),
                           ei_fp16_to_fp32(x0[b].d) * y_scale);
        sum1 = vfmaq_n_f32(sum1, vcvtq_f32_s32(dot1),
                           ei_fp16_to_fp32(x1[b].d) * y_scale);
    }
    *out0 = vaddvq_f32(sum0);
    *out1 = vaddvq_f32(sum1);
#elif defined(__AVX2__)
    int32_t nb = n / EI_QK;
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i off = _mm256_set1_epi8(8);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    for (int32_t b = 0; b < nb; b++) {
        const __m256i yq = _mm256_loadu_si256((const __m256i *)y[b].qs);
        const float y_scale = ei_fp16_to_fp32(y[b].d);
        const ei_block_q4_0 *blocks[2] = { x0 + b, x1 + b };
        __m256 *sums[2] = { &sum0, &sum1 };
        for (int i = 0; i < 2; i++) {
            const __m128i packed = _mm_loadu_si128((const __m128i *)blocks[i]->qs);
            const __m256i q4 = _mm256_inserti128_si256(
                _mm256_castsi128_si256(packed), _mm_srli_epi16(packed, 4), 1);
            const __m256i xq = _mm256_sub_epi8(_mm256_and_si256(q4, low_mask), off);
            const __m256i dot16 = _mm256_maddubs_epi16(
                _mm256_sign_epi8(xq, xq), _mm256_sign_epi8(yq, xq));
            const __m256 dot = _mm256_cvtepi32_ps(_mm256_madd_epi16(dot16, ones));
            const __m256 scale = _mm256_set1_ps(
                ei_fp16_to_fp32(blocks[i]->d) * y_scale);
            *sums[i] = _mm256_add_ps(*sums[i], _mm256_mul_ps(dot, scale));
        }
    }
    *out0 = ei_hsum_f32x8(sum0);
    *out1 = ei_hsum_f32x8(sum1);
#else
    *out0 = ei_vec_dot_q4_0_q8_0(x0, y, n);
    *out1 = ei_vec_dot_q4_0_q8_0(x1, y, n);
#endif
}

void ei_vec_dot_q4_0_q8_0_triple(const ei_block_q4_0 *x0, const ei_block_q4_0 *x1,
                                 const ei_block_q4_0 *x2, const ei_block_q8_0 *y,
                                 int32_t n, float *out0, float *out1, float *out2) {
    if (n % EI_QK != 0) ei_die("triple q4_0 dot length %d is not a multiple of 32", n);
#if defined(__ARM_NEON) && defined(__aarch64__)
    int32_t nb = n / EI_QK;
    const uint8x16_t low_mask = vdupq_n_u8(0x0Fu);
    const int8x16_t off = vdupq_n_s8(8);
    float32x4_t sums[3] = {
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
    };
    for (int32_t b = 0; b < nb; b++) {
        const int8x16_t y0 = vld1q_s8(y[b].qs);
        const int8x16_t y1 = vld1q_s8(y[b].qs + 16);
        const float y_scale = ei_fp16_to_fp32(y[b].d);
        const ei_block_q4_0 *blocks[3] = { x0 + b, x1 + b, x2 + b };
        for (int i = 0; i < 3; i++) {
            const uint8x16_t packed = vld1q_u8(blocks[i]->qs);
            const int8x16_t low = vsubq_s8(
                vreinterpretq_s8_u8(vandq_u8(packed, low_mask)), off);
            const int8x16_t high = vsubq_s8(
                vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), off);
            const int32x4_t dot = vaddq_s32(ei_neon_dot_i8x16(low, y0),
                                             ei_neon_dot_i8x16(high, y1));
            sums[i] = vfmaq_n_f32(sums[i], vcvtq_f32_s32(dot),
                                  ei_fp16_to_fp32(blocks[i]->d) * y_scale);
        }
    }
    *out0 = vaddvq_f32(sums[0]);
    *out1 = vaddvq_f32(sums[1]);
    *out2 = vaddvq_f32(sums[2]);
#elif defined(__AVX2__)
    int32_t nb = n / EI_QK;
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i off = _mm256_set1_epi8(8);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 sums[3] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
    };
    for (int32_t b = 0; b < nb; b++) {
        const __m256i yq = _mm256_loadu_si256((const __m256i *)y[b].qs);
        const float y_scale = ei_fp16_to_fp32(y[b].d);
        const ei_block_q4_0 *blocks[3] = { x0 + b, x1 + b, x2 + b };
        for (int i = 0; i < 3; i++) {
            const __m128i packed = _mm_loadu_si128((const __m128i *)blocks[i]->qs);
            const __m256i q4 = _mm256_inserti128_si256(
                _mm256_castsi128_si256(packed), _mm_srli_epi16(packed, 4), 1);
            const __m256i xq = _mm256_sub_epi8(_mm256_and_si256(q4, low_mask), off);
            const __m256i dot16 = _mm256_maddubs_epi16(
                _mm256_sign_epi8(xq, xq), _mm256_sign_epi8(yq, xq));
            const __m256 dot = _mm256_cvtepi32_ps(_mm256_madd_epi16(dot16, ones));
            const __m256 scale = _mm256_set1_ps(
                ei_fp16_to_fp32(blocks[i]->d) * y_scale);
            sums[i] = _mm256_add_ps(sums[i], _mm256_mul_ps(dot, scale));
        }
    }
    *out0 = ei_hsum_f32x8(sums[0]);
    *out1 = ei_hsum_f32x8(sums[1]);
    *out2 = ei_hsum_f32x8(sums[2]);
#else
    *out0 = ei_vec_dot_q4_0_q8_0(x0, y, n);
    *out1 = ei_vec_dot_q4_0_q8_0(x1, y, n);
    *out2 = ei_vec_dot_q4_0_q8_0(x2, y, n);
#endif
}

void ei_matmul_q4_0_q8_0_rows(const ei_tensor *w, const ei_block_q8_0 *xq, float *out,
                              int32_t row_begin, int32_t row_end) {
    if (w->type != EI_T_Q4_0 || w->ne[0] % EI_QK != 0) {
        ei_die("bad q4_0 matmul tensor");
    }
    if (row_begin < 0 || row_end < row_begin || (uint64_t)row_end > w->ne[1]) {
        ei_die("bad q4_0 matmul row range [%d, %d)", row_begin, row_end);
    }
    uint64_t row_bytes = ei_tensor_row_bytes(w);
    for (int32_t row = row_begin; row < row_end; row++) {
        const ei_block_q4_0 *wr =
            (const ei_block_q4_0 *)((const uint8_t *)w->data + (uint64_t)row * row_bytes);
        out[row] = ei_vec_dot_q4_0_q8_0(wr, xq, (int32_t)w->ne[0]);
    }
}

void ei_matmul_q4_0_q8_0_rows3(const ei_tensor *w, const ei_block_q8_0 *xq, float *out,
                               int32_t row_begin, int32_t row_end) {
    if (w->type != EI_T_Q4_0 || w->ne[0] % EI_QK != 0 || row_begin < 0 ||
        row_end < row_begin || (uint64_t)row_end > w->ne[1]) {
        ei_die("bad q4_0 rows3 matmul contract");
    }
    uint64_t row_bytes = ei_tensor_row_bytes(w);
    int32_t row = row_begin;
    for (; row + 2 < row_end; row += 3) {
        const ei_block_q4_0 *weights0 = (const ei_block_q4_0 *)(
            (const uint8_t *)w->data + (uint64_t)row * row_bytes);
        const ei_block_q4_0 *weights1 = (const ei_block_q4_0 *)(
            (const uint8_t *)w->data + (uint64_t)(row + 1) * row_bytes);
        const ei_block_q4_0 *weights2 = (const ei_block_q4_0 *)(
            (const uint8_t *)w->data + (uint64_t)(row + 2) * row_bytes);
        ei_vec_dot_q4_0_q8_0_triple(weights0, weights1, weights2, xq,
                                    (int32_t)w->ne[0], out + row,
                                    out + row + 1, out + row + 2);
    }
    if (row < row_end) ei_matmul_q4_0_q8_0_rows(w, xq, out, row, row_end);
}

void ei_matmul_q4_0_q8_0_batch_rows(const ei_tensor *w,
                                    const ei_block_q8_0 *inputs,
                                    int32_t n_inputs, float *out,
                                    int32_t row_begin, int32_t row_end) {
    if (w->type != EI_T_Q4_0 || w->ne[0] % EI_QK != 0 || n_inputs < 1 ||
        row_begin < 0 || row_end < row_begin || (uint64_t)row_end > w->ne[1]) {
        ei_die("bad q4_0 batch matmul contract");
    }
    int32_t input_stride = (int32_t)w->ne[0] / EI_QK;
    int32_t output_stride = (int32_t)w->ne[1];
    uint64_t row_bytes = ei_tensor_row_bytes(w);
    for (int32_t row = row_begin; row < row_end; row++) {
        const ei_block_q4_0 *weights = (const ei_block_q4_0 *)(
            (const uint8_t *)w->data + (uint64_t)row * row_bytes);
        for (int32_t input = 0; input < n_inputs; input += 4) {
            int32_t count = n_inputs - input < 4 ? n_inputs - input : 4;
            float values[4];
            ei_vec_dot_q4_0_q8_0_batch4(
                weights, inputs + input * input_stride,
                input_stride, count, (int32_t)w->ne[0], values);
            for (int32_t item = 0; item < count; item++) {
                out[(size_t)(input + item) * output_stride + row] = values[item];
            }
        }
    }
}

void ei_matmul_q4_0_q8_0(const ei_tensor *w, const ei_block_q8_0 *xq, float *out) {
    ei_matmul_q4_0_q8_0_rows(w, xq, out, 0, (int32_t)w->ne[1]);
}

void ei_matmul_q4_0_q8_0_dual_rows(const ei_tensor *w0, const ei_tensor *w1,
                                   const ei_block_q8_0 *xq, float *out0, float *out1,
                                   int32_t row_begin, int32_t row_end) {
    if (w0->type != EI_T_Q4_0 || w1->type != EI_T_Q4_0 ||
        w0->ne[0] != w1->ne[0] || w0->ne[1] != w1->ne[1] ||
        w0->ne[0] % EI_QK != 0 || row_begin < 0 || row_end < row_begin ||
        (uint64_t)row_end > w0->ne[1]) {
        ei_die("bad dual q4_0 matmul contract");
    }
    uint64_t row_bytes0 = ei_tensor_row_bytes(w0);
    uint64_t row_bytes1 = ei_tensor_row_bytes(w1);
    for (int32_t row = row_begin; row < row_end; row++) {
        const ei_block_q4_0 *weights0 = (const ei_block_q4_0 *)(
            (const uint8_t *)w0->data + (uint64_t)row * row_bytes0);
        const ei_block_q4_0 *weights1 = (const ei_block_q4_0 *)(
            (const uint8_t *)w1->data + (uint64_t)row * row_bytes1);
        ei_vec_dot_q4_0_q8_0_dual(weights0, weights1, xq, (int32_t)w0->ne[0],
                                  out0 + row, out1 + row);
    }
}

void ei_matmul_q4_0_q8_0_triple_rows(const ei_tensor *w0, const ei_tensor *w1,
                                     const ei_tensor *w2, const ei_block_q8_0 *xq,
                                     float *out0, float *out1, float *out2,
                                     int32_t row_begin, int32_t row_end) {
    if (w0->type != EI_T_Q4_0 || w1->type != EI_T_Q4_0 || w2->type != EI_T_Q4_0 ||
        w0->ne[0] != w1->ne[0] || w0->ne[0] != w2->ne[0] ||
        w1->ne[1] != w2->ne[1] || w0->ne[0] % EI_QK != 0 ||
        row_begin < 0 || row_end < row_begin || (uint64_t)row_end > w1->ne[1] ||
        (uint64_t)row_end > w0->ne[1]) {
        ei_die("bad triple q4_0 matmul contract");
    }
    uint64_t row_bytes0 = ei_tensor_row_bytes(w0);
    uint64_t row_bytes1 = ei_tensor_row_bytes(w1);
    uint64_t row_bytes2 = ei_tensor_row_bytes(w2);
    for (int32_t row = row_begin; row < row_end; row++) {
        const ei_block_q4_0 *weights0 = (const ei_block_q4_0 *)(
            (const uint8_t *)w0->data + (uint64_t)row * row_bytes0);
        const ei_block_q4_0 *weights1 = (const ei_block_q4_0 *)(
            (const uint8_t *)w1->data + (uint64_t)row * row_bytes1);
        const ei_block_q4_0 *weights2 = (const ei_block_q4_0 *)(
            (const uint8_t *)w2->data + (uint64_t)row * row_bytes2);
        ei_vec_dot_q4_0_q8_0_triple(weights0, weights1, weights2, xq,
                                    (int32_t)w0->ne[0], out0 + row,
                                    out1 + row, out2 + row);
    }
}

void ei_matmul_q4_0_f32(const ei_tensor *w, const float *x, float *out, ei_block_q8_0 *tmp) {
    ei_quantize_row_q8_0(x, tmp, (int32_t)w->ne[0]);
    ei_matmul_q4_0_q8_0(w, tmp, out);
}
