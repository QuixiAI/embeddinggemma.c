#include "kernels.h"

#include <math.h>

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__AVX2__) || defined(__SSE2__)
#include <immintrin.h>
#endif

const char *ei_cpu_kernel_variant(void) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    return "cpu-neon";
#elif defined(__AVX2__)
    return "cpu-avx2";
#elif defined(__SSSE3__)
    return "cpu-ssse3";
#elif defined(__SSE2__)
    return "cpu-sse2";
#else
    return "cpu-scalar";
#endif
}

#if defined(__AVX2__)
static inline float ei_hsum_f32x8(__m256 v) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
    return _mm_cvtss_f32(sum);
}
#endif

#if defined(__SSE2__) && !defined(__AVX2__)
static inline float ei_hsum_f32x4(__m128 v) {
    v = _mm_add_ps(v, _mm_movehl_ps(v, v));
    v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
    return _mm_cvtss_f32(v);
}
#endif

static float ei_sum_squares(const float *x, int32_t n) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    int32_t i = 0;
    for (; i + 15 < n; i += 16) {
        float32x4_t x0 = vld1q_f32(x + i);
        float32x4_t x1 = vld1q_f32(x + i + 4);
        float32x4_t x2 = vld1q_f32(x + i + 8);
        float32x4_t x3 = vld1q_f32(x + i + 12);
        sum0 = vfmaq_f32(sum0, x0, x0);
        sum1 = vfmaq_f32(sum1, x1, x1);
        sum2 = vfmaq_f32(sum2, x2, x2);
        sum3 = vfmaq_f32(sum3, x3, x3);
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
    for (; i < n; i++) sum += x[i] * x[i];
    return sum;
#elif defined(__AVX2__)
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
    float sum = ei_hsum_f32x8(_mm256_add_ps(_mm256_add_ps(sum0, sum1),
                                             _mm256_add_ps(sum2, sum3)));
    for (; i < n; i++) sum += x[i] * x[i];
    return sum;
#elif defined(__SSE2__)
    __m128 sum0 = _mm_setzero_ps();
    __m128 sum1 = _mm_setzero_ps();
    int32_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m128 x0 = _mm_loadu_ps(x + i);
        __m128 x1 = _mm_loadu_ps(x + i + 4);
        sum0 = _mm_add_ps(sum0, _mm_mul_ps(x0, x0));
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(x1, x1));
    }
    float sum = ei_hsum_f32x4(_mm_add_ps(sum0, sum1));
    for (; i < n; i++) sum += x[i] * x[i];
    return sum;
#else
    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) sum += x[i] * x[i];
    return sum;
#endif
}

static void ei_norm_scale(const float *x, const float *w, int32_t n,
                          float scale, float *out) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    const float32x4_t scale4 = vdupq_n_f32(scale);
    int32_t i = 0;
    for (; i + 15 < n; i += 16) {
        vst1q_f32(out + i,      vmulq_f32(vmulq_f32(vld1q_f32(x + i),      vld1q_f32(w + i)),      scale4));
        vst1q_f32(out + i + 4,  vmulq_f32(vmulq_f32(vld1q_f32(x + i + 4),  vld1q_f32(w + i + 4)),  scale4));
        vst1q_f32(out + i + 8,  vmulq_f32(vmulq_f32(vld1q_f32(x + i + 8),  vld1q_f32(w + i + 8)),  scale4));
        vst1q_f32(out + i + 12, vmulq_f32(vmulq_f32(vld1q_f32(x + i + 12), vld1q_f32(w + i + 12)), scale4));
    }
    for (; i < n; i++) out[i] = x[i] * scale * w[i];
#elif defined(__AVX2__)
    const __m256 scale8 = _mm256_set1_ps(scale);
    int32_t i = 0;
    for (; i + 15 < n; i += 16) {
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i),
                                                               _mm256_loadu_ps(w + i)), scale8));
        _mm256_storeu_ps(out + i + 8, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i + 8),
                                                                   _mm256_loadu_ps(w + i + 8)), scale8));
    }
    for (; i < n; i++) out[i] = x[i] * scale * w[i];
#elif defined(__SSE2__)
    const __m128 scale4 = _mm_set1_ps(scale);
    int32_t i = 0;
    for (; i + 7 < n; i += 8) {
        _mm_storeu_ps(out + i, _mm_mul_ps(_mm_mul_ps(_mm_loadu_ps(x + i),
                                                      _mm_loadu_ps(w + i)), scale4));
        _mm_storeu_ps(out + i + 4, _mm_mul_ps(_mm_mul_ps(_mm_loadu_ps(x + i + 4),
                                                          _mm_loadu_ps(w + i + 4)), scale4));
    }
    for (; i < n; i++) out[i] = x[i] * scale * w[i];
#else
    for (int32_t i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
#endif
}

void ei_rms_norm_quantize_q8_0(const float *x, const float *w, int32_t n,
                               float eps, ei_block_q8_0 *out) {
    if (n % EI_QK != 0) {
        ei_die("fused RMS/Q8 length %d is not a multiple of %d", n, EI_QK);
    }
    float inv = 1.0f / sqrtf(ei_sum_squares(x, n) / (float)n + eps);
    for (int32_t block = 0; block < n / EI_QK; block++) {
        float normalized[EI_QK];
        ei_norm_scale(x + block * EI_QK, w + block * EI_QK,
                      EI_QK, inv, normalized);
        ei_quantize_row_q8_0(normalized, out + block, EI_QK);
    }
}

void ei_rms_norm(const float *x, const float *w, int32_t n, float eps, float *out) {
    const float scale = 1.0f / sqrtf(ei_sum_squares(x, n) / (float)n + eps);
    ei_norm_scale(x, w, n, scale, out);
}

void ei_rms_norm_inplace(float *x, const float *w, int32_t n, float eps) {
    const float scale = 1.0f / sqrtf(ei_sum_squares(x, n) / (float)n + eps);
    ei_norm_scale(x, w, n, scale, x);
}

void ei_rms_norm_residual_inplace(float *residual, const float *x, const float *w,
                                  int32_t n, float eps) {
    const float scale = 1.0f / sqrtf(ei_sum_squares(x, n) / (float)n + eps);
#if defined(__ARM_NEON) && defined(__aarch64__)
    const float32x4_t scale4 = vdupq_n_f32(scale);
    int32_t i = 0;
    for (; i + 7 < n; i += 8) {
        float32x4_t y0 = vmulq_f32(vmulq_f32(vld1q_f32(x + i), vld1q_f32(w + i)), scale4);
        float32x4_t y1 = vmulq_f32(vmulq_f32(vld1q_f32(x + i + 4), vld1q_f32(w + i + 4)), scale4);
        vst1q_f32(residual + i, vaddq_f32(vld1q_f32(residual + i), y0));
        vst1q_f32(residual + i + 4, vaddq_f32(vld1q_f32(residual + i + 4), y1));
    }
    for (; i < n; i++) residual[i] += x[i] * scale * w[i];
#elif defined(__AVX2__)
    const __m256 scale8 = _mm256_set1_ps(scale);
    int32_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 y = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i),
                                                _mm256_loadu_ps(w + i)), scale8);
        _mm256_storeu_ps(residual + i, _mm256_add_ps(_mm256_loadu_ps(residual + i), y));
    }
    for (; i < n; i++) residual[i] += x[i] * scale * w[i];
#else
    for (int32_t i = 0; i < n; i++) residual[i] += x[i] * scale * w[i];
#endif
}

void ei_rope_neox_inplace(float *x, int32_t n_heads, int32_t pos, float base) {
    for (int32_t k = 0; k < EI_HEAD_DIM / 2; k++) {
        const float theta = (float)pos * powf(base, -2.0f * (float)k / (float)EI_HEAD_DIM);
        const float c = cosf(theta);
        const float s = sinf(theta);
        for (int32_t h = 0; h < n_heads; h++) {
            float *head = x + h * EI_HEAD_DIM;
            const float x0 = head[k];
            const float x1 = head[k + EI_HEAD_DIM / 2];
            head[k] = x0 * c - x1 * s;
            head[k + EI_HEAD_DIM / 2] = x0 * s + x1 * c;
        }
    }
}

static void ei_vec_scale_inplace(float *x, float scale, int32_t n) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    const float32x4_t scale4 = vdupq_n_f32(scale);
    int32_t i = 0;
    for (; i + 7 < n; i += 8) {
        vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), scale4));
        vst1q_f32(x + i + 4, vmulq_f32(vld1q_f32(x + i + 4), scale4));
    }
    for (; i < n; i++) x[i] *= scale;
#elif defined(__AVX2__)
    const __m256 scale8 = _mm256_set1_ps(scale);
    int32_t i = 0;
    for (; i + 7 < n; i += 8) _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), scale8));
    for (; i < n; i++) x[i] *= scale;
#else
    for (int32_t i = 0; i < n; i++) x[i] *= scale;
#endif
}

void ei_qk_norm_rope_inplace(float *x, const float *w, int32_t n_heads,
                             int32_t pos, float base, float scale, float eps) {
    for (int32_t h = 0; h < n_heads; h++) {
        ei_rms_norm_inplace(x + h * EI_HEAD_DIM, w, EI_HEAD_DIM, eps);
    }
    ei_rope_neox_inplace(x, n_heads, pos, base);
    if (scale != 1.0f) ei_vec_scale_inplace(x, scale, n_heads * EI_HEAD_DIM);
}

void ei_qk_norm_rope_qk_inplace(float *q, float *k, const float *q_weight,
                                const float *k_weight, int32_t pos,
                                float base, float eps) {
    float q_inv[EI_N_HEAD];
    for (int32_t head = 0; head < EI_N_HEAD; head++) {
        q_inv[head] = 1.0f / sqrtf(
            ei_sum_squares(q + head * EI_HEAD_DIM, EI_HEAD_DIM)
                / (float)EI_HEAD_DIM + eps);
    }
    const float k_inv = 1.0f / sqrtf(
        ei_sum_squares(k, EI_HEAD_DIM) / (float)EI_HEAD_DIM + eps);

    for (int32_t dim = 0; dim < EI_HEAD_DIM / 2; dim++) {
        const float theta = (float)pos
            * powf(base, -2.0f * (float)dim / (float)EI_HEAD_DIM);
        const float c = cosf(theta);
        const float s = sinf(theta);
        for (int32_t head = 0; head < EI_N_HEAD; head++) {
            float *values = q + head * EI_HEAD_DIM;
            const float x0 = values[dim] * q_weight[dim] * q_inv[head];
            const float x1 = values[dim + EI_HEAD_DIM / 2]
                * q_weight[dim + EI_HEAD_DIM / 2] * q_inv[head];
            values[dim] = (x0 * c - x1 * s) * 0.0625f;
            values[dim + EI_HEAD_DIM / 2] = (x0 * s + x1 * c) * 0.0625f;
        }
        const float x0 = k[dim] * k_weight[dim] * k_inv;
        const float x1 = k[dim + EI_HEAD_DIM / 2]
            * k_weight[dim + EI_HEAD_DIM / 2] * k_inv;
        k[dim] = x0 * c - x1 * s;
        k[dim + EI_HEAD_DIM / 2] = x0 * s + x1 * c;
    }
}

static float ei_dot_f32(const float *a, const float *b, int32_t n) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    int32_t i = 0;
    for (; i + 7 < n; i += 8) {
        sum0 = vfmaq_f32(sum0, vld1q_f32(a + i), vld1q_f32(b + i));
        sum1 = vfmaq_f32(sum1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float sum = vaddvq_f32(vaddq_f32(sum0, sum1));
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#elif defined(__AVX2__)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    int32_t i = 0;
    for (; i + 15 < n; i += 16) {
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8)));
    }
    float sum = ei_hsum_f32x8(_mm256_add_ps(sum0, sum1));
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#elif defined(__SSE2__)
    __m128 sum = _mm_setzero_ps();
    int32_t i = 0;
    for (; i + 3 < n; i += 4) sum = _mm_add_ps(sum, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    float scalar = ei_hsum_f32x4(sum);
    for (; i < n; i++) scalar += a[i] * b[i];
    return scalar;
#else
    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
#endif
}

static void ei_axpy(float *dst, const float *src, float scale, int32_t n) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    int32_t i = 0;
    for (; i + 7 < n; i += 8) {
        vst1q_f32(dst + i, vfmaq_n_f32(vld1q_f32(dst + i), vld1q_f32(src + i), scale));
        vst1q_f32(dst + i + 4, vfmaq_n_f32(vld1q_f32(dst + i + 4), vld1q_f32(src + i + 4), scale));
    }
    for (; i < n; i++) dst[i] += scale * src[i];
#elif defined(__AVX2__)
    const __m256 scale8 = _mm256_set1_ps(scale);
    int32_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 add = _mm256_mul_ps(_mm256_loadu_ps(src + i), scale8);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i), add));
    }
    for (; i < n; i++) dst[i] += scale * src[i];
#else
    for (int32_t i = 0; i < n; i++) dst[i] += scale * src[i];
#endif
}

void ei_attention_mha_range(const float *q, const float *k, const float *v,
                            int32_t n_tokens, int32_t window,
                            int32_t query_begin, int32_t query_end,
                            float *ctx, float *scores) {
    memset(ctx + (size_t)query_begin * EI_N_EMBD, 0,
           sizeof(float) * (size_t)(query_end - query_begin) * EI_N_EMBD);
    const int32_t half_window = window > 0 ? window / 2 : 0;

    for (int32_t qt = query_begin; qt < query_end; qt++) {
        const int32_t first = window > 0 ? (qt - half_window > 0 ? qt - half_window : 0) : 0;
        const int32_t last = window > 0
            ? (qt + half_window + 1 < n_tokens ? qt + half_window + 1 : n_tokens)
            : n_tokens;
        for (int32_t h = 0; h < EI_N_HEAD; h++) {
            const float *qv = q + (size_t)qt * EI_N_EMBD + h * EI_HEAD_DIM;
            float max_score = -INFINITY;
            for (int32_t kt = first; kt < last; kt++) {
                const float score = ei_dot_f32(qv, k + (size_t)kt * EI_HEAD_DIM, EI_HEAD_DIM);
                scores[kt] = score;
                if (score > max_score) max_score = score;
            }

            float sum = 0.0f;
            for (int32_t kt = first; kt < last; kt++) {
                const float p = expf(scores[kt] - max_score);
                scores[kt] = p;
                sum += p;
            }

            const float inv_sum = 1.0f / sum;
            float *out = ctx + (size_t)qt * EI_N_EMBD + h * EI_HEAD_DIM;
            for (int32_t kt = first; kt < last; kt++) {
                ei_axpy(out, v + (size_t)kt * EI_HEAD_DIM, scores[kt] * inv_sum, EI_HEAD_DIM);
            }
        }
    }
}

void ei_attention_mha(const float *q, const float *k, const float *v,
                      int32_t n_tokens, int32_t window, float *ctx, float *scores) {
    ei_attention_mha_range(q, k, v, n_tokens, window, 0, n_tokens, ctx, scores);
}

float ei_gelu_tanh(float x) {
    const float a = 0.044715f;
    const float s = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + tanhf(s * x * (1.0f + a * x * x)));
}

void ei_gelu_mul_inplace(float *gate, const float *up, int32_t n) {
    for (int32_t i = 0; i < n; i++) gate[i] = ei_gelu_tanh(gate[i]) * up[i];
}

void ei_gelu_mul_quantize_q8_0(const float *gate, const float *up,
                               int32_t n, ei_block_q8_0 *out) {
    if (n % EI_QK != 0) {
        ei_die("fused GELU/Q8 length %d is not a multiple of %d", n, EI_QK);
    }
    for (int32_t block = 0; block < n / EI_QK; block++) {
        float activated[EI_QK];
        const float *gate_block = gate + block * EI_QK;
        const float *up_block = up + block * EI_QK;
        for (int32_t i = 0; i < EI_QK; i++) {
            activated[i] = ei_gelu_tanh(gate_block[i]) * up_block[i];
        }
        ei_quantize_row_q8_0(activated, out + block, EI_QK);
    }
}

void ei_l2_normalize(float *x, int32_t n) {
    const float ss = ei_sum_squares(x, n);
    if (ss != 0.0f) ei_vec_scale_inplace(x, 1.0f / sqrtf(ss), n);
}

bool ei_embedding_dimensions_supported(int32_t dimensions) {
    return dimensions == 128 || dimensions == 256 || dimensions == 512 ||
           dimensions == EI_N_EMBD;
}

bool ei_embedding_normalize_prefix(float embedding[EI_N_EMBD], int32_t dimensions) {
    if (!ei_embedding_dimensions_supported(dimensions)) return false;
    if (dimensions < EI_N_EMBD) ei_l2_normalize(embedding, dimensions);
    return true;
}

void ei_vec_add_inplace(float *dst, const float *src, int32_t n) {
    ei_axpy(dst, src, 1.0f, n);
}

void ei_vec_zero(float *x, int32_t n) {
    memset(x, 0, sizeof(float) * (size_t)n);
}

void ei_mean_pool_rms_l2(float *x, int32_t n_tokens, const float *w,
                         float eps, float out[EI_N_EMBD]) {
    ei_vec_zero(out, EI_N_EMBD);
    for (int32_t t = 0; t < n_tokens; t++) {
        float *row = x + (size_t)t * EI_N_EMBD;
        ei_rms_norm_inplace(row, w, EI_N_EMBD, eps);
        ei_vec_add_inplace(out, row, EI_N_EMBD);
    }
    ei_vec_scale_inplace(out, 1.0f / (float)n_tokens, EI_N_EMBD);
    ei_l2_normalize(out, EI_N_EMBD);
}
