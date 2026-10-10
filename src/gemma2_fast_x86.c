/* x86-64 kernels for the EmbeddingGemma 2 fast CPU graph, compiled per
 * function with target attributes and selected at run time, so default builds
 * stay generic. Tiers: AVX2 + FMA (maddubs/madd int8 dots) and VNNI
 * (vpdpbusd on 256-bit lanes; AVX-512 VL + VNNI or AVX-VNNI). Following
 * QuixiCore-CPU: the signed x signed dot uses |w| (unsigned) x x * sign(w)
 * (signed); negating INT8_MIN would wrap, but activations come from the
 * symmetric +-127 quantizer, so -128 never occurs in x. AVX-512 is promoted
 * per kernel by measurement, never blanket (perf/optimization_status.md).
 * EI_GEMMA2_CPU_X86=scalar|avx2|vnni256 forces a tier (default: the best;
 * on AVX-512 VNNI CPUs the repacked 512-bit GEMM). */
#include "gemma2_fast_x86.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define AVX2 __attribute__((target("avx2,fma,f16c")))
#define VNNI __attribute__((target("avx2,fma,f16c,avx512f,avx512vl,avx512dq,avx512vnni")))
#define AVXVNNI __attribute__((target("avx2,fma,f16c,avxvnni")))

static int wide512; /* repacked 512-bit VNNI GEMM (g2x_gemm512_tile) */
int g2x_level(void) {
    static int level = -1;
    if (level >= 0)
        return level;
    __builtin_cpu_init();
    int avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
               __builtin_cpu_supports("f16c");
    int vnni512 = avx2 && __builtin_cpu_supports("avx512vl") &&
                  __builtin_cpu_supports("avx512vnni");
    int avxvnni = avx2 && __builtin_cpu_supports("avxvnni");
    level = vnni512 ? G2X_VNNI512 : avxvnni ? G2X_AVXVNNI : avx2 ? G2X_AVX2 : G2X_NONE;
    wide512 = vnni512 && __builtin_cpu_supports("avx512dq");
    const char *force = getenv("EI_GEMMA2_CPU_X86");
    if (force && !strcmp(force, "scalar"))
        level = G2X_NONE, wide512 = 0;
    else if (force && !strcmp(force, "avx2") && avx2)
        level = G2X_AVX2, wide512 = 0;
    else if (force && !strcmp(force, "vnni256"))
        wide512 = 0; /* VNNI tier, 256-bit Q8 tile */
    return level;
}
int g2x_gemm512_enabled(void) {
    g2x_level();
    return wide512;
}

/* ---- Q8_0 x Q8_0: 4 weight rows x nt (1..4) tokens -------------------------- */

AVX2 static inline __m256i dot_maddubs(__m256i w, __m256i x) {
    __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
    return _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
}
VNNI static inline __m256i dot_vnni512(__m256i w, __m256i x) {
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_abs_epi8(w),
                               _mm256_sign_epi8(x, w));
}
AVXVNNI static inline __m256i dot_avxvnni(__m256i w, __m256i x) {
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), _mm256_abs_epi8(w),
                                   _mm256_sign_epi8(x, w));
}
static inline float half_float(ei_fp16 h) { return ei_fp16_to_fp32(h); }

/* Each weight block is loaded once and applied to every token of the tile;
 * fp32 lane accumulators stay live across K, one horizontal sum at the end. */
#define G2X_Q8_TILE(NAME, ATTR, DOT)                                                           \
    ATTR static void NAME(const ei_block_q8_0 *w, const ei_block_q8_0 *x, float *y, int kb,     \
                          int m, int nt) {                                                    \
        __m256 acc[4][4];                                                                     \
        for (int r = 0; r < 4; r++)                                                           \
            for (int t = 0; t < 4; t++)                                                       \
                acc[r][t] = _mm256_setzero_ps();                                              \
        for (int p = 0; p < kb; p++) {                                                        \
            __m256i wv[4];                                                                    \
            float dw[4];                                                                      \
            for (int r = 0; r < 4; r++) {                                                     \
                const ei_block_q8_0 *b = w + (size_t)r * kb + p;                              \
                wv[r] = _mm256_loadu_si256((const __m256i *)b->qs);                           \
                dw[r] = half_float(b->d);                                                     \
            }                                                                                 \
            for (int t = 0; t < nt; t++) {                                                    \
                const ei_block_q8_0 *xb = x + (size_t)t * kb + p;                             \
                __m256i xv = _mm256_loadu_si256((const __m256i *)xb->qs);                     \
                float dx = half_float(xb->d);                                                 \
                for (int r = 0; r < 4; r++)                                                   \
                    acc[r][t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(DOT(wv[r], xv)),           \
                                                _mm256_set1_ps(dw[r] * dx), acc[r][t]);       \
            }                                                                                 \
        }                                                                                     \
        for (int t = 0; t < nt; t++)                                                          \
            for (int r = 0; r < 4; r++) {                                                     \
                __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc[r][t]),                      \
                                      _mm256_extractf128_ps(acc[r][t], 1));                   \
                s = _mm_add_ps(s, _mm_movehl_ps(s, s));                                       \
                s = _mm_add_ss(s, _mm_movehdup_ps(s));                                        \
                y[(size_t)t * m + r] = _mm_cvtss_f32(s);                                      \
            }                                                                                 \
    }
G2X_Q8_TILE(q8_tile_avx2, AVX2, dot_maddubs)
G2X_Q8_TILE(q8_tile_vnni512, VNNI, dot_vnni512)
G2X_Q8_TILE(q8_tile_avxvnni, AVXVNNI, dot_avxvnni)

void g2x_q8_tile(const ei_block_q8_0 *w, const ei_block_q8_0 *x, float *y, int kb, int m, int nt) {
    switch (g2x_level()) {
    case G2X_VNNI512:
        q8_tile_vnni512(w, x, y, kb, m, nt);
        break;
    case G2X_AVXVNNI:
        q8_tile_avxvnni(w, x, y, kb, m, nt);
        break;
    default:
        q8_tile_avx2(w, x, y, kb, m, nt);
        break;
    }
}

/* ---- AVX-512 VNNI GEMM on repacked operands ---------------------------------
 * Weights: int8 rows [m][k] contiguous, fp32 scales [m][k/32], and per-block
 * corrections -128 * sum(w) (int32). Activations: u8 = q + 128 [n][k] with fp32
 * scales. vpdpbusd(u8, s8) then yields sum(q * w) + 128 * sum(w); seeding the
 * accumulator lane 0 of each 32-byte half with the correction removes the bias
 * with no per-element sign work. 512-bit lanes cover two Q8 blocks per op;
 * the two blocks' scales fill the two halves of the scale vector. */
typedef struct {
    const void *key;
    int8_t *q;
    float *s;
    int32_t *c;
} g2x_weight;
static g2x_weight g2x_weights[512];
static int g2x_weight_count;

const void *g2x_weight_of(const void *key, const ei_block_q8_0 *blocks, int m, int k) {
    for (int i = 0; i < g2x_weight_count; i++)
        if (g2x_weights[i].key == key)
            return &g2x_weights[i];
    if (g2x_weight_count == 512)
        return NULL;
    int kb = k / 32;
    g2x_weight *w = &g2x_weights[g2x_weight_count];
    w->q = malloc((size_t)m * k);
    w->s = malloc((size_t)m * kb * sizeof(float));
    w->c = malloc((size_t)m * kb * sizeof(int32_t));
    if (!w->q || !w->s || !w->c)
        return NULL;
    for (size_t b = 0; b < (size_t)m * kb; b++) {
        memcpy(w->q + b * 32, blocks[b].qs, 32);
        w->s[b] = ei_fp16_to_fp32(blocks[b].d);
        int32_t sum = 0;
        for (int z = 0; z < 32; z++)
            sum += blocks[b].qs[z];
        w->c[b] = -128 * sum;
    }
    w->key = key;
    g2x_weight_count++;
    return w;
}

/* Tokens [t0, t1): u8 = q + 128 (a sign-bit flip) and fp32 scales. */
AVX2 void g2x_acts(const ei_block_q8_0 *xq, int t0, int t1, int k, uint8_t *xu, float *xs) {
    int kb = k / 32;
    const __m256i flip = _mm256_set1_epi8((char)0x80);
    for (size_t b = (size_t)t0 * kb; b < (size_t)t1 * kb; b++) {
        __m256i q = _mm256_loadu_si256((const __m256i *)xq[b].qs);
        _mm256_storeu_si256((__m256i *)(xu + b * 32), _mm256_xor_si256(q, flip));
        xs[b] = ei_fp16_to_fp32(xq[b].d);
    }
}

/* 4 rows x nt (1..4) tokens; rows r..r+3 of weight w, tokens t..t+nt-1. */
VNNI void g2x_gemm512_tile(const void *wp, int r, const uint8_t *xu, const float *xs, int t,
                           int nt, int k, float *y, int m) {
    const g2x_weight *w = wp;
    const int kb = k / 32;
    __m512 acc[4][4];
    for (int a = 0; a < 4; a++)
        for (int b = 0; b < 4; b++)
            acc[a][b] = _mm512_setzero_ps();
    for (int p = 0; p < kb; p += 2) {
        __m512i wv[4], cv[4];
        __m512 sw[4];
        for (int a = 0; a < 4; a++) {
            size_t row = (size_t)(r + a);
            wv[a] = _mm512_loadu_si512(w->q + row * k + p * 32);
            const int32_t *c = w->c + row * kb + p;
            cv[a] = _mm512_mask_set1_epi32(_mm512_maskz_set1_epi32(0x0001, c[0]), 0x0100, c[1]);
            const float *sc = w->s + row * kb + p;
            sw[a] = _mm512_insertf32x8(_mm512_set1_ps(sc[0]), _mm256_set1_ps(sc[1]), 1);
        }
        for (int b = 0; b < nt; b++) {
            size_t tok = (size_t)(t + b);
            __m512i xv = _mm512_loadu_si512(xu + tok * k + p * 32);
            const float *xsc = xs + tok * kb + p;
            __m512 sx = _mm512_insertf32x8(_mm512_set1_ps(xsc[0]), _mm256_set1_ps(xsc[1]), 1);
            for (int a = 0; a < 4; a++) {
                __m512i dot = _mm512_dpbusd_epi32(cv[a], xv, wv[a]);
                acc[a][b] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), _mm512_mul_ps(sw[a], sx),
                                            acc[a][b]);
            }
        }
    }
    for (int b = 0; b < nt; b++)
        for (int a = 0; a < 4; a++)
            y[(size_t)(t + b) * m + r + a] = _mm512_reduce_add_ps(acc[a][b]);
}

/* ---- activation quantization (the repo's AVX2 rounding: trunc(v +- 0.5)) ---- */

AVX2 void g2x_quantize_row_q8_0(const float *x, ei_block_q8_0 *out, int n) {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 sign = _mm256_castsi256_ps(_mm256_set1_epi32((int)0x80000000u));
    const __m256 half = _mm256_set1_ps(0.5f);
    for (int b = 0; b < n / 32; b++) {
        const float *xb = x + b * 32;
        __m256 maxv = _mm256_setzero_ps();
        for (int j = 0; j < 32; j += 8)
            maxv = _mm256_max_ps(maxv, _mm256_and_ps(_mm256_loadu_ps(xb + j), abs_mask));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(maxv), _mm256_extractf128_ps(maxv, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        float amax = _mm_cvtss_f32(m4), d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        out[b].d = ei_fp32_to_fp16(d);
        __m256 scale = _mm256_set1_ps(id);
        __m256i q[4];
        for (int j = 0; j < 4; j++) {
            __m256 v = _mm256_mul_ps(_mm256_loadu_ps(xb + j * 8), scale);
            q[j] = _mm256_cvttps_epi32(_mm256_add_ps(v, _mm256_or_ps(_mm256_and_ps(v, sign), half)));
        }
        __m256i q8 = _mm256_packs_epi16(_mm256_packs_epi32(q[0], q[1]), _mm256_packs_epi32(q[2], q[3]));
        _mm256_storeu_si256((__m256i *)out[b].qs,
                            _mm256_permutevar8x32_epi32(q8, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
    }
}

/* ---- fp32 helpers ----------------------------------------------------------- */

AVX2 static inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}
/* out[r][c] = a[r] . b[c] over k (multiple of 8): 16 accumulators, 8 loads per
 * 16 FMAs. Serves the BF16 PLE projection and attention Q.K^T. */
AVX2 void g2x_dot4x4(const float *const a[4], const float *const b[4], int k, float out[4][4]) {
    __m256 acc[4][4];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            acc[r][c] = _mm256_setzero_ps();
    for (int z = 0; z < k; z += 8) {
        __m256 av[4], bv[4];
        for (int r = 0; r < 4; r++)
            av[r] = _mm256_loadu_ps(a[r] + z);
        for (int c = 0; c < 4; c++)
            bv[c] = _mm256_loadu_ps(b[c] + z);
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                acc[r][c] = _mm256_fmadd_ps(av[r], bv[c], acc[r][c]);
    }
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            out[r][c] = hsum(acc[r][c]);
}
AVX2 float g2x_sum_squares(const float *x, int d) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int z = 0;
    for (; z + 16 <= d; z += 16) {
        __m256 a = _mm256_loadu_ps(x + z), b = _mm256_loadu_ps(x + z + 8);
        s0 = _mm256_fmadd_ps(a, a, s0);
        s1 = _mm256_fmadd_ps(b, b, s1);
    }
    float s = hsum(_mm256_add_ps(s0, s1));
    for (; z < d; z++)
        s += x[z] * x[z];
    return s;
}
AVX2 void g2x_bf16_row(const uint16_t *src, float *dst, int k) {
    int z = 0;
    for (; z + 8 <= k; z += 8) {
        __m256i w = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(src + z)));
        _mm256_storeu_ps(dst + z, _mm256_castsi256_ps(_mm256_slli_epi32(w, 16)));
    }
    for (; z < k; z++) {
        uint32_t bits = (uint32_t)src[z] << 16;
        memcpy(dst + z, &bits, 4);
    }
}
/* exp via Cody-Waite range reduction and a degree-6 polynomial (as NEON exp4). */
AVX2 static inline __m256 exp8(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.0f)), _mm256_set1_ps(88.0f));
    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.0f / 720), one = _mm256_set1_ps(1.0f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 120));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 24));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 6));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, one);
    p = _mm256_fmadd_ps(p, r, one);
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}
/* g = gelu_tanh(g) * u (u NULL: gelu only); tanh(v) = 1 - 2 / (exp(2v) + 1). */
AVX2 void g2x_gelu_mul_row(float *g, const float *u, int n) {
    const __m256 s = _mm256_set1_ps(0.79788456080286535588f), a = _mm256_set1_ps(0.044715f);
    const __m256 half = _mm256_set1_ps(0.5f), one = _mm256_set1_ps(1.0f), two = _mm256_set1_ps(2.0f);
    int z = 0;
    for (; z + 8 <= n; z += 8) {
        __m256 x = _mm256_loadu_ps(g + z);
        __m256 inner = _mm256_mul_ps(_mm256_mul_ps(s, x), _mm256_fmadd_ps(_mm256_mul_ps(a, x), x, one));
        inner = _mm256_min_ps(_mm256_max_ps(inner, _mm256_set1_ps(-9.0f)), _mm256_set1_ps(9.0f));
        __m256 t = _mm256_sub_ps(one, _mm256_div_ps(two, _mm256_add_ps(exp8(_mm256_mul_ps(two, inner)), one)));
        __m256 gelu = _mm256_mul_ps(_mm256_mul_ps(half, x), _mm256_add_ps(one, t));
        _mm256_storeu_ps(g + z, u ? _mm256_mul_ps(gelu, _mm256_loadu_ps(u + z)) : gelu);
    }
    for (; z < n; z++) {
        float x = g[z];
        float gl = 0.5f * x * (1.0f + tanhf(0.79788456080286535588f * x * (1.0f + 0.044715f * x * x)));
        g[z] = gl * (u ? u[z] : 1.0f);
    }
}
/* out_i[z..z+16) = sum_s p_i[s] v_s[z..] for nq queries, 16 dims at a time:
 * v rows at v + s * vstride, probabilities at p + i * pstride, outputs at
 * out + i * ostride. d must be a multiple of 16. */
AVX2 void g2x_pv(const float *v, size_t vstride, int nk, const float *p, size_t pstride, int nq,
                 int d, float *out, size_t ostride) {
    for (int z = 0; z < d; z += 16) {
        __m256 o[8][2];
        for (int i = 0; i < 8; i++)
            o[i][0] = o[i][1] = _mm256_setzero_ps();
        for (int s = 0; s < nk; s++) {
            const float *vr = v + (size_t)s * vstride + z;
            __m256 v0 = _mm256_loadu_ps(vr), v1 = _mm256_loadu_ps(vr + 8);
            for (int i = 0; i < nq; i++) {
                __m256 pr = _mm256_set1_ps(p[(size_t)i * pstride + s]);
                o[i][0] = _mm256_fmadd_ps(v0, pr, o[i][0]);
                o[i][1] = _mm256_fmadd_ps(v1, pr, o[i][1]);
            }
        }
        for (int i = 0; i < nq; i++) {
            _mm256_storeu_ps(out + (size_t)i * ostride + z, o[i][0]);
            _mm256_storeu_ps(out + (size_t)i * ostride + z + 8, o[i][1]);
        }
    }
}
#else
int g2x_level(void) { return G2X_NONE; }
int g2x_gemm512_enabled(void) { return 0; }
#endif
