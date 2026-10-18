#if defined(__linux__)
#define _DEFAULT_SOURCE /* madvise under -std=c11 (gemma2_release.h) */
#endif
/* Packed fp32 GEMM, y[n][m] = x[n][k] . w[m][k]^T.
 *
 * The row float-weight GEMM (gemma2_fast.c gemm_f, f_rows) converts four weight
 * rows to fp32 per task on every call and computes 4x4 dot tiles that end in
 * 16 horizontal reductions. Here the weight is expanded once (BF16/F16 -> fp32
 * is exact) into panels of 16 output columns, wp[m/16][k][16], and a 6-token x
 * 16-output register tile (12 ymm accumulators) streams one panel row per k:
 * 2 loads + 6 broadcasts per 12 FMAs, no reductions. fp32 throughout; only
 * the summation order differs from the reference. */
#include "gemma2_sgemm_x86.h"
#include "gemma2_fast_x86.h"
#include "gemma2_release.h"
#include "quants.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    const void *key;
    int m, k;
    float *p; /* [m/16][k][16] */
} g2s_weight;

int g2s_m(const void *w) { return ((const g2s_weight *)w)->m; }
int g2s_k(const void *w) { return ((const g2s_weight *)w)->k; }

#if defined(__x86_64__) || defined(_M_X64)
enum { G2S_CACHE = 64 };
static g2s_weight g2s_weights[G2S_CACHE];
static int g2s_count;
#include <immintrin.h>
#define AVX2 __attribute__((target("avx2,fma,f16c")))
#define UNROLL _Pragma("GCC unroll 8")

int g2s_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *flag = getenv("EI_GEMMA2_CPU_SGEMM"); /* default on where supported */
        cached = (!flag || strcmp(flag, "0")) && g2x_level() >= G2X_AVX2;
    }
    return cached;
}

static float element(const void *data, int type, size_t i) {
    if (type == EI_T_F32)
        return ((const float *)data)[i];
    if (type == EI_T_F16)
        return ei_fp16_to_fp32(((const ei_fp16 *)data)[i]);
    uint32_t bits = (uint32_t)((const uint16_t *)data)[i] << 16; /* BF16 */
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

typedef struct {
    float *p;
    const void *data;
    int type, k;
} g2s_pack_job;
static void g2s_pack_panels(void *opaque, int32_t begin, int32_t end) {
    const g2s_pack_job *j = opaque;
    for (int pn = begin; pn < end; pn++)
        for (int z = 0; z < j->k; z++)
            for (int c = 0; c < 16; c++)
                j->p[((size_t)pn * j->k + z) * 16 + c] = element(j->data, j->type, (size_t)(pn * 16 + c) * j->k + z);
}

const void *g2s_weight_of(const void *key, const void *data, int type, int m, int k, ei_thread_pool *pool) {
    for (int i = 0; i < g2s_count; i++)
        if (g2s_weights[i].key == key)
            return g2s_weights[i].m == m && g2s_weights[i].k == k ? &g2s_weights[i] : NULL;
    if (g2s_count == G2S_CACHE || m % 16 || (type != EI_T_F32 && type != EI_T_F16 && type != EI_T_BF16))
        return NULL;
    g2s_weight *w = &g2s_weights[g2s_count];
    w->p = ei_aligned_alloc(64, (size_t)m * k * sizeof(float));
    g2s_pack_job job = {w->p, data, type, k};
    ei_parallel_for(pool, m / 16, 4, g2s_pack_panels, &job); /* on the caller's pool (first use) */
    g2_release_source(data, (size_t)m * k * (type == EI_T_F32 ? 4 : 2));
    w->key = key;
    w->m = m;
    w->k = k;
    g2s_count++;
    return w;
}

/* g2x_ prefix: scripts/check-avx2-isolation.py's dispatched set. */
#define G2S_TILE(NT)                                                                               \
    AVX2 static void g2x_sgemm_tile##NT(const float *panel, const float *x, int k, float *y, int m) {          \
        __m256 acc[NT][2];                                                                         \
        UNROLL for (int t = 0; t < NT; t++) acc[t][0] = acc[t][1] = _mm256_setzero_ps();           \
        _Pragma("GCC unroll 2") for (int z = 0; z < k; z++) {                                      \
            __m256 b0 = _mm256_load_ps(panel + (size_t)z * 16), b1 = _mm256_load_ps(panel + (size_t)z * 16 + 8);\
            UNROLL for (int t = 0; t < NT; t++) {                                                  \
                __m256 a = _mm256_broadcast_ss(x + (size_t)t * k + z);                             \
                acc[t][0] = _mm256_fmadd_ps(a, b0, acc[t][0]);                                     \
                acc[t][1] = _mm256_fmadd_ps(a, b1, acc[t][1]);                                     \
            }                                                                                      \
        }                                                                                          \
        UNROLL for (int t = 0; t < NT; t++) {                                                      \
            _mm256_storeu_ps(y + (size_t)t * m, acc[t][0]);                                        \
            _mm256_storeu_ps(y + (size_t)t * m + 8, acc[t][1]);                                    \
        }                                                                                          \
    }
G2S_TILE(1)
G2S_TILE(2)
G2S_TILE(3)
G2S_TILE(4)
G2S_TILE(5)
G2S_TILE(6)

void g2s_block(const void *wp, const float *x, int t0, int t1, int p0, int np, float *y) {
    const g2s_weight *w = wp;
    for (int pn = p0; pn < p0 + np; pn++) {
        const float *panel = w->p + (size_t)pn * w->k * 16;
        for (int t = t0; t < t1; t += 6) {
            const float *xt = x + (size_t)t * w->k;
            float *yt = y + (size_t)t * w->m + pn * 16;
            switch (t1 - t < 6 ? t1 - t : 6) {
            case 1: g2x_sgemm_tile1(panel, xt, w->k, yt, w->m); break;
            case 2: g2x_sgemm_tile2(panel, xt, w->k, yt, w->m); break;
            case 3: g2x_sgemm_tile3(panel, xt, w->k, yt, w->m); break;
            case 4: g2x_sgemm_tile4(panel, xt, w->k, yt, w->m); break;
            case 5: g2x_sgemm_tile5(panel, xt, w->k, yt, w->m); break;
            default: g2x_sgemm_tile6(panel, xt, w->k, yt, w->m); break;
            }
        }
    }
}
#else
int g2s_enabled(void) { return 0; }
const void *g2s_weight_of(const void *key, const void *data, int type, int m, int k, ei_thread_pool *pool) {
    (void)key, (void)data, (void)type, (void)m, (void)k, (void)pool;
    return NULL;
}
void g2s_block(const void *w, const float *x, int t0, int t1, int p0, int np, float *y) {
    (void)w, (void)x, (void)t0, (void)t1, (void)p0, (void)np, (void)y;
}
#endif
