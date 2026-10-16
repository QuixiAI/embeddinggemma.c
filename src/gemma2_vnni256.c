#if defined(__linux__)
#define _DEFAULT_SOURCE /* madvise under -std=c11 (gemma2_release.h) */
#endif
/* cpu-max fork: repacked 256-bit AVX-VNNI Q8_0 x Q8_0 GEMM.
 *
 * Upstream's x86 Q8 tile (gemma2_fast_x86.c, q8_tile_avxvnni) computes each
 * (row, token, 32-block) dot with abs(w), sign(x, w), a zero-seeded vpdpbusd,
 * a conversion, a scalar scale product and an FMA, then reduces 8 lanes per
 * output at the end: about 6 vector ops per 32 MACs. Its repacked kernel
 * (g2x_gemm512_tile) needs AVX-512.
 *
 * This kernel repacks each weight once into 8-row groups so that one 32-byte
 * load holds 4 consecutive k-bytes of 8 different output rows:
 *     w8[g][block][j][r][4] = w[8g + r][32 block + 4j .. 4j + 3]   (j = 0..7)
 * Per (group, block) the 8 rows' fp16 Q8_0 scales and int16 weight sums are
 * kept (36 bytes per 32 weights, vs 34 for Q8_0 itself).
 * Activations are the u8 = q + 128 rows of g2x_acts. For each token the 4
 * k-bytes of step j are broadcast to all lanes and one vpdpbusd advances 8
 * output rows at once; seeding the int32 accumulator with -128 * sum(w) per
 * (row, block) cancels the +128 bias exactly. A block's 8 rows are then
 * scaled with one convert, one multiply and one FMA (per-row weight scales
 * times the token's block scale), and the 8 lanes are the 8 contiguous
 * outputs -- no horizontal reduction. Per 256 MACs: 8 vpdpbusd + 3 ops.
 * Integer dot products are exact, so results equal the reference Q8_0 dot up
 * to fp32 summation order. */
#include "gemma2_vnni256.h"
#include "gemma2_fast_x86.h"
#include "gemma2_release.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    const void *key;
    int m, k;
    int8_t *q;    /* [m/8][k/32][8 steps][8 rows][4] */
    ei_fp16 *s;   /* [m/8][k/32][8 rows], the Q8_0 block scales (fp16, exact) */
    int16_t *c;   /* [m/8][k/32][8 rows] = sum(w) in [-4064, 4064]; x -128 at use */
} g2v_weight;

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

enum { G2V_CACHE = 1024 };
static g2v_weight g2v_weights[G2V_CACHE];
static int g2v_count;

int g2v_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *flag = getenv("EI_GEMMA2_CPU_VNNI256"); /* default on where supported */
        __builtin_cpu_init();
        cached = (!flag || strcmp(flag, "0")) && g2x_level() == G2X_AVXVNNI; /* level implies AVX-VNNI */
    }
    return cached;
}

/* Repack 8-row groups [begin, end): runs on the caller's pool (first use of a tensor). */
typedef struct {
    g2v_weight *w;
    const ei_block_q8_0 *blocks;
    int kb;
} g2v_repack_job;
static void g2v_repack_groups(void *opaque, int32_t begin, int32_t end) {
    const g2v_repack_job *j = opaque;
    const int kb = j->kb;
    for (int g = begin; g < end; g++)
        for (int p = 0; p < kb; p++) {
            int8_t *dst = j->w->q + ((size_t)g * kb + p) * 256;
            for (int r = 0; r < 8; r++) {
                const ei_block_q8_0 *b = j->blocks + (size_t)(8 * g + r) * kb + p;
                int32_t sum = 0;
                for (int i = 0; i < 8; i++)
                    memcpy(dst + i * 32 + r * 4, b->qs + 4 * i, 4);
                for (int z = 0; z < 32; z++)
                    sum += b->qs[z];
                j->w->s[((size_t)g * kb + p) * 8 + r] = b->d;
                j->w->c[((size_t)g * kb + p) * 8 + r] = (int16_t)sum;
            }
        }
}

const void *g2v_weight_of(const void *key, const ei_block_q8_0 *blocks, int m, int k, ei_thread_pool *pool) {
    for (int i = 0; i < g2v_count; i++)
        if (g2v_weights[i].key == key) {
            if (g2v_weights[i].m != m || g2v_weights[i].k != k)
                return NULL; /* a different tensor reused the address: never serve a stale repack */
            return &g2v_weights[i];
        }
    if (g2v_count == G2V_CACHE || m % 8 || k % 32)
        return NULL;
    const int kb = k / 32;
    g2v_weight *w = &g2v_weights[g2v_count];
    w->q = aligned_alloc(64, (size_t)m * k);
    w->s = aligned_alloc(64, ((size_t)m * kb * sizeof(ei_fp16) + 63) & ~(size_t)63);
    w->c = aligned_alloc(64, ((size_t)m * kb * sizeof(int16_t) + 63) & ~(size_t)63);
    if (!w->q || !w->s || !w->c)
        return NULL;
    g2v_repack_job job = {w, blocks, kb};
    ei_parallel_for(pool, m / 8, 8, g2v_repack_groups, &job);
    g2_release_source(blocks, (size_t)m * kb * sizeof(ei_block_q8_0));
    w->key = key;
    w->m = m;
    w->k = k;
    g2v_count++;
    return w;
}

#define AVXVNNI __attribute__((target("avx2,fma,f16c,avxvnni")))

/* Explicitly unrolled (the upstream build is -O2, where GCC keeps small fixed
 * loops rolled and spills the accumulator arrays to the stack). Each token
 * carries two interleaved int32 chains (even/odd k-steps), so a 4-token tile
 * keeps 8 independent vpdpbusd chains in flight (14 ymm registers). */
#define G2V_UNROLL _Pragma("GCC unroll 16")
#define G2V_TILE(NT)                                                                               \
    AVXVNNI static void tile##NT(const g2v_weight *w, int g, const uint8_t *xu, const float *xs,     \
                                 int t0, int k, float *y, int m) {                                 \
        const int kb = k / 32;                                                                     \
        const int8_t *wq = w->q + (size_t)g * kb * 256;                                            \
        const ei_fp16 *ws = w->s + (size_t)g * kb * 8;                                             \
        const int16_t *wc = w->c + (size_t)g * kb * 8;                                             \
        const uint8_t *xr[NT];                                                                     \
        const float *sr[NT];                                                                       \
        __m256 acc[NT];                                                                            \
        G2V_UNROLL for (int t = 0; t < NT; t++) {                                                  \
            xr[t] = xu + (size_t)(t0 + t) * k;                                                     \
            sr[t] = xs + (size_t)(t0 + t) * kb;                                                    \
            acc[t] = _mm256_setzero_ps();                                                          \
        }                                                                                          \
        for (int p = 0; p < kb; p++) {                                                             \
            __m256i even[NT], odd[NT];                                                             \
            const __m256i corr = _mm256_slli_epi32(                                                \
                _mm256_sub_epi32(_mm256_setzero_si256(),                                           \
                                 _mm256_cvtepi16_epi32(_mm_load_si128((const __m128i *)(wc + (size_t)p * 8)))), \
                7); /* -128 * sum(w) */                                                            \
            G2V_UNROLL for (int t = 0; t < NT; t++) {                                              \
                even[t] = corr;                                                                    \
                odd[t] = _mm256_setzero_si256();                                                   \
            }                                                                                      \
            const int8_t *wp = wq + (size_t)p * 256;                                               \
            G2V_UNROLL for (int j = 0; j < 8; j += 2) {                                            \
                const __m256i w0 = _mm256_load_si256((const __m256i *)(wp + j * 32));              \
                G2V_UNROLL for (int t = 0; t < NT; t++) {                                          \
                    int32_t four;                                                                  \
                    memcpy(&four, xr[t] + p * 32 + j * 4, 4);                                      \
                    even[t] = _mm256_dpbusd_avx_epi32(even[t], _mm256_set1_epi32(four), w0);       \
                }                                                                                  \
                const __m256i w1 = _mm256_load_si256((const __m256i *)(wp + j * 32 + 32));         \
                G2V_UNROLL for (int t = 0; t < NT; t++) {                                          \
                    int32_t four;                                                                  \
                    memcpy(&four, xr[t] + p * 32 + j * 4 + 4, 4);                                  \
                    odd[t] = _mm256_dpbusd_avx_epi32(odd[t], _mm256_set1_epi32(four), w1);         \
                }                                                                                  \
            }                                                                                      \
            const __m256 sw = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)(ws + (size_t)p * 8))); \
            G2V_UNROLL for (int t = 0; t < NT; t++)                                                \
                acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(even[t], odd[t])),    \
                                         _mm256_mul_ps(sw, _mm256_broadcast_ss(sr[t] + p)), acc[t]);\
        }                                                                                          \
        G2V_UNROLL for (int t = 0; t < NT; t++)                                                    \
            _mm256_storeu_ps(y + (size_t)(t0 + t) * m + 8 * g, acc[t]);                            \
    }
G2V_TILE(1)
G2V_TILE(2)
G2V_TILE(3)
G2V_TILE(4)

void g2v_tile(const void *wp, int g, const uint8_t *xu, const float *xs, int t0, int nt, int k, float *y,
              int m) {
    const g2v_weight *w = wp;
    switch (nt) {
    case 1:
        tile1(w, g, xu, xs, t0, k, y, m);
        break;
    case 2:
        tile2(w, g, xu, xs, t0, k, y, m);
        break;
    case 3:
        tile3(w, g, xu, xs, t0, k, y, m);
        break;
    default:
        tile4(w, g, xu, xs, t0, k, y, m);
        break;
    }
}
#else
int g2v_enabled(void) { return 0; }
const void *g2v_weight_of(const void *key, const ei_block_q8_0 *blocks, int m, int k, ei_thread_pool *pool) {
    (void)key, (void)blocks, (void)m, (void)k, (void)pool;
    return NULL;
}
void g2v_tile(const void *w, int g, const uint8_t *xu, const float *xs, int t0, int nt, int k, float *y,
              int m) {
    (void)w, (void)g, (void)xu, (void)xs, (void)t0, (void)nt, (void)k, (void)y, (void)m;
}
#endif
