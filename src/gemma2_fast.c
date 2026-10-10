#if defined(__linux__)
#define _POSIX_C_SOURCE 200809L /* clock_gettime under -std=c11 on glibc */
#endif
/* EmbeddingGemma 2 packed-batch CPU text graph (default; EI_GEMMA2_CPU_FAST=0
 * selects the reference graph in gemma2.c).
 * Same math as ei_gemma2_embed, restructured for throughput: one packed wave
 * with per-sequence RoPE positions and block-diagonal attention, activations
 * quantized once per normalized input, a 4-row x 4-token NEON sdot Q8 GEMM
 * tile, threaded SIMD row kernels, cached RoPE tables, persistent workspace. */
#include "gemma2.h"
#include "gemma2_fast_x86.h"
#include "kernels.h"
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static const ei_tensor *tensor_of(const ei_gguf *g, const char *name) {
    return ei_gguf_tensor(g, name, true);
}
static float half_to_float(ei_fp16 h) {
#if defined(__ARM_NEON)
    __fp16 v;
    memcpy(&v, &h, 2);
    return (float)v;
#else
    return ei_fp16_to_fp32(h);
#endif
}
static float element(const ei_tensor *t, size_t i) {
    switch (t->type) {
    case EI_T_F32:
        return ((const float *)t->data)[i];
    case EI_T_F16:
        return half_to_float(((const ei_fp16 *)t->data)[i]);
    case EI_T_BF16: {
        uint32_t bits = (uint32_t)((const uint16_t *)t->data)[i] << 16;
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }
    default: {
        const ei_block_q8_0 *b = (const ei_block_q8_0 *)t->data + i / 32;
        return half_to_float(b->d) * b->qs[i % 32];
    }
    }
}

bool ei_gemma2_cpu_fast(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *flag = getenv("EI_GEMMA2_CPU_FAST"); // default on; "0" = reference graph
        cached = !flag || strcmp(flag, "0");
    }
    return cached;
}

/* ---- workspace -------------------------------------------------------- */

typedef struct {
    size_t tokens;
    float *x, *h, *q, *k, *v, *a, *up, *gate, *ple;
    ei_block_q8_0 *xq;
    uint32_t *lo, *hi, *pos;
    float norm_w[4][512]; /* dequantized norm weights for the current layer */
    /* Per-call activation repacks for the i8mm and AVX-512 GEMMs. */
    int8_t *xpair_q;
    float *xpair_s;
    size_t xpair_cap;
    uint8_t *x512_u;
    float *x512_s;
    size_t x512_cap;
} workspace;

/* One workspace per context, so independent contexts (each with its own
 * thread pool) run waves concurrently; the default context serves callers
 * of ei_gemma2_embed_batch_fast. Repacked-weight caches are shared. */
struct ei_gemma2_fast {
    pthread_mutex_t lock;
    workspace ws;
};
static ei_gemma2_fast default_ctx = {PTHREAD_MUTEX_INITIALIZER, {0}};
static pthread_mutex_t weight_cache_lock = PTHREAD_MUTEX_INITIALIZER;

static void reserve(workspace *ws, size_t n) {
    if (ws->tokens >= n)
        return;
    size_t cap = ws->tokens ? ws->tokens : 64;
    while (cap < n)
        cap *= 2;
    free(ws->x), free(ws->h), free(ws->q), free(ws->k), free(ws->v), free(ws->a), free(ws->up);
    free(ws->gate), free(ws->ple), free(ws->xq), free(ws->lo), free(ws->hi), free(ws->pos);
    ws->x = ei_xmalloc(cap * 512 * 4);
    ws->h = ei_xmalloc(cap * 512 * 4);
    ws->q = ei_xmalloc(cap * 2048 * 4);
    ws->k = ei_xmalloc(cap * 512 * 4);
    ws->v = ei_xmalloc(cap * 512 * 4);
    ws->a = ei_xmalloc(cap * 2048 * 4);
    ws->up = ei_xmalloc(cap * 2048 * 4);
    ws->gate = ei_xmalloc(cap * 2048 * 4);
    ws->ple = ei_xmalloc(cap * 512 * 24 * 4);
    ws->xq = ei_xmalloc(cap * 2048 / 32 * sizeof(ei_block_q8_0));
    ws->lo = ei_xmalloc(cap * 4);
    ws->hi = ei_xmalloc(cap * 4);
    ws->pos = ei_xmalloc(cap * 4);
    ws->tokens = cap;
}

/* RoPE tables for both bases, identical to the reference's per-element
 * theta = t * powf(base, -2z/d); built once for all 8192 positions. */
static float *rope_cos[2], *rope_sin[2];
static pthread_once_t rope_once = PTHREAD_ONCE_INIT;
static void build_rope(void) {
    for (int b = 0; b < 2; b++) {
        int d = b ? 512 : 256;
        float base = b ? 1000000.0f : 10000.0f;
        rope_cos[b] = ei_xmalloc((size_t)8192 * d / 2 * 4);
        rope_sin[b] = ei_xmalloc((size_t)8192 * d / 2 * 4);
        for (int t = 0; t < 8192; t++)
            for (int z = 0; z < d / 2; z++) {
                float theta = t * powf(base, -2.0f * z / d);
                rope_cos[b][(size_t)t * d / 2 + z] = cosf(theta);
                rope_sin[b][(size_t)t * d / 2 + z] = sinf(theta);
            }
    }
}

/* EI_GEMMA2_CPU_PROFILE=1: per-stage wall time, printed per call. */
#include <time.h>
static double prof[6];
static const char *prof_names[6] = {"gemm_q8", "gemm_f", "rows", "attention", "embed", "pool"};
static double prof_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}
#define PROF(i, stmt)                                                                           \
    do {                                                                                        \
        double prof_start = prof_now();                                                         \
        stmt;                                                                                   \
        prof[i] += prof_now() - prof_start;                                                     \
    } while (0)

/* ---- Q8_0 x Q8_0 GEMM: y[n][m] = x[n][k] . w[m][k] ----------------------- */

typedef struct {
    const ei_block_q8_0 *w, *x;
    float *y;
    int k, m, n, row_blocks, rb_rows;
} q8_job;

/* 4 weight rows x nt (1..4) tokens; each weight block is loaded once and
 * applied to every token in the tile. */
static void q8_tile(const ei_block_q8_0 *w, const ei_block_q8_0 *x, float *y, int kb, int m,
                    int nt) {
#if defined(__ARM_FEATURE_DOTPROD)
    float32x4_t acc[4][4];
    for (int r = 0; r < 4; r++)
        for (int t = 0; t < 4; t++)
            acc[r][t] = vdupq_n_f32(0);
    for (int p = 0; p < kb; p++) {
        int8x16_t wa[4], wb[4];
        float dw[4];
        for (int r = 0; r < 4; r++) {
            const ei_block_q8_0 *blk = w + (size_t)r * kb + p;
            wa[r] = vld1q_s8(blk->qs);
            wb[r] = vld1q_s8(blk->qs + 16);
            dw[r] = half_to_float(blk->d);
        }
        for (int t = 0; t < nt; t++) {
            const ei_block_q8_0 *xb = x + (size_t)t * kb + p;
            int8x16_t xa = vld1q_s8(xb->qs), xc = vld1q_s8(xb->qs + 16);
            float dx = half_to_float(xb->d);
            for (int r = 0; r < 4; r++) {
                int32x4_t s = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wa[r], xa), wb[r], xc);
                acc[r][t] = vmlaq_n_f32(acc[r][t], vcvtq_f32_s32(s), dw[r] * dx);
            }
        }
    }
    for (int t = 0; t < nt; t++)
        for (int r = 0; r < 4; r++)
            y[(size_t)t * m + r] = vaddvq_f32(acc[r][t]);
#else
#if defined(__x86_64__)
    if (g2x_level()) {
        g2x_q8_tile(w, x, y, kb, m, nt);
        return;
    }
#endif
    for (int t = 0; t < nt; t++)
        for (int r = 0; r < 4; r++) {
            float sum = 0;
            for (int p = 0; p < kb; p++) {
                const ei_block_q8_0 *a = w + (size_t)r * kb + p, *b = x + (size_t)t * kb + p;
                int dot = 0;
                for (int z = 0; z < 32; z++)
                    dot += a->qs[z] * b->qs[z];
                sum += (float)dot * half_to_float(a->d) * half_to_float(b->d);
            }
            y[(size_t)t * m + r] = sum;
        }
#endif
}
/* Tasks are 32-row x 32-token tiles so both operands stay cache-resident. */
static void q8_rows(void *opaque, int32_t begin, int32_t end) {
    q8_job *j = opaque;
    int kb = j->k / 32;
    for (int task = begin; task < end; task++) {
        int rb = task % j->row_blocks, tb = task / j->row_blocks, rr = j->rb_rows;
        int r_end = rb * rr + rr < j->m ? rb * rr + rr : j->m;
        int t_end = tb * 32 + 32 < j->n ? tb * 32 + 32 : j->n;
        for (int t = tb * 32; t < t_end; t += 4)
            for (int r = rb * rr; r < r_end; r += 4)
                q8_tile(j->w + (size_t)r * kb, j->x + (size_t)t * kb, j->y + (size_t)t * j->m + r,
                        kb, j->m, t_end - t < 4 ? t_end - t : 4);
    }
}
/* Waves of <= 8 tokens are wake-up and preemption bound: spread them over
 * ~2/3 of the pool rather than every thread (EI_GEMMA2_CPU_SHORT_THREADS
 * overrides). Measured best near 12 of 18 threads on an M5 Max and 21-32 of
 * 32 on a Ryzen 9950X; derived from the pool, not tuned per machine. */
static int short_grain(ei_thread_pool *pool, int tasks, int n) {
    if (n > 8)
        return 1;
    static int width = -1;
    if (width < 0) {
        const char *env = getenv("EI_GEMMA2_CPU_SHORT_THREADS");
        int threads = ei_thread_pool_threads(pool);
        width = env ? atoi(env) : threads * 2 / 3;
        if (width < 1)
            width = 1;
    }
    return (tasks + width - 1) / width;
}
static bool gemm_q8_i8mm(workspace *ws, ei_thread_pool *pool, const ei_tensor *w, const ei_block_q8_0 *xq, int n,
                         float *y);
static bool gemm_q8_x86_512(workspace *ws, ei_thread_pool *pool, const ei_tensor *w, const ei_block_q8_0 *xq,
                            int n, float *y);
static void gemm_q8(workspace *ws, ei_thread_pool *pool, const ei_tensor *w, const ei_block_q8_0 *xq, int n,
                    float *y) {
    if (gemm_q8_i8mm(ws, pool, w, xq, n, y))
        return;
    if (gemm_q8_x86_512(ws, pool, w, xq, n, y))
        return;
    // Short waves: 8-row blocks so every thread gets work; long: 32-row tiles.
    int rr = n <= 8 ? 8 : 32;
    q8_job j = {w->data, xq, y, (int)w->ne[0], (int)w->ne[1], n, (int)(w->ne[1] + rr - 1) / rr, rr};
    int tasks = j.row_blocks * ((n + 31) / 32), grain = short_grain(pool, tasks, n);
    PROF(0, ei_parallel_for(pool, tasks, grain, q8_rows, &j));
}

/* ---- i8mm (smmla) Q8_0 GEMM ----------------------------------------------
 * vmmlaq_s32(acc, a, b): a = 2 rows x 8 int8, b = 2 columns x 8 int8,
 * acc (2x2, row-major) += a . b^T. Weights are repacked once per tensor into
 * row pairs: per pair and 32-block, four 16-byte chunks [r0 k8c..+7 | r1 k8c..+7]
 * plus fp32 scales; activations get the same token-pair layout per call. */
#if defined(__aarch64__) && defined(__APPLE__)
#include <sys/sysctl.h>
#define G2_I8MM_TARGET __attribute__((target("i8mm")))
typedef struct {
    const ei_tensor *t;
    int8_t *q;
    float *s;
} paired_weight;
static paired_weight paired[512];
static int paired_count;
static bool i8mm_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *flag = getenv("EI_GEMMA2_CPU_I8MM");
        int have = 0;
        size_t len = sizeof have;
        if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &have, &len, NULL, 0) != 0)
            have = 0;
        cached = (!flag || strcmp(flag, "0")) && have; // default on where supported
    }
    return cached;
}
static void pair_rows(const ei_block_q8_0 *src, int rows, int kb, int8_t *q, float *s) {
    for (int rp = 0; rp < rows / 2; rp++)
        for (int p = 0; p < kb; p++) {
            const ei_block_q8_0 *a = src + (size_t)(2 * rp) * kb + p, *b = a + kb;
            int8_t *dst = q + ((size_t)rp * kb + p) * 64;
            for (int c = 0; c < 4; c++) {
                memcpy(dst + c * 16, a->qs + c * 8, 8);
                memcpy(dst + c * 16 + 8, b->qs + c * 8, 8);
            }
            s[((size_t)rp * kb + p) * 2] = half_to_float(a->d);
            s[((size_t)rp * kb + p) * 2 + 1] = half_to_float(b->d);
        }
}
static paired_weight *paired_of(const ei_tensor *t) {
    for (int i = 0; i < paired_count; i++)
        if (paired[i].t == t)
            return &paired[i];
    if (paired_count == 512)
        ei_die("i8mm weight cache full");
    int m = (int)t->ne[1], kb = (int)t->ne[0] / 32;
    paired_weight *pw = &paired[paired_count++];
    pw->t = t;
    pw->q = ei_xmalloc((size_t)m * kb * 32);
    pw->s = ei_xmalloc((size_t)m * kb * sizeof(float));
    pair_rows(t->data, m, kb, pw->q, pw->s);
    return pw;
}
typedef struct {
    const int8_t *wq, *xq;
    const float *ws, *xs;
    float *y;
    int kb, m, n, row_blocks;
} mm_job_i8;
/* 8 rows (4 pairs) x 4 tokens (2 pairs). */
G2_I8MM_TARGET static void i8mm_tile(const mm_job_i8 *j, int r, int t) {
    float32x4_t acc[4][2];
    for (int a = 0; a < 4; a++)
        acc[a][0] = acc[a][1] = vdupq_n_f32(0);
    const int8_t *w = j->wq + (size_t)(r / 2) * j->kb * 64, *x = j->xq + (size_t)(t / 2) * j->kb * 64;
    const float *ws = j->ws + (size_t)(r / 2) * j->kb * 2, *xs = j->xs + (size_t)(t / 2) * j->kb * 2;
    size_t wstride = (size_t)j->kb * 64, xstride = (size_t)j->kb * 64;
    for (int p = 0; p < j->kb; p++) {
        int8x16_t xa[2][4];
        float32x4_t xv[2];
        for (int b = 0; b < 2; b++) {
            for (int c = 0; c < 4; c++)
                xa[b][c] = vld1q_s8(x + b * xstride + (size_t)p * 64 + c * 16);
            float32x2_t sx = vld1_f32(xs + b * j->kb * 2 + p * 2);
            xv[b] = vcombine_f32(sx, sx); /* (x0, x1, x0, x1) */
        }
        for (int a = 0; a < 4; a++) {
            const int8_t *wp = w + a * wstride + (size_t)p * 64;
            int8x16_t w0 = vld1q_s8(wp), w1 = vld1q_s8(wp + 16), w2 = vld1q_s8(wp + 32),
                      w3 = vld1q_s8(wp + 48);
            float32x2_t sw = vld1_f32(ws + a * j->kb * 2 + p * 2);
            float32x4_t wv = vcombine_f32(vdup_lane_f32(sw, 0), vdup_lane_f32(sw, 1));
            for (int b = 0; b < 2; b++) {
                int32x4_t s = vmmlaq_s32(vdupq_n_s32(0), w0, xa[b][0]);
                s = vmmlaq_s32(s, w1, xa[b][1]);
                s = vmmlaq_s32(s, w2, xa[b][2]);
                s = vmmlaq_s32(s, w3, xa[b][3]);
                acc[a][b] = vfmaq_f32(acc[a][b], vcvtq_f32_s32(s), vmulq_f32(wv, xv[b]));
            }
        }
    }
    for (int a = 0; a < 4; a++)
        for (int b = 0; b < 2; b++) {
            int r0 = r + 2 * a, t0 = t + 2 * b;
            float out[4];
            vst1q_f32(out, acc[a][b]); /* (r0 t0, r0 t1, r1 t0, r1 t1) */
            if (t0 < j->n) {
                j->y[(size_t)t0 * j->m + r0] = out[0];
                j->y[(size_t)t0 * j->m + r0 + 1] = out[2];
            }
            if (t0 + 1 < j->n) {
                j->y[(size_t)(t0 + 1) * j->m + r0] = out[1];
                j->y[(size_t)(t0 + 1) * j->m + r0 + 1] = out[3];
            }
        }
}
static void i8mm_rows(void *opaque, int32_t begin, int32_t end) {
    mm_job_i8 *j = opaque;
    for (int task = begin; task < end; task++) {
        int rb = task % j->row_blocks, tb = task / j->row_blocks;
        int r_end = rb * 32 + 32 < j->m ? rb * 32 + 32 : j->m;
        int t_end = tb * 32 + 32 < j->n ? tb * 32 + 32 : j->n;
        for (int t = tb * 32; t < t_end; t += 4)
            for (int r = rb * 32; r < r_end; r += 8)
                i8mm_tile(j, r, t);
    }
}
/* Activations to token-pair layout (odd tail paired with zeros). */
static bool gemm_q8_i8mm(workspace *ws, ei_thread_pool *pool, const ei_tensor *w, const ei_block_q8_0 *xq, int n,
                         float *y) {
    int k = (int)w->ne[0], m = (int)w->ne[1], kb = k / 32;
    if (!i8mm_enabled() || m % 32 || n < 4)
        return false;
    pthread_mutex_lock(&weight_cache_lock);
    paired_weight *pw = paired_of(w);
    pthread_mutex_unlock(&weight_cache_lock);
    int np = (n + 3) / 4 * 4;
    size_t need = (size_t)np * kb * 32;
    if (need > ws->xpair_cap) {
        free(ws->xpair_q), free(ws->xpair_s);
        ws->xpair_q = ei_xmalloc(need);
        ws->xpair_s = ei_xmalloc((size_t)np * kb * sizeof(float));
        ws->xpair_cap = need;
    }
    pair_rows(xq, n & ~1, kb, ws->xpair_q, ws->xpair_s);
    for (int t = n & ~1; t < np; t += 2) {
        /* tail pairs: real token (if any) then zeros */
        for (int p = 0; p < kb; p++) {
            int8_t *dst = ws->xpair_q + ((size_t)(t / 2) * kb + p) * 64;
            memset(dst, 0, 64);
            ws->xpair_s[((size_t)(t / 2) * kb + p) * 2] = 0;
            ws->xpair_s[((size_t)(t / 2) * kb + p) * 2 + 1] = 0;
            if (t < n) {
                const ei_block_q8_0 *a = xq + (size_t)t * kb + p;
                for (int c = 0; c < 4; c++)
                    memcpy(dst + c * 16, a->qs + c * 8, 8);
                ws->xpair_s[((size_t)(t / 2) * kb + p) * 2] = half_to_float(a->d);
            }
        }
    }
    mm_job_i8 j = {pw->q, ws->xpair_q, pw->s, ws->xpair_s, y, kb, m, n, m / 32};
    int tasks = j.row_blocks * ((np + 31) / 32);
    PROF(0, ei_parallel_for(pool, tasks, short_grain(pool, tasks, n), i8mm_rows, &j));
    return true;
}
#else
static bool gemm_q8_i8mm(workspace *ws, ei_thread_pool *pool, const ei_tensor *w, const ei_block_q8_0 *xq, int n,
                         float *y) {
    (void)ws, (void)pool, (void)w, (void)xq, (void)n, (void)y;
    return false;
}
#endif

/* ---- x86 AVX-512 VNNI GEMM on repacked operands (gemma2_fast_x86.c) ---------- */
#if defined(__x86_64__)
typedef struct {
    const void *w;
    uint8_t *xu;
    float *xs;
    float *y;
    int k, m, n, row_blocks;
    const ei_block_q8_0 *xq;
} x512_job;
static void x512_acts(void *opaque, int32_t begin, int32_t end) {
    x512_job *j = opaque;
    g2x_acts(j->xq, begin, end, j->k, j->xu, j->xs);
}
static void x512_rows(void *opaque, int32_t begin, int32_t end) {
    x512_job *j = opaque;
    for (int task = begin; task < end; task++) {
        int rb = task % j->row_blocks, tb = task / j->row_blocks;
        int r_end = rb * 32 + 32 < j->m ? rb * 32 + 32 : j->m;
        int t_end = tb * 32 + 32 < j->n ? tb * 32 + 32 : j->n;
        for (int t = tb * 32; t < t_end; t += 4)
            for (int r = rb * 32; r < r_end; r += 4)
                g2x_gemm512_tile(j->w, r, j->xu, j->xs, t, t_end - t < 4 ? t_end - t : 4, j->k, j->y,
                                 j->m);
    }
}
static bool gemm_q8_x86_512(workspace *ws, ei_thread_pool *pool, const ei_tensor *w, const ei_block_q8_0 *xq,
                            int n, float *y) {
    int k = (int)w->ne[0], m = (int)w->ne[1];
    // Short waves stay on the 4-row tile with its short-wave task width.
    if (!g2x_gemm512_enabled() || n <= 8 || k % 64 || m % 4)
        return false;
    pthread_mutex_lock(&weight_cache_lock);
    const void *pw = g2x_weight_of(w, w->data, m, k);
    pthread_mutex_unlock(&weight_cache_lock);
    if (!pw)
        return false;
    if ((size_t)n * k > ws->x512_cap) {
        free(ws->x512_u), free(ws->x512_s);
        ws->x512_u = ei_xmalloc((size_t)n * k);
        ws->x512_s = ei_xmalloc((size_t)n * (k / 32) * sizeof(float));
        ws->x512_cap = (size_t)n * k;
    }
    x512_job j = {pw, ws->x512_u, ws->x512_s, y, k, m, n, (m + 31) / 32, xq};
    PROF(0, ei_parallel_for(pool, n, 16, x512_acts, &j));
    PROF(0, ei_parallel_for(pool, j.row_blocks * ((n + 31) / 32), 1, x512_rows, &j));
    return true;
}
#else
static bool gemm_q8_x86_512(workspace *ws, ei_thread_pool *pool, const ei_tensor *w, const ei_block_q8_0 *xq,
                            int n, float *y) {
    (void)ws, (void)pool, (void)w, (void)xq, (void)n, (void)y;
    return false;
}
#endif

/* Float-activation GEMM for BF16/F16/F32 weights (the PLE projection). */
typedef struct {
    const ei_tensor *w;
    const float *x;
    float *y;
    int k, m, n;
} f_job;
static void f_rows(void *opaque, int32_t begin, int32_t end) {
    f_job *j = opaque;
    int k = j->k;
    float rows[4][1024];
    for (int rb = begin; rb < end; rb++) {
        int r0 = rb * 4, nr = j->m - r0 < 4 ? j->m - r0 : 4;
        for (int r = 0; r < nr; r++) {
#if defined(__ARM_NEON)
            if (j->w->type == EI_T_BF16 && k % 8 == 0) {
                const uint16_t *src = (const uint16_t *)j->w->data + (size_t)(r0 + r) * k;
                for (int z = 0; z < k; z += 8) {
                    uint16x8_t b = vld1q_u16(src + z);
                    vst1q_f32(rows[r] + z, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b), 16)));
                    vst1q_f32(rows[r] + z + 4,
                              vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b), 16)));
                }
                continue;
            }
#elif defined(__x86_64__)
            if (j->w->type == EI_T_BF16 && g2x_level()) {
                g2x_bf16_row((const uint16_t *)j->w->data + (size_t)(r0 + r) * k, rows[r], k);
                continue;
            }
#endif
            for (int z = 0; z < k; z++)
                rows[r][z] = element(j->w, (size_t)(r0 + r) * k + z);
        }
        for (int r = nr; r < 4; r++)
            memset(rows[r], 0, (size_t)k * 4);
        for (int t0 = 0; t0 < j->n; t0 += 4) {
            int nt = j->n - t0 < 4 ? j->n - t0 : 4;
#if defined(__ARM_NEON)
            /* 4 rows x 4 tokens: 16 accumulators, 8 loads per 16 FMAs. */
            float32x4_t acc[4][4];
            for (int r = 0; r < 4; r++)
                for (int t = 0; t < 4; t++)
                    acc[r][t] = vdupq_n_f32(0);
            const float *xt[4];
            for (int t = 0; t < 4; t++)
                xt[t] = j->x + (size_t)(t0 + (t < nt ? t : 0)) * k;
            for (int z = 0; z < k; z += 4) {
                float32x4_t xv[4], wv[4];
                for (int t = 0; t < 4; t++)
                    xv[t] = vld1q_f32(xt[t] + z);
                for (int r = 0; r < 4; r++)
                    wv[r] = vld1q_f32(rows[r] + z);
                for (int r = 0; r < 4; r++)
                    for (int t = 0; t < 4; t++)
                        acc[r][t] = vfmaq_f32(acc[r][t], wv[r], xv[t]);
            }
            for (int t = 0; t < nt; t++)
                for (int r = 0; r < nr; r++)
                    j->y[(size_t)(t0 + t) * j->m + r0 + r] = vaddvq_f32(acc[r][t]);
#else
#if defined(__x86_64__)
            if (g2x_level() && k % 8 == 0) {
                const float *xt[4], *wr[4] = {rows[0], rows[1], rows[2], rows[3]};
                for (int t = 0; t < 4; t++)
                    xt[t] = j->x + (size_t)(t0 + (t < nt ? t : 0)) * k;
                float dots[4][4];
                g2x_dot4x4(wr, xt, k, dots);
                for (int t = 0; t < nt; t++)
                    for (int r = 0; r < nr; r++)
                        j->y[(size_t)(t0 + t) * j->m + r0 + r] = dots[r][t];
                continue;
            }
#endif
            for (int t = 0; t < nt; t++)
                for (int r = 0; r < nr; r++) {
                    float sum = 0;
                    for (int z = 0; z < k; z++)
                        sum += rows[r][z] * j->x[(size_t)(t0 + t) * k + z];
                    j->y[(size_t)(t0 + t) * j->m + r0 + r] = sum;
                }
#endif
        }
    }
}
static void gemm_f(ei_thread_pool *pool, const ei_tensor *w, const float *x, int n, float *y) {
    f_job j = {w, x, y, (int)w->ne[0], (int)w->ne[1], n};
    if (j.k > 1024 || j.k % 4)
        ei_die("gemm_f supports k <= 1024, multiple of 4");
    PROF(1, ei_parallel_for(pool, (j.m + 3) / 4, 2, f_rows, &j));
}

/* ---- row kernels -------------------------------------------------------- */

static float sum_squares(const float *x, int d) {
#if defined(__ARM_NEON)
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    int z = 0;
    for (; z + 8 <= d; z += 8) {
        float32x4_t a = vld1q_f32(x + z), b = vld1q_f32(x + z + 4);
        s0 = vfmaq_f32(s0, a, a);
        s1 = vfmaq_f32(s1, b, b);
    }
    float s = vaddvq_f32(vaddq_f32(s0, s1));
    for (; z < d; z++)
        s += x[z] * x[z];
    return s;
#else
#if defined(__x86_64__)
    if (g2x_level())
        return g2x_sum_squares(x, d);
#endif
    double s = 0;
    for (int z = 0; z < d; z++)
        s += (double)x[z] * x[z];
    return (float)s;
#endif
}
static void dequant_norm(const ei_tensor *w, int d, float *out) {
    for (int z = 0; z < d; z++)
        out[z] = element(w, z);
}

typedef enum {
    ROW_NORM_QUANT,     /* y = rms(x) * w; q8(y) */
    ROW_QKV_POST,       /* per head: rms (weighted q/k, plain v), RoPE q/k */
    ROW_POST_ADD,       /* x = (x + rms(h) * w) * scale */
    ROW_GELU_QUANT,     /* gate = gelu(gate) * up; q8(gate) */
    ROW_PLE_GATE_QUANT, /* h = gelu(h) * ple[t, layer]; q8(h) */
    ROW_QUANT,          /* q8(x) */
} row_op;
typedef struct {
    row_op op;
    const float *in;
    float *out;
    const float *w;
    float scale;
    int width, layer, d, kv, rope_base;
    workspace *ws;
} row_job;

/* gelu_tanh over a row. NEON: tanh(u) = 1 - 2 / (exp(2u) + 1) with a
 * Cody-Waite range-reduced degree-6 exp polynomial (~1 ulp), inputs clamped
 * where tanh is +-1 in float. */
#if defined(__ARM_NEON)
static float32x4_t exp4(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0f)), vdupq_n_f32(88.0f));
    float32x4_t n = vrndnq_f32(vmulq_n_f32(x, 1.44269504088896341f));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
    r = vfmsq_f32(r, n, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t p = vdupq_n_f32(1.0f / 720), one = vdupq_n_f32(1);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 120), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 24), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 6), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(one, p, r);
    p = vfmaq_f32(one, p, r);
    int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(e));
}
#endif
static void gelu_mul_row(float *g, const float *u, int n) {
    int z = 0;
#if defined(__ARM_NEON)
    const float32x4_t s = vdupq_n_f32(0.79788456080286535588f), a = vdupq_n_f32(0.044715f);
    const float32x4_t half = vdupq_n_f32(0.5f), one = vdupq_n_f32(1), two = vdupq_n_f32(2);
    for (; z + 4 <= n; z += 4) {
        float32x4_t x = vld1q_f32(g + z);
        float32x4_t inner = vmulq_f32(vmulq_f32(s, x), vfmaq_f32(one, vmulq_f32(a, x), x));
        inner = vminq_f32(vmaxq_f32(inner, vdupq_n_f32(-9.0f)), vdupq_n_f32(9.0f));
        float32x4_t t = vsubq_f32(one, vdivq_f32(two, vaddq_f32(exp4(vmulq_f32(two, inner)), one)));
        float32x4_t gelu = vmulq_f32(vmulq_f32(half, x), vaddq_f32(one, t));
        vst1q_f32(g + z, u ? vmulq_f32(gelu, vld1q_f32(u + z)) : gelu);
    }
#elif defined(__x86_64__)
    if (g2x_level()) {
        g2x_gelu_mul_row(g, u, n);
        return;
    }
#endif
    for (; z < n; z++)
        g[z] = ei_gelu_tanh(g[z]) * (u ? u[z] : 1.0f);
}
static void quant_row(const float *x, ei_block_q8_0 *q, int width) {
#if defined(__x86_64__) && !defined(__AVX2__)
    if (g2x_level()) {
        g2x_quantize_row_q8_0(x, q, width); /* the AVX2 build's rounding, dispatched */
        return;
    }
#endif
    ei_quantize_row_q8_0(x, q, width);
}
static void rms_row(float *r, int d, const float *w) {
    float scale = 1.0f / sqrtf(sum_squares(r, d) / d + 1e-6f);
    for (int z = 0; z < d; z++)
        r[z] *= scale * (w ? w[z] : 1.0f);
}
static void rope_row(float *r, int d, int pos, int base) {
    const float *c = rope_cos[base] + (size_t)pos * d / 2, *s = rope_sin[base] + (size_t)pos * d / 2;
    for (int z = 0; z < d / 2; z++) {
        float a = r[z], b = r[z + d / 2];
        r[z] = a * c[z] - b * s[z];
        r[z + d / 2] = a * s[z] + b * c[z];
    }
}
static void row_tokens(void *opaque, int32_t begin, int32_t end) {
    row_job *j = opaque;
    workspace *ws = j->ws;
    for (int t = begin; t < end; t++) {
        switch (j->op) {
        case ROW_NORM_QUANT: {
            float *y = j->out + (size_t)t * 512;
            memcpy(y, j->in + (size_t)t * 512, 512 * 4);
            rms_row(y, 512, j->w);
            quant_row(y, ws->xq + (size_t)t * 16, 512);
            break;
        }
        case ROW_QKV_POST: {
            int d = j->d;
            for (int hd = 0; hd < 4; hd++) {
                float *r = ws->q + ((size_t)t * 4 + hd) * d;
                rms_row(r, d, ws->norm_w[0]);
                rope_row(r, d, (int)ws->pos[t], j->rope_base);
            }
            for (int hd = 0; hd < j->kv; hd++) {
                float *r = ws->k + ((size_t)t * j->kv + hd) * d;
                rms_row(r, d, ws->norm_w[1]);
                rope_row(r, d, (int)ws->pos[t], j->rope_base);
                rms_row(ws->v + ((size_t)t * j->kv + hd) * d, d, NULL);
            }
            break;
        }
        case ROW_POST_ADD: {
            float *h = j->out + (size_t)t * 512, *x = ws->x + (size_t)t * 512;
            rms_row(h, 512, j->w);
            for (int z = 0; z < 512; z++)
                x[z] = (x[z] + h[z]) * j->scale;
            break;
        }
        case ROW_GELU_QUANT: {
            float *g = ws->gate + (size_t)t * 2048;
            const float *u = ws->up + (size_t)t * 2048;
            gelu_mul_row(g, u, 2048);
            quant_row(g, ws->xq + (size_t)t * 64, 2048);
            break;
        }
        case ROW_PLE_GATE_QUANT: {
            float *h = ws->h + (size_t)t * 512;
            const float *p = ws->ple + ((size_t)t * 24 + j->layer) * 512;
            gelu_mul_row(h, p, 512);
            quant_row(h, ws->xq + (size_t)t * 16, 512);
            break;
        }
        case ROW_QUANT:
            quant_row(j->in + (size_t)t * j->width, ws->xq + (size_t)t * j->width / 32, j->width);
            break;
        }
    }
}
static void rows(workspace *ws, ei_thread_pool *pool, int n, row_job job) {
    job.ws = ws;
    PROF(2, ei_parallel_for(pool, n, 8, row_tokens, &job));
}

/* ---- attention ---------------------------------------------------------- */

typedef struct {
    int n, kv, d, window;
    workspace *ws;
} attn_job;
#if !defined(__ARM_NEON)
static float dot_f(const float *a, const float *b, int d) {
#if defined(__ARM_NEON)
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0), s2 = vdupq_n_f32(0),
                s3 = vdupq_n_f32(0);
    for (int z = 0; z < d; z += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + z), vld1q_f32(b + z));
        s1 = vfmaq_f32(s1, vld1q_f32(a + z + 4), vld1q_f32(b + z + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + z + 8), vld1q_f32(b + z + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + z + 12), vld1q_f32(b + z + 12));
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
#else
    float s = 0;
    for (int z = 0; z < d; z++)
        s += a[z] * b[z];
    return s;
#endif
}
#endif
/* Tasks are (block of QB queries, head): each K/V row is loaded once for all
 * queries of the block; each query keeps its own sequence/window mask. */
enum { QB = 8 };
static void attention_rows(void *opaque, int32_t begin, int32_t end) {
    attn_job *j = opaque;
    workspace *ws = j->ws;
    int d = j->d, blocks = (j->n + QB - 1) / QB;
    static _Thread_local float scores[QB][8192];
    for (int task = begin; task < end; task++) {
        int hd = task / blocks, t0 = task % blocks * QB, kh = hd / (4 / j->kv);
        int nq = j->n - t0 < QB ? j->n - t0 : QB;
        int lo[QB], hi[QB], klo = 1 << 30, khi = 0;
        for (int i = 0; i < nq; i++) {
            int t = t0 + i;
            lo[i] = (int)ws->lo[t], hi[i] = (int)ws->hi[t];
            if (j->window) {
                if (t - j->window > lo[i])
                    lo[i] = t - j->window;
                if (t + j->window + 1 < hi[i])
                    hi[i] = t + j->window + 1;
            }
            klo = lo[i] < klo ? lo[i] : klo;
            khi = hi[i] > khi ? hi[i] : khi;
        }
        const float *q[QB];
        float max[QB], total[QB];
        for (int i = 0; i < nq; i++) {
            q[i] = ws->q + ((size_t)(t0 + i) * 4 + hd) * d;
            max[i] = -INFINITY;
            total[i] = 0;
        }
        /* S = Q K^T in 4-query x 4-key register tiles. */
        int nk = khi - klo;
        for (int s0 = 0; s0 < nk; s0 += 4) {
            const float *kp[4];
            for (int c = 0; c < 4; c++)
                kp[c] = ws->k + ((size_t)(klo + (s0 + c < nk ? s0 + c : s0)) * j->kv + kh) * d;
            for (int i0 = 0; i0 < nq; i0 += 4) {
                const float *qp[4];
                for (int r = 0; r < 4; r++)
                    qp[r] = q[i0 + r < nq ? i0 + r : i0];
                float dots[4][4];
#if defined(__ARM_NEON)
                float32x4_t acc[4][4];
                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++)
                        acc[r][c] = vdupq_n_f32(0);
                for (int z = 0; z < d; z += 4) {
                    float32x4_t qv[4], kv[4];
                    for (int r = 0; r < 4; r++)
                        qv[r] = vld1q_f32(qp[r] + z);
                    for (int c = 0; c < 4; c++)
                        kv[c] = vld1q_f32(kp[c] + z);
                    for (int r = 0; r < 4; r++)
                        for (int c = 0; c < 4; c++)
                            acc[r][c] = vfmaq_f32(acc[r][c], qv[r], kv[c]);
                }
                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++)
                        dots[r][c] = vaddvq_f32(acc[r][c]);
#else
#if defined(__x86_64__)
                if (g2x_level())
                    g2x_dot4x4(qp, kp, d, dots);
                else
#endif
                    for (int r = 0; r < 4; r++)
                        for (int c = 0; c < 4; c++)
                            dots[r][c] = dot_f(qp[r], kp[c], d);
#endif
                for (int r = 0; r < 4 && i0 + r < nq; r++)
                    for (int c = 0; c < 4 && s0 + c < nk; c++) {
                        int i = i0 + r, sk = klo + s0 + c;
                        float v = sk >= lo[i] && sk < hi[i] ? dots[r][c] : -INFINITY;
                        scores[i][s0 + c] = v;
                        if (v > max[i])
                            max[i] = v;
                    }
            }
        }
        for (int i = 0; i < nq; i++) {
            for (int sk = 0; sk < nk; sk++) {
                float e = scores[i][sk] == -INFINITY ? 0.0f : expf(scores[i][sk] - max[i]);
                scores[i][sk] = e;
                total[i] += e;
            }
            for (int sk = 0; sk < nk; sk++)
                scores[i][sk] /= total[i];
        }
        /* O = P V with 8 queries x 8 dims in registers across the key loop. */
#if defined(__x86_64__)
        if (g2x_level()) {
            g2x_pv(ws->v + ((size_t)klo * j->kv + kh) * d, (size_t)j->kv * d, nk, scores[0], 8192, nq,
                   d, ws->a + ((size_t)t0 * 4 + hd) * d, (size_t)4 * d);
            continue;
        }
#endif
        for (int z = 0; z < d; z += 8) {
#if defined(__ARM_NEON)
            float32x4_t o[QB][2];
            for (int i = 0; i < QB; i++)
                o[i][0] = o[i][1] = vdupq_n_f32(0);
            for (int sk = 0; sk < nk; sk++) {
                const float *v = ws->v + ((size_t)(klo + sk) * j->kv + kh) * d + z;
                float32x4_t v0 = vld1q_f32(v), v1 = vld1q_f32(v + 4);
                for (int i = 0; i < nq; i++) {
                    float pr = scores[i][sk];
                    o[i][0] = vfmaq_n_f32(o[i][0], v0, pr);
                    o[i][1] = vfmaq_n_f32(o[i][1], v1, pr);
                }
            }
            for (int i = 0; i < nq; i++) {
                float *out = ws->a + ((size_t)(t0 + i) * 4 + hd) * d + z;
                vst1q_f32(out, o[i][0]);
                vst1q_f32(out + 4, o[i][1]);
            }
#else
            for (int i = 0; i < nq; i++) {
                float *out = ws->a + ((size_t)(t0 + i) * 4 + hd) * d + z;
                memset(out, 0, 8 * 4);
                for (int sk = 0; sk < nk; sk++) {
                    const float *v = ws->v + ((size_t)(klo + sk) * j->kv + kh) * d + z;
                    for (int c = 0; c < 8; c++)
                        out[c] += scores[i][sk] * v[c];
                }
            }
#endif
        }
    }
}

/* ---- graph -------------------------------------------------------------- */

ei_gemma2_fast *ei_gemma2_fast_create(void) {
    ei_gemma2_fast *ctx = ei_xcalloc(1, sizeof *ctx);
    pthread_mutex_init(&ctx->lock, NULL);
    return ctx;
}
void ei_gemma2_fast_free(ei_gemma2_fast *ctx) {
    if (!ctx)
        return;
    workspace *ws = &ctx->ws;
    free(ws->x), free(ws->h), free(ws->q), free(ws->k), free(ws->v), free(ws->a), free(ws->up);
    free(ws->gate), free(ws->ple), free(ws->xq), free(ws->lo), free(ws->hi), free(ws->pos);
    free(ws->xpair_q), free(ws->xpair_s), free(ws->x512_u), free(ws->x512_s);
    pthread_mutex_destroy(&ctx->lock);
    free(ctx);
}
bool ei_gemma2_embed_batch_fast(const ei_gguf *g, ei_thread_pool *pool, const int32_t *ids,
                                const float *inputs, const size_t *offsets, size_t batch,
                                float *out, char *err, size_t err_len) {
    return ei_gemma2_embed_batch_fast_ctx(&default_ctx, g, pool, ids, inputs, offsets, batch, out,
                                          err, err_len);
}
bool ei_gemma2_embed_batch_fast_ctx(ei_gemma2_fast *ctx, const ei_gguf *g, ei_thread_pool *pool,
                                    const int32_t *ids, const float *inputs,
                                    const size_t *offsets, size_t batch, float *out, char *err,
                                    size_t err_len) {
    size_t count = offsets[batch];
    if (!count || count > 8192) {
        snprintf(err, err_len, "EmbeddingGemma 2 accepts 1..8192 input tokens");
        return false;
    }
    pthread_once(&rope_once, build_rope);
    pthread_mutex_lock(&ctx->lock);
    workspace *ws = &ctx->ws;
    reserve(ws, count);
    int n = (int)count;
    for (size_t s = 0; s < batch; s++)
        for (size_t t = offsets[s]; t < offsets[s + 1]; t++) {
            ws->lo[t] = (uint32_t)offsets[s];
            ws->hi[t] = (uint32_t)offsets[s + 1];
            ws->pos[t] = (uint32_t)(t - offsets[s]);
        }
    const ei_tensor *emb = tensor_of(g, "token_embd.weight");
    if (inputs)
        memcpy(ws->x, inputs, count * 512 * 4);
    for (int t = 0; t < n && !inputs; t++) {
        const ei_block_q8_0 *row = (const ei_block_q8_0 *)emb->data + (size_t)ids[t] * 16;
        // (d * q) * sqrt(512), the reference and media rounding order.
        for (int b = 0; b < 16; b++) {
            float d = half_to_float(row[b].d);
            for (int z = 0; z < 32; z++)
                ws->x[(size_t)t * 512 + b * 32 + z] = d * row[b].qs[z] * sqrtf(512.0f);
        }
    }
    gemm_f(pool, tensor_of(g, "per_layer_model_proj.weight"), ws->x, n, ws->ple);
    for (size_t i = 0; i < count * 512 * 24; i++)
        ws->ple[i] /= sqrtf(512.0f);
    dequant_norm(tensor_of(g, "per_layer_proj_norm.weight"), 512, ws->norm_w[0]);
    for (size_t r = 0; r < count * 24; r++)
        rms_row(ws->ple + r * 512, 512, ws->norm_w[0]);
    for (int l = 0; l < 24; l++) {
        char name[128];
#define T(suffix) (snprintf(name, sizeof name, "blk.%d." suffix ".weight", l), tensor_of(g, name))
        int d = l % 6 == 5 ? 512 : 256, kv = 512 / d;
        dequant_norm(T("attn_norm"), 512, ws->norm_w[2]);
        rows(ws, pool, n, (row_job){.op = ROW_NORM_QUANT, .in = ws->x, .out = ws->h, .w = ws->norm_w[2]});
        gemm_q8(ws, pool, T("attn_q"), ws->xq, n, ws->q);
        gemm_q8(ws, pool, T("attn_k"), ws->xq, n, ws->k);
        gemm_q8(ws, pool, T("attn_v"), ws->xq, n, ws->v);
        dequant_norm(T("attn_q_norm"), d, ws->norm_w[0]);
        dequant_norm(T("attn_k_norm"), d, ws->norm_w[1]);
        rows(ws, pool, n,
             (row_job){.op = ROW_QKV_POST, .d = d, .kv = kv, .rope_base = l % 6 == 5 ? 1 : 0});
        attn_job aj = {n, kv, d, l % 6 == 5 ? 0 : 512, ws};
        PROF(3, ei_parallel_for(pool, 4 * ((n + QB - 1) / QB), 1, attention_rows, &aj));
        rows(ws, pool, n, (row_job){.op = ROW_QUANT, .in = ws->a, .width = 4 * d});
        gemm_q8(ws, pool, T("attn_output"), ws->xq, n, ws->h);
        dequant_norm(T("post_attention_norm"), 512, ws->norm_w[2]);
        rows(ws, pool, n, (row_job){.op = ROW_POST_ADD, .out = ws->h, .w = ws->norm_w[2], .scale = 1});
        dequant_norm(T("ffn_norm"), 512, ws->norm_w[2]);
        rows(ws, pool, n, (row_job){.op = ROW_NORM_QUANT, .in = ws->x, .out = ws->h, .w = ws->norm_w[2]});
        gemm_q8(ws, pool, T("ffn_up"), ws->xq, n, ws->up);
        gemm_q8(ws, pool, T("ffn_gate"), ws->xq, n, ws->gate);
        rows(ws, pool, n, (row_job){.op = ROW_GELU_QUANT});
        gemm_q8(ws, pool, T("ffn_down"), ws->xq, n, ws->h);
        dequant_norm(T("post_ffw_norm"), 512, ws->norm_w[2]);
        rows(ws, pool, n, (row_job){.op = ROW_POST_ADD, .out = ws->h, .w = ws->norm_w[2], .scale = 1});
        rows(ws, pool, n, (row_job){.op = ROW_QUANT, .in = ws->x, .width = 512});
        gemm_q8(ws, pool, T("inp_gate"), ws->xq, n, ws->h);
        rows(ws, pool, n, (row_job){.op = ROW_PLE_GATE_QUANT, .layer = l});
        gemm_q8(ws, pool, T("proj"), ws->xq, n, ws->a);
        dequant_norm(T("post_norm"), 512, ws->norm_w[2]);
        float scale = element(T("layer_output_scale"), 0);
        rows(ws, pool, n, (row_job){.op = ROW_POST_ADD, .out = ws->a, .w = ws->norm_w[2], .scale = scale});
#undef T
    }
    dequant_norm(tensor_of(g, "output_norm.weight"), 512, ws->norm_w[2]);
    rows(ws, pool, n, (row_job){.op = ROW_NORM_QUANT, .in = ws->x, .out = ws->h, .w = ws->norm_w[2]});
    const ei_tensor *output = tensor_of(g, "output.weight");
    if (output->type == EI_T_Q8_0)
        gemm_q8(ws, pool, output, ws->xq, n, ws->a);
    else
        gemm_f(pool, output, ws->h, n, ws->a);
    for (size_t s = 0; s < batch; s++) {
        float *o = out + s * 768;
        size_t len = offsets[s + 1] - offsets[s];
        memset(o, 0, 768 * 4);
        for (size_t t = offsets[s]; t < offsets[s + 1]; t++)
            for (int z = 0; z < 768; z++)
                o[z] += ws->a[t * 768 + z] / (float)len;
        ei_l2_normalize(o, 768);
    }
    if (getenv("EI_GEMMA2_CPU_PROFILE")) {
        for (int i = 0; i < 6; i++)
            fprintf(stderr, "%s %.2f ms  ", prof_names[i], prof[i]);
        fprintf(stderr, "\n");
    }
    memset(prof, 0, sizeof prof);
    pthread_mutex_unlock(&ctx->lock);
    return true;
}
