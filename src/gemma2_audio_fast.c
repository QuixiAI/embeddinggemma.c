/* cpu-max fork: SIMD + threaded CPU audio tower for EmbeddingGemma 2.
 *
 * Upstream runs the CPU audio tower in the scalar reference graph (gemma2.c:
 * per-element GEMM loops, both subsampling convolutions through a per-element
 * type switch, single-threaded norms/clamps/activations/quantization). This
 * path keeps the reference math -- Q8_0 activations for Q8_0 weights, the
 * clipped-linear input/output clamps, fp32 convolutions, norms and local
 * attention, the reference's tensor mapping -- and executes it with:
 *   - the subsampling convolutions as channel-vectorized AVX2 kernels with
 *     fused LayerNorm + ReLU,
 *   - Q8_0 GEMMs on the fast graph's routes (AVX-VNNI 256 repacked here),
 *   - per-token fused row passes (norm + clamp + quantize, clamp + SiLU +
 *     quantize, GLU, causal depthwise conv + norm + SiLU + quantize,
 *     post-norm + residual), threaded over tokens,
 *   - relative-position keys computed once per layer and cached,
 *   - a persistent workspace.
 * EI_GEMMA2_CPU_MEDIA_FAST=0 selects the reference graph. */
#include "gemma2.h"
#include "gemma2_fast_x86.h"
#include "gemma2_media_x86.h"
#include "kernels.h"
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define AVX2 __attribute__((target("avx2,fma,f16c")))
#define UNROLL _Pragma("GCC unroll 8")
#endif

enum { AD = 1024, AF = 4096, AHEADS = 8, AHD = 128, CTX = 12, NREL = 13 };

/* EI_GEMMA2_CPU_PROFILE=1: per-stage wall time of each audio call on stderr. */
#include <time.h>
enum { AP_CONV, AP_GEMM, AP_ROWS, AP_ATTN, AP_N };
static const char *ap_names[AP_N] = {"subsample", "gemm", "rows", "attention"};
static double ap_ms[AP_N];
static int ap_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("EI_GEMMA2_CPU_PROFILE");
        on = e && strcmp(e, "0") != 0;
    }
    return on;
}
static double ap_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}
#define AP(i, ...)                                                                                 \
    do {                                                                                           \
        double ap_t0_ = ap_on() ? ap_now() : 0;                                                    \
        __VA_ARGS__;                                                                               \
        if (ap_on())                                                                               \
            ap_ms[i] += ap_now() - ap_t0_;                                                         \
    } while (0)

static const ei_tensor *tensor_of(const ei_gguf *g, const char *name) {
    return ei_gguf_tensor(g, name, true);
}
static const float *f32_of(const ei_gguf *g, const char *name) {
    const ei_tensor *t = tensor_of(g, name);
    if (t->type != EI_T_F32)
        ei_die("EmbeddingGemma 2 audio: %s must be F32", name);
    return t->data;
}
static float scalar_or(const ei_gguf *g, const char *name, float fallback) {
    const ei_tensor *t = ei_gguf_tensor(g, name, false);
    return t ? ((const float *)t->data)[0] : fallback;
}

/* A clipped linear: Q8_0 weight plus optional input/output clamps. */
typedef struct {
    const ei_tensor *w;
    float in_lo, in_hi, out_lo, out_hi;
} clinear;
static clinear clinear_of(const ei_gguf *g, const char *base) {
    char key[160];
    clinear c;
    snprintf(key, sizeof key, "%s.weight", base);
    c.w = tensor_of(g, key);
    snprintf(key, sizeof key, "%s.input_min", base);
    c.in_lo = scalar_or(g, key, -INFINITY);
    snprintf(key, sizeof key, "%s.input_max", base);
    c.in_hi = scalar_or(g, key, INFINITY);
    snprintf(key, sizeof key, "%s.output_min", base);
    c.out_lo = scalar_or(g, key, -INFINITY);
    snprintf(key, sizeof key, "%s.output_max", base);
    c.out_hi = scalar_or(g, key, INFINITY);
    return c;
}

/* ---- workspace ------------------------------------------------------------ */

typedef struct {
    size_t cap;                         /* tokens */
    float *x, *h, *hn, *q, *k, *v, *a;  /* [cap][1024] */
    float *up;                          /* [cap][4096] */
    ei_block_q8_0 *xq;                  /* [cap][4096/32] */
    size_t conv1_cap;
    float *c1;                          /* subsampled activations [t1][32][32] */
    float rel[12][NREL * AD];           /* per-layer relative keys */
    const void *rel_key;                /* projector the cache belongs to */
    float qscale[12][AHD], kscale;
    float w0t[9][128];                  /* conv0 weights, [tap][out] */
    float *w1t;                         /* conv1 weights, [tap][in][out] = [9][128][32] */
} audio_ws;

static pthread_mutex_t audio_lock = PTHREAD_MUTEX_INITIALIZER;
static audio_ws aws;
static ei_gemma2_fast *gemm_ctx;

static void reserve(audio_ws *w, size_t n, size_t conv1_floats) {
    if (w->cap < n) {
        free(w->x), free(w->h), free(w->hn), free(w->q), free(w->k), free(w->v), free(w->a);
        free(w->up), free(w->xq);
        size_t row = (size_t)AD * sizeof(float);
        w->x = ei_xmalloc(n * row);
        w->h = ei_xmalloc(n * row);
        w->hn = ei_xmalloc(n * row);
        w->q = ei_xmalloc(n * row);
        w->k = ei_xmalloc(n * row);
        w->v = ei_xmalloc(n * row);
        w->a = ei_xmalloc(n * row);
        w->up = ei_xmalloc(n * AF * sizeof(float));
        w->xq = ei_xmalloc(n * (AF / 32) * sizeof(ei_block_q8_0));
        w->cap = n;
    }
    if (w->conv1_cap < conv1_floats) {
        free(w->c1);
        w->c1 = ei_xmalloc(conv1_floats * sizeof(float));
        w->conv1_cap = conv1_floats;
    }
}

/* Constant per projector: relative keys of the 13 positions (the reference
 * recomputes them per call with the same Q8 GEMM), q/k scales, conv weights. */
static void prepare(audio_ws *w, const ei_gguf *g, ei_thread_pool *pool) {
    if (w->rel_key == g)
        return;
    float pos[NREL * AD];
    for (int p = 0; p < NREL; ++p)
        for (int z = 0; z < 512; ++z) {
            float theta = (12 - p) * expf(-z * logf(10000.0f) / 511);
            pos[p * AD + z] = sinf(theta);
            pos[p * AD + z + 512] = cosf(theta);
        }
    ei_block_q8_0 posq[NREL * AD / 32];
    for (int p = 0; p < NREL; ++p)
        ei_quantize_row_q8_0(pos + p * AD, posq + p * (AD / 32), AD);
    for (int l = 0; l < 12; ++l) {
        char name[128];
        snprintf(name, sizeof name, "a.blk.%d.attn_k_rel.weight", l);
        ei_gemma2_fast_gemm_q8(gemm_ctx, pool, tensor_of(g, name), posq, NREL, w->rel[l]);
        snprintf(name, sizeof name, "a.blk.%d.per_dim_scale.weight", l);
        const float *qs = f32_of(g, name);
        for (int z = 0; z < AHD; ++z)
            w->qscale[l][z] = qs[z] / (sqrtf(128.0f) * logf(2.0f));
    }
    w->kscale = log1pf(expf(1.0f)) / logf(2.0f);
    const float *c0 = f32_of(g, "a.conv1d.0.weight"), *c1 = f32_of(g, "a.conv1d.1.weight");
    for (int o = 0; o < 128; ++o)
        for (int tap = 0; tap < 9; ++tap)
            w->w0t[tap][o] = c0[(size_t)o * 9 + tap];
    free(w->w1t);
    w->w1t = ei_xmalloc((size_t)9 * 128 * 32 * sizeof(float));
    for (int o = 0; o < 32; ++o)
        for (int c = 0; c < 128; ++c)
            for (int tap = 0; tap < 9; ++tap)
                w->w1t[((size_t)tap * 128 + c) * 32 + o] = c1[((size_t)o * 128 + c) * 9 + tap];
    w->rel_key = g;
}

/* ---- subsampling convolutions ------------------------------------------------ */


static void layer_norm_relu(float *r, int c, const float *w) {
    double mean = 0, var = 0;
    for (int i = 0; i < c; ++i)
        mean += r[i];
    mean /= c;
    for (int i = 0; i < c; ++i)
        var += (r[i] - mean) * (r[i] - mean);
    float scale = 1.0f / sqrtf((float)(var / c) + 1e-6f);
    for (int i = 0; i < c; ++i)
        r[i] = fmaxf(0, ((float)(r[i] - mean)) * scale * w[i]);
}

/* conv0: 1 -> 128 channels, 3x3, stride 2, pad 1, over (time, mel): one
 * output time row (64 mel positions x 128 channels), LayerNorm + ReLU. */
static void conv0_row(const audio_ws *w, const float *mel, int frames, int t, const float *norm, float *dst) {
    for (int f = 0; f < 64; f++) {
        float acc[128] = {0};
        for (int ky = 0; ky < 3; ky++)
            for (int kx = 0; kx < 3; kx++) {
                int ti = t * 2 + ky - 1, fi = f * 2 + kx - 1;
                if (ti < 0 || ti >= frames || fi < 0 || fi >= 128)
                    continue;
                float x = mel[(size_t)ti * 128 + fi];
                const float *wr = w->w0t[ky * 3 + kx];
                for (int o = 0; o < 128; o++)
                    acc[o] += x * wr[o];
            }
        float *d = dst + (size_t)f * 128;
        memcpy(d, acc, sizeof acc);
        layer_norm_relu(d, 128, norm);
    }
}

#if defined(__x86_64__) || defined(_M_X64)
/* conv1 output (t, f) = 32 channels in 4 ymm; 1152 taps x inputs. `in` holds
 * conv0 rows [row0, ...); bounds use the global time index. */
AVX2 static void conv1_point_avx2(const float *in, int row0, int nt, int nf, int t, int f, const float *w1t,
                                  float *out) {
    __m256 acc[4];
    UNROLL for (int o = 0; o < 4; o++) acc[o] = _mm256_setzero_ps();
    for (int ky = 0; ky < 3; ky++)
        for (int kx = 0; kx < 3; kx++) {
            int ti = t * 2 + ky - 1, fi = f * 2 + kx - 1;
            if (ti < 0 || ti >= nt || fi < 0 || fi >= nf)
                continue;
            const float *x = in + ((size_t)(ti - row0) * nf + fi) * 128;
            const float *wr = w1t + (size_t)(ky * 3 + kx) * 128 * 32;
            for (int c = 0; c < 128; c++) {
                __m256 xb = _mm256_broadcast_ss(x + c);
                UNROLL for (int o = 0; o < 4; o++)
                    acc[o] = _mm256_fmadd_ps(xb, _mm256_loadu_ps(wr + c * 32 + o * 8), acc[o]);
            }
        }
    UNROLL for (int o = 0; o < 4; o++) _mm256_storeu_ps(out + o * 8, acc[o]);
}
#endif

/* Both subsampling convolutions, streamed: a task producing conv1 rows
 * [begin, end) computes the conv0 rows they read (2*begin-1 .. 2*end-1) into a
 * task-local buffer instead of materializing conv0 for the whole clip (49 MB
 * at 30 s); boundary rows are recomputed by neighbouring tasks. */
typedef struct {
    const audio_ws *w;
    const float *mel, *norm0, *norm1;
    float *out;
    int frames, t0, t1;
} conv_job;
static void conv01_rows(void *opaque, int32_t begin, int32_t end) {
    conv_job *j = opaque;
    int r0 = 2 * begin - 1 < 0 ? 0 : 2 * begin - 1, r1 = 2 * (end - 1) + 2 < j->t0 ? 2 * (end - 1) + 2 : j->t0;
    static _Thread_local float *buf; /* per pool thread, kept: <= (2 * grain + 1) conv0 rows */
    static _Thread_local size_t cap;
    size_t need = (size_t)(r1 - r0) * 64 * 128;
    if (cap < need) {
        free(buf);
        buf = ei_xmalloc(need * sizeof(float));
        cap = need;
    }
    for (int r = r0; r < r1; r++)
        conv0_row(j->w, j->mel, j->frames, r, j->norm0, buf + (size_t)(r - r0) * 64 * 128);
    for (int t = begin; t < end; t++)
        for (int f = 0; f < 32; f++) {
            float *dst = j->out + ((size_t)t * 32 + f) * 32;
#if defined(__x86_64__) || defined(_M_X64)
            if (g2x_level() >= G2X_AVX2)
                conv1_point_avx2(buf, r0, j->t0, 64, t, f, j->w->w1t, dst);
            else
#endif
            {
                float acc[32] = {0};
                for (int tap = 0; tap < 9; tap++) {
                    int ti = t * 2 + tap / 3 - 1, fi = f * 2 + tap % 3 - 1;
                    if (ti < 0 || ti >= j->t0 || fi < 0 || fi >= 64)
                        continue;
                    const float *x = buf + ((size_t)(ti - r0) * 64 + fi) * 128;
                    for (int c = 0; c < 128; c++)
                        for (int o = 0; o < 32; o++)
                            acc[o] += x[c] * j->w->w1t[((size_t)tap * 128 + c) * 32 + o];
                }
                memcpy(dst, acc, sizeof acc);
            }
            layer_norm_relu(dst, 32, j->norm1);
        }
}

/* ---- row passes ------------------------------------------------------------- */

typedef enum {
    R_NORM,          /* hn = rms(x) * w */
    R_QUANT,         /* xq = q8(clamp(src, lo, hi)), width wd */
    R_SILU_QUANT,    /* up = silu(clamp(up, olo, ohi)); xq = q8(clamp(up, lo, hi)) */
    R_POST_ADD,      /* x += sc * rms(clamp(h, olo, ohi)) * w */
    R_ADD,           /* x += clamp(h, olo, ohi) (light-conv output, no norm) */
    R_QKV,           /* clamp q/k/v outputs; scale q per dim, k by kscale */
    R_GLU,           /* h = a * sigmoid(b), a/b = clamp(up)[0:1024] / [1024:2048] */
    R_DWCONV,        /* a = rms(causal dwconv5(h)) * w; a = silu(a); xq = q8(clamp(a)) */
    R_FINAL_NORM,    /* x = rms(x) * w */
    R_BIAS_NORM_QUANT, /* up = rms(clamp(up) + bias) (1536); xq = q8(clamp(up, lo, hi)) */
} arow;
typedef struct {
    arow op;
    audio_ws *w;
    const float *nw, *bias;
    float *src;
    int wd, layer;
    float lo, hi, olo, ohi, sc;
} arow_job;

static void rms_into(float *y, const float *x, int d, const float *w) {
    float scale = 1.0f / sqrtf(ei_gemma2_fast_sum_squares(x, d) / d + 1e-6f);
    if (w)
        for (int z = 0; z < d; z++)
            y[z] = x[z] * (scale * w[z]);
    else
        for (int z = 0; z < d; z++)
            y[z] = x[z] * scale;
}
static void clamp_row(float *x, int d, float lo, float hi) {
    if (lo == -INFINITY && hi == INFINITY)
        return;
    for (int z = 0; z < d; z++)
        x[z] = fminf(hi, fmaxf(lo, x[z]));
}
static void quant_clamped(const float *x, int d, float lo, float hi, ei_block_q8_0 *q) {
    if (lo == -INFINITY && hi == INFINITY) {
        ei_gemma2_fast_quant_row(x, q, d);
        return;
    }
    float tmp[AF];
    for (int z = 0; z < d; z++)
        tmp[z] = fminf(hi, fmaxf(lo, x[z]));
    ei_gemma2_fast_quant_row(tmp, q, d);
}
static void arow_tokens(void *opaque, int32_t begin, int32_t end) {
    arow_job *j = opaque;
    audio_ws *w = j->w;
    float tmp[AF];
    for (int t = begin; t < end; t++) {
        float *x = w->x + (size_t)t * AD;
        switch (j->op) {
        case R_NORM:
            rms_into(w->hn + (size_t)t * AD, x, AD, j->nw);
            break;
        case R_QUANT:
            quant_clamped(j->src + (size_t)t * j->wd, j->wd, j->lo, j->hi, w->xq + (size_t)t * (j->wd / 32));
            break;
        case R_SILU_QUANT: {
            float *u = w->up + (size_t)t * AF;
            clamp_row(u, AF, j->olo, j->ohi);
            g2m_silu(u, u, AF);
            quant_clamped(u, AF, j->lo, j->hi, w->xq + (size_t)t * (AF / 32));
            break;
        }
        case R_POST_ADD: {
            float *h = w->h + (size_t)t * AD;
            clamp_row(h, AD, j->olo, j->ohi);
            rms_into(tmp, h, AD, j->nw);
            for (int z = 0; z < AD; z++)
                x[z] += j->sc * tmp[z];
            break;
        }
        case R_ADD: {
            float *h = w->h + (size_t)t * AD;
            clamp_row(h, AD, j->olo, j->ohi);
            for (int z = 0; z < AD; z++)
                x[z] += h[z];
            break;
        }
        case R_QKV: {
            float *q = w->q + (size_t)t * AD, *k = w->k + (size_t)t * AD, *v = w->v + (size_t)t * AD;
            clamp_row(q, AD, j->olo, j->ohi);
            clamp_row(k, AD, j->olo, j->ohi);
            clamp_row(v, AD, j->olo, j->ohi);
            const float *qs = w->qscale[j->layer];
            for (int z = 0; z < AD; z++) {
                q[z] *= qs[z % AHD];
                k[z] *= w->kscale;
            }
            break;
        }
        case R_GLU: {
            float *u = w->up + (size_t)t * 2 * AD, *h = w->h + (size_t)t * AD; /* pw1 rows are 2048 wide */
            clamp_row(u, 2 * AD, j->olo, j->ohi);
            for (int z = 0; z < AD; z++)
                h[z] = u[z] / (1 + expf(-u[AD + z]));
            break;
        }
        case R_DWCONV: {
            const float *dw = j->bias; /* [1024][5] */
            float *a = w->a + (size_t)t * AD;
            for (int z = 0; z < AD; z++) {
                float sum = 0;
                for (int i = 0; i < 5; i++)
                    if (t + i >= 4)
                        sum += w->h[(size_t)(t + i - 4) * AD + z] * dw[z * 5 + i];
                a[z] = sum;
            }
            rms_into(a, a, AD, j->nw);
            g2m_silu(a, a, AD);
            quant_clamped(a, AD, j->lo, j->hi, w->xq + (size_t)t * (AD / 32));
            break;
        }
        case R_FINAL_NORM:
            rms_into(x, x, AD, j->nw);
            break;
        case R_BIAS_NORM_QUANT: {
            float *u = w->up + (size_t)t * 1536;
            clamp_row(u, 1536, j->olo, j->ohi);
            for (int z = 0; z < 1536; z++)
                u[z] += j->bias[z];
            rms_into(u, u, 1536, NULL);
            quant_clamped(u, 1536, j->lo, j->hi, w->xq + (size_t)t * (1536 / 32));
            break;
        }
        }
    }
}
static void arows(ei_thread_pool *pool, int n, arow_job job) { ei_parallel_for(pool, n, 4, arow_tokens, &job); }

/* ---- local relative-position attention ---------------------------------------- */

typedef struct {
    audio_ws *w;
    int n, layer;
} aattn_job;
static void aattn_tokens(void *opaque, int32_t begin, int32_t end) {
    aattn_job *j = opaque;
    audio_ws *w = j->w;
    const float *rel = w->rel[j->layer];
    for (int t = begin; t < end; t++)
        for (int hd = 0; hd < AHEADS; hd++) {
            const float *q = w->q + (size_t)t * AD + hd * AHD;
            float scores[CTX], mx = -INFINITY;
            int lo = t > CTX - 1 ? t - (CTX - 1) : 0;
            for (int s = lo; s <= t; s++) {
                const float *k = w->k + (size_t)s * AD + hd * AHD, *r = rel + (size_t)(12 - t + s) * AD + hd * AHD;
                float dot = 0;
                for (int z = 0; z < AHD; z++)
                    dot += q[z] * (k[z] + r[z]);
                dot = 50 * tanhf(dot / 50);
                scores[s - lo] = dot;
                mx = dot > mx ? dot : mx;
            }
            float total = 0;
            for (int s = lo; s <= t; s++) {
                scores[s - lo] = expf(scores[s - lo] - mx);
                total += scores[s - lo];
            }
            float *o = w->a + (size_t)t * AD + hd * AHD;
            memset(o, 0, AHD * sizeof(float));
            for (int s = lo; s <= t; s++) {
                float p = scores[s - lo] / total;
                const float *v = w->v + (size_t)s * AD + hd * AHD;
                for (int z = 0; z < AHD; z++)
                    o[z] += p * v[z];
            }
        }
}

/* ---- tower --------------------------------------------------------------------- */

static void create_ctx(void) { gemm_ctx = ei_gemma2_fast_create(); }

static void gemm(ei_thread_pool *pool, const clinear *c, const ei_block_q8_0 *xq, int n, float *y) {
    AP(AP_GEMM, ei_gemma2_fast_gemm_q8(gemm_ctx, pool, c->w, xq, n, y));
}

bool ei_gemma2_audio_fast(const ei_gguf *g, ei_thread_pool *pool, const float *mel, int frames,
                          float **output, size_t *tokens, char *err, size_t err_len) {
    if (frames < 1 || frames > 3000) {
        snprintf(err, err_len, "audio chunk must contain 1..3000 mel frames");
        return false;
    }
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, create_ctx);
    pthread_mutex_lock(&audio_lock);
    ei_cpu_binding binding; /* hybrid pools: the caller joins on a P-core */
    ei_thread_pool_bind_caller(pool, &binding);
    audio_ws *w = &aws;
    const int t0 = (frames + 1) / 2, t1 = (t0 + 1) / 2, n = t1; /* conv0: t0 x 64 mel x 128 ch; conv1: t1 x 32 x 32 */
    reserve(w, (size_t)n, (size_t)t1 * 32 * 32);
    prepare(w, g, pool);

    /* Subsampling: (frames x 128) -> (t0 x 64 x 128) -> (t1 x 32 x 32), LayerNorm + ReLU. */
    conv_job cj = {w, mel, f32_of(g, "a.conv1d.0.norm.weight"), f32_of(g, "a.conv1d.1.norm.weight"), w->c1,
                   frames, t0, t1};
    AP(AP_CONV, ei_parallel_for(pool, t1, 4, conv01_rows, &cj));
    AP(AP_GEMM, ei_gemma2_fast_gemm_f(pool, tensor_of(g, "a.input_projection.weight"), w->c1, n, w->x));

    for (int l = 0; l < 12; ++l) {
        char base[96], name[128];
#define CL(suffix) (snprintf(base, sizeof base, "a.blk.%d." suffix, l), clinear_of(g, base))
#define AF32(suffix) (snprintf(name, sizeof name, "a.blk.%d." suffix ".weight", l), f32_of(g, name))
        clinear up = CL("ffn_up"), down = CL("ffn_down"), q = CL("attn_q"), k = CL("attn_k"), v = CL("attn_v");
        clinear out = CL("attn_out"), pw1 = CL("conv_pw1"), pw2 = CL("conv_pw2");
        clinear up1 = CL("ffn_up_1"), down1 = CL("ffn_down_1");
        /* Half-step FFN 1. */
        AP(AP_ROWS, arows(pool, n, (arow_job){.op = R_NORM, .w = w, .nw = AF32("ffn_norm")});
           arows(pool, n, (arow_job){.op = R_QUANT, .w = w, .src = w->hn, .wd = AD, .lo = up.in_lo, .hi = up.in_hi}));
        gemm(pool, &up, w->xq, n, w->up);
        AP(AP_ROWS, arows(pool, n, (arow_job){.op = R_SILU_QUANT, .w = w, .olo = up.out_lo, .ohi = up.out_hi,
                                              .lo = down.in_lo, .hi = down.in_hi}));
        gemm(pool, &down, w->xq, n, w->h);
        AP(AP_ROWS, arows(pool, n, (arow_job){.op = R_POST_ADD, .w = w, .nw = AF32("ffn_post_norm"), .sc = 0.5f,
                                              .olo = down.out_lo, .ohi = down.out_hi}));
        /* Local attention with relative positions. */
        AP(AP_ROWS, arows(pool, n, (arow_job){.op = R_NORM, .w = w, .nw = AF32("attn_pre_norm")}));
        const clinear *qkv[3] = {&q, &k, &v};
        float *dst[3] = {w->q, w->k, w->v};
        float qlo = NAN, qhi = NAN;
        for (int i = 0; i < 3; i++) {
            if (!(qkv[i]->in_lo == qlo && qkv[i]->in_hi == qhi)) /* q/k/v usually share clamps */
                AP(AP_ROWS, arows(pool, n, (arow_job){.op = R_QUANT, .w = w, .src = w->hn, .wd = AD,
                                                      .lo = qkv[i]->in_lo, .hi = qkv[i]->in_hi}));
            qlo = qkv[i]->in_lo, qhi = qkv[i]->in_hi;
            gemm(pool, qkv[i], w->xq, n, dst[i]);
        }
        if (!(q.out_lo == k.out_lo && k.out_lo == v.out_lo && q.out_hi == k.out_hi && k.out_hi == v.out_hi))
            ei_die("EmbeddingGemma 2 audio: q/k/v output clamps differ"); /* not seen in any checkpoint */
        AP(AP_ROWS, arows(pool, n, (arow_job){.op = R_QKV, .w = w, .layer = l, .olo = q.out_lo, .ohi = q.out_hi}));
        aattn_job aj = {w, n, l};
        AP(AP_ATTN, ei_parallel_for(pool, n, 4, aattn_tokens, &aj));
        AP(AP_ROWS, arows(pool, n, (arow_job){.op = R_QUANT, .w = w, .src = w->a, .wd = AD, .lo = out.in_lo,
                                              .hi = out.in_hi}));
        gemm(pool, &out, w->xq, n, w->h);
        AP(AP_ROWS, arows(pool, n, (arow_job){.op = R_POST_ADD, .w = w, .nw = AF32("attn_post_norm"), .sc = 1.0f,
                                              .olo = out.out_lo, .ohi = out.out_hi}));
        /* Light convolution. The GGUF conversion swaps norm_conv and conv_norm;
         * the reference loader's mapping is used (conv_norm before pw1). */
        AP(AP_ROWS, arows(pool, n, (arow_job){.op = R_NORM, .w = w, .nw = AF32("conv_norm")});
           arows(pool, n, (arow_job){.op = R_QUANT, .w = w, .src = w->hn, .wd = AD, .lo = pw1.in_lo, .hi = pw1.in_hi}));
        gemm(pool, &pw1, w->xq, n, w->up);
        AP(AP_ROWS, arows(pool, n, (arow_job){.op = R_GLU, .w = w, .olo = pw1.out_lo, .ohi = pw1.out_hi});
           arows(pool, n, (arow_job){.op = R_DWCONV, .w = w, .bias = AF32("conv_dw"), .nw = AF32("norm_conv"),
                                     .lo = pw2.in_lo, .hi = pw2.in_hi}));
        gemm(pool, &pw2, w->xq, n, w->h);
        AP(AP_ROWS, arows(pool, n, (arow_job){.op = R_ADD, .w = w, .olo = pw2.out_lo, .ohi = pw2.out_hi}));
        /* Half-step FFN 2, then the block's output norm. */
        AP(AP_ROWS, arows(pool, n, (arow_job){.op = R_NORM, .w = w, .nw = AF32("ffn_norm_1")});
           arows(pool, n, (arow_job){.op = R_QUANT, .w = w, .src = w->hn, .wd = AD, .lo = up1.in_lo, .hi = up1.in_hi}));
        gemm(pool, &up1, w->xq, n, w->up);
        AP(AP_ROWS, arows(pool, n, (arow_job){.op = R_SILU_QUANT, .w = w, .olo = up1.out_lo, .ohi = up1.out_hi,
                                              .lo = down1.in_lo, .hi = down1.in_hi}));
        gemm(pool, &down1, w->xq, n, w->h);
        AP(AP_ROWS, arows(pool, n, (arow_job){.op = R_POST_ADD, .w = w, .nw = AF32("ffn_post_norm_1"), .sc = 0.5f,
                                              .olo = down1.out_lo, .ohi = down1.out_hi});
           arows(pool, n, (arow_job){.op = R_FINAL_NORM, .w = w, .nw = AF32("ln2")}));
#undef CL
#undef AF32
    }
    clinear pre = clinear_of(g, "a.pre_encode.out"), proj = clinear_of(g, "mm.a.input_projection");
    AP(AP_ROWS, arows(pool, n, (arow_job){.op = R_QUANT, .w = w, .src = w->x, .wd = AD, .lo = pre.in_lo,
                                          .hi = pre.in_hi}));
    gemm(pool, &pre, w->xq, n, w->up);
    AP(AP_ROWS, arows(pool, n, (arow_job){.op = R_BIAS_NORM_QUANT, .w = w, .bias = f32_of(g, "a.pre_encode.out.bias"),
                                          .olo = pre.out_lo, .ohi = pre.out_hi, .lo = proj.in_lo, .hi = proj.in_hi}));
    *output = ei_xmalloc((size_t)n * 512 * sizeof(float));
    *tokens = (size_t)n;
    gemm(pool, &proj, w->xq, n, *output);
    for (size_t i = 0; i < (size_t)n * 512; i++)
        (*output)[i] = fminf(proj.out_hi, fmaxf(proj.out_lo, (*output)[i]));
    if (ap_on()) {
        double total = 0;
        for (int i = 0; i < AP_N; i++)
            total += ap_ms[i];
        fprintf(stderr, "audio-fast frames=%d tokens=%d total=%.1f ms:", frames, n, total);
        for (int i = 0; i < AP_N; i++)
            fprintf(stderr, " %s=%.1f", ap_names[i], ap_ms[i]);
        fprintf(stderr, "\n");
        memset(ap_ms, 0, sizeof ap_ms);
    }
    ei_thread_pool_unbind_caller(&binding);
    pthread_mutex_unlock(&audio_lock);
    return true;
}
