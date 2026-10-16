/* cpu-max fork: SIMD + threaded CPU vision tower for EmbeddingGemma 2.
 *
 * Upstream runs the CPU vision tower in the scalar reference graph
 * (gemma2.c: per-element GEMM loops, scalar attention with a calloc per job,
 * single-threaded norms/RoPE/activations/quantization). This path keeps the
 * reference math -- Q8_0 activations for Q8_0 weights, fp32 attention and
 * norms, the fp16-rounded patch input, the selected GELU variant -- and
 * changes only how it is executed:
 *   - Q8_0 GEMMs on the fast text graph's x86 VNNI/AVX2 (or NEON) tiles,
 *   - g2m_attn64: tiled online-softmax attention for the 64-wide heads,
 *   - per-token fused row passes (norm+quantize, q/k/v norms + axial RoPE from
 *     tables, GELU+quantize, post-norm+residual), threaded over tokens,
 *   - the 3072-wide FFN intermediate processed in token chunks, so its buffers
 *     are O(chunk) instead of O(patches),
 *   - a persistent workspace instead of nine callocs per image.
 * EI_GEMMA2_CPU_MEDIA_FAST=0 selects the reference graph. */
#include "gemma2.h"
#include "gemma2_fast_x86.h"
#include "gemma2_media_x86.h"
#include "kernels.h"
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

enum { VD = 768, VF = 3072, VHEADS = 12, FFN_CHUNK = 256 };

/* EI_GEMMA2_CPU_PROFILE=1: per-stage wall time of each vision call on stderr. */
#include <time.h>
enum { VP_PATCH, VP_GEMM_QKV, VP_ROWS, VP_ATTN, VP_GEMM_O, VP_GEMM_FFN, VP_POOL, VP_N };
static const char *vp_names[VP_N] = {"patch", "gemm_qkv", "rows", "attention", "gemm_out", "gemm_ffn", "pool_proj"};
static double vp_ms[VP_N];
static int vp_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("EI_GEMMA2_CPU_PROFILE");
        on = e && strcmp(e, "0") != 0;
    }
    return on;
}
static double vp_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}
#define VP(i, ...)                                                                                 \
    do {                                                                                           \
        double vp_t0_ = vp_on() ? vp_now() : 0;                                                    \
        __VA_ARGS__;                                                                               \
        if (vp_on())                                                                               \
            vp_ms[i] += vp_now() - vp_t0_;                                                         \
    } while (0)

bool ei_gemma2_cpu_media_fast(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *flag = getenv("EI_GEMMA2_CPU_MEDIA_FAST"); /* default on; "0" = reference */
        cached = !flag || strcmp(flag, "0");
    }
    return cached;
}

/* EI_GEMMA2_SEMANTICS=upstream restores upstream's three deviations from the
 * official model; the default ("official") follows the Transformers
 * implementation (measured against its FP32 embeddings: reports/). */
static void (*cpu_yield_hook)(void); /* set once at startup */
void ei_gemma2_set_cpu_yield(void (*hook)(void)) { cpu_yield_hook = hook; }
void ei_gemma2_cpu_yield(void) {
    if (cpu_yield_hook)
        cpu_yield_hook();
}

bool ei_gemma2_official_semantics(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("EI_GEMMA2_SEMANTICS");
        cached = !(e && !strcmp(e, "upstream"));
    }
    return cached;
}
bool ei_gemma2_vision_tanh_gelu(const ei_gguf *g) {
    const char *gelu = getenv("EI_GEMMA2_VISION_GELU"); /* explicit override: tanh | quick */
    if (gelu)
        return !strcmp(gelu, "tanh");
    /* The GGUF lacks clip.use_gelu, so upstream (like llama.cpp) selects QuickGELU;
     * the official config's vision hidden_activation is gelu_pytorch_tanh. */
    return ei_gemma2_official_semantics() || ei_gguf_kv_bool(g, "clip.use_gelu", false);
}

static const ei_tensor *tensor_of(const ei_gguf *g, const char *name) {
    return ei_gguf_tensor(g, name, true);
}
static const float *f32_of(const ei_gguf *g, const char *name) {
    const ei_tensor *t = tensor_of(g, name);
    if (t->type != EI_T_F32)
        ei_die("EmbeddingGemma 2 vision: %s must be F32", name);
    return t->data;
}

/* ---- workspace ------------------------------------------------------------ */

typedef struct {
    size_t cap;                    /* patches */
    float *x, *h, *q, *k, *v;      /* [cap][768]; v doubles as the attention output */
    float *up, *gate;              /* [FFN_CHUNK][3072] */
    ei_block_q8_0 *xq;             /* [max(cap*24, FFN_CHUNK*96)] */
    float *rope;                   /* [positions][32]: 16 cos then 16 sin */
    int rope_positions;
} vision_ws;

static pthread_mutex_t vision_lock = PTHREAD_MUTEX_INITIALIZER;
static vision_ws vws;
static ei_gemma2_fast *gemm_ctx;

static void reserve(vision_ws *w, size_t n) {
    if (w->cap >= n)
        return;
    free(w->x), free(w->h), free(w->q), free(w->k), free(w->v), free(w->xq);
    size_t row = (size_t)VD * sizeof(float);
    w->x = ei_xmalloc(n * row);
    w->h = ei_xmalloc(n * row);
    w->q = ei_xmalloc(n * row);
    w->k = ei_xmalloc(n * row);
    w->v = ei_xmalloc(n * row);
    size_t blocks = n * VD / 32 > (size_t)FFN_CHUNK * VF / 32 ? n * VD / 32 : (size_t)FFN_CHUNK * VF / 32;
    w->xq = ei_xmalloc(blocks * sizeof(ei_block_q8_0));
    if (!w->up) {
        w->up = ei_xmalloc((size_t)FFN_CHUNK * VF * sizeof(float));
        w->gate = ei_xmalloc((size_t)FFN_CHUNK * VF * sizeof(float));
    }
    w->cap = n;
}

/* theta = p * 100^(-2z/32), the reference's exact expression, z = 0..15. */
static void rope_table(vision_ws *w, int positions) {
    if (w->rope_positions >= positions)
        return;
    free(w->rope);
    w->rope = ei_xmalloc((size_t)positions * 32 * sizeof(float));
    for (int p = 0; p < positions; p++)
        for (int z = 0; z < 16; z++) {
            float theta = p * powf(100.0f, -2.0f * z / 32);
            w->rope[(size_t)p * 32 + z] = cosf(theta);
            w->rope[(size_t)p * 32 + 16 + z] = sinf(theta);
        }
    w->rope_positions = positions;
}

/* ---- row passes ------------------------------------------------------------- */

typedef enum { ROW_NORM_QUANT, ROW_QKV_POST, ROW_V_HEADS, ROW_QUANT, ROW_POST_ADD, ROW_GELU_QUANT, ROW_POOL } row_op;
typedef struct {
    row_op op;
    vision_ws *w;
    const float *nw, *qn, *kn; /* norm weights */
    float *src, *dst;
    int cols, out_cols, t0;    /* ROW_GELU_QUANT: first token of the chunk */
    int n;                     /* ROW_QKV_POST / ROW_V_HEADS: tokens (head-major planes) */
    bool tanh_gelu;
} row_job;

static void rms(float *y, const float *x, int d, const float *w) {
    float scale = 1.0f / sqrtf(ei_gemma2_fast_sum_squares(x, d) / d + 1e-6f);
    if (w)
        for (int z = 0; z < d; z++)
            y[z] = x[z] * (scale * w[z]);
    else
        for (int z = 0; z < d; z++)
            y[z] = x[z] * scale;
}
static void rope_head(float *r, const float *cx, const float *cy) {
    for (int axis = 0; axis < 2; axis++) {
        const float *c = axis ? cy : cx, *s = c + 16;
        float *h = r + axis * 32;
        for (int z = 0; z < 16; z++) {
            float a = h[z], b = h[z + 16];
            h[z] = a * c[z] - b * s[z];
            h[z + 16] = a * s[z] + b * c[z];
        }
    }
}
static void row_tokens(void *opaque, int32_t begin, int32_t end) {
    row_job *j = opaque;
    vision_ws *w = j->w;
    float tmp[VD];
    for (int t = begin; t < end; t++) {
        switch (j->op) {
        case ROW_NORM_QUANT: /* xq = q8(rms(x) * w) */
            rms(tmp, w->x + (size_t)t * VD, VD, j->nw);
            ei_gemma2_fast_quant_row(tmp, w->xq + (size_t)t * (VD / 32), VD);
            break;
        case ROW_QKV_POST: { /* q in place; k -> h as [head][n][64] (see attn_tasks) */
            const float *cx = w->rope + (size_t)(t % j->cols) * 32, *cy = w->rope + (size_t)(t / j->cols) * 32;
            for (int hd = 0; hd < VHEADS; hd++) {
                float *qr = w->q + (size_t)t * VD + hd * 64, *kh = w->h + ((size_t)hd * j->n + t) * 64;
                rms(qr, qr, 64, j->qn);
                rms(kh, w->k + (size_t)t * VD + hd * 64, 64, j->kn);
                rope_head(qr, cx, cy);
                rope_head(kh, cx, cy);
            }
            break;
        }
        case ROW_V_HEADS: /* v -> the k buffer (free after ROW_QKV_POST) as [head][n][64] */
            for (int hd = 0; hd < VHEADS; hd++)
                rms(w->k + ((size_t)hd * j->n + t) * 64, w->v + (size_t)t * VD + hd * 64, 64, NULL);
            break;
        case ROW_QUANT:
            ei_gemma2_fast_quant_row(j->src + (size_t)t * VD, w->xq + (size_t)t * (VD / 32), VD);
            break;
        case ROW_POST_ADD: { /* x += rms(h) * w */
            float *x = w->x + (size_t)t * VD;
            rms(tmp, w->h + (size_t)t * VD, VD, j->nw);
            for (int z = 0; z < VD; z++)
                x[z] += tmp[z];
            break;
        }
        case ROW_GELU_QUANT: { /* chunk-local row t: gate = act(gate) * up; q8 */
            float *g = w->gate + (size_t)t * VF;
            const float *u = w->up + (size_t)t * VF;
            if (j->tanh_gelu)
                ei_gemma2_fast_gelu_mul(g, u, VF);
            else
                g2m_quick_gelu_mul(g, u, VF);
            ei_gemma2_fast_quant_row(g, w->xq + (size_t)t * (VF / 32), VF);
            break;
        }
        case ROW_POOL: { /* 3x3 mean * sqrt(768), unweighted rms, q8 (reference order) */
            float *o = tmp;
            memset(o, 0, sizeof tmp);
            for (int yy = 0; yy < 3; yy++)
                for (int xx = 0; xx < 3; xx++) {
                    const float *s = w->x + ((size_t)(t / j->out_cols * 3 + yy) * j->cols + t % j->out_cols * 3 + xx) * VD;
                    for (int z = 0; z < VD; z++)
                        o[z] += s[z] / 9.0f * sqrtf(768.0f);
                }
            rms(o, o, VD, NULL);
            ei_gemma2_fast_quant_row(o, w->xq + (size_t)t * (VD / 32), VD);
            break;
        }
        }
    }
}
static void rows(ei_thread_pool *pool, int n, row_job job) { ei_parallel_for(pool, n, 4, row_tokens, &job); }

/* ---- patch embedding ------------------------------------------------------- */

typedef struct {
    const uint8_t *rgb;
    float *patches;
    int width, cols;
} patch_job;
static void patch_tokens(void *opaque, int32_t begin, int32_t end) {
    patch_job *j = opaque;
    for (int t = begin; t < end; t++)
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < 16; y++)
                for (int z = 0; z < 16; z++) {
                    float p = 2.0f * j->rgb[((size_t)(t / j->cols * 16 + y) * j->width + t % j->cols * 16 + z) * 3 + c] / 255.0f - 1;
                    /* The reference convolution's fp16 input rounding. */
                    j->patches[(size_t)t * VD + c * 256 + y * 16 + z] = ei_fp16_to_fp32(ei_fp32_to_fp16(p));
                }
}
typedef struct {
    float *x;
    const float *pos;
    int cols;
} pos_job;
static void pos_tokens(void *opaque, int32_t begin, int32_t end) {
    pos_job *j = opaque;
    for (int t = begin; t < end; t++) {
        const float *px = j->pos + (size_t)(t % j->cols) * VD, *py = j->pos + ((size_t)10240 + t / j->cols) * VD;
        float *x = j->x + (size_t)t * VD;
        for (int z = 0; z < VD; z++)
            x[z] += px[z] + py[z];
    }
}

/* ---- attention ----------------------------------------------------------------- */

typedef struct {
    const vision_ws *w;
    int n, blocks;
} attn_job;
/* Q and the output (into v, free once V moved) are token-major [n][768]; K (in h) and V (in k) are
 * head-major [12][n][64], so the K/V stream of a query block is contiguous:
 * read in place from the token-major buffers, every key of a head sat on its
 * own 4 KB page (3 KB row stride) and the per-head stream at 10080 patches
 * (5.2 MB) defeated the L2 prefetchers. */
static void attn_tasks(void *opaque, int32_t begin, int32_t end) {
    attn_job *j = opaque;
    for (int task = begin; task < end; task++) {
        const int qb = g2m_attn_qblock(64);
        int hd = task / j->blocks, q0 = task % j->blocks * qb;
        int nq = j->n - q0 < qb ? j->n - q0 : qb;
        const size_t plane = (size_t)hd * j->n * 64;
        g2m_attn_args a = {.q = j->w->q, .k = j->w->h + plane, .v = j->w->k + plane, .out = j->w->v,
                           .qstride = VD, .kvstride = 64, .ostride = VD, .qcol = (size_t)hd * 64, .kvcol = 0,
                           .ocol = (size_t)hd * 64, .d = 64, .q0 = q0, .nq = nq, .klo = 0, .khi = j->n,
                           .scale = 1.0f};
        g2m_attn(&a);
    }
}

/* ---- tower --------------------------------------------------------------------- */

static void create_ctx(void) { gemm_ctx = ei_gemma2_fast_create(); }

bool ei_gemma2_vision_fast(const ei_gguf *g, ei_thread_pool *pool, const uint8_t *rgb, int width,
                           int height, float **output, size_t *tokens, char *err, size_t err_len) {
    if (width < 48 || height < 48 || width % 48 || height % 48 ||
        (size_t)width * height > 1120 * 48 * 48) {
        snprintf(err, err_len, "vision input must be aligned to 48 pixels and at most 1120 soft tokens");
        return false;
    }
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, create_ctx);
    const int cols = width / 16, prow = height / 16, n = cols * prow;
    const bool tanh_gelu = ei_gemma2_vision_tanh_gelu(g);

    pthread_mutex_lock(&vision_lock);
    ei_cpu_binding binding; /* hybrid pools: the caller joins on a P-core */
    ei_thread_pool_bind_caller(pool, &binding);
    vision_ws *w = &vws;
    reserve(w, (size_t)n);
    rope_table(w, cols > prow ? cols : prow);

    /* Patch embedding (F32 weights) + learned x/y positions. Patches stage in q. */
    patch_job pj = {rgb, w->q, width, cols};
    ei_tensor patch = *tensor_of(g, "v.patch_embd.weight");
    patch.ne[0] = VD;
    patch.ne[1] = VD;
    pos_job po = {w->x, f32_of(g, "v.position_embd.weight"), cols};
    VP(VP_PATCH, ei_parallel_for(pool, n, 16, patch_tokens, &pj);
       ei_gemma2_fast_gemm_f(pool, &patch, w->q, n, w->x);
       ei_parallel_for(pool, n, 16, pos_tokens, &po));

    attn_job aj = {w, n, (n + g2m_attn_qblock(64) - 1) / g2m_attn_qblock(64)};
    for (int l = 0; l < 16; l++) {
        char name[128];
#define V(suffix) (snprintf(name, sizeof name, "v.blk.%d." suffix ".weight", l), tensor_of(g, name))
#define VF32(suffix) (snprintf(name, sizeof name, "v.blk.%d." suffix ".weight", l), f32_of(g, name))
        if (l)
            ei_gemma2_cpu_yield(); /* between layers: waiting text waves may run */
        VP(VP_ROWS, rows(pool, n, (row_job){.op = ROW_NORM_QUANT, .w = w, .nw = VF32("ln1")}));
        VP(VP_GEMM_QKV, ei_gemma2_fast_gemm_q8(gemm_ctx, pool, V("attn_q"), w->xq, n, w->q);
           ei_gemma2_fast_gemm_q8(gemm_ctx, pool, V("attn_k"), w->xq, n, w->k);
           ei_gemma2_fast_gemm_q8(gemm_ctx, pool, V("attn_v"), w->xq, n, w->v));
        const float *qn = VF32("attn_q_norm"), *kn = VF32("attn_k_norm");
        VP(VP_ROWS, rows(pool, n, (row_job){.op = ROW_QKV_POST, .w = w, .qn = qn, .kn = kn, .cols = cols, .n = n});
           rows(pool, n, (row_job){.op = ROW_V_HEADS, .w = w, .n = n}));
        VP(VP_ATTN, ei_parallel_for(pool, VHEADS * aj.blocks, 1, attn_tasks, &aj));
        ei_gemma2_cpu_yield();
        VP(VP_ROWS, rows(pool, n, (row_job){.op = ROW_QUANT, .w = w, .src = w->v}));
        VP(VP_GEMM_O, ei_gemma2_fast_gemm_q8(gemm_ctx, pool, V("attn_out"), w->xq, n, w->h));
        VP(VP_ROWS, rows(pool, n, (row_job){.op = ROW_POST_ADD, .w = w, .nw = VF32("attn_post_norm")}));
        /* ln2 activations for every token first (q buffer is free now), then the
         * FFN in chunks: up/gate/act/down per FFN_CHUNK tokens. */
        VP(VP_ROWS, rows(pool, n, (row_job){.op = ROW_NORM_QUANT, .w = w, .nw = VF32("ln2")}));
        memcpy(w->q, w->xq, (size_t)n * (VD / 32) * sizeof(ei_block_q8_0));
        const ei_block_q8_0 *ln2q = (const ei_block_q8_0 *)w->q;
        const ei_tensor *up = V("ffn_up"), *gate = V("ffn_gate"), *down = V("ffn_down");
        for (int c0 = 0; c0 < n; c0 += FFN_CHUNK) {
            int cn = n - c0 < FFN_CHUNK ? n - c0 : FFN_CHUNK;
            if (c0)
                ei_gemma2_cpu_yield();
            const ei_block_q8_0 *xin = ln2q + (size_t)c0 * (VD / 32);
            VP(VP_GEMM_FFN, ei_gemma2_fast_gemm_q8(gemm_ctx, pool, up, xin, cn, w->up);
               ei_gemma2_fast_gemm_q8(gemm_ctx, pool, gate, xin, cn, w->gate));
            VP(VP_ROWS, rows(pool, cn, (row_job){.op = ROW_GELU_QUANT, .w = w, .tanh_gelu = tanh_gelu}));
            VP(VP_GEMM_FFN, ei_gemma2_fast_gemm_q8(gemm_ctx, pool, down, w->xq, cn, w->h + (size_t)c0 * VD));
        }
        VP(VP_ROWS, rows(pool, n, (row_job){.op = ROW_POST_ADD, .w = w, .nw = VF32("ffn_post_norm")}));
#undef V
#undef VF32
    }
    const int out_cols = cols / 3, nt = out_cols * (prow / 3);
    *output = ei_xmalloc((size_t)nt * 512 * sizeof(float));
    *tokens = (size_t)nt;
    VP(VP_POOL, rows(pool, nt, (row_job){.op = ROW_POOL, .w = w, .cols = cols, .out_cols = out_cols});
       ei_gemma2_fast_gemm_q8(gemm_ctx, pool, tensor_of(g, "mm.input_projection.weight"), w->xq, nt, *output));
    if (vp_on()) {
        double total = 0;
        for (int i = 0; i < VP_N; i++)
            total += vp_ms[i];
        fprintf(stderr, "vision-fast patches=%d total=%.1f ms:", n, total);
        for (int i = 0; i < VP_N; i++)
            fprintf(stderr, " %s=%.1f", vp_names[i], vp_ms[i]);
        fprintf(stderr, "\n");
        memset(vp_ms, 0, sizeof vp_ms);
    }
    ei_thread_pool_unbind_caller(&binding);
    pthread_mutex_unlock(&vision_lock);
    return true;
}
