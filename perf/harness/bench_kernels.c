#define _POSIX_C_SOURCE 200809L

#include "kernels.h"
#include "quants.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define PERF_SCHEMA 1
#define DEFAULT_MIN_SAMPLE_NS 2000000.0

static volatile float g_sink = 0.0f;

typedef void (*bench_fn)(void *);

typedef struct {
    double ms;
    double p20_ms;
    double p80_ms;
    double cv;
    int batch;
} timing_result;

typedef struct {
    uint32_t s;
} rng_state;

static uint32_t rng_next(rng_state *rng) {
    uint32_t x = rng->s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->s = x ? x : 0x9e3779b9u;
    return rng->s;
}

static float rng_f32(rng_state *rng) {
    uint32_t x = rng_next(rng);
    return ((float)(x & 0xFFFFu) / 32768.0f) - 1.0f;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static uint64_t now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        ei_die("clock_gettime failed: %s", strerror(errno));
    }
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void *perf_xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) ei_die("out of memory");
    return p;
}

static void fill_f32(float *x, int32_t n, uint32_t seed) {
    rng_state rng = { seed };
    for (int32_t i = 0; i < n; i++) x[i] = 3.0f * rng_f32(&rng);
}

static void fill_q4_blocks(ei_block_q4_0 *x, int32_t n_blocks, uint32_t seed) {
    rng_state rng = { seed };
    for (int32_t b = 0; b < n_blocks; b++) {
        float d = 0.005f + 0.05f * (float)(rng_next(&rng) & 0xFFu) / 255.0f;
        x[b].d = ei_fp32_to_fp16(d);
        for (int j = 0; j < 16; j++) {
            uint8_t lo = (uint8_t)(rng_next(&rng) & 0x0Fu);
            uint8_t hi = (uint8_t)(rng_next(&rng) & 0x0Fu);
            x[b].qs[j] = (uint8_t)(lo | (hi << 4));
        }
    }
}

static void fill_q8_blocks(ei_block_q8_0 *x, int32_t n_blocks, uint32_t seed) {
    float tmp[EI_QK];
    rng_state rng = { seed };
    for (int32_t b = 0; b < n_blocks; b++) {
        for (int j = 0; j < EI_QK; j++) tmp[j] = 2.0f * rng_f32(&rng);
        ei_quantize_row_q8_0(tmp, x + b, EI_QK);
    }
}

static float scalar_dot_q4_q8(const ei_block_q4_0 *x, const ei_block_q8_0 *y, int32_t k) {
    int32_t nb = k / EI_QK;
    float sumf = 0.0f;
    for (int32_t b = 0; b < nb; b++) {
        int32_t sumi = 0;
        for (int j = 0; j < 16; j++) {
            int32_t v0 = (int32_t)(x[b].qs[j] & 0x0Fu) - 8;
            int32_t v1 = (int32_t)(x[b].qs[j] >> 4) - 8;
            sumi += v0 * (int32_t)y[b].qs[j];
            sumi += v1 * (int32_t)y[b].qs[j + 16];
        }
        sumf += (float)sumi * ei_fp16_to_fp32(x[b].d) * ei_fp16_to_fp32(y[b].d);
    }
    return sumf;
}

static void scalar_matmul_q4_q8(const ei_block_q4_0 *w, const ei_block_q8_0 *xq,
                                int32_t n, int32_t k, float *out) {
    int32_t blocks = k / EI_QK;
    for (int32_t row = 0; row < n; row++) {
        out[row] = scalar_dot_q4_q8(w + (size_t)row * (size_t)blocks, xq, k);
    }
}

static timing_result time_bench(bench_fn fn, void *ctx, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) fn(ctx);

    uint64_t t0 = now_ns();
    fn(ctx);
    uint64_t t1 = now_ns();
    double one_ns = (double)(t1 - t0);
    if (one_ns < 1.0) one_ns = 1.0;
    int batch = (int)ceil(DEFAULT_MIN_SAMPLE_NS / one_ns);
    if (batch < 1) batch = 1;
    if (batch > 8192) batch = 8192;

    double *samples = perf_xmalloc(sizeof(double) * (size_t)iters);
    for (int i = 0; i < iters; i++) {
        t0 = now_ns();
        for (int j = 0; j < batch; j++) fn(ctx);
        t1 = now_ns();
        samples[i] = ((double)(t1 - t0) / 1000000.0) / (double)batch;
    }

    qsort(samples, (size_t)iters, sizeof(double), cmp_double);
    double sum = 0.0;
    for (int i = 0; i < iters; i++) sum += samples[i];
    double mean = sum / (double)iters;
    double var = 0.0;
    for (int i = 0; i < iters; i++) {
        double d = samples[i] - mean;
        var += d * d;
    }
    var /= (double)iters;

    timing_result r = {
        .ms = (iters & 1) ? samples[iters / 2]
                          : 0.5 * (samples[iters / 2 - 1] + samples[iters / 2]),
        .p20_ms = samples[(int)floor(0.20 * (double)(iters - 1))],
        .p80_ms = samples[(int)floor(0.80 * (double)(iters - 1))],
        .cv = mean > 0.0 ? sqrt(var) / mean : 0.0,
        .batch = batch,
    };
    free(samples);
    return r;
}

static void emit_row(const char *kernel, const char *variant, const char *shape_json,
                     const char *dtype, const char *fmt, timing_result t,
                     double bytes_moved, double weight_bytes, double flops,
                     double max_abs_err, double max_rel_err) {
    double sec = t.ms / 1000.0;
    double gbps = bytes_moved > 0.0 && sec > 0.0 ? bytes_moved / sec / 1e9 : 0.0;
    double wgbps = weight_bytes > 0.0 && sec > 0.0 ? weight_bytes / sec / 1e9 : 0.0;
    double gflops = flops > 0.0 && sec > 0.0 ? flops / sec / 1e9 : 0.0;

    printf("{\"schema\":%d,\"backend\":\"cpu\",\"kernel\":\"%s\",\"variant\":\"%s\","
           "\"shape\":{%s},\"dtype\":\"%s\",\"format\":",
           PERF_SCHEMA, kernel, variant, shape_json, dtype);
    if (fmt) printf("\"%s\"", fmt);
    else printf("null");
    printf(",\"target_ms\":%.9g,\"target_p20_ms\":%.9g,\"target_p80_ms\":%.9g,"
           "\"target_cv\":%.6g,\"batch\":%d,\"bytes_moved\":%.0f,"
           "\"weight_bytes\":%.0f,\"flops\":%.0f,\"gbps\":%.9g,"
           "\"weight_gbps\":%.9g,\"gflops\":%.9g,\"max_abs_err\":%.9g,"
           "\"max_rel_err\":%.9g,\"cpu_kernel\":\"%s\",\"quant_kernel\":\"%s\"}\n",
           t.ms, t.p20_ms, t.p80_ms, t.cv, t.batch, bytes_moved, weight_bytes,
           flops, gbps, wgbps, gflops, max_abs_err, max_rel_err,
           ei_cpu_kernel_variant(), ei_quants_kernel_variant());
}

typedef struct {
    ei_block_q4_0 *w;
    ei_block_q8_0 *xq;
    int32_t k;
    float out;
} dot_state;

static void bench_dot(void *ctx) {
    dot_state *s = ctx;
    s->out = ei_vec_dot_q4_0_q8_0(s->w, s->xq, s->k);
    g_sink = g_sink + s->out * 1e-30f;
}

static void run_q4dot(int32_t k, int warmup, int iters) {
    int32_t blocks = k / EI_QK;
    dot_state s = {
        .w = perf_xmalloc(sizeof(ei_block_q4_0) * (size_t)blocks),
        .xq = perf_xmalloc(sizeof(ei_block_q8_0) * (size_t)blocks),
        .k = k,
        .out = 0.0f,
    };
    fill_q4_blocks(s.w, blocks, 0x1000u + (uint32_t)k);
    fill_q8_blocks(s.xq, blocks, 0x2000u + (uint32_t)k);
    float ref = scalar_dot_q4_q8(s.w, s.xq, k);
    bench_dot(&s);
    double abs_err = fabs((double)s.out - (double)ref);
    double rel_err = abs_err / (fabs((double)ref) + 1e-12);

    timing_result t = time_bench(bench_dot, &s, warmup, iters);
    char shape[64];
    snprintf(shape, sizeof shape, "\"k\":%d", k);
    emit_row("q4dot", "q4_0_q8_0", shape, "f32", "q4_0",
             t, (double)blocks * (double)(sizeof(ei_block_q4_0) + sizeof(ei_block_q8_0)),
             (double)blocks * (double)sizeof(ei_block_q4_0),
             2.0 * (double)k, abs_err, rel_err);
    free(s.xq);
    free(s.w);
}

typedef struct {
    ei_tensor tensor;
    ei_block_q4_0 *w;
    ei_block_q8_0 *xq;
    float *out;
} gemv_state;

static void bench_gemv(void *ctx) {
    gemv_state *s = ctx;
    ei_matmul_q4_0_q8_0(&s->tensor, s->xq, s->out);
    g_sink = g_sink + s->out[(size_t)s->tensor.ne[1] / 2u] * 1e-30f;
}

static void run_q4gemv(int32_t n, int32_t k, int warmup, int iters) {
    int32_t blocks = k / EI_QK;
    gemv_state s;
    memset(&s, 0, sizeof s);
    s.w = perf_xmalloc(sizeof(ei_block_q4_0) * (size_t)n * (size_t)blocks);
    s.xq = perf_xmalloc(sizeof(ei_block_q8_0) * (size_t)blocks);
    s.out = perf_xmalloc(sizeof(float) * (size_t)n);
    fill_q4_blocks(s.w, n * blocks, 0x3000u + (uint32_t)n + (uint32_t)k);
    fill_q8_blocks(s.xq, blocks, 0x4000u + (uint32_t)k);
    s.tensor.type = EI_T_Q4_0;
    s.tensor.ne[0] = (uint64_t)k;
    s.tensor.ne[1] = (uint64_t)n;
    s.tensor.n_dims = 2;
    s.tensor.data = s.w;

    float *ref = perf_xmalloc(sizeof(float) * (size_t)n);
    scalar_matmul_q4_q8(s.w, s.xq, n, k, ref);
    bench_gemv(&s);
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (int32_t i = 0; i < n; i++) {
        double err = fabs((double)s.out[i] - (double)ref[i]);
        double rel = err / (fabs((double)ref[i]) + 1e-12);
        if (err > max_abs) max_abs = err;
        if (rel > max_rel) max_rel = rel;
    }

    timing_result t = time_bench(bench_gemv, &s, warmup, iters);
    char shape[96];
    snprintf(shape, sizeof shape, "\"n\":%d,\"k\":%d", n, k);
    emit_row("q4gemv", "q4_0_q8_0", shape, "f32", "q4_0",
             t, (double)n * (double)blocks * (double)sizeof(ei_block_q4_0) +
                    (double)blocks * (double)sizeof(ei_block_q8_0) +
                    (double)n * (double)sizeof(float),
             (double)n * (double)blocks * (double)sizeof(ei_block_q4_0),
             2.0 * (double)n * (double)k, max_abs, max_rel);
    free(ref);
    free(s.out);
    free(s.xq);
    free(s.w);
}

typedef struct {
    ei_tensor tensor;
    ei_block_q8_0 *rows;
    float *out;
    int32_t row;
    int32_t n_rows;
} dequant_state;

static void bench_dequant(void *ctx) {
    dequant_state *s = ctx;
    ei_dequantize_row_q8_0(&s->tensor, s->row, s->out);
    s->row++;
    if (s->row == s->n_rows) s->row = 0;
    g_sink = g_sink + s->out[EI_N_EMBD / 2] * 1e-30f;
}

static void run_q8dequant(int32_t rows, int32_t k, int warmup, int iters) {
    int32_t blocks = k / EI_QK;
    dequant_state s;
    memset(&s, 0, sizeof s);
    s.rows = perf_xmalloc(sizeof(ei_block_q8_0) * (size_t)rows * (size_t)blocks);
    s.out = perf_xmalloc(sizeof(float) * (size_t)k);
    fill_q8_blocks(s.rows, rows * blocks, 0x5000u + (uint32_t)rows);
    s.n_rows = rows;
    s.tensor.type = EI_T_Q8_0;
    s.tensor.ne[0] = (uint64_t)k;
    s.tensor.ne[1] = (uint64_t)rows;
    s.tensor.n_dims = 2;
    s.tensor.data = s.rows;

    timing_result t = time_bench(bench_dequant, &s, warmup, iters);
    char shape[96];
    snprintf(shape, sizeof shape, "\"rows\":%d,\"k\":%d", rows, k);
    emit_row("q8dequant", "q8_0_row", shape, "f32", "q8_0",
             t, (double)blocks * (double)sizeof(ei_block_q8_0) +
                    (double)k * (double)sizeof(float),
             (double)blocks * (double)sizeof(ei_block_q8_0),
             0.0, 0.0, 0.0);
    free(s.out);
    free(s.rows);
}

typedef struct {
    float *x;
    ei_block_q8_0 *out;
    int32_t n;
} quant_state;

static void bench_quant(void *ctx) {
    quant_state *s = ctx;
    ei_quantize_row_q8_0(s->x, s->out, s->n);
    g_sink = g_sink + (float)s->out[s->n / EI_QK / 2].qs[0] * 1e-30f;
}

static void run_q8quant(int32_t n, int warmup, int iters) {
    quant_state s = {
        .x = perf_xmalloc(sizeof(float) * (size_t)n),
        .out = perf_xmalloc(sizeof(ei_block_q8_0) * (size_t)(n / EI_QK)),
        .n = n,
    };
    fill_f32(s.x, n, 0x5800u + (uint32_t)n);
    timing_result t = time_bench(bench_quant, &s, warmup, iters);
    char shape[64];
    snprintf(shape, sizeof shape, "\"n\":%d", n);
    emit_row("q8quant", "q8_0_row", shape, "f32", "q8_0", t,
             (double)n * (double)sizeof(float) +
                 (double)(n / EI_QK) * (double)sizeof(ei_block_q8_0),
             0.0, 0.0, 0.0, 0.0);
    free(s.out);
    free(s.x);
}

typedef struct {
    float *x;
    float *w;
    float *out;
    int32_t n;
} rms_state;

static void bench_rms(void *ctx) {
    rms_state *s = ctx;
    ei_rms_norm(s->x, s->w, s->n, 1e-6f, s->out);
    g_sink = g_sink + s->out[s->n / 2] * 1e-30f;
}

static void run_rms(int32_t n, int warmup, int iters) {
    rms_state s = {
        .x = perf_xmalloc(sizeof(float) * (size_t)n),
        .w = perf_xmalloc(sizeof(float) * (size_t)n),
        .out = perf_xmalloc(sizeof(float) * (size_t)n),
        .n = n,
    };
    fill_f32(s.x, n, 0x6000u + (uint32_t)n);
    fill_f32(s.w, n, 0x7000u + (uint32_t)n);
    timing_result t = time_bench(bench_rms, &s, warmup, iters);
    char shape[64];
    snprintf(shape, sizeof shape, "\"n\":%d", n);
    emit_row("rms_norm", "f32", shape, "f32", NULL, t,
             3.0 * (double)n * (double)sizeof(float), 0.0,
             3.0 * (double)n, 0.0, 0.0);
    free(s.out);
    free(s.w);
    free(s.x);
}

typedef struct {
    float *embedding;
    int32_t dimensions;
} embedding_prefix_state;

static void bench_embedding_prefix(void *ctx) {
    embedding_prefix_state *s = ctx;
    if (!ei_embedding_normalize_prefix(s->embedding, s->dimensions)) {
        ei_die("unsupported embedding prefix dimension %d", s->dimensions);
    }
    g_sink = g_sink + s->embedding[s->dimensions / 2] * 1e-30f;
}

static void run_embedding_prefix(int32_t dimensions, int warmup, int iters) {
    embedding_prefix_state s = {
        .embedding = perf_xmalloc(sizeof(float) * EI_N_EMBD),
        .dimensions = dimensions,
    };
    fill_f32(s.embedding, EI_N_EMBD, 0x7800u + (uint32_t)dimensions);
    ei_l2_normalize(s.embedding, EI_N_EMBD);
    timing_result t = time_bench(bench_embedding_prefix, &s, warmup, iters);
    char shape[64];
    snprintf(shape, sizeof shape, "\"dimensions\":%d", dimensions);
    const double bytes = dimensions < EI_N_EMBD
        ? 3.0 * (double)dimensions * (double)sizeof(float)
        : 0.0;
    emit_row("embedding_prefix", "l2_normalize", shape, "f32", NULL, t,
             bytes, 0.0, dimensions < EI_N_EMBD ? 3.0 * dimensions : 0.0,
             0.0, 0.0);
    free(s.embedding);
}

typedef struct {
    float *x;
    float *weight;
    float *out;
    int32_t tokens;
} mean_pool_state;

static void bench_mean_pool(void *ctx) {
    mean_pool_state *s = ctx;
    ei_mean_pool_rms_l2(s->x, s->tokens, s->weight, 0.0f, s->out);
    g_sink = g_sink + s->out[EI_N_EMBD / 2] * 1e-30f;
}

static void run_mean_pool(int32_t tokens, int warmup, int iters) {
    mean_pool_state s = {
        .x = perf_xmalloc(sizeof(float) * (size_t)tokens * EI_N_EMBD),
        .weight = perf_xmalloc(sizeof(float) * EI_N_EMBD),
        .out = perf_xmalloc(sizeof(float) * EI_N_EMBD),
        .tokens = tokens,
    };
    fill_f32(s.x, tokens * EI_N_EMBD, 0x7c00u + (uint32_t)tokens);
    for (int32_t i = 0; i < EI_N_EMBD; i++) s.weight[i] = 1.0f;
    bench_mean_pool(&s);

    timing_result t = time_bench(bench_mean_pool, &s, warmup, iters);
    char shape[80];
    snprintf(shape, sizeof shape, "\"tokens\":%d,\"hidden\":%d",
             tokens, EI_N_EMBD);
    const double bytes = (28.0 * tokens + 20.0) * EI_N_EMBD * sizeof(float);
    emit_row("mean_pool", "rms_mean_l2", shape, "f32", NULL, t,
             bytes, 0.0, 0.0, 0.0, 0.0);
    free(s.out);
    free(s.weight);
    free(s.x);
}

typedef struct {
    float *gate;
    float *up;
    int32_t n;
} gelu_state;

static void bench_gelu(void *ctx) {
    gelu_state *s = ctx;
    ei_gelu_mul_inplace(s->gate, s->up, s->n);
    g_sink = g_sink + s->gate[s->n / 2] * 1e-30f;
}

static void run_gelu(int32_t n, int warmup, int iters) {
    gelu_state s = {
        .gate = perf_xmalloc(sizeof(float) * (size_t)n),
        .up = perf_xmalloc(sizeof(float) * (size_t)n),
        .n = n,
    };
    fill_f32(s.gate, n, 0x8000u + (uint32_t)n);
    fill_f32(s.up, n, 0x9000u + (uint32_t)n);
    timing_result t = time_bench(bench_gelu, &s, warmup, iters);
    char shape[64];
    snprintf(shape, sizeof shape, "\"n\":%d", n);
    emit_row("gelu_mul", "tanh", shape, "f32", NULL, t,
             3.0 * (double)n * (double)sizeof(float), 0.0,
             0.0, 0.0, 0.0);
    free(s.up);
    free(s.gate);
}

typedef struct {
    float *x;
    int32_t heads;
} rope_state;

static void bench_rope(void *ctx) {
    rope_state *s = ctx;
    ei_rope_neox_inplace(s->x, s->heads, 127, 10000.0f);
    g_sink = g_sink + s->x[(s->heads * EI_HEAD_DIM) / 2] * 1e-30f;
}

static void run_rope(int32_t heads, int warmup, int iters) {
    int32_t n = heads * EI_HEAD_DIM;
    rope_state s = {
        .x = perf_xmalloc(sizeof(float) * (size_t)n),
        .heads = heads,
    };
    fill_f32(s.x, n, 0xa000u + (uint32_t)heads);
    timing_result t = time_bench(bench_rope, &s, warmup, iters);
    char shape[80];
    snprintf(shape, sizeof shape, "\"heads\":%d,\"head_dim\":%d", heads, EI_HEAD_DIM);
    emit_row("rope", "neox", shape, "f32", NULL, t,
             2.0 * (double)n * (double)sizeof(float), 0.0,
             0.0, 0.0, 0.0);
    free(s.x);
}

typedef struct {
    float *dst;
    float *src;
    int32_t n;
} add_state;

static void bench_add(void *ctx) {
    add_state *s = ctx;
    ei_vec_add_inplace(s->dst, s->src, s->n);
    g_sink = g_sink + s->dst[s->n / 2] * 1e-30f;
}

static void run_add(int32_t n, int warmup, int iters) {
    add_state s = {
        .dst = perf_xmalloc(sizeof(float) * (size_t)n),
        .src = perf_xmalloc(sizeof(float) * (size_t)n),
        .n = n,
    };
    fill_f32(s.dst, n, 0xb000u + (uint32_t)n);
    fill_f32(s.src, n, 0xc000u + (uint32_t)n);
    timing_result t = time_bench(bench_add, &s, warmup, iters);
    char shape[64];
    snprintf(shape, sizeof shape, "\"n\":%d", n);
    emit_row("vec_add", "inplace", shape, "f32", NULL, t,
             3.0 * (double)n * (double)sizeof(float), 0.0,
             (double)n, 0.0, 0.0);
    free(s.src);
    free(s.dst);
}

typedef struct {
    float *q;
    float *k;
    float *v;
    float *out;
    float *scores;
    int32_t tokens;
    int32_t window;
} attention_state;

static void bench_attention(void *ctx) {
    attention_state *s = ctx;
    ei_attention_mha(s->q, s->k, s->v, s->tokens, s->window, s->out, s->scores);
    g_sink = g_sink + s->out[(size_t)s->tokens * EI_N_EMBD / 2] * 1e-30f;
}

static void run_attention(int32_t tokens, int32_t window, int warmup, int iters) {
    attention_state s = {
        .q = perf_xmalloc(sizeof(float) * (size_t)tokens * EI_N_EMBD),
        .k = perf_xmalloc(sizeof(float) * (size_t)tokens * EI_HEAD_DIM),
        .v = perf_xmalloc(sizeof(float) * (size_t)tokens * EI_HEAD_DIM),
        .out = perf_xmalloc(sizeof(float) * (size_t)tokens * EI_N_EMBD),
        .scores = perf_xmalloc(sizeof(float) * (size_t)tokens),
        .tokens = tokens,
        .window = window,
    };
    fill_f32(s.q, tokens * EI_N_EMBD, 0xd000u + (uint32_t)tokens);
    fill_f32(s.k, tokens * EI_HEAD_DIM, 0xe000u + (uint32_t)tokens);
    fill_f32(s.v, tokens * EI_HEAD_DIM, 0xf000u + (uint32_t)tokens);
    timing_result t = time_bench(bench_attention, &s, warmup, iters);
    char shape[96];
    snprintf(shape, sizeof shape, "\"tokens\":%d,\"heads\":%d,\"head_dim\":%d,\"window\":%d",
             tokens, EI_N_HEAD, EI_HEAD_DIM, window);
    int64_t keys_per_query = window == 0 || window >= tokens ? tokens : window + 1;
    double pairs = (double)tokens * (double)keys_per_query * (double)EI_N_HEAD;
    emit_row("attention", window == 0 ? "full" : "swa", shape, "f32", NULL, t,
             0.0, 0.0, 4.0 * pairs * (double)EI_HEAD_DIM, 0.0, 0.0);
    free(s.scores);
    free(s.out);
    free(s.v);
    free(s.k);
    free(s.q);
}

static bool kernel_enabled(const char *list, const char *name) {
    if (strcmp(list, "all") == 0) return true;
    const char *p = list;
    size_t name_len = strlen(name);
    while (*p) {
        const char *comma = strchr(p, ',');
        size_t len = comma ? (size_t)(comma - p) : strlen(p);
        if (len == name_len && memcmp(p, name, len) == 0) return true;
        if (!comma) break;
        p = comma + 1;
    }
    return false;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--preset smoke|quick|comprehensive] [--kernel list] "
            "[--warmup n] [--iters n]\n",
            argv0);
}

int main(int argc, char **argv) {
    const char *preset = "smoke";
    const char *kernels = "all";
    int warmup = 5;
    int iters = 20;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--preset") == 0 && i + 1 < argc) {
            preset = argv[++i];
        } else if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            kernels = argv[++i];
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (warmup < 0 || iters <= 0) {
        usage(argv[0]);
        return 2;
    }

    bool smoke = strcmp(preset, "smoke") == 0;
    bool quick = strcmp(preset, "quick") == 0;
    bool comprehensive = strcmp(preset, "comprehensive") == 0;
    if (!smoke && !quick && !comprehensive) {
        usage(argv[0]);
        return 2;
    }

    if (kernel_enabled(kernels, "q4dot")) {
        run_q4dot(EI_N_EMBD, warmup, iters);
        if (!smoke) run_q4dot(EI_N_FF, warmup, iters);
    }
    if (kernel_enabled(kernels, "q4gemv")) {
        run_q4gemv(EI_HEAD_DIM, EI_N_EMBD, warmup, iters);
        if (!smoke) {
            run_q4gemv(EI_N_EMBD, EI_N_EMBD, warmup, iters);
            run_q4gemv(EI_N_FF, EI_N_EMBD, warmup, iters);
            run_q4gemv(EI_N_EMBD, EI_N_FF, warmup, iters);
        }
        if (comprehensive) {
            run_q4gemv(EI_N_HEAD * EI_HEAD_DIM, EI_N_EMBD, warmup, iters);
            run_q4gemv(EI_N_EMBD, EI_N_HEAD * EI_HEAD_DIM, warmup, iters);
        }
    }
    if (kernel_enabled(kernels, "q8dequant")) {
        run_q8dequant(smoke ? 16 : 256, EI_N_EMBD, warmup, iters);
        if (comprehensive) run_q8dequant(4096, EI_N_EMBD, warmup, iters);
    }
    if (kernel_enabled(kernels, "q8quant")) {
        run_q8quant(EI_N_EMBD, warmup, iters);
        if (!smoke) run_q8quant(EI_N_FF, warmup, iters);
    }
    if (kernel_enabled(kernels, "rms_norm")) {
        run_rms(EI_HEAD_DIM, warmup, iters);
        if (!smoke) run_rms(EI_N_EMBD, warmup, iters);
    }
    if (kernel_enabled(kernels, "embedding_prefix")) {
        run_embedding_prefix(128, warmup, iters);
        run_embedding_prefix(256, warmup, iters);
        run_embedding_prefix(512, warmup, iters);
        run_embedding_prefix(EI_N_EMBD, warmup, iters);
    }
    if (kernel_enabled(kernels, "mean_pool")) {
        run_mean_pool(1, warmup, iters);
        if (!smoke) run_mean_pool(32, warmup, iters);
        if (comprehensive) run_mean_pool(512, warmup, iters);
    }
    if (kernel_enabled(kernels, "gelu_mul")) {
        run_gelu(EI_N_FF, warmup, iters);
    }
    if (kernel_enabled(kernels, "rope")) {
        run_rope(EI_N_HEAD_KV, warmup, iters);
        if (!smoke) run_rope(EI_N_HEAD, warmup, iters);
    }
    if (kernel_enabled(kernels, "vec_add")) {
        run_add(EI_N_EMBD, warmup, iters);
    }
    if (kernel_enabled(kernels, "attention")) {
        run_attention(smoke ? 16 : (comprehensive ? 512 : 128), 0, warmup, iters);
        if (!smoke) run_attention(comprehensive ? 512 : 128, 512, warmup, iters);
    }

    if (g_sink == 12345.0f) {
        fprintf(stderr, "sink: %f\n", (float)g_sink);
    }
    return 0;
}
