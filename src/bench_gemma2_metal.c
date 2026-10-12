#define _POSIX_C_SOURCE 200809L
// Compare and time an EmbeddingGemma 2 Metal route against the default path.
// usage: [EI_BENCH_BASE=FLAGS] bench_gemma2_metal text.gguf "FLAG=1,FLAG=1" [TxB ...]
// Each case packs B sequences of T tokens (T+TxB+... packs a mixed wave in that
// order); flags name EI_GEMMA2_METAL_<FLAG>.
#include "engine.h"
#include "gemma2_metal.h"
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *route_flags[] = {"TENSOR", "FLASH",  "GEMV",   "FUSED",  "TENSOR2",
                                    "FLASH2", "NOEPI", "FLASH64", "VFUSED", "AFUSED",
                                    "LAYER",  "FLASH2_TILE", "LAYER_GROUPS"};

static void set_route(const char *spec) {
    char name[64];
    for (size_t i = 0; i < sizeof route_flags / sizeof *route_flags; i++) {
        snprintf(name, sizeof name, "EI_GEMMA2_METAL_%s", route_flags[i]);
        // Routes are explicit: everything not named is off ("0"), since the
        // batched stack defaults on.
        if (strcmp(route_flags[i], "FLASH2_TILE") && strcmp(route_flags[i], "LAYER_GROUPS"))
            setenv(name, "0", 1);
        else
            unsetenv(name);
    }
    char *copy = strdup(spec), *save = NULL;
    for (char *item = strtok_r(copy, ",", &save); item; item = strtok_r(NULL, ",", &save)) {
        char *eq = strchr(item, '=');
        if (!eq)
            ei_die("bad flag '%s'", item);
        *eq = '\0';
        snprintf(name, sizeof name, "EI_GEMMA2_METAL_%s", item);
        setenv(name, eq + 1, 1);
    }
    free(copy);
}
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}
// A case is T (one sequence), TxB (B sequences of T) or several joined by '+'
// (a mixed wave in the given order, e.g. 2000+20x15). Returns the sequence count.
static size_t parse_case(const char *spec, size_t *lens, size_t cap, size_t *total) {
    size_t count = 0;
    *total = 0;
    char *copy = strdup(spec), *save = NULL;
    for (char *item = strtok_r(copy, "+", &save); item; item = strtok_r(NULL, "+", &save)) {
        size_t t = 0, b = 1;
        if (sscanf(item, "%zux%zu", &t, &b) < 1 || !t || !b)
            ei_die("bad case '%s' (T, TxB or T+TxB..., at most 8192 tokens)", spec);
        for (size_t i = 0; i < b; i++) {
            if (count == cap)
                ei_die("case '%s' has too many sequences", spec);
            lens[count++] = t;
            *total += t;
        }
    }
    free(copy);
    if (!count || *total > 8192)
        ei_die("bad case '%s' (T, TxB or T+TxB..., at most 8192 tokens)", spec);
    return count;
}
static void fill_case(const size_t *lens, size_t count, int32_t *ids, size_t *offsets) {
    offsets[0] = 0;
    for (size_t s = 0; s < count; s++)
        offsets[s + 1] = offsets[s] + lens[s];
    for (size_t s = 0; s < count; s++)
        for (size_t i = offsets[s]; i < offsets[s + 1]; i++)
            ids[i] = (int32_t)(100 + (i * 137 + s * 7919) % 20000);
}
static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double run(void *ctx, const int32_t *ids, const size_t *offsets, size_t batch, float *out,
                  int iterations) {
    char err[512];
    double times[16];
    for (int i = -2; i < iterations; i++) {
        double start = now_ms();
        if (!ei_gemma2_metal_batch(ctx, ids, offsets, batch, out, err, sizeof err))
            ei_die("%s", err);
        if (i >= 0)
            times[i] = now_ms() - start;
    }
    qsort(times, (size_t)iterations, sizeof *times, cmp);
    return times[iterations / 2];
}
typedef struct {
    void *ctx;
    const int32_t *ids;
    const size_t *offsets;
    size_t batch;
    int iterations;
    double ms;
} stream_job;
static void *stream_main(void *opaque) {
    stream_job *j = opaque;
    float *out = ei_xmalloc(j->batch * 768 * sizeof *out);
    char err[512];
    double start = now_ms();
    for (int i = 0; i < j->iterations; i++)
        if (!ei_gemma2_metal_batch(j->ctx, j->ids, j->offsets, j->batch, out, err, sizeof err))
            ei_die("%s", err);
    j->ms = now_ms() - start;
    free(out);
    return NULL;
}
// EI_BENCH_STREAMS=N: N threads, each with its own route context (queue and
// workspace), run the case concurrently; reports aggregate tokens/s vs one.
static void streams(const char *model_route, ei_engine *e, size_t n_streams, const char *const *cases,
                    size_t n_cases) {
    char err[512];
    void **ctx = ei_xmalloc(n_streams * sizeof *ctx);
    for (size_t i = 0; i < n_streams; i++)
        if (!(ctx[i] = ei_gemma2_metal_create(&e->model.gguf, err, sizeof err)))
            ei_die("%s", err);
    printf("route %s, %zu streams\n%-10s %14s %14s %8s\n", model_route, n_streams, "case",
           "1-stream tok/s", "N-stream tok/s", "scaling");
    for (size_t c = 0; c < n_cases; c++) {
        size_t lens[8192], total = 0;
        size_t b = parse_case(cases[c], lens, 8192, &total);
        int32_t *ids = ei_xmalloc(total * sizeof *ids);
        size_t *offsets = ei_xmalloc((b + 1) * sizeof *offsets);
        fill_case(lens, b, ids, offsets);
        size_t t = total / b;
        int iterations = (int)(2000 / (t * b) + 20);
        stream_job one = {ctx[0], ids, offsets, b, iterations, 0};
        stream_main(&one); // warm
        stream_main(&one);
        stream_job *jobs = ei_xmalloc(n_streams * sizeof *jobs);
        pthread_t *threads = ei_xmalloc(n_streams * sizeof *threads);
        for (size_t i = 0; i < n_streams; i++) {
            jobs[i] = (stream_job){ctx[i], ids, offsets, b, iterations, 0};
            pthread_create(&threads[i], NULL, stream_main, &jobs[i]);
        }
        double worst = 0;
        for (size_t i = 0; i < n_streams; i++) {
            pthread_join(threads[i], NULL);
            if (jobs[i].ms > worst)
                worst = jobs[i].ms;
        }
        double single = (double)(t * b) * iterations / one.ms * 1000.0;
        double multi = (double)(t * b) * iterations * n_streams / worst * 1000.0;
        printf("%-10s %14.0f %14.0f %7.2fx\n", cases[c], single, multi, multi / single);
        fflush(stdout);
        free(jobs);
        free(threads);
        free(ids);
        free(offsets);
    }
    for (size_t i = 0; i < n_streams; i++)
        ei_gemma2_metal_free(ctx[i]);
    free(ctx);
}
int main(int argc, char **argv) {
    if (argc < 3)
        ei_die("usage: %s text.gguf FLAGS [TxB ...]", argv[0]);
    const char *defaults[] = {"1x1", "8x1", "32x1", "1x32", "32x4", "32x16", "128x1",
                              "128x8", "512x1", "512x16", "2048x1"};
    const char **cases = argc > 3 ? (const char **)argv + 3 : defaults;
    size_t n_cases = argc > 3 ? (size_t)argc - 3 : sizeof defaults / sizeof *defaults;
    // EI_BENCH_BASE names the baseline route (default: every route off).
    const char *base_route = getenv("EI_BENCH_BASE");
    set_route(base_route ? base_route : "");
    ei_engine e;
    ei_engine_load_backend(&e, argv[1], "metal");
    set_route(argv[2]);
    const char *stream_count = getenv("EI_BENCH_STREAMS");
    if (stream_count && atoi(stream_count) > 1) {
        streams(argv[2], &e, (size_t)atoi(stream_count), cases, n_cases);
        ei_engine_free(&e);
        return 0;
    }
    char err[512];
    void *candidate = ei_gemma2_metal_create(&e.model.gguf, err, sizeof err);
    if (!candidate)
        ei_die("%s", err);
    int iterations = getenv("EI_BENCH_ITERS") ? atoi(getenv("EI_BENCH_ITERS")) : 5;
    if (iterations < 1 || iterations > 16)
        ei_die("EI_BENCH_ITERS must be 1..16");
    printf("%-10s %12s %12s %8s %14s %12s\n", "case", "base ms", "route ms", "speedup",
           "min cosine", "route tok/s");
    for (size_t c = 0; c < n_cases; c++) {
        size_t lens[8192], total = 0;
        size_t b = parse_case(cases[c], lens, 8192, &total);
        int32_t *ids = ei_xmalloc(total * sizeof *ids);
        size_t *offsets = ei_xmalloc((b + 1) * sizeof *offsets);
        fill_case(lens, b, ids, offsets);
        float *a = ei_xmalloc(b * 768 * sizeof *a), *r = ei_xmalloc(b * 768 * sizeof *r);
        double base = run(e.gemma2_metal, ids, offsets, b, a, iterations);
        double fast = run(candidate, ids, offsets, b, r, iterations);
        double worst = 1;
        for (size_t s = 0; s < b; s++) {
            double ab = 0, aa = 0, rr = 0;
            for (size_t i = 0; i < 768; i++) {
                float x = a[s * 768 + i], y = r[s * 768 + i];
                if (!isfinite(y))
                    ei_die("non-finite route output at case %s", cases[c]);
                ab += (double)x * y;
                aa += (double)x * x;
                rr += (double)y * y;
            }
            double cosine = ab / sqrt(aa * rr);
            if (cosine < worst)
                worst = cosine;
        }
        printf("%-10s %12.3f %12.3f %7.2fx %14.9f %12.0f\n", cases[c], base, fast, base / fast,
               worst, (double)total / fast * 1000.0);
        fflush(stdout);
        free(ids);
        free(offsets);
        free(a);
        free(r);
    }
    ei_gemma2_metal_free(candidate);
    ei_engine_free(&e);
    return 0;
}
