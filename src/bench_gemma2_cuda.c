#define _POSIX_C_SOURCE 200809L
// Time EmbeddingGemma 2 CUDA waves and check packed-vs-single parity.
// usage: bench_gemma2_cuda text.gguf [TxB ...]   (B sequences of T tokens per wave)
#include "engine.h"
#include "gemma2_cuda.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}
static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
int main(int argc, char **argv) {
    if (argc < 2)
        ei_die("usage: %s text.gguf [TxB ...]", argv[0]);
    const char *defaults[] = {"1x1", "8x1", "32x1", "32x4", "32x16", "128x1", "128x16",
                              "512x1", "512x4", "512x16", "2048x1", "2048x4", "8192x1"};
    const char **cases = argc > 2 ? (const char **)argv + 2 : defaults;
    size_t n_cases = argc > 2 ? (size_t)argc - 2 : sizeof defaults / sizeof *defaults;
    ei_engine e;
    ei_engine_load_backend(&e, argv[1], "cuda");
    static const char text[] =
        "The mitochondrion is the organelle that powers the cell, converting nutrients into "
        "adenosine triphosphate through oxidative phosphorylation. Researchers in Lisbon "
        "measured how temperature, oxygen and diet change that output over several weeks, "
        "and published the data with their code so others could repeat the analysis. ";
    char *corpus_text = ei_xmalloc(sizeof text * 400);
    corpus_text[0] = 0;
    for (int i = 0; i < 400; i++)
        strcat(corpus_text, text);
    ei_tokens corpus;
    ei_tokenize_spm(&e.tokenizer, corpus_text, strlen(corpus_text), false, false, &corpus);
    char err[512];
    // Reference context with the CUDA-core attention kernel for route parity.
    setenv("EI_GEMMA2_CUDA_MMA_ATTN", "0", 1);
    void *scalar = ei_gemma2_cuda_create(&e.model.gguf, err, sizeof err);
    unsetenv("EI_GEMMA2_CUDA_MMA_ATTN");
    if (!scalar)
        ei_die("%s", err);
    printf("%-10s %10s %12s %14s %14s\n", "case", "ms", "tok/s", "packed/single", "vs cuda-core");
    for (size_t c = 0; c < n_cases; c++) {
        size_t t = 0, b = 0;
        if (sscanf(cases[c], "%zux%zu", &t, &b) != 2 || !t || !b || t * b > 32768)
            ei_die("bad case '%s'", cases[c]);
        int32_t *ids = ei_xmalloc(t * b * sizeof *ids);
        size_t *offsets = ei_xmalloc((b + 1) * sizeof *offsets);
        for (size_t i = 0; i < t * b; i++)
            ids[i] = corpus.ids[(i / t * 53 + i % t) % corpus.n];
        for (size_t s = 0; s <= b; s++)
            offsets[s] = s * t;
        float *out = ei_xmalloc(b * 768 * sizeof *out);
        double times[9];
        for (int i = -2; i < 9; i++) {
            double start = now_ms();
            if (!ei_gemma2_cuda_batch(e.gemma2_cuda, ids, offsets, b, out, err, sizeof err))
                ei_die("%s", err);
            if (i >= 0)
                times[i] = now_ms() - start;
        }
        qsort(times, 9, sizeof *times, cmp);
        double worst = 1;
        for (size_t s = 0; s < b && s < 4; s++) {
            float one[768];
            size_t off1[2] = {0, t};
            if (!ei_gemma2_cuda_batch(e.gemma2_cuda, ids + s * t, off1, 1, one, err, sizeof err))
                ei_die("%s", err);
            double ab = 0;
            for (int z = 0; z < 768; z++)
                ab += (double)one[z] * out[s * 768 + z];
            if (ab < worst)
                worst = ab;
        }
        float *ref = ei_xmalloc(b * 768 * sizeof *ref);
        if (!ei_gemma2_cuda_batch(scalar, ids, offsets, b, ref, err, sizeof err))
            ei_die("%s", err);
        double route = 1;
        for (size_t s = 0; s < b; s++) {
            double ab = 0;
            for (int z = 0; z < 768; z++)
                ab += (double)ref[s * 768 + z] * out[s * 768 + z];
            if (ab < route)
                route = ab;
        }
        free(ref);
        printf("%-10s %10.3f %12.0f %14.9f %14.9f\n", cases[c], times[4],
               (double)(t * b) / times[4] * 1e3, worst, route);
        fflush(stdout);
        free(ids), free(offsets), free(out);
    }
    return 0;
}
