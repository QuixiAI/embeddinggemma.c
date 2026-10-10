#define _POSIX_C_SOURCE 200809L
// Compare and time the EmbeddingGemma 2 fast CPU graph against the reference.
// usage: bench_gemma2_cpu text.gguf [TxB ...]   (EI_THREADS sets the pool size)
// Each case packs B sequences of T tokens. The reference runs them one by one,
// as the engine did before the fast path.
#include "engine.h"
#include "gemma2.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}
int main(int argc, char **argv) {
    if (argc < 2)
        ei_die("usage: %s text.gguf [TxB ...]", argv[0]);
    const char *defaults[] = {"1x1", "8x1", "32x1", "32x4", "128x1", "32x16", "512x1"};
    const char **cases = argc > 2 ? (const char **)argv + 2 : defaults;
    size_t n_cases = argc > 2 ? (size_t)argc - 2 : sizeof defaults / sizeof *defaults;
    ei_engine e;
    ei_engine_load_backend(&e, argv[1], "cpu");
    bool skip_reference = getenv("EI_BENCH_FAST_ONLY") != NULL;
    bool random_ids = getenv("EI_BENCH_RANDOM_IDS") != NULL;
    static const char text[] =
        "The mitochondrion is the organelle that powers the cell, converting nutrients into "
        "adenosine triphosphate through oxidative phosphorylation. Researchers in Lisbon "
        "measured how temperature, oxygen and diet change that output over several weeks, "
        "and published the data with their code so others could repeat the analysis. ";
    char *corpus_text = ei_xmalloc(sizeof text * 40);
    corpus_text[0] = 0;
    for (int i = 0; i < 40; i++)
        strcat(corpus_text, text);
    ei_tokens corpus;
    ei_tokenize_spm(&e.tokenizer, corpus_text, strlen(corpus_text), false, false, &corpus);
    printf("%d threads\n%-10s %12s %12s %8s %14s %12s\n", ei_engine_threads(&e), "case",
           "reference ms", "fast ms", "speedup", "min cosine", "fast tok/s");
    char err[512];
    for (size_t c = 0; c < n_cases; c++) {
        size_t t = 0, b = 0;
        if (sscanf(cases[c], "%zux%zu", &t, &b) != 2 || !t || !b || t * b > 8192)
            ei_die("bad case '%s'", cases[c]);
        int32_t *ids = ei_xmalloc(t * b * sizeof *ids);
        size_t *offsets = ei_xmalloc((b + 1) * sizeof *offsets);
        // Natural-text token windows: random ids amplify Q8 activation rounding
        // far beyond real inputs (CPU reference vs FP32 Metal drops to 0.93).
        for (size_t i = 0; i < t * b; i++)
            ids[i] = random_ids ? (int32_t)(100 + (i * 137 + i / t * 7919) % 20000)
                                : corpus.ids[(i / t * 53 + i % t) % corpus.n];
        for (size_t s = 0; s <= b; s++)
            offsets[s] = s * t;
        float *ref = ei_xmalloc(b * 768 * sizeof *ref), *fast = ei_xmalloc(b * 768 * sizeof *fast);
        double ref_ms = 0;
        if (!skip_reference) {
            double start = now_ms();
            for (size_t s = 0; s < b; s++)
                if (!ei_gemma2_embed(&e.model.gguf, e.thread_pool, ids + s * t, NULL, t,
                                     ref + s * 768, err, sizeof err))
                    ei_die("%s", err);
            ref_ms = now_ms() - start;
        }
        if (!ei_gemma2_embed_batch_fast(&e.model.gguf, e.thread_pool, ids, NULL, offsets, b, fast, err,
                                        sizeof err)) // warm
            ei_die("%s", err);
        double times[5];
        for (int i = 0; i < 5; i++) {
            double start = now_ms();
            if (!ei_gemma2_embed_batch_fast(&e.model.gguf, e.thread_pool, ids, NULL, offsets, b,
                                            fast, err, sizeof err))
                ei_die("%s", err);
            times[i] = now_ms() - start;
        }
        for (int i = 1; i < 5; i++)
            for (int j = i; j > 0 && times[j] < times[j - 1]; j--) {
                double tmp = times[j];
                times[j] = times[j - 1];
                times[j - 1] = tmp;
            }
        double fast_ms = times[2], worst = 1;
        for (size_t s = 0; s < b && !skip_reference; s++) {
            double ab = 0, aa = 0, rr = 0;
            for (size_t i = 0; i < 768; i++) {
                double x = ref[s * 768 + i], y = fast[s * 768 + i];
                ab += x * y, aa += x * x, rr += y * y;
            }
            if (ab / sqrt(aa * rr) < worst)
                worst = ab / sqrt(aa * rr);
        }
        printf("%-10s %12.2f %12.2f %7.2fx %14.9f %12.0f\n", cases[c], ref_ms, fast_ms,
               skip_reference ? 0 : ref_ms / fast_ms, worst, (double)(t * b) / fast_ms * 1000);
        fflush(stdout);
        free(ids), free(offsets), free(ref), free(fast);
    }
    ei_tokens_free(&corpus);
    free(corpus_text);
    ei_engine_free(&e);
    return 0;
}
