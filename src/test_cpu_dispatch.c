/* x86-64 runtime dispatch parity: embeds the same inputs through the forced
 * SSE2 baseline and the AVX2/FMA/F16C kernels in one process and compares them at
 * every Matryoshka dimension. Also times one ~150-token chunk per route. */
#define _POSIX_C_SOURCE 200809L

#include "engine.h"
#include "kernels_avx2.h"

#include <errno.h>
#include <math.h>
#include <time.h>

/* The routes differ only in float summation order and FMA, but a last-bit
 * difference can flip a Q8_0 activation rounding and cascade through 24
 * layers, so the gate matches the llama.cpp golden gate rather than
 * bit-equality. Real-model results sit near cosine 0.9999. */
#define PARITY_MIN_COSINE 0.999
#define PARITY_MAX_ABS_DIFF 1e-2

static const int32_t dimensions[] = { EI_N_EMBD, 512, 256, 128 };
#define N_DIMS ((int)(sizeof dimensions / sizeof dimensions[0]))

typedef struct {
    char **items;
    size_t n;
} string_list;

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) ei_die("%s: cannot open", path);
    if (fseek(f, 0, SEEK_END) != 0) ei_die("%s: seek failed", path);
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) ei_die("%s: tell failed", path);
    char *data = ei_xmalloc((size_t)n + 1u);
    if (fread(data, 1, (size_t)n, f) != (size_t)n) ei_die("%s: read failed", path);
    fclose(f);
    data[n] = '\0';
    return data;
}

/* A flat JSON array of strings with simple escapes (testdata/test-strings.json). */
static string_list parse_strings(const char *path) {
    char *data = read_file(path);
    string_list list = { 0 };
    const char *p = strchr(data, '[');
    if (!p) ei_die("%s: expected a JSON array", path);
    for (p++; *p && *p != ']'; p++) {
        if (*p != '"') continue;
        char *s = ei_xmalloc(strlen(p) + 1u);
        size_t len = 0;
        for (p++; *p != '"'; p++) {
            if (!*p) ei_die("%s: unterminated string", path);
            char c = *p;
            if (c == '\\') {
                c = *++p;
                if (c == 'n') c = '\n';
                else if (c == 't') c = '\t';
                else if (c != '"' && c != '\\' && c != '/') ei_die("%s: unsupported escape", path);
            }
            s[len++] = c;
        }
        s[len] = '\0';
        list.items = ei_xrealloc(list.items, sizeof(char *) * (list.n + 1u));
        list.items[list.n++] = s;
    }
    free(data);
    return list;
}

/* An input of exactly n_tokens ids, cycling through the tokenized samples. */
static ei_tokens repeated_tokens(const ei_tokens *samples, size_t n_samples, size_t n_tokens) {
    ei_tokens out = { ei_xmalloc(sizeof(int32_t) * n_tokens), n_tokens };
    size_t len = 0;
    for (size_t i = 0; len < n_tokens; i = (i + 1) % n_samples) {
        for (size_t j = 0; j < samples[i].n && len < n_tokens; j++) {
            out.ids[len++] = samples[i].ids[j];
        }
    }
    return out;
}

static void embed_as(ei_engine *e, const char *isa, const ei_tokens *tokens,
                     float out[EI_N_EMBD]) {
    if (!ei_cpu_set_isa(isa)) ei_die("cannot select CPU ISA %s", isa);
    char err[256];
    if (!ei_engine_embed_tokens(e, tokens->ids, tokens->n, out, err, sizeof err)) {
        ei_die("embedding failed under %s: %s", isa, err);
    }
}

static void embed_batch_as(ei_engine *e, const char *isa, const ei_tokens *tokens,
                           size_t n, float *out) {
    if (!ei_cpu_set_isa(isa)) ei_die("cannot select CPU ISA %s", isa);
    size_t total = 0;
    for (size_t i = 0; i < n; i++) total += tokens[i].n;
    int32_t *ids = ei_xmalloc(sizeof(int32_t) * total);
    size_t *offsets = ei_xmalloc(sizeof(size_t) * (n + 1u));
    offsets[0] = 0;
    for (size_t i = 0; i < n; i++) {
        memcpy(ids + offsets[i], tokens[i].ids, sizeof(int32_t) * tokens[i].n);
        offsets[i + 1] = offsets[i] + tokens[i].n;
    }
    char err[256];
    if (!ei_engine_embed_tokens_batch(e, ids, offsets, n, out, err, sizeof err)) {
        ei_die("batch embedding failed under %s: %s", isa, err);
    }
    free(ids);
    free(offsets);
}

static double max_abs[N_DIMS];
static double min_cos[N_DIMS];
static int failures;

static void compare(const char *label, const float *baseline, const float *avx2) {
    for (int d = 0; d < N_DIMS; d++) {
        float a[EI_N_EMBD], b[EI_N_EMBD];
        memcpy(a, baseline, sizeof a);
        memcpy(b, avx2, sizeof b);
        if (!ei_embedding_normalize_prefix(a, dimensions[d]) ||
            !ei_embedding_normalize_prefix(b, dimensions[d])) {
            ei_die("bad dimension %d", dimensions[d]);
        }
        double dot = 0.0, na = 0.0, nb = 0.0, diff = 0.0;
        for (int32_t i = 0; i < dimensions[d]; i++) {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
            double delta = fabs((double)a[i] - b[i]);
            if (!(delta <= diff)) diff = delta; /* also catches NaN */
        }
        double cosine = dot / sqrt(na * nb);
        if (!(cosine >= min_cos[d])) min_cos[d] = cosine;
        if (!(diff <= max_abs[d])) max_abs[d] = diff;
        if (!(cosine >= PARITY_MIN_COSINE) || !(diff <= PARITY_MAX_ABS_DIFF)) {
            printf("FAIL %s D=%d: SSE2 vs AVX2 cosine %.9f, max abs diff %.3g\n",
                   label, dimensions[d], cosine, diff);
            failures++;
        }
    }
}

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        ei_die("clock_gettime failed: %s", strerror(errno));
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double time_chunk(ei_engine *e, const char *isa, const ei_tokens *tokens, int reps) {
    float out[EI_N_EMBD];
    embed_as(e, isa, tokens, out); /* warm-up */
    double best = INFINITY;
    for (int r = 0; r < reps; r++) {
        double start = now_seconds();
        embed_as(e, isa, tokens, out);
        double elapsed = now_seconds() - start;
        if (elapsed < best) best = elapsed;
    }
    return best;
}

int main(int argc, char **argv) {
    if (argc != 3) ei_die("usage: %s <model.gguf> <test-strings.json>", argv[0]);
    if (!ei_cpu_avx2_supported()) {
        printf("cpu dispatch parity: skipped (no x86-64 AVX2+FMA+F16C route on this host)\n");
        return 0;
    }

    ei_engine engine;
    ei_engine_load_backend(&engine, argv[1], "cpu");
    string_list texts = parse_strings(argv[2]);
    /* The samples, then 150 tokens (one typical chunk) and 600 tokens (past
     * the 512-token sliding window). */
    const size_t n_samples = texts.n, n_inputs = texts.n + 2u;
    ei_tokens *tokens = ei_xmalloc(sizeof(ei_tokens) * n_inputs);
    for (size_t i = 0; i < n_samples; i++) {
        ei_tokenize_spm(&engine.tokenizer, texts.items[i], strlen(texts.items[i]),
                        true, false, &tokens[i]);
    }
    tokens[n_samples] = repeated_tokens(tokens, n_samples, 150u);
    tokens[n_samples + 1u] = repeated_tokens(tokens, n_samples, 600u);

    for (int d = 0; d < N_DIMS; d++) {
        max_abs[d] = 0.0;
        min_cos[d] = 1.0;
    }

    for (size_t i = 0; i < n_inputs; i++) {
        float baseline[EI_N_EMBD], avx2[EI_N_EMBD];
        embed_as(&engine, "baseline", &tokens[i], baseline);
        embed_as(&engine, "avx2", &tokens[i], avx2);
        char label[64];
        snprintf(label, sizeof label, "input %zu (%zu tokens)", i, tokens[i].n);
        compare(label, baseline, avx2);
    }

    /* The batched route uses the multi-input Q4_0 kernel. */
    float *batch_baseline = ei_xmalloc(sizeof(float) * EI_N_EMBD * n_samples);
    float *batch_avx2 = ei_xmalloc(sizeof(float) * EI_N_EMBD * n_samples);
    embed_batch_as(&engine, "baseline", tokens, n_samples, batch_baseline);
    embed_batch_as(&engine, "avx2", tokens, n_samples, batch_avx2);
    for (size_t i = 0; i < n_samples; i++) {
        char label[64];
        snprintf(label, sizeof label, "batch item %zu", i);
        compare(label, batch_baseline + i * EI_N_EMBD, batch_avx2 + i * EI_N_EMBD);
    }

    for (int d = 0; d < N_DIMS; d++) {
        printf("cpu dispatch parity D=%d: min cosine %.9f, max abs diff %.3g\n",
               dimensions[d], min_cos[d], max_abs[d]);
    }
    if (failures) {
        ei_die("cpu dispatch parity: %d comparisons outside cosine >= %.3f, "
               "max abs <= %.0e", failures, PARITY_MIN_COSINE, PARITY_MAX_ABS_DIFF);
    }
    printf("cpu dispatch parity: %zu inputs + %zu batched, SSE2 vs AVX2 within "
           "cosine >= %.3f and max abs <= %.0e\n",
           n_inputs, n_samples, PARITY_MIN_COSINE, PARITY_MAX_ABS_DIFF);

    const ei_tokens *chunk = &tokens[n_samples];
    size_t chunk_tokens = chunk->n;
    double sse2 = time_chunk(&engine, "baseline", chunk, 3);
    double avx2 = time_chunk(&engine, "avx2", chunk, 3);
    printf("cpu dispatch timing: %zu-token chunk, %d threads: SSE2 %.1f ms "
           "(%.0f tok/s), AVX2 %.1f ms (%.0f tok/s), speedup %.2fx\n",
           chunk_tokens, ei_engine_threads(&engine), sse2 * 1e3,
           (double)chunk_tokens / sse2, avx2 * 1e3, (double)chunk_tokens / avx2,
           sse2 / avx2);

    for (size_t i = 0; i < n_samples; i++) {
        ei_tokens_free(&tokens[i]);
        free(texts.items[i]);
    }
    free(tokens[n_samples].ids);
    free(tokens[n_samples + 1u].ids);
    free(tokens);
    free(texts.items);
    free(batch_baseline);
    free(batch_avx2);
    ei_engine_free(&engine);
    return 0;
}
