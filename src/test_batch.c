#define _POSIX_C_SOURCE 200809L

#include "engine.h"

#include <math.h>

static float cosine(const float *a, const float *b) {
    double dot = 0.0;
    double aa = 0.0;
    double bb = 0.0;
    for (int32_t i = 0; i < EI_N_EMBD; i++) {
        dot += (double)a[i] * (double)b[i];
        aa += (double)a[i] * (double)a[i];
        bb += (double)b[i] * (double)b[i];
    }
    return (float)(dot / sqrt(aa * bb));
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        ei_die("usage: %s <model.gguf> [cpu|metal|cuda]", argv[0]);
    }
    const char *backend = argc == 3 ? argv[2] : "cpu";
    const size_t lengths[] = { 1, 7, 32, 129 };
    const size_t batch_size = sizeof(lengths) / sizeof(lengths[0]);
    size_t offsets[batch_size + 1];
    offsets[0] = 0;
    for (size_t i = 0; i < batch_size; i++) offsets[i + 1] = offsets[i] + lengths[i];

    int32_t *ids = ei_xmalloc(sizeof(*ids) * offsets[batch_size]);
    for (size_t sequence = 0; sequence < batch_size; sequence++) {
        for (size_t token = offsets[sequence]; token < offsets[sequence + 1]; token++) {
            ids[token] = 1000 + (int32_t)((token * 7919 + sequence * 104729) % 240000);
        }
    }

    ei_engine engine;
    ei_engine_load_backend(&engine, argv[1], backend);
    float *reference = ei_xmalloc(sizeof(float) * batch_size * EI_N_EMBD);
    float *batched = ei_xmalloc(sizeof(float) * batch_size * EI_N_EMBD);
    char err[256];
    for (size_t sequence = 0; sequence < batch_size; sequence++) {
        if (!ei_engine_embed_tokens(
                &engine, ids + offsets[sequence], lengths[sequence],
                reference + sequence * EI_N_EMBD, err, sizeof err)) {
            ei_die("%s single sequence %zu failed: %s", backend, sequence, err);
        }
    }
    if (!ei_engine_embed_tokens_batch(
            &engine, ids, offsets, batch_size, batched, err, sizeof err)) {
        ei_die("%s packed batch failed: %s", backend, err);
    }

    float minimum = 1.0f;
    for (size_t sequence = 0; sequence < batch_size; sequence++) {
        float similarity = cosine(reference + sequence * EI_N_EMBD,
                                  batched + sequence * EI_N_EMBD);
        if (!isfinite(similarity)) {
            ei_die("%s batch parity produced a nonfinite cosine", backend);
        }
        if (similarity < minimum) minimum = similarity;
        printf("%s batch parity sequence=%zu tokens=%zu cosine=%.9f\n",
               backend, sequence, lengths[sequence], similarity);
    }

    free(batched);
    free(reference);
    free(ids);
    ei_engine_free(&engine);
    float minimum_expected = 0.9999f;
    if (strcmp(backend, "cuda") == 0) minimum_expected = 0.998f;
    if (strcmp(backend, "rocm") == 0) minimum_expected = 0.999f;
    if (strcmp(backend, "xpu") == 0) minimum_expected = 0.999f;
    if (minimum < minimum_expected) {
        fprintf(stderr, "%s packed batch parity failed: minimum cosine %.9f\n",
                backend, minimum);
        return 1;
    }
    printf("%s packed batch parity: %.9f\n", backend, minimum);
    return 0;
}
