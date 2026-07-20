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

static void fill_ids(int32_t *ids, int32_t tokens) {
    for (int32_t token = 0; token < tokens; token++) {
        ids[token] = 1000 + (token * 7919) % 240000;
    }
}

static float compare_routes(ei_engine *first, ei_engine *second,
                            const int32_t *shapes, size_t shape_count,
                            const char *label) {
    float minimum = 1.0f;
    for (size_t shape = 0; shape < shape_count; shape++) {
        const int32_t tokens = shapes[shape];
        int32_t *ids = ei_xmalloc(sizeof(*ids) * (size_t)tokens);
        fill_ids(ids, tokens);
        float first_output[EI_N_EMBD];
        float second_output[EI_N_EMBD];
        char err[256];
        if (!ei_engine_embed_tokens(first, ids, (size_t)tokens,
                                    first_output, err, sizeof err)) {
            ei_die("XPU %s first route T=%d failed: %s", label, tokens, err);
        }
        if (!ei_engine_embed_tokens(second, ids, (size_t)tokens,
                                    second_output, err, sizeof err)) {
            ei_die("XPU %s second route T=%d failed: %s", label, tokens, err);
        }
        const float similarity = cosine(first_output, second_output);
        if (!isfinite(similarity)) {
            ei_die("XPU %s T=%d produced a nonfinite cosine", label, tokens);
        }
        if (similarity < minimum) minimum = similarity;
        printf("XPU %s T=%d cosine=%.9f\n", label, tokens, similarity);
        free(ids);
    }
    printf("XPU %s minimum cosine=%.9f\n", label, minimum);
    return minimum;
}

#ifdef EI_XPU_XE2_FLASH
static float compare_packed_routes(
    ei_engine *first, ei_engine *second, const size_t *lengths,
    size_t batch_size, int repeats, const char *label) {
    size_t *offsets = ei_xmalloc((batch_size + 1) * sizeof(*offsets));
    offsets[0] = 0;
    for (size_t sequence = 0; sequence < batch_size; sequence++) {
        offsets[sequence + 1] = offsets[sequence] + lengths[sequence];
    }
    int32_t *ids = ei_xmalloc(offsets[batch_size] * sizeof(*ids));
    for (size_t sequence = 0; sequence < batch_size; sequence++) {
        for (size_t token = offsets[sequence]; token < offsets[sequence + 1];
             token++) {
            ids[token] = 1000 + (int32_t)(
                (token * 7919u + sequence * 104729u) % 240000u);
        }
    }
    float *first_output = ei_xmalloc(
        batch_size * EI_N_EMBD * sizeof(*first_output));
    float *second_output = ei_xmalloc(
        batch_size * EI_N_EMBD * sizeof(*second_output));
    float minimum = 1.0f;
    char err[256];
    for (int repeat = 0; repeat < repeats; repeat++) {
        if (!ei_engine_embed_tokens_batch(
                first, ids, offsets, batch_size,
                first_output, err, sizeof err)) {
            ei_die("XPU %s first route failed: %s", label, err);
        }
        if (!ei_engine_embed_tokens_batch(
                second, ids, offsets, batch_size,
                second_output, err, sizeof err)) {
            ei_die("XPU %s second route failed: %s", label, err);
        }
        for (size_t sequence = 0; sequence < batch_size; sequence++) {
            const float similarity = cosine(
                first_output + sequence * EI_N_EMBD,
                second_output + sequence * EI_N_EMBD);
            if (!isfinite(similarity)) {
                ei_die("XPU %s produced a nonfinite cosine", label);
            }
            if (similarity < minimum) minimum = similarity;
        }
    }
    printf("XPU %s repeated minimum cosine=%.9f\n", label, minimum);
    free(second_output);
    free(first_output);
    free(ids);
    free(offsets);
    return minimum;
}
#endif

int main(int argc, char **argv) {
    if (argc != 2) ei_die("usage: %s <model.gguf>", argv[0]);
    const int32_t shapes[] = {1, 2, 7, 32, 128};
    ei_engine cpu;
    ei_engine xpu;
    ei_engine_load_backend(&cpu, argv[1], "cpu");
    ei_engine_load_backend(&xpu, argv[1], "xpu");

    float minimum = 1.0f;
    for (size_t shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]); shape++) {
        const int32_t tokens = shapes[shape];
        int32_t *ids = ei_xmalloc(sizeof(*ids) * (size_t)tokens);
        fill_ids(ids, tokens);
        float cpu_output[EI_N_EMBD];
        float xpu_output[EI_N_EMBD];
        char err[256];
        if (!ei_engine_embed_tokens(&cpu, ids, (size_t)tokens,
                                    cpu_output, err, sizeof err)) {
            ei_die("CPU T=%d failed: %s", tokens, err);
        }
        if (!ei_engine_embed_tokens(&xpu, ids, (size_t)tokens,
                                    xpu_output, err, sizeof err)) {
            ei_die("XPU T=%d failed: %s", tokens, err);
        }
        const float similarity = cosine(cpu_output, xpu_output);
        if (!isfinite(similarity)) {
            ei_die("CPU/XPU T=%d produced a nonfinite cosine", tokens);
        }
        if (similarity < minimum) minimum = similarity;
        printf("synthetic backend drift T=%d cpu/xpu %.9f\n",
               tokens, similarity);
        free(ids);
        if (similarity < 0.99f) {
            ei_engine_free(&xpu);
            ei_engine_free(&cpu);
            return 1;
        }
    }
    printf("synthetic backend drift: cpu/xpu %.9f\n", minimum);
    ei_engine_free(&xpu);
    ei_engine_free(&cpu);

    const int32_t projection_shapes[] = {1, 7, 32, 129};
    ei_engine direct_projection;
    ei_engine gemm_projection;
    setenv("EI_XPU_GEMM_MIN_TOKENS", "65536", 1);
    ei_engine_load_backend(&direct_projection, argv[1], "xpu");
    setenv("EI_XPU_GEMM_MIN_TOKENS", "1", 1);
    ei_engine_load_backend(&gemm_projection, argv[1], "xpu");
    unsetenv("EI_XPU_GEMM_MIN_TOKENS");
    minimum = compare_routes(
        &direct_projection, &gemm_projection, projection_shapes,
        sizeof(projection_shapes) / sizeof(projection_shapes[0]),
        "packed-Q4/FP16-GEMM projection parity");
    ei_engine_free(&gemm_projection);
    ei_engine_free(&direct_projection);
    if (minimum < 0.9998f) return 1;

    const int32_t attention_shapes[] = {128, 512, 2048};
    ei_engine online_attention;
    ei_engine tensor_attention;
    setenv("EI_XPU_TENSOR_ATTENTION_MIN_TOKENS", "65536", 1);
    ei_engine_load_backend(&online_attention, argv[1], "xpu");
    setenv("EI_XPU_TENSOR_ATTENTION_MIN_TOKENS", "1", 1);
    ei_engine_load_backend(&tensor_attention, argv[1], "xpu");
    unsetenv("EI_XPU_TENSOR_ATTENTION_MIN_TOKENS");
    minimum = compare_routes(
        &online_attention, &tensor_attention, attention_shapes,
        sizeof(attention_shapes) / sizeof(attention_shapes[0]),
        "online/tensor attention parity");
    ei_engine_free(&tensor_attention);
    ei_engine_free(&online_attention);
    if (minimum < 0.999f) return 1;

#ifdef EI_XPU_XE2_FLASH
    const size_t packed_flash_lengths[] = {256, 257, 384, 511};
    ei_engine conventional_attention;
    ei_engine flash_attention;
    setenv("EI_XPU_XE2_FLASH", "0", 1);
    ei_engine_load_backend(&conventional_attention, argv[1], "xpu");
    setenv("EI_XPU_XE2_FLASH", "1", 1);
    ei_engine_load_backend(&flash_attention, argv[1], "xpu");
    unsetenv("EI_XPU_XE2_FLASH");
    minimum = compare_packed_routes(
        &conventional_attention, &flash_attention, packed_flash_lengths,
        sizeof(packed_flash_lengths) / sizeof(packed_flash_lengths[0]), 20,
        "mixed packed/Flash attention parity");
    const size_t packed_flash_boundary_lengths[] = {128, 129, 192, 255};
    const float boundary_minimum = compare_packed_routes(
        &conventional_attention, &flash_attention,
        packed_flash_boundary_lengths,
        sizeof(packed_flash_boundary_lengths) /
            sizeof(packed_flash_boundary_lengths[0]),
        20, "boundary packed/Flash attention parity");
    if (boundary_minimum < minimum) minimum = boundary_minimum;
    ei_engine_free(&flash_attention);
    ei_engine_free(&conventional_attention);
    if (minimum < 0.9997f) return 1;
#endif
    return 0;
}
