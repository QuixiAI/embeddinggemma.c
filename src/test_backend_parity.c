#define _POSIX_C_SOURCE 200809L

#include "engine.h"

#include <math.h>

#define MIN_SYNTHETIC_CPU_METAL_COSINE 0.99f
#define MIN_ROUTE_COSINE 0.999999f

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
    if (argc != 2) ei_die("usage: %s <model.gguf>", argv[0]);
    const int32_t production_shapes[] = { 1, 6, 7, 32, 63, 128, 512 };
    const int32_t tile_shapes[] = { 1, 7, 64, 512, 2048 };
    ei_engine cpu;
    ei_engine metal;
    ei_engine metal_r1;
    ei_engine metal8;
    ei_engine metal16;
    ei_engine metal_kv_f32;
    ei_engine metal_kv_f16;
    ei_engine_load_backend(&cpu, argv[1], "cpu");

    unsetenv("EI_METAL_GEMM_MIN_TOKENS");
    unsetenv("EI_METAL_GEMM_TILE_TOKENS");
    unsetenv("EI_METAL_GEMV_R4_MIN_TOKENS");
    ei_engine_load_backend(&metal, argv[1], "metal");
    setenv("EI_METAL_GEMV_R4_MIN_TOKENS", "64", 1);
    ei_engine_load_backend(&metal_r1, argv[1], "metal");

    float minimum_cpu_metal = 1.0f;
    float minimum_r1_r4 = 1.0f;
    for (size_t shape = 0;
         shape < sizeof(production_shapes) / sizeof(production_shapes[0]); shape++) {
        int32_t tokens = production_shapes[shape];
        int32_t *ids = ei_xmalloc(sizeof(*ids) * (size_t)tokens);
        for (int32_t i = 0; i < tokens; i++) ids[i] = 1000 + (i * 7919) % 240000;
        float cpu_output[EI_N_EMBD];
        float metal_output[EI_N_EMBD];
        float metal_r1_output[EI_N_EMBD];
        char err[256];
        if (!ei_engine_embed_tokens(&cpu, ids, (size_t)tokens,
                                    cpu_output, err, sizeof err)) {
            ei_die("CPU T=%d failed: %s", tokens, err);
        }
        if (!ei_engine_embed_tokens(&metal, ids, (size_t)tokens,
                                    metal_output, err, sizeof err)) {
            ei_die("Metal production T=%d failed: %s", tokens, err);
        }
        float similarity = cosine(cpu_output, metal_output);
        if (similarity < minimum_cpu_metal) minimum_cpu_metal = similarity;
        float route_similarity = 1.0f;
        if (tokens < 64) {
            if (!ei_engine_embed_tokens(&metal_r1, ids, (size_t)tokens,
                                        metal_r1_output, err, sizeof err)) {
                ei_die("Metal forced-R1 T=%d failed: %s", tokens, err);
            }
            route_similarity = cosine(metal_output, metal_r1_output);
            if (route_similarity < minimum_r1_r4) minimum_r1_r4 = route_similarity;
        }
        printf("synthetic backend drift T=%d cpu/metal %.9f production/R1 %.9f\n",
               tokens, similarity, route_similarity);
        free(ids);
        /* Arbitrary token IDs catch gross backend drift. Release acceptance
         * against llama.cpp goldens is enforced separately by test_embed.c. */
        if (similarity < MIN_SYNTHETIC_CPU_METAL_COSINE ||
            route_similarity < MIN_ROUTE_COSINE) {
            ei_engine_free(&metal_r1);
            ei_engine_free(&metal);
            ei_engine_free(&cpu);
            return 1;
        }
    }
    ei_engine_free(&metal_r1);
    ei_engine_free(&metal);

    unsetenv("EI_METAL_GEMV_R4_MIN_TOKENS");
    setenv("EI_METAL_FP16_KV_MIN_TOKENS", "65536", 1);
    setenv("EI_METAL_FLASH_ATTN_MIN_TOKENS", "65536", 1);
    ei_engine_load_backend(&metal_kv_f32, argv[1], "metal");
    setenv("EI_METAL_FP16_KV_MIN_TOKENS", "1", 1);
    setenv("EI_METAL_FLASH_ATTN_MIN_TOKENS", "1", 1);
    ei_engine_load_backend(&metal_kv_f16, argv[1], "metal");
    unsetenv("EI_METAL_FP16_KV_MIN_TOKENS");
    unsetenv("EI_METAL_FLASH_ATTN_MIN_TOKENS");
    const int32_t kv_shapes[] = { 1024, 2048 };
    float minimum_kv = 1.0f;
    for (size_t shape = 0; shape < sizeof(kv_shapes) / sizeof(kv_shapes[0]); shape++) {
        int32_t tokens = kv_shapes[shape];
        int32_t *ids = ei_xmalloc(sizeof(*ids) * (size_t)tokens);
        for (int32_t i = 0; i < tokens; i++) ids[i] = 1000 + (i * 7919) % 240000;
        float f32_output[EI_N_EMBD];
        float f16_output[EI_N_EMBD];
        char err[256];
        if (!ei_engine_embed_tokens(&metal_kv_f32, ids, (size_t)tokens,
                                    f32_output, err, sizeof err) ||
            !ei_engine_embed_tokens(&metal_kv_f16, ids, (size_t)tokens,
                                    f16_output, err, sizeof err)) {
            ei_die("Metal attention parity T=%d failed: %s", tokens, err);
        }
        float similarity = cosine(f32_output, f16_output);
        if (similarity < minimum_kv) minimum_kv = similarity;
        printf("Metal attention parity T=%d legacy/flash %.9f\n",
               tokens, similarity);
        free(ids);
        if (similarity < 0.99999f) {
            ei_engine_free(&metal_kv_f16);
            ei_engine_free(&metal_kv_f32);
            ei_engine_free(&cpu);
            return 1;
        }
    }
    ei_engine_free(&metal_kv_f16);
    ei_engine_free(&metal_kv_f32);

    setenv("EI_METAL_GEMM_MIN_TOKENS", "1", 1);
    setenv("EI_METAL_GEMM_TILE_TOKENS", "8", 1);
    ei_engine_load_backend(&metal8, argv[1], "metal");
    setenv("EI_METAL_GEMM_TILE_TOKENS", "16", 1);
    ei_engine_load_backend(&metal16, argv[1], "metal");

    float minimum_tile = 1.0f;
    for (size_t shape = 0; shape < sizeof(tile_shapes) / sizeof(tile_shapes[0]); shape++) {
        int32_t tokens = tile_shapes[shape];
        int32_t *ids = ei_xmalloc(sizeof(*ids) * (size_t)tokens);
        for (int32_t i = 0; i < tokens; i++) ids[i] = 1000 + (i * 7919) % 240000;
        float metal8_output[EI_N_EMBD];
        float metal16_output[EI_N_EMBD];
        char err[256];
        if (!ei_engine_embed_tokens(&metal8, ids, (size_t)tokens,
                                    metal8_output, err, sizeof err)) {
            ei_die("Metal 32x8 T=%d failed: %s", tokens, err);
        }
        if (!ei_engine_embed_tokens(&metal16, ids, (size_t)tokens,
                                    metal16_output, err, sizeof err)) {
            ei_die("Metal 16x16 T=%d failed: %s", tokens, err);
        }
        float tile_similarity = cosine(metal8_output, metal16_output);
        if (tile_similarity < minimum_tile) minimum_tile = tile_similarity;
        printf("Metal diagnostic tile parity T=%d tile8/tile16 %.9f\n",
               tokens, tile_similarity);
        free(ids);
        if (tile_similarity < 0.999999f) {
            ei_engine_free(&metal16);
            ei_engine_free(&metal8);
            ei_engine_free(&cpu);
            return 1;
        }
    }
    printf("synthetic backend drift: cpu/metal %.9f, production/R1 %.9f, "
           "legacy/flash attention %.9f, tile8/tile16 %.9f\n",
           minimum_cpu_metal, minimum_r1_r4, minimum_kv, minimum_tile);
    ei_engine_free(&metal16);
    ei_engine_free(&metal8);
    ei_engine_free(&cpu);
    return 0;
}
