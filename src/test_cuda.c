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
    if (argc != 2) ei_die("usage: %s <model.gguf>", argv[0]);
    const int32_t shapes[] = {1, 7, 32, 128};
    ei_engine cpu;
    ei_engine cuda;
    ei_engine_load_backend(&cpu, argv[1], "cpu");
    ei_engine_load_backend(&cuda, argv[1], "cuda");

    float minimum = 1.0f;
    for (size_t shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]); shape++) {
        const int32_t tokens = shapes[shape];
        int32_t *ids = ei_xmalloc(sizeof(*ids) * (size_t)tokens);
        for (int32_t token = 0; token < tokens; token++) {
            ids[token] = 1000 + (token * 7919) % 240000;
        }
        float cpu_output[EI_N_EMBD];
        float cuda_output[EI_N_EMBD];
        char err[256];
        if (!ei_engine_embed_tokens(&cpu, ids, (size_t)tokens,
                                    cpu_output, err, sizeof err)) {
            ei_die("CPU T=%d failed: %s", tokens, err);
        }
        if (!ei_engine_embed_tokens(&cuda, ids, (size_t)tokens,
                                    cuda_output, err, sizeof err)) {
            ei_die("CUDA T=%d failed: %s", tokens, err);
        }
        const float similarity = cosine(cpu_output, cuda_output);
        if (similarity < minimum) minimum = similarity;
        printf("synthetic backend drift T=%d cpu/cuda %.9f\n",
               tokens, similarity);
        free(ids);
        if (similarity < 0.99f) {
            ei_engine_free(&cuda);
            ei_engine_free(&cpu);
            return 1;
        }
    }
    printf("synthetic backend drift: cpu/cuda %.9f\n", minimum);
    ei_engine_free(&cuda);
    ei_engine_free(&cpu);

    const int32_t attention_shapes[] = {256, 512, 2048};
    ei_engine online_attention;
    ei_engine tensor_attention;
    setenv("EI_CUDA_TENSOR_ATTENTION_MIN_TOKENS", "65536", 1);
    ei_engine_load_backend(&online_attention, argv[1], "cuda");
    setenv("EI_CUDA_TENSOR_ATTENTION_MIN_TOKENS", "1", 1);
    ei_engine_load_backend(&tensor_attention, argv[1], "cuda");
    unsetenv("EI_CUDA_TENSOR_ATTENTION_MIN_TOKENS");
    minimum = 1.0f;
    for (size_t shape = 0;
         shape < sizeof(attention_shapes) / sizeof(attention_shapes[0]);
         shape++) {
        const int32_t tokens = attention_shapes[shape];
        int32_t *ids = ei_xmalloc(sizeof(*ids) * (size_t)tokens);
        for (int32_t token = 0; token < tokens; token++) {
            ids[token] = 1000 + (token * 7919) % 240000;
        }
        float online_output[EI_N_EMBD];
        float tensor_output[EI_N_EMBD];
        char err[256];
        if (!ei_engine_embed_tokens(&online_attention, ids, (size_t)tokens,
                                    online_output, err, sizeof err)) {
            ei_die("CUDA online attention T=%d failed: %s", tokens, err);
        }
        if (!ei_engine_embed_tokens(&tensor_attention, ids, (size_t)tokens,
                                    tensor_output, err, sizeof err)) {
            ei_die("CUDA tensor attention T=%d failed: %s", tokens, err);
        }
        const float similarity = cosine(online_output, tensor_output);
        if (similarity < minimum) minimum = similarity;
        printf("CUDA attention parity T=%d online/tensor %.9f\n",
               tokens, similarity);
        free(ids);
    }
    ei_engine_free(&tensor_attention);
    ei_engine_free(&online_attention);
    if (minimum < 0.999f) return 1;
    printf("CUDA attention parity: %.9f\n", minimum);

    const int32_t qkv_shapes[] = {7, 128, 2048};
    ei_engine split_qkv;
    ei_engine direct_qkv;
    setenv("EI_CUDA_DIRECT_FP16_QKV", "0", 1);
    ei_engine_load_backend(&split_qkv, argv[1], "cuda");
    setenv("EI_CUDA_DIRECT_FP16_QKV", "1", 1);
    ei_engine_load_backend(&direct_qkv, argv[1], "cuda");
    unsetenv("EI_CUDA_DIRECT_FP16_QKV");
    minimum = 1.0f;
    for (size_t shape = 0;
         shape < sizeof(qkv_shapes) / sizeof(qkv_shapes[0]); shape++) {
        const int32_t tokens = qkv_shapes[shape];
        int32_t *ids = ei_xmalloc(sizeof(*ids) * (size_t)tokens);
        for (int32_t token = 0; token < tokens; token++) {
            ids[token] = 1000 + (token * 7919) % 240000;
        }
        float split_output[EI_N_EMBD];
        float direct_output[EI_N_EMBD];
        char err[256];
        if (!ei_engine_embed_tokens(&split_qkv, ids, (size_t)tokens,
                                    split_output, err, sizeof err)) {
            ei_die("CUDA split QKV T=%d failed: %s", tokens, err);
        }
        if (!ei_engine_embed_tokens(&direct_qkv, ids, (size_t)tokens,
                                    direct_output, err, sizeof err)) {
            ei_die("CUDA direct FP16 QKV T=%d failed: %s", tokens, err);
        }
        const float similarity = cosine(split_output, direct_output);
        if (similarity < minimum) minimum = similarity;
        printf("CUDA QKV epilogue parity T=%d split/direct %.9f\n",
               tokens, similarity);
        free(ids);
    }
    ei_engine_free(&direct_qkv);
    ei_engine_free(&split_qkv);
    if (minimum < 0.999999f) return 1;
    printf("CUDA QKV epilogue parity: %.9f\n", minimum);

    const int32_t projection_shapes[] = {7, 32, 128, 512};
    ei_engine expanded_projection;
    ei_engine native_projection;
    setenv("EI_CUDA_NATIVE_Q4_GEMM", "0", 1);
    ei_engine_load_backend(&expanded_projection, argv[1], "cuda");
    setenv("EI_CUDA_NATIVE_Q4_GEMM", "1", 1);
    ei_engine_load_backend(&native_projection, argv[1], "cuda");
    unsetenv("EI_CUDA_NATIVE_Q4_GEMM");
    minimum = 1.0f;
    for (size_t shape = 0;
         shape < sizeof(projection_shapes) / sizeof(projection_shapes[0]);
         shape++) {
        const int32_t tokens = projection_shapes[shape];
        int32_t *ids = ei_xmalloc(sizeof(*ids) * (size_t)tokens);
        for (int32_t token = 0; token < tokens; token++) {
            ids[token] = 1000 + (token * 7919) % 240000;
        }
        float expanded_output[EI_N_EMBD];
        float native_output[EI_N_EMBD];
        char err[256];
        if (!ei_engine_embed_tokens(&expanded_projection, ids, (size_t)tokens,
                                    expanded_output, err, sizeof err)) {
            ei_die("CUDA expanded projection T=%d failed: %s", tokens, err);
        }
        if (!ei_engine_embed_tokens(&native_projection, ids, (size_t)tokens,
                                    native_output, err, sizeof err)) {
            ei_die("CUDA native Q4 projection T=%d failed: %s", tokens, err);
        }
        const float similarity = cosine(expanded_output, native_output);
        if (similarity < minimum) minimum = similarity;
        printf("CUDA Q4 MMA parity T=%d expanded/native %.9f\n",
               tokens, similarity);
        free(ids);
    }
    ei_engine_free(&native_projection);
    ei_engine_free(&expanded_projection);
    if (minimum < 0.999f) return 1;
    printf("CUDA Q4 MMA parity: %.9f\n", minimum);

    const int32_t latency_shapes[] = {1, 4};
    ei_engine fp32_latency;
    ei_engine q8_latency;
    setenv("EI_CUDA_Q8_LATENCY", "0", 1);
    ei_engine_load_backend(&fp32_latency, argv[1], "cuda");
    setenv("EI_CUDA_Q8_LATENCY", "1", 1);
    ei_engine_load_backend(&q8_latency, argv[1], "cuda");
    unsetenv("EI_CUDA_Q8_LATENCY");
    minimum = 1.0f;
    for (size_t shape = 0;
         shape < sizeof(latency_shapes) / sizeof(latency_shapes[0]); shape++) {
        const int32_t tokens = latency_shapes[shape];
        int32_t *ids = ei_xmalloc(sizeof(*ids) * (size_t)tokens);
        for (int32_t token = 0; token < tokens; token++) {
            ids[token] = 1000 + (token * 7919) % 240000;
        }
        float fp32_output[EI_N_EMBD];
        float q8_output[EI_N_EMBD];
        char err[256];
        if (!ei_engine_embed_tokens(&fp32_latency, ids, (size_t)tokens,
                                    fp32_output, err, sizeof err)) {
            ei_die("CUDA FP32 latency route T=%d failed: %s", tokens, err);
        }
        if (!ei_engine_embed_tokens(&q8_latency, ids, (size_t)tokens,
                                    q8_output, err, sizeof err)) {
            ei_die("CUDA Q8 latency route T=%d failed: %s", tokens, err);
        }
        const float similarity = cosine(fp32_output, q8_output);
        if (similarity < minimum) minimum = similarity;
        printf("CUDA latency parity T=%d fp32/q8 %.9f\n", tokens, similarity);
        free(ids);
    }
    ei_engine_free(&q8_latency);
    ei_engine_free(&fp32_latency);
    if (minimum < 0.99f) return 1;
    printf("CUDA latency parity: %.9f\n", minimum);
    return 0;
}
