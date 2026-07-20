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
            ei_die("ROCm %s first route T=%d failed: %s", label, tokens, err);
        }
        if (!ei_engine_embed_tokens(second, ids, (size_t)tokens,
                                    second_output, err, sizeof err)) {
            ei_die("ROCm %s second route T=%d failed: %s", label, tokens, err);
        }
        const float similarity = cosine(first_output, second_output);
        if (!isfinite(similarity)) {
            ei_die("ROCm %s T=%d produced a nonfinite cosine", label, tokens);
        }
        if (similarity < minimum) minimum = similarity;
        printf("ROCm %s T=%d cosine=%.9f\n", label, tokens, similarity);
        free(ids);
    }
    printf("ROCm %s minimum cosine=%.9f\n", label, minimum);
    return minimum;
}

static float compare_singleton_batches(ei_engine *first, ei_engine *second,
                                       const int32_t *batch_sizes,
                                       size_t batch_count, const char *label) {
    float minimum = 1.0f;
    for (size_t shape = 0; shape < batch_count; shape++) {
        const int32_t batch_size = batch_sizes[shape];
        int32_t *ids = ei_xmalloc(sizeof(*ids) * (size_t)batch_size);
        size_t *offsets = ei_xmalloc(sizeof(*offsets) *
                                     ((size_t)batch_size + 1));
        float *first_output = ei_xmalloc(sizeof(*first_output) *
            (size_t)batch_size * EI_N_EMBD);
        float *second_output = ei_xmalloc(sizeof(*second_output) *
            (size_t)batch_size * EI_N_EMBD);
        for (int32_t sequence = 0; sequence < batch_size; sequence++) {
            ids[sequence] = 1000 + (sequence * 7919) % 240000;
            offsets[sequence] = (size_t)sequence;
        }
        offsets[batch_size] = (size_t)batch_size;
        char err[256];
        if (!ei_engine_embed_tokens_batch(first, ids, offsets,
                                          (size_t)batch_size, first_output,
                                          err, sizeof err)) {
            ei_die("ROCm %s first route B=%d failed: %s",
                   label, batch_size, err);
        }
        if (!ei_engine_embed_tokens_batch(second, ids, offsets,
                                          (size_t)batch_size, second_output,
                                          err, sizeof err)) {
            ei_die("ROCm %s second route B=%d failed: %s",
                   label, batch_size, err);
        }
        for (int32_t sequence = 0; sequence < batch_size; sequence++) {
            const float similarity = cosine(
                first_output + (size_t)sequence * EI_N_EMBD,
                second_output + (size_t)sequence * EI_N_EMBD);
            if (!isfinite(similarity)) {
                ei_die("ROCm %s B=%d produced a nonfinite cosine",
                       label, batch_size);
            }
            if (similarity < minimum) minimum = similarity;
        }
        printf("ROCm %s B=%d minimum cosine=%.9f\n",
               label, batch_size, minimum);
        free(second_output);
        free(first_output);
        free(offsets);
        free(ids);
    }
    printf("ROCm %s minimum cosine=%.9f\n", label, minimum);
    return minimum;
}

int main(int argc, char **argv) {
    if (argc != 2) ei_die("usage: %s <model.gguf>", argv[0]);
    const int32_t shapes[] = {1, 7, 32, 128};
    ei_engine cpu;
    ei_engine rocm;
    ei_engine_load_backend(&cpu, argv[1], "cpu");
    ei_engine_load_backend(&rocm, argv[1], "rocm");

    float minimum = 1.0f;
    for (size_t shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]); shape++) {
        const int32_t tokens = shapes[shape];
        int32_t *ids = ei_xmalloc(sizeof(*ids) * (size_t)tokens);
        fill_ids(ids, tokens);
        float cpu_output[EI_N_EMBD];
        float rocm_output[EI_N_EMBD];
        char err[256];
        if (!ei_engine_embed_tokens(&cpu, ids, (size_t)tokens,
                                    cpu_output, err, sizeof err)) {
            ei_die("CPU T=%d failed: %s", tokens, err);
        }
        if (!ei_engine_embed_tokens(&rocm, ids, (size_t)tokens,
                                    rocm_output, err, sizeof err)) {
            ei_die("ROCm T=%d failed: %s", tokens, err);
        }
        const float similarity = cosine(cpu_output, rocm_output);
        if (!isfinite(similarity)) {
            ei_die("CPU/ROCm T=%d produced a nonfinite cosine", tokens);
        }
        if (similarity < minimum) minimum = similarity;
        printf("synthetic backend drift T=%d cpu/rocm %.9f\n",
               tokens, similarity);
        free(ids);
    }
    printf("synthetic backend drift: cpu/rocm %.9f\n", minimum);
    ei_engine_free(&rocm);
    ei_engine_free(&cpu);
    if (minimum < 0.99f) return 1;

    const int32_t projection_shapes[] = {1, 7, 32, 128};
    ei_engine packed_projection;
    ei_engine expanded_projection;
    setenv("EI_ROCM_NATIVE_Q4_GEMM", "0", 1);
    setenv("EI_ROCM_GEMM_MIN_TOKENS", "65536", 1);
    ei_engine_load_backend(&packed_projection, argv[1], "rocm");
    setenv("EI_ROCM_GEMM_MIN_TOKENS", "1", 1);
    ei_engine_load_backend(&expanded_projection, argv[1], "rocm");
    unsetenv("EI_ROCM_GEMM_MIN_TOKENS");
    unsetenv("EI_ROCM_NATIVE_Q4_GEMM");
    minimum = compare_routes(
        &packed_projection, &expanded_projection, projection_shapes,
        sizeof(projection_shapes) / sizeof(projection_shapes[0]),
        "packed-Q4/FP16-hipBLAS projection parity");
    ei_engine_free(&expanded_projection);
    ei_engine_free(&packed_projection);
    if (minimum < 0.99f) return 1;

    const int32_t native_shapes[] = {128, 192, 256};
    ei_engine hipblas_projection;
    ei_engine native_projection;
    setenv("EI_ROCM_NATIVE_Q4_GEMM", "0", 1);
    ei_engine_load_backend(&hipblas_projection, argv[1], "rocm");
    setenv("EI_ROCM_NATIVE_Q4_GEMM", "1", 1);
    setenv("EI_ROCM_NATIVE_Q4_MAX_TOKENS", "65536", 1);
    ei_engine_load_backend(&native_projection, argv[1], "rocm");
    unsetenv("EI_ROCM_NATIVE_Q4_MAX_TOKENS");
    unsetenv("EI_ROCM_NATIVE_Q4_GEMM");
    minimum = compare_routes(
        &hipblas_projection, &native_projection, native_shapes,
        sizeof(native_shapes) / sizeof(native_shapes[0]),
        "hipBLAS/native-Q4-MFMA projection parity");
    ei_engine_free(&native_projection);
    ei_engine_free(&hipblas_projection);
    if (minimum < 0.999f) return 1;

    const int32_t attention_shapes[] = {128, 512, 2048};
    ei_engine online_attention;
    ei_engine tensor_attention;
    setenv("EI_ROCM_TENSOR_ATTENTION_MIN_TOKENS", "65536", 1);
    ei_engine_load_backend(&online_attention, argv[1], "rocm");
    setenv("EI_ROCM_TENSOR_ATTENTION_MIN_TOKENS", "1", 1);
    ei_engine_load_backend(&tensor_attention, argv[1], "rocm");
    unsetenv("EI_ROCM_TENSOR_ATTENTION_MIN_TOKENS");
    minimum = compare_routes(
        &online_attention, &tensor_attention, attention_shapes,
        sizeof(attention_shapes) / sizeof(attention_shapes[0]),
        "online/hipBLAS attention parity");
    ei_engine_free(&tensor_attention);
    ei_engine_free(&online_attention);
    if (minimum < 0.999f) return 1;

    const int32_t batched_attention_shapes[] = {96, 512, 2048};
    ei_engine per_head_attention;
    ei_engine batched_attention;
    setenv("EI_ROCM_TENSOR_ATTENTION_MIN_TOKENS", "1", 1);
    setenv("EI_ROCM_BATCHED_TENSOR_ATTENTION", "0", 1);
    ei_engine_load_backend(&per_head_attention, argv[1], "rocm");
    setenv("EI_ROCM_BATCHED_TENSOR_ATTENTION", "1", 1);
    ei_engine_load_backend(&batched_attention, argv[1], "rocm");
    unsetenv("EI_ROCM_BATCHED_TENSOR_ATTENTION");
    unsetenv("EI_ROCM_TENSOR_ATTENTION_MIN_TOKENS");
    minimum = compare_routes(
        &per_head_attention, &batched_attention, batched_attention_shapes,
        sizeof(batched_attention_shapes) / sizeof(batched_attention_shapes[0]),
        "per-head/batched-head hipBLAS attention parity");
    ei_engine_free(&batched_attention);
    ei_engine_free(&per_head_attention);
    if (minimum < 0.99999f) return 1;

    const int32_t fp16_score_shapes[] = {96, 512, 896};
    ei_engine fp32_scores;
    ei_engine fp16_scores;
    setenv("EI_ROCM_TENSOR_ATTENTION_MIN_TOKENS", "1", 1);
    setenv("EI_ROCM_FP16_ATTENTION_SCORES", "0", 1);
    ei_engine_load_backend(&fp32_scores, argv[1], "rocm");
    setenv("EI_ROCM_FP16_ATTENTION_SCORES", "1", 1);
    ei_engine_load_backend(&fp16_scores, argv[1], "rocm");
    unsetenv("EI_ROCM_FP16_ATTENTION_SCORES");
    unsetenv("EI_ROCM_TENSOR_ATTENTION_MIN_TOKENS");
    minimum = compare_routes(
        &fp32_scores, &fp16_scores, fp16_score_shapes,
        sizeof(fp16_score_shapes) / sizeof(fp16_score_shapes[0]),
        "FP32/FP16 hipBLAS attention-score parity");
    ei_engine_free(&fp16_scores);
    ei_engine_free(&fp32_scores);
    if (minimum < 0.9999f) return 1;

    const int32_t fused_activation_shapes[] = {320, 352, 368};
    ei_engine split_activation;
    ei_engine fused_activation;
    setenv("EI_ROCM_NATIVE_Q4_GEMM", "1", 1);
    setenv("EI_ROCM_NATIVE_Q4_MAX_TOKENS", "65536", 1);
    setenv("EI_ROCM_NATIVE_Q4_FUSED_ACTIVATION", "0", 1);
    ei_engine_load_backend(&split_activation, argv[1], "rocm");
    setenv("EI_ROCM_NATIVE_Q4_FUSED_ACTIVATION", "1", 1);
    ei_engine_load_backend(&fused_activation, argv[1], "rocm");
    unsetenv("EI_ROCM_NATIVE_Q4_FUSED_ACTIVATION");
    unsetenv("EI_ROCM_NATIVE_Q4_MAX_TOKENS");
    unsetenv("EI_ROCM_NATIVE_Q4_GEMM");
    minimum = compare_routes(
        &split_activation, &fused_activation, fused_activation_shapes,
        sizeof(fused_activation_shapes) / sizeof(fused_activation_shapes[0]),
        "split/fused native-Q4 FFN-activation parity");
    ei_engine_free(&fused_activation);
    ei_engine_free(&split_activation);
    if (minimum < 0.99999f) return 1;

    const int32_t mfma_attention_shapes[] = {16, 33, 128, 191};
    ei_engine scalar_flash_attention;
    ei_engine mfma_attention;
    setenv("EI_ROCM_MFMA_ATTENTION", "0", 1);
    ei_engine_load_backend(&scalar_flash_attention, argv[1], "rocm");
    setenv("EI_ROCM_MFMA_ATTENTION", "1", 1);
    ei_engine_load_backend(&mfma_attention, argv[1], "rocm");
    unsetenv("EI_ROCM_MFMA_ATTENTION");
    minimum = compare_routes(
        &scalar_flash_attention, &mfma_attention, mfma_attention_shapes,
        sizeof(mfma_attention_shapes) / sizeof(mfma_attention_shapes[0]),
        "scalar/MFMA attention parity");
    ei_engine_free(&mfma_attention);
    ei_engine_free(&scalar_flash_attention);
    if (minimum < 0.9999f) return 1;

    const int32_t qkv_shapes[] = {7, 128, 2048};
    ei_engine split_qkv;
    ei_engine direct_qkv;
    setenv("EI_ROCM_DIRECT_FP16_QKV", "0", 1);
    ei_engine_load_backend(&split_qkv, argv[1], "rocm");
    setenv("EI_ROCM_DIRECT_FP16_QKV", "1", 1);
    ei_engine_load_backend(&direct_qkv, argv[1], "rocm");
    unsetenv("EI_ROCM_DIRECT_FP16_QKV");
    minimum = compare_routes(
        &split_qkv, &direct_qkv, qkv_shapes,
        sizeof(qkv_shapes) / sizeof(qkv_shapes[0]),
        "split/direct-FP16 QKV parity");
    ei_engine_free(&direct_qkv);
    ei_engine_free(&split_qkv);
    if (minimum < 0.99999f) return 1;

    const int32_t v_only_shapes[] = {1};
    ei_engine full_singleton_attention;
    ei_engine v_only_attention;
    setenv("EI_ROCM_SINGLE_TOKEN_V_ONLY", "0", 1);
    ei_engine_load_backend(&full_singleton_attention, argv[1], "rocm");
    setenv("EI_ROCM_SINGLE_TOKEN_V_ONLY", "1", 1);
    ei_engine_load_backend(&v_only_attention, argv[1], "rocm");
    unsetenv("EI_ROCM_SINGLE_TOKEN_V_ONLY");
    minimum = compare_routes(
        &full_singleton_attention, &v_only_attention, v_only_shapes,
        sizeof(v_only_shapes) / sizeof(v_only_shapes[0]),
        "full/singleton-V-only attention parity");
    ei_engine_free(&v_only_attention);
    ei_engine_free(&full_singleton_attention);
    if (minimum < 0.99999f) return 1;

    const int32_t direct_rms_shapes[] = {1, 7, 16, 31};
    ei_engine split_direct_rms;
    ei_engine fused_direct_rms;
    setenv("EI_ROCM_DIRECT_RMS_FUSION", "0", 1);
    ei_engine_load_backend(&split_direct_rms, argv[1], "rocm");
    setenv("EI_ROCM_DIRECT_RMS_FUSION", "1", 1);
    ei_engine_load_backend(&fused_direct_rms, argv[1], "rocm");
    unsetenv("EI_ROCM_DIRECT_RMS_FUSION");
    minimum = compare_routes(
        &split_direct_rms, &fused_direct_rms, direct_rms_shapes,
        sizeof(direct_rms_shapes) / sizeof(direct_rms_shapes[0]),
        "split/fused direct RMS parity");
    ei_engine_free(&fused_direct_rms);
    ei_engine_free(&split_direct_rms);
    if (minimum < 0.9999f) return 1;

    const int32_t pair_shapes[] = {16, 24, 31};
    ei_engine one_row_q4;
    ei_engine paired_q4;
    setenv("EI_ROCM_DIRECT_Q4_PAIR", "0", 1);
    ei_engine_load_backend(&one_row_q4, argv[1], "rocm");
    setenv("EI_ROCM_DIRECT_Q4_PAIR", "1", 1);
    ei_engine_load_backend(&paired_q4, argv[1], "rocm");
    unsetenv("EI_ROCM_DIRECT_Q4_PAIR");
    minimum = compare_routes(
        &one_row_q4, &paired_q4, pair_shapes,
        sizeof(pair_shapes) / sizeof(pair_shapes[0]),
        "one-row/paired direct Q4 parity");
    ei_engine_free(&paired_q4);
    ei_engine_free(&one_row_q4);
    if (minimum < 0.99999f) return 1;

    const int32_t embedding_rms_shapes[] = {1, 16, 31};
    ei_engine split_embedding_rms;
    ei_engine fused_embedding_rms;
    setenv("EI_ROCM_FUSED_EMBEDDING_RMS", "0", 1);
    ei_engine_load_backend(&split_embedding_rms, argv[1], "rocm");
    setenv("EI_ROCM_FUSED_EMBEDDING_RMS", "1", 1);
    ei_engine_load_backend(&fused_embedding_rms, argv[1], "rocm");
    unsetenv("EI_ROCM_FUSED_EMBEDDING_RMS");
    minimum = compare_routes(
        &split_embedding_rms, &fused_embedding_rms, embedding_rms_shapes,
        sizeof(embedding_rms_shapes) / sizeof(embedding_rms_shapes[0]),
        "split/fused embedding RMS parity");
    ei_engine_free(&fused_embedding_rms);
    ei_engine_free(&split_embedding_rms);
    if (minimum < 0.9999f) return 1;

    ei_engine split_singleton_pool;
    ei_engine fused_singleton_pool;
    setenv("EI_ROCM_FINAL_SINGLETON_POOL", "0", 1);
    ei_engine_load_backend(&split_singleton_pool, argv[1], "rocm");
    setenv("EI_ROCM_FINAL_SINGLETON_POOL", "1", 1);
    ei_engine_load_backend(&fused_singleton_pool, argv[1], "rocm");
    unsetenv("EI_ROCM_FINAL_SINGLETON_POOL");
    minimum = compare_routes(
        &split_singleton_pool, &fused_singleton_pool, v_only_shapes,
        sizeof(v_only_shapes) / sizeof(v_only_shapes[0]),
        "split/fused singleton final-pool parity");
    ei_engine_free(&fused_singleton_pool);
    ei_engine_free(&split_singleton_pool);
    if (minimum < 0.99999f) return 1;

    const int32_t singleton_batches[] = {1, 16, 32, 64, 72, 96};
    ei_engine native_singletons;
    ei_engine routed_singletons;
    setenv("EI_ROCM_SINGLETON_DIRECT_MAX_TOKENS", "0", 1);
    ei_engine_load_backend(&native_singletons, argv[1], "rocm");
    setenv("EI_ROCM_SINGLETON_DIRECT_MAX_TOKENS", "72", 1);
    ei_engine_load_backend(&routed_singletons, argv[1], "rocm");
    unsetenv("EI_ROCM_SINGLETON_DIRECT_MAX_TOKENS");
    minimum = compare_singleton_batches(
        &native_singletons, &routed_singletons, singleton_batches,
        sizeof(singleton_batches) / sizeof(singleton_batches[0]),
        "native/singleton-direct routing parity");
    ei_engine_free(&routed_singletons);
    ei_engine_free(&native_singletons);
    return minimum < 0.9999f;
}
