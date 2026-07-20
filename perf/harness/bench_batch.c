#define _POSIX_C_SOURCE 200809L

#include "engine.h"

#include <errno.h>
#include <math.h>
#include <time.h>

#define MAX_BATCH_LEVELS 16

static uint64_t now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        ei_die("clock_gettime failed: %s", strerror(errno));
    }
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_sorted(const double *values, int32_t count) {
    return (count & 1) ? values[count / 2]
                       : 0.5 * (values[count / 2 - 1] + values[count / 2]);
}

static int32_t parse_levels(const char *value, int32_t out[MAX_BATCH_LEVELS]) {
    char *copy = strdup(value);
    if (!copy) ei_die("out of memory");
    int32_t count = 0;
    char *save = NULL;
    for (char *item = strtok_r(copy, ",", &save); item;
         item = strtok_r(NULL, ",", &save)) {
        char *end = NULL;
        long level = strtol(item, &end, 10);
        if (*end != '\0' || level < 1 || level > 256 || count == MAX_BATCH_LEVELS) {
            free(copy);
            ei_die("--batch-sizes must contain 1..256 comma-separated values");
        }
        out[count++] = (int32_t)level;
    }
    free(copy);
    return count;
}

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

static double run_packed(ei_engine *engine, const int32_t *ids,
                         const size_t *offsets, int32_t batch_size,
                         float *output) {
    char err[256];
    uint64_t begin = now_ns();
    if (!ei_engine_embed_tokens_batch(engine, ids, offsets, (size_t)batch_size,
                                      output, err, sizeof err)) {
        ei_die("packed batch failed: %s", err);
    }
    return (double)(now_ns() - begin) / 1000000.0;
}

static double run_serial(ei_engine *engine, const int32_t *ids,
                         const size_t *offsets, int32_t batch_size,
                         float *output) {
    char err[256];
    uint64_t begin = now_ns();
    for (int32_t sequence = 0; sequence < batch_size; sequence++) {
        if (!ei_engine_embed_tokens(
                engine, ids + offsets[sequence],
                offsets[sequence + 1] - offsets[sequence],
                output + (size_t)sequence * EI_N_EMBD, err, sizeof err)) {
            ei_die("serialized sequence failed: %s", err);
        }
    }
    return (double)(now_ns() - begin) / 1000000.0;
}

static void bench_level(ei_engine *engine, int32_t tokens, int32_t batch_size,
                        int32_t warmup, int32_t iterations) {
    size_t total = (size_t)tokens * (size_t)batch_size;
    int32_t *ids = ei_xmalloc(total * sizeof(*ids));
    size_t *offsets = ei_xmalloc(((size_t)batch_size + 1) * sizeof(*offsets));
    offsets[0] = 0;
    for (int32_t sequence = 0; sequence < batch_size; sequence++) {
        offsets[sequence + 1] = offsets[sequence] + (size_t)tokens;
        for (int32_t token = 0; token < tokens; token++) {
            ids[offsets[sequence] + (size_t)token] = 1000 +
                (int32_t)(((uint64_t)token * 7919u +
                           (uint64_t)sequence * 104729u) % 240000u);
        }
    }
    float *packed = ei_xmalloc((size_t)batch_size * EI_N_EMBD * sizeof(*packed));
    float *serial = ei_xmalloc((size_t)batch_size * EI_N_EMBD * sizeof(*serial));
    for (int32_t i = 0; i < warmup; i++) {
        (void)run_packed(engine, ids, offsets, batch_size, packed);
        (void)run_serial(engine, ids, offsets, batch_size, serial);
    }

    double *packed_ms = ei_xmalloc((size_t)iterations * sizeof(*packed_ms));
    double *serial_ms = ei_xmalloc((size_t)iterations * sizeof(*serial_ms));
    for (int32_t i = 0; i < iterations; i++) {
        if ((i & 1) == 0) {
            packed_ms[i] = run_packed(engine, ids, offsets, batch_size, packed);
            serial_ms[i] = run_serial(engine, ids, offsets, batch_size, serial);
        } else {
            serial_ms[i] = run_serial(engine, ids, offsets, batch_size, serial);
            packed_ms[i] = run_packed(engine, ids, offsets, batch_size, packed);
        }
    }
    float minimum = 1.0f;
    for (int32_t sequence = 0; sequence < batch_size; sequence++) {
        float similarity = cosine(packed + (size_t)sequence * EI_N_EMBD,
                                  serial + (size_t)sequence * EI_N_EMBD);
        if (!isfinite(similarity)) {
            ei_die("packed/serial parity produced a nonfinite cosine");
        }
        if (similarity < minimum) minimum = similarity;
    }
    float minimum_expected = 0.9999f;
    if (strcmp(ei_engine_backend(engine), "cuda") == 0) {
        minimum_expected = 0.998f;
    } else if (strcmp(ei_engine_backend(engine), "xpu") == 0) {
        minimum_expected = 0.9998f;
    }
    if (minimum < minimum_expected) {
        ei_die("packed/serial parity failed: %.9f", minimum);
    }

    qsort(packed_ms, (size_t)iterations, sizeof(*packed_ms), compare_double);
    qsort(serial_ms, (size_t)iterations, sizeof(*serial_ms), compare_double);
    double packed_median = median_sorted(packed_ms, iterations);
    double serial_median = median_sorted(serial_ms, iterations);
    double requests_per_second = 1000.0 * (double)batch_size / packed_median;
    printf("{\"schema\":1,\"backend\":\"%s\",\"kernel\":\"engine_embed_tokens_batch\","
           "\"variant\":\"packed_vs_serialized\","
           "\"shape\":{\"tokens\":%d,\"batch_size\":%d,\"total_tokens\":%zu},"
           "\"threads\":%d,\"packed_ms\":%.9g,\"serialized_ms\":%.9g,"
           "\"throughput_speedup\":%.9g,\"requests_per_second\":%.9g,"
           "\"tokens_per_second\":%.9g,\"minimum_cosine\":%.9g}\n",
           ei_engine_backend(engine), tokens, batch_size, total,
           ei_engine_threads(engine), packed_median, serial_median,
           serial_median / packed_median, requests_per_second,
           requests_per_second * tokens, minimum);

    free(serial_ms);
    free(packed_ms);
    free(serial);
    free(packed);
    free(offsets);
    free(ids);
}

static void bench_level_fp16_ab(ei_engine *baseline, ei_engine *fp16,
                                int32_t tokens, int32_t batch_size,
                                int32_t warmup, int32_t iterations,
                                const char *backend, const char *variant,
                                const char *candidate_field,
                                float minimum_expected) {
    size_t total = (size_t)tokens * (size_t)batch_size;
    int32_t *ids = ei_xmalloc(total * sizeof(*ids));
    size_t *offsets = ei_xmalloc(((size_t)batch_size + 1) * sizeof(*offsets));
    offsets[0] = 0;
    for (int32_t sequence = 0; sequence < batch_size; sequence++) {
        offsets[sequence + 1] = offsets[sequence] + (size_t)tokens;
        for (int32_t token = 0; token < tokens; token++) {
            ids[offsets[sequence] + (size_t)token] = 1000 +
                (int32_t)(((uint64_t)token * 7919u +
                           (uint64_t)sequence * 104729u) % 240000u);
        }
    }
    size_t output_count = (size_t)batch_size * EI_N_EMBD;
    float *baseline_output = ei_xmalloc(output_count * sizeof(*baseline_output));
    float *fp16_output = ei_xmalloc(output_count * sizeof(*fp16_output));
    for (int32_t i = 0; i < warmup; i++) {
        (void)run_packed(baseline, ids, offsets, batch_size, baseline_output);
        (void)run_packed(fp16, ids, offsets, batch_size, fp16_output);
    }
    double *baseline_ms = ei_xmalloc((size_t)iterations * sizeof(*baseline_ms));
    double *fp16_ms = ei_xmalloc((size_t)iterations * sizeof(*fp16_ms));
    for (int32_t i = 0; i < iterations; i++) {
        if ((i & 1) == 0) {
            baseline_ms[i] = run_packed(baseline, ids, offsets, batch_size, baseline_output);
            fp16_ms[i] = run_packed(fp16, ids, offsets, batch_size, fp16_output);
        } else {
            fp16_ms[i] = run_packed(fp16, ids, offsets, batch_size, fp16_output);
            baseline_ms[i] = run_packed(baseline, ids, offsets, batch_size, baseline_output);
        }
    }
    float minimum = 1.0f;
    for (int32_t sequence = 0; sequence < batch_size; sequence++) {
        float similarity = cosine(baseline_output + (size_t)sequence * EI_N_EMBD,
                                  fp16_output + (size_t)sequence * EI_N_EMBD);
        if (!isfinite(similarity)) {
            ei_die("A/B parity produced a nonfinite cosine");
        }
        if (similarity < minimum) minimum = similarity;
    }
    if (minimum < minimum_expected) {
        ei_die("FP16 attention parity failed: %.9f", minimum);
    }
    qsort(baseline_ms, (size_t)iterations, sizeof(*baseline_ms), compare_double);
    qsort(fp16_ms, (size_t)iterations, sizeof(*fp16_ms), compare_double);
    double baseline_median = median_sorted(baseline_ms, iterations);
    double fp16_median = median_sorted(fp16_ms, iterations);
    printf("{\"schema\":1,\"backend\":\"%s\","
           "\"kernel\":\"engine_embed_tokens_batch\","
           "\"variant\":\"%s\","
           "\"shape\":{\"tokens\":%d,\"batch_size\":%d,\"total_tokens\":%zu},"
           "\"threads\":1,\"%s\":%.9g,\"baseline_ms\":%.9g,"
           "\"throughput_speedup\":%.9g,\"requests_per_second\":%.9g,"
           "\"minimum_cosine\":%.9g}\n",
           backend, variant, tokens, batch_size, total, candidate_field,
           fp16_median, baseline_median,
           baseline_median / fp16_median, 1000.0 * (double)batch_size / fp16_median,
           minimum);
    free(fp16_ms);
    free(baseline_ms);
    free(fp16_output);
    free(baseline_output);
    free(offsets);
    free(ids);
}

int main(int argc, char **argv) {
    const char *model = NULL;
    const char *backend = "cpu";
    const char *batch_value = "1,2,4,8,16,32";
    int32_t tokens = 32;
    int32_t max_total_tokens = 4096;
    int32_t warmup = 2;
    int32_t iterations = 7;
    bool ab_metal_fp16_kv = false;
    bool ab_xpu_fp16_attention = false;
    bool ab_xpu_attention_routing = false;
    bool ab_xpu_v_only = false;
    bool ab_xpu_xe2_flash = false;
    bool ab_xpu_command_graph = false;
    bool ab_xpu_xe2_w4 = false;
    bool ab_xpu_rms_register_cache = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model = argv[++i];
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) backend = argv[++i];
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch-sizes") == 0 && i + 1 < argc) batch_value = argv[++i];
        else if (strcmp(argv[i], "--max-total-tokens") == 0 && i + 1 < argc) max_total_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ab-metal-fp16-kv") == 0) ab_metal_fp16_kv = true;
        else if (strcmp(argv[i], "--ab-xpu-fp16-attention") == 0) ab_xpu_fp16_attention = true;
        else if (strcmp(argv[i], "--ab-xpu-attention-routing") == 0) ab_xpu_attention_routing = true;
        else if (strcmp(argv[i], "--ab-xpu-v-only") == 0) ab_xpu_v_only = true;
        else if (strcmp(argv[i], "--ab-xpu-xe2-flash") == 0) ab_xpu_xe2_flash = true;
        else if (strcmp(argv[i], "--ab-xpu-command-graph") == 0) ab_xpu_command_graph = true;
        else if (strcmp(argv[i], "--ab-xpu-xe2-w4") == 0) ab_xpu_xe2_w4 = true;
        else if (strcmp(argv[i], "--ab-xpu-rms-register-cache") == 0) ab_xpu_rms_register_cache = true;
        else ei_die("usage: %s --model model.gguf [--backend cpu|metal|cuda|xpu] [--tokens N] "
                    "[--batch-sizes 1,2,4] [--max-total-tokens N]", argv[0]);
    }
    if (!model || tokens < 1 || tokens > EI_N_CTX || max_total_tokens < tokens ||
        warmup < 0 || iterations < 1) ei_die("invalid batch benchmark arguments");
    int32_t levels[MAX_BATCH_LEVELS];
    int32_t n_levels = parse_levels(batch_value, levels);
    if (ab_xpu_rms_register_cache) {
        if (strcmp(backend, "xpu") != 0) {
            ei_die("--ab-xpu-rms-register-cache requires --backend xpu");
        }
        ei_engine baseline;
        ei_engine candidate;
        setenv("EI_XPU_RMS_REGISTER_CACHE", "0", 1);
        ei_engine_load_backend(&baseline, model, "xpu");
        setenv("EI_XPU_RMS_REGISTER_CACHE", "1", 1);
        ei_engine_load_backend(&candidate, model, "xpu");
        unsetenv("EI_XPU_RMS_REGISTER_CACHE");
        for (int32_t i = 0; i < n_levels; i++) {
            if ((int64_t)levels[i] * tokens <= max_total_tokens) {
                bench_level_fp16_ab(&baseline, &candidate, tokens, levels[i],
                                    warmup, iterations, "xpu",
                                    "rms_register_cache_ab", "register_ms",
                                    0.999f);
            }
        }
        ei_engine_free(&candidate);
        ei_engine_free(&baseline);
        return 0;
    }
    if (ab_xpu_xe2_w4) {
        if (strcmp(backend, "xpu") != 0) {
            ei_die("--ab-xpu-xe2-w4 requires --backend xpu");
        }
        ei_engine baseline;
        ei_engine candidate;
        setenv("EI_XPU_XE2_W4", "0", 1);
        ei_engine_load_backend(&baseline, model, "xpu");
        setenv("EI_XPU_XE2_W4", "1", 1);
        ei_engine_load_backend(&candidate, model, "xpu");
        unsetenv("EI_XPU_XE2_W4");
        for (int32_t i = 0; i < n_levels; i++) {
            if ((int64_t)levels[i] * tokens <= max_total_tokens) {
                bench_level_fp16_ab(&baseline, &candidate, tokens, levels[i],
                                    warmup, iterations, "xpu",
                                    "xe2_w4_ab", "w4_ms", 0.999f);
            }
        }
        ei_engine_free(&candidate);
        ei_engine_free(&baseline);
        return 0;
    }
    if (ab_xpu_command_graph) {
        if (strcmp(backend, "xpu") != 0) {
            ei_die("--ab-xpu-command-graph requires --backend xpu");
        }
        ei_engine baseline;
        ei_engine candidate;
        setenv("EI_XPU_COMMAND_GRAPH", "0", 1);
        ei_engine_load_backend(&baseline, model, "xpu");
        setenv("EI_XPU_COMMAND_GRAPH", "1", 1);
        ei_engine_load_backend(&candidate, model, "xpu");
        unsetenv("EI_XPU_COMMAND_GRAPH");
        for (int32_t i = 0; i < n_levels; i++) {
            if ((int64_t)levels[i] * tokens <= max_total_tokens) {
                bench_level_fp16_ab(&baseline, &candidate, tokens, levels[i],
                                    warmup, iterations, "xpu",
                                    "command_graph_ab", "graph_ms", 0.999f);
            }
        }
        ei_engine_free(&candidate);
        ei_engine_free(&baseline);
        return 0;
    }
    if (ab_xpu_xe2_flash) {
        if (strcmp(backend, "xpu") != 0) {
            ei_die("--ab-xpu-xe2-flash requires --backend xpu");
        }
        ei_engine baseline;
        ei_engine candidate;
        setenv("EI_XPU_XE2_FLASH", "0", 1);
        ei_engine_load_backend(&baseline, model, "xpu");
        setenv("EI_XPU_XE2_FLASH", "1", 1);
        ei_engine_load_backend(&candidate, model, "xpu");
        unsetenv("EI_XPU_XE2_FLASH");
        for (int32_t i = 0; i < n_levels; i++) {
            if ((int64_t)levels[i] * tokens <= max_total_tokens) {
                bench_level_fp16_ab(&baseline, &candidate, tokens, levels[i],
                                    warmup, iterations, "xpu",
                                    "xe2_flash_ab", "flash_ms", 0.9997f);
            }
        }
        ei_engine_free(&candidate);
        ei_engine_free(&baseline);
        return 0;
    }
    if (ab_xpu_v_only) {
        if (strcmp(backend, "xpu") != 0) {
            ei_die("--ab-xpu-v-only requires --backend xpu");
        }
        ei_engine baseline;
        ei_engine candidate;
        setenv("EI_XPU_SINGLE_TOKEN_V_ONLY", "0", 1);
        ei_engine_load_backend(&baseline, model, "xpu");
        setenv("EI_XPU_SINGLE_TOKEN_V_ONLY", "1", 1);
        ei_engine_load_backend(&candidate, model, "xpu");
        unsetenv("EI_XPU_SINGLE_TOKEN_V_ONLY");
        for (int32_t i = 0; i < n_levels; i++) {
            if ((int64_t)levels[i] * tokens <= max_total_tokens) {
                bench_level_fp16_ab(&baseline, &candidate, tokens, levels[i],
                                    warmup, iterations, "xpu",
                                    "single_token_v_only_ab", "v_only_ms",
                                    0.9999f);
            }
        }
        ei_engine_free(&candidate);
        ei_engine_free(&baseline);
        return 0;
    }
    if (ab_xpu_attention_routing) {
        if (strcmp(backend, "xpu") != 0) {
            ei_die("--ab-xpu-attention-routing requires --backend xpu");
        }
        ei_engine baseline;
        ei_engine candidate;
        setenv("EI_XPU_TENSOR_ATTENTION_MIN_TOKENS", "128", 1);
        ei_engine_load_backend(&baseline, model, "xpu");
        unsetenv("EI_XPU_TENSOR_ATTENTION_MIN_TOKENS");
        ei_engine_load_backend(&candidate, model, "xpu");
        for (int32_t i = 0; i < n_levels; i++) {
            if ((int64_t)levels[i] * tokens <= max_total_tokens) {
                bench_level_fp16_ab(&baseline, &candidate, tokens, levels[i],
                                    warmup, iterations, "xpu",
                                    "batch_aware_attention_ab", "candidate_ms",
                                    0.9997f);
            }
        }
        ei_engine_free(&candidate);
        ei_engine_free(&baseline);
        return 0;
    }
    if (ab_metal_fp16_kv || ab_xpu_fp16_attention) {
        const char *expected_backend =
            ab_metal_fp16_kv ? "metal" : "xpu";
        const char *environment = ab_metal_fp16_kv
            ? "EI_METAL_FP16_KV_MIN_TOKENS" : "EI_XPU_FP16_ATTENTION";
        const char *baseline_value = ab_metal_fp16_kv ? "65536" : "0";
        const char *variant = ab_metal_fp16_kv
            ? "fp16_kv_ab" : "fp16_dense_auto_ab";
        const float minimum_expected =
            ab_metal_fp16_kv ? 0.9999f : 0.9998f;
        if (strcmp(backend, expected_backend) != 0) {
            ei_die("FP16 A/B mode requires --backend %s", expected_backend);
        }
        ei_engine baseline;
        ei_engine fp16;
        setenv(environment, baseline_value, 1);
        ei_engine_load_backend(&baseline, model, expected_backend);
        setenv(environment, ab_metal_fp16_kv ? "1" : "auto", 1);
        ei_engine_load_backend(&fp16, model, expected_backend);
        unsetenv(environment);
        for (int32_t i = 0; i < n_levels; i++) {
            if ((int64_t)levels[i] * tokens <= max_total_tokens) {
                bench_level_fp16_ab(&baseline, &fp16, tokens, levels[i],
                                    warmup, iterations, expected_backend,
                                    variant, "fp16_ms", minimum_expected);
            }
        }
        ei_engine_free(&fp16);
        ei_engine_free(&baseline);
        return 0;
    }
    ei_engine engine;
    ei_engine_load_backend(&engine, model, backend);
    for (int32_t i = 0; i < n_levels; i++) {
        if ((int64_t)levels[i] * tokens <= max_total_tokens) {
            bench_level(&engine, tokens, levels[i], warmup, iterations);
        }
    }
    ei_engine_free(&engine);
    return 0;
}
