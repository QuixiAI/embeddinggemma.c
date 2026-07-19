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
        if (similarity < minimum) minimum = similarity;
    }
    if (minimum < 0.9999f) ei_die("packed/serial parity failed: %.9f", minimum);

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
        if (similarity < minimum) minimum = similarity;
    }
    if (minimum < 0.9999f) ei_die("FP16 K/V parity failed: %.9f", minimum);
    qsort(baseline_ms, (size_t)iterations, sizeof(*baseline_ms), compare_double);
    qsort(fp16_ms, (size_t)iterations, sizeof(*fp16_ms), compare_double);
    double baseline_median = median_sorted(baseline_ms, iterations);
    double fp16_median = median_sorted(fp16_ms, iterations);
    printf("{\"schema\":1,\"backend\":\"metal\","
           "\"kernel\":\"engine_embed_tokens_batch\","
           "\"variant\":\"fp16_kv_ab\","
           "\"shape\":{\"tokens\":%d,\"batch_size\":%d,\"total_tokens\":%zu},"
           "\"threads\":1,\"fp16_ms\":%.9g,\"baseline_ms\":%.9g,"
           "\"throughput_speedup\":%.9g,\"requests_per_second\":%.9g,"
           "\"minimum_cosine\":%.9g}\n",
           tokens, batch_size, total, fp16_median, baseline_median,
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
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model = argv[++i];
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) backend = argv[++i];
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch-sizes") == 0 && i + 1 < argc) batch_value = argv[++i];
        else if (strcmp(argv[i], "--max-total-tokens") == 0 && i + 1 < argc) max_total_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ab-metal-fp16-kv") == 0) ab_metal_fp16_kv = true;
        else ei_die("usage: %s --model model.gguf [--backend cpu|metal] [--tokens N] "
                    "[--batch-sizes 1,2,4] [--max-total-tokens N]", argv[0]);
    }
    if (!model || tokens < 1 || tokens > EI_N_CTX || max_total_tokens < tokens ||
        warmup < 0 || iterations < 1) ei_die("invalid batch benchmark arguments");
    int32_t levels[MAX_BATCH_LEVELS];
    int32_t n_levels = parse_levels(batch_value, levels);
    if (ab_metal_fp16_kv) {
        if (strcmp(backend, "metal") != 0) {
            ei_die("--ab-metal-fp16-kv requires --backend metal");
        }
        ei_engine baseline;
        ei_engine fp16;
        setenv("EI_METAL_FP16_KV_MIN_TOKENS", "65536", 1);
        ei_engine_load_backend(&baseline, model, "metal");
        setenv("EI_METAL_FP16_KV_MIN_TOKENS", "1", 1);
        ei_engine_load_backend(&fp16, model, "metal");
        unsetenv("EI_METAL_FP16_KV_MIN_TOKENS");
        for (int32_t i = 0; i < n_levels; i++) {
            if ((int64_t)levels[i] * tokens <= max_total_tokens) {
                bench_level_fp16_ab(&baseline, &fp16, tokens, levels[i],
                                    warmup, iterations);
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
