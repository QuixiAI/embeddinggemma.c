#define _POSIX_C_SOURCE 200809L

#include "engine.h"

#include <errno.h>
#include <math.h>
#include <time.h>

#define PERF_SCHEMA 1
#define MAX_TOKEN_SHAPES 16

static uint64_t now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        ei_die("clock_gettime failed: %s", strerror(errno));
    }
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int compare_double(const void *a, const void *b) {
    const double x = *(const double *)a;
    const double y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_sorted(const double *values, int count) {
    return (count & 1) ? values[count / 2]
                       : 0.5 * (values[count / 2 - 1] + values[count / 2]);
}

static double cosine(const float *a, const float *b) {
    double dot = 0.0;
    double aa = 0.0;
    double bb = 0.0;
    for (int i = 0; i < EI_N_EMBD; i++) {
        dot += (double)a[i] * (double)b[i];
        aa += (double)a[i] * (double)a[i];
        bb += (double)b[i] * (double)b[i];
    }
    return dot / sqrt(aa * bb);
}

static int parse_token_shapes(const char *value, int32_t out[MAX_TOKEN_SHAPES]) {
    char *copy = strdup(value);
    if (!copy) ei_die("out of memory");
    int count = 0;
    char *save = NULL;
    for (char *item = strtok_r(copy, ",", &save); item; item = strtok_r(NULL, ",", &save)) {
        char *end = NULL;
        long tokens = strtol(item, &end, 10);
        if (*end != '\0' || tokens < 1 || tokens > EI_N_CTX || count == MAX_TOKEN_SHAPES) {
            free(copy);
            ei_die("--tokens must contain 1-%d comma-separated values", EI_N_CTX);
        }
        out[count++] = (int32_t)tokens;
    }
    free(copy);
    if (count == 0) ei_die("--tokens cannot be empty");
    return count;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s --model model.gguf [--backend auto|cpu|metal] "
            "[--tokens 1,8,32,128] [--warmup n] [--iters n] "
            "[--ab-gelu-quant] [--ab-metal-fp16-kv]\n",
            argv0);
}

static double timed_embed(ei_engine *engine, const int32_t *ids, int32_t tokens,
                          float output[EI_N_EMBD]) {
    char err[256];
    uint64_t start = now_ns();
    if (!ei_engine_embed_tokens(engine, ids, (size_t)tokens,
                                output, err, sizeof err)) {
        ei_die("benchmark failed: %s", err);
    }
    return (double)(now_ns() - start) / 1000000.0;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *backend = "auto";
    const char *token_shapes = "1,8,32,128";
    int warmup = 1;
    int iters = 5;
    bool ab_gelu_quant = false;
    bool ab_metal_fp16_kv = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            token_shapes = argv[++i];
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ab-gelu-quant") == 0) {
            ab_gelu_quant = true;
        } else if (strcmp(argv[i], "--ab-metal-fp16-kv") == 0) {
            ab_metal_fp16_kv = true;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!model_path || warmup < 0 || iters < 1) {
        usage(argv[0]);
        return 2;
    }

    int32_t shapes[MAX_TOKEN_SHAPES];
    int shape_count = parse_token_shapes(token_shapes, shapes);
    if (ab_metal_fp16_kv) {
        if (strcmp(backend, "metal") != 0) {
            ei_die("--ab-metal-fp16-kv requires --backend metal");
        }
        ei_engine baseline;
        ei_engine fp16;
        setenv("EI_METAL_FP16_KV_MIN_TOKENS", "65536", 1);
        ei_engine_load_backend(&baseline, model_path, "metal");
        setenv("EI_METAL_FP16_KV_MIN_TOKENS", "1", 1);
        ei_engine_load_backend(&fp16, model_path, "metal");
        unsetenv("EI_METAL_FP16_KV_MIN_TOKENS");

        for (int shape = 0; shape < shape_count; shape++) {
            int32_t tokens = shapes[shape];
            int32_t *ids = ei_xmalloc(sizeof(*ids) * (size_t)tokens);
            for (int32_t i = 0; i < tokens; i++) ids[i] = 1000 + (i * 7919) % 240000;
            float baseline_output[EI_N_EMBD];
            float fp16_output[EI_N_EMBD];
            for (int i = 0; i < warmup; i++) {
                (void)timed_embed(&baseline, ids, tokens, baseline_output);
                (void)timed_embed(&fp16, ids, tokens, fp16_output);
            }
            double *baseline_samples = ei_xmalloc(sizeof(*baseline_samples) * (size_t)iters);
            double *fp16_samples = ei_xmalloc(sizeof(*fp16_samples) * (size_t)iters);
            for (int i = 0; i < iters; i++) {
                if ((i & 1) == 0) {
                    baseline_samples[i] = timed_embed(&baseline, ids, tokens, baseline_output);
                    fp16_samples[i] = timed_embed(&fp16, ids, tokens, fp16_output);
                } else {
                    fp16_samples[i] = timed_embed(&fp16, ids, tokens, fp16_output);
                    baseline_samples[i] = timed_embed(&baseline, ids, tokens, baseline_output);
                }
            }
            double similarity = cosine(baseline_output, fp16_output);
            qsort(baseline_samples, (size_t)iters, sizeof(*baseline_samples), compare_double);
            qsort(fp16_samples, (size_t)iters, sizeof(*fp16_samples), compare_double);
            double baseline_median = median_sorted(baseline_samples, iters);
            double fp16_median = median_sorted(fp16_samples, iters);
            printf("{\"schema\":%d,\"backend\":\"metal\","
                   "\"kernel\":\"engine_embed_tokens\","
                   "\"variant\":\"fp16_kv_ab\",\"shape\":{\"tokens\":%d},"
                   "\"threads\":1,\"fp16_ms\":%.9g,\"baseline_ms\":%.9g,"
                   "\"throughput_speedup\":%.9g,\"cosine\":%.9g}\n",
                   PERF_SCHEMA, tokens, fp16_median, baseline_median,
                   baseline_median / fp16_median, similarity);
            free(fp16_samples);
            free(baseline_samples);
            free(ids);
        }
        ei_engine_free(&fp16);
        ei_engine_free(&baseline);
        return 0;
    }

    ei_engine engine;
    ei_engine_load_backend(&engine, model_path, backend);
    if (ab_gelu_quant && strcmp(ei_engine_backend(&engine), "cpu") != 0) {
        ei_die("--ab-gelu-quant requires the CPU backend");
    }

    for (int shape = 0; shape < shape_count; shape++) {
        int32_t tokens = shapes[shape];
        int32_t *ids = ei_xmalloc(sizeof(*ids) * (size_t)tokens);
        for (int32_t i = 0; i < tokens; i++) ids[i] = 1000 + (i * 7919) % 240000;

        float output[EI_N_EMBD];
        char err[256];
        if (ab_gelu_quant) {
            for (int i = 0; i < warmup; i++) {
                engine.fused_gelu_quant = true;
                (void)timed_embed(&engine, ids, tokens, output);
                engine.fused_gelu_quant = false;
                (void)timed_embed(&engine, ids, tokens, output);
            }
            double *fused = ei_xmalloc(sizeof(*fused) * (size_t)iters);
            double *baseline = ei_xmalloc(sizeof(*baseline) * (size_t)iters);
            double checksum_fused = 0.0;
            double checksum_baseline = 0.0;
            for (int i = 0; i < iters; i++) {
                if ((i & 1) == 0) {
                    engine.fused_gelu_quant = true;
                    fused[i] = timed_embed(&engine, ids, tokens, output);
                    checksum_fused += output[(i * 97) % EI_N_EMBD];
                    engine.fused_gelu_quant = false;
                    baseline[i] = timed_embed(&engine, ids, tokens, output);
                    checksum_baseline += output[(i * 97) % EI_N_EMBD];
                } else {
                    engine.fused_gelu_quant = false;
                    baseline[i] = timed_embed(&engine, ids, tokens, output);
                    checksum_baseline += output[(i * 97) % EI_N_EMBD];
                    engine.fused_gelu_quant = true;
                    fused[i] = timed_embed(&engine, ids, tokens, output);
                    checksum_fused += output[(i * 97) % EI_N_EMBD];
                }
            }
            qsort(fused, (size_t)iters, sizeof(*fused), compare_double);
            qsort(baseline, (size_t)iters, sizeof(*baseline), compare_double);
            double fused_median = fused[iters / 2];
            double baseline_median = baseline[iters / 2];
            printf("{\"schema\":%d,\"backend\":\"cpu\","
                   "\"kernel\":\"engine_embed_tokens\","
                   "\"variant\":\"fused_gelu_quant_ab\","
                   "\"shape\":{\"tokens\":%d},\"threads\":%d,"
                   "\"fused_ms\":%.9g,\"baseline_ms\":%.9g,"
                   "\"throughput_speedup\":%.9g,"
                   "\"checksum_delta\":%.9g}\n",
                   PERF_SCHEMA, tokens, ei_engine_threads(&engine),
                   fused_median, baseline_median,
                   baseline_median / fused_median,
                   checksum_fused - checksum_baseline);
            free(baseline);
            free(fused);
            free(ids);
            continue;
        }

        for (int i = 0; i < warmup; i++) {
            if (!ei_engine_embed_tokens(&engine, ids, (size_t)tokens, output, err, sizeof err)) {
                ei_die("warmup failed: %s", err);
            }
        }

        double *samples = ei_xmalloc(sizeof(*samples) * (size_t)iters);
        double checksum = 0.0;
        for (int i = 0; i < iters; i++) {
            uint64_t start = now_ns();
            if (!ei_engine_embed_tokens(&engine, ids, (size_t)tokens, output, err, sizeof err)) {
                ei_die("benchmark failed: %s", err);
            }
            uint64_t end = now_ns();
            samples[i] = (double)(end - start) / 1000000.0;
            checksum += output[(i * 97) % EI_N_EMBD];
        }
        qsort(samples, (size_t)iters, sizeof(*samples), compare_double);
        double median = median_sorted(samples, iters);
        double p20 = samples[(int)(0.20 * (double)(iters - 1))];
        double p80 = samples[(int)(0.80 * (double)(iters - 1))];
        printf("{\"schema\":%d,\"backend\":\"%s\",\"kernel\":\"engine_embed_tokens\","
               "\"variant\":\"full_graph\",\"shape\":{\"tokens\":%d},\"threads\":%d,"
               "\"target_ms\":%.9g,\"target_p20_ms\":%.9g,\"target_p80_ms\":%.9g,"
               "\"tokens_per_second\":%.9g,\"checksum\":%.9g}\n",
               PERF_SCHEMA, ei_engine_backend(&engine), tokens, ei_engine_threads(&engine),
               median, p20, p80, 1000.0 * (double)tokens / median, checksum);
        free(samples);
        free(ids);
    }

    ei_engine_free(&engine);
    return 0;
}
