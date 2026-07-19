#define _POSIX_C_SOURCE 200809L

#include "inference_service.h"

#include <errno.h>
#include <time.h>

#define MAX_WORKER_LEVELS 16

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

static int parse_workers(const char *value, size_t out[MAX_WORKER_LEVELS]) {
    char *copy = strdup(value);
    if (!copy) ei_die("out of memory");
    int count = 0;
    char *save = NULL;
    for (char *item = strtok_r(copy, ",", &save); item;
         item = strtok_r(NULL, ",", &save)) {
        char *end = NULL;
        long workers = strtol(item, &end, 10);
        if (*end != '\0' || workers < 0 || workers > 64 ||
            count == MAX_WORKER_LEVELS) {
            free(copy);
            ei_die("--workers must contain 0..64 comma-separated values");
        }
        out[count++] = (size_t)workers;
    }
    free(copy);
    return count;
}

static bool fake_execute(void *opaque, const int32_t *ids,
                         const size_t *offsets, size_t batch_size,
                         float *out, char *err, size_t err_len) {
    (void)opaque;
    for (size_t sequence = 0; sequence < batch_size; sequence++) {
        float value = (float)(ids[offsets[sequence]] % 97) * 0.001f;
        for (int32_t d = 0; d < EI_N_EMBD; d++) {
            out[sequence * EI_N_EMBD + d] = value + (float)d * 0.0001f;
        }
    }
    if (err && err_len) err[0] = '\0';
    return true;
}

static char *make_text(size_t chars, int iteration, int sequence) {
    char *text = ei_xmalloc(chars + 1);
    static const char pattern[] = "search_query: efficient embedding inference ";
    size_t pattern_len = sizeof(pattern) - 1;
    for (size_t i = 0; i < chars; i++) text[i] = pattern[i % pattern_len];
    char suffix[48];
    int n = snprintf(suffix, sizeof suffix, " iteration-%d-sequence-%d",
                     iteration, sequence);
    size_t suffix_len = n > 0 ? (size_t)n : 0;
    if (suffix_len > chars) suffix_len = chars;
    memcpy(text + chars - suffix_len, suffix, suffix_len);
    text[chars] = '\0';
    return text;
}

static void bench_workers(const ei_tokenizer *tokenizer, size_t workers,
                          size_t chars, size_t batch_size,
                          int warmup, int iterations) {
    size_t total_runs = (size_t)warmup + (size_t)iterations;
    char **texts = ei_xmalloc(total_runs * batch_size * sizeof(*texts));
    size_t *lengths = ei_xmalloc(batch_size * sizeof(*lengths));
    const char **run_texts = ei_xmalloc(batch_size * sizeof(*run_texts));
    for (size_t i = 0; i < batch_size; i++) lengths[i] = chars;
    for (size_t run = 0; run < total_runs; run++) {
        for (size_t sequence = 0; sequence < batch_size; sequence++) {
            texts[run * batch_size + sequence] =
                make_text(chars, (int)run, (int)sequence);
        }
    }

    ei_inference_service_config config = {
        .cache_entries = 0,
        .max_batch_tokens = 65536,
        .max_batch_requests = 256,
        .max_batch_sequence_tokens = EI_N_CTX,
        .tokenizer_workers = workers,
        .batch_wait_us = 0,
    };
    ei_inference_service *service = ei_inference_service_create(
        tokenizer, fake_execute, NULL, &config);
    if (!service) ei_die("failed to create tokenization benchmark service");
    float *output = ei_xmalloc(batch_size * EI_N_EMBD * sizeof(*output));
    double *samples = ei_xmalloc((size_t)iterations * sizeof(*samples));
    char err[256];
    for (size_t run = 0; run < total_runs; run++) {
        for (size_t sequence = 0; sequence < batch_size; sequence++) {
            run_texts[sequence] = texts[run * batch_size + sequence];
        }
        uint64_t begin = now_ns();
        if (!ei_inference_service_embed_batch(
                service, run_texts, lengths, batch_size,
                output, err, sizeof err)) {
            ei_die("tokenization benchmark failed: %s", err);
        }
        double elapsed = (double)(now_ns() - begin) / 1000000.0;
        if (run >= (size_t)warmup) samples[run - (size_t)warmup] = elapsed;
    }
    qsort(samples, (size_t)iterations, sizeof(*samples), compare_double);
    double median = samples[iterations / 2];
    printf("{\"schema\":1,\"variant\":\"tokenizer_workers\","
           "\"shape\":{\"characters\":%zu,\"batch_size\":%zu},"
           "\"workers\":%zu,\"median_ms\":%.9g,"
           "\"inputs_per_second\":%.9g}\n",
           chars, batch_size, workers, median,
           1000.0 * (double)batch_size / median);

    free(samples);
    free(output);
    ei_inference_service_free(service);
    for (size_t i = 0; i < total_runs * batch_size; i++) free(texts[i]);
    free(run_texts);
    free(lengths);
    free(texts);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *worker_values = "0,2,4,8";
    size_t chars = 256;
    size_t batch_size = 32;
    int warmup = 3;
    int iterations = 15;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
            worker_values = argv[++i];
        } else if (strcmp(argv[i], "--chars") == 0 && i + 1 < argc) {
            chars = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            batch_size = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else {
            ei_die("usage: %s --model model.gguf [--workers 0,2,4,8] "
                   "[--chars N] [--batch-size N] [--warmup N] [--iters N]",
                   argv[0]);
        }
    }
    if (!model_path || chars < 32 || chars > 65536 || batch_size < 1 ||
        batch_size > 256 || warmup < 0 || iterations < 1) {
        ei_die("invalid tokenization benchmark arguments");
    }

    ei_model model;
    ei_model_load(&model, model_path);
    ei_tokenizer tokenizer;
    ei_tokenizer_init(&tokenizer, &model);
    size_t workers[MAX_WORKER_LEVELS];
    int worker_count = parse_workers(worker_values, workers);
    for (int i = 0; i < worker_count; i++) {
        bench_workers(&tokenizer, workers[i], chars, batch_size,
                      warmup, iterations);
    }
    ei_tokenizer_free(&tokenizer);
    ei_model_free(&model);
    return 0;
}
