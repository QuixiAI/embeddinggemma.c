#define _POSIX_C_SOURCE 200809L

#include "inference_service.h"
#include "engine.h"

#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

#define PERF_SCHEMA 1
#define MAX_CONCURRENCY_LEVELS 16
#define MAX_TOKEN_PATTERNS 16

typedef struct {
    ei_engine *engine;
    ei_inference_service *service;
    const int32_t *ids;
    const size_t *offsets;
    const int32_t *token_counts;
    int32_t request_count;

    pthread_mutex_t start_mutex;
    pthread_cond_t ready_cond;
    pthread_cond_t start_cond;
    int32_t ready;
    bool start;

    _Atomic int32_t next_request;
    _Atomic bool failed;
    char error[256];
    double *latency_ms;
    double *checksums;
} concurrency_run;

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

static int32_t parse_concurrency(const char *value,
                                 int32_t out[MAX_CONCURRENCY_LEVELS]) {
    char *copy = strdup(value);
    if (!copy) ei_die("out of memory");
    int32_t count = 0;
    char *save = NULL;
    for (char *item = strtok_r(copy, ",", &save); item;
         item = strtok_r(NULL, ",", &save)) {
        char *end = NULL;
        long concurrency = strtol(item, &end, 10);
        if (*end != '\0' || concurrency < 1 || concurrency > 64 ||
            count == MAX_CONCURRENCY_LEVELS) {
            free(copy);
            ei_die("--concurrency must contain 1-64 comma-separated values");
        }
        out[count++] = (int32_t)concurrency;
    }
    free(copy);
    if (count == 0) ei_die("--concurrency cannot be empty");
    return count;
}

static void *request_worker(void *opaque) {
    concurrency_run *run = opaque;
    pthread_mutex_lock(&run->start_mutex);
    run->ready++;
    pthread_cond_signal(&run->ready_cond);
    while (!run->start) pthread_cond_wait(&run->start_cond, &run->start_mutex);
    pthread_mutex_unlock(&run->start_mutex);

    for (;;) {
        int32_t request = atomic_fetch_add_explicit(
            &run->next_request, 1, memory_order_relaxed);
        if (request >= run->request_count) break;

        float output[EI_N_EMBD];
        char err[256];
        uint64_t begin = now_ns();
        bool ok = ei_inference_service_embed_tokens(
            run->service, run->ids + run->offsets[request],
            (size_t)run->token_counts[request], output, err, sizeof err);
        uint64_t end = now_ns();

        run->latency_ms[request] = (double)(end - begin) / 1000000.0;
        if (!ok) {
            bool expected = false;
            if (atomic_compare_exchange_strong(&run->failed, &expected, true)) {
                snprintf(run->error, sizeof run->error, "%s", err);
            }
            run->checksums[request] = 0.0;
        } else {
            run->checksums[request] = output[(request * 97) % EI_N_EMBD];
        }
    }
    return NULL;
}

static void run_level(ei_engine *engine, ei_inference_service *service,
                      const int32_t *token_pattern, int32_t pattern_count,
                      const char *pattern_label,
                      int32_t concurrency, int32_t min_requests) {
    int32_t request_count = concurrency > min_requests ? concurrency : min_requests;
    int32_t *token_counts = ei_xmalloc(
        sizeof(*token_counts) * (size_t)request_count);
    size_t *offsets = ei_xmalloc(sizeof(*offsets) * ((size_t)request_count + 1u));
    offsets[0] = 0;
    for (int32_t request = 0; request < request_count; request++) {
        token_counts[request] = token_pattern[request % pattern_count];
        offsets[request + 1] = offsets[request] + (size_t)token_counts[request];
    }
    int32_t *ids = ei_xmalloc(sizeof(*ids) * offsets[request_count]);
    for (int32_t request = 0; request < request_count; request++) {
        for (int32_t token = 0; token < token_counts[request]; token++) {
            ids[offsets[request] + (size_t)token] = 1000 +
                (int32_t)(((uint64_t)token * 7919u +
                           (uint64_t)request * 104729u) % 240000u);
        }
    }
    concurrency_run run = {
        .engine = engine,
        .service = service,
        .ids = ids,
        .offsets = offsets,
        .token_counts = token_counts,
        .request_count = request_count,
    };
    run.latency_ms = ei_xmalloc(sizeof(*run.latency_ms) * (size_t)request_count);
    run.checksums = ei_xmalloc(sizeof(*run.checksums) * (size_t)request_count);
    if (pthread_mutex_init(&run.start_mutex, NULL) != 0 ||
        pthread_cond_init(&run.ready_cond, NULL) != 0 ||
        pthread_cond_init(&run.start_cond, NULL) != 0) {
        ei_die("failed to initialize concurrency benchmark synchronization");
    }

    ei_inference_service_stats stats_before;
    ei_inference_service_get_stats(service, &stats_before);

    pthread_t *workers = ei_xmalloc(sizeof(*workers) * (size_t)concurrency);
    for (int32_t i = 0; i < concurrency; i++) {
        if (pthread_create(&workers[i], NULL, request_worker, &run) != 0) {
            ei_die("failed to create request worker %d", i);
        }
    }

    pthread_mutex_lock(&run.start_mutex);
    while (run.ready != concurrency) {
        pthread_cond_wait(&run.ready_cond, &run.start_mutex);
    }
    uint64_t begin = now_ns();
    run.start = true;
    pthread_cond_broadcast(&run.start_cond);
    pthread_mutex_unlock(&run.start_mutex);

    for (int32_t i = 0; i < concurrency; i++) pthread_join(workers[i], NULL);
    uint64_t end = now_ns();
    if (atomic_load(&run.failed)) ei_die("concurrent request failed: %s", run.error);
    ei_inference_service_stats stats_after;
    ei_inference_service_get_stats(service, &stats_after);
    uint64_t batches = stats_after.batches - stats_before.batches;

    double checksum = 0.0;
    for (int32_t i = 0; i < request_count; i++) checksum += run.checksums[i];
    qsort(run.latency_ms, (size_t)request_count, sizeof(*run.latency_ms), compare_double);
    double wall_ms = (double)(end - begin) / 1000000.0;
    double requests_per_second = 1000.0 * (double)request_count / wall_ms;
    size_t p50_index = ((size_t)request_count * 50u + 99u) / 100u - 1u;
    size_t p95_index = ((size_t)request_count * 95u + 99u) / 100u - 1u;

    printf("{\"schema\":%d,\"backend\":\"%s\","
           "\"kernel\":\"engine_embed_tokens\","
           "\"variant\":\"dynamic_token_batch\","
           "\"shape\":{\"tokens\":%d,\"token_pattern\":\"%s\",\"concurrency\":%d},"
           "\"threads\":%d,\"requests\":%d,\"wall_ms\":%.9g,"
           "\"latency_p50_ms\":%.9g,\"latency_p95_ms\":%.9g,"
           "\"requests_per_second\":%.9g,\"tokens_per_second\":%.9g,"
           "\"batches\":%llu,\"average_batch_size\":%.9g,"
           "\"checksum\":%.9g}\n",
           PERF_SCHEMA, ei_engine_backend(engine),
           pattern_count == 1 ? token_pattern[0] : -1,
           pattern_label, concurrency,
           ei_engine_threads(engine), request_count, wall_ms,
           run.latency_ms[p50_index], run.latency_ms[p95_index],
           requests_per_second,
           1000.0 * (double)offsets[request_count] / wall_ms,
           (unsigned long long)batches,
           batches ? (double)request_count / (double)batches : 0.0,
           checksum);

    free(workers);
    pthread_cond_destroy(&run.start_cond);
    pthread_cond_destroy(&run.ready_cond);
    pthread_mutex_destroy(&run.start_mutex);
    free(run.checksums);
    free(run.latency_ms);
    free(ids);
    free(offsets);
    free(token_counts);
}

static int32_t parse_token_pattern(const char *value,
                                   int32_t out[MAX_TOKEN_PATTERNS]) {
    char *copy = strdup(value);
    if (!copy) ei_die("out of memory");
    int32_t count = 0;
    char *save = NULL;
    for (char *item = strtok_r(copy, ",", &save); item;
         item = strtok_r(NULL, ",", &save)) {
        char *end = NULL;
        long tokens = strtol(item, &end, 10);
        if (*end != '\0' || tokens < 1 || tokens > EI_N_CTX ||
            count == MAX_TOKEN_PATTERNS) {
            free(copy);
            ei_die("--tokens must contain 1-2048 comma-separated values");
        }
        out[count++] = (int32_t)tokens;
    }
    free(copy);
    if (count == 0) ei_die("--tokens cannot be empty");
    return count;
}

static bool execute_engine_batch(void *opaque, const int32_t *ids,
                                 const size_t *offsets, size_t batch_size,
                                 float *out, char *err, size_t err_len) {
    return ei_engine_embed_tokens_batch(opaque, ids, offsets, batch_size,
                                        out, err, err_len);
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s --model model.gguf [--backend cpu|metal] "
            "[--tokens n[,n...]] [--concurrency 1,2,4] [--warmup n] "
            "[--min-requests n] [--max-batch-tokens n] "
            "[--max-batch-requests n] [--max-batch-sequence-tokens n] "
            "[--batch-wait-us n]\n",
            argv0);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *backend = "cpu";
    const char *concurrency_value = "1,2,4,8,16,32";
    const char *tokens_value = "2048";
    int32_t warmup = 3;
    int32_t min_requests = 4;
    int32_t max_batch_tokens = 4096;
    int32_t max_batch_requests = 64;
    int32_t max_batch_sequence_tokens = 512;
    int32_t batch_wait_us = 200;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            tokens_value = argv[++i];
        } else if (strcmp(argv[i], "--concurrency") == 0 && i + 1 < argc) {
            concurrency_value = argv[++i];
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--min-requests") == 0 && i + 1 < argc) {
            min_requests = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-batch-tokens") == 0 && i + 1 < argc) {
            max_batch_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-batch-requests") == 0 && i + 1 < argc) {
            max_batch_requests = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-batch-sequence-tokens") == 0 &&
                   i + 1 < argc) {
            max_batch_sequence_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batch-wait-us") == 0 && i + 1 < argc) {
            batch_wait_us = atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!model_path || (strcmp(backend, "cpu") != 0 && strcmp(backend, "metal") != 0) ||
        warmup < 0 || min_requests < 1) {
        usage(argv[0]);
        return 2;
    }
    if (max_batch_tokens < EI_N_CTX || max_batch_requests < 1 ||
        max_batch_sequence_tokens < 1 ||
        max_batch_sequence_tokens > EI_N_CTX || batch_wait_us < 0) {
        usage(argv[0]);
        return 2;
    }

    int32_t concurrency[MAX_CONCURRENCY_LEVELS];
    int32_t level_count = parse_concurrency(concurrency_value, concurrency);
    int32_t token_pattern[MAX_TOKEN_PATTERNS];
    int32_t pattern_count = parse_token_pattern(tokens_value, token_pattern);
    int32_t warmup_tokens = 0;
    for (int32_t i = 0; i < pattern_count; i++) {
        if (token_pattern[i] > warmup_tokens) warmup_tokens = token_pattern[i];
    }
    int32_t *warmup_ids = ei_xmalloc(
        sizeof(*warmup_ids) * (size_t)warmup_tokens);
    for (int32_t i = 0; i < warmup_tokens; i++) {
        warmup_ids[i] = 1000 + (i * 7919) % 240000;
    }

    ei_engine engine;
    ei_engine_load_backend(&engine, model_path, backend);
    float output[EI_N_EMBD];
    char err[256];
    for (int32_t i = 0; i < warmup; i++) {
        if (!ei_engine_embed_tokens(&engine, warmup_ids, (size_t)warmup_tokens,
                                    output, err, sizeof err)) {
            ei_die("warmup failed: %s", err);
        }
    }
    ei_inference_service_config config = {
        .cache_entries = 0,
        .max_batch_tokens = (size_t)max_batch_tokens,
        .max_batch_requests = (size_t)max_batch_requests,
        .max_batch_sequence_tokens = (size_t)max_batch_sequence_tokens,
        .tokenizer_workers = 0,
        .batch_wait_us = (uint32_t)batch_wait_us,
    };
    ei_inference_service *service = ei_inference_service_create(
        NULL, execute_engine_batch, &engine, &config);
    if (!service) ei_die("failed to create benchmark inference service");
    for (int32_t i = 0; i < level_count; i++) {
        run_level(&engine, service, token_pattern, pattern_count, tokens_value,
                  concurrency[i], min_requests);
    }

    ei_inference_service_free(service);
    ei_engine_free(&engine);
    free(warmup_ids);
    return 0;
}
