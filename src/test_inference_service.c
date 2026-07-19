#define _POSIX_C_SOURCE 200809L

#include "inference_service.h"

#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

typedef struct {
    _Atomic int calls;
    _Atomic int max_batch;
} fake_backend;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int ready;
    int target;
    bool start;
} start_gate;

typedef struct {
    ei_inference_service *service;
    start_gate *gate;
    int32_t ids[8];
    size_t n_tokens;
    bool ok;
    float first;
    char error[256];
} request_context;

static float expected_first(const int32_t *ids, size_t n_tokens) {
    int64_t sum = 0;
    for (size_t i = 0; i < n_tokens; i++) sum += ids[i];
    return (float)sum;
}

static bool fake_execute(void *opaque, const int32_t *ids,
                         const size_t *offsets, size_t batch_size,
                         float *out, char *err, size_t err_len) {
    fake_backend *backend = opaque;
    atomic_fetch_add(&backend->calls, 1);
    int observed = atomic_load(&backend->max_batch);
    while (observed < (int)batch_size &&
           !atomic_compare_exchange_weak(&backend->max_batch, &observed,
                                         (int)batch_size)) {}
    struct timespec delay = { .tv_nsec = 20000000l };
    nanosleep(&delay, NULL);
    for (size_t sequence = 0; sequence < batch_size; sequence++) {
        float value = expected_first(ids + offsets[sequence],
                                     offsets[sequence + 1] - offsets[sequence]);
        for (int32_t d = 0; d < EI_N_EMBD; d++) {
            out[sequence * EI_N_EMBD + d] = value + (float)d * 0.001f;
        }
    }
    if (err && err_len) err[0] = '\0';
    return true;
}

static void gate_wait(start_gate *gate) {
    pthread_mutex_lock(&gate->mutex);
    gate->ready++;
    pthread_cond_broadcast(&gate->cond);
    while (!gate->start) pthread_cond_wait(&gate->cond, &gate->mutex);
    pthread_mutex_unlock(&gate->mutex);
}

static void gate_release(start_gate *gate) {
    pthread_mutex_lock(&gate->mutex);
    while (gate->ready != gate->target) pthread_cond_wait(&gate->cond, &gate->mutex);
    gate->start = true;
    pthread_cond_broadcast(&gate->cond);
    pthread_mutex_unlock(&gate->mutex);
}

static void *request_main(void *opaque) {
    request_context *request = opaque;
    gate_wait(request->gate);
    float output[EI_N_EMBD];
    request->ok = ei_inference_service_embed_tokens(
        request->service, request->ids, request->n_tokens, output,
        request->error, sizeof request->error);
    request->first = output[0];
    return NULL;
}

static void run_prepared_group(ei_inference_service *service,
                               request_context *requests, size_t count) {
    start_gate gate = { .target = (int)count };
    if (pthread_mutex_init(&gate.mutex, NULL) != 0 ||
        pthread_cond_init(&gate.cond, NULL) != 0) {
        ei_die("failed to initialize test gate");
    }
    pthread_t *threads = ei_xmalloc(sizeof(*threads) * count);
    for (size_t i = 0; i < count; i++) {
        requests[i].service = service;
        requests[i].gate = &gate;
        if (pthread_create(&threads[i], NULL, request_main, &requests[i]) != 0) {
            ei_die("failed to create service test worker");
        }
    }
    gate_release(&gate);
    for (size_t i = 0; i < count; i++) {
        pthread_join(threads[i], NULL);
        if (!requests[i].ok) ei_die("service request failed: %s", requests[i].error);
        float expected = expected_first(requests[i].ids, requests[i].n_tokens);
        if (requests[i].first != expected) {
            ei_die("service result mismatch: got %.9g expected %.9g",
                   requests[i].first, expected);
        }
    }
    free(threads);
    pthread_cond_destroy(&gate.cond);
    pthread_mutex_destroy(&gate.mutex);
}

static void run_group(ei_inference_service *service, request_context *requests,
                      size_t count, size_t n_tokens, bool identical) {
    for (size_t i = 0; i < count; i++) {
        requests[i].n_tokens = n_tokens;
        for (size_t token = 0; token < n_tokens; token++) {
            requests[i].ids[token] = identical
                ? 11 + (int32_t)token
                : 100 * (int32_t)(token + 1) + (int32_t)i;
        }
    }
    run_prepared_group(service, requests, count);
}

int main(void) {
    if (unsetenv("EI_BATCH_LOOKAHEAD") != 0) {
        ei_die("failed to clear lookahead environment override");
    }
    fake_backend backend = {0};
    ei_inference_service_config config = {
        .cache_entries = 2,
        .max_batch_tokens = EI_N_CTX,
        .max_batch_requests = 16,
        .max_batch_sequence_tokens = 4,
        .tokenizer_workers = 0,
        .batch_wait_us = 50000,
    };
    ei_inference_service *service = ei_inference_service_create(
        NULL, fake_execute, &backend, &config);
    if (!service) ei_die("failed to create inference service");

    request_context duplicates[8] = {0};
    run_group(service, duplicates, 8, 3, true);
    if (atomic_load(&backend.calls) != 1) {
        ei_die("singleflight executed %d backend calls, expected 1",
               atomic_load(&backend.calls));
    }

    float output[EI_N_EMBD];
    char err[256];
    const int32_t duplicate_ids[] = { 11, 12, 13 };
    if (!ei_inference_service_embed_tokens(
            service, duplicate_ids, 3, output, err, sizeof err)) {
        ei_die("cache hit failed: %s", err);
    }
    if (atomic_load(&backend.calls) != 1) ei_die("cache hit reached backend");

    request_context distinct[4] = {0};
    run_group(service, distinct, 4, 3, false);
    if (atomic_load(&backend.calls) != 2 || atomic_load(&backend.max_batch) != 4) {
        ei_die("batching got calls=%d max_batch=%d, expected 2 and 4",
               atomic_load(&backend.calls), atomic_load(&backend.max_batch));
    }

    if (!ei_inference_service_embed_tokens(
            service, duplicate_ids, 3, output, err, sizeof err)) {
        ei_die("post-eviction request failed: %s", err);
    }
    if (atomic_load(&backend.calls) != 3) {
        ei_die("LRU did not evict the oldest entry");
    }

    ei_inference_service_stats stats;
    ei_inference_service_get_stats(service, &stats);
    printf("service cache hits=%llu misses=%llu coalesced=%llu "
           "batches=%llu sequences=%llu max_batch=%d cache_entries=%zu\n",
           (unsigned long long)stats.cache_hits,
           (unsigned long long)stats.cache_misses,
           (unsigned long long)stats.coalesced_requests,
           (unsigned long long)stats.batches,
           (unsigned long long)stats.sequences,
           atomic_load(&backend.max_batch), stats.cache_entries);
    if (stats.cache_hits != 1 || stats.cache_misses != 6 ||
        stats.coalesced_requests != 7 || stats.cache_entries != 2) {
        ei_die("unexpected inference service statistics");
    }

    ei_inference_service_free(service);

    fake_backend mixed_backend = {0};
    ei_inference_service_config mixed_config = config;
    mixed_config.cache_entries = 0;
    ei_inference_service *mixed_service = ei_inference_service_create(
        NULL, fake_execute, &mixed_backend, &mixed_config);
    if (!mixed_service) ei_die("failed to create mixed-request service");
    request_context mixed_requests[8] = {0};
    for (size_t i = 0; i < 8; i++) {
        mixed_requests[i].n_tokens = i % 2 == 0 ? 3 : 5;
        for (size_t token = 0; token < mixed_requests[i].n_tokens; token++) {
            mixed_requests[i].ids[token] =
                1000 * (int32_t)(i + 1) + (int32_t)token;
        }
    }
    run_prepared_group(mixed_service, mixed_requests, 8);
    if (atomic_load(&mixed_backend.calls) > 6 ||
        atomic_load(&mixed_backend.max_batch) < 3) {
        ei_die("lookahead batching got calls=%d max_batch=%d, expected <=6 and >=3",
               atomic_load(&mixed_backend.calls),
               atomic_load(&mixed_backend.max_batch));
    }
    ei_inference_service_free(mixed_service);

    if (setenv("EI_BATCH_LOOKAHEAD", "0", 1) != 0) {
        ei_die("failed to set FIFO environment override");
    }
    fake_backend fifo_backend = {0};
    ei_inference_service *fifo_service = ei_inference_service_create(
        NULL, fake_execute, &fifo_backend, &mixed_config);
    if (!fifo_service) ei_die("failed to create FIFO service");
    run_prepared_group(fifo_service, mixed_requests, 8);
    ei_inference_service_free(fifo_service);
    if (unsetenv("EI_BATCH_LOOKAHEAD") != 0) {
        ei_die("failed to clear FIFO environment override");
    }

    fake_backend long_backend = {0};
    ei_inference_service *long_service = ei_inference_service_create(
        NULL, fake_execute, &long_backend, &config);
    if (!long_service) ei_die("failed to create long-request service");
    request_context long_requests[3] = {0};
    run_group(long_service, long_requests, 3, 5, false);
    if (atomic_load(&long_backend.calls) != 3 ||
        atomic_load(&long_backend.max_batch) != 1) {
        ei_die("long requests were packed: calls=%d max_batch=%d",
               atomic_load(&long_backend.calls),
               atomic_load(&long_backend.max_batch));
    }
    ei_inference_service_free(long_service);
    return 0;
}
