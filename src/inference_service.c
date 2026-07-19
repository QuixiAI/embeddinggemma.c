#define _POSIX_C_SOURCE 200809L

#include "inference_service.h"

#include <errno.h>
#include <pthread.h>
#include <time.h>

typedef enum {
    EI_ENTRY_PENDING,
    EI_ENTRY_READY,
    EI_ENTRY_FAILED,
} entry_state;

typedef struct cache_entry cache_entry;
typedef struct tokenize_job tokenize_job;
typedef struct tokenize_batch tokenize_batch;

struct tokenize_batch {
    pthread_mutex_t mutex;
    pthread_cond_t done;
    size_t remaining;
    const char *const *texts;
    const size_t *text_lengths;
    ei_tokens *tokens;
};

struct tokenize_job {
    tokenize_batch *batch;
    size_t index;
    tokenize_job *next;
};

struct cache_entry {
    uint64_t hash;
    int32_t *ids;
    size_t n_tokens;
    size_t users;
    entry_state state;
    bool in_lru;
    float embedding[EI_N_EMBD];
    char error[256];
    pthread_cond_t done;
    cache_entry *hash_next;
    cache_entry *queue_next;
    cache_entry *lru_prev;
    cache_entry *lru_next;
};

struct ei_inference_service {
    const ei_tokenizer *tokenizer;
    ei_inference_execute_fn execute;
    void *execute_opaque;
    ei_inference_service_config config;

    pthread_t backend_thread;
    pthread_t *tokenizer_threads;
    size_t tokenizer_thread_count;
    pthread_mutex_t mutex;
    pthread_cond_t queue_ready;
    bool stopping;
    bool batch_lookahead;
    bool adaptive_batch_wait;
    bool backend_busy;
    uint64_t batch_hot_until_ns;
    uint64_t batch_hot_window_ns;

    pthread_mutex_t tokenizer_mutex;
    pthread_cond_t tokenizer_ready;
    bool tokenizer_stopping;
    tokenize_job *tokenizer_head;
    tokenize_job *tokenizer_tail;

    cache_entry **buckets;
    size_t bucket_count;
    cache_entry *queue_head;
    cache_entry *queue_tail;
    cache_entry *lru_head;
    cache_entry *lru_tail;
    size_t ready_entries;
    size_t queued_requests;
    ei_inference_service_stats stats;
};

static void set_error(char *err, size_t err_len, const char *message) {
    if (err && err_len) snprintf(err, err_len, "%s", message);
}

static size_t next_power_of_two(size_t value) {
    size_t result = 1;
    while (result < value) result *= 2;
    return result;
}

static uint64_t hash_tokens(const int32_t *ids, size_t n_tokens) {
    uint64_t hash = 1469598103934665603ull;
    const uint8_t *bytes = (const uint8_t *)ids;
    size_t n = n_tokens * sizeof(*ids);
    for (size_t i = 0; i < n; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    hash ^= n_tokens;
    hash *= 1099511628211ull;
    return hash;
}

static cache_entry **entry_slot(ei_inference_service *service,
                                uint64_t hash, const int32_t *ids,
                                size_t n_tokens) {
    cache_entry **slot = &service->buckets[hash & (service->bucket_count - 1)];
    while (*slot) {
        cache_entry *entry = *slot;
        if (entry->hash == hash && entry->n_tokens == n_tokens &&
            memcmp(entry->ids, ids, n_tokens * sizeof(*ids)) == 0) {
            break;
        }
        slot = &entry->hash_next;
    }
    return slot;
}

static void lru_remove(ei_inference_service *service, cache_entry *entry) {
    if (!entry->in_lru) return;
    if (entry->lru_prev) entry->lru_prev->lru_next = entry->lru_next;
    else service->lru_head = entry->lru_next;
    if (entry->lru_next) entry->lru_next->lru_prev = entry->lru_prev;
    else service->lru_tail = entry->lru_prev;
    entry->lru_prev = NULL;
    entry->lru_next = NULL;
    entry->in_lru = false;
}

static void lru_push_front(ei_inference_service *service, cache_entry *entry) {
    lru_remove(service, entry);
    entry->lru_next = service->lru_head;
    if (service->lru_head) service->lru_head->lru_prev = entry;
    else service->lru_tail = entry;
    service->lru_head = entry;
    entry->in_lru = true;
}

static void destroy_entry(cache_entry *entry) {
    pthread_cond_destroy(&entry->done);
    free(entry->ids);
    free(entry);
}

static void remove_entry(ei_inference_service *service, cache_entry *entry) {
    cache_entry **slot = entry_slot(service, entry->hash, entry->ids, entry->n_tokens);
    if (*slot == entry) *slot = entry->hash_next;
    if (entry->in_lru) {
        lru_remove(service, entry);
        service->ready_entries--;
    }
    destroy_entry(entry);
}

static void prune_cache(ei_inference_service *service) {
    while (service->ready_entries > service->config.cache_entries) {
        cache_entry *entry = service->lru_tail;
        while (entry && entry->users != 0) entry = entry->lru_prev;
        if (!entry) break;
        remove_entry(service, entry);
    }
}

static void release_entry(ei_inference_service *service, cache_entry *entry) {
    entry->users--;
    if (entry->state == EI_ENTRY_FAILED && entry->users == 0) {
        remove_entry(service, entry);
    } else {
        prune_cache(service);
    }
}

static void collection_deadline(uint32_t wait_us, struct timespec *deadline) {
    if (clock_gettime(CLOCK_REALTIME, deadline) != 0) {
        ei_die("clock_gettime failed: %s", strerror(errno));
    }
    deadline->tv_nsec += (long)(wait_us % 1000000u) * 1000l;
    deadline->tv_sec += (time_t)(wait_us / 1000000u) + deadline->tv_nsec / 1000000000l;
    deadline->tv_nsec %= 1000000000l;
}

static uint64_t monotonic_ns(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        ei_die("clock_gettime failed: %s", strerror(errno));
    }
    return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static void mark_batch_hot(ei_inference_service *service) {
    uint64_t now = monotonic_ns();
    service->batch_hot_until_ns = UINT64_MAX - now < service->batch_hot_window_ns
        ? UINT64_MAX : now + service->batch_hot_window_ns;
}

static bool should_collect_batch(const ei_inference_service *service) {
    if (service->config.batch_wait_us == 0) return false;
    if (!service->adaptive_batch_wait) return true;
    if (service->queue_head && service->queue_head->queue_next) return true;
    return monotonic_ns() < service->batch_hot_until_ns;
}

static size_t batch_lookahead_limit(size_t max_requests) {
    return max_requests <= SIZE_MAX / 8u ? max_requests * 8u : SIZE_MAX;
}

static bool batch_collection_ready(const ei_inference_service *service) {
    const cache_entry *entry = service->queue_head;
    if (!entry) return false;
    if (entry->n_tokens > service->config.max_batch_sequence_tokens) return true;

    if (service->batch_lookahead) {
        size_t requests = 1;
        size_t tokens = entry->n_tokens;
        size_t scanned = 0;
        const size_t max_scan = batch_lookahead_limit(
            service->config.max_batch_requests);
        entry = entry->queue_next;
        while (entry && scanned < max_scan) {
            scanned++;
            if (entry->n_tokens <= service->config.max_batch_sequence_tokens &&
                tokens + entry->n_tokens <= service->config.max_batch_tokens) {
                requests++;
                tokens += entry->n_tokens;
                if (requests == service->config.max_batch_requests ||
                    tokens == service->config.max_batch_tokens) {
                    return true;
                }
            }
            entry = entry->queue_next;
        }
        return false;
    }

    size_t requests = 0;
    size_t tokens = 0;
    while (entry) {
        if (requests > 0 &&
            (entry->n_tokens > service->config.max_batch_sequence_tokens ||
             tokens + entry->n_tokens > service->config.max_batch_tokens)) {
            return true;
        }
        requests++;
        tokens += entry->n_tokens;
        if (requests == service->config.max_batch_requests ||
            tokens == service->config.max_batch_tokens) {
            return true;
        }
        entry = entry->queue_next;
    }
    return false;
}

static void *tokenizer_main(void *opaque) {
    ei_inference_service *service = opaque;
    pthread_mutex_lock(&service->tokenizer_mutex);
    for (;;) {
        while (!service->tokenizer_stopping && !service->tokenizer_head) {
            pthread_cond_wait(&service->tokenizer_ready,
                              &service->tokenizer_mutex);
        }
        if (service->tokenizer_stopping && !service->tokenizer_head) break;
        tokenize_job *job = service->tokenizer_head;
        service->tokenizer_head = job->next;
        if (!service->tokenizer_head) service->tokenizer_tail = NULL;
        pthread_mutex_unlock(&service->tokenizer_mutex);

        tokenize_batch *batch = job->batch;
        ei_tokenize_spm(service->tokenizer, batch->texts[job->index],
                        batch->text_lengths[job->index], true, false,
                        &batch->tokens[job->index]);
        pthread_mutex_lock(&batch->mutex);
        batch->remaining--;
        if (batch->remaining == 0) pthread_cond_signal(&batch->done);
        pthread_mutex_unlock(&batch->mutex);

        pthread_mutex_lock(&service->tokenizer_mutex);
    }
    pthread_mutex_unlock(&service->tokenizer_mutex);
    return NULL;
}

static void *backend_main(void *opaque) {
    ei_inference_service *service = opaque;
    size_t max_requests = service->config.max_batch_requests;
    cache_entry **batch = ei_xmalloc(sizeof(*batch) * max_requests);
    size_t *offsets = ei_xmalloc(sizeof(*offsets) * (max_requests + 1));
    int32_t *ids = ei_xmalloc(sizeof(*ids) * service->config.max_batch_tokens);
    float *output = ei_xmalloc(sizeof(*output) * max_requests * EI_N_EMBD);

    pthread_mutex_lock(&service->mutex);
    for (;;) {
        while (!service->stopping && !service->queue_head) {
            pthread_cond_wait(&service->queue_ready, &service->mutex);
        }
        if (service->stopping && !service->queue_head) break;

        if (should_collect_batch(service)) {
            struct timespec deadline;
            collection_deadline(service->config.batch_wait_us, &deadline);
            while (!service->stopping && !batch_collection_ready(service)) {
                int rc = pthread_cond_timedwait(
                    &service->queue_ready, &service->mutex, &deadline);
                if (rc == ETIMEDOUT) break;
                if (rc != 0) ei_die("batch collection wait failed: %s", strerror(rc));
            }
        }

        size_t batch_size = 0;
        size_t total_tokens = 0;
        offsets[0] = 0;
        while (!service->batch_lookahead && service->queue_head &&
               batch_size < max_requests) {
            cache_entry *entry = service->queue_head;
            if (batch_size > 0) {
                if (batch[0]->n_tokens >
                        service->config.max_batch_sequence_tokens ||
                    entry->n_tokens >
                        service->config.max_batch_sequence_tokens ||
                    total_tokens + entry->n_tokens >
                        service->config.max_batch_tokens) {
                    break;
                }
            }
            service->queue_head = entry->queue_next;
            if (!service->queue_head) service->queue_tail = NULL;
            entry->queue_next = NULL;
            service->queued_requests--;
            batch[batch_size++] = entry;
            total_tokens += entry->n_tokens;
            offsets[batch_size] = total_tokens;
        }
        if (service->batch_lookahead) {
            cache_entry *entry = service->queue_head;
            service->queue_head = entry->queue_next;
            if (!service->queue_head) service->queue_tail = NULL;
            entry->queue_next = NULL;
            service->queued_requests--;
            batch[batch_size++] = entry;
            total_tokens = entry->n_tokens;
            offsets[batch_size] = total_tokens;

            if (entry->n_tokens <= service->config.max_batch_sequence_tokens) {
                const size_t max_scan = batch_lookahead_limit(max_requests);
                size_t scanned = 0;
                cache_entry *previous = NULL;
                entry = service->queue_head;
                while (entry && batch_size < max_requests && scanned < max_scan) {
                    cache_entry *next = entry->queue_next;
                    scanned++;
                    if (entry->n_tokens <= service->config.max_batch_sequence_tokens &&
                        total_tokens + entry->n_tokens <=
                            service->config.max_batch_tokens) {
                        if (previous) previous->queue_next = next;
                        else service->queue_head = next;
                        if (service->queue_tail == entry) {
                            service->queue_tail = previous;
                        }
                        entry->queue_next = NULL;
                        service->queued_requests--;
                        batch[batch_size++] = entry;
                        total_tokens += entry->n_tokens;
                        offsets[batch_size] = total_tokens;
                    } else {
                        previous = entry;
                    }
                    entry = next;
                }
                if (!service->queue_head) service->queue_tail = NULL;
            }
        }

        for (size_t i = 0; i < batch_size; i++) {
            memcpy(ids + offsets[i], batch[i]->ids,
                   batch[i]->n_tokens * sizeof(*ids));
        }
        service->backend_busy = true;
        pthread_mutex_unlock(&service->mutex);

        char error[256];
        bool ok = service->execute(service->execute_opaque, ids, offsets,
                                   batch_size, output, error, sizeof error);

        if (ok) {
            for (size_t i = 0; i < batch_size; i++) {
                memcpy(batch[i]->embedding, output + i * EI_N_EMBD,
                       sizeof(batch[i]->embedding));
            }
        }
        pthread_mutex_lock(&service->mutex);
        service->backend_busy = false;
        service->stats.batches++;
        service->stats.sequences += batch_size;
        service->stats.tokens += total_tokens;
        for (size_t i = 0; i < batch_size; i++) {
            cache_entry *entry = batch[i];
            if (ok) {
                entry->state = EI_ENTRY_READY;
                lru_push_front(service, entry);
                service->ready_entries++;
            } else {
                entry->state = EI_ENTRY_FAILED;
                snprintf(entry->error, sizeof entry->error, "%s", error);
            }
            pthread_cond_broadcast(&entry->done);
        }
        prune_cache(service);
    }
    pthread_mutex_unlock(&service->mutex);
    free(output);
    free(ids);
    free(offsets);
    free(batch);
    return NULL;
}

ei_inference_service *ei_inference_service_create(
    const ei_tokenizer *tokenizer, ei_inference_execute_fn execute,
    void *execute_opaque, const ei_inference_service_config *config) {
    if (!execute || !config || config->max_batch_tokens < EI_N_CTX ||
        config->max_batch_requests == 0 ||
        config->max_batch_sequence_tokens == 0 ||
        config->max_batch_sequence_tokens > EI_N_CTX ||
        (config->tokenizer_workers > 0 && !tokenizer)) {
        return NULL;
    }
    ei_inference_service *service = ei_xcalloc(1, sizeof(*service));
    service->tokenizer = tokenizer;
    service->execute = execute;
    service->execute_opaque = execute_opaque;
    service->config = *config;
    const char *lookahead = getenv("EI_BATCH_LOOKAHEAD");
    service->batch_lookahead = !lookahead || strcmp(lookahead, "0") != 0;
    const char *adaptive_wait = getenv("EI_ADAPTIVE_BATCH_WAIT");
    service->adaptive_batch_wait = !adaptive_wait ||
        strcmp(adaptive_wait, "0") != 0;
    uint64_t wait_ns = (uint64_t)config->batch_wait_us * 1000ull;
    service->batch_hot_window_ns = wait_ns > 100000ull
        ? wait_ns * 10ull : 1000000ull;
    size_t desired_buckets = config->cache_entries > 32
        ? config->cache_entries * 2 : 64;
    service->bucket_count = next_power_of_two(desired_buckets);
    service->buckets = ei_xcalloc(service->bucket_count, sizeof(*service->buckets));
    if (pthread_mutex_init(&service->mutex, NULL) != 0 ||
        pthread_cond_init(&service->queue_ready, NULL) != 0 ||
        pthread_mutex_init(&service->tokenizer_mutex, NULL) != 0 ||
        pthread_cond_init(&service->tokenizer_ready, NULL) != 0) {
        ei_die("failed to initialize inference service synchronization");
    }
    service->tokenizer_thread_count = config->tokenizer_workers;
    if (service->tokenizer_thread_count > 0) {
        service->tokenizer_threads = ei_xmalloc(
            service->tokenizer_thread_count * sizeof(*service->tokenizer_threads));
        for (size_t i = 0; i < service->tokenizer_thread_count; i++) {
            if (pthread_create(&service->tokenizer_threads[i], NULL,
                               tokenizer_main, service) != 0) {
                ei_die("failed to create tokenizer worker %zu", i);
            }
        }
    }
    if (pthread_create(&service->backend_thread, NULL, backend_main, service) != 0) {
        ei_die("failed to create inference backend thread");
    }
    return service;
}

void ei_inference_service_free(ei_inference_service *service) {
    if (!service) return;
    pthread_mutex_lock(&service->mutex);
    service->stopping = true;
    pthread_cond_signal(&service->queue_ready);
    pthread_mutex_unlock(&service->mutex);
    pthread_join(service->backend_thread, NULL);

    pthread_mutex_lock(&service->tokenizer_mutex);
    service->tokenizer_stopping = true;
    pthread_cond_broadcast(&service->tokenizer_ready);
    pthread_mutex_unlock(&service->tokenizer_mutex);
    for (size_t i = 0; i < service->tokenizer_thread_count; i++) {
        pthread_join(service->tokenizer_threads[i], NULL);
    }

    for (size_t bucket = 0; bucket < service->bucket_count; bucket++) {
        cache_entry *entry = service->buckets[bucket];
        while (entry) {
            cache_entry *next = entry->hash_next;
            destroy_entry(entry);
            entry = next;
        }
    }
    pthread_cond_destroy(&service->queue_ready);
    pthread_mutex_destroy(&service->mutex);
    pthread_cond_destroy(&service->tokenizer_ready);
    pthread_mutex_destroy(&service->tokenizer_mutex);
    free(service->tokenizer_threads);
    free(service->buckets);
    free(service);
}

static cache_entry *submit_tokens_locked(ei_inference_service *service,
                                         const int32_t *ids, size_t n_tokens) {
    uint64_t hash = hash_tokens(ids, n_tokens);
    cache_entry **slot = entry_slot(service, hash, ids, n_tokens);
    cache_entry *entry = *slot;
    if (entry) {
        entry->users++;
        if (entry->state == EI_ENTRY_READY) {
            service->stats.cache_hits++;
            lru_push_front(service, entry);
        } else if (entry->state == EI_ENTRY_PENDING) {
            service->stats.coalesced_requests++;
            if (service->backend_busy || service->queue_head) {
                mark_batch_hot(service);
            }
        }
        return entry;
    }

    entry = ei_xcalloc(1, sizeof(*entry));
    entry->hash = hash;
    entry->ids = ei_xmalloc(n_tokens * sizeof(*entry->ids));
    memcpy(entry->ids, ids, n_tokens * sizeof(*entry->ids));
    entry->n_tokens = n_tokens;
    entry->users = 1;
    entry->state = EI_ENTRY_PENDING;
    if (pthread_cond_init(&entry->done, NULL) != 0) {
        ei_die("failed to initialize inference completion");
    }
    entry->hash_next = *slot;
    *slot = entry;
    if (service->queue_head || service->backend_busy) mark_batch_hot(service);
    if (service->queue_tail) service->queue_tail->queue_next = entry;
    else service->queue_head = entry;
    service->queue_tail = entry;
    service->queued_requests++;
    service->stats.cache_misses++;
    return entry;
}

static bool await_entry_locked(ei_inference_service *service, cache_entry *entry,
                               float *out, char *err, size_t err_len) {
    while (entry->state == EI_ENTRY_PENDING) {
        pthread_cond_wait(&entry->done, &service->mutex);
    }
    bool ok = entry->state == EI_ENTRY_READY;
    if (ok) {
        memcpy(out, entry->embedding, sizeof(entry->embedding));
    } else {
        set_error(err, err_len, entry->error);
    }
    release_entry(service, entry);
    return ok;
}

bool ei_inference_service_embed_tokens(ei_inference_service *service,
                                       const int32_t *ids, size_t n_tokens,
                                       float out[EI_N_EMBD],
                                       char *err, size_t err_len) {
    if (!service || !ids || n_tokens == 0 || n_tokens > EI_N_CTX) {
        set_error(err, err_len, "input token count must be 1..2048");
        return false;
    }
    pthread_mutex_lock(&service->mutex);
    cache_entry *entry = submit_tokens_locked(service, ids, n_tokens);
    pthread_cond_signal(&service->queue_ready);
    bool ok = await_entry_locked(service, entry, out, err, err_len);
    if (ok && err && err_len) err[0] = '\0';
    pthread_mutex_unlock(&service->mutex);
    return ok;
}

bool ei_inference_service_embed(ei_inference_service *service,
                                const char *text, size_t text_len,
                                float out[EI_N_EMBD], char *err, size_t err_len) {
    if (!service || !service->tokenizer) {
        set_error(err, err_len, "inference service has no tokenizer");
        return false;
    }
    ei_tokens tokens;
    ei_tokenize_spm(service->tokenizer, text, text_len, true, false, &tokens);
    bool ok = ei_inference_service_embed_tokens(
        service, tokens.ids, tokens.n, out, err, err_len);
    ei_tokens_free(&tokens);
    return ok;
}

bool ei_inference_service_embed_batch(ei_inference_service *service,
                                      const char *const *texts,
                                      const size_t *text_lengths,
                                      size_t batch_size, float *out,
                                      char *err, size_t err_len) {
    if (!service || !service->tokenizer || !texts || !text_lengths ||
        batch_size == 0) {
        set_error(err, err_len, "embedding batch must contain at least one input");
        return false;
    }
    ei_tokens *tokens = ei_xcalloc(batch_size, sizeof(*tokens));
    cache_entry **entries = ei_xmalloc(batch_size * sizeof(*entries));
    if (batch_size > 1 && service->tokenizer_thread_count > 0) {
        tokenize_batch batch = {
            .remaining = batch_size, .texts = texts,
            .text_lengths = text_lengths, .tokens = tokens,
        };
        if (pthread_mutex_init(&batch.mutex, NULL) != 0 ||
            pthread_cond_init(&batch.done, NULL) != 0) {
            ei_die("failed to initialize tokenization batch");
        }
        tokenize_job *jobs = ei_xcalloc(batch_size, sizeof(*jobs));
        pthread_mutex_lock(&service->tokenizer_mutex);
        for (size_t i = 0; i < batch_size; i++) {
            jobs[i].batch = &batch;
            jobs[i].index = i;
            if (service->tokenizer_tail) service->tokenizer_tail->next = &jobs[i];
            else service->tokenizer_head = &jobs[i];
            service->tokenizer_tail = &jobs[i];
        }
        pthread_cond_broadcast(&service->tokenizer_ready);
        pthread_mutex_unlock(&service->tokenizer_mutex);
        pthread_mutex_lock(&batch.mutex);
        while (batch.remaining != 0) {
            pthread_cond_wait(&batch.done, &batch.mutex);
        }
        pthread_mutex_unlock(&batch.mutex);
        pthread_cond_destroy(&batch.done);
        pthread_mutex_destroy(&batch.mutex);
        free(jobs);
    } else {
        for (size_t i = 0; i < batch_size; i++) {
            ei_tokenize_spm(service->tokenizer, texts[i], text_lengths[i],
                            true, false, &tokens[i]);
        }
    }
    for (size_t i = 0; i < batch_size; i++) {
        if (tokens[i].n == 0 || tokens[i].n > EI_N_CTX) {
            for (size_t j = 0; j < batch_size; j++) ei_tokens_free(&tokens[j]);
            free(entries);
            free(tokens);
            set_error(err, err_len, "input token count must be 1..2048");
            return false;
        }
    }

    pthread_mutex_lock(&service->mutex);
    for (size_t i = 0; i < batch_size; i++) {
        entries[i] = submit_tokens_locked(service, tokens[i].ids, tokens[i].n);
    }
    pthread_cond_signal(&service->queue_ready);
    bool ok = true;
    char first_error[256] = {0};
    for (size_t i = 0; i < batch_size; i++) {
        char item_error[256];
        if (!await_entry_locked(service, entries[i],
                                out + i * EI_N_EMBD,
                                item_error, sizeof item_error)) {
            if (ok) snprintf(first_error, sizeof first_error, "%s", item_error);
            ok = false;
        }
    }
    pthread_mutex_unlock(&service->mutex);
    for (size_t i = 0; i < batch_size; i++) ei_tokens_free(&tokens[i]);
    free(entries);
    free(tokens);
    if (ok) {
        if (err && err_len) err[0] = '\0';
    } else {
        set_error(err, err_len, first_error);
    }
    return ok;
}

void ei_inference_service_get_stats(ei_inference_service *service,
                                    ei_inference_service_stats *stats) {
    pthread_mutex_lock(&service->mutex);
    *stats = service->stats;
    stats->cache_entries = service->ready_entries;
    stats->queued_requests = service->queued_requests;
    pthread_mutex_unlock(&service->mutex);
}
