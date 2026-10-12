#define _POSIX_C_SOURCE 200809L

#include "inference_service.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

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

    pthread_t *backend_threads;
    size_t slots;
    pthread_t *tokenizer_threads;
    size_t tokenizer_thread_count;
    pthread_mutex_t mutex;
    pthread_cond_t queue_ready;
    bool stopping;
    bool batch_lookahead;
    bool sort_waves; // EI_BATCH_SORT=0 keeps queue order within a wave
    bool adaptive_batch_wait;
    bool batch_wave;
    bool solo_inline;
    size_t busy_slots;
    uint32_t slot_mask;
    uint32_t long_mask; /* busy slots running long waves (bypass mode) */
    uint32_t group_mask[32]; /* slots of each device */
    uint32_t overflow_mask;  /* CPU-worker slots beside a GPU */
    size_t groups;
    uint64_t batch_hot_until_ns;
    uint64_t batch_hot_window_ns;
    uint64_t batch_wait_cap_ns;
    /* Requests accepted but still tokenizing. While any are pending, the wave
     * collector keeps waiting (up to arrival_wait_ns from the start of the
     * collection) so a wave is not cut just ahead of requests that are about
     * to queue: at 512 tokens x c16 the 16 clients otherwise phase-lock into
     * a 15-request wave plus a lone one. */
    size_t arriving_requests;
    uint64_t arrival_wait_ns;
    size_t active_requests;
    size_t active_peak;

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

static size_t context_limit(const ei_tokenizer *tokenizer) {
    return tokenizer && tokenizer->model && tokenizer->model->embeddinggemma2 ? 8192 : EI_N_CTX;
}

/* Over-long text is truncated to the profile context, like the reference
 * tokenizer: keep the leading tokens and the trailing EOS when one was added. */
static void truncate_tokens(const ei_inference_service *service, ei_tokens *tokens) {
    size_t limit = context_limit(service->tokenizer);
    if (tokens->n <= limit) return;
    const ei_model *model = service->tokenizer->model;
    if (model->add_eos && tokens->ids[tokens->n - 1] == model->eos_id) {
        tokens->ids[limit - 1] = model->eos_id;
    }
    tokens->n = limit;
}

static void set_error(char *err, size_t err_len, const char *message) {
    if (err && err_len) snprintf(err, err_len, "%s", message);
}

static bool bypass_mode(const ei_inference_service *service) {
    return service->config.short_bypass_tokens > 0 && service->slots > 1;
}
static bool is_short(const ei_inference_service *service, const cache_entry *entry) {
    return entry->n_tokens <= service->config.short_bypass_tokens;
}
static bool queue_has_short(const ei_inference_service *service) {
    for (const cache_entry *e = service->queue_head; e; e = e->queue_next)
        if (is_short(service, e)) return true;
    return false;
}
static bool overflow_eligible(const ei_inference_service *service, const cache_entry *entry) {
    return entry->n_tokens <= service->config.overflow_max_sequence_tokens;
}
/* Overflow waves serve real backlog only: requests queued behind the work
 * the next wave of every device absorbs (otherwise the GPUs clear them sooner
 * than a CPU wave would), and only requests small enough for a CPU worker.
 * `ahead` is the queued work before an entry; take_class_locked uses the same
 * rule. */
static bool in_backlog(const ei_inference_service *service, size_t ahead) {
    return ahead >= service->config.max_batch_tokens * service->groups;
}
static bool queue_has_overflow(const ei_inference_service *service) {
    size_t ahead = 0;
    for (const cache_entry *e = service->queue_head; e; e = e->queue_next) {
        if (in_backlog(service, ahead) && overflow_eligible(service, e)) return true;
        ahead += e->n_tokens;
    }
    return false;
}
static size_t popcount32(uint32_t v) {
    size_t n = 0;
    for (; v; v &= v - 1) n++;
    return n;
}
static size_t lowest_bit(uint32_t v) {
    size_t i = 0;
    while (!(v & (1u << i))) i++;
    return i;
}

enum { WAVE_NONE, WAVE_ANY, WAVE_SHORT, WAVE_OVERFLOW };
/* Choose the slot for the next wave, or WAVE_NONE. An idle device starts any
 * wave, the first idle device winning. Otherwise, without bypass, the
 * least-loaded device with a free pipeline slot does; with bypass, a device
 * whose in-flight waves are all long may start a short-only wave. Overflow
 * (CPU) slots start a wave of small requests only when no device can. */
static int pick_slot_locked(const ei_inference_service *service, size_t *slot) {
    if (!service->queue_head) return WAVE_NONE;
    size_t best = SIZE_MAX, best_busy = SIZE_MAX;
    bool shorts = bypass_mode(service) && queue_has_short(service);
    int mode = WAVE_NONE;
    for (size_t g = 0; g < service->groups; g++) {
        uint32_t free = service->group_mask[g] & ~service->slot_mask;
        uint32_t busy = service->group_mask[g] & service->slot_mask;
        if (!free) continue;
        if (!busy) {
            *slot = lowest_bit(free);
            return WAVE_ANY;
        }
        if (!bypass_mode(service)) {
            if (popcount32(busy) < best_busy) {
                best_busy = popcount32(busy);
                best = lowest_bit(free);
                mode = WAVE_ANY;
            }
        } else if (shorts && (busy & ~service->long_mask) == 0 && mode == WAVE_NONE) {
            best = lowest_bit(free);
            mode = WAVE_SHORT;
        }
    }
    if (mode != WAVE_NONE) {
        *slot = best;
        return mode;
    }
    uint32_t overflow = service->overflow_mask & ~service->slot_mask;
    if (overflow && queue_has_overflow(service)) {
        *slot = lowest_bit(overflow);
        return WAVE_OVERFLOW;
    }
    return WAVE_NONE;
}
static bool can_start_wave(const ei_inference_service *service) {
    size_t slot;
    return pick_slot_locked(service, &slot) != WAVE_NONE;
}
static size_t wave_budget(const ei_inference_service *service, size_t slot) {
    const size_t *caps = service->config.slot_max_batch_tokens;
    return caps && caps[slot] ? caps[slot] : service->config.max_batch_tokens;
}
/* Unlink queued entries of one class (short or long) into a wave, in queue
 * order, within the token/request budgets; others stay queued. */
static size_t take_class_locked(ei_inference_service *service, int wave_class, size_t budget,
                                cache_entry **batch, size_t *offsets, size_t max_requests,
                                size_t *total_tokens) {
    size_t batch_size = 0, total = 0;
    cache_entry *previous = NULL, *entry = service->queue_head;
    bool overflow = wave_class == WAVE_OVERFLOW;
    const size_t seq_limit = service->config.max_batch_sequence_tokens;
    // Overflow waves leave the head of the queue (the devices' next waves)
    // alone and take from the backlog behind it.
    size_t ahead = 0;
    offsets[0] = 0;
    while (entry && batch_size < max_requests) {
        cache_entry *next = entry->queue_next;
        bool member = overflow ? in_backlog(service, ahead) && overflow_eligible(service, entry)
                               : is_short(service, entry) == (wave_class == WAVE_SHORT);
        ahead += entry->n_tokens;
        bool fits = member &&
            (batch_size == 0 ||
             (batch[0]->n_tokens <= seq_limit && entry->n_tokens <= seq_limit &&
              total + entry->n_tokens <= budget));
        if (fits) {
            if (previous) previous->queue_next = next;
            else service->queue_head = next;
            if (service->queue_tail == entry) service->queue_tail = previous;
            entry->queue_next = NULL;
            service->queued_requests--;
            batch[batch_size++] = entry;
            total += entry->n_tokens;
            offsets[batch_size] = total;
            if (batch[0]->n_tokens > seq_limit) break; /* oversized runs alone */
        } else {
            previous = entry;
        }
        entry = next;
    }
    *total_tokens = total;
    return batch_size;
}

static void claim_slot_locked(ei_inference_service *service, size_t slot) {
    if (slot >= service->slots || (service->slot_mask & (1u << slot)))
        ei_die("inference service claimed busy slot %zu of %zu", slot, service->slots);
    service->slot_mask |= 1u << slot;
    service->busy_slots++;
}

static void release_slot_locked(ei_inference_service *service, size_t slot) {
    service->slot_mask &= ~(1u << slot);
    service->long_mask &= ~(1u << slot);
    service->busy_slots--;
    // Another backend thread may be waiting for a free slot.
    if (service->slots > 1) pthread_cond_broadcast(&service->queue_ready);
}

static bool run_execute(ei_inference_service *service, size_t slot, const int32_t *ids,
                        const size_t *offsets, size_t batch_size, float *out, char *err,
                        size_t err_len) {
    if (service->config.execute_slot)
        return service->config.execute_slot(service->execute_opaque, slot, ids, offsets,
                                            batch_size, out, err, err_len);
    return service->execute(service->execute_opaque, ids, offsets, batch_size, out, err,
                            err_len);
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

static int longer_first(const void *a, const void *b) {
    const cache_entry *x = *(cache_entry *const *)a, *y = *(cache_entry *const *)b;
    return (x->n_tokens < y->n_tokens) - (x->n_tokens > y->n_tokens);
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

static size_t expected_wave(const ei_inference_service *service);

static bool should_collect_batch(const ei_inference_service *service) {
    if (service->config.batch_wait_us == 0) return false;
    if (service->arriving_requests && service->arrival_wait_ns) return true;
    if (!service->adaptive_batch_wait) return true;
    if (service->queue_head && service->queue_head->queue_next) return true;
    if (service->batch_wave &&
        service->active_requests > service->queued_requests &&
        expected_wave(service) > service->queued_requests) {
        return true;
    }
    return monotonic_ns() < service->batch_hot_until_ns;
}

static size_t batch_lookahead_limit(size_t max_requests) {
    return max_requests <= SIZE_MAX / 8u ? max_requests * 8u : SIZE_MAX;
}

static size_t expected_wave(const ei_inference_service *service) {
    size_t wave = service->active_peak;
    return wave < service->config.max_batch_requests
        ? wave : service->config.max_batch_requests;
}

static void note_active_request_locked(ei_inference_service *service,
                                       size_t count) {
    service->active_requests += count;
    if (service->active_requests > service->active_peak) {
        service->active_peak = service->active_requests;
    }
}

static bool batch_collection_ready(const ei_inference_service *service, size_t budget) {
    const cache_entry *entry = service->queue_head;
    if (!entry) return false;
    // A capped slot's wave is ready once the queue fills its budget.
    if (budget < service->config.max_batch_tokens) {
        size_t tokens = 0;
        for (const cache_entry *e = entry; e; e = e->queue_next)
            if ((tokens += e->n_tokens) >= budget) return true;
    }
    if (entry->n_tokens > service->config.max_batch_sequence_tokens) return true;
    if (service->batch_wave) {
        size_t wave = expected_wave(service);
        if (wave > 0 && service->queued_requests >= wave) return true;
    }

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
                tokens + entry->n_tokens <= budget) {
                requests++;
                tokens += entry->n_tokens;
                if (requests == service->config.max_batch_requests ||
                    tokens == budget) {
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
             tokens + entry->n_tokens > budget)) {
            return true;
        }
        requests++;
        tokens += entry->n_tokens;
        if (requests == service->config.max_batch_requests ||
            tokens == budget) {
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
    uint64_t last_end = 0;
    size_t max_requests = service->config.max_batch_requests;
    cache_entry **batch = ei_xmalloc(sizeof(*batch) * max_requests);
    size_t *offsets = ei_xmalloc(sizeof(*offsets) * (max_requests + 1));
    size_t capacity = service->config.max_batch_tokens;
    if (capacity < context_limit(service->tokenizer)) capacity = context_limit(service->tokenizer);
    int32_t *ids = ei_xmalloc(sizeof(*ids) * capacity);
    float *output = ei_xmalloc(sizeof(*output) * max_requests * EI_N_EMBD);

    pthread_mutex_lock(&service->mutex);
    for (;;) {
        // Busy slots also cover solo inline executions on worker threads;
        // dispatching onto a busy slot would race on its engine context.
        // Releasing a slot always signals queue_ready, so waiting on a full
        // backend is safe even during shutdown.
        while ((!service->queue_head && !service->stopping) ||
               (service->queue_head && !can_start_wave(service))) {
            pthread_cond_wait(&service->queue_ready, &service->mutex);
        }
        if (!service->queue_head) break;
        // Short-only bypass and overflow waves start at once with the requests
        // already queued; a wave on a device collects as before.
        size_t slot = 0;
        int wave = pick_slot_locked(service, &slot);

        if (wave == WAVE_ANY && should_collect_batch(service)) {
            const uint64_t wait_ns =
                (uint64_t)service->config.batch_wait_us * 1000ull;
            const uint64_t start_ns = monotonic_ns();
            const uint64_t cap_ns = start_ns + service->batch_wait_cap_ns;
            const uint64_t arrival_ns = start_ns + service->arrival_wait_ns;
            uint64_t deadline_ns = start_ns + wait_ns;
            size_t seen_queued = service->queued_requests;
            while (!service->stopping &&
                   !batch_collection_ready(service, wave_budget(service, slot))) {
                uint64_t now = monotonic_ns();
                if (service->queued_requests > seen_queued) {
                    seen_queued = service->queued_requests;
                    uint64_t extended = now + wait_ns;
                    if (extended > cap_ns) extended = cap_ns;
                    if (extended > deadline_ns) deadline_ns = extended;
                }
                if (service->arriving_requests && service->arrival_wait_ns) {
                    // Accepted requests are still tokenizing: hold the wave
                    // for them, one wait at a time, within the arrival bound.
                    uint64_t extended = now + wait_ns;
                    if (extended > arrival_ns) extended = arrival_ns;
                    if (extended > deadline_ns) deadline_ns = extended;
                }
                if (now >= deadline_ns) break;
                struct timespec deadline;
                collection_deadline(
                    (uint32_t)((deadline_ns - now) / 1000ull) + 1u, &deadline);
                int rc = pthread_cond_timedwait(
                    &service->queue_ready, &service->mutex, &deadline);
                if (rc != 0 && rc != ETIMEDOUT) {
                    ei_die("batch collection wait failed: %s", strerror(rc));
                }
            }
        }

        // With several backend threads, others (and solo inline runs) may have
        // taken the queue or every slot while this one waited for the wave.
        wave = pick_slot_locked(service, &slot);
        if (wave == WAVE_NONE) continue;
        const size_t budget = wave_budget(service, slot);
        size_t batch_size = 0;
        size_t total_tokens = 0;
        offsets[0] = 0;
        bool wave_long = false;
        bool taken = wave == WAVE_OVERFLOW || bypass_mode(service);
        if (taken) {
            // Homogeneous waves: short-only while bypassing, otherwise the
            // class of the queue head; overflow waves take small requests.
            int wave_class = wave;
            if (wave == WAVE_ANY)
                wave_class = bypass_mode(service) && !is_short(service, service->queue_head)
                                 ? WAVE_ANY : WAVE_SHORT;
            batch_size = take_class_locked(service, wave_class, budget, batch, offsets,
                                           max_requests, &total_tokens);
            wave_long = wave_class == WAVE_ANY;
            if (batch_size == 0) {
                // pick_slot_locked promised a member. Never retry while
                // holding the mutex: finishing waves need it to release slots.
                struct timespec deadline;
                collection_deadline(1000, &deadline);
                pthread_cond_timedwait(&service->queue_ready, &service->mutex, &deadline);
                continue;
            }
        }
        while (!taken && !service->batch_lookahead && service->queue_head &&
               batch_size < max_requests) {
            cache_entry *entry = service->queue_head;
            if (batch_size > 0) {
                if (batch[0]->n_tokens >
                        service->config.max_batch_sequence_tokens ||
                    entry->n_tokens >
                        service->config.max_batch_sequence_tokens ||
                    total_tokens + entry->n_tokens > budget) {
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
        if (!taken && service->batch_lookahead) {
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
                        total_tokens + entry->n_tokens <= budget) {
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

        // Longest sequences first: packed attention tiles then start aligned
        // to the long sequences, 2-4% on mixed waves in-process (2026-10-11).
        // Output slots follow the batch order, so the wave's results are unaffected.
        if (batch_size > 1 && service->sort_waves) {
            qsort(batch, batch_size, sizeof *batch, longer_first);
            for (size_t i = 0; i < batch_size; i++)
                offsets[i + 1] = offsets[i] + batch[i]->n_tokens;
        }
        for (size_t i = 0; i < batch_size; i++) {
            memcpy(ids + offsets[i], batch[i]->ids,
                   batch[i]->n_tokens * sizeof(*ids));
        }
        claim_slot_locked(service, slot);
        if (wave_long) service->long_mask |= 1u << slot;
        pthread_mutex_unlock(&service->mutex);

        // Diagnostic: EI_WAVE_LOG=1 prints each wave and the backend idle gap
        // (time this thread spent since its previous wave finished).
        static int wave_log = -1;
        if (wave_log < 0) wave_log = getenv("EI_WAVE_LOG") != NULL;
        uint64_t exec_start = wave_log ? monotonic_ns() : 0;
        char error[256];
        bool ok = run_execute(service, slot, ids, offsets, batch_size, output, error,
                              sizeof error);
        if (wave_log) {
            uint64_t exec_end = monotonic_ns();
            fprintf(stderr, "wave slot=%zu seqs=%zu tokens=%zu exec=%.3fms gap=%.3fms busy=%zu\n",
                    slot, batch_size, total_tokens, (exec_end - exec_start) / 1e6,
                    last_end ? (exec_start - last_end) / 1e6 : 0.0, service->busy_slots);
            last_end = exec_end;
        }

        if (ok) {
            for (size_t i = 0; i < batch_size; i++) {
                memcpy(batch[i]->embedding, output + i * EI_N_EMBD,
                       sizeof(batch[i]->embedding));
            }
        }
        pthread_mutex_lock(&service->mutex);
        release_slot_locked(service, slot);
        service->active_peak = service->active_requests;
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

/* On-disk exact-cache format (little-endian host layout; guarded by the model
 * fingerprint so a file from a different model or build is ignored, never
 * served). Header: magic[8] "EIEMBED1", u32 version, u32 n_embd, u64
 * fingerprint, u64 entry_count. Then entry_count records of: u64 hash, u32
 * n_tokens, i32 ids[n_tokens], f32 embedding[EI_N_EMBD]. */
static const char EI_CACHE_MAGIC[8] = {'E', 'I', 'E', 'M', 'B', 'E', 'D', '1'};
#define EI_CACHE_VERSION 1u
#define EI_CACHE_COUNT_OFFSET 24 /* magic(8)+version(4)+n_embd(4)+fingerprint(8) */

static void load_persistent_cache(ei_inference_service *service) {
    const char *path = service->config.cache_path;
    if (!path || !*path) return;
    FILE *file = fopen(path, "rb");
    if (!file) return;
    char magic[8];
    uint32_t version = 0, n_embd = 0;
    uint64_t fingerprint = 0, count = 0;
    if (fread(magic, 1, sizeof magic, file) != sizeof magic ||
        memcmp(magic, EI_CACHE_MAGIC, sizeof magic) != 0 ||
        fread(&version, sizeof version, 1, file) != 1 ||
        version != EI_CACHE_VERSION ||
        fread(&n_embd, sizeof n_embd, 1, file) != 1 || n_embd != EI_N_EMBD ||
        fread(&fingerprint, sizeof fingerprint, 1, file) != 1 ||
        fingerprint != service->config.cache_fingerprint ||
        fread(&count, sizeof count, 1, file) != 1) {
        fclose(file);
        return; /* absent/stale/other-model: ignore, it will be overwritten */
    }
    size_t loaded = 0;
    for (uint64_t i = 0; i < count &&
             service->ready_entries < service->config.cache_entries; i++) {
        uint64_t hash = 0;
        uint32_t n_tokens = 0;
        if (fread(&hash, sizeof hash, 1, file) != 1 ||
            fread(&n_tokens, sizeof n_tokens, 1, file) != 1 || n_tokens == 0 ||
            n_tokens > context_limit(service->tokenizer)) {
            break; /* truncated or corrupt: keep what loaded cleanly */
        }
        cache_entry *entry = ei_xcalloc(1, sizeof(*entry));
        entry->hash = hash;
        entry->n_tokens = n_tokens;
        entry->ids = ei_xmalloc(n_tokens * sizeof(*entry->ids));
        entry->state = EI_ENTRY_READY;
        if (fread(entry->ids, sizeof(*entry->ids), n_tokens, file) != n_tokens ||
            fread(entry->embedding, sizeof(float), EI_N_EMBD, file) != EI_N_EMBD) {
            free(entry->ids);
            free(entry);
            break;
        }
        if (pthread_cond_init(&entry->done, NULL) != 0) {
            ei_die("failed to initialize inference completion");
        }
        cache_entry **slot =
            &service->buckets[hash & (service->bucket_count - 1)];
        entry->hash_next = *slot;
        *slot = entry;
        lru_push_front(service, entry);
        service->ready_entries++;
        loaded++;
    }
    fclose(file);
    if (loaded) {
        fprintf(stderr, "loaded %zu cached embeddings from %s\n", loaded, path);
    }
}

/* Caller guarantees the bucket table is not mutated concurrently (holds the
 * mutex, or runs single-threaded). Best-effort and atomic via a temp file +
 * rename; any failure is silent. */
static void write_persistent_cache_locked(ei_inference_service *service) {
    const char *path = service->config.cache_path;
    if (!path || !*path) return;
    size_t path_len = strlen(path);
    char *tmp = ei_xmalloc(path_len + 8);
    memcpy(tmp, path, path_len);
    memcpy(tmp + path_len, ".XXXXXX", 8);
    int fd = mkstemp(tmp);
    if (fd < 0) {
        free(tmp);
        return;
    }
    FILE *file = fdopen(fd, "wb");
    if (!file) {
        close(fd);
        unlink(tmp);
        free(tmp);
        return;
    }
    uint32_t version = EI_CACHE_VERSION, n_embd = EI_N_EMBD;
    uint64_t fingerprint = service->config.cache_fingerprint, count = 0;
    bool ok =
        fwrite(EI_CACHE_MAGIC, 1, sizeof EI_CACHE_MAGIC, file) == sizeof EI_CACHE_MAGIC &&
        fwrite(&version, sizeof version, 1, file) == 1 &&
        fwrite(&n_embd, sizeof n_embd, 1, file) == 1 &&
        fwrite(&fingerprint, sizeof fingerprint, 1, file) == 1 &&
        fwrite(&count, sizeof count, 1, file) == 1;
    for (size_t bucket = 0; ok && bucket < service->bucket_count; bucket++) {
        for (cache_entry *entry = service->buckets[bucket];
             ok && entry; entry = entry->hash_next) {
            if (entry->state != EI_ENTRY_READY) continue;
            uint32_t n_tokens = (uint32_t)entry->n_tokens;
            ok = fwrite(&entry->hash, sizeof entry->hash, 1, file) == 1 &&
                 fwrite(&n_tokens, sizeof n_tokens, 1, file) == 1 &&
                 fwrite(entry->ids, sizeof(*entry->ids), entry->n_tokens, file)
                     == entry->n_tokens &&
                 fwrite(entry->embedding, sizeof(float), EI_N_EMBD, file)
                     == (size_t)EI_N_EMBD;
            if (ok) count++;
        }
    }
    if (ok) {
        ok = fseek(file, EI_CACHE_COUNT_OFFSET, SEEK_SET) == 0 &&
             fwrite(&count, sizeof count, 1, file) == 1;
    }
    if (fclose(file) != 0) ok = false;
    if (ok && rename(tmp, path) == 0) {
        fprintf(stderr, "persisted %llu cached embeddings to %s\n",
                (unsigned long long)count, path);
    } else {
        unlink(tmp);
    }
    free(tmp);
}

void ei_inference_service_dump_cache(ei_inference_service *service) {
    if (!service || !service->config.cache_path || !*service->config.cache_path) {
        return;
    }
    pthread_mutex_lock(&service->mutex);
    write_persistent_cache_locked(service);
    pthread_mutex_unlock(&service->mutex);
}

ei_inference_service *ei_inference_service_create(
    const ei_tokenizer *tokenizer, ei_inference_execute_fn execute,
    void *execute_opaque, const ei_inference_service_config *config) {
    if (!execute || !config || config->max_batch_tokens < EI_N_CTX ||
        config->max_batch_requests == 0 || config->max_batch_sequence_tokens == 0 ||
        config->max_batch_sequence_tokens > context_limit(tokenizer) ||
        (config->tokenizer_workers > 0 && !tokenizer) ||
        config->pipeline_slots > 32 || (config->pipeline_slots > 1 && !config->execute_slot)) {
        return NULL;
    }
    // Every executor layout needs a device slot: long requests never overflow.
    size_t slot_count = config->pipeline_slots > 1 ? config->pipeline_slots : 1;
    bool device_slot = false;
    for (size_t i = 0; i < slot_count; i++) {
        if (config->slot_group && config->slot_group[i] >= 32) return NULL;
        if (!config->slot_overflow || !config->slot_overflow[i]) device_slot = true;
    }
    if (!device_slot) return NULL;
    ei_inference_service *service = ei_xcalloc(1, sizeof(*service));
    service->tokenizer = tokenizer;
    service->execute = execute;
    service->execute_opaque = execute_opaque;
    service->config = *config;
    const char *lookahead = getenv("EI_BATCH_LOOKAHEAD");
    service->batch_lookahead = !lookahead || strcmp(lookahead, "0") != 0;
    const char *sort_waves = getenv("EI_BATCH_SORT");
    service->sort_waves = !sort_waves || strcmp(sort_waves, "0") != 0;
    const char *adaptive_wait = getenv("EI_ADAPTIVE_BATCH_WAIT");
    service->adaptive_batch_wait = !adaptive_wait ||
        strcmp(adaptive_wait, "0") != 0;
    const char *wave = getenv("EI_BATCH_WAVE");
    service->batch_wave = !wave || strcmp(wave, "0") != 0;
    const char *solo = getenv("EI_SOLO_INLINE");
    service->solo_inline = !solo || strcmp(solo, "0") != 0;
    const char *wait_cap = getenv("EI_BATCH_WAIT_MAX_US");
    uint64_t cap_us = wait_cap ? strtoull(wait_cap, NULL, 10)
                               : (uint64_t)config->batch_wait_us * 4ull;
    if (cap_us < config->batch_wait_us) cap_us = config->batch_wait_us;
    service->batch_wait_cap_ns = cap_us * 1000ull;
    const char *arrival_wait = getenv("EI_BATCH_ARRIVAL_WAIT_US");
    service->arrival_wait_ns =
        (arrival_wait ? strtoull(arrival_wait, NULL, 10) : 5000ull) * 1000ull;
    uint64_t wait_ns = (uint64_t)config->batch_wait_us * 1000ull;
    service->batch_hot_window_ns = wait_ns > 100000ull
        ? wait_ns * 10ull : 1000000ull;
    size_t desired_buckets = config->cache_entries > 32
        ? config->cache_entries * 2 : 64;
    service->bucket_count = next_power_of_two(desired_buckets);
    service->buckets = ei_xcalloc(service->bucket_count, sizeof(*service->buckets));
    load_persistent_cache(service);
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
    service->slots = config->pipeline_slots > 1 ? config->pipeline_slots : 1;
    for (size_t i = 0; i < service->slots; i++) {
        if (config->slot_overflow && config->slot_overflow[i]) {
            service->overflow_mask |= 1u << i;
            continue;
        }
        size_t g = config->slot_group ? config->slot_group[i] : 0;
        service->group_mask[g] |= 1u << i;
        if (g + 1 > service->groups) service->groups = g + 1;
    }
    service->backend_threads = ei_xmalloc(service->slots * sizeof(*service->backend_threads));
    for (size_t i = 0; i < service->slots; i++)
        if (pthread_create(&service->backend_threads[i], NULL, backend_main, service) != 0)
            ei_die("failed to create inference backend thread %zu", i);
    return service;
}

void ei_inference_service_free(ei_inference_service *service) {
    if (!service) return;
    pthread_mutex_lock(&service->mutex);
    service->stopping = true;
    pthread_cond_broadcast(&service->queue_ready);
    pthread_mutex_unlock(&service->mutex);
    for (size_t i = 0; i < service->slots; i++)
        pthread_join(service->backend_threads[i], NULL);
    free(service->backend_threads);

    pthread_mutex_lock(&service->tokenizer_mutex);
    service->tokenizer_stopping = true;
    pthread_cond_broadcast(&service->tokenizer_ready);
    pthread_mutex_unlock(&service->tokenizer_mutex);
    for (size_t i = 0; i < service->tokenizer_thread_count; i++) {
        pthread_join(service->tokenizer_threads[i], NULL);
    }

    /* Threads are joined, so the bucket table is now single-threaded. */
    write_persistent_cache_locked(service);
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
            if (service->busy_slots || service->queue_head) {
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
    if (service->queue_head || service->busy_slots) mark_batch_hot(service);
    if (service->queue_tail) service->queue_tail->queue_next = entry;
    else service->queue_head = entry;
    service->queue_tail = entry;
    service->queued_requests++;
    service->stats.cache_misses++;
    return entry;
}

static bool try_execute_solo_locked(ei_inference_service *service,
                                    cache_entry *entry) {
    if (!service->solo_inline) return false;
    // A solo request obeys the wave rules: a device slot (short-only beside
    // long waves in bypass mode); overflow slots are left to the backend.
    size_t slot = 0;
    int wave = pick_slot_locked(service, &slot);
    if (wave == WAVE_NONE || wave == WAVE_OVERFLOW ||
        (wave == WAVE_SHORT && !is_short(service, entry)))
        return false;
    if (entry->state != EI_ENTRY_PENDING) return false;
    if (service->queue_head != entry || entry->queue_next) return false;
    if (service->queued_requests != 1) return false;
    if (expected_wave(service) > 1) return false;
    if (service->arriving_requests && service->arrival_wait_ns) return false;
    service->queue_head = NULL;
    service->queue_tail = NULL;
    service->queued_requests = 0;
    claim_slot_locked(service, slot);
    if (bypass_mode(service) && !is_short(service, entry)) service->long_mask |= 1u << slot;
    pthread_mutex_unlock(&service->mutex);

    size_t offsets[2] = {0, entry->n_tokens};
    float output[EI_N_EMBD];
    char error[256];
    bool ok = run_execute(service, slot, entry->ids, offsets, 1, output, error, sizeof error);

    pthread_mutex_lock(&service->mutex);
    release_slot_locked(service, slot);
    service->active_peak = service->active_requests;
    service->stats.batches++;
    service->stats.sequences++;
    service->stats.tokens += entry->n_tokens;
    if (ok) {
        memcpy(entry->embedding, output, sizeof(entry->embedding));
        entry->state = EI_ENTRY_READY;
        lru_push_front(service, entry);
        service->ready_entries++;
    } else {
        entry->state = EI_ENTRY_FAILED;
        snprintf(entry->error, sizeof entry->error, "%s", error);
    }
    pthread_cond_broadcast(&entry->done);
    // The backend waits out busy slots (including during shutdown); wake
    // it so it can drain anything that queued during the inline run.
    pthread_cond_signal(&service->queue_ready);
    prune_cache(service);
    return true;
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
    if (!service || !ids || n_tokens == 0 || n_tokens > context_limit(service->tokenizer)) {
        set_error(err, err_len, "input token count exceeds the selected profile context limit");
        return false;
    }
    pthread_mutex_lock(&service->mutex);
    note_active_request_locked(service, 1);
    cache_entry *entry = submit_tokens_locked(service, ids, n_tokens);
    if (!try_execute_solo_locked(service, entry)) {
        pthread_cond_signal(&service->queue_ready);
    }
    bool ok = await_entry_locked(service, entry, out, err, err_len);
    if (ok && err && err_len) err[0] = '\0';
    service->active_requests--;
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
    truncate_tokens(service, &tokens);
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
    return ei_inference_service_embed_batch_with_usage(
        service, texts, text_lengths, batch_size, out, NULL, err, err_len);
}

bool ei_inference_service_embed_batch_with_usage(
    ei_inference_service *service, const char *const *texts,
    const size_t *text_lengths, size_t batch_size, float *out,
    size_t *prompt_tokens, char *err, size_t err_len) {
    if (!service || !service->tokenizer || !texts || !text_lengths ||
        batch_size == 0) {
        set_error(err, err_len, "embedding batch must contain at least one input");
        return false;
    }
    ei_tokens *tokens = ei_xcalloc(batch_size, sizeof(*tokens));
    cache_entry **entries = ei_xmalloc(batch_size * sizeof(*entries));
    pthread_mutex_lock(&service->mutex);
    service->arriving_requests += batch_size;
    pthread_mutex_unlock(&service->mutex);
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
        truncate_tokens(service, &tokens[i]);
        if (tokens[i].n == 0) {
            for (size_t j = 0; j < batch_size; j++) ei_tokens_free(&tokens[j]);
            free(entries);
            free(tokens);
            pthread_mutex_lock(&service->mutex);
            service->arriving_requests -= batch_size;
            pthread_cond_broadcast(&service->queue_ready);
            pthread_mutex_unlock(&service->mutex);
            set_error(err, err_len, "input produced no tokens");
            return false;
        }
    }
    if (prompt_tokens) {
        *prompt_tokens = 0;
        for (size_t i = 0; i < batch_size; i++) {
            *prompt_tokens += tokens[i].n;
        }
    }

    pthread_mutex_lock(&service->mutex);
    service->arriving_requests -= batch_size;
    note_active_request_locked(service, batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        entries[i] = submit_tokens_locked(service, tokens[i].ids, tokens[i].n);
    }
    if (batch_size != 1 || !try_execute_solo_locked(service, entries[0])) {
        pthread_cond_signal(&service->queue_ready);
    }
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
    service->active_requests -= batch_size;
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
