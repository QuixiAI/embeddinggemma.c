#if defined(__linux__)
#define _POSIX_C_SOURCE 200809L /* clock_gettime under -std=c11 on glibc */
#endif
#include "parallel.h"

#include <pthread.h>
#include <stdatomic.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    struct ei_thread_pool *pool;
    pthread_cond_t start;
    bool assigned;
    int32_t index;
    bool sleeping; /* spin mode: parked on start (guarded by the pool mutex) */
} worker_context;

struct ei_thread_pool {
    pthread_t *workers;
    worker_context *contexts;
    int32_t worker_count;
    int32_t total_threads;

    pthread_mutex_t mutex;
    pthread_cond_t done;
    bool stop;
    int32_t active_workers;

    _Atomic int32_t next;
    int32_t count;
    int32_t grain;
    ei_parallel_fn function;
    void *context;

    /* Spin mode (EI_POOL_SPIN_US > 0): workers poll a generation counter for
     * spin_ns after each job before parking, so back-to-back calls skip the
     * condition-variable wake. Off by default; the mutex/condvar path above is
     * unchanged when off. */
    uint64_t spin_ns;
    /* generation << 32 | participants, published together so a worker never
     * pairs one call's generation with another call's participant count. */
    _Atomic uint64_t ticket;
    _Atomic int32_t pending;
};

static uint64_t pool_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void run_jobs(ei_thread_pool *pool) {
    for (;;) {
        int32_t begin = atomic_fetch_add_explicit(&pool->next, pool->grain,
                                                   memory_order_relaxed);
        if (begin >= pool->count) return;
        int32_t end = begin + pool->grain;
        if (end > pool->count) end = pool->count;
        pool->function(pool->context, begin, end);
    }
}

static void *spin_worker_main(worker_context *worker) {
    ei_thread_pool *pool = worker->pool;
    uint64_t seen = 0; /* tickets start at generation 0; a late worker still sees call 1 */
    for (;;) {
        uint64_t now = atomic_load_explicit(&pool->ticket, memory_order_acquire);
        if (now >> 32 == seen >> 32) {
            uint64_t deadline = pool_now_ns() + pool->spin_ns;
            while ((now = atomic_load_explicit(&pool->ticket, memory_order_acquire)) >> 32 ==
                       seen >> 32 &&
                   pool_now_ns() < deadline) {
            }
        }
        if (now >> 32 == seen >> 32) {
            pthread_mutex_lock(&pool->mutex);
            worker->sleeping = true;
            while (!pool->stop &&
                   (now = atomic_load_explicit(&pool->ticket, memory_order_acquire)) >> 32 ==
                       seen >> 32)
                pthread_cond_wait(&worker->start, &pool->mutex);
            worker->sleeping = false;
            bool stop = pool->stop;
            pthread_mutex_unlock(&pool->mutex);
            if (stop) return NULL;
        }
        seen = now;
        if (pool->stop) return NULL;
        /* A participant's call cannot complete (or be replaced) without it. */
        if (worker->index < (int32_t)(uint32_t)now) {
            run_jobs(pool);
            atomic_fetch_sub_explicit(&pool->pending, 1, memory_order_acq_rel);
        }
    }
}

static void *worker_main(void *opaque) {
    worker_context *worker = opaque;
    ei_thread_pool *pool = worker->pool;
    if (pool->spin_ns) return spin_worker_main(worker);
    pthread_mutex_lock(&pool->mutex);
    for (;;) {
        while (!pool->stop && !worker->assigned) {
            pthread_cond_wait(&worker->start, &pool->mutex);
        }
        if (pool->stop) break;
        worker->assigned = false;
        pthread_mutex_unlock(&pool->mutex);
        run_jobs(pool);
        pthread_mutex_lock(&pool->mutex);
        pool->active_workers--;
        if (pool->active_workers == 0) pthread_cond_signal(&pool->done);
    }
    pthread_mutex_unlock(&pool->mutex);
    return NULL;
}

static int32_t default_thread_count(void) {
    const char *value = getenv("EI_THREADS");
    if (value && *value) {
        char *end = NULL;
        long parsed = strtol(value, &end, 10);
        if (*end != '\0' || parsed < 1 || parsed > 64) {
            ei_die("EI_THREADS must be an integer from 1 to 64");
        }
        return (int32_t)parsed;
    }
    long available = sysconf(_SC_NPROCESSORS_ONLN);
    if (available < 1) available = 1;
    if (available > 32) available = 32;
    return (int32_t)available;
}

static uint32_t create_spin_us;
ei_thread_pool *ei_thread_pool_create_spin(int32_t requested_threads, uint32_t default_spin_us) {
    create_spin_us = default_spin_us;
    ei_thread_pool *pool = ei_thread_pool_create(requested_threads);
    create_spin_us = 0;
    return pool;
}

ei_thread_pool *ei_thread_pool_create(int32_t requested_threads) {
    int32_t total = requested_threads > 0 ? requested_threads : default_thread_count();
    if (total < 1) total = 1;
    ei_thread_pool *pool = calloc(1, sizeof *pool);
    if (!pool) ei_die("out of memory creating CPU thread pool");
    pool->total_threads = total;
    pool->worker_count = total - 1;
    if (pthread_mutex_init(&pool->mutex, NULL) != 0 ||
        pthread_cond_init(&pool->done, NULL) != 0) {
        ei_die("failed to initialize CPU thread pool");
    }
    if (pool->worker_count == 0) return pool;

    const char *spin = getenv("EI_POOL_SPIN_US");
    pool->spin_ns = (spin && *spin ? strtoull(spin, NULL, 10) : create_spin_us) * 1000ull;
    pool->workers = calloc((size_t)pool->worker_count, sizeof *pool->workers);
    pool->contexts = calloc((size_t)pool->worker_count, sizeof *pool->contexts);
    if (!pool->workers || !pool->contexts) ei_die("out of memory creating CPU workers");
    for (int32_t i = 0; i < pool->worker_count; i++) {
        pool->contexts[i].pool = pool;
        pool->contexts[i].index = i;
        if (pthread_cond_init(&pool->contexts[i].start, NULL) != 0) {
            ei_die("failed to initialize CPU worker %d", i);
        }
        if (pthread_create(&pool->workers[i], NULL, worker_main, &pool->contexts[i]) != 0) {
            ei_die("failed to create CPU worker %d", i);
        }
    }
    return pool;
}

void ei_thread_pool_destroy(ei_thread_pool *pool) {
    if (!pool) return;
    pthread_mutex_lock(&pool->mutex);
    pool->stop = true;
    atomic_fetch_add_explicit(&pool->ticket, 1ull << 32, memory_order_release);
    for (int32_t i = 0; i < pool->worker_count; i++) {
        pool->contexts[i].assigned = true;
        pthread_cond_signal(&pool->contexts[i].start);
    }
    pthread_mutex_unlock(&pool->mutex);
    for (int32_t i = 0; i < pool->worker_count; i++) {
        pthread_join(pool->workers[i], NULL);
        pthread_cond_destroy(&pool->contexts[i].start);
    }
    pthread_cond_destroy(&pool->done);
    pthread_mutex_destroy(&pool->mutex);
    free(pool->contexts);
    free(pool->workers);
    free(pool);
}

int32_t ei_thread_pool_threads(const ei_thread_pool *pool) {
    return pool ? pool->total_threads : 1;
}

void ei_parallel_for(ei_thread_pool *pool, int32_t count, int32_t grain,
                     ei_parallel_fn function, void *context) {
    if (count <= 0) return;
    if (!pool || pool->worker_count == 0 || count <= grain) {
        function(context, 0, count);
        return;
    }
    if (grain < 1) grain = 1;

    if (pool->spin_ns) {
        int32_t jobs = (count + grain - 1) / grain, helpers = jobs - 1;
        if (helpers > pool->worker_count) helpers = pool->worker_count;
        pool->count = count;
        pool->grain = grain;
        pool->function = function;
        pool->context = context;
        atomic_store_explicit(&pool->next, 0, memory_order_relaxed);
        atomic_store_explicit(&pool->pending, helpers, memory_order_relaxed);
        pthread_mutex_lock(&pool->mutex);
        uint64_t generation = (atomic_load_explicit(&pool->ticket, memory_order_relaxed) >> 32) + 1;
        atomic_store_explicit(&pool->ticket, generation << 32 | (uint32_t)helpers,
                              memory_order_release);
        for (int32_t i = 0; i < helpers; i++)
            if (pool->contexts[i].sleeping) pthread_cond_signal(&pool->contexts[i].start);
        pthread_mutex_unlock(&pool->mutex);
        run_jobs(pool);
        while (atomic_load_explicit(&pool->pending, memory_order_acquire) != 0) {
        }
        return;
    }

    pthread_mutex_lock(&pool->mutex);
    pool->count = count;
    pool->grain = grain;
    pool->function = function;
    pool->context = context;
    int32_t jobs = (count + grain - 1) / grain;
    pool->active_workers = jobs - 1;
    if (pool->active_workers > pool->worker_count) {
        pool->active_workers = pool->worker_count;
    }
    atomic_store_explicit(&pool->next, 0, memory_order_relaxed);
    for (int32_t i = 0; i < pool->active_workers; i++) {
        pool->contexts[i].assigned = true;
        pthread_cond_signal(&pool->contexts[i].start);
    }
    pthread_mutex_unlock(&pool->mutex);

    run_jobs(pool);

    pthread_mutex_lock(&pool->mutex);
    while (pool->active_workers != 0) pthread_cond_wait(&pool->done, &pool->mutex);
    pthread_mutex_unlock(&pool->mutex);
}
