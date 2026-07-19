#include "parallel.h"

#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>

typedef struct {
    struct ei_thread_pool *pool;
    pthread_cond_t start;
    bool assigned;
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
};

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

static void *worker_main(void *opaque) {
    worker_context *worker = opaque;
    ei_thread_pool *pool = worker->pool;
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

    pool->workers = calloc((size_t)pool->worker_count, sizeof *pool->workers);
    pool->contexts = calloc((size_t)pool->worker_count, sizeof *pool->contexts);
    if (!pool->workers || !pool->contexts) ei_die("out of memory creating CPU workers");
    for (int32_t i = 0; i < pool->worker_count; i++) {
        pool->contexts[i].pool = pool;
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
