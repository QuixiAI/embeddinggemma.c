#ifndef EI_PARALLEL_H
#define EI_PARALLEL_H

#include "common.h"

typedef struct ei_thread_pool ei_thread_pool;
typedef void (*ei_parallel_fn)(void *context, int32_t begin, int32_t end);

ei_thread_pool *ei_thread_pool_create(int32_t requested_threads);
/* Like ei_thread_pool_create with workers spinning default_spin_us after each
 * job before parking; EI_POOL_SPIN_US overrides (0 = the condvar-only pool). */
ei_thread_pool *ei_thread_pool_create_spin(int32_t requested_threads, uint32_t default_spin_us);
/* Spin pool whose threads are pinned: worker i to cpus[1 + i]; cpus[0] is for
 * the caller (see ei_thread_pool_bind_caller). Pinning is Linux-only. */
ei_thread_pool *ei_thread_pool_create_pinned(int32_t threads, uint32_t default_spin_us,
                                             const int32_t *cpus);
/* Pin the calling thread to a pinned pool's caller CPU for one call, then
 * restore its previous affinity. No-ops for unpinned pools. */
typedef struct {
    _Alignas(8) unsigned char mask[128];
    bool bound;
} ei_cpu_binding;
void ei_thread_pool_bind_caller(const ei_thread_pool *pool, ei_cpu_binding *saved);
void ei_thread_pool_unbind_caller(const ei_cpu_binding *saved);
/* Logical CPUs in placement order: one hardware thread per core first, grouped
 * by shared L3, then SMT siblings (Linux sysfs; identity elsewhere). */
int32_t ei_cpu_placement(int32_t *cpus, int32_t max);
void ei_thread_pool_destroy(ei_thread_pool *pool);
int32_t ei_thread_pool_threads(const ei_thread_pool *pool);
/* Online logical CPUs (at least 1). */
int32_t ei_cpu_count(void);
void ei_parallel_for(ei_thread_pool *pool, int32_t count, int32_t grain,
                     ei_parallel_fn function, void *context);

#endif /* EI_PARALLEL_H */
