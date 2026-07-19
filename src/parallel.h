#ifndef EI_PARALLEL_H
#define EI_PARALLEL_H

#include "common.h"

typedef struct ei_thread_pool ei_thread_pool;
typedef void (*ei_parallel_fn)(void *context, int32_t begin, int32_t end);

ei_thread_pool *ei_thread_pool_create(int32_t requested_threads);
void ei_thread_pool_destroy(ei_thread_pool *pool);
int32_t ei_thread_pool_threads(const ei_thread_pool *pool);
void ei_parallel_for(ei_thread_pool *pool, int32_t count, int32_t grain,
                     ei_parallel_fn function, void *context);

#endif /* EI_PARALLEL_H */
