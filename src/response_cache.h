#ifndef EI_RESPONSE_CACHE_H
#define EI_RESPONSE_CACHE_H

#include "common.h"

typedef struct ei_response_cache ei_response_cache;

typedef struct {
    const char *data;
    size_t len;
    void *entry;
} ei_response_cache_value;

ei_response_cache *ei_response_cache_create(size_t max_bytes,
                                            size_t bucket_count);
void ei_response_cache_free(ei_response_cache *cache);

bool ei_response_cache_acquire(ei_response_cache *cache,
                               const char *key, size_t key_len,
                               ei_response_cache_value *value);
void ei_response_cache_release(ei_response_cache *cache,
                               ei_response_cache_value *value);

void ei_response_cache_insert(ei_response_cache *cache,
                              const char *key, size_t key_len,
                              const char *value, size_t value_len);

#endif
