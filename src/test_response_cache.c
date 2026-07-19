#include "response_cache.h"

static void require_miss(ei_response_cache *cache,
                         const char *key, size_t key_len) {
    ei_response_cache_value value;
    if (ei_response_cache_acquire(cache, key, key_len, &value)) {
        ei_response_cache_release(cache, &value);
        ei_die("unexpected response cache hit");
    }
}

int main(void) {
    ei_response_cache *cache = ei_response_cache_create(200, 3);
    if (!cache) ei_die("failed to create response cache");

    const char key_a[] = { 'a', '\0', '1' };
    const char key_b[] = { 'b', '\0', '2' };
    char value_a[64];
    char value_b[64];
    memset(value_a, 'A', sizeof value_a);
    memset(value_b, 'B', sizeof value_b);

    require_miss(cache, key_a, sizeof key_a);
    ei_response_cache_insert(
        cache, key_a, sizeof key_a, value_a, sizeof value_a);
    ei_response_cache_value held;
    if (!ei_response_cache_acquire(cache, key_a, sizeof key_a, &held) ||
        held.len != sizeof value_a ||
        memcmp(held.data, value_a, sizeof value_a) != 0 ||
        held.data[held.len] != '\0') {
        ei_die("response cache returned an invalid value");
    }

    ei_response_cache_insert(
        cache, key_b, sizeof key_b, value_b, sizeof value_b);
    require_miss(cache, key_b, sizeof key_b);
    ei_response_cache_release(cache, &held);

    ei_response_cache_insert(
        cache, key_b, sizeof key_b, value_b, sizeof value_b);
    require_miss(cache, key_a, sizeof key_a);
    ei_response_cache_value value;
    if (!ei_response_cache_acquire(cache, key_b, sizeof key_b, &value) ||
        value.len != sizeof value_b ||
        memcmp(value.data, value_b, sizeof value_b) != 0) {
        ei_die("response cache LRU insertion failed");
    }
    ei_response_cache_release(cache, &value);
    ei_response_cache_free(cache);

    if (ei_response_cache_create(0, 64) != NULL) {
        ei_die("zero-byte response cache must be disabled");
    }
    return 0;
}
