#include "response_cache.h"

#include <pthread.h>

typedef struct response_cache_entry response_cache_entry;

struct response_cache_entry {
    uint64_t hash;
    size_t key_len;
    size_t value_len;
    size_t users;
    response_cache_entry *hash_next;
    response_cache_entry *lru_prev;
    response_cache_entry *lru_next;
    char data[];
};

struct ei_response_cache {
    size_t max_bytes;
    size_t used_bytes;
    size_t bucket_count;
    response_cache_entry **buckets;
    response_cache_entry *lru_head;
    response_cache_entry *lru_tail;
    pthread_mutex_t mutex;
};

static uint64_t hash_bytes(const char *data, size_t len) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < len; i++) {
        hash ^= (unsigned char)data[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static size_t next_power_of_two(size_t value) {
    size_t result = 1;
    while (result < value && result <= SIZE_MAX / 2u) result *= 2u;
    return result < value ? value : result;
}

static size_t entry_bytes(const response_cache_entry *entry) {
    return sizeof(*entry) + entry->key_len + entry->value_len + 1u;
}

static char *entry_value(response_cache_entry *entry) {
    return entry->data + entry->key_len;
}

static response_cache_entry **entry_slot(ei_response_cache *cache,
                                         uint64_t hash,
                                         const char *key, size_t key_len) {
    response_cache_entry **slot =
        &cache->buckets[hash & (cache->bucket_count - 1u)];
    while (*slot) {
        response_cache_entry *entry = *slot;
        if (entry->hash == hash && entry->key_len == key_len &&
            memcmp(entry->data, key, key_len) == 0) {
            break;
        }
        slot = &entry->hash_next;
    }
    return slot;
}

static void lru_remove(ei_response_cache *cache,
                       response_cache_entry *entry) {
    if (entry->lru_prev) entry->lru_prev->lru_next = entry->lru_next;
    else cache->lru_head = entry->lru_next;
    if (entry->lru_next) entry->lru_next->lru_prev = entry->lru_prev;
    else cache->lru_tail = entry->lru_prev;
    entry->lru_prev = NULL;
    entry->lru_next = NULL;
}

static void lru_push_front(ei_response_cache *cache,
                           response_cache_entry *entry) {
    if (cache->lru_head == entry) return;
    if (entry->lru_prev || entry->lru_next || cache->lru_tail == entry) {
        lru_remove(cache, entry);
    }
    entry->lru_prev = NULL;
    entry->lru_next = cache->lru_head;
    if (cache->lru_head) cache->lru_head->lru_prev = entry;
    else cache->lru_tail = entry;
    cache->lru_head = entry;
}

static void remove_entry(ei_response_cache *cache,
                         response_cache_entry *entry) {
    response_cache_entry **slot = entry_slot(
        cache, entry->hash, entry->data, entry->key_len);
    if (*slot == entry) *slot = entry->hash_next;
    lru_remove(cache, entry);
    cache->used_bytes -= entry_bytes(entry);
    free(entry);
}

ei_response_cache *ei_response_cache_create(size_t max_bytes,
                                            size_t bucket_count) {
    if (max_bytes == 0) return NULL;
    ei_response_cache *cache = ei_xcalloc(1, sizeof(*cache));
    cache->max_bytes = max_bytes;
    cache->bucket_count = next_power_of_two(bucket_count > 0 ? bucket_count : 64);
    cache->buckets = ei_xcalloc(cache->bucket_count, sizeof(*cache->buckets));
    if (pthread_mutex_init(&cache->mutex, NULL) != 0) {
        ei_die("failed to initialize HTTP response cache");
    }
    return cache;
}

void ei_response_cache_free(ei_response_cache *cache) {
    if (!cache) return;
    response_cache_entry *entry = cache->lru_head;
    while (entry) {
        response_cache_entry *next = entry->lru_next;
        free(entry);
        entry = next;
    }
    pthread_mutex_destroy(&cache->mutex);
    free(cache->buckets);
    free(cache);
}

bool ei_response_cache_acquire(ei_response_cache *cache,
                               const char *key, size_t key_len,
                               ei_response_cache_value *value) {
    if (!cache || !key || !value) return false;
    uint64_t hash = hash_bytes(key, key_len);
    pthread_mutex_lock(&cache->mutex);
    response_cache_entry *entry = *entry_slot(cache, hash, key, key_len);
    if (!entry) {
        pthread_mutex_unlock(&cache->mutex);
        return false;
    }
    entry->users++;
    lru_push_front(cache, entry);
    *value = (ei_response_cache_value){
        .data = entry_value(entry), .len = entry->value_len, .entry = entry,
    };
    pthread_mutex_unlock(&cache->mutex);
    return true;
}

void ei_response_cache_release(ei_response_cache *cache,
                               ei_response_cache_value *value) {
    if (!cache || !value || !value->entry) return;
    pthread_mutex_lock(&cache->mutex);
    response_cache_entry *entry = value->entry;
    entry->users--;
    pthread_mutex_unlock(&cache->mutex);
    *value = (ei_response_cache_value){0};
}

void ei_response_cache_insert(ei_response_cache *cache,
                              const char *key, size_t key_len,
                              const char *value, size_t value_len) {
    if (!cache || !key || !value ||
        key_len > SIZE_MAX - value_len - sizeof(response_cache_entry) - 1u) {
        return;
    }
    size_t bytes = sizeof(response_cache_entry) + key_len + value_len + 1u;
    if (bytes > cache->max_bytes) return;

    response_cache_entry *candidate = ei_xcalloc(1, bytes);
    candidate->hash = hash_bytes(key, key_len);
    candidate->key_len = key_len;
    candidate->value_len = value_len;
    memcpy(candidate->data, key, key_len);
    memcpy(entry_value(candidate), value, value_len);
    entry_value(candidate)[value_len] = '\0';

    pthread_mutex_lock(&cache->mutex);
    response_cache_entry **slot = entry_slot(
        cache, candidate->hash, key, key_len);
    if (*slot) {
        pthread_mutex_unlock(&cache->mutex);
        free(candidate);
        return;
    }
    while (cache->used_bytes > cache->max_bytes - bytes) {
        response_cache_entry *victim = cache->lru_tail;
        while (victim && victim->users != 0) victim = victim->lru_prev;
        if (!victim) {
            pthread_mutex_unlock(&cache->mutex);
            free(candidate);
            return;
        }
        remove_entry(cache, victim);
    }
    slot = entry_slot(cache, candidate->hash, key, key_len);
    candidate->hash_next = *slot;
    *slot = candidate;
    cache->used_bytes += bytes;
    lru_push_front(cache, candidate);
    pthread_mutex_unlock(&cache->mutex);
}
