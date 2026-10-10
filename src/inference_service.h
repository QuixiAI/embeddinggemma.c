#ifndef EI_INFERENCE_SERVICE_H
#define EI_INFERENCE_SERVICE_H

#include "tokenizer.h"

typedef bool (*ei_inference_execute_fn)(void *opaque, const int32_t *ids,
                                        const size_t *offsets, size_t batch_size,
                                        float *out, char *err, size_t err_len);

/* Slot-aware execute: waves on distinct slots may run concurrently. */
typedef bool (*ei_inference_execute_slot_fn)(void *opaque, size_t slot, const int32_t *ids,
                                             const size_t *offsets, size_t batch_size,
                                             float *out, char *err, size_t err_len);

typedef struct {
    size_t cache_entries;
    size_t max_batch_tokens;
    size_t max_batch_requests;
    size_t max_batch_sequence_tokens;
    size_t tokenizer_workers;
    uint32_t batch_wait_us;
    /* Optional on-disk exact-cache persistence. When cache_path is set, the
     * service loads matching entries at startup and dumps ready entries on
     * ei_inference_service_dump_cache / free. cache_fingerprint must uniquely
     * identify the model so a cache from a different model is never served. */
    const char *cache_path;
    uint64_t cache_fingerprint;
    /* Waves in flight (0 or 1: one, the historical behaviour). With more,
     * execute_slot must be set; it receives the wave's slot in [0, slots). */
    size_t pipeline_slots;
    ei_inference_execute_slot_fn execute_slot;
    /* Mixed traffic: requests of at most this many tokens are "short". While
     * every in-flight wave is long, a free slot may start a short-only wave,
     * so short requests bypass long waves; waves never mix the two classes.
     * 0 disables (the historical collection). Needs pipeline_slots > 1. */
    size_t short_bypass_tokens;
    /* Executors (optional, pipeline_slots entries each). slot_group[i] names
     * the device of slot i: waves go to an idle device first, then to the
     * least-loaded one (NULL: every slot on one device). slot_max_batch_tokens
     * caps a slot's waves (0: max_batch_tokens), so slow executors such as CPU
     * workers take waves they finish promptly. Overflow slots (CPU workers
     * beside a GPU) start a wave only when no GPU slot can, and take only
     * requests of at most overflow_max_sequence_tokens. */
    const uint8_t *slot_group;
    const bool *slot_overflow;
    const size_t *slot_max_batch_tokens;
    size_t overflow_max_sequence_tokens;
} ei_inference_service_config;

typedef struct {
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t coalesced_requests;
    uint64_t batches;
    uint64_t sequences;
    uint64_t tokens;
    size_t cache_entries;
    size_t queued_requests;
} ei_inference_service_stats;

typedef struct ei_inference_service ei_inference_service;

ei_inference_service *ei_inference_service_create(
    const ei_tokenizer *tokenizer, ei_inference_execute_fn execute,
    void *execute_opaque, const ei_inference_service_config *config);
void ei_inference_service_free(ei_inference_service *service);

bool ei_inference_service_embed(ei_inference_service *service,
                                const char *text, size_t text_len,
                                float out[EI_N_EMBD], char *err, size_t err_len);
bool ei_inference_service_embed_batch(ei_inference_service *service,
                                      const char *const *texts,
                                      const size_t *text_lengths,
                                      size_t batch_size, float *out,
                                      char *err, size_t err_len);
bool ei_inference_service_embed_batch_with_usage(
    ei_inference_service *service, const char *const *texts,
    const size_t *text_lengths, size_t batch_size, float *out,
    size_t *prompt_tokens, char *err, size_t err_len);
bool ei_inference_service_embed_tokens(ei_inference_service *service,
                                       const int32_t *ids, size_t n_tokens,
                                       float out[EI_N_EMBD],
                                       char *err, size_t err_len);
void ei_inference_service_get_stats(ei_inference_service *service,
                                    ei_inference_service_stats *stats);

/* Atomically write the current exact cache to config.cache_path. Safe to call
 * while the service is live (locks internally). No-op when cache_path is
 * unset. Best-effort: persistence failures are silent and never disrupt
 * serving. */
void ei_inference_service_dump_cache(ei_inference_service *service);

#endif /* EI_INFERENCE_SERVICE_H */
