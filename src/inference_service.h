#ifndef EI_INFERENCE_SERVICE_H
#define EI_INFERENCE_SERVICE_H

#include "tokenizer.h"

typedef bool (*ei_inference_execute_fn)(void *opaque, const int32_t *ids,
                                        const size_t *offsets, size_t batch_size,
                                        float *out, char *err, size_t err_len);

typedef struct {
    size_t cache_entries;
    size_t max_batch_tokens;
    size_t max_batch_requests;
    size_t max_batch_sequence_tokens;
    size_t tokenizer_workers;
    uint32_t batch_wait_us;
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
bool ei_inference_service_embed_tokens(ei_inference_service *service,
                                       const int32_t *ids, size_t n_tokens,
                                       float out[EI_N_EMBD],
                                       char *err, size_t err_len);
void ei_inference_service_get_stats(ei_inference_service *service,
                                    ei_inference_service_stats *stats);

#endif /* EI_INFERENCE_SERVICE_H */
