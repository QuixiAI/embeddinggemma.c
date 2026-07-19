#ifndef EI_ENGINE_METAL_H
#define EI_ENGINE_METAL_H

#include "model.h"

void *ei_metal_engine_create(const ei_model *model, const char *library_path,
                             char *err, size_t err_len);
void ei_metal_engine_free(void *handle);
bool ei_metal_engine_reserve(void *handle, size_t total_tokens,
                             size_t batch_size, char *err, size_t err_len);
bool ei_metal_engine_embed_tokens(void *handle, const int32_t *ids, size_t n_tokens,
                                  float out[EI_N_EMBD], char *err, size_t err_len);
bool ei_metal_engine_embed_tokens_batch(void *handle, const int32_t *ids,
                                        const size_t *offsets, size_t batch_size,
                                        float *out, char *err, size_t err_len);

#endif /* EI_ENGINE_METAL_H */
