#ifndef EI_GEMMA2_METAL_H
#define EI_GEMMA2_METAL_H
#include "gguf.h"
void *ei_gemma2_metal_create(const ei_gguf *g, char *err, size_t len);
void ei_gemma2_metal_free(void *ctx);
/* A pipeline slot: shares the device, pipelines and weights of ctx, with its
 * own command queue and workspace so waves on different slots overlap. */
void *ei_gemma2_metal_clone(void *ctx, char *err, size_t len);
bool ei_gemma2_metal_embed(void *ctx, const int32_t *ids, const float *inputs, size_t n,
                           float out[768], char *err, size_t len);
bool ei_gemma2_metal_vision(void *ctx, const ei_gguf *g, const uint8_t *rgb, int width, int height,
                            float **out, size_t *tokens, char *err, size_t len);
bool ei_gemma2_metal_audio(void *ctx, const ei_gguf *g, const float *mel, int frames, float **out,
                           size_t *tokens, char *err, size_t len);
bool ei_gemma2_metal_batch(void *ctx, const int32_t *ids, const size_t *offsets, size_t batch,
                           float *out, char *err, size_t len);
#endif
