#ifndef EI_GEMMA2_CUDA_H
#define EI_GEMMA2_CUDA_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#include "gguf.h"
/* EmbeddingGemma 2 text tower on CUDA (packed multi-sequence waves). */
void *ei_gemma2_cuda_create(const ei_gguf *g, char *err, size_t len);
void ei_gemma2_cuda_free(void *ctx);
bool ei_gemma2_cuda_batch(void *ctx, const int32_t *ids, const size_t *offsets, size_t batch,
                          float *out, char *err, size_t len);
/* One sequence of 512-wide input rows (media soft tokens interleaved with text). */
bool ei_gemma2_cuda_embed_inputs(void *ctx, const float *inputs, size_t n, float out[768],
                                 char *err, size_t len);
/* Media towers on the GPU: soft tokens (512 wide) for the text tower. */
bool ei_gemma2_cuda_vision(void *ctx, const ei_gguf *projector, const uint8_t *rgb, int width,
                           int height, float **out, size_t *tokens, char *err, size_t len);
bool ei_gemma2_cuda_audio(void *ctx, const ei_gguf *projector, const float *mel, int frames,
                          float **out, size_t *tokens, char *err, size_t len);
#ifdef __cplusplus
}
#endif
#endif
