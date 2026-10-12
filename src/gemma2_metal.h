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
/* Frames of one size (video) as one packed vision wave: out holds the frames'
 * soft tokens in order, tokens counts all of them. At most
 * EI_GEMMA2_VISION_WAVE_TOKENS patch tokens (width * height / 256 per frame). */
#define EI_GEMMA2_VISION_WAVE_TOKENS 10080
bool ei_gemma2_metal_vision_frames(void *ctx, const ei_gguf *g, const uint8_t *const *rgb,
                                   int frames, int width, int height, float **out, size_t *tokens,
                                   char *err, size_t len);
bool ei_gemma2_metal_audio(void *ctx, const ei_gguf *g, const float *mel, int frames, float **out,
                           size_t *tokens, char *err, size_t len);
/* Audio clips (1..3000 mel frames each) as one packed wave: out holds every
 * clip's rows in order, (frames + 3) / 4 rows per clip; tokens counts all of
 * them. At most EI_GEMMA2_AUDIO_WAVE_FRAMES frames and 64 clips per wave. */
#define EI_GEMMA2_AUDIO_WAVE_FRAMES 12000
bool ei_gemma2_metal_audio_clips(void *ctx, const ei_gguf *g, const float *const *mel,
                                 const int *frames, int clips, float **out, size_t *tokens,
                                 char *err, size_t len);
bool ei_gemma2_metal_batch(void *ctx, const int32_t *ids, const size_t *offsets, size_t batch,
                           float *out, char *err, size_t len);
/* Packed sequences of 512-float input rows (media requests after their towers):
 * inputs holds offsets[batch] rows; waves are split at 8192 tokens. */
bool ei_gemma2_metal_batch_inputs(void *ctx, const float *inputs, const size_t *offsets,
                                  size_t batch, float *out, char *err, size_t len);
#endif
