/* Native EmbeddingGemma 2 graph. No ggml runtime dependency. */
#ifndef EI_GEMMA2_H
#define EI_GEMMA2_H
#include "parallel.h"
#include "quants.h"
void ei_gemma2_validate(const ei_gguf *g);
void ei_gemma2_validate_projector(const ei_gguf *g);
bool ei_gemma2_embed(const ei_gguf *g, ei_thread_pool *pool, const int32_t *ids,
                     const float *inputs, size_t n, float out[768], char *err, size_t err_len);
/* Packed-batch fast CPU text graph (EI_GEMMA2_CPU_FAST=1): batch sequences
 * delimited by offsets[0..batch], one 768-float embedding each. Token ids, or
 * (ids NULL) 512-wide input rows such as media soft tokens. */
bool ei_gemma2_cpu_fast(void);
bool ei_gemma2_embed_batch_fast(const ei_gguf *g, ei_thread_pool *pool, const int32_t *ids,
                                const float *inputs, const size_t *offsets, size_t batch,
                                float *out, char *err, size_t err_len);
/* Independent fast-graph contexts (own workspace; weights and repack caches
 * shared): waves on distinct contexts and pools may run concurrently. */
typedef struct ei_gemma2_fast ei_gemma2_fast;
ei_gemma2_fast *ei_gemma2_fast_create(void);
void ei_gemma2_fast_free(ei_gemma2_fast *ctx);
bool ei_gemma2_embed_batch_fast_ctx(ei_gemma2_fast *ctx, const ei_gguf *g, ei_thread_pool *pool,
                                    const int32_t *ids, const float *inputs,
                                    const size_t *offsets, size_t batch, float *out, char *err,
                                    size_t err_len);
/* Fast-graph kernels reused by the CPU media towers (cpu-max fork). The GEMM
 * uses ctx's activation-repack workspace; callers serialize per context. */
void ei_gemma2_fast_gemm_q8(ei_gemma2_fast *ctx, ei_thread_pool *pool, const ei_tensor *w,
                            const ei_block_q8_0 *xq, int n, float *y);
void ei_gemma2_fast_gemm_f(ei_thread_pool *pool, const ei_tensor *w, const float *x, int n, float *y);
void ei_gemma2_fast_quant_row(const float *x, ei_block_q8_0 *q, int width);
float ei_gemma2_fast_sum_squares(const float *x, int d);
void ei_gemma2_fast_gelu_mul(float *g, const float *u, int n);
bool ei_gemma2_vision(const ei_gguf *g, ei_thread_pool *pool, const uint8_t *rgb, int width,
                      int height, float **output, size_t *tokens, char *err, size_t err_len);
/* cpu-max fork: SIMD/threaded CPU vision tower (default; EI_GEMMA2_CPU_MEDIA_FAST=0
 * restores the reference graph above). Same math as the reference: Q8 activations
 * for Q8 weights, fp32 attention/norms, the GGUF-selected GELU variant. */
bool ei_gemma2_cpu_media_fast(void);
bool ei_gemma2_vision_ref(const ei_gguf *g, ei_thread_pool *pool, const uint8_t *rgb, int width,
                          int height, float **output, size_t *tokens, char *err, size_t err_len);
bool ei_gemma2_vision_fast(const ei_gguf *g, ei_thread_pool *pool, const uint8_t *rgb, int width,
                           int height, float **output, size_t *tokens, char *err, size_t err_len);
bool ei_gemma2_audio(const ei_gguf *g, ei_thread_pool *pool, const float *mel, int frames,
                     float **output, size_t *tokens, char *err, size_t err_len);
#endif
