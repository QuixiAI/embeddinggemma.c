#ifndef EI_ENGINE_H
#define EI_ENGINE_H

#include "kernels.h"
#include "parallel.h"
#include "quants.h"
#include "tokenizer.h"

typedef struct {
    ei_model model;
    ei_tokenizer tokenizer;
    void *metal;
    void *cuda;
    void *xpu;
    const char *backend_name;
    ei_thread_pool *thread_pool;
    int32_t short_projection_threads;
    int32_t multirow_min_tokens;
    bool dual_projection;
    bool fused_rms_quant;
    bool fused_gelu_quant;

    size_t workspace_tokens;
    int32_t *workspace_positions;
    float *workspace_x;
    float *workspace_attn_in;
    float *workspace_q;
    float *workspace_k;
    float *workspace_v;
    float *workspace_ctx;
    float *workspace_up;
    float *workspace_gate;
    ei_block_q8_0 *workspace_q8;
} ei_engine;

void ei_engine_load(ei_engine *e, const char *model_path);
void ei_engine_load_backend(ei_engine *e, const char *model_path, const char *backend);
void ei_engine_free(ei_engine *e);
const char *ei_engine_backend(const ei_engine *e);
int32_t ei_engine_threads(const ei_engine *e);
bool ei_engine_reserve(ei_engine *e, size_t total_tokens, size_t batch_size,
                       char *err, size_t err_len);

bool ei_engine_embed(ei_engine *e, const char *text, size_t len, float out[EI_N_EMBD],
                     char *err, size_t err_len);
bool ei_engine_embed_tokens(ei_engine *e, const int32_t *ids, size_t n_tokens,
                            float out[EI_N_EMBD], char *err, size_t err_len);
bool ei_engine_embed_tokens_batch(ei_engine *e, const int32_t *ids,
                                  const size_t *offsets, size_t batch_size,
                                  float *out, char *err, size_t err_len);

#endif /* EI_ENGINE_H */
