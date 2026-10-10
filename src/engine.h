#ifndef EI_ENGINE_H
#define EI_ENGINE_H

#include "kernels.h"
#include "parallel.h"
#include "quants.h"
#include "tokenizer.h"

#define EI_MAX_PIPELINE_SLOTS 8 /* per GPU */
#define EI_MAX_EXECUTORS 32      /* slots across all GPUs and CPU workers */

typedef enum { EI_EXECUTOR_METAL = 1, EI_EXECUTOR_CUDA, EI_EXECUTOR_CPU } ei_executor_kind;
/* One wave runs on an executor at a time; distinct executors run concurrently. */
typedef struct {
    ei_executor_kind kind;
    uint8_t group;  /* device: pipeline slots of one GPU share it; each CPU worker has its own */
    bool overflow;  /* CPU worker beside a GPU */
    bool owned;     /* ctx is freed by the engine (not the engine's primary context) */
    void *ctx;      /* Metal/CUDA context, or ei_gemma2_fast for a CPU worker */
    ei_thread_pool *pool; /* CPU worker */
} ei_executor;
typedef struct {
    size_t pipeline_slots;     /* waves in flight per GPU (1..8) */
    const int *cuda_devices;   /* CUDA device indices; NULL: the engine's device */
    size_t cuda_device_count;
    size_t cpu_workers;        /* independent CPU executors */
    int32_t cpu_threads;       /* threads shared by all CPU workers (0: every core) */
    bool cpu_pin;              /* pin worker threads to distinct cores (Linux) */
} ei_executor_config;

typedef struct {
    ei_model model;
    ei_tokenizer tokenizer;
    void *metal;
    void *gemma2_metal;
    /* Executors created by ei_engine_configure_executors (none: the engine
     * runs every wave itself), and a dedicated media context beside them. */
    ei_executor executors[EI_MAX_EXECUTORS];
    size_t executor_count;
    void *gemma2_media;
    void *gemma2_cuda;
    void *gemma2_cuda_media;
    void *cuda;
    void *rocm;
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
/* Overlapping waves for EmbeddingGemma 2 on Metal or CUDA: slots > 1 creates
 * per-slot queues/streams and workspaces sharing the weights, plus a media
 * context. Other backends accept only 1. */
bool ei_engine_set_pipeline_slots(ei_engine *e, size_t slots, char *err, size_t err_len);
/* EmbeddingGemma 2 executors: pipeline slots on every listed CUDA device (each
 * holds its own weights) or on Metal, plus CPU workers with private thread
 * pools (beside a GPU they are overflow executors). Executor i is slot i of
 * ei_engine_embed_tokens_batch_slot. */
bool ei_engine_configure_executors(ei_engine *e, const ei_executor_config *config, char *err,
                                   size_t err_len);
/* Like ei_engine_embed_tokens_batch on pipeline slot `slot` (< slot count);
 * distinct slots may run concurrently from different threads. */
bool ei_engine_embed_tokens_batch_slot(ei_engine *e, size_t slot, const int32_t *ids,
                                       const size_t *offsets, size_t batch_size,
                                       float *out, char *err, size_t err_len);

#endif /* EI_ENGINE_H */
