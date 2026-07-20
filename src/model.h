/* Typed view of embeddinggemma-300M-qat-Q4_0.gguf: hyperparameters and
 * per-layer tensor pointers, resolved and validated at load. */
#ifndef EI_MODEL_H
#define EI_MODEL_H

#include "gguf.h"

#define EI_N_LAYER   24
#define EI_N_EMBD    768
#define EI_N_FF      1152
#define EI_N_HEAD    3
#define EI_N_HEAD_KV 1
#define EI_HEAD_DIM  256
#define EI_N_CTX     2048
#define EI_VOCAB     262144

typedef struct {
    /* q4_0 matmul weights; shapes noted as [in, out] element counts */
    const ei_tensor *attn_q;      /* [768, 768]  (3 heads × 256) */
    const ei_tensor *attn_k;      /* [768, 256] */
    const ei_tensor *attn_v;      /* [768, 256] */
    const ei_tensor *attn_output; /* [768, 768] */
    const ei_tensor *ffn_up;      /* [768, 1152] */
    const ei_tensor *ffn_gate;    /* [768, 1152] */
    const ei_tensor *ffn_down;    /* [1152, 768] */
    /* f32 norm weights */
    const float *attn_norm;        /* [768] */
    const float *attn_q_norm;      /* [256] */
    const float *attn_k_norm;      /* [256] */
    const float *post_attention_norm; /* [768] */
    const float *ffn_norm;         /* [768] */
    const float *post_ffw_norm;    /* [768] */
} ei_layer;

typedef struct {
    ei_gguf gguf;

    /* hparams read from the file (validated against the EI_* constants) */
    float    rms_eps;
    float    rope_base_full;  /* full-attention layers */
    float    rope_base_swa;   /* sliding-window layers */
    uint32_t swa_window;      /* 512 */
    uint32_t swa_period;      /* every Nth layer is full attention */

    const ei_tensor *token_embd;  /* q8_0 [768, 262144] */
    const float     *output_norm; /* f32 [768] */
    ei_layer layers[EI_N_LAYER];

    /* tokenizer data (borrowed from the gguf mapping) */
    const ei_kv *tok_tokens;  /* string array, 262144 */
    const ei_kv *tok_scores;  /* f32 array */
    const ei_kv *tok_types;   /* i32 array */
    int32_t bos_id, eos_id, unk_id, pad_id;
    bool add_bos, add_eos, add_space_prefix;
} ei_model;

#ifdef __cplusplus
extern "C" {
#endif

void ei_model_load(ei_model *m, const char *path);
void ei_model_free(ei_model *m);

/* True when layer il uses the 512-token symmetric sliding window. */
bool ei_layer_is_swa(const ei_model *m, int il);

#ifdef __cplusplus
}
#endif

#endif /* EI_MODEL_H */
