#include "model.h"

static const ei_tensor *want(const ei_gguf *g, const char *name,
                             uint32_t type, uint64_t ne0, uint64_t ne1) {
    const ei_tensor *t = ei_gguf_tensor(g, name, true);
    if (t->type != type) {
        ei_die("tensor %s: expected ggml type %u, file has %u", name, type, t->type);
    }
    if (t->ne[0] != ne0 || t->ne[1] != ne1) {
        ei_die("tensor %s: expected shape [%llu, %llu], file has [%llu, %llu]",
               name, (unsigned long long)ne0, (unsigned long long)ne1,
               (unsigned long long)t->ne[0], (unsigned long long)t->ne[1]);
    }
    return t;
}

static const float *want_f32_vec(const ei_gguf *g, const char *name, uint64_t n) {
    return (const float *)want(g, name, EI_T_F32, n, 1)->data;
}

static const ei_kv *want_arr(const ei_gguf *g, const char *key,
                             uint32_t arr_type, uint64_t n) {
    const ei_kv *kv = ei_gguf_kv(g, key, true);
    if (kv->type != EI_GGUF_ARRAY || kv->arr_type != arr_type || kv->arr_n != n) {
        ei_die("gguf key %s: expected array(type %u) x %llu",
               key, arr_type, (unsigned long long)n);
    }
    return kv;
}

void ei_model_load(ei_model *m, const char *path) {
    memset(m, 0, sizeof *m);
    ei_gguf *g = &m->gguf;
    ei_gguf_open(g, path);

    const ei_kv *arch = ei_gguf_kv(g, "general.architecture", true);
    if (arch->type != EI_GGUF_STRING ||
        arch->v.s.len != strlen("gemma-embedding") ||
        memcmp(arch->v.s.str, "gemma-embedding", arch->v.s.len) != 0) {
        ei_die("%s: not a gemma-embedding GGUF (this engine serves exactly one model)", path);
    }

    const ei_kv *tok_model = ei_gguf_kv(g, "tokenizer.ggml.model", true);
    if (tok_model->type != EI_GGUF_STRING ||
        tok_model->v.s.len != strlen("llama") ||
        memcmp(tok_model->v.s.str, "llama", tok_model->v.s.len) != 0) {
        ei_die("%s: unsupported tokenizer model (expected llama/SPM)", path);
    }

    #define CHECK_HPARAM(key, expect) do { \
        uint64_t v = ei_gguf_kv_u64(g, key, 0); \
        if (v != (expect)) ei_die("%s: %s = %llu, engine is built for %llu", \
                                  path, key, (unsigned long long)v, \
                                  (unsigned long long)(expect)); \
    } while (0)

    CHECK_HPARAM("gemma-embedding.block_count",           EI_N_LAYER);
    CHECK_HPARAM("gemma-embedding.embedding_length",      EI_N_EMBD);
    CHECK_HPARAM("gemma-embedding.feed_forward_length",   EI_N_FF);
    CHECK_HPARAM("gemma-embedding.attention.head_count",  EI_N_HEAD);
    CHECK_HPARAM("gemma-embedding.attention.head_count_kv", EI_N_HEAD_KV);
    CHECK_HPARAM("gemma-embedding.attention.key_length",  EI_HEAD_DIM);
    CHECK_HPARAM("gemma-embedding.attention.value_length", EI_HEAD_DIM);
    CHECK_HPARAM("gemma-embedding.context_length",        EI_N_CTX);
    #undef CHECK_HPARAM

    m->rms_eps = (float)ei_gguf_kv_f64(g, "gemma-embedding.attention.layer_norm_rms_epsilon", 1e-6);
    m->rope_base_full = (float)ei_gguf_kv_f64(g, "gemma-embedding.rope.freq_base", 1000000.0);
    /* SWA layers train with their own (lower) rope base; absent key = 10000
     * per llama.cpp's default for this arch. */
    m->rope_base_swa = (float)ei_gguf_kv_f64(g, "gemma-embedding.rope.freq_base_swa", 10000.0);
    m->swa_window = (uint32_t)ei_gguf_kv_u64(g, "gemma-embedding.attention.sliding_window", 512);
    m->swa_period = (uint32_t)ei_gguf_kv_u64(g, "gemma-embedding.attention.sliding_window_pattern", 6);

    m->token_embd  = want(g, "token_embd.weight", EI_T_Q8_0, EI_N_EMBD, EI_VOCAB);
    m->output_norm = want_f32_vec(g, "output_norm.weight", EI_N_EMBD);

    for (int il = 0; il < EI_N_LAYER; il++) {
        char n[128];
        ei_layer *L = &m->layers[il];
        #define T(field, suffix, ty, a, b) do { \
            snprintf(n, sizeof n, "blk.%d." suffix, il); \
            L->field = want(g, n, ty, a, b); \
        } while (0)
        #define V(field, suffix, len) do { \
            snprintf(n, sizeof n, "blk.%d." suffix, il); \
            L->field = want_f32_vec(g, n, len); \
        } while (0)
        T(attn_q,      "attn_q.weight",      EI_T_Q4_0, EI_N_EMBD, EI_N_HEAD * EI_HEAD_DIM);
        T(attn_k,      "attn_k.weight",      EI_T_Q4_0, EI_N_EMBD, EI_N_HEAD_KV * EI_HEAD_DIM);
        T(attn_v,      "attn_v.weight",      EI_T_Q4_0, EI_N_EMBD, EI_N_HEAD_KV * EI_HEAD_DIM);
        T(attn_output, "attn_output.weight", EI_T_Q4_0, EI_N_HEAD * EI_HEAD_DIM, EI_N_EMBD);
        T(ffn_up,      "ffn_up.weight",      EI_T_Q4_0, EI_N_EMBD, EI_N_FF);
        T(ffn_gate,    "ffn_gate.weight",    EI_T_Q4_0, EI_N_EMBD, EI_N_FF);
        T(ffn_down,    "ffn_down.weight",    EI_T_Q4_0, EI_N_FF, EI_N_EMBD);
        V(attn_norm,           "attn_norm.weight",           EI_N_EMBD);
        V(attn_q_norm,         "attn_q_norm.weight",         EI_HEAD_DIM);
        V(attn_k_norm,         "attn_k_norm.weight",         EI_HEAD_DIM);
        V(post_attention_norm, "post_attention_norm.weight", EI_N_EMBD);
        V(ffn_norm,            "ffn_norm.weight",            EI_N_EMBD);
        V(post_ffw_norm,       "post_ffw_norm.weight",       EI_N_EMBD);
        #undef T
        #undef V
    }

    m->tok_tokens = want_arr(g, "tokenizer.ggml.tokens", EI_GGUF_STRING, EI_VOCAB);
    m->tok_scores = want_arr(g, "tokenizer.ggml.scores", EI_GGUF_F32, EI_VOCAB);
    m->tok_types  = want_arr(g, "tokenizer.ggml.token_type", EI_GGUF_I32, EI_VOCAB);
    m->bos_id = (int32_t)ei_gguf_kv_u64(g, "tokenizer.ggml.bos_token_id", 2);
    m->eos_id = (int32_t)ei_gguf_kv_u64(g, "tokenizer.ggml.eos_token_id", 1);
    m->unk_id = (int32_t)ei_gguf_kv_u64(g, "tokenizer.ggml.unknown_token_id", 3);
    m->pad_id = (int32_t)ei_gguf_kv_u64(g, "tokenizer.ggml.padding_token_id", 0);
    m->add_bos = ei_gguf_kv_bool(g, "tokenizer.ggml.add_bos_token", true);
    m->add_eos = ei_gguf_kv_bool(g, "tokenizer.ggml.add_eos_token", false);
    m->add_space_prefix = ei_gguf_kv_bool(g, "tokenizer.ggml.add_space_prefix", true);
}

bool ei_layer_is_swa(const ei_model *m, int il) {
    /* llama.cpp set_swa_pattern(6): within each period the first 5 layers are
     * sliding-window, the last is full attention (layers 5, 11, 17, 23). */
    return ((uint32_t)il % m->swa_period) < (m->swa_period - 1);
}

void ei_model_free(ei_model *m) {
    ei_gguf_close(&m->gguf);
    memset(m, 0, sizeof *m);
}
