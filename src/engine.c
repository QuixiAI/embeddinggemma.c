#include "engine.h"

#if defined(EI_ENABLE_METAL)
#include "engine_metal.h"
#endif

#include <math.h>

static void set_err(char *err, size_t err_len, const char *msg) {
    if (err && err_len) {
        snprintf(err, err_len, "%s", msg);
    }
}

void ei_engine_load_backend(ei_engine *e, const char *model_path, const char *backend) {
    memset(e, 0, sizeof *e);
    ei_model_load(&e->model, model_path);
    ei_tokenizer_init(&e->tokenizer, &e->model);
    e->backend_name = "cpu";
    e->short_projection_threads = 6;
    e->multirow_min_tokens = 512;
    const char *dual_projection = getenv("EI_CPU_DUAL_PROJECTION");
    e->dual_projection = !dual_projection || strcmp(dual_projection, "0") != 0;
    const char *fused_rms_quant = getenv("EI_CPU_FUSED_RMS_QUANT");
    e->fused_rms_quant = !fused_rms_quant || strcmp(fused_rms_quant, "0") != 0;
    const char *fused_gelu_quant = getenv("EI_CPU_FUSED_GELU_QUANT");
    e->fused_gelu_quant = fused_gelu_quant &&
        strcmp(fused_gelu_quant, "0") != 0;
    const char *short_threads = getenv("EI_CPU_SHORT_THREADS");
    if (short_threads && *short_threads) {
        char *end = NULL;
        long parsed = strtol(short_threads, &end, 10);
        if (*end != '\0' || parsed < 1 || parsed > 16) {
            ei_die("EI_CPU_SHORT_THREADS must be an integer from 1 to 16");
        }
        e->short_projection_threads = (int32_t)parsed;
    }
    const char *multirow_min = getenv("EI_CPU_MULTIROW_MIN_TOKENS");
    if (multirow_min && *multirow_min) {
        char *end = NULL;
        long parsed = strtol(multirow_min, &end, 10);
        if (*end != '\0' || parsed < 0 || parsed > 65536) {
            ei_die("EI_CPU_MULTIROW_MIN_TOKENS must be an integer from 0 to 65536");
        }
        e->multirow_min_tokens = parsed == 0 ? INT32_MAX : (int32_t)parsed;
    }

    const char *requested = backend && *backend ? backend : "auto";
    if (strcmp(requested, "auto") != 0 && strcmp(requested, "cpu") != 0 &&
        strcmp(requested, "metal") != 0) {
        ei_die("unknown inference backend '%s' (expected auto, cpu, or metal)", requested);
    }
#if defined(EI_ENABLE_METAL)
    if (strcmp(requested, "cpu") != 0) {
        char metal_err[512];
        e->metal = ei_metal_engine_create(&e->model, NULL, metal_err, sizeof metal_err);
        if (e->metal) {
            e->backend_name = "metal";
        } else if (strcmp(requested, "metal") == 0) {
            ei_die("Metal backend initialization failed: %s", metal_err);
        } else {
            fprintf(stderr, "Metal unavailable, using CPU: %s\n", metal_err);
        }
    }
#else
    if (strcmp(requested, "metal") == 0) {
        ei_die("this binary was built without Metal support");
    }
#endif
    if (!e->metal) e->thread_pool = ei_thread_pool_create(0);
}

void ei_engine_load(ei_engine *e, const char *model_path) {
    const char *backend = getenv("EI_BACKEND");
    ei_engine_load_backend(e, model_path, backend ? backend : "auto");
}

void ei_engine_free(ei_engine *e) {
#if defined(EI_ENABLE_METAL)
    ei_metal_engine_free(e->metal);
#endif
    ei_thread_pool_destroy(e->thread_pool);
    free(e->workspace_positions);
    free(e->workspace_ctx);
    free(e->workspace_gate);
    free(e->workspace_up);
    free(e->workspace_q8);
    free(e->workspace_v);
    free(e->workspace_k);
    free(e->workspace_q);
    free(e->workspace_attn_in);
    free(e->workspace_x);
    ei_tokenizer_free(&e->tokenizer);
    ei_model_free(&e->model);
    memset(e, 0, sizeof *e);
}

const char *ei_engine_backend(const ei_engine *e) {
    return e->backend_name ? e->backend_name : "cpu";
}

int32_t ei_engine_threads(const ei_engine *e) {
    return ei_thread_pool_threads(e->thread_pool);
}

static bool token_ids_valid(const int32_t *ids, size_t n, char *err, size_t err_len) {
    if (n == 0) {
        set_err(err, err_len, "tokenizer produced no tokens");
        return false;
    }
    if (n > EI_N_CTX) {
        if (err && err_len) {
            snprintf(err, err_len, "input has %zu tokens, maximum context is %d", n, EI_N_CTX);
        }
        return false;
    }
    for (size_t i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= EI_VOCAB) {
            if (err && err_len) {
                snprintf(err, err_len, "token id %d at index %zu is outside vocab", ids[i], i);
            }
            return false;
        }
    }
    return true;
}

static void input_embeddings(const ei_model *model, const int32_t *ids,
                             int32_t n_tokens, float *output) {
    const float scale = sqrtf((float)EI_N_EMBD);
    for (int32_t t = 0; t < n_tokens; t++) {
        float *row = output + (size_t)t * EI_N_EMBD;
        ei_dequantize_row_q8_0_scaled(model->token_embd, ids[t], scale, row);
    }
}

typedef struct {
    const ei_tensor *weight;
    const ei_block_q8_0 *input;
    float *output;
} q4_projection_context;

static void q4_projection_rows(void *opaque, int32_t begin, int32_t end) {
    q4_projection_context *context = opaque;
    ei_matmul_q4_0_q8_0_rows3(context->weight, context->input, context->output, begin, end);
}

static void parallel_q4_projection(const ei_engine *e, const ei_tensor *weight,
                                   const ei_block_q8_0 *input, float *output) {
    q4_projection_context context = { weight, input, output };
    int32_t rows = (int32_t)weight->ne[1];
    int32_t width = e->short_projection_threads;
    ei_parallel_for(e->thread_pool, rows, (rows + width - 1) / width,
                    q4_projection_rows, &context);
}

typedef struct {
    const ei_layer *layer;
    const ei_block_q8_0 *input;
    float *q;
    float *k;
    float *v;
} qkv_projection_context;

static void qkv_projection_rows(void *opaque, int32_t begin, int32_t end) {
    qkv_projection_context *context = opaque;
    if (begin < EI_N_EMBD) {
        int32_t stop = end < EI_N_EMBD ? end : EI_N_EMBD;
        ei_matmul_q4_0_q8_0_rows(context->layer->attn_q, context->input,
                                 context->q, begin, stop);
    }
    if (end > EI_N_EMBD && begin < EI_N_EMBD + EI_HEAD_DIM) {
        int32_t start = begin > EI_N_EMBD ? begin - EI_N_EMBD : 0;
        int32_t stop = end - EI_N_EMBD;
        if (stop > EI_HEAD_DIM) stop = EI_HEAD_DIM;
        ei_matmul_q4_0_q8_0_rows(context->layer->attn_k, context->input,
                                 context->k, start, stop);
    }
    if (end > EI_N_EMBD + EI_HEAD_DIM) {
        int32_t start = begin > EI_N_EMBD + EI_HEAD_DIM
            ? begin - EI_N_EMBD - EI_HEAD_DIM : 0;
        int32_t stop = end - EI_N_EMBD - EI_HEAD_DIM;
        if (stop > EI_HEAD_DIM) stop = EI_HEAD_DIM;
        ei_matmul_q4_0_q8_0_rows(context->layer->attn_v, context->input,
                                 context->v, start, stop);
    }
}

static void qkv_projection_rows_triple(void *opaque, int32_t begin, int32_t end) {
    qkv_projection_context *context = opaque;
    if (begin < EI_HEAD_DIM) {
        int32_t stop = end < EI_HEAD_DIM ? end : EI_HEAD_DIM;
        ei_matmul_q4_0_q8_0_triple_rows(
            context->layer->attn_q, context->layer->attn_k,
            context->layer->attn_v, context->input, context->q,
            context->k, context->v, begin, stop);
    }
    if (end > EI_HEAD_DIM) {
        int32_t start = begin > EI_HEAD_DIM ? begin : EI_HEAD_DIM;
        ei_matmul_q4_0_q8_0_rows3(context->layer->attn_q, context->input,
                                  context->q, start, end);
    }
}

typedef struct {
    const ei_layer *layer;
    const ei_block_q8_0 *input;
    float *up;
    float *gate;
} up_gate_projection_context;

static void up_gate_projection_rows(void *opaque, int32_t begin, int32_t end) {
    up_gate_projection_context *context = opaque;
    if (begin < EI_N_FF) {
        int32_t stop = end < EI_N_FF ? end : EI_N_FF;
        ei_matmul_q4_0_q8_0_rows(context->layer->ffn_up, context->input,
                                 context->up, begin, stop);
    }
    if (end > EI_N_FF) {
        int32_t start = begin > EI_N_FF ? begin - EI_N_FF : 0;
        int32_t stop = end - EI_N_FF;
        if (stop > EI_N_FF) stop = EI_N_FF;
        ei_matmul_q4_0_q8_0_rows(context->layer->ffn_gate, context->input,
                                 context->gate, start, stop);
    }
}

static void up_gate_projection_rows_dual(void *opaque, int32_t begin, int32_t end) {
    up_gate_projection_context *context = opaque;
    ei_matmul_q4_0_q8_0_dual_rows(context->layer->ffn_up, context->layer->ffn_gate,
                                  context->input, context->up, context->gate,
                                  begin, end);
}

typedef struct {
    const ei_engine *engine;
    const ei_layer *layer;
    const float *x;
    float *attn_in;
    float *q;
    float *k;
    float *v;
    float rope_base;
    const int32_t *positions;
} build_qkv_context;

typedef struct {
    const float *input;
    const float *weight;
    ei_block_q8_0 *output;
    int32_t width;
    int32_t blocks;
    float eps;
} norm_quant_context;

static void norm_quant_rows(void *opaque, int32_t begin, int32_t end) {
    norm_quant_context *context = opaque;
    for (int32_t token = begin; token < end; token++) {
        ei_rms_norm_quantize_q8_0(
            context->input + (size_t)token * context->width,
            context->weight, context->width, context->eps,
            context->output + (size_t)token * context->blocks);
    }
}

typedef struct {
    const float *input;
    ei_block_q8_0 *output;
    int32_t width;
    int32_t blocks;
} quant_context;

static void quant_rows(void *opaque, int32_t begin, int32_t end) {
    quant_context *context = opaque;
    for (int32_t token = begin; token < end; token++) {
        ei_quantize_row_q8_0(
            context->input + (size_t)token * context->width,
            context->output + (size_t)token * context->blocks,
            context->width);
    }
}

typedef struct {
    const ei_tensor *weight;
    const ei_block_q8_0 *input;
    int32_t tokens;
    float *output;
} batch_projection_context;

static void batch_projection_rows(void *opaque, int32_t begin, int32_t end) {
    batch_projection_context *context = opaque;
    ei_matmul_q4_0_q8_0_batch_rows(
        context->weight, context->input, context->tokens,
        context->output, begin, end);
}

static void parallel_batch_projection(const ei_engine *e, const ei_tensor *weight,
                                      const ei_block_q8_0 *input, int32_t tokens,
                                      float *output) {
    batch_projection_context context = {
        .weight = weight, .input = input, .tokens = tokens, .output = output,
    };
    ei_parallel_for(e->thread_pool, (int32_t)weight->ne[1], 1,
                    batch_projection_rows, &context);
}

typedef struct {
    const ei_engine *engine;
    const ei_layer *layer;
    float *q;
    float *k;
    const int32_t *positions;
    float rope_base;
} qk_post_context;

static void qk_post_rows(void *opaque, int32_t begin, int32_t end) {
    qk_post_context *context = opaque;
    for (int32_t token = begin; token < end; token++) {
        int32_t position = context->positions ? context->positions[token] : token;
        ei_qk_norm_rope_qk_inplace(
            context->q + (size_t)token * EI_N_EMBD,
            context->k + (size_t)token * EI_HEAD_DIM,
            context->layer->attn_q_norm, context->layer->attn_k_norm,
            position, context->rope_base, context->engine->model.rms_eps);
    }
}

static void build_qkv_multirow(const ei_engine *e, const ei_layer *layer,
                               int32_t tokens, const int32_t *positions,
                               const float *x, float *q, float *k, float *v) {
    norm_quant_context quant = {
        .input = x, .weight = layer->attn_norm,
        .output = e->workspace_q8, .width = EI_N_EMBD,
        .blocks = EI_N_EMBD / EI_QK, .eps = e->model.rms_eps,
    };
    ei_parallel_for(e->thread_pool, tokens, 1, norm_quant_rows, &quant);
    parallel_batch_projection(e, layer->attn_q, e->workspace_q8, tokens, q);
    parallel_batch_projection(e, layer->attn_k, e->workspace_q8, tokens, k);
    parallel_batch_projection(e, layer->attn_v, e->workspace_q8, tokens, v);

    bool swa = ei_layer_is_swa(&e->model, (int)(layer - e->model.layers));
    qk_post_context post = {
        .engine = e, .layer = layer, .q = q, .k = k,
        .positions = positions,
        .rope_base = swa ? e->model.rope_base_swa : e->model.rope_base_full,
    };
    ei_parallel_for(e->thread_pool, tokens, 1, qk_post_rows, &post);
}

static void build_qkv_rows(void *opaque, int32_t begin, int32_t end) {
    build_qkv_context *context = opaque;
    const ei_engine *e = context->engine;
    const ei_layer *layer = context->layer;
    for (int32_t t = begin; t < end; t++) {
        ei_block_q8_0 qtmp[EI_N_EMBD / EI_QK];
        const float *xrow = context->x + (size_t)t * EI_N_EMBD;
        float *arow = context->attn_in + (size_t)t * EI_N_EMBD;
        float *qrow = context->q + (size_t)t * EI_N_EMBD;
        float *krow = context->k + (size_t)t * EI_HEAD_DIM;
        float *vrow = context->v + (size_t)t * EI_HEAD_DIM;

        if (e->fused_rms_quant) {
            ei_rms_norm_quantize_q8_0(xrow, layer->attn_norm, EI_N_EMBD,
                                      e->model.rms_eps, qtmp);
        } else {
            ei_rms_norm(xrow, layer->attn_norm, EI_N_EMBD, e->model.rms_eps, arow);
            ei_quantize_row_q8_0(arow, qtmp, EI_N_EMBD);
        }
        ei_matmul_q4_0_q8_0(layer->attn_q, qtmp, qrow);
        if (e->dual_projection) {
            ei_matmul_q4_0_q8_0_dual_rows(layer->attn_k, layer->attn_v, qtmp,
                                          krow, vrow, 0, EI_HEAD_DIM);
        } else {
            ei_matmul_q4_0_q8_0(layer->attn_k, qtmp, krow);
            ei_matmul_q4_0_q8_0(layer->attn_v, qtmp, vrow);
        }

        int32_t position = context->positions ? context->positions[t] : t;
        ei_qk_norm_rope_qk_inplace(qrow, krow, layer->attn_q_norm,
                                   layer->attn_k_norm, position, context->rope_base,
                                   e->model.rms_eps);
    }
}

static void build_qkv(const ei_engine *e, const ei_layer *layer, int32_t n_tokens,
                      const float *x, float *attn_in, float *q, float *k, float *v) {
    if (n_tokens >= e->multirow_min_tokens) {
        build_qkv_multirow(e, layer, n_tokens, NULL, x, q, k, v);
        return;
    }
    bool swa = ei_layer_is_swa(&e->model, (int)(layer - e->model.layers));
    float rope_base = swa ? e->model.rope_base_swa : e->model.rope_base_full;
    if (n_tokens == 1 && ei_engine_threads(e) > 1) {
        ei_block_q8_0 input[EI_N_EMBD / EI_QK];
        if (e->fused_rms_quant) {
            ei_rms_norm_quantize_q8_0(x, layer->attn_norm, EI_N_EMBD,
                                      e->model.rms_eps, input);
        } else {
            ei_rms_norm(x, layer->attn_norm, EI_N_EMBD, e->model.rms_eps, attn_in);
            ei_quantize_row_q8_0(attn_in, input, EI_N_EMBD);
        }
        qkv_projection_context projection = { layer, input, q, k, v };
        int32_t projection_rows = e->dual_projection
            ? EI_N_EMBD : EI_N_EMBD + 2 * EI_HEAD_DIM;
        int32_t width = e->short_projection_threads;
        ei_parallel_for(e->thread_pool, projection_rows,
                        (projection_rows + width - 1) / width,
                        e->dual_projection ? qkv_projection_rows_triple
                                           : qkv_projection_rows,
                        &projection);
        ei_qk_norm_rope_qk_inplace(q, k, layer->attn_q_norm,
                                   layer->attn_k_norm, 0, rope_base,
                                   e->model.rms_eps);
        return;
    }
    build_qkv_context context = {
        .engine = e, .layer = layer, .x = x, .attn_in = attn_in,
        .q = q, .k = k, .v = v,
        .rope_base = rope_base,
        .positions = NULL,
    };
    ei_parallel_for(e->thread_pool, n_tokens, 1, build_qkv_rows, &context);
}

static void build_qkv_batch(const ei_engine *e, const ei_layer *layer,
                            int32_t total_tokens, const int32_t *positions,
                            const float *x, float *attn_in, float *q,
                            float *k, float *v) {
    if (total_tokens >= e->multirow_min_tokens) {
        build_qkv_multirow(e, layer, total_tokens, positions, x, q, k, v);
        return;
    }
    bool swa = ei_layer_is_swa(&e->model, (int)(layer - e->model.layers));
    build_qkv_context context = {
        .engine = e, .layer = layer, .x = x, .attn_in = attn_in,
        .q = q, .k = k, .v = v,
        .rope_base = swa ? e->model.rope_base_swa : e->model.rope_base_full,
        .positions = positions,
    };
    ei_parallel_for(e->thread_pool, total_tokens, 1, build_qkv_rows, &context);
}

typedef struct {
    const ei_engine *engine;
    const ei_layer *layer;
    const float *ctx;
    float *x;
} attention_output_context;

static void attention_output_rows(void *opaque, int32_t begin, int32_t end) {
    attention_output_context *context = opaque;
    for (int32_t t = begin; t < end; t++) {
        ei_block_q8_0 qtmp[EI_N_EMBD / EI_QK];
        float proj[EI_N_EMBD];
        float *xrow = context->x + (size_t)t * EI_N_EMBD;
        ei_matmul_q4_0_f32(context->layer->attn_output,
                           context->ctx + (size_t)t * EI_N_EMBD, proj, qtmp);
        ei_rms_norm_residual_inplace(xrow, proj, context->layer->post_attention_norm,
                                     EI_N_EMBD, context->engine->model.rms_eps);
    }
}

typedef struct {
    const ei_engine *engine;
    const float *projection;
    float *residual;
    const float *weight;
} residual_context;

static void residual_rows(void *opaque, int32_t begin, int32_t end) {
    residual_context *context = opaque;
    for (int32_t token = begin; token < end; token++) {
        ei_rms_norm_residual_inplace(
            context->residual + (size_t)token * EI_N_EMBD,
            context->projection + (size_t)token * EI_N_EMBD,
            context->weight, EI_N_EMBD, context->engine->model.rms_eps);
    }
}

static void apply_attention_output_multirow(const ei_engine *e,
                                            const ei_layer *layer,
                                            int32_t tokens,
                                            const float *ctx, float *x) {
    quant_context quant = {
        .input = ctx, .output = e->workspace_q8,
        .width = EI_N_EMBD, .blocks = EI_N_EMBD / EI_QK,
    };
    ei_parallel_for(e->thread_pool, tokens, 1, quant_rows, &quant);
    parallel_batch_projection(e, layer->attn_output, e->workspace_q8,
                              tokens, e->workspace_attn_in);
    residual_context residual = {
        .engine = e, .projection = e->workspace_attn_in,
        .residual = x, .weight = layer->post_attention_norm,
    };
    ei_parallel_for(e->thread_pool, tokens, 1, residual_rows, &residual);
}

static void apply_attention_output(const ei_engine *e, const ei_layer *layer,
                                   int32_t n_tokens, const float *ctx, float *x) {
    if (n_tokens >= e->multirow_min_tokens) {
        apply_attention_output_multirow(e, layer, n_tokens, ctx, x);
        return;
    }
    if (n_tokens == 1 && ei_engine_threads(e) > 1) {
        ei_block_q8_0 input[EI_N_EMBD / EI_QK];
        float projection[EI_N_EMBD];
        ei_quantize_row_q8_0(ctx, input, EI_N_EMBD);
        parallel_q4_projection(e, layer->attn_output, input, projection);
        ei_rms_norm_residual_inplace(x, projection, layer->post_attention_norm,
                                     EI_N_EMBD, e->model.rms_eps);
        return;
    }
    attention_output_context context = {
        .engine = e, .layer = layer, .ctx = ctx, .x = x,
    };
    ei_parallel_for(e->thread_pool, n_tokens, 1, attention_output_rows, &context);
}

typedef struct {
    const ei_engine *engine;
    const ei_layer *layer;
    float *x;
} ffn_context;

static void ffn_rows(void *opaque, int32_t begin, int32_t end) {
    ffn_context *context = opaque;
    const ei_engine *e = context->engine;
    const ei_layer *layer = context->layer;
    for (int32_t t = begin; t < end; t++) {
        ei_block_q8_0 qtmp[EI_N_FF / EI_QK];
        float ffn_in[EI_N_EMBD];
        float up[EI_N_FF];
        float gate[EI_N_FF];
        float out[EI_N_EMBD];
        float *xrow = context->x + (size_t)t * EI_N_EMBD;
        if (e->fused_rms_quant) {
            ei_rms_norm_quantize_q8_0(xrow, layer->ffn_norm, EI_N_EMBD,
                                      e->model.rms_eps, qtmp);
        } else {
            ei_rms_norm(xrow, layer->ffn_norm, EI_N_EMBD, e->model.rms_eps, ffn_in);
            ei_quantize_row_q8_0(ffn_in, qtmp, EI_N_EMBD);
        }
        if (e->dual_projection) {
            ei_matmul_q4_0_q8_0_dual_rows(layer->ffn_up, layer->ffn_gate, qtmp,
                                          up, gate, 0, EI_N_FF);
        } else {
            ei_matmul_q4_0_q8_0(layer->ffn_up, qtmp, up);
            ei_matmul_q4_0_q8_0(layer->ffn_gate, qtmp, gate);
        }
        if (e->fused_gelu_quant) {
            ei_gelu_mul_quantize_q8_0(gate, up, EI_N_FF, qtmp);
            ei_matmul_q4_0_q8_0(layer->ffn_down, qtmp, out);
        } else {
            ei_gelu_mul_inplace(gate, up, EI_N_FF);
            ei_matmul_q4_0_f32(layer->ffn_down, gate, out, qtmp);
        }
        ei_rms_norm_residual_inplace(xrow, out, layer->post_ffw_norm,
                                     EI_N_EMBD, e->model.rms_eps);
    }
}

typedef struct {
    float *gate;
    const float *up;
    ei_block_q8_0 *output;
} gelu_context;

static void gelu_rows(void *opaque, int32_t begin, int32_t end) {
    gelu_context *context = opaque;
    for (int32_t token = begin; token < end; token++) {
        ei_gelu_mul_inplace(context->gate + (size_t)token * EI_N_FF,
                            context->up + (size_t)token * EI_N_FF, EI_N_FF);
    }
}

static void gelu_quant_rows(void *opaque, int32_t begin, int32_t end) {
    gelu_context *context = opaque;
    for (int32_t token = begin; token < end; token++) {
        ei_gelu_mul_quantize_q8_0(
            context->gate + (size_t)token * EI_N_FF,
            context->up + (size_t)token * EI_N_FF,
            EI_N_FF, context->output + (size_t)token * (EI_N_FF / EI_QK));
    }
}

static void apply_ffn_multirow(const ei_engine *e, const ei_layer *layer,
                               int32_t tokens, float *x) {
    norm_quant_context input_quant = {
        .input = x, .weight = layer->ffn_norm,
        .output = e->workspace_q8, .width = EI_N_EMBD,
        .blocks = EI_N_EMBD / EI_QK, .eps = e->model.rms_eps,
    };
    ei_parallel_for(e->thread_pool, tokens, 1, norm_quant_rows, &input_quant);
    parallel_batch_projection(e, layer->ffn_up, e->workspace_q8,
                              tokens, e->workspace_up);
    parallel_batch_projection(e, layer->ffn_gate, e->workspace_q8,
                              tokens, e->workspace_gate);
    gelu_context gelu = {
        .gate = e->workspace_gate, .up = e->workspace_up,
        .output = e->workspace_q8,
    };
    if (e->fused_gelu_quant) {
        ei_parallel_for(e->thread_pool, tokens, 1, gelu_quant_rows, &gelu);
    } else {
        ei_parallel_for(e->thread_pool, tokens, 1, gelu_rows, &gelu);
        quant_context gate_quant = {
            .input = e->workspace_gate, .output = e->workspace_q8,
            .width = EI_N_FF, .blocks = EI_N_FF / EI_QK,
        };
        ei_parallel_for(e->thread_pool, tokens, 1, quant_rows, &gate_quant);
    }
    parallel_batch_projection(e, layer->ffn_down, e->workspace_q8,
                              tokens, e->workspace_ctx);
    residual_context residual = {
        .engine = e, .projection = e->workspace_ctx,
        .residual = x, .weight = layer->post_ffw_norm,
    };
    ei_parallel_for(e->thread_pool, tokens, 1, residual_rows, &residual);
}

static void apply_ffn(const ei_engine *e, const ei_layer *layer,
                      int32_t n_tokens, float *x) {
    if (n_tokens >= e->multirow_min_tokens) {
        apply_ffn_multirow(e, layer, n_tokens, x);
        return;
    }
    if (n_tokens == 1 && ei_engine_threads(e) > 1) {
        ei_block_q8_0 input[EI_N_FF / EI_QK];
        float ffn_in[EI_N_EMBD];
        float up[EI_N_FF];
        float gate[EI_N_FF];
        float output[EI_N_EMBD];
        if (e->fused_rms_quant) {
            ei_rms_norm_quantize_q8_0(x, layer->ffn_norm, EI_N_EMBD,
                                      e->model.rms_eps, input);
        } else {
            ei_rms_norm(x, layer->ffn_norm, EI_N_EMBD, e->model.rms_eps, ffn_in);
            ei_quantize_row_q8_0(ffn_in, input, EI_N_EMBD);
        }
        up_gate_projection_context projection = { layer, input, up, gate };
        int32_t width = e->short_projection_threads;
        int32_t projection_rows = e->dual_projection ? EI_N_FF : 2 * EI_N_FF;
        ei_parallel_for(e->thread_pool, projection_rows,
                        (projection_rows + width - 1) / width,
                        e->dual_projection ? up_gate_projection_rows_dual
                                           : up_gate_projection_rows,
                        &projection);
        if (e->fused_gelu_quant) {
            ei_gelu_mul_quantize_q8_0(gate, up, EI_N_FF, input);
        } else {
            ei_gelu_mul_inplace(gate, up, EI_N_FF);
            ei_quantize_row_q8_0(gate, input, EI_N_FF);
        }
        parallel_q4_projection(e, layer->ffn_down, input, output);
        ei_rms_norm_residual_inplace(x, output, layer->post_ffw_norm,
                                     EI_N_EMBD, e->model.rms_eps);
        return;
    }
    ffn_context context = { .engine = e, .layer = layer, .x = x };
    ei_parallel_for(e->thread_pool, n_tokens, 1, ffn_rows, &context);
}

typedef struct {
    const float *q;
    const float *k;
    const float *v;
    float *ctx;
    int32_t tokens;
    int32_t window;
} attention_context;

static void attention_rows(void *opaque, int32_t begin, int32_t end) {
    attention_context *context = opaque;
    float scores[EI_N_CTX];
    ei_attention_mha_range(context->q, context->k, context->v,
                           context->tokens, context->window, begin, end,
                           context->ctx, scores);
}

typedef struct {
    const float *q;
    const float *k;
    const float *v;
    float *ctx;
    const size_t *offsets;
    size_t batch_size;
    int32_t window;
} batch_attention_context;

static void batch_attention_rows(void *opaque, int32_t begin, int32_t end) {
    batch_attention_context *context = opaque;
    float scores[EI_N_CTX];
    size_t sequence = 0;
    while (sequence + 1 < context->batch_size &&
           context->offsets[sequence + 1] <= (size_t)begin) {
        sequence++;
    }
    while (begin < end) {
        size_t start = context->offsets[sequence];
        size_t stop = context->offsets[sequence + 1];
        int32_t segment_end = end < (int32_t)stop ? end : (int32_t)stop;
        int32_t tokens = (int32_t)(stop - start);
        ei_attention_mha_range(
            context->q + start * EI_N_EMBD,
            context->k + start * EI_HEAD_DIM,
            context->v + start * EI_HEAD_DIM,
            tokens, context->window, begin - (int32_t)start,
            segment_end - (int32_t)start,
            context->ctx + start * EI_N_EMBD, scores);
        begin = segment_end;
        sequence++;
    }
}

static void ensure_workspace(ei_engine *e, size_t n_tokens) {
    if (e->workspace_tokens >= n_tokens) return;
    size_t capacity = e->workspace_tokens ? e->workspace_tokens : 16;
    while (capacity < n_tokens) {
        if (capacity > SIZE_MAX / 2) ei_die("activation workspace is too large");
        capacity *= 2;
    }
    e->workspace_positions = ei_xrealloc(
        e->workspace_positions, sizeof(*e->workspace_positions) * capacity);
    e->workspace_x = ei_xrealloc(e->workspace_x, sizeof(float) * capacity * EI_N_EMBD);
    e->workspace_attn_in = ei_xrealloc(e->workspace_attn_in, sizeof(float) * capacity * EI_N_EMBD);
    e->workspace_q = ei_xrealloc(e->workspace_q, sizeof(float) * capacity * EI_N_EMBD);
    e->workspace_k = ei_xrealloc(e->workspace_k, sizeof(float) * capacity * EI_HEAD_DIM);
    e->workspace_v = ei_xrealloc(e->workspace_v, sizeof(float) * capacity * EI_HEAD_DIM);
    e->workspace_ctx = ei_xrealloc(e->workspace_ctx, sizeof(float) * capacity * EI_N_EMBD);
    e->workspace_up = ei_xrealloc(e->workspace_up, sizeof(float) * capacity * EI_N_FF);
    e->workspace_gate = ei_xrealloc(e->workspace_gate, sizeof(float) * capacity * EI_N_FF);
    e->workspace_q8 = ei_xrealloc(
        e->workspace_q8,
        sizeof(*e->workspace_q8) * capacity * (EI_N_FF / EI_QK));
    e->workspace_tokens = capacity;
}

bool ei_engine_reserve(ei_engine *e, size_t total_tokens, size_t batch_size,
                       char *err, size_t err_len) {
    if (!e || total_tokens == 0 || total_tokens > 65536 ||
        batch_size == 0 || batch_size > 256) {
        set_err(err, err_len, "workspace reservation is outside supported limits");
        return false;
    }
#if defined(EI_ENABLE_METAL)
    if (e->metal) {
        return ei_metal_engine_reserve(e->metal, total_tokens, batch_size,
                                       err, err_len);
    }
#endif
    ensure_workspace(e, total_tokens);
    if (err && err_len) err[0] = '\0';
    return true;
}

bool ei_engine_embed_tokens(ei_engine *e, const int32_t *ids, size_t n_tokens,
                            float out[EI_N_EMBD], char *err, size_t err_len) {
    if (!token_ids_valid(ids, n_tokens, err, err_len)) return false;

#if defined(EI_ENABLE_METAL)
    if (e->metal) {
        return ei_metal_engine_embed_tokens(e->metal, ids, n_tokens, out, err, err_len);
    }
#endif

    int32_t T = (int32_t)n_tokens;
    ensure_workspace(e, n_tokens);
    float *x = e->workspace_x;
    float *attn_in = e->workspace_attn_in;
    float *q = e->workspace_q;
    float *k = e->workspace_k;
    float *v = e->workspace_v;
    float *ctx = e->workspace_ctx;

    input_embeddings(&e->model, ids, T, x);

    for (int il = 0; il < EI_N_LAYER; il++) {
        const ei_layer *L = &e->model.layers[il];
        bool is_swa = ei_layer_is_swa(&e->model, il);
        build_qkv(e, L, T, x, attn_in, q, k, v);
        attention_context attention = {
            .q = q, .k = k, .v = v, .ctx = ctx, .tokens = T,
            .window = is_swa ? (int32_t)e->model.swa_window : 0,
        };
        ei_parallel_for(e->thread_pool, T, 1, attention_rows, &attention);
        apply_attention_output(e, L, T, ctx, x);
        apply_ffn(e, L, T, x);
    }

    ei_mean_pool_rms_l2(x, T, e->model.output_norm, e->model.rms_eps, out);

    if (err && err_len) err[0] = '\0';
    return true;
}

bool ei_engine_embed_tokens_batch(ei_engine *e, const int32_t *ids,
                                  const size_t *offsets, size_t batch_size,
                                  float *out, char *err, size_t err_len) {
    if (!offsets || batch_size == 0 || offsets[0] != 0) {
        set_err(err, err_len, "batch offsets must start at zero");
        return false;
    }
    for (size_t sequence = 0; sequence < batch_size; sequence++) {
        size_t start = offsets[sequence];
        size_t stop = offsets[sequence + 1];
        if (stop < start || !token_ids_valid(ids + start, stop - start, err, err_len)) {
            return false;
        }
    }
    size_t total_tokens = offsets[batch_size];
    if (total_tokens > INT32_MAX) {
        set_err(err, err_len, "batch has too many total tokens");
        return false;
    }
    if (batch_size == 1) {
        return ei_engine_embed_tokens(e, ids, total_tokens, out, err, err_len);
    }

#if defined(EI_ENABLE_METAL)
    if (e->metal) {
        return ei_metal_engine_embed_tokens_batch(
            e->metal, ids, offsets, batch_size, out, err, err_len);
    }
#endif

    // Two one-token CPU requests do not amortize the packed-path setup.
    if (total_tokens < 4) {
        for (size_t sequence = 0; sequence < batch_size; sequence++) {
            size_t start = offsets[sequence];
            size_t count = offsets[sequence + 1] - start;
            if (!ei_engine_embed_tokens(e, ids + start, count,
                                        out + sequence * EI_N_EMBD,
                                        err, err_len)) {
                return false;
            }
        }
        return true;
    }

    ensure_workspace(e, total_tokens);
    int32_t *positions = e->workspace_positions;
    for (size_t sequence = 0; sequence < batch_size; sequence++) {
        for (size_t token = offsets[sequence]; token < offsets[sequence + 1]; token++) {
            positions[token] = (int32_t)(token - offsets[sequence]);
        }
    }

    int32_t total = (int32_t)total_tokens;
    float *x = e->workspace_x;
    float *attn_in = e->workspace_attn_in;
    float *q = e->workspace_q;
    float *k = e->workspace_k;
    float *v = e->workspace_v;
    float *ctx = e->workspace_ctx;
    input_embeddings(&e->model, ids, total, x);

    for (int il = 0; il < EI_N_LAYER; il++) {
        const ei_layer *layer = &e->model.layers[il];
        build_qkv_batch(e, layer, total, positions, x, attn_in, q, k, v);
        batch_attention_context attention = {
            .q = q, .k = k, .v = v, .ctx = ctx,
            .offsets = offsets, .batch_size = batch_size,
            .window = ei_layer_is_swa(&e->model, il)
                ? (int32_t)e->model.swa_window : 0,
        };
        ei_parallel_for(e->thread_pool, total, 1, batch_attention_rows, &attention);
        apply_attention_output(e, layer, total, ctx, x);
        apply_ffn(e, layer, total, x);
    }

    for (size_t sequence = 0; sequence < batch_size; sequence++) {
        size_t start = offsets[sequence];
        ei_mean_pool_rms_l2(x + start * EI_N_EMBD,
                            (int32_t)(offsets[sequence + 1] - start),
                            e->model.output_norm, e->model.rms_eps,
                            out + sequence * EI_N_EMBD);
    }
    if (err && err_len) err[0] = '\0';
    return true;
}

bool ei_engine_embed(ei_engine *e, const char *text, size_t len, float out[EI_N_EMBD],
                     char *err, size_t err_len) {
    ei_tokens toks;
    ei_tokenize_spm(&e->tokenizer, text, len, true, false, &toks);
    bool ok = ei_engine_embed_tokens(e, toks.ids, toks.n, out, err, err_len);
    ei_tokens_free(&toks);
    return ok;
}
