#include "gemma2.h"
#include "kernels.h"
#include <math.h>

/* Tensor storage is [input, output]; activations are contiguous token rows. */
static const ei_tensor *tensor(const ei_gguf *g, const char *name) {
    return ei_gguf_tensor(g, name, true);
}
static float *alloc_f(size_t n) { return ei_xcalloc(n, sizeof(float)); }
static float value(const ei_tensor *t, size_t i) {
    if (t->type == EI_T_F32)
        return ((const float *)t->data)[i];
    if (t->type == EI_T_BF16) {
        uint32_t bits = (uint32_t)((const uint16_t *)t->data)[i] << 16;
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }
    if (t->type == EI_T_F16)
        return ei_fp16_to_fp32(((const ei_fp16 *)t->data)[i]);
    const ei_block_q8_0 *b = (const ei_block_q8_0 *)t->data + i / 32;
    return ei_fp16_to_fp32(b->d) * b->qs[i % 32];
}
static void shape(const ei_gguf *g, const char *name, int k, int m) {
    const ei_tensor *t = tensor(g, name);
    if (t->ne[0] != (uint64_t)k || t->ne[1] != (uint64_t)m || t->ne[2] != 1 || t->ne[3] != 1 ||
        (t->type != EI_T_Q8_0 && t->type != EI_T_F32 && t->type != EI_T_BF16))
        ei_die("EmbeddingGemma 2: unsupported tensor %s", name);
}
void ei_gemma2_validate(const ei_gguf *g) {
    const char *keys[] = {"gemma-embedding2.block_count", "gemma-embedding2.embedding_length",
                          "gemma-embedding2.feed_forward_length",
                          "gemma-embedding2.embedding_length_out",
                          "gemma-embedding2.attention.sliding_window"};
    const uint64_t expected[] = {24, 512, 2048, 768, 1024};
    for (size_t j = 0; j < sizeof(expected) / sizeof(expected[0]); ++j)
        if (ei_gguf_kv_u64(g, keys[j], 0) != expected[j])
            ei_die("EmbeddingGemma 2: unsupported hyperparameter %s", keys[j]);
    if (tensor(g, "token_embd.weight")->type != EI_T_Q8_0)
        ei_die("EmbeddingGemma 2 requires Q8_0 token embeddings");
    shape(g, "token_embd.weight", 512, 262144);
    shape(g, "output.weight", 512, 768);
    shape(g, "output_norm.weight", 512, 1);
    shape(g, "per_layer_model_proj.weight", 512, 24 * 512);
    shape(g, "per_layer_proj_norm.weight", 512, 1);
    for (int l = 0; l < 24; ++l) {
        int d = l % 6 == 5 ? 512 : 256;
        const char *names[] = {
            "attn_norm",         "attn_q",      "attn_k",      "attn_v",
            "attn_output",       "attn_q_norm", "attn_k_norm", "post_attention_norm",
            "ffn_norm",          "ffn_up",      "ffn_gate",    "ffn_down",
            "post_ffw_norm",     "inp_gate",    "proj",        "post_norm",
            "layer_output_scale"};
        int k[] = {512, 512, 512,  512, 4 * d, d,   d,   512, 512,
                   512, 512, 2048, 512, 512,   512, 512, 1};
        int m[] = {1, 4 * d, 512, 512, 512, 1, 1, 1, 1, 2048, 2048, 512, 1, 512, 512, 1, 1};
        for (size_t j = 0; j < sizeof(k) / sizeof(k[0]); ++j) {
            char name[128];
            snprintf(name, sizeof name, "blk.%d.%s.weight", l, names[j]);
            shape(g, name, k[j], m[j]);
        }
    }
}
void ei_gemma2_validate_projector(const ei_gguf *g) {
    if (!ei_gguf_kv_bool(g, "clip.has_vision_encoder", false) ||
        !ei_gguf_kv_bool(g, "clip.has_audio_encoder", false))
        ei_die("EmbeddingGemma 2 projector must contain both vision and audio encoders");
    shape(g, "mm.input_projection.weight", 768, 512);
    shape(g, "mm.a.input_projection.weight", 1536, 512);
    shape(g, "a.input_projection.weight", 1024, 1024);
    shape(g, "a.pre_encode.out.weight", 1024, 1536);
    shape(g, "a.pre_encode.out.bias", 1536, 1);
    const char *special[] = {"v.patch_embd.weight", "v.position_embd.weight", "a.conv1d.0.weight",
                             "a.conv1d.1.weight"};
    const uint64_t sizes[][4] = {
        {16, 16, 3, 768}, {768, 10240, 2, 1}, {3, 3, 1, 128}, {3, 3, 128, 32}};
    for (int i = 0; i < 4; ++i) {
        const ei_tensor *t = tensor(g, special[i]);
        if (t->type != EI_T_F32 || memcmp(t->ne, sizes[i], sizeof(t->ne)))
            ei_die("EmbeddingGemma 2: unsupported projector tensor %s", special[i]);
    }
    shape(g, "a.conv1d.0.norm.weight", 128, 1);
    shape(g, "a.conv1d.1.norm.weight", 32, 1);
    const char *vn[] = {"ln1",         "ln2",      "attn_post_norm", "ffn_post_norm", "attn_q_norm",
                        "attn_k_norm", "attn_q",   "attn_k",         "attn_v",        "attn_out",
                        "ffn_up",      "ffn_gate", "ffn_down"};
    const int vk[] = {768, 768, 768, 768, 64, 64, 768, 768, 768, 768, 768, 768, 3072};
    const int vm[] = {1, 1, 1, 1, 1, 1, 768, 768, 768, 768, 3072, 3072, 768};
    const char *an[] = {"ffn_norm",      "ffn_post_norm",  "ffn_norm_1", "ffn_post_norm_1",
                        "attn_pre_norm", "attn_post_norm", "conv_norm",  "norm_conv",
                        "ln2",           "per_dim_scale",  "ffn_up",     "ffn_down",
                        "ffn_up_1",      "ffn_down_1",     "attn_q",     "attn_k",
                        "attn_v",        "attn_out",       "attn_k_rel", "conv_pw1",
                        "conv_pw2",      "conv_dw"};
    const int ak[] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 128,  1024,
                      4096, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 5};
    const int am[] = {1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    4096,
                      1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 2048, 1024, 1024};
    for (int l = 0; l < 16; ++l)
        for (size_t j = 0; j < sizeof(vk) / sizeof(vk[0]); ++j) {
            char name[128];
            snprintf(name, sizeof name, "v.blk.%d.%s.weight", l, vn[j]);
            shape(g, name, vk[j], vm[j]);
        }
    for (int l = 0; l < 12; ++l)
        for (size_t j = 0; j < sizeof(ak) / sizeof(ak[0]); ++j) {
            char name[128];
            snprintf(name, sizeof name, "a.blk.%d.%s.weight", l, an[j]);
            shape(g, name, ak[j], am[j]);
        }
}

typedef struct {
    const ei_tensor *w;
    const float *x;
    const ei_block_q8_0 *xq;
    float *y;
    int k, m;
} mm_job;
static void mm_rows(void *opaque, int32_t begin, int32_t end) {
    mm_job *j = opaque;
    for (int at = begin; at < end; ++at) {
        int token = at / j->m, row = at % j->m;
        float sum = 0;
        if (j->w->type == EI_T_Q8_0) {
            const ei_block_q8_0 *a = (const ei_block_q8_0 *)j->w->data + (size_t)row * j->k / 32;
            const ei_block_q8_0 *b = j->xq + (size_t)token * j->k / 32;
            for (int p = 0; p < j->k / 32; ++p) {
                int dot = 0;
                for (int z = 0; z < 32; ++z)
                    dot += (int)a[p].qs[z] * (int)b[p].qs[z];
                sum += (float)dot * ei_fp16_to_fp32(a[p].d) * ei_fp16_to_fp32(b[p].d);
            }
        } else {
            const float *x = j->x + (size_t)token * j->k;
            for (int z = 0; z < j->k; ++z)
                sum += value(j->w, (size_t)row * j->k + z) * x[z];
        }
        j->y[at] = sum;
    }
}
static void mm(ei_thread_pool *pool, const ei_tensor *w, const float *x, int n, float *y) {
    int k = (int)w->ne[0], m = (int)w->ne[1];
    ei_block_q8_0 *q = NULL;
    if (w->type == EI_T_Q8_0) {
        q = ei_xcalloc((size_t)n * k / 32, sizeof(*q));
        for (int t = 0; t < n; ++t)
            ei_quantize_row_q8_0(x + (size_t)t * k, q + (size_t)t * k / 32, k);
    }
    mm_job job = {w, x, q, y, k, m};
    ei_parallel_for(pool, n * m, 32, mm_rows, &job);
    free(q);
}
static void norm(float *x, int n, int d, const ei_tensor *w) {
    for (int t = 0; t < n; ++t) {
        float *r = x + (size_t)t * d;
        double ss = 0;
        for (int z = 0; z < d; ++z)
            ss += (double)r[z] * r[z];
        float scale = 1.0f / sqrtf((float)(ss / d) + 1e-6f);
        for (int z = 0; z < d; ++z)
            r[z] *= scale * (w ? value(w, z) : 1.0f);
    }
}
static void rope(float *x, int n, int heads, int d, float base) {
    for (int t = 0; t < n; ++t)
        for (int h = 0; h < heads; ++h) {
            float *r = x + ((size_t)t * heads + h) * d;
            for (int z = 0; z < d / 2; ++z) {
                float theta = t * powf(base, -2.0f * z / d), c = cosf(theta), s = sinf(theta);
                float a = r[z], b = r[z + d / 2];
                r[z] = a * c - b * s;
                r[z + d / 2] = a * s + b * c;
            }
        }
}
typedef struct {
    const float *q, *k, *v;
    float *out;
    int n, heads, kv, d, window;
} attn_job;
static void attention_rows(void *opaque, int32_t begin, int32_t end) {
    attn_job *a = opaque;
    float *scores = alloc_f(a->n);
    for (int at = begin; at < end; ++at) {
        int t = at / a->heads, h = at % a->heads, kh = h / (a->heads / a->kv);
        int lo = a->window && t > a->window ? t - a->window : 0;
        int hi = a->window && t + a->window + 1 < a->n ? t + a->window + 1 : a->n;
        const float *q = a->q + (size_t)at * a->d;
        float max = -INFINITY;
        for (int s = lo; s < hi; ++s) {
            const float *k = a->k + ((size_t)s * a->kv + kh) * a->d;
            float dot = 0;
            for (int z = 0; z < a->d; ++z)
                dot += q[z] * k[z];
            scores[s] = dot;
            if (dot > max)
                max = dot;
        }
        float total = 0;
        for (int s = lo; s < hi; ++s) {
            scores[s] = expf(scores[s] - max);
            total += scores[s];
        }
        float *out = a->out + (size_t)at * a->d;
        memset(out, 0, a->d * sizeof(float));
        for (int s = lo; s < hi; ++s) {
            float p = scores[s] / total;
            const float *v = a->v + ((size_t)s * a->kv + kh) * a->d;
            for (int z = 0; z < a->d; ++z)
                out[z] += p * v[z];
        }
    }
    free(scores);
}
static void attention(ei_thread_pool *pool, const float *q, const float *k, const float *v, int n,
                      int heads, int kv, int d, int window, float *out) {
    attn_job j = {q, k, v, out, n, heads, kv, d, window};
    ei_parallel_for(pool, n * heads, 1, attention_rows, &j);
}
bool ei_gemma2_embed(const ei_gguf *g, ei_thread_pool *pool, const int32_t *ids,
                     const float *inputs, size_t count, float out[768], char *err, size_t err_len) {
    if (!count || count > 8192 || (!ids && !inputs)) {
        snprintf(err, err_len, "EmbeddingGemma 2 accepts 1..8192 input tokens");
        return false;
    }
    int n = (int)count;
    float *x = alloc_f(count * 512), *h = alloc_f(count * 512), *q = alloc_f(count * 2048),
          *k = alloc_f(count * 512), *v = alloc_f(count * 512);
    float *a = alloc_f(count * 2048), *up = alloc_f(count * 2048), *gate = alloc_f(count * 2048),
          *ple = alloc_f(count * 512 * 24);
    const ei_tensor *emb = tensor(g, "token_embd.weight");
    for (int t = 0; t < n; ++t)
        for (int z = 0; z < 512; ++z)
            x[(size_t)t * 512 + z] = inputs ? inputs[(size_t)t * 512 + z]
                                            : value(emb, (size_t)ids[t] * 512 + z) * sqrtf(512.0f);
    mm(pool, tensor(g, "per_layer_model_proj.weight"), x, n, ple);
    for (size_t i = 0; i < count * 512 * 24; ++i)
        ple[i] /= sqrtf(512.0f);
    norm(ple, n * 24, 512, tensor(g, "per_layer_proj_norm.weight"));
    for (int l = 0; l < 24; ++l) {
        char name[128];
#define T(suffix) (snprintf(name, sizeof name, "blk.%d." suffix ".weight", l), tensor(g, name))
        int d = l % 6 == 5 ? 512 : 256, kv = 512 / d;
        memcpy(h, x, count * 512 * sizeof(float));
        norm(h, n, 512, T("attn_norm"));
        mm(pool, T("attn_q"), h, n, q);
        mm(pool, T("attn_k"), h, n, k);
        mm(pool, T("attn_v"), h, n, v);
        norm(q, n * 4, d, T("attn_q_norm"));
        norm(k, n * kv, d, T("attn_k_norm"));
        norm(v, n * kv, d, NULL);
        float base = l % 6 == 5 ? 1000000.0f : 10000.0f;
        rope(q, n, 4, d, base);
        rope(k, n, kv, d, base);
        attention(pool, q, k, v, n, 4, kv, d, l % 6 == 5 ? 0 : 512, a);
        mm(pool, T("attn_output"), a, n, h);
        norm(h, n, 512, T("post_attention_norm"));
        for (size_t i = 0; i < count * 512; ++i)
            x[i] += h[i];
        memcpy(h, x, count * 512 * sizeof(float));
        norm(h, n, 512, T("ffn_norm"));
        mm(pool, T("ffn_up"), h, n, up);
        mm(pool, T("ffn_gate"), h, n, gate);
        for (size_t i = 0; i < count * 2048; ++i)
            gate[i] = ei_gelu_tanh(gate[i]) * up[i];
        mm(pool, T("ffn_down"), gate, n, h);
        norm(h, n, 512, T("post_ffw_norm"));
        for (size_t i = 0; i < count * 512; ++i)
            x[i] += h[i];
        mm(pool, T("inp_gate"), x, n, h);
        for (int t = 0; t < n; ++t)
            for (int z = 0; z < 512; ++z)
                h[(size_t)t * 512 + z] =
                    ei_gelu_tanh(h[(size_t)t * 512 + z]) * ple[((size_t)t * 24 + l) * 512 + z];
        mm(pool, T("proj"), h, n, a);
        norm(a, n, 512, T("post_norm"));
        float scale = value(T("layer_output_scale"), 0);
        for (size_t i = 0; i < count * 512; ++i)
            x[i] = (x[i] + a[i]) * scale;
#undef T
    }
    norm(x, n, 512, tensor(g, "output_norm.weight"));
    mm(pool, tensor(g, "output.weight"), x, n, a);
    memset(out, 0, 768 * sizeof(float));
    for (int t = 0; t < n; ++t)
        for (int z = 0; z < 768; ++z)
            out[z] += a[(size_t)t * 768 + z] / n;
    ei_l2_normalize(out, 768);
    free(x);
    free(h);
    free(q);
    free(k);
    free(v);
    free(a);
    free(up);
    free(gate);
    free(ple);
    return true;
}

/* Gemma4 vision tower: RGB patch projection, axial RoPE, 3x3 mean pool. */
bool ei_gemma2_vision(const ei_gguf *g, ei_thread_pool *pool, const uint8_t *rgb, int width,
                      int height, float **output, size_t *tokens, char *err, size_t err_len) {
    if (width < 48 || height < 48 || width % 48 || height % 48 ||
        (size_t)width * height > 1120 * 48 * 48) {
        snprintf(err, err_len,
                 "vision input must be aligned to 48 pixels and at most 1120 soft tokens");
        return false;
    }
    int cols = width / 16, rows = height / 16, n = cols * rows;
    bool use_gelu = ei_gguf_kv_bool(g, "clip.use_gelu", false);
    float *patches = alloc_f((size_t)n * 768), *x = alloc_f((size_t)n * 768),
          *h = alloc_f((size_t)n * 768);
    float *q = alloc_f((size_t)n * 768), *k = alloc_f((size_t)n * 768),
          *v = alloc_f((size_t)n * 768);
    float *a = alloc_f((size_t)n * 768), *up = alloc_f((size_t)n * 3072),
          *gate = alloc_f((size_t)n * 3072);
    for (int t = 0; t < n; ++t)
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < 16; ++y)
                for (int z = 0; z < 16; ++z)
                    patches[(size_t)t * 768 + c * 256 + y * 16 + z] =
                        2.0f *
                            rgb[((size_t)(t / cols * 16 + y) * width + t % cols * 16 + z) * 3 + c] /
                            255.0f -
                        1;
    ei_tensor patch = *tensor(g, "v.patch_embd.weight");
    patch.ne[0] = 768;
    patch.ne[1] = 768;
    for (size_t i = 0; i < (size_t)n * 768; ++i)
        patches[i] = ei_fp16_to_fp32(ei_fp32_to_fp16(patches[i]));
    mm(pool, &patch, patches, n, x);
    free(patches);
    const ei_tensor *pos = tensor(g, "v.position_embd.weight");
    for (int t = 0; t < n; ++t)
        for (int z = 0; z < 768; ++z)
            x[(size_t)t * 768 + z] += value(pos, (size_t)(t % cols) * 768 + z) +
                                      value(pos, ((size_t)10240 + t / cols) * 768 + z);
    for (int l = 0; l < 16; ++l) {
        char name[128];
#define V(suffix) (snprintf(name, sizeof name, "v.blk.%d." suffix ".weight", l), tensor(g, name))
        memcpy(h, x, (size_t)n * 768 * sizeof(float));
        norm(h, n, 768, V("ln1"));
        mm(pool, V("attn_q"), h, n, q);
        mm(pool, V("attn_k"), h, n, k);
        mm(pool, V("attn_v"), h, n, v);
        norm(q, n * 12, 64, V("attn_q_norm"));
        norm(k, n * 12, 64, V("attn_k_norm"));
        norm(v, n * 12, 64, NULL);
        for (int t = 0; t < n; ++t)
            for (int head = 0; head < 12; ++head)
                for (int axis = 0; axis < 2; ++axis)
                    for (int z = 0; z < 16; ++z) {
                        int p = axis ? t / cols : t % cols;
                        float theta = p * powf(100.0f, -2.0f * z / 32), co = cosf(theta),
                              si = sinf(theta);
                        size_t i = ((size_t)t * 12 + head) * 64 + axis * 32 + z;
                        float qa = q[i], qb = q[i + 16], ka = k[i], kb = k[i + 16];
                        q[i] = qa * co - qb * si;
                        q[i + 16] = qa * si + qb * co;
                        k[i] = ka * co - kb * si;
                        k[i + 16] = ka * si + kb * co;
                    }
        attention(pool, q, k, v, n, 12, 12, 64, 0, a);
        mm(pool, V("attn_out"), a, n, h);
        norm(h, n, 768, V("attn_post_norm"));
        for (size_t i = 0; i < (size_t)n * 768; ++i)
            x[i] += h[i];
        memcpy(h, x, (size_t)n * 768 * sizeof(float));
        norm(h, n, 768, V("ln2"));
        mm(pool, V("ffn_up"), h, n, up);
        mm(pool, V("ffn_gate"), h, n, gate);
        for (size_t i = 0; i < (size_t)n * 3072; ++i)
            gate[i] =
                (use_gelu ? ei_gelu_tanh(gate[i]) : gate[i] / (1.0f + expf(-1.702f * gate[i]))) *
                up[i];
        mm(pool, V("ffn_down"), gate, n, h);
        norm(h, n, 768, V("ffn_post_norm"));
        for (size_t i = 0; i < (size_t)n * 768; ++i)
            x[i] += h[i];
#undef V
    }
    int out_cols = cols / 3, out_rows = rows / 3, nt = out_cols * out_rows;
    memset(h, 0, (size_t)nt * 768 * sizeof(float));
    for (int t = 0; t < nt; ++t)
        for (int yy = 0; yy < 3; ++yy)
            for (int xx = 0; xx < 3; ++xx)
                for (int z = 0; z < 768; ++z)
                    h[(size_t)t * 768 + z] +=
                        x[((size_t)(t / out_cols * 3 + yy) * cols + t % out_cols * 3 + xx) * 768 +
                          z] /
                        9.0f * sqrtf(768.0f);
    norm(h, nt, 768, NULL);
    *output = alloc_f((size_t)nt * 512);
    *tokens = nt;
    mm(pool, tensor(g, "mm.input_projection.weight"), h, nt, *output);
    free(x);
    free(h);
    free(q);
    free(k);
    free(v);
    free(a);
    free(up);
    free(gate);
    return true;
}

static float scalar_or(const ei_gguf *g, const char *name, float fallback) {
    const ei_tensor *t = ei_gguf_tensor(g, name, false);
    return t ? value(t, 0) : fallback;
}
static void clipped_mm(const ei_gguf *g, ei_thread_pool *pool, const char *name, const float *x,
                       int n, float *y) {
    char key[160];
    size_t len = strlen(name) - strlen(".weight");
    snprintf(key, sizeof key, "%.*s.input_min", (int)len, name);
    float lo = scalar_or(g, key, -INFINITY);
    snprintf(key, sizeof key, "%.*s.input_max", (int)len, name);
    float hi = scalar_or(g, key, INFINITY);
    const ei_tensor *w = tensor(g, name);
    size_t count = (size_t)n * w->ne[0];
    float *clamped = alloc_f(count);
    for (size_t i = 0; i < count; ++i)
        clamped[i] = fminf(hi, fmaxf(lo, x[i]));
    mm(pool, w, clamped, n, y);
    free(clamped);
    snprintf(key, sizeof key, "%.*s.output_min", (int)len, name);
    lo = scalar_or(g, key, -INFINITY);
    snprintf(key, sizeof key, "%.*s.output_max", (int)len, name);
    hi = scalar_or(g, key, INFINITY);
    for (size_t i = 0; i < (size_t)n * w->ne[1]; ++i)
        y[i] = fminf(hi, fmaxf(lo, y[i]));
}
static float silu(float x) { return x / (1.0f + expf(-x)); }
/* Audio features are time-major, 128 HTK log-magnitude mel bins. */
bool ei_gemma2_audio(const ei_gguf *g, ei_thread_pool *pool, const float *mel, int frames,
                     float **output, size_t *tokens, char *err, size_t err_len) {
    if (frames < 1 || frames > 3000) {
        snprintf(err, err_len, "audio chunk must contain 1..3000 mel frames");
        return false;
    }
    int nt = frames, nf = 128, nc = 1;
    float *cur = alloc_f((size_t)nt * nf);
    memcpy(cur, mel, (size_t)nt * nf * sizeof(float));
    for (int layer = 0; layer < 2; ++layer) {
        char name[96];
        snprintf(name, sizeof name, "a.conv1d.%d.weight", layer);
        const ei_tensor *w = tensor(g, name);
        int ot = (nt + 1) / 2, of = (nf + 1) / 2, oc = (int)w->ne[3];
        float *next = alloc_f((size_t)ot * of * oc);
        for (int t = 0; t < ot; ++t)
            for (int f = 0; f < of; ++f)
                for (int o = 0; o < oc; ++o) {
                    float sum = 0;
                    for (int ky = 0; ky < 3; ++ky)
                        for (int kx = 0; kx < 3; ++kx) {
                            int ti = t * 2 + ky - 1, fi = f * 2 + kx - 1;
                            if (ti < 0 || ti >= nt || fi < 0 || fi >= nf)
                                continue;
                            for (int c = 0; c < nc; ++c)
                                sum += cur[((size_t)ti * nf + fi) * nc + c] *
                                       value(w, ((size_t)o * nc + c) * 9 + ky * 3 + kx);
                        }
                    next[((size_t)t * of + f) * oc + o] = sum;
                }
        snprintf(name, sizeof name, "a.conv1d.%d.norm.weight", layer);
        const ei_tensor *weight = tensor(g, name);
        for (int t = 0; t < ot * of; ++t) {
            float *r = next + (size_t)t * oc;
            double mean = 0, var = 0;
            for (int c = 0; c < oc; ++c)
                mean += r[c];
            mean /= oc;
            for (int c = 0; c < oc; ++c)
                var += (r[c] - mean) * (r[c] - mean);
            float scale = 1.0f / sqrtf((float)(var / oc) + 1e-6f);
            for (int c = 0; c < oc; ++c)
                r[c] = fmaxf(0, ((float)(r[c] - mean)) * scale * value(weight, c));
        }
        free(cur);
        cur = next;
        nt = ot;
        nf = of;
        nc = oc;
    }
    int n = nt;
    size_t count = (size_t)n * 1024;
    float *x = alloc_f(count), *h = alloc_f(count), *q = alloc_f(count), *k = alloc_f(count),
          *v = alloc_f(count), *a = alloc_f(count), *up = alloc_f((size_t)n * 4096);
    mm(pool, tensor(g, "a.input_projection.weight"), cur, n, x);
    free(cur);
    float *pos = alloc_f(13 * 1024), *rpe = alloc_f(13 * 1024);
    for (int p = 0; p < 13; ++p)
        for (int z = 0; z < 512; ++z) {
            float theta = (12 - p) * expf(-z * logf(10000.0f) / 511);
            pos[p * 1024 + z] = sinf(theta);
            pos[p * 1024 + z + 512] = cosf(theta);
        }
    for (int l = 0; l < 12; ++l) {
        char name[128];
#define A(suffix) (snprintf(name, sizeof name, "a.blk.%d." suffix ".weight", l), tensor(g, name))
#define AM(suffix, input, output)                                                                  \
    do {                                                                                           \
        snprintf(name, sizeof name, "a.blk.%d." suffix ".weight", l);                              \
        clipped_mm(g, pool, name, input, n, output);                                               \
    } while (0)
        memcpy(h, x, count * sizeof(float));
        norm(h, n, 1024, A("ffn_norm"));
        AM("ffn_up", h, up);
        for (size_t i = 0; i < (size_t)n * 4096; ++i)
            up[i] = silu(up[i]);
        AM("ffn_down", up, h);
        norm(h, n, 1024, A("ffn_post_norm"));
        for (size_t i = 0; i < count; ++i)
            x[i] += 0.5f * h[i];
        memcpy(h, x, count * sizeof(float));
        norm(h, n, 1024, A("attn_pre_norm"));
        AM("attn_q", h, q);
        AM("attn_k", h, k);
        AM("attn_v", h, v);
        const ei_tensor *qs = A("per_dim_scale");
        for (size_t i = 0; i < count; ++i) {
            q[i] *= value(qs, i % 128) / (sqrtf(128.0f) * logf(2.0f));
            k[i] *= log1pf(expf(1.0f)) / logf(2.0f);
        }
        mm(pool, A("attn_k_rel"), pos, 13, rpe);
        memset(a, 0, count * sizeof(float));
        for (int t = 0; t < n; ++t)
            for (int head = 0; head < 8; ++head) {
                float scores[12], mx = -INFINITY;
                int lo = t > 11 ? t - 11 : 0;
                for (int s = lo; s <= t; ++s) {
                    float dot = 0;
                    for (int z = 0; z < 128; ++z)
                        dot += q[(size_t)t * 1024 + head * 128 + z] *
                               (k[(size_t)s * 1024 + head * 128 + z] +
                                rpe[(12 - t + s) * 1024 + head * 128 + z]);
                    dot = 50 * tanhf(dot / 50);
                    scores[s - lo] = dot;
                    if (dot > mx)
                        mx = dot;
                }
                float total = 0;
                for (int s = lo; s <= t; ++s) {
                    scores[s - lo] = expf(scores[s - lo] - mx);
                    total += scores[s - lo];
                }
                for (int s = lo; s <= t; ++s)
                    for (int z = 0; z < 128; ++z)
                        a[(size_t)t * 1024 + head * 128 + z] +=
                            scores[s - lo] / total * v[(size_t)s * 1024 + head * 128 + z];
            }
        AM("attn_out", a, h);
        norm(h, n, 1024, A("attn_post_norm"));
        for (size_t i = 0; i < count; ++i)
            x[i] += h[i];
        /* GGUF conversion swaps norm_conv and conv_norm; match the reference loader. */
        memcpy(h, x, count * sizeof(float));
        norm(h, n, 1024, A("conv_norm"));
        AM("conv_pw1", h, up);
        for (int t = 0; t < n; ++t)
            for (int z = 0; z < 1024; ++z)
                h[(size_t)t * 1024 + z] =
                    up[(size_t)t * 2048 + z] / (1 + expf(-up[(size_t)t * 2048 + 1024 + z]));
        const ei_tensor *dw = A("conv_dw");
        for (int t = 0; t < n; ++t)
            for (int z = 0; z < 1024; ++z) {
                float sum = 0;
                for (int j = 0; j < 5; ++j)
                    if (t + j >= 4)
                        sum += h[(size_t)(t + j - 4) * 1024 + z] * value(dw, z * 5 + j);
                a[(size_t)t * 1024 + z] = sum;
            }
        norm(a, n, 1024, A("norm_conv"));
        for (size_t i = 0; i < count; ++i)
            a[i] = silu(a[i]);
        AM("conv_pw2", a, h);
        for (size_t i = 0; i < count; ++i)
            x[i] += h[i];
        memcpy(h, x, count * sizeof(float));
        norm(h, n, 1024, A("ffn_norm_1"));
        AM("ffn_up_1", h, up);
        for (size_t i = 0; i < (size_t)n * 4096; ++i)
            up[i] = silu(up[i]);
        AM("ffn_down_1", up, h);
        norm(h, n, 1024, A("ffn_post_norm_1"));
        for (size_t i = 0; i < count; ++i)
            x[i] += 0.5f * h[i];
        norm(x, n, 1024, A("ln2"));
#undef AM
#undef A
    }
    clipped_mm(g, pool, "a.pre_encode.out.weight", x, n, up);
    const ei_tensor *bias = tensor(g, "a.pre_encode.out.bias");
    for (int t = 0; t < n; ++t)
        for (int z = 0; z < 1536; ++z)
            up[(size_t)t * 1536 + z] += value(bias, z);
    norm(up, n, 1536, NULL);
    *output = alloc_f((size_t)n * 512);
    *tokens = n;
    clipped_mm(g, pool, "mm.a.input_projection.weight", up, n, *output);
    free(x);
    free(h);
    free(q);
    free(k);
    free(v);
    free(a);
    free(up);
    free(pos);
    free(rpe);
    return true;
}
