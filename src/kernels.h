#ifndef EI_KERNELS_H
#define EI_KERNELS_H

#include "model.h"
#include "quants.h"

void ei_rms_norm(const float *x, const float *w, int32_t n, float eps, float *out);
void ei_rms_norm_inplace(float *x, const float *w, int32_t n, float eps);
void ei_rms_norm_residual_inplace(float *residual, const float *x, const float *w,
                                  int32_t n, float eps);
void ei_rms_norm_quantize_q8_0(const float *x, const float *w, int32_t n,
                               float eps, ei_block_q8_0 *out);
void ei_rope_neox_inplace(float *x, int32_t n_heads, int32_t pos, float base);
void ei_qk_norm_rope_inplace(float *x, const float *w, int32_t n_heads,
                             int32_t pos, float base, float scale, float eps);
void ei_qk_norm_rope_qk_inplace(float *q, float *k, const float *q_weight,
                                const float *k_weight, int32_t pos,
                                float base, float eps);
void ei_attention_mha(const float *q, const float *k, const float *v,
                      int32_t n_tokens, int32_t window, float *ctx, float *scores);
void ei_attention_mha_range(const float *q, const float *k, const float *v,
                            int32_t n_tokens, int32_t window,
                            int32_t query_begin, int32_t query_end,
                            float *ctx, float *scores);
float ei_gelu_tanh(float x);
void ei_gelu_mul_inplace(float *gate, const float *up, int32_t n);
void ei_gelu_mul_quantize_q8_0(const float *gate, const float *up,
                               int32_t n, ei_block_q8_0 *out);
void ei_l2_normalize(float *x, int32_t n);
bool ei_embedding_dimensions_supported(int32_t dimensions);
bool ei_embedding_normalize_prefix(float embedding[EI_N_EMBD], int32_t dimensions);
void ei_mean_pool_rms_l2(float *x, int32_t n_tokens, const float *w,
                         float eps, float out[EI_N_EMBD]);
void ei_vec_add_inplace(float *dst, const float *src, int32_t n);
void ei_vec_zero(float *x, int32_t n);
const char *ei_cpu_kernel_variant(void);

#endif /* EI_KERNELS_H */
