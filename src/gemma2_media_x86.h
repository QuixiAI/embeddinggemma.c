/* cpu-max fork: SIMD kernels for the EmbeddingGemma 2 CPU media towers
 * (vision/audio). x86-64 AVX2+FMA bodies are selected at run time with
 * portable C fallbacks, so generic builds stay generic. */
#ifndef EI_GEMMA2_MEDIA_X86_H
#define EI_GEMMA2_MEDIA_X86_H
#include <stddef.h>
#include <stdint.h>

/* Key block of the tiled kernels. Query blocks depend on the head width and
 * kernel variant: callers must size tasks with g2m_attn_qblock(d). */
#ifndef G2M_KB
#define G2M_KB 64 /* compile-time key block (tuning experiments: -DG2M_KB=32|128) */
#endif
enum { G2M_ATTN_KB = G2M_KB };

/* Online-softmax attention for one head and a block of queries [q0, q0 + nq),
 * nq <= g2m_attn_qblock(d). Head width d: multiple of 16, <= 512 (fp32, any d
 * via the scalar path). Rows of q / k,v / out are qstride / kvstride / ostride
 * floats apart and the head starts at column qcol / kvcol / ocol. Query i sees
 * keys [lo[i], hi[i]) when lo/hi are given (sliding windows, packed
 * sequences), else [klo, khi); klo/khi must bound every lo/hi. Scores are
 * scaled by `scale`. Masked keys contribute exactly zero. */
#include <stdbool.h>
typedef struct {
    const float *q, *k, *v;
    float *out;
    size_t qstride, kvstride, ostride, qcol, kvcol, ocol;
    int d, q0, nq;
    const int32_t *lo, *hi;
    int klo, khi;
    float scale;
} g2m_attn_args;
int g2m_attn_qblock(int d);
void g2m_attn(const g2m_attn_args *a);
/* Scalar implementation (the non-AVX2 fallback; used by tests as the reference). */
void g2m_attn_reference(const g2m_attn_args *a);
/* Full bidirectional softmax attention, scale 1, for one 64-wide head:
 * rows of q/k/v/out are `stride` floats apart; the head starts at column
 * `col`. Keys [k0, k1) are visible to every query of the task. */
void g2m_attn64(const float *q, const float *k, const float *v, float *out, size_t stride, size_t col,
                int q0, int nq, int k0, int k1);
/* g[i] = act(g[i]) * u[i] (u NULL: act only); act = x * sigmoid(1.702 x) (QuickGELU). */
void g2m_quick_gelu_mul(float *g, const float *u, int n);
/* y[i] = x[i] / (1 + exp(-x[i])) (SiLU), in place when y == x. */
void g2m_silu(float *y, const float *x, int n);
/* The attention kernel's vector exp, in place (accuracy checked by test_gemma2_media). */
void g2m_exp_inplace(float *x, int n);
#endif
