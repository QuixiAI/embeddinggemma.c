/* cpu-max fork: SIMD kernels for the EmbeddingGemma 2 CPU media towers.
 *
 * Attention (g2m_attn64) is a tiled online-softmax ("flash") kernel written in
 * a transposed form so that every softmax step is a vertical vector op:
 *   Q^T   [64][QB]   packed once per task (QB queries, zero padded to 16)
 *   S^T   [KB][QB]   = K_blk . Q^T, K rows read in place (broadcast per key)
 *   P^T   = exp(S^T - m), running max m and sum l kept per query lane
 *   O^T   [64][QB]  += V_blk^T . P^T, V rows read in place (broadcast per dim)
 * so neither K nor V is ever copied or transposed; the only packed operand is
 * the task's Q block (16 KB). Arithmetic is fp32 throughout (no reduced
 * precision); only summation order differs from the scalar reference. */
#include "gemma2_media_x86.h"
#include "gemma2_fast_x86.h"

#include <math.h>
#include <string.h>

/* ---- portable fallbacks -------------------------------------------------- */

static void attn64_scalar(const float *q, const float *k, const float *v, float *out, size_t stride,
                          size_t col, int q0, int nq, int k0, int k1) {
    float scores[G2M_ATTN_KB];
    for (int i = 0; i < nq; i++) {
        const float *qi = q + (size_t)(q0 + i) * stride + col;
        float m = -INFINITY, l = 0, o[64] = {0};
        for (int b = k0; b < k1; b += G2M_ATTN_KB) {
            int kn = k1 - b < G2M_ATTN_KB ? k1 - b : G2M_ATTN_KB;
            float mb = -INFINITY;
            for (int j = 0; j < kn; j++) {
                const float *kj = k + (size_t)(b + j) * stride + col;
                float s = 0;
                for (int d = 0; d < 64; d++)
                    s += qi[d] * kj[d];
                scores[j] = s;
                mb = s > mb ? s : mb;
            }
            float mn = mb > m ? mb : m, alpha = expf(m - mn);
            l *= alpha;
            for (int d = 0; d < 64; d++)
                o[d] *= alpha;
            for (int j = 0; j < kn; j++) {
                float p = expf(scores[j] - mn);
                l += p;
                const float *vj = v + (size_t)(b + j) * stride + col;
                for (int d = 0; d < 64; d++)
                    o[d] += p * vj[d];
            }
            m = mn;
        }
        float *oi = out + (size_t)(q0 + i) * stride + col;
        for (int d = 0; d < 64; d++)
            oi[d] = o[d] / l;
    }
}

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define AVX2 __attribute__((target("avx2,fma,f16c")))
/* Fixed-size register tiles must unroll at the upstream -O2 (GCC keeps such
 * loops rolled there and spills the accumulator arrays). */
#define UNROLL _Pragma("GCC unroll 8")

/* exp via x = n ln2 + r, |r| <= ln2/2, degree-7 Taylor in Horner form.
 * Inputs below -87.3 return ~1.2e-38 (2^-126), never a denormal or NaN. */
AVX2 static inline __m256 exp8(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.3f)), _mm256_set1_ps(88.7f));
    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.0f / 5040);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 720));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 120));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 24));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 6));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

AVX2 static void attn64_avx2(const float *q, const float *k, const float *v, float *out, size_t stride,
                             size_t col, int q0, int nq, int k0, int k1) {
    enum { QB = G2M_ATTN_QB, KB = G2M_ATTN_KB };
    _Alignas(32) float qt[64][QB];
    _Alignas(32) float ot[64][QB];
    _Alignas(32) float st[KB][QB];
    _Alignas(32) float m[QB], l[QB], alpha[QB];
    const int np = (nq + 15) & ~15; /* query lanes, padded to whole 16-wide tiles */
    for (int d = 0; d < 64; d++)
        for (int i = 0; i < np; i++)
            qt[d][i] = i < nq ? q[(size_t)(q0 + i) * stride + col + d] : 0.0f;
    memset(ot, 0, sizeof ot);
    for (int i = 0; i < np; i++)
        m[i] = -INFINITY, l[i] = 0;

    for (int b = k0; b < k1; b += KB) {
        const int kn = k1 - b < KB ? k1 - b : KB;
        /* S^T = K_blk . Q^T: 4 keys x 16 queries per register tile. */
        for (int j0 = 0; j0 < kn; j0 += 4) {
            const float *kr[4];
            UNROLL for (int c = 0; c < 4; c++)
                kr[c] = k + (size_t)(b + (j0 + c < kn ? j0 + c : kn - 1)) * stride + col;
            for (int i0 = 0; i0 < np; i0 += 16) {
                __m256 acc[4][2];
                UNROLL for (int c = 0; c < 4; c++)
                    acc[c][0] = acc[c][1] = _mm256_setzero_ps();
                _Pragma("GCC unroll 4") for (int d = 0; d < 64; d++) {
                    __m256 a0 = _mm256_load_ps(&qt[d][i0]), a1 = _mm256_load_ps(&qt[d][i0 + 8]);
                    UNROLL for (int c = 0; c < 4; c++) {
                        __m256 kb = _mm256_broadcast_ss(kr[c] + d);
                        acc[c][0] = _mm256_fmadd_ps(kb, a0, acc[c][0]);
                        acc[c][1] = _mm256_fmadd_ps(kb, a1, acc[c][1]);
                    }
                }
                for (int c = 0; c < 4 && j0 + c < kn; c++) {
                    _mm256_store_ps(&st[j0 + c][i0], acc[c][0]);
                    _mm256_store_ps(&st[j0 + c][i0 + 8], acc[c][1]);
                }
            }
        }
        /* Online softmax, 8 query lanes at a time. */
        for (int i0 = 0; i0 < np; i0 += 8) {
            __m256 mb = _mm256_set1_ps(-INFINITY);
            for (int j = 0; j < kn; j++)
                mb = _mm256_max_ps(mb, _mm256_load_ps(&st[j][i0]));
            __m256 mo = _mm256_load_ps(&m[i0]), mn = _mm256_max_ps(mo, mb);
            __m256 al = exp8(_mm256_sub_ps(mo, mn)), sum = _mm256_setzero_ps();
            for (int j = 0; j < kn; j++) {
                __m256 p = exp8(_mm256_sub_ps(_mm256_load_ps(&st[j][i0]), mn));
                _mm256_store_ps(&st[j][i0], p);
                sum = _mm256_add_ps(sum, p);
            }
            _mm256_store_ps(&l[i0], _mm256_fmadd_ps(_mm256_load_ps(&l[i0]), al, sum));
            _mm256_store_ps(&m[i0], mn);
            _mm256_store_ps(&alpha[i0], al);
        }
        /* O^T = O^T * alpha + V_blk^T . P^T: 4 dims x 16 queries per tile. */
        for (int d0 = 0; d0 < 64; d0 += 4)
            for (int i0 = 0; i0 < np; i0 += 16) {
                __m256 al0 = _mm256_load_ps(&alpha[i0]), al1 = _mm256_load_ps(&alpha[i0 + 8]);
                __m256 acc[4][2];
                UNROLL for (int c = 0; c < 4; c++) {
                    acc[c][0] = _mm256_mul_ps(_mm256_load_ps(&ot[d0 + c][i0]), al0);
                    acc[c][1] = _mm256_mul_ps(_mm256_load_ps(&ot[d0 + c][i0 + 8]), al1);
                }
                const float *vr = v + (size_t)b * stride + col + d0;
                for (int j = 0; j < kn; j++, vr += stride) {
                    __m256 p0 = _mm256_load_ps(&st[j][i0]), p1 = _mm256_load_ps(&st[j][i0 + 8]);
                    UNROLL for (int c = 0; c < 4; c++) {
                        __m256 vb = _mm256_broadcast_ss(vr + c);
                        acc[c][0] = _mm256_fmadd_ps(vb, p0, acc[c][0]);
                        acc[c][1] = _mm256_fmadd_ps(vb, p1, acc[c][1]);
                    }
                }
                UNROLL for (int c = 0; c < 4; c++) {
                    _mm256_store_ps(&ot[d0 + c][i0], acc[c][0]);
                    _mm256_store_ps(&ot[d0 + c][i0 + 8], acc[c][1]);
                }
            }
    }
    for (int i = 0; i < nq; i++) {
        float inv = 1.0f / l[i], *o = out + (size_t)(q0 + i) * stride + col;
        for (int d = 0; d < 64; d++)
            o[d] = ot[d][i] * inv;
    }
}

AVX2 static void quick_gelu_mul_avx2(float *g, const float *u, int n) {
    const __m256 c = _mm256_set1_ps(-1.702f), one = _mm256_set1_ps(1.0f);
    int z = 0;
    for (; z + 8 <= n; z += 8) {
        __m256 x = _mm256_loadu_ps(g + z);
        __m256 y = _mm256_div_ps(x, _mm256_add_ps(one, exp8(_mm256_mul_ps(c, x))));
        _mm256_storeu_ps(g + z, u ? _mm256_mul_ps(y, _mm256_loadu_ps(u + z)) : y);
    }
    for (; z < n; z++)
        g[z] = g[z] / (1.0f + expf(-1.702f * g[z])) * (u ? u[z] : 1.0f);
}

AVX2 static void silu_avx2(float *y, const float *x, int n) {
    const __m256 one = _mm256_set1_ps(1.0f), neg = _mm256_set1_ps(-1.0f);
    int z = 0;
    for (; z + 8 <= n; z += 8) {
        __m256 a = _mm256_loadu_ps(x + z);
        _mm256_storeu_ps(y + z, _mm256_div_ps(a, _mm256_add_ps(one, exp8(_mm256_mul_ps(neg, a)))));
    }
    for (; z < n; z++)
        y[z] = x[z] / (1.0f + expf(-x[z]));
}

AVX2 static void exp_avx2(float *x, int n) {
    int z = 0;
    for (; z + 8 <= n; z += 8)
        _mm256_storeu_ps(x + z, exp8(_mm256_loadu_ps(x + z)));
    for (; z < n; z++)
        x[z] = expf(x[z]);
}
#endif

void g2m_attn64(const float *q, const float *k, const float *v, float *out, size_t stride, size_t col,
                int q0, int nq, int k0, int k1) {
#if defined(__x86_64__) || defined(_M_X64)
    if (g2x_level() >= G2X_AVX2) {
        attn64_avx2(q, k, v, out, stride, col, q0, nq, k0, k1);
        return;
    }
#endif
    attn64_scalar(q, k, v, out, stride, col, q0, nq, k0, k1);
}

void g2m_quick_gelu_mul(float *g, const float *u, int n) {
#if defined(__x86_64__) || defined(_M_X64)
    if (g2x_level() >= G2X_AVX2) {
        quick_gelu_mul_avx2(g, u, n);
        return;
    }
#endif
    for (int z = 0; z < n; z++)
        g[z] = g[z] / (1.0f + expf(-1.702f * g[z])) * (u ? u[z] : 1.0f);
}

void g2m_silu(float *y, const float *x, int n) {
#if defined(__x86_64__) || defined(_M_X64)
    if (g2x_level() >= G2X_AVX2) {
        silu_avx2(y, x, n);
        return;
    }
#endif
    for (int z = 0; z < n; z++)
        y[z] = x[z] / (1.0f + expf(-x[z]));
}

void g2m_exp_inplace(float *x, int n) {
#if defined(__x86_64__) || defined(_M_X64)
    if (g2x_level() >= G2X_AVX2) {
        exp_avx2(x, n);
        return;
    }
#endif
    for (int z = 0; z < n; z++)
        x[z] = expf(x[z]);
}
