/* SIMD kernels for the EmbeddingGemma 2 CPU text graph and media towers.
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
#include <stdlib.h>
#include <string.h>

/* ---- portable fallbacks -------------------------------------------------- */

int g2m_attn_qblock(int d) {
    /* EI_G2M_QB=16..128 (multiple of 16) overrides the query block for widths <= 64, and up
     * to 64 for widths <= 128 (tuning experiments). */
    static int qb = -1;
    if (qb < 0) {
        const char *e = getenv("EI_G2M_QB");
        int v = e ? atoi(e) : 0;
        qb = v >= 16 && v <= 128 && v % 16 == 0 ? v : 0;
    }
    if (qb && (d <= 64 || (d <= 128 && qb <= 64)))
        return qb; /* Q^T/O^T hold d x QB <= 8192 floats */
    return d <= 128 ? 64 : d <= 256 ? 32 : 16;
}

void g2m_attn_reference(const g2m_attn_args *a) {
    float scores[G2M_ATTN_KB], o[512];
    for (int i = 0; i < a->nq; i++) {
        const float *qi = a->q + (size_t)(a->q0 + i) * a->qstride + a->qcol;
        int lo = a->lo ? a->lo[i] : a->klo, hi = a->hi ? a->hi[i] : a->khi;
        float m = -INFINITY, l = 0;
        memset(o, 0, (size_t)a->d * sizeof(float));
        for (int b = lo; b < hi; b += G2M_ATTN_KB) {
            int kn = hi - b < G2M_ATTN_KB ? hi - b : G2M_ATTN_KB;
            float mb = -INFINITY;
            for (int j = 0; j < kn; j++) {
                const float *kj = a->k + (size_t)(b + j) * a->kvstride + a->kvcol;
                float s = 0;
                for (int dd = 0; dd < a->d; dd++)
                    s += qi[dd] * a->scale * kj[dd];
                scores[j] = s;
                mb = s > mb ? s : mb;
            }
            float mn = mb > m ? mb : m, alpha = expf(m - mn);
            l *= alpha;
            for (int dd = 0; dd < a->d; dd++)
                o[dd] *= alpha;
            for (int j = 0; j < kn; j++) {
                float p = expf(scores[j] - mn);
                l += p;
                const float *vj = a->v + (size_t)(b + j) * a->kvstride + a->kvcol;
                for (int dd = 0; dd < a->d; dd++)
                    o[dd] += p * vj[dd];
            }
            m = mn;
        }
        float *oi = a->out + (size_t)(a->q0 + i) * a->ostride + a->ocol;
        for (int dd = 0; dd < a->d; dd++)
            oi[dd] = l > 0 ? o[dd] / l : 0.0f;
    }
}

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define AVX2 __attribute__((target("avx2,fma,f16c")))
/* Fixed-size register tiles must unroll at the upstream -O2 (GCC keeps such
 * loops rolled there and spills the accumulator arrays). */
#define UNROLL _Pragma("GCC unroll 8")

/* exp via x = n ln2 + r, |r| <= ln2/2, degree-7 Taylor in Horner form.
 * Inputs below -87.3 (including -inf, e.g. masked attention scores) and NaN
 * return exactly 0; inputs above 88.7 saturate. Every AVX body here carries
 * the g2x_ prefix: scripts/check-avx2-isolation.py allows AVX instructions
 * only in the runtime-dispatched kernels. */
AVX2 static inline __m256 g2x_exp8(__m256 x) {
    const __m256 live = _mm256_cmp_ps(x, _mm256_set1_ps(-87.3f), _CMP_GE_OQ);
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
    return _mm256_and_ps(_mm256_mul_ps(p, _mm256_castsi256_ps(e)), live);
}

/* Scratch per thread: Q^T and O^T hold d x QB floats (<= 8192 each by the
 * choice of QB), S^T/P^T KB x QB. 80 KB, so not on small worker stacks. */
static _Thread_local _Alignas(32) float tl_qt[8192], tl_ot[8192], tl_st[G2M_ATTN_KB * G2M_ATTN_QB_MAX];

AVX2 static void g2x_media_attn(const g2m_attn_args *a) {
    enum { KB = G2M_ATTN_KB };
    const int d = a->d, QB = g2m_attn_qblock(d), nq = a->nq;
    float *qt = tl_qt, *ot = tl_ot, *st = tl_st;
    _Alignas(32) float m[G2M_ATTN_QB_MAX], l[G2M_ATTN_QB_MAX], alpha[G2M_ATTN_QB_MAX],
        lof[G2M_ATTN_QB_MAX], hif[G2M_ATTN_QB_MAX];
    const int np = (nq + 15) & ~15; /* query lanes, padded to whole 16-wide tiles */
    for (int dd = 0; dd < d; dd++)
        for (int i = 0; i < np; i++)
            qt[dd * QB + i] = i < nq ? a->q[(size_t)(a->q0 + i) * a->qstride + a->qcol + dd] * a->scale : 0.0f;
    memset(ot, 0, (size_t)d * QB * sizeof(float));
    for (int i = 0; i < np; i++) {
        m[i] = -INFINITY, l[i] = 0;
        /* Padded lanes see every key (their q is zero, the results are dropped). */
        lof[i] = a->lo && i < nq ? (float)a->lo[i] : (float)a->klo;
        hif[i] = a->hi && i < nq ? (float)a->hi[i] : (float)a->khi;
    }
    const bool masked = a->lo != NULL;
    const float *kbase = a->k + a->kvcol, *vbase = a->v + a->kvcol;

    for (int b = a->klo; b < a->khi; b += KB) {
        const int kn = a->khi - b < KB ? a->khi - b : KB;
        /* S^T = K_blk . Q^T: 4 keys x 16 queries per register tile. */
        for (int j0 = 0; j0 < kn; j0 += 4) {
            const float *kr[4];
            UNROLL for (int c = 0; c < 4; c++)
                kr[c] = kbase + (size_t)(b + (j0 + c < kn ? j0 + c : kn - 1)) * a->kvstride;
            for (int i0 = 0; i0 < np; i0 += 16) {
                __m256 acc[4][2];
                UNROLL for (int c = 0; c < 4; c++)
                    acc[c][0] = acc[c][1] = _mm256_setzero_ps();
                const float *qp = qt + i0;
                _Pragma("GCC unroll 4") for (int dd = 0; dd < d; dd++, qp += QB) {
                    __m256 a0 = _mm256_load_ps(qp), a1 = _mm256_load_ps(qp + 8);
                    UNROLL for (int c = 0; c < 4; c++) {
                        __m256 kb = _mm256_broadcast_ss(kr[c] + dd);
                        acc[c][0] = _mm256_fmadd_ps(kb, a0, acc[c][0]);
                        acc[c][1] = _mm256_fmadd_ps(kb, a1, acc[c][1]);
                    }
                }
                for (int c = 0; c < 4 && j0 + c < kn; c++) {
                    _mm256_store_ps(st + (j0 + c) * QB + i0, acc[c][0]);
                    _mm256_store_ps(st + (j0 + c) * QB + i0 + 8, acc[c][1]);
                }
            }
        }
        /* Mask, then online softmax, 8 query lanes at a time. */
        for (int i0 = 0; i0 < np; i0 += 8) {
            __m256 mb = _mm256_set1_ps(-INFINITY);
            if (masked) {
                const __m256 lov = _mm256_load_ps(lof + i0), hiv = _mm256_load_ps(hif + i0);
                for (int j = 0; j < kn; j++) {
                    __m256 key = _mm256_set1_ps((float)(b + j)), sv = _mm256_load_ps(st + j * QB + i0);
                    __m256 ok = _mm256_and_ps(_mm256_cmp_ps(key, lov, _CMP_GE_OQ), _mm256_cmp_ps(key, hiv, _CMP_LT_OQ));
                    sv = _mm256_blendv_ps(_mm256_set1_ps(-INFINITY), sv, ok);
                    _mm256_store_ps(st + j * QB + i0, sv);
                    mb = _mm256_max_ps(mb, sv);
                }
            } else
                for (int j = 0; j < kn; j++)
                    mb = _mm256_max_ps(mb, _mm256_load_ps(st + j * QB + i0));
            __m256 mo = _mm256_load_ps(m + i0), mn = _mm256_max_ps(mo, mb);
            /* A lane with no visible key yet keeps m = -inf; subtract a finite
             * floor so exp sees -inf - floor = -inf -> 0, never inf - inf. */
            __m256 ms = _mm256_max_ps(mn, _mm256_set1_ps(-3.0e38f));
            __m256 al = g2x_exp8(_mm256_sub_ps(mo, ms)), sum = _mm256_setzero_ps();
            for (int j = 0; j < kn; j++) {
                __m256 pv = g2x_exp8(_mm256_sub_ps(_mm256_load_ps(st + j * QB + i0), ms));
                _mm256_store_ps(st + j * QB + i0, pv);
                sum = _mm256_add_ps(sum, pv);
            }
            _mm256_store_ps(l + i0, _mm256_fmadd_ps(_mm256_load_ps(l + i0), al, sum));
            _mm256_store_ps(m + i0, mn);
            _mm256_store_ps(alpha + i0, al);
        }
        /* O^T = O^T * alpha + V_blk^T . P^T: 4 dims x 16 queries per tile. */
        for (int d0 = 0; d0 < d; d0 += 4)
            for (int i0 = 0; i0 < np; i0 += 16) {
                __m256 al0 = _mm256_load_ps(alpha + i0), al1 = _mm256_load_ps(alpha + i0 + 8);
                __m256 acc[4][2];
                UNROLL for (int c = 0; c < 4; c++) {
                    acc[c][0] = _mm256_mul_ps(_mm256_load_ps(ot + (d0 + c) * QB + i0), al0);
                    acc[c][1] = _mm256_mul_ps(_mm256_load_ps(ot + (d0 + c) * QB + i0 + 8), al1);
                }
                const float *vr = vbase + (size_t)b * a->kvstride + d0;
                const float *pp = st + i0;
                for (int j = 0; j < kn; j++, vr += a->kvstride, pp += QB) {
                    __m256 p0 = _mm256_load_ps(pp), p1 = _mm256_load_ps(pp + 8);
                    UNROLL for (int c = 0; c < 4; c++) {
                        __m256 vb = _mm256_broadcast_ss(vr + c);
                        acc[c][0] = _mm256_fmadd_ps(vb, p0, acc[c][0]);
                        acc[c][1] = _mm256_fmadd_ps(vb, p1, acc[c][1]);
                    }
                }
                UNROLL for (int c = 0; c < 4; c++) {
                    _mm256_store_ps(ot + (d0 + c) * QB + i0, acc[c][0]);
                    _mm256_store_ps(ot + (d0 + c) * QB + i0 + 8, acc[c][1]);
                }
            }
    }
    for (int i = 0; i < nq; i++) {
        float inv = l[i] > 0 ? 1.0f / l[i] : 0.0f, *o = a->out + (size_t)(a->q0 + i) * a->ostride + a->ocol;
        for (int dd = 0; dd < d; dd++)
            o[dd] = ot[dd * QB + i] * inv;
    }
}

AVX2 static void g2x_media_quick_gelu_mul(float *g, const float *u, int n) {
    const __m256 c = _mm256_set1_ps(-1.702f), one = _mm256_set1_ps(1.0f);
    int z = 0;
    for (; z + 8 <= n; z += 8) {
        __m256 x = _mm256_loadu_ps(g + z);
        __m256 y = _mm256_div_ps(x, _mm256_add_ps(one, g2x_exp8(_mm256_mul_ps(c, x))));
        _mm256_storeu_ps(g + z, u ? _mm256_mul_ps(y, _mm256_loadu_ps(u + z)) : y);
    }
    for (; z < n; z++)
        g[z] = g[z] / (1.0f + expf(-1.702f * g[z])) * (u ? u[z] : 1.0f);
}

AVX2 static void g2x_media_silu(float *y, const float *x, int n) {
    const __m256 one = _mm256_set1_ps(1.0f), neg = _mm256_set1_ps(-1.0f);
    int z = 0;
    for (; z + 8 <= n; z += 8) {
        __m256 a = _mm256_loadu_ps(x + z);
        _mm256_storeu_ps(y + z, _mm256_div_ps(a, _mm256_add_ps(one, g2x_exp8(_mm256_mul_ps(neg, a)))));
    }
    for (; z < n; z++)
        y[z] = x[z] / (1.0f + expf(-x[z]));
}

AVX2 static void g2x_media_exp(float *x, int n) {
    int z = 0;
    for (; z + 8 <= n; z += 8)
        _mm256_storeu_ps(x + z, g2x_exp8(_mm256_loadu_ps(x + z)));
    for (; z < n; z++)
        x[z] = expf(x[z]);
}
#endif

void g2m_attn(const g2m_attn_args *a) {
#if defined(__x86_64__) || defined(_M_X64)
    if (g2x_level() >= G2X_AVX2 && a->d % 16 == 0 && a->d <= 512) {
        g2x_media_attn(a);
        return;
    }
#endif
    g2m_attn_reference(a);
}

void g2m_attn64(const float *q, const float *k, const float *v, float *out, size_t stride, size_t col,
                int q0, int nq, int k0, int k1) {
    g2m_attn_args a = {.q = q, .k = k, .v = v, .out = out, .qstride = stride, .kvstride = stride,
                       .ostride = stride, .qcol = col, .kvcol = col, .ocol = col, .d = 64, .q0 = q0, .nq = nq,
                       .klo = k0, .khi = k1, .scale = 1.0f};
    g2m_attn(&a);
}

void g2m_quick_gelu_mul(float *g, const float *u, int n) {
#if defined(__x86_64__) || defined(_M_X64)
    if (g2x_level() >= G2X_AVX2) {
        g2x_media_quick_gelu_mul(g, u, n);
        return;
    }
#endif
    for (int z = 0; z < n; z++)
        g[z] = g[z] / (1.0f + expf(-1.702f * g[z])) * (u ? u[z] : 1.0f);
}

void g2m_silu(float *y, const float *x, int n) {
#if defined(__x86_64__) || defined(_M_X64)
    if (g2x_level() >= G2X_AVX2) {
        g2x_media_silu(y, x, n);
        return;
    }
#endif
    for (int z = 0; z < n; z++)
        y[z] = x[z] / (1.0f + expf(-x[z]));
}

void g2m_exp_inplace(float *x, int n) {
#if defined(__x86_64__) || defined(_M_X64)
    if (g2x_level() >= G2X_AVX2) {
        g2x_media_exp(x, n);
        return;
    }
#endif
    for (int z = 0; z < n; z++)
        x[z] = expf(x[z]);
}
