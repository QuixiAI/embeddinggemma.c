/* cpu-max fork: g2m_attn (AVX2 tiled online softmax) against its scalar
 * reference on random data: head widths 64/128/256/512, GQA strides,
 * sliding windows and packed-sequence bounds, partial query/key blocks.
 * Both are compared with a float64 ground truth (max |diff| in a row / row RMS);
 * exits 1 if the tiled kernel is more than 2x less accurate than plain scalar
 * fp32 or above 1e-3. */
#include "gemma2_media_x86.h"
#include "common.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = 0x243F6A8885A308D3ull;
static float frand(void) {
    rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
    return (float)((rng >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f;
}

/* max over (token, head) rows of max|a - ref| / rms(ref) */
static double row_err(const float *a, const double *ref, int n, int heads, int d) {
    double worst = 0;
    for (size_t r = 0; r < (size_t)n * heads; r++) {
        double ss = 0, md = 0;
        for (int z = 0; z < d; z++) {
            double x = a[r * d + z], y = ref[r * d + z];
            ss += y * y;
            md = x != x ? INFINITY : fabs(x - y) > md ? fabs(x - y) : md;
        }
        double e = md / sqrt(ss / d + 1e-30);
        if (!(e <= worst))
            worst = e;
    }
    return worst;
}
static void run(int d, int n, int heads, int kvh, int window, int split, double *err_avx2, double *err_scalar) {
    size_t qs = (size_t)heads * d, ks = (size_t)kvh * d;
    float *q = ei_xmalloc(n * qs * 4), *k = ei_xmalloc(n * ks * 4), *v = ei_xmalloc(n * ks * 4);
    float *o1 = ei_xcalloc(n * qs, 4), *o2 = ei_xcalloc(n * qs, 4);
    double *exact = ei_xcalloc(n * qs, sizeof(double)), *sc = ei_xcalloc(n, sizeof(double));
    /* Unit-scale rows as after the model's q/k/v RMS norms; scores reach ~d. */
    for (size_t i = 0; i < n * qs; i++)
        q[i] = frand() * 1.7f;
    for (size_t i = 0; i < n * ks; i++)
        k[i] = frand() * 1.7f, v[i] = frand() * 1.7f;
    int qb = g2m_attn_qblock(d);
    int32_t lo[64], hi[64];
    for (int hd = 0; hd < heads; hd++)
        for (int q0 = 0; q0 < n; q0 += qb) {
            int nq = n - q0 < qb ? n - q0 : qb, klo = n, khi = 0;
            for (int i = 0; i < nq; i++) {
                int t = q0 + i, s0 = t < split ? 0 : split, s1 = t < split ? split : n;
                lo[i] = window && t - window > s0 ? t - window : s0;
                hi[i] = window && t + window + 1 < s1 ? t + window + 1 : s1;
                klo = lo[i] < klo ? lo[i] : klo;
                khi = hi[i] > khi ? hi[i] : khi;
                /* double-precision ground truth */
                const float *qi = q + (size_t)t * qs + (size_t)hd * d;
                int kh = hd / (heads / kvh);
                double mx = -INFINITY, total = 0;
                for (int j = lo[i]; j < hi[i]; j++) {
                    double dot = 0;
                    for (int z = 0; z < d; z++)
                        dot += (double)qi[z] * k[(size_t)j * ks + (size_t)kh * d + z];
                    sc[j] = dot;
                    mx = dot > mx ? dot : mx;
                }
                double *ex = exact + (size_t)t * qs + (size_t)hd * d;
                for (int j = lo[i]; j < hi[i]; j++) {
                    double p = exp(sc[j] - mx);
                    total += p;
                    for (int z = 0; z < d; z++)
                        ex[z] += p * v[(size_t)j * ks + (size_t)kh * d + z];
                }
                for (int z = 0; z < d; z++)
                    ex[z] /= total;
            }
            g2m_attn_args a = {.q = q, .k = k, .v = v, .qstride = qs, .kvstride = ks, .ostride = qs,
                               .qcol = (size_t)hd * d, .kvcol = (size_t)(hd / (heads / kvh)) * d,
                               .ocol = (size_t)hd * d, .d = d, .q0 = q0, .nq = nq, .lo = lo, .hi = hi,
                               .klo = klo, .khi = khi, .scale = 1.0f};
            a.out = o1;
            g2m_attn(&a);
            a.out = o2;
            g2m_attn_reference(&a);
        }
    *err_avx2 = row_err(o1, exact, n, heads, d);
    *err_scalar = row_err(o2, exact, n, heads, d);
    free(q), free(k), free(v), free(o1), free(o2), free(exact), free(sc);
}

int main(void) {
    struct { int d, n, heads, kvh, window, split; } cases[] = {
        {64, 630, 12, 12, 0, 630},   /* vision i70 */
        {64, 77, 12, 12, 0, 77},     /* partial blocks */
        {128, 300, 8, 8, 11, 300},   /* audio-like width */
        {256, 1100, 4, 2, 512, 1100}, /* text sliding layer, GQA 4:2 */
        {256, 700, 4, 2, 512, 333},  /* two packed sequences */
        {512, 900, 4, 1, 0, 900},    /* text full layer, MQA */
        {512, 300, 4, 1, 0, 37},     /* packed, short first sequence */
    };
    bool ok = true;
    for (size_t c = 0; c < sizeof cases / sizeof *cases; c++) {
        double ea, es;
        run(cases[c].d, cases[c].n, cases[c].heads, cases[c].kvh, cases[c].window, cases[c].split, &ea, &es);
        printf("d=%3d n=%5d heads=%2d kv=%2d window=%3d split=%4d  err/rms vs fp64: tiled %.2e  scalar fp32 %.2e\n",
               cases[c].d, cases[c].n, cases[c].heads, cases[c].kvh, cases[c].window, cases[c].split, ea, es);
        ok &= ea <= 2 * es + 1e-6 && ea < 1e-3;
    }
    return ok ? 0 : 1;
}
