/* cpu-max fork: exactness and throughput of the Q8_0 GEMM routes.
 *
 *   test_gemma2_vnni256 [NxMxK ...]     (default: the vision/audio/text shapes)
 *   fNxMxK / FNxMxK: the float-weight GEMM with BF16 / F32 weights (EI_GEMMA2_CPU_SGEMM=0: upstream)
 *
 * Random Q8_0 weights and activations; the result of ei_gemma2_fast_gemm_q8
 * (whatever route the dispatch picks: set EI_GEMMA2_CPU_VNNI256=0 for the
 * upstream tile) is compared with a double-precision scalar Q8_0 dot product.
 * Prints max relative error and GMAC/s (median of 5) per shape; exits 1 if a
 * relative error exceeds 1e-5. EI_THREADS sets the pool width. */
#if defined(__linux__)
#define _POSIX_C_SOURCE 200809L
#endif
#include "gemma2.h"
#include "gemma2_vnni256.h"
#include "parallel.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}
static uint64_t rng = 0x9E3779B97F4A7C15ull;
static float frand(void) {
    rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
    return (float)((rng >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f;
}
static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv) {
    const char *defaults[] = {"2520x768x768", "2520x3072x768", "2520x768x3072", "750x4096x1024",
                              "750x1024x4096", "512x2048x512", "512x512x2048", "16x2048x512"};
    const char **cases = argc > 1 ? (const char **)argv + 1 : defaults;
    int n_cases = argc > 1 ? argc - 1 : (int)(sizeof defaults / sizeof *defaults);
    ei_thread_pool *pool = ei_thread_pool_create_spin(0, 50);
    ei_gemma2_fast *ctx = ei_gemma2_fast_create();
    printf("threads=%d vnni256=%d\n", ei_thread_pool_threads(pool), g2v_enabled());
    bool ok = true;
    for (int c = 0; c < n_cases; c++) {
        int n, m, k;
        if (cases[c][0] == 'f' || cases[c][0] == 'F') { /* float-weight GEMM: f = BF16, F = F32 weights */
            bool bf16 = cases[c][0] == 'f';
            if (sscanf(cases[c] + 1, "%dx%dx%d", &n, &m, &k) != 3)
                ei_die("bad case %s", cases[c]);
            float *wf = ei_xmalloc((size_t)m * k * sizeof(float)), *x = ei_xmalloc((size_t)n * k * sizeof(float));
            uint16_t *wb = ei_xmalloc((size_t)m * k * sizeof(uint16_t));
            for (size_t i = 0; i < (size_t)m * k; i++) {
                float f = frand() * 0.1f;
                uint32_t bits;
                memcpy(&bits, &f, 4);
                wb[i] = (uint16_t)(bits >> 16);
                bits = (uint32_t)wb[i] << 16;
                memcpy(&wf[i], &bits, 4); /* the exact value the BF16 tensor holds */
            }
            for (size_t i = 0; i < (size_t)n * k; i++)
                x[i] = frand() * 3.0f;
            ei_tensor *wt = ei_xmalloc(sizeof *wt);
            *wt = (ei_tensor){.n_dims = 2, .ne = {(uint64_t)k, (uint64_t)m, 1, 1},
                              .type = bf16 ? EI_T_BF16 : EI_T_F32, .data = bf16 ? (void *)wb : (void *)wf};
            float *yf = ei_xmalloc((size_t)n * m * sizeof(float));
            double times[5];
            for (int rep = 0; rep < 6; rep++) {
                double t0 = now_ms();
                ei_gemma2_fast_gemm_f(pool, wt, x, n, yf);
                if (rep)
                    times[rep - 1] = now_ms() - t0;
            }
            qsort(times, 5, sizeof *times, cmp_double);
            double max_rel = 0;
            for (int t = 0; t < n; t += n > 64 ? 29 : 1)
                for (int r = 0; r < m; r += m > 256 ? 11 : 1) {
                    double ref = 0, mag = 0;
                    for (int z = 0; z < k; z++)
                        ref += (double)wf[(size_t)r * k + z] * x[(size_t)t * k + z],
                            mag += fabs((double)wf[(size_t)r * k + z] * x[(size_t)t * k + z]);
                    double rel = fabs(yf[(size_t)t * m + r] - ref) / (mag > 0 ? mag : 1);
                    max_rel = rel > max_rel ? rel : max_rel;
                }
            printf("%-16s median %8.3f ms  %8.1f GMAC/s  max_rel_err %.2e (float weights, %s)\n", cases[c], times[2],
                   (double)n * m * k / (times[2] * 1e6), max_rel, bf16 ? "BF16" : "F32");
            if (!(max_rel < 1e-5))
                ok = false;
            free(x), free(yf);
            continue;
        }
        if (sscanf(cases[c], "%dx%dx%d", &n, &m, &k) != 3 || k % 32)
            ei_die("bad case %s (NxMxK, K multiple of 32)", cases[c]);
        ei_block_q8_0 *wq = ei_xmalloc((size_t)m * k / 32 * sizeof *wq);
        ei_block_q8_0 *xq = ei_xmalloc((size_t)n * k / 32 * sizeof *xq);
        float *row = ei_xmalloc((size_t)k * sizeof(float)), *y = ei_xmalloc((size_t)n * m * sizeof(float));
        for (int r = 0; r < m; r++) {
            for (int z = 0; z < k; z++)
                row[z] = frand() * (1.0f + (z % 97 == 0) * 6.0f); /* a few outliers per row */
            ei_quantize_row_q8_0(row, wq + (size_t)r * k / 32, k);
        }
        for (int t = 0; t < n; t++) {
            for (int z = 0; z < k; z++)
                row[z] = frand() * 3.0f;
            ei_quantize_row_q8_0(row, xq + (size_t)t * k / 32, k);
        }
        /* Repack caches are keyed by tensor address: one live tensor object per case. */
        ei_tensor *wt = ei_xmalloc(sizeof *wt);
        *wt = (ei_tensor){.n_dims = 2, .ne = {(uint64_t)k, (uint64_t)m, 1, 1}, .type = EI_T_Q8_0, .data = wq};
        ei_tensor w = *wt;
        double times[5];
        for (int rep = 0; rep < 6; rep++) {
            double t0 = now_ms();
            ei_gemma2_fast_gemm_q8(ctx, pool, wt, xq, n, y);
            if (rep)
                times[rep - 1] = now_ms() - t0; /* rep 0 includes the one-time repack */
        }
        qsort(times, 5, sizeof *times, cmp_double);
        double max_rel = 0;
        int checked = 0;
        for (int t = 0; t < n; t += n > 64 ? 37 : 1)
            for (int r = 0; r < m; r += m > 256 ? 13 : 1) {
                double ref = 0, mag = 0;
                for (int p = 0; p < k / 32; p++) {
                    const ei_block_q8_0 *a = wq + (size_t)r * k / 32 + p, *b = xq + (size_t)t * k / 32 + p;
                    int dot = 0, absdot = 0;
                    for (int z = 0; z < 32; z++)
                        dot += a->qs[z] * b->qs[z], absdot += abs(a->qs[z] * b->qs[z]);
                    double s = (double)ei_fp16_to_fp32(a->d) * ei_fp16_to_fp32(b->d);
                    ref += dot * s;
                    mag += absdot * s;
                }
                double rel = fabs(y[(size_t)t * m + r] - ref) / (mag > 0 ? mag : 1);
                if (rel > max_rel)
                    max_rel = rel;
                checked++;
            }
        double gmacs = (double)n * m * k / (times[2] * 1e6);
        printf("%-16s median %8.3f ms  %8.1f GMAC/s  max_rel_err %.2e (%d outputs checked)\n", cases[c], times[2],
               gmacs, max_rel, checked);
        if (!(max_rel < 1e-5))
            ok = false;
        (void)w;
        free(xq), free(row), free(y); /* wq/wt stay alive: the repack cache keys on them */
    }
    ei_gemma2_fast_free(ctx);
    ei_thread_pool_destroy(pool);
    return ok ? 0 : 1;
}
