#include "kernels.h"
#include "quants.h"

#include <math.h>
#include <stdio.h>

static uint32_t rng_state = 0x12345678u;

static float next_f32(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return ((float)(x & 0xffffu) / 32768.0f) - 1.0f;
}

static void fill(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = next_f32();
}

static void require_close(const char *name, const float *got, const float *want,
                          size_t n, float atol) {
    float max_error = 0.0f;
    size_t max_index = 0;
    for (size_t i = 0; i < n; i++) {
        float error = fabsf(got[i] - want[i]);
        if (error > max_error) {
            max_error = error;
            max_index = i;
        }
    }
    if (max_error > atol) {
        ei_die("%s mismatch at %zu: got %.9g, want %.9g, error %.9g",
               name, max_index, got[max_index], want[max_index], max_error);
    }
    printf("%s max error %.9g\n", name, max_error);
}

static void reference_attention(const float *q, const float *k, const float *v,
                                int32_t tokens, int32_t window, float *out) {
    float scores[16];
    memset(out, 0, sizeof(float) * (size_t)tokens * EI_N_EMBD);
    int32_t half = window > 0 ? window / 2 : 0;
    for (int32_t qt = 0; qt < tokens; qt++) {
        int32_t first = window > 0 && qt > half ? qt - half : 0;
        int32_t last = window > 0 && qt + half + 1 < tokens ? qt + half + 1 : tokens;
        for (int32_t h = 0; h < EI_N_HEAD; h++) {
            const float *qv = q + (size_t)qt * EI_N_EMBD + h * EI_HEAD_DIM;
            float max_score = -INFINITY;
            for (int32_t kt = first; kt < last; kt++) {
                float score = 0.0f;
                for (int32_t d = 0; d < EI_HEAD_DIM; d++) {
                    score += qv[d] * k[(size_t)kt * EI_HEAD_DIM + d];
                }
                scores[kt] = score;
                if (score > max_score) max_score = score;
            }
            float denominator = 0.0f;
            for (int32_t kt = first; kt < last; kt++) {
                scores[kt] = expf(scores[kt] - max_score);
                denominator += scores[kt];
            }
            float *dst = out + (size_t)qt * EI_N_EMBD + h * EI_HEAD_DIM;
            for (int32_t kt = first; kt < last; kt++) {
                float probability = scores[kt] / denominator;
                for (int32_t d = 0; d < EI_HEAD_DIM; d++) {
                    dst[d] += probability * v[(size_t)kt * EI_HEAD_DIM + d];
                }
            }
        }
    }
}

static void test_attention(void) {
    enum { TOKENS = 9 };
    float *q = ei_xmalloc(sizeof(float) * TOKENS * EI_N_EMBD);
    float *k = ei_xmalloc(sizeof(float) * TOKENS * EI_HEAD_DIM);
    float *v = ei_xmalloc(sizeof(float) * TOKENS * EI_HEAD_DIM);
    float *got = ei_xmalloc(sizeof(float) * TOKENS * EI_N_EMBD);
    float *want = ei_xmalloc(sizeof(float) * TOKENS * EI_N_EMBD);
    float scores[TOKENS];
    fill(q, TOKENS * EI_N_EMBD);
    fill(k, TOKENS * EI_HEAD_DIM);
    fill(v, TOKENS * EI_HEAD_DIM);

    reference_attention(q, k, v, TOKENS, 0, want);
    ei_attention_mha(q, k, v, TOKENS, 0, got, scores);
    require_close("attention full", got, want, TOKENS * EI_N_EMBD, 2e-5f);

    reference_attention(q, k, v, TOKENS, 6, want);
    ei_attention_mha(q, k, v, TOKENS, 6, got, scores);
    require_close("attention swa", got, want, TOKENS * EI_N_EMBD, 2e-5f);

    free(want);
    free(got);
    free(v);
    free(k);
    free(q);
}

static void test_norm_residual(void) {
    float x[EI_N_EMBD];
    float w[EI_N_EMBD];
    float got[EI_N_EMBD];
    float want[EI_N_EMBD];
    float normalized[EI_N_EMBD];
    fill(x, EI_N_EMBD);
    fill(w, EI_N_EMBD);
    fill(got, EI_N_EMBD);
    memcpy(want, got, sizeof got);
    ei_rms_norm(x, w, EI_N_EMBD, 1e-6f, normalized);
    ei_vec_add_inplace(want, normalized, EI_N_EMBD);
    ei_rms_norm_residual_inplace(got, x, w, EI_N_EMBD, 1e-6f);
    require_close("rms residual", got, want, EI_N_EMBD, 2e-6f);
}

static void test_qk_norm_rope(void) {
    float x[EI_N_EMBD];
    float got[EI_N_EMBD];
    float want[EI_N_EMBD];
    float w[EI_HEAD_DIM];
    fill(x, EI_N_EMBD);
    fill(w, EI_HEAD_DIM);
    memcpy(got, x, sizeof x);
    memcpy(want, x, sizeof x);
    for (int32_t h = 0; h < EI_N_HEAD; h++) {
        ei_rms_norm_inplace(want + h * EI_HEAD_DIM, w, EI_HEAD_DIM, 1e-6f);
    }
    ei_rope_neox_inplace(want, EI_N_HEAD, 127, 10000.0f);
    for (int32_t i = 0; i < EI_N_EMBD; i++) want[i] *= 0.0625f;
    ei_qk_norm_rope_inplace(got, w, EI_N_HEAD, 127, 10000.0f, 0.0625f, 1e-6f);
    require_close("qk norm rope", got, want, EI_N_EMBD, 1e-6f);

    float q[EI_N_EMBD];
    float k[EI_HEAD_DIM];
    float q_want[EI_N_EMBD];
    float k_want[EI_HEAD_DIM];
    fill(q, EI_N_EMBD);
    fill(k, EI_HEAD_DIM);
    memcpy(q_want, q, sizeof q);
    memcpy(k_want, k, sizeof k);
    ei_qk_norm_rope_inplace(q_want, w, EI_N_HEAD, 127, 10000.0f, 0.0625f, 1e-6f);
    ei_qk_norm_rope_inplace(k_want, w, EI_N_HEAD_KV, 127, 10000.0f, 1.0f, 1e-6f);
    ei_qk_norm_rope_qk_inplace(q, k, w, w, 127, 10000.0f, 1e-6f);
    require_close("qk fused q", q, q_want, EI_N_EMBD, 1e-6f);
    require_close("qk fused k", k, k_want, EI_HEAD_DIM, 1e-6f);
}

static void test_q4_triple(void) {
    enum { BLOCKS = EI_N_EMBD / EI_QK };
    ei_block_q4_0 q0[BLOCKS];
    ei_block_q4_0 q1[BLOCKS];
    ei_block_q4_0 q2[BLOCKS];
    ei_block_q8_0 xq[BLOCKS];
    float x[EI_N_EMBD];
    fill(x, EI_N_EMBD);
    ei_quantize_row_q8_0(x, xq, EI_N_EMBD);
    ei_block_q4_0 *weights[3] = { q0, q1, q2 };
    for (int matrix = 0; matrix < 3; matrix++) {
        for (int block = 0; block < BLOCKS; block++) {
            weights[matrix][block].d = ei_fp32_to_fp16(
                0.01f + 0.001f * (float)(matrix + block));
            for (int i = 0; i < 16; i++) {
                uint8_t lo = (uint8_t)((matrix * 3 + block + i) & 15);
                uint8_t hi = (uint8_t)((matrix * 5 + block + 2 * i) & 15);
                weights[matrix][block].qs[i] = (uint8_t)(lo | (hi << 4));
            }
        }
    }
    float want[3] = {
        ei_vec_dot_q4_0_q8_0(q0, xq, EI_N_EMBD),
        ei_vec_dot_q4_0_q8_0(q1, xq, EI_N_EMBD),
        ei_vec_dot_q4_0_q8_0(q2, xq, EI_N_EMBD),
    };
    float got[3];
    ei_vec_dot_q4_0_q8_0_triple(q0, q1, q2, xq, EI_N_EMBD,
                                &got[0], &got[1], &got[2]);
    require_close("q4 triple", got, want, 3, 1e-6f);
}

static void test_mean_pool(void) {
    enum { TOKENS = 7 };
    float x[TOKENS * EI_N_EMBD];
    float got_x[TOKENS * EI_N_EMBD];
    float want_x[TOKENS * EI_N_EMBD];
    float w[EI_N_EMBD];
    float got[EI_N_EMBD];
    float want[EI_N_EMBD];
    fill(x, TOKENS * EI_N_EMBD);
    fill(w, EI_N_EMBD);
    memcpy(got_x, x, sizeof x);
    memcpy(want_x, x, sizeof x);
    memset(want, 0, sizeof want);
    for (int32_t t = 0; t < TOKENS; t++) {
        float *row = want_x + t * EI_N_EMBD;
        ei_rms_norm_inplace(row, w, EI_N_EMBD, 1e-6f);
        ei_vec_add_inplace(want, row, EI_N_EMBD);
    }
    for (int32_t i = 0; i < EI_N_EMBD; i++) want[i] /= TOKENS;
    ei_l2_normalize(want, EI_N_EMBD);
    ei_mean_pool_rms_l2(got_x, TOKENS, w, 1e-6f, got);
    require_close("mean pool rms l2", got, want, EI_N_EMBD, 2e-6f);
}

static void test_matryoshka_dimensions(void) {
    static const int32_t dimensions[] = { 128, 256, 512, EI_N_EMBD };
    static const int32_t invalid[] = { -1, 0, 127, 129, 767, 769 };
    float canonical[EI_N_EMBD];
    float got[EI_N_EMBD];
    float want[EI_N_EMBD];
    fill(canonical, EI_N_EMBD);
    ei_l2_normalize(canonical, EI_N_EMBD);

    for (size_t case_index = 0;
         case_index < sizeof dimensions / sizeof dimensions[0]; case_index++) {
        const int32_t n = dimensions[case_index];
        memcpy(got, canonical, sizeof got);
        memcpy(want, canonical, sizeof want);

        double sum_squares = 0.0;
        for (int32_t i = 0; i < n; i++) {
            sum_squares += (double)canonical[i] * (double)canonical[i];
        }
        const float scale = (float)(1.0 / sqrt(sum_squares));
        for (int32_t i = 0; i < n; i++) want[i] *= scale;

        if (!ei_embedding_dimensions_supported(n) ||
            !ei_embedding_normalize_prefix(got, n)) {
            ei_die("Matryoshka dimension %d was rejected", n);
        }
        char name[64];
        snprintf(name, sizeof name, "Matryoshka D=%d", n);
        require_close(name, got, want, (size_t)n, 2e-6f);
        if (n < EI_N_EMBD &&
            memcmp(got + n, canonical + n,
                   sizeof(float) * (size_t)(EI_N_EMBD - n)) != 0) {
            ei_die("Matryoshka D=%d modified the unused suffix", n);
        }
        if (n == EI_N_EMBD && memcmp(got, canonical, sizeof got) != 0) {
            ei_die("Matryoshka D=%d must preserve the canonical embedding", n);
        }
    }

    for (size_t i = 0; i < sizeof invalid / sizeof invalid[0]; i++) {
        memcpy(got, canonical, sizeof got);
        if (ei_embedding_dimensions_supported(invalid[i]) ||
            ei_embedding_normalize_prefix(got, invalid[i])) {
            ei_die("invalid Matryoshka dimension %d was accepted", invalid[i]);
        }
        if (memcmp(got, canonical, sizeof got) != 0) {
            ei_die("invalid Matryoshka dimension %d modified the embedding", invalid[i]);
        }
    }
}

int main(void) {
    test_q4_triple();
    test_attention();
    test_norm_residual();
    test_qk_norm_rope();
    test_mean_pool();
    test_matryoshka_dimensions();
    return 0;
}
