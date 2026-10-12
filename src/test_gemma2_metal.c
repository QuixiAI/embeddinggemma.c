#define _POSIX_C_SOURCE 200809L
#include "engine.h"
#include "gemma2_metal.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void compare(const char *kind, size_t n, const float *a, const float *b, size_t count,
                    double gate) {
    double aa = 0, bb = 0, ab = 0;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i]))
            ei_die("non-finite %s", kind);
        aa += (double)a[i] * a[i];
        bb += (double)b[i] * b[i];
        ab += (double)a[i] * b[i];
    }
    double cosine = ab / sqrt(aa * bb);
    printf("%s %zu cosine %.9f\n", kind, n, cosine);
    fflush(stdout);
    if (cosine < gate)
        ei_die("%s parity gate %.9f < %.9f", kind, cosine, gate);
}
int main(int argc, char **argv) {
    if (argc != 3)
        ei_die("usage: %s text.gguf projector.gguf", argv[0]);
    // The fast route is EI_GEMMA2_ROUTE_FLAGS (comma-separated EI_GEMMA2_METAL_*
    // names; the rest stay "0"), default FUSED,TENSOR2,FLASH2, the shipped
    // default. The reference route has every flag off.
    const char *all[] = {"TENSOR", "FLASH",   "GEMV",   "FUSED",  "TENSOR2",
                         "FLASH2", "FLASH64", "VFUSED", "AFUSED", "LAYER"};
    char name[64];
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) {
        snprintf(name, sizeof name, "EI_GEMMA2_METAL_%s", all[i]);
        setenv(name, "0", 1); // the batched stack is on by default; reference is all off
    }
    ei_engine e;
    ei_engine_load_backend(&e, argv[1], "metal");
    const char *route = getenv("EI_GEMMA2_ROUTE_FLAGS");
    char *flags = strdup(route && *route ? route : "FUSED,TENSOR2,FLASH2,FLASH64,VFUSED,AFUSED"),
         *save = NULL;
    for (char *f = strtok_r(flags, ",", &save); f; f = strtok_r(NULL, ",", &save)) {
        snprintf(name, sizeof name, "EI_GEMMA2_METAL_%s", f);
        setenv(name, "1", 1);
    }
    printf("fast route: %s\n", route && *route ? route : "FUSED,TENSOR2,FLASH2,FLASH64,VFUSED,AFUSED");
    free(flags);
    char err[512];
    void *fast = ei_gemma2_metal_create(&e.model.gguf, err, sizeof err);
    if (!fast)
        ei_die("%s", err);
    int32_t *ids = ei_xmalloc(8192 * sizeof *ids);
    for (int i = 0; i < 8192; i++)
        ids[i] = 100 + (i * 137) % 10000;
    const size_t lengths[] = {1, 7, 31, 32, 33, 127, 128, 129, 512, 1025, 2048, 8192};
    for (size_t i = 0; i < sizeof lengths / sizeof *lengths; i++) {
        float a[768], b[768];
        size_t n = lengths[i];
        if (!ei_gemma2_metal_embed(e.gemma2_metal, ids, NULL, n, a, err, sizeof err) ||
            !ei_gemma2_metal_embed(fast, ids, NULL, n, b, err, sizeof err))
            ei_die("%s", err);
        compare("text routes", n, a, b, 768, .999);
    }
    // Each side of both thresholds, odd boundaries, and different absolute
    // positions catch RoPE reset and cross-sequence attention leakage.
    const size_t offsets[] = {0, 1, 8, 39, 71, 104, 231, 359, 488, 1513};
    float packed[9 * 768];
    for (int route = 0; route < 2; route++) {
        void *ctx = route ? fast : e.gemma2_metal;
        if (!ei_gemma2_metal_batch(ctx, ids, offsets, 9, packed, err, sizeof err))
            ei_die("%s", err);
        for (size_t s = 0; s < 9; s++) {
            float single[768];
            size_t n = offsets[s + 1] - offsets[s];
            if (!ei_gemma2_metal_embed(ctx, ids + offsets[s], NULL, n, single, err, sizeof err))
                ei_die("%s", err);
            compare(route ? "fast batch" : "direct batch", n, single, packed + s * 768, 768,
                    route ? .999 : .999999);
        }
    }
    ei_gguf g;
    ei_gguf_open(&g, argv[2]);
    for (int size = 96; size <= 432; size += 336) {
        uint8_t *rgb = ei_xmalloc((size_t)size * size * 3);
        for (int i = 0; i < size * size * 3; i++)
            rgb[i] = (i * 37 + i / 71) % 256;
        float *a = NULL, *b = NULL;
        size_t na = 0, nb = 0;
        if (!ei_gemma2_metal_vision(e.gemma2_metal, &g, rgb, size, size, &a, &na, err,
                                    sizeof err) ||
            !ei_gemma2_metal_vision(fast, &g, rgb, size, size, &b, &nb, err, sizeof err))
            ei_die("%s", err);
        if (na != nb)
            ei_die("vision token count mismatch");
        compare("vision routes", size, a, b, na * 512, .999);
        free(a);
        free(b);
        free(rgb);
    }
    const int frames[] = {96, 129, 600};
    for (size_t i = 0; i < sizeof frames / sizeof *frames; i++) {
        float *mel = ei_xmalloc((size_t)frames[i] * 128 * 4);
        for (int j = 0; j < frames[i] * 128; j++)
            mel[j] = sinf(j * .017f);
        float *a = NULL, *b = NULL;
        size_t na = 0, nb = 0;
        if (!ei_gemma2_metal_audio(e.gemma2_metal, &g, mel, frames[i], &a, &na, err, sizeof err) ||
            !ei_gemma2_metal_audio(fast, &g, mel, frames[i], &b, &nb, err, sizeof err))
            ei_die("%s", err);
        if (na != nb)
            ei_die("audio token count mismatch");
        compare("audio routes", frames[i], a, b, na * 512, .999);
        free(a);
        free(b);
        free(mel);
    }
    // Packed audio clips (one wave) against the same clips encoded alone: the
    // clip bounds keep the depthwise conv and attention inside each clip.
    {
        const int clip_frames[] = {96, 129, 600, 1000};
        const int clips = 4;
        float *mels[4];
        for (int i = 0; i < clips; i++) {
            mels[i] = ei_xmalloc((size_t)clip_frames[i] * 128 * 4);
            for (int j = 0; j < clip_frames[i] * 128; j++)
                mels[i][j] = sinf(j * .013f + i);
        }
        float *packed_rows = NULL;
        size_t packed_n = 0;
        if (!ei_gemma2_metal_audio_clips(fast, &g, (const float *const *)mels, clip_frames, clips,
                                         &packed_rows, &packed_n, err, sizeof err))
            ei_die("%s", err);
        size_t at = 0;
        for (int i = 0; i < clips; i++) {
            float *single = NULL;
            size_t n = 0;
            if (!ei_gemma2_metal_audio(fast, &g, mels[i], clip_frames[i], &single, &n, err,
                                       sizeof err))
                ei_die("%s", err);
            if (n != (size_t)(clip_frames[i] + 3) / 4 || at + n > packed_n)
                ei_die("packed audio token count mismatch");
            compare("audio packed clip", (size_t)clip_frames[i], single, packed_rows + at * 512,
                    n * 512, .9999);
            at += n;
            free(single);
            free(mels[i]);
        }
        if (at != packed_n)
            ei_die("packed audio token count mismatch");
        free(packed_rows);
    }
    // Persistent layer kernel (EI_GEMMA2_METAL_LAYER=1): bit-identical to the
    // GEMV route with scalar attention (FLASH2 off below 40 tokens) on short
    // waves, single and packed, or it is disabled here (probe failed).
    {
        setenv("EI_GEMMA2_METAL_FLASH2_MIN_TOKENS", "40", 1);
        setenv("EI_GEMMA2_METAL_LAYER", "0", 1);
        void *ref = ei_gemma2_metal_create(&e.model.gguf, err, sizeof err);
        setenv("EI_GEMMA2_METAL_LAYER", "1", 1);
        void *layer = ei_gemma2_metal_create(&e.model.gguf, err, sizeof err);
        unsetenv("EI_GEMMA2_METAL_FLASH2_MIN_TOKENS");
        if (!ref || !layer)
            ei_die("%s", err);
        const size_t sizes[] = {1, 7, 8, 9, 16, 31, 39};
        for (size_t i = 0; i < sizeof sizes / sizeof *sizes; i++) {
            float a[768], b[768];
            if (!ei_gemma2_metal_embed(ref, ids, NULL, sizes[i], a, err, sizeof err) ||
                !ei_gemma2_metal_embed(layer, ids, NULL, sizes[i], b, err, sizeof err))
                ei_die("%s", err);
            if (memcmp(a, b, sizeof a))
                ei_die("persistent layer route differs from the GEMV route at %zu tokens", sizes[i]);
        }
        const size_t packed[] = {0, 3, 11, 20, 39};
        float pa[4 * 768], pb[4 * 768];
        if (!ei_gemma2_metal_batch(ref, ids, packed, 4, pa, err, sizeof err) ||
            !ei_gemma2_metal_batch(layer, ids, packed, 4, pb, err, sizeof err))
            ei_die("%s", err);
        if (memcmp(pa, pb, sizeof pa))
            ei_die("persistent layer route differs from the GEMV route on a packed wave");
        ei_gemma2_metal_free(ref);
        ei_gemma2_metal_free(layer);
        puts("persistent layer route: bit-identical to the GEMV route (1..39 tokens, packed)");
    }
    free(ids);
    ei_gemma2_metal_free(fast);
    ei_engine_free(&e);
    ei_gguf_close(&g);
    puts("EmbeddingGemma 2 Metal routes and packed batches passed");
    return 0;
}
