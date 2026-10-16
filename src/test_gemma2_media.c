/* cpu-max fork: parity and timing of the fast CPU media towers against the
 * unchanged reference graph (same process, same pool, same inputs).
 *
 *   test_gemma2_media text.gguf mmproj.gguf vision A.png [B.png ...]
 *   test_gemma2_media text.gguf mmproj.gguf vision-synthetic
 *
 * Images must already be 48-pixel aligned (the fixtures are). Prints one JSON
 * line per input: soft-token count, flattened and minimum per-token cosine,
 * max |diff|, reference and fast tower milliseconds (median of
 * EI_MEDIA_TEST_REPEAT fast runs). EI_MEDIA_TEST_REF=0 skips the reference.
 * Exit status 1 if a gate in check() fails. */
#if defined(__linux__)
#define _POSIX_C_SOURCE 200809L
#endif
#include "engine.h"
#include "gemma2.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define STBI_NO_STDIO
#include "vendor/stb_image.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}
static uint8_t *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f)
        ei_die("cannot open %s", path);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = ei_xmalloc((size_t)n);
    if (fread(data, 1, (size_t)n, f) != (size_t)n)
        ei_die("cannot read %s", path);
    fclose(f);
    *size = (size_t)n;
    return data;
}
static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
static int env_int(const char *name, int fallback) {
    const char *v = getenv(name);
    return v ? atoi(v) : fallback;
}

/* Gate: flattened cosine >= EI_MEDIA_TEST_MIN_COS (default 0.999) and every token
 * >= EI_MEDIA_TEST_MIN_TOKEN_COS (default 0.95). Both graphs quantize activations
 * to Q8_0, so they differ by rounding noise; against the HF FP32 tower on real
 * inputs each graph's worst token is ~0.95 (scripts/compare_*_towers.py). */
static bool check(const char *label, const float *ref, const float *fast, size_t tokens, double ref_ms,
                  double fast_ms, double min_allowed) {
    double min_token_allowed =
        getenv("EI_MEDIA_TEST_MIN_TOKEN_COS") ? atof(getenv("EI_MEDIA_TEST_MIN_TOKEN_COS")) : 0.95;
    double dot = 0, rr = 0, ff = 0, min_tok = 1, max_diff = 0;
    for (size_t t = 0; t < tokens; t++) {
        double d = 0, a = 0, b = 0;
        for (int z = 0; z < 512; z++) {
            double x = ref[t * 512 + z], y = fast[t * 512 + z];
            d += x * y, a += x * x, b += y * y;
            if (fabs(x - y) > max_diff)
                max_diff = fabs(x - y);
        }
        dot += d, rr += a, ff += b;
        double c = d / sqrt(a * b);
        if (c < min_tok)
            min_tok = c;
    }
    double cos = dot / sqrt(rr * ff);
    printf("{\"input\":\"%s\",\"tokens\":%zu,\"cosine\":%.9f,\"min_token_cosine\":%.9f,\"max_abs_diff\":%.6g,"
           "\"ref_ms\":%.2f,\"fast_ms\":%.2f,\"speedup\":%.2f}\n",
           label, tokens, cos, min_tok, max_diff, ref_ms, fast_ms, ref_ms > 0 ? ref_ms / fast_ms : 0.0);
    fflush(stdout);
    return cos >= min_allowed && min_tok >= min_token_allowed && isfinite(cos);
}

typedef struct {
    const uint8_t *rgb;
    int w, h;
    const float *mel;
    int frames;
} media_input;
static bool tower(const ei_gguf *proj, ei_thread_pool *pool, const media_input *in, bool fast, float **out,
                  size_t *n, char *err, size_t len) {
    if (in->mel)
        return fast ? ei_gemma2_audio_fast(proj, pool, in->mel, in->frames, out, n, err, len)
                    : ei_gemma2_audio_ref(proj, pool, in->mel, in->frames, out, n, err, len);
    return fast ? ei_gemma2_vision_fast(proj, pool, in->rgb, in->w, in->h, out, n, err, len)
                : ei_gemma2_vision_ref(proj, pool, in->rgb, in->w, in->h, out, n, err, len);
}
static bool run_vision(ei_engine *e, const ei_gguf *proj, const char *label, const media_input *in) {
    char err[256];
    int repeat = env_int("EI_MEDIA_TEST_REPEAT", 3);
    bool do_ref = env_int("EI_MEDIA_TEST_REF", 1) != 0;
    double min_allowed = getenv("EI_MEDIA_TEST_MIN_COS") ? atof(getenv("EI_MEDIA_TEST_MIN_COS")) : 0.999;
    float *fast = NULL, *ref = NULL;
    size_t nf = 0, nr = 0;
    double times[32];
    if (repeat > 32)
        repeat = 32;
    for (int r = 0; r < repeat; r++) {
        free(fast);
        double t0 = now_ms();
        if (!tower(proj, e->thread_pool, in, true, &fast, &nf, err, sizeof err))
            ei_die("%s", err);
        times[r] = now_ms() - t0;
    }
    qsort(times, (size_t)repeat, sizeof *times, cmp_double);
    double ref_ms = 0;
    if (do_ref) {
        double t0 = now_ms();
        if (!tower(proj, e->thread_pool, in, false, &ref, &nr, err, sizeof err))
            ei_die("%s", err);
        ref_ms = now_ms() - t0;
        if (nr != nf)
            ei_die("token count mismatch %zu vs %zu", nr, nf);
    }
    const char *dump = getenv("EI_MEDIA_TEST_DUMP"); /* prefix: <prefix>.fast.f32 / .ref.f32 */
    if (dump) {
        char path[512];
        for (int which = 0; which < 2; which++) {
            const float *src = which ? ref : fast;
            if (!src)
                continue;
            snprintf(path, sizeof path, "%s.%s.f32", dump, which ? "ref" : "fast");
            FILE *f = fopen(path, "wb");
            if (!f || fwrite(src, sizeof(float), nf * 512, f) != nf * 512)
                ei_die("cannot write %s", path);
            fclose(f);
        }
    }
    bool ok = true;
    if (do_ref)
        ok = check(label, ref, fast, nf, ref_ms, times[repeat / 2], min_allowed);
    else
        printf("{\"input\":\"%s\",\"tokens\":%zu,\"fast_ms\":%.2f}\n", label, nf, times[repeat / 2]);
    for (size_t i = 0; i < nf * 512; i++)
        if (!isfinite(fast[i]))
            ok = false;
    free(fast);
    free(ref);
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 4)
        ei_die("usage: %s text.gguf mmproj.gguf vision IMAGE.png... | vision-synthetic | audio MEL.f32... | audio-synthetic", argv[0]);
    ei_engine e;
    ei_engine_load(&e, argv[1]);
    ei_gguf proj;
    ei_gguf_open(&proj, argv[2]);
    bool ok = true;
    if (!strcmp(argv[3], "vision-synthetic")) {
        /* Gradients and a checkerboard at three aspect ratios / budgets. */
        const int sizes[][2] = {{96, 96}, {480, 336}, {144, 432}};
        for (int s = 0; s < 3; s++) {
            int w = sizes[s][0], h = sizes[s][1];
            uint8_t *rgb = ei_xmalloc((size_t)w * h * 3);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) {
                    uint8_t *p = rgb + ((size_t)y * w + x) * 3;
                    p[0] = (uint8_t)(x * 255 / w);
                    p[1] = (uint8_t)(y * 255 / h);
                    p[2] = ((x / 8 + y / 8) & 1) ? 230 : 20;
                }
            char label[64];
            snprintf(label, sizeof label, "synthetic-%dx%d", w, h);
            media_input in = {.rgb = rgb, .w = w, .h = h};
            ok &= run_vision(&e, &proj, label, &in);
            free(rgb);
        }
    } else if (!strcmp(argv[3], "vision")) {
        for (int i = 4; i < argc; i++) {
            size_t size;
            uint8_t *data = read_file(argv[i], &size);
            int w, h, c;
            uint8_t *rgb = stbi_load_from_memory(data, (int)size, &w, &h, &c, 3);
            free(data);
            if (!rgb)
                ei_die("cannot decode %s", argv[i]);
            media_input in = {.rgb = rgb, .w = w, .h = h};
            ok &= run_vision(&e, &proj, argv[i], &in);
            stbi_image_free(rgb);
        }
    } else if (!strcmp(argv[3], "audio")) {
        /* Raw float32 log-mel features, [frames][128] (e.g. from the HF extractor). */
        for (int i = 4; i < argc; i++) {
            size_t size;
            float *mel = (float *)read_file(argv[i], &size);
            media_input in = {.mel = mel, .frames = (int)(size / (128 * sizeof(float)))};
            ok &= run_vision(&e, &proj, argv[i], &in);
            free(mel);
        }
    } else if (!strcmp(argv[3], "audio-synthetic")) {
        const int lens[] = {96, 499, 1499};
        for (int s = 0; s < 3; s++) {
            float *mel = ei_xmalloc((size_t)lens[s] * 128 * sizeof(float));
            for (int i = 0; i < lens[s] * 128; i++)
                mel[i] = sinf(i * 0.017f) * 2.0f - 4.0f;
            char label[64];
            snprintf(label, sizeof label, "synthetic-mel-%d", lens[s]);
            media_input in = {.mel = mel, .frames = lens[s]};
            ok &= run_vision(&e, &proj, label, &in);
            free(mel);
        }
    } else
        ei_die("unknown mode %s", argv[3]);
    ei_gguf_close(&proj);
    ei_engine_free(&e);
    return ok ? 0 : 1;
}
