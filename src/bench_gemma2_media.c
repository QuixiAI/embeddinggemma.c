#define _POSIX_C_SOURCE 200809L
// Time the EmbeddingGemma 2 Metal media towers and the mixed-input text forward.
// usage: [EI_BENCH_STREAMS=N] bench_gemma2_media text.gguf mmproj.gguf [ITEMS...]
// Items: v432 v768 v912x672 video4 a10 a30 mixed (default: all) or vWxH /
// vWxHxF (a packed F-frame wave). Median of 5 after 2 warmups; with
// EI_BENCH_STREAMS=N, N slot clones run the item concurrently and the
// aggregate rate is reported; EI_BENCH_REPEAT=N repeats the list round-robin
// (route flags come from the environment).
#include "engine.h"
#include "gemma2_metal.h"
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}
static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
typedef struct {
    const char *name;
    int width, height, frames; // vision when width > 0 (frames > 1: one packed
                               // video wave), audio when width == 0 && frames > 0
    int clips;                 // audio: clips of `frames` frames as one packed wave
} item;
typedef struct {
    void *ctx;
    const ei_gguf *proj;
    const item *it;
    const uint8_t *rgb;
    const float *mel;
    const float *inputs;
    size_t inputs_n;
    int iterations;
    size_t tokens;
    double ms; // median per call
} job;
static void run_once(job *j) {
    char err[512];
    float *soft = NULL;
    size_t n = 0;
    bool ok;
    if (j->it->width && j->it->frames > 1) {
        const uint8_t *rgb[64];
        size_t stride = (size_t)j->it->width * j->it->height * 3;
        for (int f = 0; f < j->it->frames && f < 64; f++)
            rgb[f] = j->rgb + stride * f;
        ok = ei_gemma2_metal_vision_frames(j->ctx, j->proj, rgb, j->it->frames, j->it->width,
                                           j->it->height, &soft, &n, err, sizeof err);
    } else if (j->it->width)
        ok = ei_gemma2_metal_vision(j->ctx, j->proj, j->rgb, j->it->width, j->it->height, &soft, &n,
                                    err, sizeof err);
    else if (j->it->frames && j->it->clips > 1) {
        const float *mel[64];
        int frames[64];
        for (int i = 0; i < j->it->clips && i < 64; i++) {
            mel[i] = j->mel;
            frames[i] = j->it->frames;
        }
        ok = ei_gemma2_metal_audio_clips(j->ctx, j->proj, mel, frames, j->it->clips, &soft, &n,
                                         err, sizeof err);
    } else if (j->it->frames)
        ok = ei_gemma2_metal_audio(j->ctx, j->proj, j->mel, j->it->frames, &soft, &n, err,
                                   sizeof err);
    else {
        float out[768];
        ok = ei_gemma2_metal_embed(j->ctx, NULL, j->inputs, j->inputs_n, out, err, sizeof err);
        n = j->inputs_n;
    }
    if (!ok)
        ei_die("%s", err);
    free(soft);
    j->tokens = n;
}
static void *run_job(void *arg) {
    job *j = arg;
    double times[64];
    for (int i = -2; i < j->iterations; i++) {
        double start = now_ms();
        run_once(j);
        if (i >= 0)
            times[i] = now_ms() - start;
    }
    qsort(times, (size_t)j->iterations, sizeof *times, cmp);
    j->ms = times[j->iterations / 2];
    return NULL;
}
int main(int argc, char **argv) {
    if (argc < 3)
        ei_die("usage: bench_gemma2_media text.gguf mmproj.gguf [ITEMS...]");
    ei_engine e;
    ei_engine_load_backend(&e, argv[1], "metal");
    if (!e.gemma2_metal)
        ei_die("Metal EmbeddingGemma 2 context unavailable");
    ei_gguf proj;
    ei_gguf_open(&proj, argv[2]);
    const char *streams_env = getenv("EI_BENCH_STREAMS");
    int streams = streams_env ? atoi(streams_env) : 1;
    if (streams < 1 || streams > 16)
        ei_die("EI_BENCH_STREAMS must be 1..16");
    static const item items[] = {
        {"v432", 432, 432, 0, 0}, {"v768", 768, 768, 0, 0}, {"v912x672", 912, 672, 0, 0},
        {"video4", 432, 432, 4, 0}, {"a10", 0, 0, 1000, 0}, {"a30", 0, 0, 3000, 0},
        {"mixed", 0, 0, 0, 0},
    };
    size_t count = sizeof items / sizeof *items;
    char err[512];
    void *ctxs[16];
    ctxs[0] = e.gemma2_metal;
    for (int s = 1; s < streams; s++)
        if (!(ctxs[s] = ei_gemma2_metal_clone(e.gemma2_metal, err, sizeof err)))
            ei_die("%s", err);
    // Synthetic inputs: textured pixels, a two-tone mel, and 335 float rows
    // (text + 256-token image + 2 delimiters + 75 audio-like rows) for the mixed forward.
    size_t pixels = (size_t)1536 * 1536 * 3 * 2; // room for 8 432x432 frames too
    uint8_t *rgb = ei_xmalloc(pixels);
    for (size_t i = 0; i < pixels; i++)
        rgb[i] = (uint8_t)((i * 37 + i / 71) % 256);
    float *mel = ei_xmalloc((size_t)3000 * 128 * sizeof *mel);
    for (size_t i = 0; i < (size_t)3000 * 128; i++)
        mel[i] = sinf((float)i * .017f);
    size_t mixed_n = 335;
    float *inputs = ei_xmalloc(mixed_n * 512 * sizeof *inputs);
    for (size_t i = 0; i < mixed_n * 512; i++)
        inputs[i] = sinf((float)i * .0013f) * 2;
    printf("streams %d (median of 5 after 2 warmups; aggregate rate across streams)\n", streams);
    // Extra sizes on the command line: vWIDTHxHEIGHT (48-pixel multiples),
    // vWIDTHxHEIGHTxFRAMES for a packed video wave, aFRAMESxCLIPS for a packed
    // audio wave of CLIPS clips of FRAMES mel frames (10 ms each).
    item extra[32];
    size_t extras = 0;
    for (int a = 3; a < argc && extras < 32; a++) {
        int w = 0, h = 0, f = 0;
        bool known = false;
        for (size_t i = 0; i < count; i++)
            known |= !strcmp(argv[a], items[i].name);
        if (known)
            continue;
        int got = sscanf(argv[a], "v%dx%dx%d", &w, &h, &f), clips = 0;
        if (got >= 2 && w > 0 && h > 0 && f <= 64 &&
            (size_t)w * h * (f > 1 ? f : 1) * 3 <= pixels)
            extra[extras++] = (item){argv[a], w, h, got == 3 ? f : 0, 0};
        else if (sscanf(argv[a], "a%dx%d", &f, &clips) == 2 && f >= 1 && f <= 3000 && clips >= 1 &&
                 clips <= 64)
            extra[extras++] = (item){argv[a], 0, 0, f, clips};
    }
    // EI_BENCH_REPEAT=N runs the item list N times round-robin so configurations
    // interleave on this shared host instead of drifting apart.
    const char *repeat_env = getenv("EI_BENCH_REPEAT");
    int repeat = repeat_env ? atoi(repeat_env) : 1;
    for (int pass = 0; pass < (repeat > 0 ? repeat : 1); pass++)
    for (size_t i = 0; i < count + extras; i++) {
        const item *it = i < count ? &items[i] : &extra[i - count];
        bool wanted = argc == 3;
        for (int a = 3; a < argc; a++)
            wanted |= !strcmp(argv[a], it->name);
        if (!wanted)
            continue;
        job jobs[16];
        pthread_t threads[16];
        for (int s = 0; s < streams; s++) {
            jobs[s] = (job){.ctx = ctxs[s], .proj = &proj, .it = it, .rgb = rgb, .mel = mel,
                            .inputs = inputs, .inputs_n = mixed_n, .iterations = 5};
            pthread_create(&threads[s], NULL, run_job, &jobs[s]);
        }
        double sum_rate = 0, ms = 0;
        for (int s = 0; s < streams; s++) {
            pthread_join(threads[s], NULL);
            sum_rate += 1000.0 / jobs[s].ms;
            ms += jobs[s].ms / streams;
        }
        printf("%-9s %5zu tokens  %8.2f ms/call  %7.2f items/s  %9.0f tok/s\n", it->name,
               jobs[0].tokens, ms, sum_rate, sum_rate * (double)jobs[0].tokens);
    }
    return 0;
}
