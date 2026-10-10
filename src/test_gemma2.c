#include "engine.h"
#include "gemma2.h"
#include <math.h>
#if defined(EI_ENABLE_METAL)
#include "gemma2_metal.h"
#endif
#if defined(EI_ENABLE_CUDA)
#include "gemma2_cuda.h"
#endif

static void vector(const float *x, size_t n) {
    putchar('[');
    for (size_t i = 0; i < n; ++i)
        printf("%s%.9g", i ? "," : "", x[i]);
    putchar(']');
}
int main(int argc, char **argv) {
    if (argc < 4)
        ei_die("usage: %s text.gguf projector.gguf text|vision|audio [text]", argv[0]);
    ei_engine e;
    ei_engine_load(&e, argv[1]);
    char err[256];
    float out[768];
    if (!strcmp(argv[3], "text")) {
        if (argc != 5)
            return 2;
        ei_tokens t;
        ei_tokenize_spm(&e.tokenizer, argv[4], strlen(argv[4]), true, false, &t);
        if (!ei_engine_embed_tokens(&e, t.ids, t.n, out, err, sizeof err))
            ei_die("%s", err);
        printf("{\"backend\":\"%s\",\"tokens\":[", ei_engine_backend(&e));
        for (size_t i = 0; i < t.n; ++i)
            printf("%s%d", i ? "," : "", t.ids[i]);
        printf("],\"embedding\":");
        vector(out, 768);
        puts("}");
        ei_tokens_free(&t);
    } else {
        ei_gguf projector;
        ei_gguf_open(&projector, argv[2]);
        bool audio = !strcmp(argv[3], "audio");
        float *tower = NULL;
        size_t n = 0;
        if (audio) {
            float mel[96 * 128];
            for (int i = 0; i < 96 * 128; ++i)
                mel[i] = sinf(i * 0.017f);
#if defined(EI_ENABLE_METAL)
            if (e.gemma2_metal) {
                if (!ei_gemma2_metal_audio(e.gemma2_metal, &projector, mel, 96, &tower, &n, err,
                                           sizeof err))
                    ei_die("%s", err);
            } else
#endif
#if defined(EI_ENABLE_CUDA)
                if (e.gemma2_cuda) {
                if (!ei_gemma2_cuda_audio(e.gemma2_cuda, &projector, mel, 96, &tower, &n, err,
                                          sizeof err))
                    ei_die("%s", err);
            } else
#endif
                if (!ei_gemma2_audio(&projector, e.thread_pool, mel, 96, &tower, &n, err,
                                     sizeof err))
                ei_die("%s", err);
        } else {
            uint8_t rgb[96 * 96 * 3];
            for (size_t i = 0; i < sizeof rgb; ++i)
                rgb[i] = (i * 37 + i / 71) % 256;
#if defined(EI_ENABLE_METAL)
            if (e.gemma2_metal) {
                if (!ei_gemma2_metal_vision(e.gemma2_metal, &projector, rgb, 96, 96, &tower, &n,
                                            err, sizeof err))
                    ei_die("%s", err);
            } else
#endif
#if defined(EI_ENABLE_CUDA)
                if (e.gemma2_cuda) {
                if (!ei_gemma2_cuda_vision(e.gemma2_cuda, &projector, rgb, 96, 96, &tower, &n,
                                           err, sizeof err))
                    ei_die("%s", err);
            } else
#endif
                if (!ei_gemma2_vision(&projector, e.thread_pool, rgb, 96, 96, &tower, &n, err,
                                      sizeof err))
                ei_die("%s", err);
        }
        float *inputs = ei_xmalloc((n + 4) * 512 * sizeof(float));
        const ei_tensor *w = ei_gguf_tensor(&e.model.gguf, "token_embd.weight", true);
        int ids[] = {2, audio ? 256000 : 255999, audio ? 258883 : 258882, 1};
        size_t pos[] = {0, 1, n + 2, n + 3};
        for (int i = 0; i < 4; ++i)
            ei_dequantize_row_q8_0_scaled(w, ids[i], sqrtf(512.0f), inputs + pos[i] * 512);
        memcpy(inputs + 1024, tower, n * 512 * sizeof(float));
#if defined(EI_ENABLE_METAL)
        if (e.gemma2_metal) {
            if (!ei_gemma2_metal_embed(e.gemma2_metal, NULL, inputs, n + 4, out, err, sizeof err))
                ei_die("%s", err);
        } else
#endif
#if defined(EI_ENABLE_CUDA)
            if (e.gemma2_cuda) {
            if (!ei_gemma2_cuda_embed_inputs(e.gemma2_cuda, inputs, n + 4, out, err, sizeof err))
                ei_die("%s", err);
        } else
#endif
            if (!ei_gemma2_embed(&e.model.gguf, e.thread_pool, NULL, inputs, n + 4, out, err,
                                 sizeof err))
            ei_die("%s", err);
        printf("{\"backend\":\"%s\",\"tower\":", ei_engine_backend(&e));
        vector(tower, n * 512);
        printf(",\"embedding\":");
        vector(out, 768);
        puts("}");
        free(inputs);
        free(tower);
        ei_gguf_close(&projector);
    }
    ei_engine_free(&e);
    return 0;
}
