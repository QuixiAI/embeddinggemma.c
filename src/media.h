#ifndef EI_MEDIA_H
#define EI_MEDIA_H
#include "gemma2.h"
#include "tokenizer.h"
#include <pthread.h>
typedef struct {
    const ei_gguf *text_model;
    const ei_tokenizer *tokenizer;
    const ei_gguf *projector;
    ei_thread_pool *pool;
    void *metal;
    void *cuda; /* CUDA engine: towers and the final text pass */
    /* Optional: held only around engine calls (towers and the final text
     * pass), so decoding and feature extraction run concurrently. */
    pthread_mutex_t *lock;
    float *inputs;
    size_t n;
} ei_media_input;
void ei_media_input_init(ei_media_input *m, const ei_gguf *text, const ei_tokenizer *tok,
                         const ei_gguf *projector, ei_thread_pool *pool);
void ei_media_input_free(ei_media_input *m);
bool ei_media_text(ei_media_input *m, const char *text, size_t len, char *err, size_t err_len);
bool ei_media_image(ei_media_input *m, const char *base64, size_t len, int budget, char *err,
                    size_t err_len);
bool ei_media_audio(ei_media_input *m, const char *base64, size_t len, char *err, size_t err_len);
bool ei_media_finish(ei_media_input *m, float out[768], char *err, size_t err_len);
#endif
