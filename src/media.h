#ifndef EI_MEDIA_H
#define EI_MEDIA_H
#include "gemma2.h"
#include "response_cache.h"
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
    /* Optional: runs the final text pass over the assembled input rows
     * (e.g. packed with other requests' finishes) instead of the engine call. */
    bool (*finish)(void *ctx, const float *inputs, size_t n, float *out, char *err,
                   size_t err_len);
    void *finish_ctx;
    /* Optional: runs the audio tower over a clip's mel frames (e.g. packed with
     * other requests' clips into one wave) instead of the engine call. On
     * success *out is a malloc'd (frames + 3) / 4 x 512 block. */
    bool (*audio)(void *ctx, const float *mel, int frames, float **out, size_t *n, char *err,
                  size_t err_len);
    void *audio_ctx;
    /* Optional media cache (ei_response_cache, thread-safe): tower outputs by
     * content hash, so a repeated image, frame or clip skips decode and tower. */
    ei_response_cache *cache;
    float *inputs;
    size_t n;
} ei_media_input;
void ei_media_input_init(ei_media_input *m, const ei_gguf *text, const ei_tokenizer *tok,
                         const ei_gguf *projector, ei_thread_pool *pool);
void ei_media_input_free(ei_media_input *m);
bool ei_media_text(ei_media_input *m, const char *text, size_t len, char *err, size_t err_len);
bool ei_media_image(ei_media_input *m, const char *base64, size_t len, int budget, char *err,
                    size_t err_len);
/* A decoded, resized image canvas ready for the vision tower (video frames are
 * prepared in parallel, then committed in order). */
typedef struct {
    uint8_t *canvas; /* NULL: nothing to add (dropped or decode failed, see ok) */
    int width, height, budget;
    bool ok;
    char err[160];
    float *soft; /* set by ei_media_frames_encode or a cache hit: the tower's output rows */
    size_t tokens;
    unsigned char key[24]; /* media cache key when keyed */
    bool keyed;
} ei_media_frame;
void ei_media_image_prepare(const ei_media_input *m, const char *base64, size_t len, int budget,
                            ei_media_frame *frame);
/* Run the vision tower over prepared frames ahead of their commits. On Metal,
 * runs of same-sized frames go through one packed wave
 * (EI_GEMMA2_VIDEO_BATCH=0: one frame per wave). Frames without room at
 * commit time are re-encoded there. Errors surface at the frame's commit. */
void ei_media_frames_encode(ei_media_input *m, ei_media_frame *frames, size_t n);
void ei_media_frame_free(ei_media_frame *frame);
bool ei_media_image_commit(ei_media_input *m, ei_media_frame *frame, const char *base64,
                           size_t len, char *err, size_t err_len);
bool ei_media_audio(ei_media_input *m, const char *base64, size_t len, char *err, size_t err_len);
bool ei_media_finish(ei_media_input *m, float out[768], char *err, size_t err_len);
#endif
