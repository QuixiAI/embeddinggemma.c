/* Included by server.c after the JSON string helpers. */
#ifndef EI_HTTP_MEDIA_H
#define EI_HTTP_MEDIA_H
static bool json_expect(const char **p, char ch, char *err, size_t len) {
    skip_ws(p);
    if (**p != ch) {
        snprintf(err, len, "expected '%c' in multimodal input", ch);
        return false;
    }
    ++*p;
    return true;
}
static bool string_is(const sv_string *s, const char *z) {
    return s->len == strlen(z) && !memcmp(s->s, z, s->len);
}
typedef struct {
    const ei_media_input *m;
    const string_list *frames;
    int budget;
    ei_media_frame *out;
    _Atomic size_t next;
} video_decode_job;
static void *video_decode_worker(void *arg) {
    video_decode_job *job = arg;
    for (size_t i; (i = atomic_fetch_add(&job->next, 1)) < job->frames->n;)
        ei_media_image_prepare(job->m, job->frames->items[i].s, job->frames->items[i].len,
                               job->budget, &job->out[i]);
    return NULL;
}
static void video_decode_all(const ei_media_input *m, const string_list *frames, int budget,
                             ei_media_frame *out) {
    video_decode_job job = {.m = m, .frames = frames, .budget = budget, .out = out};
    size_t threads = frames->n < 8 ? frames->n : 8;
    pthread_t workers[8];
    size_t started = 0;
    for (size_t t = 1; t < threads; ++t)
        if (!pthread_create(&workers[started], NULL, video_decode_worker, &job)) started++;
    video_decode_worker(&job);
    for (size_t t = 0; t < started; ++t)
        pthread_join(workers[t], NULL);
}
static bool media_part(const char **p, ei_media_input *m, char *err, size_t len) {
    if (!json_expect(p, '{', err, len))
        return false;
    sv_string type = {0}, text = {0}, data = {0};
    string_list frames = {0};
    bool ok = false;
    double fps = 1;
    long vision_tokens = 0;
    unsigned fields = 0;
    for (;;) {
        sv_string key = {0};
        if (!parse_json_string(p, &key, err, len))
            goto done;
        if (!json_expect(p, ':', err, len)) {
            free(key.s);
            goto done;
        }
        unsigned bit = 0;
        if (string_is(&key, "type"))
            bit = 1;
        else if (string_is(&key, "text"))
            bit = 2;
        else if (string_is(&key, "data"))
            bit = 4;
        else if (string_is(&key, "frames"))
            bit = 8;
        else if (string_is(&key, "fps"))
            bit = 16;
        else if (string_is(&key, "vision_tokens"))
            bit = 32;
        free(key.s);
        if (!bit || (fields & bit)) {
            snprintf(err, len, "unknown or duplicate content-part field");
            goto done;
        }
        fields |= bit;
        if (bit == 1 || bit == 2 || bit == 4) {
            sv_string *s = bit == 1 ? &type : bit == 2 ? &text : &data;
            if (!parse_json_string(p, s, err, len))
                goto done;
        } else if (bit == 8) {
            if (!json_expect(p, '[', err, len))
                goto done;
            for (;;) {
                sv_string frame = {0};
                if (!parse_json_string(p, &frame, err, len))
                    goto done;
                /* 114 frames at the 70-token minimum fill the context; later
                 * frames could only be truncated away. */
                if (frames.n < 114)
                    append_input(&frames, frame);
                else
                    free(frame.s);
                skip_ws(p);
                if (**p == ']') {
                    ++*p;
                    break;
                }
                if (!json_expect(p, ',', err, len))
                    goto done;
            }
        } else if (bit == 32) {
            skip_ws(p);
            char *end;
            errno = 0;
            vision_tokens = strtol(*p, &end, 10);
            if (errno || end == *p || vision_tokens < 70 || vision_tokens > 1120) {
                snprintf(err, len, "vision_tokens must be an integer from 70 through 1120");
                goto done;
            }
            *p = end;
        } else {
            skip_ws(p);
            char *end;
            errno = 0;
            fps = strtod(*p, &end);
            if (errno || end == *p || !isfinite(fps) || fps < 0.001 || fps > 120) {
                snprintf(err, len, "fps must be between 0.001 and 120");
                goto done;
            }
            *p = end;
        }
        skip_ws(p);
        if (**p == '}') {
            ++*p;
            break;
        }
        if (!json_expect(p, ',', err, len))
            goto done;
    }
    /* Soft tokens per image or frame: the model card defaults, 70..1120 when set. */
    int image_tokens = vision_tokens ? (int)vision_tokens : 280;
    int frame_tokens = vision_tokens ? (int)vision_tokens : 140;
    unsigned media_fields = fields & ~32u;
    if (string_is(&type, "text") && fields == 3)
        ok = ei_media_text(m, text.s, text.len, err, len);
    else if (string_is(&type, "image") && media_fields == 5)
        ok = ei_media_image(m, data.s, data.len, image_tokens, err, len);
    else if (string_is(&type, "audio") && fields == 5)
        ok = ei_media_audio(m, data.s, data.len, err, len);
    else if (string_is(&type, "video") && (media_fields == 9 || media_fields == 25)) {
        ok = frames.n > 0;
        // Frames decode and resize in parallel (up to 8 threads), run their
        // towers as packed waves, then commit in stream order.
        ei_media_frame *prepared = ei_xcalloc(frames.n, sizeof *prepared);
        video_decode_all(m, &frames, frame_tokens, prepared);
        ei_media_frames_encode(m, prepared, frames.n);
        for (size_t i = 0; ok && i < frames.n; ++i) {
            unsigned seconds = (unsigned)(i / fps);
            char stamp[40];
            int n = snprintf(stamp, sizeof stamp, "%s%02u:%02u ", i ? " " : "", seconds / 60,
                             seconds % 60);
            ok = ei_media_text(m, stamp, (size_t)n, err, len) &&
                 ei_media_image_commit(m, &prepared[i], frames.items[i].s, frames.items[i].len,
                                       err, len);
        }
        for (size_t i = 0; i < frames.n; ++i)
            ei_media_frame_free(&prepared[i]);
        free(prepared);
    } else
        snprintf(err, len,
                 "content requires text/text, image/data, audio/data, or video/frames with "
                 "optional fps; images and video accept optional vision_tokens");
done:
    free(type.s);
    free(text.s);
    free(data.s);
    free_inputs(&frames);
    return ok;
}
static bool embed_structured(const server_opts *opts, const sv_string *input, float *out,
                             size_t *usage, char *err, size_t len) {
    if (!opts->engine->model.embeddinggemma2 || !opts->projector) {
        snprintf(err, len, "multimodal input requires the embeddinggemma-2 profile");
        return false;
    }
    // Whole-input cache: the same content object (in any batch) is a lookup.
    // Key: a tag byte and the input text; value: [u64 usage][768 floats].
    char *input_key = NULL;
    size_t input_key_len = 0;
    if (media_cache) {
        input_key_len = input->len + 1;
        input_key = ei_xmalloc(input_key_len);
        input_key[0] = 'E';
        memcpy(input_key + 1, input->s, input->len);
        ei_response_cache_value hit;
        if (ei_response_cache_acquire(media_cache, input_key, input_key_len, &hit) &&
            hit.len == 8 + 768 * sizeof(float)) {
            uint64_t used = 0;
            memcpy(&used, hit.data, 8);
            memcpy(out, hit.data + 8, 768 * sizeof(float));
            ei_response_cache_release(media_cache, &hit);
            *usage += (size_t)used;
            free(input_key);
            return true;
        }
    }
    ei_media_input m;
    ei_media_input_init(&m, &opts->engine->model.gguf, &opts->engine->tokenizer, opts->projector,
                        opts->engine->thread_pool);
    m.cache = media_cache;
    // GPU media runs on a slot of the media pool (own context and lock); the
    // CPU backend shares the engine pool with text waves under inference_mutex.
    // Locks cover engine calls only; parsing and decoding run unlocked.
    media_slot *slot = media_slot_acquire();
    if (slot) {
        m.metal = slot->metal;
        m.cuda = slot->cuda;
        m.lock = &slot->lock;
#if defined(EI_ENABLE_METAL)
        if (media_finisher.enabled) m.finish = media_finish_submit;
        if (media_audio_batcher.enabled) m.audio = media_audio_submit;
#endif
    } else {
        m.metal = opts->engine->gemma2_media ? opts->engine->gemma2_media : opts->engine->gemma2_metal;
        m.cuda = opts->engine->gemma2_cuda_media ? opts->engine->gemma2_cuda_media
                                                 : opts->engine->gemma2_cuda;
        m.lock = &inference_mutex;
    }
    const char *p = input->s;
    bool ok = false;
    sv_string key = {0};
    if (!json_expect(&p, '{', err, len) || !parse_json_string(&p, &key, err, len))
        goto done;
    if (!string_is(&key, "content")) {
        snprintf(err, len, "multimodal input object must contain content");
        goto done;
    }
    if (!json_expect(&p, ':', err, len) || !json_expect(&p, '[', err, len))
        goto done;
    size_t parts = 0;
    for (;;) {
        if (++parts > 256) {
            snprintf(err, len, "content accepts at most 256 parts");
            goto done;
        }
        if (!media_part(&p, &m, err, len))
            goto done;
        skip_ws(&p);
        if (*p == ']') {
            ++p;
            break;
        }
        if (!json_expect(&p, ',', err, len))
            goto done;
    }
    if (!json_expect(&p, '}', err, len))
        goto done;
    skip_ws(&p);
    if (*p) {
        snprintf(err, len, "unexpected content after multimodal input");
        goto done;
    }
    ok = ei_media_finish(&m, out, err, len);
    if (ok) {
        *usage += m.n;
        if (input_key) {
            char blob[8 + 768 * sizeof(float)];
            uint64_t used = m.n;
            memcpy(blob, &used, 8);
            memcpy(blob + 8, out, 768 * sizeof(float));
            ei_response_cache_insert(media_cache, input_key, input_key_len, blob, sizeof blob);
        }
    }
done:
    free(input_key);
    free(key.s);
    ei_media_input_free(&m);
    media_slot_release(slot);
    return ok;
}
#endif
