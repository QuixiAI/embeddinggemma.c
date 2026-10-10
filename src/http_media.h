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
static bool media_part(const char **p, ei_media_input *m, char *err, size_t len) {
    if (!json_expect(p, '{', err, len))
        return false;
    sv_string type = {0}, text = {0}, data = {0};
    string_list frames = {0};
    bool ok = false;
    double fps = 1;
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
                append_input(&frames, frame);
                if (frames.n > 114) {
                    snprintf(err, len, "video accepts at most 114 sampled frames");
                    goto done;
                }
                skip_ws(p);
                if (**p == ']') {
                    ++*p;
                    break;
                }
                if (!json_expect(p, ',', err, len))
                    goto done;
            }
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
    if (string_is(&type, "text") && fields == 3)
        ok = ei_media_text(m, text.s, text.len, err, len);
    else if (string_is(&type, "image") && fields == 5)
        ok = ei_media_image(m, data.s, data.len, 280, err, len);
    else if (string_is(&type, "audio") && fields == 5)
        ok = ei_media_audio(m, data.s, data.len, err, len);
    else if (string_is(&type, "video") && (fields == 9 || fields == 25)) {
        ok = frames.n > 0;
        for (size_t i = 0; ok && i < frames.n; ++i) {
            unsigned seconds = (unsigned)(i / fps);
            char stamp[40];
            int n = snprintf(stamp, sizeof stamp, "%s%02u:%02u ", i ? " " : "", seconds / 60,
                             seconds % 60);
            ok = ei_media_text(m, stamp, (size_t)n, err, len) &&
                 ei_media_image(m, frames.items[i].s, frames.items[i].len, 140, err, len);
        }
    } else
        snprintf(err, len,
                 "content requires text/text, image/data, audio/data, or video/frames with "
                 "optional fps");
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
    ei_media_input m;
    ei_media_input_init(&m, &opts->engine->model.gguf, &opts->engine->tokenizer, opts->projector,
                        opts->engine->thread_pool);
    // With pipeline slots, media owns a context separate from every text slot.
    m.metal = opts->engine->gemma2_media ? opts->engine->gemma2_media : opts->engine->gemma2_metal;
    m.cuda = opts->engine->gemma2_cuda_media ? opts->engine->gemma2_cuda_media
                                             : opts->engine->gemma2_cuda;
    m.lock = &inference_mutex; // engine calls only; parsing and decoding run unlocked
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
    if (ok)
        *usage += m.n;
done:
    free(key.s);
    ei_media_input_free(&m);
    return ok;
}
#endif
