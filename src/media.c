#define _POSIX_C_SOURCE 200809L /* clock_gettime */
#if defined(EI_ENABLE_METAL)
#include "gemma2_metal.h"
#endif
#if defined(EI_ENABLE_CUDA)
#include "gemma2_cuda.h"
#endif
#include "media.h"
#include <pthread.h>
#include <math.h>
#include <limits.h>
#if defined(__APPLE__)
/* PNG IDAT inflate through libcompression (several times stb's inflate on a
 * 768x768 photo); stb's own path remains for the zlib header checks it does
 * not do, for output sizes other than the exact raw length (interlaced files)
 * and when EI_PNG_INFLATE=0. Output bytes are identical: inflate is exact. */
#include <compression.h>
static char *stb_zlib_fallback(const char *buffer, int len, int initial_size, int *outlen,
                               int parse_header);
static char *ei_png_inflate(const char *buffer, int len, int initial_size, int *outlen,
                            int parse_header) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *e = getenv("EI_PNG_INFLATE");
        enabled = !(e && !strcmp(e, "0"));
    }
    const unsigned char *src = (const unsigned char *)buffer;
    size_t src_len = (size_t)len;
    if (enabled && parse_header && len > 6) {
        unsigned cmf = src[0], flg = src[1];
        /* deflate method, no preset dictionary, header check. */
        if ((cmf & 15) == 8 && !(flg & 32) && (cmf * 256 + flg) % 31 == 0) {
            src += 2;
            src_len -= 6; /* header and adler32 trailer */
            size_t guess = (size_t)initial_size, cap = guess + 64;
            char *dst = malloc(cap);
            if (dst) {
                size_t got = compression_decode_buffer((uint8_t *)dst, cap, src, src_len, NULL,
                                                       COMPRESSION_ZLIB);
                if (got == guess) {
                    *outlen = (int)got;
                    return dst;
                }
                free(dst);
            }
        }
    }
    return stb_zlib_fallback(buffer, len, initial_size, outlen, parse_header);
}
#define STBI_ZLIB_DECODE_HOOK ei_png_inflate
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_PNM
#define STBI_MAX_DIMENSIONS 8192
#include "vendor/stb_image.h"
#if defined(__APPLE__)
static char *stb_zlib_fallback(const char *buffer, int len, int initial_size, int *outlen,
                               int parse_header) {
    return stbi_zlib_decode_malloc_guesssize_headerflag(buffer, len, initial_size, outlen,
                                                        parse_header);
}
#endif
#include <time.h>
/* EI_MEDIA_TIMING=1: per-stage wall times of the media front ends on stderr. */
static bool media_timing(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("EI_MEDIA_TIMING");
        on = e && strcmp(e, "0") != 0;
    }
    return on;
}
static double media_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
#define MEDIA_STAGE(label, since)                                                              \
    do {                                                                                      \
        if (media_timing()) {                                                                 \
            double now_ = media_now_ms();                                                     \
            fprintf(stderr, "media %-10s %7.2f ms\n", label, now_ - (since));               \
            (since) = now_;                                                                   \
        }                                                                                     \
    } while (0)

#define MEDIA_CONTEXT 8192
static bool fail(char *err, size_t n, const char *message) {
    snprintf(err, n, "%s", message);
    return false;
}
/* Inputs never exceed the shared context: text is clipped and media shrunk or
 * trimmed to the room left, with a slot always reserved for the final EOS. */
static size_t avail(const ei_media_input *m) {
    size_t cap = MEDIA_CONTEXT - (m->tokenizer->model->add_eos ? 1 : 0);
    return m->n < cap ? cap - m->n : 0;
}
static void rows(ei_media_input *m, const float *src, size_t n) {
    m->inputs = ei_xrealloc(m->inputs, (m->n + n) * 512 * sizeof(float));
    memcpy(m->inputs + m->n * 512, src, n * 512 * sizeof(float));
    m->n += n;
}
static void token(ei_media_input *m, int32_t id) {
    const ei_tensor *w = ei_gguf_tensor(m->text_model, "token_embd.weight", true);
    m->inputs = ei_xrealloc(m->inputs, (m->n + 1) * 512 * sizeof(float));
    ei_dequantize_row_q8_0(w, id, m->inputs + m->n * 512);
    for (int z = 0; z < 512; ++z) m->inputs[m->n * 512 + z] *= sqrtf(512.0f);
    m->n++;
}
void ei_media_input_init(ei_media_input *m, const ei_gguf *text, const ei_tokenizer *tok,
                         const ei_gguf *projector, ei_thread_pool *pool) {
    *m = (ei_media_input){
        .text_model = text, .tokenizer = tok, .projector = projector, .pool = pool};
    if (tok->model->add_bos)
        token(m, tok->model->bos_id);
}
void ei_media_input_free(ei_media_input *m) {
    free(m->inputs);
    m->inputs = NULL;
    m->n = 0;
}
bool ei_media_text(ei_media_input *m, const char *text, size_t len, char *err, size_t err_len) {
    ei_tokens t;
    ei_tokenize_spm(m->tokenizer, text, len, false, false, &t);
    (void)err;
    (void)err_len;
    for (size_t i = 0; i < t.n && avail(m); ++i)
        token(m, t.ids[i]);
    ei_tokens_free(&t);
    return true;
}
static int b64digit(unsigned char c) {
    if (c >= 'A' && c <= 'Z')
        return c - 'A';
    if (c >= 'a' && c <= 'z')
        return c - 'a' + 26;
    if (c >= '0' && c <= '9')
        return c - '0' + 52;
    if (c == '+')
        return 62;
    if (c == '/')
        return 63;
    return -1;
}
static uint8_t *decode(const char *data, size_t len, size_t *size) {
    if (len > 5 && !memcmp(data, "data:", 5)) {
        const char *comma = memchr(data, ',', len);
        if (!comma || comma - data < 7 || memcmp(comma - 7, ";base64", 7))
            return NULL;
        len -= comma + 1 - data;
        data = comma + 1;
    }
    if (!len || len % 4 || len > 16 * 1024 * 1024)
        return NULL;
    uint8_t *out = ei_xmalloc(len / 4 * 3);
    size_t n = 0;
    /* Every quad but the last holds no padding: a 256-entry table decodes
     * them 4 at a time with one validity check per quad (invalid bytes and
     * '=' map to 0xFF). The last quad takes the padding rules below. */
    static uint8_t table[256];
    static bool table_ready = false;
    if (!table_ready) {
        memset(table, 0xFF, sizeof table);
        for (int c = 0; c < 256; ++c) {
            int v = b64digit((unsigned char)c);
            if (v >= 0)
                table[c] = (uint8_t)v;
        }
        table_ready = true;
    }
    size_t i = 0, fast = len - 4;
    const uint8_t *in = (const uint8_t *)data;
    for (; i < fast; i += 4) {
        uint8_t a = table[in[i]], b = table[in[i + 1]], c = table[in[i + 2]],
                d = table[in[i + 3]];
        if ((a | b | c | d) & 0xC0) {
            free(out);
            return NULL;
        }
        uint32_t v = (uint32_t)a << 18 | (uint32_t)b << 12 | (uint32_t)c << 6 | d;
        out[n] = (uint8_t)(v >> 16);
        out[n + 1] = (uint8_t)(v >> 8);
        out[n + 2] = (uint8_t)v;
        n += 3;
    }
    for (; i < len; i += 4) {
        int a = b64digit(data[i]), b = b64digit(data[i + 1]);
        int c = data[i + 2] == '=' ? 0 : b64digit(data[i + 2]),
            d = data[i + 3] == '=' ? 0 : b64digit(data[i + 3]);
        bool p2 = data[i + 2] == '=', p3 = data[i + 3] == '=';
        if (a < 0 || b < 0 || c < 0 || d < 0 || (p2 && !p3) || ((p2 || p3) && i + 4 != len) ||
            (p2 && (b & 15)) || (p3 && !p2 && (c & 3))) {
            free(out);
            return NULL;
        }
        out[n++] = (uint8_t)((a << 2) | (b >> 4));
        if (!p2)
            out[n++] = (uint8_t)((b << 4) | (c >> 2));
        if (!p3)
            out[n++] = (uint8_t)((c << 6) | d);
    }
    *size = n;
    return out;
}
static double cubic(double x) {
    x = fabs(x);
    if (x < 1)
        return (1.5 * x - 2.5) * x * x + 1;
    if (x < 2)
        return ((-0.5 * x + 2.5) * x - 4) * x + 2;
    return 0;
}
/* Separable antialiased Pillow-compatible bicubic filter, rounded after each axis. */
static uint8_t *resize_axis(const uint8_t *src, int width, int height, int target, bool vertical) {
    if (target == (vertical ? height : width)) {
        size_t bytes = (size_t)width * height * 3;
        uint8_t *copy = ei_xmalloc(bytes);
        memcpy(copy, src, bytes);
        return copy;
    }
    int before = vertical ? height : width, lines = vertical ? width : height;
    uint8_t *out = ei_xmalloc((size_t)target * lines * 3);
    double scale = (double)before / target, filter = fmax(1, scale), support = 2 * filter;
    int max = (int)ceil(support) * 2 + 1;
    int32_t *weights = ei_xmalloc((size_t)max * sizeof(*weights));
    for (int p = 0; p < target; ++p) {
        double center = (p + 0.5) * scale;
        int lo = (int)(center - support + 0.5), hi = (int)(center + support + 0.5);
        if (lo < 0)
            lo = 0;
        if (hi > before)
            hi = before;
        double sum = 0;
        for (int i = lo; i < hi; ++i)
            sum += cubic((i - center + 0.5) / filter);
        for (int i = lo; i < hi; ++i) {
            double w = cubic((i - center + 0.5) / filter) / sum * 4194304.0;
            weights[i - lo] = (int32_t)(w + (w < 0 ? -0.5 : 0.5));
        }
        for (int line = 0; line < lines; ++line)
            for (int c = 0; c < 3; ++c) {
                int64_t total = 1 << 21;
                for (int i = lo; i < hi; ++i) {
                    size_t at = vertical ? ((size_t)i * width + line) * 3 + c
                                         : ((size_t)line * width + i) * 3 + c;
                    total += (int64_t)src[at] * weights[i - lo];
                }
                int v = (int)(total >> 22);
                if (v < 0)
                    v = 0;
                if (v > 255)
                    v = 255;
                size_t at = vertical ? ((size_t)p * width + line) * 3 + c
                                     : ((size_t)line * target + p) * 3 + c;
                out[at] = (uint8_t)v;
            }
    }
    free(weights);
    return out;
}
static void append_soft(ei_media_input *m, const float *soft, size_t n, int start, int end) {
    token(m, start);
    rows(m, soft, n);
    token(m, end);
}
/* Media cache: tower outputs keyed by kind, the parameter that fixes the
 * output (image token budget, audio samples used) and a 128-bit hash of the
 * base64 payload. Values are [u64 rows][rows x 512 floats]. */
static uint64_t mix64(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}
static uint64_t hash64(const char *data, size_t len, uint64_t seed) {
    uint64_t h = mix64(seed ^ 0x9E3779B97F4A7C15ULL ^ (uint64_t)len);
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        uint64_t w;
        memcpy(&w, data + i, 8);
        h = mix64(h ^ w) + 0x632BE59BD9B4E019ULL;
    }
    uint64_t tail = 0;
    if (i < len)
        memcpy(&tail, data + i, len - i);
    return mix64(h ^ tail ^ ((uint64_t)(len - i) << 56));
}
static void media_key(unsigned char key[24], char kind, uint32_t param, const char *data,
                      size_t len) {
    uint64_t a = hash64(data, len, 1), b = hash64(data, len, 2);
    memset(key, 0, 24);
    key[0] = (unsigned char)kind;
    memcpy(key + 4, &param, 4);
    memcpy(key + 8, &a, 8);
    memcpy(key + 16, &b, 8);
}
/* On a hit *soft is a malloc'd copy of the rows. */
static bool cache_soft_lookup(const ei_media_input *m, const unsigned char key[24], float **soft,
                              size_t *n) {
    ei_response_cache_value v;
    if (!m->cache || !ei_response_cache_acquire(m->cache, (const char *)key, 24, &v))
        return false;
    uint64_t rows_n = 0;
    memcpy(&rows_n, v.data, 8);
    *n = (size_t)rows_n;
    *soft = ei_xmalloc(*n * 512 * sizeof(float) + sizeof(float));
    memcpy(*soft, v.data + 8, *n * 512 * sizeof(float));
    ei_response_cache_release(m->cache, &v);
    return true;
}
static void cache_soft_store(const ei_media_input *m, const unsigned char key[24],
                             const float *soft, size_t n) {
    if (!m->cache || !soft)
        return;
    size_t bytes = 8 + n * 512 * sizeof(float);
    char *blob = ei_xmalloc(bytes);
    uint64_t rows_n = n;
    memcpy(blob, &rows_n, 8);
    memcpy(blob + 8, soft, n * 512 * sizeof(float));
    ei_response_cache_insert(m->cache, (const char *)key, 24, blob, bytes);
    free(blob);
}
bool ei_media_image(ei_media_input *m, const char *base64, size_t len, int budget, char *err,
                    size_t err_len) {
    size_t size = 0;
    double t0 = media_timing() ? media_now_ms() : 0;
    /* The output is fixed by the payload and the budget it will actually get. */
    unsigned char key[24];
    bool keyed = m->cache != NULL;
    if (keyed) {
        int eff = budget;
        if ((size_t)eff + 2 > avail(m))
            eff = avail(m) >= 72 ? (int)avail(m) - 2 : 0;
        media_key(key, 'I', (uint32_t)eff, base64, len);
        float *soft = NULL;
        size_t n = 0;
        if (eff >= 70 && cache_soft_lookup(m, key, &soft, &n)) {
            if (n + 2 <= avail(m))
                append_soft(m, soft, n, 255999, 258882);
            free(soft);
            MEDIA_STAGE("cached", t0);
            return true;
        }
    }
    uint8_t *bytes = decode(base64, len, &size);
    MEDIA_STAGE("base64", t0);
    if (!bytes)
        return fail(err, err_len, "invalid image base64");
    int w, h, c;
    if (!stbi_info_from_memory(bytes, (int)size, &w, &h, &c) || w < 1 || h < 1 ||
        (size_t)w * h > 16 * 1024 * 1024) {
        free(bytes);
        return fail(err, err_len, "unsupported image or image exceeds 16 megapixels");
    }
    /* Shrink the token budget to the room left; drop the image below the minimum. */
    if ((size_t)budget + 2 > avail(m))
        budget = avail(m) >= 72 ? (int)avail(m) - 2 : 0;
    if (budget < 70) {
        free(bytes);
        return true;
    }
    uint8_t *rgb = stbi_load_from_memory(bytes, (int)size, &w, &h, &c, 3);
    free(bytes);
    MEDIA_STAGE("decode", t0);
    if (!rgb)
        return fail(err, err_len, "cannot decode image (expected PNG, JPEG, BMP, or PNM)");
    int tw = (int)fmax(48, round(w / 48.0) * 48), th = (int)fmax(48, round(h / 48.0) * 48);
    double area = (double)tw * th, max = (double)budget * 48 * 48, min = 70.0 * 48 * 48;
    if (area > max) {
        double scale = sqrt(max / ((double)w * h));
        tw = (int)fmax(48, floor(w * scale / 48) * 48);
        th = (int)fmax(48, floor(h * scale / 48) * 48);
    } else if (area < min) {
        double scale = sqrt(min / ((double)w * h));
        tw = (int)ceil(w * scale / 48) * 48;
        th = (int)ceil(h * scale / 48) * 48;
    }
    if ((size_t)tw * th / 2304 + 2 > avail(m)) {
        stbi_image_free(rgb); /* extreme aspect ratio: even the 48-pixel floor overflows */
        return true;
    }
    /* Preserve aspect ratio within the aligned canvas, center-pad with black. */
    double scale = fmin((double)tw / w, (double)th / h);
    int rw = (int)ceil(w * scale), rh = (int)ceil(h * scale);
    if (rw > tw)
        rw = tw;
    if (rh > th)
        rh = th;
    uint8_t *horizontal = resize_axis(rgb, w, h, rw, false);
    stbi_image_free(rgb);
    uint8_t *resized = resize_axis(horizontal, rw, h, rh, true);
    free(horizontal);
    uint8_t *canvas = ei_xcalloc((size_t)tw * th, 3);
    int ox = (tw - rw) / 2, oy = (th - rh) / 2;
    for (int y = 0; y < rh; ++y)
        memcpy(canvas + ((size_t)(y + oy) * tw + ox) * 3, resized + (size_t)y * rw * 3,
               (size_t)rw * 3);
    free(resized);
    MEDIA_STAGE("resize", t0);
    float *soft = NULL;
    size_t n = 0;
    bool ok;
    if (m->lock)
        pthread_mutex_lock(m->lock);
#if defined(EI_ENABLE_METAL)
    if (m->metal) ok = ei_gemma2_metal_vision(m->metal, m->projector, canvas, tw, th, &soft, &n, err, err_len);
    else
#endif
#if defined(EI_ENABLE_CUDA)
    if (m->cuda) ok = ei_gemma2_cuda_vision(m->cuda, m->projector, canvas, tw, th, &soft, &n, err, err_len);
    else
#endif
    ok = ei_gemma2_vision(m->projector, m->pool, canvas, tw, th, &soft, &n, err, err_len);
    if (m->lock)
        pthread_mutex_unlock(m->lock);
    MEDIA_STAGE("vision", t0);
    free(canvas);
    if (ok && keyed)
        cache_soft_store(m, key, soft, n);
    if (ok && n + 2 <= avail(m))
        append_soft(m, soft, n, 255999, 258882);
    free(soft);
    return ok;
}
/* Decode and resize for `budget` soft tokens (no room check: the stream
 * position is only known at commit). */
void ei_media_image_prepare(const ei_media_input *m, const char *base64, size_t len, int budget,
                            ei_media_frame *frame) {
    *frame = (ei_media_frame){.budget = budget, .ok = true};
    if (m->cache && budget >= 70) {
        frame->keyed = true;
        media_key(frame->key, 'I', (uint32_t)budget, base64, len);
        if (cache_soft_lookup(m, frame->key, &frame->soft, &frame->tokens))
            return;
    }
    size_t size = 0;
    uint8_t *bytes = decode(base64, len, &size);
    if (!bytes) {
        frame->ok = false;
        snprintf(frame->err, sizeof frame->err, "invalid image base64");
        return;
    }
    int w, h, c;
    if (!stbi_info_from_memory(bytes, (int)size, &w, &h, &c) || w < 1 || h < 1 ||
        (size_t)w * h > 16 * 1024 * 1024) {
        free(bytes);
        frame->ok = false;
        snprintf(frame->err, sizeof frame->err, "unsupported image or image exceeds 16 megapixels");
        return;
    }
    if (budget < 70) {
        free(bytes);
        return;
    }
    uint8_t *rgb = stbi_load_from_memory(bytes, (int)size, &w, &h, &c, 3);
    free(bytes);
    if (!rgb) {
        frame->ok = false;
        snprintf(frame->err, sizeof frame->err,
                 "cannot decode image (expected PNG, JPEG, BMP, or PNM)");
        return;
    }
    int tw = (int)fmax(48, round(w / 48.0) * 48), th = (int)fmax(48, round(h / 48.0) * 48);
    double area = (double)tw * th, max = (double)budget * 48 * 48, min = 70.0 * 48 * 48;
    if (area > max) {
        double scale = sqrt(max / ((double)w * h));
        tw = (int)fmax(48, floor(w * scale / 48) * 48);
        th = (int)fmax(48, floor(h * scale / 48) * 48);
    } else if (area < min) {
        double scale = sqrt(min / ((double)w * h));
        tw = (int)ceil(w * scale / 48) * 48;
        th = (int)ceil(h * scale / 48) * 48;
    }
    /* Preserve aspect ratio within the aligned canvas, center-pad with black. */
    double scale = fmin((double)tw / w, (double)th / h);
    int rw = (int)ceil(w * scale), rh = (int)ceil(h * scale);
    if (rw > tw)
        rw = tw;
    if (rh > th)
        rh = th;
    uint8_t *horizontal = resize_axis(rgb, w, h, rw, false);
    stbi_image_free(rgb);
    uint8_t *resized = resize_axis(horizontal, rw, h, rh, true);
    free(horizontal);
    uint8_t *canvas = ei_xcalloc((size_t)tw * th, 3);
    int ox = (tw - rw) / 2, oy = (th - rh) / 2;
    for (int y = 0; y < rh; ++y)
        memcpy(canvas + ((size_t)(y + oy) * tw + ox) * 3, resized + (size_t)y * rw * 3,
               (size_t)rw * 3);
    free(resized);
    frame->canvas = canvas;
    frame->width = tw;
    frame->height = th;
}
void ei_media_frame_free(ei_media_frame *frame) {
    free(frame->canvas);
    free(frame->soft);
    frame->canvas = NULL;
    frame->soft = NULL;
}
void ei_media_frames_encode(ei_media_input *m, ei_media_frame *frames, size_t n) {
#if defined(EI_ENABLE_METAL)
    const char *flag = getenv("EI_GEMMA2_VIDEO_BATCH");
    if (!m->metal || (flag && !strcmp(flag, "0")))
        return;
    for (size_t i = 0; i < n;) {
        ei_media_frame *f = &frames[i];
        if (!f->ok || !f->canvas) {
            i++;
            continue;
        }
        /* The run of same-sized frames that fits one wave. */
        size_t per = (size_t)f->width * f->height / 256, count = 1;
        while (i + count < n && frames[i + count].ok && frames[i + count].canvas &&
               frames[i + count].width == f->width && frames[i + count].height == f->height &&
               per * (count + 1) <= EI_GEMMA2_VISION_WAVE_TOKENS)
            count++;
        const uint8_t *rgb[64];
        size_t batch = count < 64 ? count : 64;
        for (size_t j = 0; j < batch; j++)
            rgb[j] = frames[i + j].canvas;
        float *soft = NULL;
        size_t total = 0;
        char err[160];
        if (m->lock)
            pthread_mutex_lock(m->lock);
        bool ok = ei_gemma2_metal_vision_frames(m->metal, m->projector, rgb, (int)batch, f->width,
                                                f->height, &soft, &total, err, sizeof err);
        if (m->lock)
            pthread_mutex_unlock(m->lock);
        if (ok) {
            size_t each = total / batch;
            for (size_t j = 0; j < batch; j++) {
                frames[i + j].soft = ei_xmalloc(each * 512 * sizeof *soft);
                memcpy(frames[i + j].soft, soft + j * each * 512, each * 512 * sizeof *soft);
                frames[i + j].tokens = each;
                if (frames[i + j].keyed)
                    cache_soft_store(m, frames[i + j].key, frames[i + j].soft, each);
            }
        } else
            for (size_t j = 0; j < batch; j++) {
                frames[i + j].ok = false;
                snprintf(frames[i + j].err, sizeof frames[i + j].err, "%s", err);
            }
        free(soft);
        i += batch;
    }
#else
    (void)m;
    (void)frames;
    (void)n;
#endif
}
/* Run the tower on a prepared canvas (or take the rows ei_media_frames_encode
 * produced) and append its soft tokens. When the stream no longer has room for
 * the prepared budget, the image is re-prepared at the room left (the
 * sequential path's exact behaviour). */
bool ei_media_image_commit(ei_media_input *m, ei_media_frame *frame, const char *base64,
                           size_t len, char *err, size_t err_len) {
    bool ok = frame->ok;
    if (!ok)
        snprintf(err, err_len, "%s", frame->err);
    uint8_t *canvas = frame->canvas;
    float *soft = frame->soft;
    frame->canvas = NULL;
    frame->soft = NULL;
    if (!ok || (!canvas && !soft)) {
        free(canvas);
        free(soft);
        return ok;
    }
    size_t need = soft ? frame->tokens : (size_t)frame->width * frame->height / 2304;
    if ((size_t)frame->budget + 2 > avail(m) || need + 2 > avail(m)) {
        free(canvas);
        free(soft);
        return ei_media_image(m, base64, len, frame->budget, err, err_len);
    }
    if (soft) {
        free(canvas);
        append_soft(m, soft, frame->tokens, 255999, 258882);
        free(soft);
        return true;
    }
    size_t n = 0;
    if (m->lock)
        pthread_mutex_lock(m->lock);
#if defined(EI_ENABLE_METAL)
    if (m->metal) ok = ei_gemma2_metal_vision(m->metal, m->projector, canvas, frame->width, frame->height, &soft, &n, err, err_len);
    else
#endif
#if defined(EI_ENABLE_CUDA)
    if (m->cuda) ok = ei_gemma2_cuda_vision(m->cuda, m->projector, canvas, frame->width, frame->height, &soft, &n, err, err_len);
    else
#endif
    ok = ei_gemma2_vision(m->projector, m->pool, canvas, frame->width, frame->height, &soft, &n, err, err_len);
    if (m->lock)
        pthread_mutex_unlock(m->lock);
    free(canvas);
    if (ok && frame->keyed)
        cache_soft_store(m, frame->key, soft, n);
    if (ok && n + 2 <= avail(m))
        append_soft(m, soft, n, 255999, 258882);
    free(soft);
    return ok;
}
static uint32_t u32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static uint16_t u16(const uint8_t *p) {
    return (uint16_t)p[0] | (uint16_t)p[1] << 8;
}
/* Radix-2 FFT, unnormalized, for the model's 512-point STFT. Twiddles are
 * tabulated once with the same expression the butterflies used to evaluate
 * per call (2,304 cosf/sinf pairs per frame), so results are unchanged. */
static float fft_cos[512], fft_sin[512];
static pthread_once_t fft_once = PTHREAD_ONCE_INIT;
static void fft_tables(void) {
    for (int width = 2; width <= 512; width *= 2)
        for (int j = 0; j < width / 2; ++j) {
            float theta = -6.283185307179586f * j / width;
            fft_cos[width / 2 + j] = cosf(theta);
            fft_sin[width / 2 + j] = sinf(theta);
        }
}
static void fft(float re[512], float im[512]) {
    pthread_once(&fft_once, fft_tables);
    for (int i = 1, j = 0; i < 512; ++i) {
        int bit = 256;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            float a = re[i];
            re[i] = re[j];
            re[j] = a;
        }
    }
    for (int width = 2; width <= 512; width *= 2)
        for (int start = 0; start < 512; start += width)
            for (int j = 0; j < width / 2; ++j) {
                float co = fft_cos[width / 2 + j], si = fft_sin[width / 2 + j];
                int a = start + j, b = a + width / 2;
                float r = co * re[b] - si * im[b], v = si * re[b] + co * im[b];
                re[b] = re[a] - r;
                im[b] = im[a] - v;
                re[a] += r;
                im[a] += v;
            }
}
bool ei_media_audio(ei_media_input *m, const char *base64, size_t len, char *err, size_t err_len) {
    size_t size = 0;
    double t0 = media_timing() ? media_now_ms() : 0;
    uint8_t *bytes = decode(base64, len, &size);
    MEDIA_STAGE("base64", t0);
    if (!bytes || size < 12 || memcmp(bytes, "RIFF", 4) || memcmp(bytes + 8, "WAVE", 4)) {
        free(bytes);
        return fail(err, err_len, "audio must be a base64 WAV file");
    }
    int format = 0, channels = 0, rate = 0, bits = 0;
    const uint8_t *data = NULL;
    size_t data_len = 0;
    for (size_t at = 12; at + 8 <= size;) {
        size_t n = u32(bytes + at + 4);
        if (n > size - at - 8) {
            free(bytes);
            return fail(err, err_len, "truncated WAV chunk");
        }
        if (!memcmp(bytes + at, "fmt ", 4) && n >= 16) {
            format = u16(bytes + at + 8);
            channels = u16(bytes + at + 10);
            rate = (int)u32(bytes + at + 12);
            bits = u16(bytes + at + 22);
        }
        if (!memcmp(bytes + at, "data", 4)) {
            data = bytes + at + 8;
            data_len = n;
        }
        at += 8 + n + (n & 1);
    }
    if (!data || rate != 16000 || channels != 1 ||
        !((format == 1 && bits == 16) || (format == 3 && bits == 32)) || data_len % (bits / 8)) {
        free(bytes);
        return fail(err, err_len, "WAV must be mono 16000 Hz PCM16 or float32");
    }
    size_t samples = data_len / (bits / 8);
    if (samples < 321) {
        free(bytes);
        return fail(err, err_len, "audio must be at least 21 ms long");
    }
    /* Trim the clip to the room left (25 tokens per second); drop it below one token. */
    size_t room = avail(m) > 2 ? avail(m) - 2 : 0;
    if (samples > room * 640)
        samples = room * 640;
    if (samples < 321) {
        free(bytes);
        return true;
    }
    /* The rows are fixed by the payload and the samples used after the trim. */
    unsigned char key[24];
    bool keyed = m->cache != NULL;
    if (keyed) {
        media_key(key, 'A', (uint32_t)samples, base64, len);
        float *soft = NULL;
        size_t n = 0;
        if (cache_soft_lookup(m, key, &soft, &n)) {
            free(bytes);
            token(m, 256000);
            size_t left = avail(m) > 1 ? avail(m) - 1 : 0;
            rows(m, soft, n < left ? n : left);
            free(soft);
            token(m, 258883);
            MEDIA_STAGE("cached", t0);
            return true;
        }
    }
    float *all = NULL;
    size_t all_n = 0;
    float *pcm = ei_xmalloc(samples * sizeof(float));
    for (size_t i = 0; i < samples; ++i) {
        if (bits == 16)
            pcm[i] = (int16_t)u16(data + i * 2) / 32768.0f;
        else
            memcpy(pcm + i, data + i * 4, 4);
        if (!isfinite(pcm[i]) || fabsf(pcm[i]) > 1.0f) {
            free(pcm);
            free(bytes);
            return fail(err, err_len, "audio samples must be finite and in [-1,1]");
        }
    }
    free(bytes);
    float filters[128][257], edges[130];
    int first[128], last[128];
    float maxmel = 2595.0f * log10f(1 + 8000.0f / 700);
    for (int i = 0; i < 130; ++i)
        edges[i] = 700 * (powf(10, maxmel * i / 129 / 2595) - 1);
    for (int bin = 0; bin < 128; ++bin) {
        first[bin] = 257;
        last[bin] = -1;
        for (int f = 0; f < 257; ++f) {
            float hz = f * 31.25f;
            filters[bin][f] =
                fmaxf(0, fminf((hz - edges[bin]) / (edges[bin + 1] - edges[bin]),
                               (edges[bin + 2] - hz) / (edges[bin + 2] - edges[bin + 1])));
            if (filters[bin][f] != 0) {
                if (f < first[bin]) first[bin] = f;
                last[bin] = f;
            }
        }
    }
    token(m, 256000);
    bool ok = true;
    for (size_t offset = 0; ok && offset < samples; offset += 480000) {
        size_t count = samples - offset;
        if (count > 480000)
            count = 480000;
        int frames = ((int)count - 161) / 160 + 1;
        /* A sub-frame tail still carries signal; pad it to one analysis frame. */
        if (frames < 1) frames = 1;
        MEDIA_STAGE("wav", t0);
        float *mel = ei_xcalloc((size_t)frames * 128, sizeof(float));
        for (int t = 0; t < frames; ++t) {
            float re[512] = {0}, im[512] = {0}, mag[257];
            for (int j = 0; j < 320; ++j) {
                int pos = t * 160 + j - 160;
                if (pos >= 0 && (size_t)pos < count)
                    re[j] = pcm[offset + pos] * (0.5f - 0.5f * cosf(6.283185307179586f * j / 320));
            }
            fft(re, im);
            for (int f = 0; f < 257; ++f)
                mag[f] = sqrtf(re[f] * re[f] + im[f] * im[f]);
            for (int bin = 0; bin < 128; ++bin) {
                double sum = 0;
                for (int f = first[bin]; f <= last[bin]; ++f)
                    sum += mag[f] * filters[bin][f];
                mel[(size_t)t * 128 + bin] = logf(fmaxf(0.001f, (float)sum));
            }
        }
        MEDIA_STAGE("logmel", t0);
        float *soft = NULL;
        size_t n = 0;
        if (m->audio)
            ok = m->audio(m->audio_ctx, mel, frames, &soft, &n, err, err_len);
        else {
            if (m->lock)
                pthread_mutex_lock(m->lock);
#if defined(EI_ENABLE_METAL)
            if (m->metal) ok = ei_gemma2_metal_audio(m->metal, m->projector, mel, frames, &soft, &n, err, err_len);
            else
#endif
#if defined(EI_ENABLE_CUDA)
            if (m->cuda) ok = ei_gemma2_cuda_audio(m->cuda, m->projector, mel, frames, &soft, &n, err, err_len);
            else
#endif
            ok = ei_gemma2_audio(m->projector, m->pool, mel, frames, &soft, &n, err, err_len);
            if (m->lock)
                pthread_mutex_unlock(m->lock);
        }
        MEDIA_STAGE("audio", t0);
        free(mel);
        if (ok) {
            /* Per-chunk rounding can overshoot the estimate; keep the end marker's slot. */
            size_t left = avail(m) > 1 ? avail(m) - 1 : 0;
            rows(m, soft, n < left ? n : left);
            if (keyed) {
                all = ei_xrealloc(all, (all_n + n) * 512 * sizeof(float));
                memcpy(all + all_n * 512, soft, n * 512 * sizeof(float));
                all_n += n;
            }
        }
        free(soft);
    }
    free(pcm);
    if (ok && keyed)
        cache_soft_store(m, key, all, all_n);
    free(all);
    if (ok)
        token(m, 258883);
    return ok;
}
static bool finish_text(ei_media_input *m, float out[768], char *err, size_t err_len);
bool ei_media_finish(ei_media_input *m, float out[768], char *err, size_t err_len) {
    if (m->tokenizer->model->add_eos)
        token(m, m->tokenizer->model->eos_id);
    double t0 = media_timing() ? media_now_ms() : 0;
    bool ok;
    if (m->finish)
        ok = m->finish(m->finish_ctx, m->inputs, m->n, out, err, err_len);
    else {
        if (m->lock)
            pthread_mutex_lock(m->lock);
        ok = finish_text(m, out, err, err_len);
        if (m->lock)
            pthread_mutex_unlock(m->lock);
    }
    if (media_timing())
        fprintf(stderr, "media finish     %7.2f ms (%zu rows)\n", media_now_ms() - t0, m->n);
    return ok;
}
static bool finish_text(ei_media_input *m, float out[768], char *err, size_t err_len) {
#if defined(EI_ENABLE_CUDA)
    if (m->cuda)
        return ei_gemma2_cuda_embed_inputs(m->cuda, m->inputs, m->n, out, err, err_len);
#endif
#if defined(EI_ENABLE_METAL)
    if (m->metal) return ei_gemma2_metal_embed(m->metal, NULL, m->inputs, m->n, out, err, err_len);
#endif
    if (ei_gemma2_cpu_fast()) {
        size_t offsets[2] = {0, m->n};
        return ei_gemma2_embed_batch_fast(m->text_model, m->pool, NULL, m->inputs, offsets, 1, out,
                                          err, err_len);
    }
    return ei_gemma2_embed(m->text_model, m->pool, NULL, m->inputs, m->n, out, err, err_len);
}
