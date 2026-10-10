#if defined(EI_ENABLE_METAL)
#include "gemma2_metal.h"
#endif
#if defined(EI_ENABLE_CUDA)
#include "gemma2_cuda.h"
#endif
#include "media.h"
#include <math.h>
#include <limits.h>
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_PNM
#define STBI_MAX_DIMENSIONS 8192
#include "vendor/stb_image.h"

#define MEDIA_CONTEXT 8192
static bool fail(char *err, size_t n, const char *message) {
    snprintf(err, n, "%s", message);
    return false;
}
static bool room(ei_media_input *m, size_t n, char *err, size_t len) {
    if (n > MEDIA_CONTEXT - m->n)
        return fail(err, len, "combined input exceeds 8192 tokens");
    m->inputs = ei_xrealloc(m->inputs, (m->n + n) * 512 * sizeof(float));
    return true;
}
static bool token(ei_media_input *m, int32_t id, char *err, size_t len) {
    if (!room(m, 1, err, len))
        return false;
    const ei_tensor *w = ei_gguf_tensor(m->text_model, "token_embd.weight", true);
    ei_dequantize_row_q8_0(w, id, m->inputs + m->n * 512);
    for (int z = 0; z < 512; ++z) m->inputs[m->n * 512 + z] *= sqrtf(512.0f);
    m->n++;
    return true;
}
void ei_media_input_init(ei_media_input *m, const ei_gguf *text, const ei_tokenizer *tok,
                         const ei_gguf *projector, ei_thread_pool *pool) {
    *m = (ei_media_input){
        .text_model = text, .tokenizer = tok, .projector = projector, .pool = pool};
    char err[80];
    if (tok->model->add_bos)
        token(m, tok->model->bos_id, err, sizeof err);
}
void ei_media_input_free(ei_media_input *m) {
    free(m->inputs);
    m->inputs = NULL;
    m->n = 0;
}
bool ei_media_text(ei_media_input *m, const char *text, size_t len, char *err, size_t err_len) {
    ei_tokens t;
    ei_tokenize_spm(m->tokenizer, text, len, false, false, &t);
    bool ok = true;
    for (size_t i = 0; i < t.n && ok; ++i)
        ok = token(m, t.ids[i], err, err_len);
    ei_tokens_free(&t);
    return ok;
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
    for (size_t i = 0; i < len; i += 4) {
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
static bool append_soft(ei_media_input *m, float *soft, size_t n, int start, int end, char *err,
                        size_t len) {
    if (!room(m, n + 2, err, len))
        return false;
    if (!token(m, start, err, len))
        return false;
    if (!room(m, n, err, len))
        return false;
    memcpy(m->inputs + m->n * 512, soft, n * 512 * sizeof(float));
    m->n += n;
    return token(m, end, err, len);
}
bool ei_media_image(ei_media_input *m, const char *base64, size_t len, int budget, char *err,
                    size_t err_len) {
    size_t size = 0;
    uint8_t *bytes = decode(base64, len, &size);
    if (!bytes)
        return fail(err, err_len, "invalid image base64");
    int w, h, c;
    if (!stbi_info_from_memory(bytes, (int)size, &w, &h, &c) || w < 1 || h < 1 ||
        (size_t)w * h > 16 * 1024 * 1024) {
        free(bytes);
        return fail(err, err_len, "unsupported image or image exceeds 16 megapixels");
    }
    uint8_t *rgb = stbi_load_from_memory(bytes, (int)size, &w, &h, &c, 3);
    free(bytes);
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
    if ((size_t)tw * th / 2304 + 2 > MEDIA_CONTEXT - m->n) {
        stbi_image_free(rgb);
        return fail(err, err_len, "image exceeds remaining token budget");
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
    free(canvas);
    if (ok)
        ok = append_soft(m, soft, n, 255999, 258882, err, err_len);
    free(soft);
    return ok;
}
static uint32_t u32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static uint16_t u16(const uint8_t *p) {
    return (uint16_t)p[0] | (uint16_t)p[1] << 8;
}
/* Radix-2 FFT, unnormalized, for the model's 512-point STFT. */
static void fft(float re[512], float im[512]) {
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
                float theta = -6.283185307179586f * j / width, co = cosf(theta), si = sinf(theta);
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
    uint8_t *bytes = decode(base64, len, &size);
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
    if (samples < 321 || samples > 16000 * 300) {
        free(bytes);
        return fail(err, err_len, "audio duration must be 21 ms through 300 seconds");
    }
    if ((samples + 639) / 640 + 2 > MEDIA_CONTEXT - m->n) {
        free(bytes);
        return fail(err, err_len, "audio exceeds remaining token budget");
    }
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
    float maxmel = 2595.0f * log10f(1 + 8000.0f / 700);
    for (int i = 0; i < 130; ++i)
        edges[i] = 700 * (powf(10, maxmel * i / 129 / 2595) - 1);
    for (int bin = 0; bin < 128; ++bin)
        for (int f = 0; f < 257; ++f) {
            float hz = f * 31.25f;
            filters[bin][f] =
                fmaxf(0, fminf((hz - edges[bin]) / (edges[bin + 1] - edges[bin]),
                               (edges[bin + 2] - hz) / (edges[bin + 2] - edges[bin + 1])));
        }
    bool ok = token(m, 256000, err, err_len);
    for (size_t offset = 0; ok && offset < samples; offset += 480000) {
        size_t count = samples - offset;
        if (count > 480000)
            count = 480000;
        int frames = ((int)count - 161) / 160 + 1;
        /* A sub-frame tail still carries signal; pad it to one analysis frame. */
        if (frames < 1) frames = 1;
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
                for (int f = 0; f < 257; ++f)
                    sum += mag[f] * filters[bin][f];
                mel[(size_t)t * 128 + bin] = logf(fmaxf(0.001f, (float)sum));
            }
        }
        float *soft = NULL;
        size_t n = 0;
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
        free(mel);
        if (ok)
            ok = room(m, n, err, err_len);
        if (ok) {
            memcpy(m->inputs + m->n * 512, soft, n * 512 * sizeof(float));
            m->n += n;
        }
        free(soft);
    }
    free(pcm);
    if (ok)
        ok = token(m, 258883, err, err_len);
    return ok;
}
static bool finish_text(ei_media_input *m, float out[768], char *err, size_t err_len);
bool ei_media_finish(ei_media_input *m, float out[768], char *err, size_t err_len) {
    if (m->tokenizer->model->add_eos && !token(m, m->tokenizer->model->eos_id, err, err_len))
        return false;
    if (m->lock)
        pthread_mutex_lock(m->lock);
    bool ok = finish_text(m, out, err, err_len);
    if (m->lock)
        pthread_mutex_unlock(m->lock);
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
