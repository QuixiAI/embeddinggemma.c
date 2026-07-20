#include "engine.h"
#include "inference_service.h"
#include "response_cache.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <curl/curl.h>
#include <errno.h>
#include <limits.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#define MODEL_URL "https://huggingface.co/ggml-org/embeddinggemma-300M-qat-q4_0-GGUF/resolve/main/embeddinggemma-300M-qat-Q4_0.gguf"
#define DEFAULT_MODEL_NAME "embeddinggemma-300m"
#define MAX_BODY_BYTES (16u * 1024u * 1024u)
#define MAX_HEADER_BYTES (64u * 1024u)

#if defined(EI_ENABLE_CUDA)
#define DEFAULT_INFERENCE_BACKEND "cuda"
#elif defined(EI_ENABLE_METAL)
#define DEFAULT_INFERENCE_BACKEND "metal"
#else
#define DEFAULT_INFERENCE_BACKEND "cpu"
#endif

typedef struct {
    char  *s;
    size_t len;
} sv_string;

typedef struct {
    sv_string *items;
    size_t n;
} string_list;

typedef enum {
    EMBEDDING_ENCODING_FLOAT,
    EMBEDDING_ENCODING_BASE64,
} embedding_encoding;

typedef struct {
    char *data;
    size_t n;
    size_t cap;
} sbuf;

typedef struct {
    const char *bind_host;
    int port;
    const char *backend;
    const char *model_path;
    size_t workers;
    size_t max_queue;
    size_t cache_entries;
    size_t max_batch_tokens;
    size_t max_batch_requests;
    size_t max_batch_sequence_tokens;
    size_t max_client_batch_size;
    size_t tokenizer_workers;
    uint32_t batch_wait_us;
    size_t keepalive_connections;
    size_t keepalive_max_requests;
    uint32_t keepalive_timeout_ms;
    size_t response_cache_bytes;
} server_opts;

static void sbuf_reserve(sbuf *b, size_t additional) {
    size_t required = b->n + additional + 1u;
    if (required <= b->cap) return;
    size_t capacity = b->cap ? b->cap : 4096u;
    while (capacity < required) capacity *= 2u;
    b->data = ei_xrealloc(b->data, capacity);
    b->cap = capacity;
}

static void sbuf_append(sbuf *b, const char *s, size_t n) {
    sbuf_reserve(b, n);
    memcpy(b->data + b->n, s, n);
    b->n += n;
    b->data[b->n] = '\0';
}

static void sbuf_append_z(sbuf *b, const char *s) {
    sbuf_append(b, s, strlen(s));
}

static void sbuf_append_json_string(sbuf *b, const char *s, size_t n) {
    sbuf_append_z(b, "\"");
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        char tmp[8];
        switch (c) {
            case '"': sbuf_append_z(b, "\\\""); break;
            case '\\': sbuf_append_z(b, "\\\\"); break;
            case '\b': sbuf_append_z(b, "\\b"); break;
            case '\f': sbuf_append_z(b, "\\f"); break;
            case '\n': sbuf_append_z(b, "\\n"); break;
            case '\r': sbuf_append_z(b, "\\r"); break;
            case '\t': sbuf_append_z(b, "\\t"); break;
            default:
                if (c < 0x20) {
                    snprintf(tmp, sizeof tmp, "\\u%04x", c);
                    sbuf_append_z(b, tmp);
                } else {
                    sbuf_append(b, (const char *)&c, 1);
                }
        }
    }
    sbuf_append_z(b, "\"");
}

static void sbuf_append_base64_f32(sbuf *b, const float *values, size_t count) {
    static const char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    uint8_t raw[EI_N_EMBD * sizeof(float)];
    for (size_t i = 0; i < count; i++) {
        uint32_t bits;
        memcpy(&bits, values + i, sizeof bits);
        raw[i * 4u] = (uint8_t)bits;
        raw[i * 4u + 1u] = (uint8_t)(bits >> 8);
        raw[i * 4u + 2u] = (uint8_t)(bits >> 16);
        raw[i * 4u + 3u] = (uint8_t)(bits >> 24);
    }

    const size_t raw_len = count * sizeof(float);
    const size_t encoded_len = 4u * ((raw_len + 2u) / 3u);
    sbuf_reserve(b, encoded_len);
    char *dst = b->data + b->n;
    size_t out = 0;
    for (size_t i = 0; i < raw_len; i += 3u) {
        const uint32_t a = raw[i];
        const uint32_t c1 = i + 1u < raw_len ? raw[i + 1u] : 0u;
        const uint32_t c2 = i + 2u < raw_len ? raw[i + 2u] : 0u;
        dst[out++] = alphabet[a >> 2];
        dst[out++] = alphabet[((a & 3u) << 4) | (c1 >> 4)];
        dst[out++] = i + 1u < raw_len
            ? alphabet[((c1 & 15u) << 2) | (c2 >> 6)] : '=';
        dst[out++] = i + 2u < raw_len ? alphabet[c2 & 63u] : '=';
    }
    b->n += encoded_len;
    b->data[b->n] = '\0';
}

static void skip_ws(const char **p) {
    while (isspace((unsigned char)**p)) (*p)++;
}

static uint32_t hex4(const char *p, char *err, size_t err_len) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
        else {
            snprintf(err, err_len, "invalid unicode escape in JSON string");
            return UINT32_MAX;
        }
    }
    return v;
}

static void append_byte(char **buf, size_t *n, size_t *cap, uint8_t c) {
    if (*n + 2u > *cap) {
        *cap = *cap ? *cap * 2u : 64u;
        *buf = ei_xrealloc(*buf, *cap);
    }
    (*buf)[(*n)++] = (char)c;
    (*buf)[*n] = '\0';
}

static void append_utf8(char **buf, size_t *n, size_t *cap, uint32_t cp) {
    if (cp <= 0x7Fu) {
        append_byte(buf, n, cap, (uint8_t)cp);
    } else if (cp <= 0x7FFu) {
        append_byte(buf, n, cap, (uint8_t)(0xC0u | (cp >> 6)));
        append_byte(buf, n, cap, (uint8_t)(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        append_byte(buf, n, cap, (uint8_t)(0xE0u | (cp >> 12)));
        append_byte(buf, n, cap, (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu)));
        append_byte(buf, n, cap, (uint8_t)(0x80u | (cp & 0x3Fu)));
    } else {
        append_byte(buf, n, cap, (uint8_t)(0xF0u | (cp >> 18)));
        append_byte(buf, n, cap, (uint8_t)(0x80u | ((cp >> 12) & 0x3Fu)));
        append_byte(buf, n, cap, (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu)));
        append_byte(buf, n, cap, (uint8_t)(0x80u | (cp & 0x3Fu)));
    }
}

static bool parse_json_string(const char **p, sv_string *out, char *err, size_t err_len) {
    skip_ws(p);
    if (**p != '"') {
        snprintf(err, err_len, "expected JSON string");
        return false;
    }
    (*p)++;

    char *buf = NULL;
    size_t n = 0;
    size_t cap = 0;
    while (**p && **p != '"') {
        unsigned char c = (unsigned char)**p;
        if (c != '\\') {
            append_byte(&buf, &n, &cap, c);
            (*p)++;
            continue;
        }

        (*p)++;
        char esc = **p;
        if (!esc) {
            free(buf);
            snprintf(err, err_len, "unfinished JSON string escape");
            return false;
        }
        (*p)++;
        switch (esc) {
            case '"': append_byte(&buf, &n, &cap, '"'); break;
            case '\\': append_byte(&buf, &n, &cap, '\\'); break;
            case '/': append_byte(&buf, &n, &cap, '/'); break;
            case 'b': append_byte(&buf, &n, &cap, '\b'); break;
            case 'f': append_byte(&buf, &n, &cap, '\f'); break;
            case 'n': append_byte(&buf, &n, &cap, '\n'); break;
            case 'r': append_byte(&buf, &n, &cap, '\r'); break;
            case 't': append_byte(&buf, &n, &cap, '\t'); break;
            case 'u': {
                uint32_t cp = hex4(*p, err, err_len);
                if (cp == UINT32_MAX) {
                    free(buf);
                    return false;
                }
                *p += 4;
                if (0xD800u <= cp && cp <= 0xDBFFu) {
                    if ((*p)[0] != '\\' || (*p)[1] != 'u') {
                        free(buf);
                        snprintf(err, err_len, "missing low surrogate in JSON string");
                        return false;
                    }
                    *p += 2;
                    uint32_t lo = hex4(*p, err, err_len);
                    if (lo == UINT32_MAX) {
                        free(buf);
                        return false;
                    }
                    *p += 4;
                    if (lo < 0xDC00u || lo > 0xDFFFu) {
                        free(buf);
                        snprintf(err, err_len, "invalid low surrogate in JSON string");
                        return false;
                    }
                    cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                } else if (0xDC00u <= cp && cp <= 0xDFFFu) {
                    free(buf);
                    snprintf(err, err_len, "unexpected low surrogate in JSON string");
                    return false;
                }
                append_utf8(&buf, &n, &cap, cp);
            } break;
            default:
                free(buf);
                snprintf(err, err_len, "unsupported JSON escape '\\%c'", esc);
                return false;
        }
    }
    if (**p != '"') {
        free(buf);
        snprintf(err, err_len, "unterminated JSON string");
        return false;
    }
    (*p)++;
    if (!buf) {
        buf = ei_xmalloc(1);
        buf[0] = '\0';
    }
    *out = (sv_string){ buf, n };
    return true;
}

static bool append_input(string_list *list, sv_string s) {
    list->items = ei_xrealloc(list->items, sizeof(sv_string) * (list->n + 1u));
    list->items[list->n++] = s;
    return true;
}

static void free_inputs(string_list *list) {
    for (size_t i = 0; i < list->n; i++) free(list->items[i].s);
    free(list->items);
    memset(list, 0, sizeof *list);
}

static const char *find_json_field(const char *body, const char *name) {
    char pattern[64];
    int pattern_len = snprintf(pattern, sizeof pattern, "\"%s\"", name);
    if (pattern_len < 0 || (size_t)pattern_len >= sizeof pattern) return NULL;

    const char *p = body;
    while ((p = strstr(p, pattern)) != NULL) {
        const char *q = p + (size_t)pattern_len;
        skip_ws(&q);
        if (*q == ':') return q + 1;
        p += (size_t)pattern_len;
    }
    return NULL;
}

static bool parse_embed_request(const char *body, string_list *inputs,
                                int32_t *dimensions, embedding_encoding *encoding,
                                char *err, size_t err_len) {
    const char *p = find_json_field(body, "input");
    if (!p) {
        snprintf(err, err_len, "request JSON must contain an input field");
        return false;
    }
    skip_ws(&p);
    if (*p == '"') {
        sv_string s;
        if (!parse_json_string(&p, &s, err, err_len)) return false;
        append_input(inputs, s);
    } else {
        if (*p != '[') {
            snprintf(err, err_len, "input must be a string or an array of strings");
            return false;
        }
        p++;
        skip_ws(&p);
        if (*p == ']') {
            snprintf(err, err_len, "input array must contain at least one string");
            return false;
        }
        for (;;) {
            sv_string s;
            if (!parse_json_string(&p, &s, err, err_len)) return false;
            append_input(inputs, s);
            skip_ws(&p);
            if (*p == ']') break;
            if (*p != ',') {
                snprintf(err, err_len, "expected comma or closing bracket in input array");
                return false;
            }
            p++;
        }
    }

    *dimensions = EI_N_EMBD;
    p = find_json_field(body, "dimensions");
    if (p) {
        skip_ws(&p);
        if (!isdigit((unsigned char)*p)) {
            snprintf(err, err_len, "dimensions must be an integer");
            return false;
        }
        char *end = NULL;
        errno = 0;
        unsigned long long parsed = strtoull(p, &end, 10);
        if (errno || end == p) {
            snprintf(err, err_len, "invalid dimensions value");
            return false;
        }
        const char *after = end;
        skip_ws(&after);
        if (*after != ',' && *after != '}') {
            snprintf(err, err_len, "dimensions must be an integer");
            return false;
        }
        if (parsed > INT32_MAX ||
            !ei_embedding_dimensions_supported((int32_t)parsed)) {
            snprintf(err, err_len,
                     "dimensions must be one of 128, 256, 512, or 768");
            return false;
        }
        *dimensions = (int32_t)parsed;
    }

    *encoding = EMBEDDING_ENCODING_FLOAT;
    p = find_json_field(body, "encoding_format");
    if (p) {
        sv_string format;
        if (!parse_json_string(&p, &format, err, err_len)) {
            snprintf(err, err_len, "encoding_format must be \"float\" or \"base64\"");
            return false;
        }
        if (format.len == strlen("float") &&
            memcmp(format.s, "float", format.len) == 0) {
            *encoding = EMBEDDING_ENCODING_FLOAT;
        } else if (format.len == strlen("base64") &&
                   memcmp(format.s, "base64", format.len) == 0) {
            *encoding = EMBEDDING_ENCODING_BASE64;
        } else {
            free(format.s);
            snprintf(err, err_len, "encoding_format must be \"float\" or \"base64\"");
            return false;
        }
        free(format.s);
    }
    return true;
}

static void send_all(int fd, const char *buf, size_t n) {
    while (n > 0) {
        ssize_t sent = send(fd, buf, n, 0);
        if (sent < 0) {
            if (errno == EINTR) continue;
            return;
        }
        buf += sent;
        n -= (size_t)sent;
    }
}

static void http_response(int fd, int status, const char *reason,
                          const char *body, bool keep_alive) {
    char hdr[512];
    int n = snprintf(hdr, sizeof hdr,
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: %s\r\n"
        "\r\n",
        status, reason, strlen(body), keep_alive ? "keep-alive" : "close");
    if (n > 0) send_all(fd, hdr, (size_t)n);
    send_all(fd, body, strlen(body));
}

static void http_error(int fd, int status, const char *reason, const char *msg,
                       bool keep_alive) {
    sbuf b = {0};
    sbuf_append_z(&b, "{\"error\":");
    sbuf_append_json_string(&b, msg, strlen(msg));
    sbuf_append_z(&b, "}\n");
    http_response(fd, status, reason,
                  b.data ? b.data : "{\"error\":\"unknown\"}\n",
                  keep_alive);
    free(b.data);
}

static bool ascii_equal_ci(const char *a, const char *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (tolower((unsigned char)a[i]) != tolower((unsigned char)b[i])) {
            return false;
        }
    }
    return true;
}

static bool header_value(const char *headers, const char *name, sv_string *value) {
    size_t name_len = strlen(name);
    const char *line = headers;
    while (*line) {
        const char *next = strstr(line, "\r\n");
        const char *line_end = next ? next : line + strlen(line);
        const char *colon = memchr(line, ':', (size_t)(line_end - line));
        if (colon != NULL) {
            size_t key_len = (size_t)(colon - line);
            if (key_len == name_len && ascii_equal_ci(line, name, name_len)) {
                const char *begin = colon + 1;
                while (begin < line_end && isspace((unsigned char)*begin)) begin++;
                while (line_end > begin &&
                       isspace((unsigned char)line_end[-1])) line_end--;
                *value = (sv_string){ (char *)begin, (size_t)(line_end - begin) };
                return true;
            }
        }
        if (!next) break;
        line = next + 2;
    }
    return false;
}

static bool header_has_token(const char *headers, const char *name,
                             const char *token) {
    sv_string value;
    if (!header_value(headers, name, &value)) return false;
    const size_t token_len = strlen(token);
    const char *p = value.s;
    const char *end = value.s + value.len;
    while (p < end) {
        while (p < end && (isspace((unsigned char)*p) || *p == ',')) p++;
        const char *item = p;
        while (p < end && *p != ',') p++;
        const char *item_end = p;
        while (item_end > item && isspace((unsigned char)item_end[-1])) item_end--;
        if ((size_t)(item_end - item) == token_len &&
            ascii_equal_ci(item, token, token_len)) {
            return true;
        }
    }
    return false;
}

static char *find_header_end(char *buf, size_t n) {
    if (n < 4) return NULL;
    for (size_t i = 0; i + 3 < n; i++) {
        if (buf[i] == '\r' && buf[i + 1] == '\n' && buf[i + 2] == '\r' && buf[i + 3] == '\n') {
            return buf + i;
        }
    }
    return NULL;
}

typedef struct {
    int fd;
    sbuf buffered;
} http_connection;

typedef enum {
    HTTP_READ_OK,
    HTTP_READ_CLOSED,
    HTTP_READ_ERROR,
} http_read_result;

static http_read_result read_request(http_connection *connection,
                                     char **headers_out, char **body_out,
                                     size_t *body_len_out,
                                     char *err, size_t err_len) {
    char tmp[8192];
    char *hdr_end = find_header_end(
        connection->buffered.data, connection->buffered.n);
    while (!hdr_end) {
        ssize_t n = recv(connection->fd, tmp, sizeof tmp, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            snprintf(err, err_len, "failed to read request");
            return HTTP_READ_ERROR;
        }
        if (n == 0) {
            if (connection->buffered.n == 0) return HTTP_READ_CLOSED;
            snprintf(err, err_len,
                     "client closed connection before headers completed");
            return HTTP_READ_ERROR;
        }
        sbuf_append(&connection->buffered, tmp, (size_t)n);
        hdr_end = find_header_end(
            connection->buffered.data, connection->buffered.n);
        if (!hdr_end && connection->buffered.n > MAX_HEADER_BYTES) {
            snprintf(err, err_len, "request headers are too large");
            return HTTP_READ_ERROR;
        }
    }

    size_t header_len = (size_t)(hdr_end - connection->buffered.data);
    size_t body_start = header_len + 4u;
    char *headers = ei_xmalloc(header_len + 1u);
    memcpy(headers, connection->buffered.data, header_len);
    headers[header_len] = '\0';

    size_t body_len = 0;
    sv_string content_length;
    if (header_value(headers, "content-length", &content_length)) {
        if (content_length.len == 0) {
            snprintf(err, err_len, "invalid Content-Length");
            free(headers);
            return HTTP_READ_ERROR;
        }
        for (size_t i = 0; i < content_length.len; i++) {
            unsigned char c = (unsigned char)content_length.s[i];
            if (!isdigit(c) || body_len > (MAX_BODY_BYTES - (size_t)(c - '0')) / 10u) {
                snprintf(err, err_len, "invalid Content-Length");
                free(headers);
                return HTTP_READ_ERROR;
            }
            body_len = body_len * 10u + (size_t)(c - '0');
        }
    }
    sv_string transfer_encoding;
    if (header_value(headers, "transfer-encoding", &transfer_encoding)) {
        (void)transfer_encoding;
        snprintf(err, err_len, "Transfer-Encoding is not supported");
        free(headers);
        return HTTP_READ_ERROR;
    }

    while (connection->buffered.n - body_start < body_len) {
        ssize_t n = recv(connection->fd, tmp, sizeof tmp, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            snprintf(err, err_len, "failed to read request body");
            free(headers);
            return HTTP_READ_ERROR;
        }
        if (n == 0) {
            snprintf(err, err_len, "client closed connection before body completed");
            free(headers);
            return HTTP_READ_ERROR;
        }
        sbuf_append(&connection->buffered, tmp, (size_t)n);
    }

    char *body = ei_xmalloc(body_len + 1u);
    memcpy(body, connection->buffered.data + body_start, body_len);
    body[body_len] = '\0';
    size_t consumed = body_start + body_len;
    size_t remaining = connection->buffered.n - consumed;
    memmove(connection->buffered.data,
            connection->buffered.data + consumed, remaining);
    connection->buffered.n = remaining;
    connection->buffered.data[remaining] = '\0';

    *headers_out = headers;
    *body_out = body;
    *body_len_out = body_len;
    return HTTP_READ_OK;
}

static void handle_tags(int fd, const server_opts *opts, bool keep_alive) {
    (void)opts;
    sbuf b = {0};
    sbuf_append_z(&b, "{\"models\":[{\"name\":");
    sbuf_append_json_string(&b, DEFAULT_MODEL_NAME, strlen(DEFAULT_MODEL_NAME));
    sbuf_append_z(&b, "}]}");
    http_response(fd, 200, "OK", b.data, keep_alive);
    free(b.data);
}

static void handle_embed(int fd, ei_inference_service *service,
                         const server_opts *opts, const char *body,
                         size_t body_len, bool keep_alive,
                         ei_response_cache *response_cache) {
    char err[256];
    string_list inputs = {0};
    int32_t dimensions;
    embedding_encoding encoding;
    if (!parse_embed_request(body, &inputs, &dimensions, &encoding,
                             err, sizeof err)) {
        free_inputs(&inputs);
        http_error(fd, 400, "Bad Request", err, keep_alive);
        return;
    }
    if (inputs.n > opts->max_client_batch_size) {
        snprintf(err, sizeof err, "batch size %zu exceeds maximum %zu",
                 inputs.n, opts->max_client_batch_size);
        free_inputs(&inputs);
        http_error(fd, 400, "Bad Request", err, keep_alive);
        return;
    }
    if (encoding == EMBEDDING_ENCODING_FLOAT) {
        ei_response_cache_value cached;
        if (ei_response_cache_acquire(
                response_cache, body, body_len, &cached)) {
            http_response(fd, 200, "OK", cached.data, keep_alive);
            ei_response_cache_release(response_cache, &cached);
            free_inputs(&inputs);
            return;
        }
    }

    const char **texts = ei_xmalloc(inputs.n * sizeof(*texts));
    size_t *lengths = ei_xmalloc(inputs.n * sizeof(*lengths));
    float *embeddings = ei_xmalloc(inputs.n * EI_N_EMBD * sizeof(*embeddings));
    for (size_t i = 0; i < inputs.n; i++) {
        texts[i] = inputs.items[i].s;
        lengths[i] = inputs.items[i].len;
    }
    if (!ei_inference_service_embed_batch(
            service, texts, lengths, inputs.n, embeddings, err, sizeof err)) {
        free(embeddings);
        free(lengths);
        free(texts);
        free_inputs(&inputs);
        http_error(fd, 400, "Bad Request", err, keep_alive);
        return;
    }

    sbuf out = {0};
    sbuf_append_z(&out, "{\"embeddings\":[");
    for (size_t i = 0; i < inputs.n; i++) {
        float *emb = embeddings + i * EI_N_EMBD;
        if (!ei_embedding_normalize_prefix(emb, dimensions)) {
            free_inputs(&inputs);
            free(embeddings);
            free(lengths);
            free(texts);
            free(out.data);
            http_error(fd, 500, "Internal Server Error",
                       "unsupported embedding dimensions", keep_alive);
            return;
        }
        if (i) sbuf_append_z(&out, ",");
        if (encoding == EMBEDDING_ENCODING_BASE64) {
            sbuf_append_z(&out, "\"");
            sbuf_append_base64_f32(&out, emb, (size_t)dimensions);
            sbuf_append_z(&out, "\"");
        } else {
            sbuf_append_z(&out, "[");
            for (int32_t d = 0; d < dimensions; d++) {
                char num[32];
                int n = snprintf(num, sizeof num, "%s%.9g", d ? "," : "", emb[d]);
                if (n < 0 || (size_t)n >= sizeof num) {
                    free_inputs(&inputs);
                    free(embeddings);
                    free(lengths);
                    free(texts);
                    free(out.data);
                    http_error(fd, 500, "Internal Server Error",
                               "failed to format embedding", keep_alive);
                    return;
                }
                sbuf_append(&out, num, (size_t)n);
            }
            sbuf_append_z(&out, "]");
        }
    }
    sbuf_append_z(&out, "]}");
    if (encoding == EMBEDDING_ENCODING_FLOAT) {
        ei_response_cache_insert(
            response_cache, body, body_len, out.data, out.n);
    }
    http_response(fd, 200, "OK", out.data, keep_alive);
    free(out.data);
    free(embeddings);
    free(lengths);
    free(texts);
    free_inputs(&inputs);
}

static bool request_keep_alive(const char *headers, const char *version) {
    if (strcmp(version, "HTTP/1.1") == 0) {
        return !header_has_token(headers, "connection", "close");
    }
    if (strcmp(version, "HTTP/1.0") == 0) {
        return header_has_token(headers, "connection", "keep-alive");
    }
    return false;
}

static bool handle_client(http_connection *connection,
                          ei_inference_service *service,
                          const server_opts *opts,
                          ei_response_cache *response_cache,
                          bool allow_keep_alive) {
    char err[256];
    char *headers = NULL;
    char *body = NULL;
    size_t body_len = 0;
    http_read_result read_result = read_request(
        connection, &headers, &body, &body_len, err, sizeof err);
    if (read_result != HTTP_READ_OK) {
        if (read_result == HTTP_READ_ERROR) {
            http_error(connection->fd, 400, "Bad Request", err, false);
        }
        return false;
    }
    (void)body_len;

    char method[16] = {0};
    char path[256] = {0};
    char version[16] = {0};
    bool parsed = sscanf(headers, "%15s %255s %15s", method, path, version) == 3;
    bool keep_alive = parsed && allow_keep_alive &&
        request_keep_alive(headers, version);
    if (!parsed) {
        http_error(connection->fd, 400, "Bad Request",
                   "malformed request line", false);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/api/tags") == 0) {
        handle_tags(connection->fd, opts, keep_alive);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/api/embed") == 0) {
        handle_embed(connection->fd, service, opts, body, body_len,
                     keep_alive, response_cache);
    } else {
        http_error(connection->fd, 404, "Not Found",
                   "supported routes are GET /api/tags and POST /api/embed",
                   keep_alive);
    }

    free(headers);
    free(body);
    return keep_alive;
}

static char *dup_range(const char *s, size_t n) {
    char *out = ei_xmalloc(n + 1u);
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static char *dup_z(const char *s) {
    return dup_range(s, strlen(s));
}

static char *path_join(const char *a, const char *b) {
    size_t an = strlen(a);
    size_t bn = strlen(b);
    bool slash = an > 0 && a[an - 1] == '/';
    char *out = ei_xmalloc(an + (slash ? 0u : 1u) + bn + 1u);
    memcpy(out, a, an);
    size_t n = an;
    if (!slash) out[n++] = '/';
    memcpy(out + n, b, bn);
    out[n + bn] = '\0';
    return out;
}

static char *dirname_copy(const char *path) {
    const char *slash = strrchr(path, '/');
    if (!slash) return dup_z(".");
    if (slash == path) return dup_z("/");
    return dup_range(path, (size_t)(slash - path));
}

static char *resolve_model_path(const server_opts *opts) {
    if (opts->model_path) return dup_z(opts->model_path);
    const char *override = getenv("EI_MODEL_PATH");
    if (override && *override) return dup_z(override);

    const char *cache_home = getenv("XDG_CACHE_HOME");
    char *root = NULL;
    if (cache_home && *cache_home) {
        root = dup_z(cache_home);
    } else {
        const char *home = getenv("HOME");
        root = path_join(home && *home ? home : ".", ".cache");
    }
    char *directory = path_join(root, "embeddinggemma.c");
    char *model = path_join(directory, "embeddinggemma-300M-qat-Q4_0.gguf");
    free(directory);
    free(root);
    return model;
}

static bool regular_file_exists(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        if (!S_ISREG(st.st_mode)) ei_die("%s exists but is not a regular file", path);
        return true;
    }
    if (errno == ENOENT) return false;
    ei_die("cannot stat %s: %s", path, strerror(errno));
}

static void ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        if (!S_ISDIR(st.st_mode)) ei_die("%s exists but is not a directory", path);
        return;
    }
    if (errno != ENOENT) ei_die("cannot stat %s: %s", path, strerror(errno));
    if (mkdir(path, 0755) != 0 && errno != EEXIST) {
        ei_die("cannot create directory %s: %s", path, strerror(errno));
    }
}

static void mkdir_p(const char *path) {
    char *tmp = dup_z(path);
    size_t n = strlen(tmp);
    if (n == 0) {
        free(tmp);
        return;
    }
    for (size_t i = 1; i < n; i++) {
        if (tmp[i] == '/') {
            tmp[i] = '\0';
            ensure_dir(tmp);
            tmp[i] = '/';
        }
    }
    ensure_dir(tmp);
    free(tmp);
}

static size_t curl_write_file(void *ptr, size_t size, size_t nmemb, void *userdata) {
    return fwrite(ptr, size, nmemb, (FILE *)userdata) * size;
}

typedef struct {
    int last_percent;
} download_progress;

static int curl_progress(void *userdata, curl_off_t total, curl_off_t now,
                         curl_off_t ultotal, curl_off_t ulnow) {
    (void)ultotal;
    (void)ulnow;
    download_progress *p = (download_progress *)userdata;
    if (total <= 0) return 0;
    int percent = (int)((now * 100) / total);
    if (percent >= p->last_percent + 10 || (percent == 100 && p->last_percent != 100)) {
        fprintf(stderr, "download progress: %d%%\n", percent);
        p->last_percent = percent;
    }
    return 0;
}

static void download_model(const char *model_path) {
    char *model_dir = dirname_copy(model_path);
    mkdir_p(model_dir);
    free(model_dir);

    size_t tmp_len = strlen(model_path) + strlen(".download") + 1u;
    char *tmp_path = ei_xmalloc(tmp_len);
    snprintf(tmp_path, tmp_len, "%s.download", model_path);

    fprintf(stderr, "model missing; downloading %s\n", MODEL_URL);
    fprintf(stderr, "destination: %s\n", model_path);

    FILE *f = fopen(tmp_path, "wb");
    if (!f) {
        free(tmp_path);
        ei_die("cannot open temporary download file: %s", strerror(errno));
    }

    CURLcode init = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (init != CURLE_OK) {
        fclose(f);
        unlink(tmp_path);
        free(tmp_path);
        ei_die("curl_global_init failed: %s", curl_easy_strerror(init));
    }

    CURL *curl = curl_easy_init();
    if (!curl) {
        fclose(f);
        unlink(tmp_path);
        free(tmp_path);
        curl_global_cleanup();
        ei_die("curl_easy_init failed");
    }

    download_progress progress = { .last_percent = -10 };
    curl_easy_setopt(curl, CURLOPT_URL, MODEL_URL);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "embeddinggemma.c/1.0");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_file);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, f);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, curl_progress);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progress);

    CURLcode rc = curl_easy_perform(curl);
    long http = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http);
    curl_easy_cleanup(curl);
    curl_global_cleanup();

    if (fclose(f) != 0 && rc == CURLE_OK) {
        rc = CURLE_WRITE_ERROR;
    }

    if (rc != CURLE_OK) {
        unlink(tmp_path);
        free(tmp_path);
        if (http) {
            ei_die("download failed: HTTP %ld: %s", http, curl_easy_strerror(rc));
        }
        ei_die("download failed: %s", curl_easy_strerror(rc));
    }

    if (regular_file_exists(model_path)) {
        fprintf(stderr, "model appeared during download; keeping existing file\n");
        unlink(tmp_path);
        free(tmp_path);
        return;
    }
    if (rename(tmp_path, model_path) != 0) {
        unlink(tmp_path);
        free(tmp_path);
        ei_die("cannot move downloaded model into place: %s", strerror(errno));
    }
    free(tmp_path);
}

static void ensure_model_available(const char *model_path) {
    if (regular_file_exists(model_path)) return;
    download_model(model_path);
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s [--bind ADDR] [--port PORT] [--backend auto|cpu|metal|cuda]\n"
        "          [--model PATH] [--workers N] [--max-queue N]\n"
        "          [--cache-entries N] [--max-batch-tokens N]\n"
        "          [--max-batch-requests N]\n"
        "          [--max-batch-sequence-tokens N]\n"
        "          [--max-client-batch-size N] [--tokenizer-workers N]\n"
        "          [--batch-wait-us N] [--keepalive-connections N]\n"
        "          [--keepalive-max-requests N] [--keepalive-timeout-ms N]\n"
        "          [--response-cache-mb N]\n"
        "default model: $XDG_CACHE_HOME/embeddinggemma.c/%s\n"
        "               or $HOME/.cache/embeddinggemma.c/%s\n",
        argv0, "embeddinggemma-300M-qat-Q4_0.gguf",
        "embeddinggemma-300M-qat-Q4_0.gguf");
}

static bool parse_size_arg(const char *value, size_t minimum, size_t maximum,
                           size_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (errno || end == value || *end != '\0' || parsed < minimum ||
        parsed > maximum) return false;
    *out = (size_t)parsed;
    return true;
}

static bool parse_args(int argc, char **argv, server_opts *opts) {
    bool keepalive_connections_set = false;
    opts->bind_host = "0.0.0.0";
    opts->port = 11434;
    opts->backend = DEFAULT_INFERENCE_BACKEND;
    opts->model_path = NULL;
    opts->workers = 64;
    opts->max_queue = 256;
    opts->cache_entries = 4096;
    opts->max_batch_tokens = 4096;
    opts->max_batch_requests = 64;
    opts->max_batch_sequence_tokens = 512;
    opts->max_client_batch_size = 32;
    opts->tokenizer_workers = 8;
    opts->batch_wait_us = 200;
    opts->keepalive_connections = 0;
    opts->keepalive_max_requests = 100;
    opts->keepalive_timeout_ms = 1000;
    opts->response_cache_bytes = 64u * 1024u * 1024u;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bind") == 0 && i + 1 < argc) {
            opts->bind_host = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            opts->port = atoi(argv[++i]);
            if (opts->port <= 0 || opts->port > 65535) return false;
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            opts->backend = argv[++i];
            if (strcmp(opts->backend, "auto") != 0 && strcmp(opts->backend, "cpu") != 0 &&
                strcmp(opts->backend, "metal") != 0 &&
                strcmp(opts->backend, "cuda") != 0) return false;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            opts->model_path = argv[++i];
        } else if (strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
            if (!parse_size_arg(argv[++i], 1, 256, &opts->workers)) return false;
        } else if (strcmp(argv[i], "--max-queue") == 0 && i + 1 < argc) {
            if (!parse_size_arg(argv[++i], 1, 4096, &opts->max_queue)) return false;
        } else if (strcmp(argv[i], "--cache-entries") == 0 && i + 1 < argc) {
            if (!parse_size_arg(argv[++i], 0, 1048576, &opts->cache_entries)) return false;
        } else if (strcmp(argv[i], "--max-batch-tokens") == 0 && i + 1 < argc) {
            if (!parse_size_arg(argv[++i], EI_N_CTX, 65536,
                                &opts->max_batch_tokens)) return false;
        } else if (strcmp(argv[i], "--max-batch-requests") == 0 && i + 1 < argc) {
            if (!parse_size_arg(argv[++i], 1, 256,
                                &opts->max_batch_requests)) return false;
        } else if (strcmp(argv[i], "--max-batch-sequence-tokens") == 0 &&
                   i + 1 < argc) {
            if (!parse_size_arg(argv[++i], 1, EI_N_CTX,
                                &opts->max_batch_sequence_tokens)) return false;
        } else if (strcmp(argv[i], "--max-client-batch-size") == 0 &&
                   i + 1 < argc) {
            if (!parse_size_arg(argv[++i], 1, 256,
                                &opts->max_client_batch_size)) return false;
        } else if (strcmp(argv[i], "--tokenizer-workers") == 0 &&
                   i + 1 < argc) {
            if (!parse_size_arg(argv[++i], 0, 64,
                                &opts->tokenizer_workers)) return false;
        } else if (strcmp(argv[i], "--batch-wait-us") == 0 && i + 1 < argc) {
            size_t value;
            if (!parse_size_arg(argv[++i], 0, 1000000, &value)) return false;
            opts->batch_wait_us = (uint32_t)value;
        } else if (strcmp(argv[i], "--keepalive-connections") == 0 &&
                   i + 1 < argc) {
            if (!parse_size_arg(argv[++i], 0, 255,
                                &opts->keepalive_connections)) return false;
            keepalive_connections_set = true;
        } else if (strcmp(argv[i], "--keepalive-max-requests") == 0 &&
                   i + 1 < argc) {
            if (!parse_size_arg(argv[++i], 1, 1000000,
                                &opts->keepalive_max_requests)) return false;
        } else if (strcmp(argv[i], "--keepalive-timeout-ms") == 0 &&
                   i + 1 < argc) {
            size_t value;
            if (!parse_size_arg(argv[++i], 1, 60000, &value)) return false;
            opts->keepalive_timeout_ms = (uint32_t)value;
        } else if (strcmp(argv[i], "--response-cache-mb") == 0 &&
                   i + 1 < argc) {
            size_t value;
            if (!parse_size_arg(argv[++i], 0, 4096, &value)) return false;
            opts->response_cache_bytes = value * 1024u * 1024u;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            exit(0);
        } else {
            return false;
        }
    }
    if (!keepalive_connections_set) {
        opts->keepalive_connections = opts->workers / 2u;
    }
    if (opts->keepalive_connections >= opts->workers) return false;
    const char *keepalive = getenv("EI_HTTP_KEEPALIVE");
    if (keepalive && strcmp(keepalive, "0") == 0) {
        opts->keepalive_connections = 0;
    }
    return true;
}

typedef struct {
    int *fds;
    size_t capacity;
    size_t head;
    size_t count;
    pthread_mutex_t mutex;
    pthread_cond_t ready;
} socket_queue;

typedef struct {
    size_t active;
    size_t capacity;
    pthread_mutex_t mutex;
} keepalive_limiter;

typedef struct {
    socket_queue *queue;
    keepalive_limiter *keepalive;
    ei_inference_service *service;
    ei_response_cache *response_cache;
    const server_opts *opts;
} server_worker;

static void socket_queue_init(socket_queue *queue, size_t capacity) {
    memset(queue, 0, sizeof(*queue));
    queue->fds = ei_xmalloc(capacity * sizeof(*queue->fds));
    queue->capacity = capacity;
    if (pthread_mutex_init(&queue->mutex, NULL) != 0 ||
        pthread_cond_init(&queue->ready, NULL) != 0) {
        ei_die("failed to initialize HTTP connection queue");
    }
}

static bool socket_queue_try_push(socket_queue *queue, int fd) {
    pthread_mutex_lock(&queue->mutex);
    if (queue->count == queue->capacity) {
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    size_t tail = (queue->head + queue->count) % queue->capacity;
    queue->fds[tail] = fd;
    queue->count++;
    pthread_cond_signal(&queue->ready);
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

static int socket_queue_pop(socket_queue *queue) {
    pthread_mutex_lock(&queue->mutex);
    while (queue->count == 0) pthread_cond_wait(&queue->ready, &queue->mutex);
    int fd = queue->fds[queue->head];
    queue->head = (queue->head + 1) % queue->capacity;
    queue->count--;
    pthread_mutex_unlock(&queue->mutex);
    return fd;
}

static void keepalive_limiter_init(keepalive_limiter *limiter, size_t capacity) {
    *limiter = (keepalive_limiter){ .capacity = capacity };
    if (pthread_mutex_init(&limiter->mutex, NULL) != 0) {
        ei_die("failed to initialize HTTP keep-alive limiter");
    }
}

static bool keepalive_limiter_try_acquire(keepalive_limiter *limiter) {
    pthread_mutex_lock(&limiter->mutex);
    bool acquired = limiter->active < limiter->capacity;
    if (acquired) limiter->active++;
    pthread_mutex_unlock(&limiter->mutex);
    return acquired;
}

static void keepalive_limiter_release(keepalive_limiter *limiter) {
    pthread_mutex_lock(&limiter->mutex);
    limiter->active--;
    pthread_mutex_unlock(&limiter->mutex);
}

static bool wait_for_next_request(int fd, uint32_t timeout_ms) {
    struct pollfd descriptor = { .fd = fd, .events = POLLIN };
    int result;
    do {
        result = poll(&descriptor, 1, (int)timeout_ms);
    } while (result < 0 && errno == EINTR);
    return result > 0 && (descriptor.revents & POLLIN) != 0;
}

static void *server_worker_main(void *opaque) {
    server_worker *worker = opaque;
    for (;;) {
        int fd = socket_queue_pop(worker->queue);
        http_connection connection = { .fd = fd };
        bool keepalive_slot = worker->opts->keepalive_max_requests > 1 &&
            keepalive_limiter_try_acquire(worker->keepalive);
        for (size_t request = 0;; request++) {
            bool allow_keep_alive = keepalive_slot &&
                request + 1u < worker->opts->keepalive_max_requests;
            if (!handle_client(&connection, worker->service, worker->opts,
                               worker->response_cache, allow_keep_alive)) {
                break;
            }
            if (connection.buffered.n == 0 &&
                !wait_for_next_request(fd, worker->opts->keepalive_timeout_ms)) {
                break;
            }
        }
        if (keepalive_slot) keepalive_limiter_release(worker->keepalive);
        free(connection.buffered.data);
        close(fd);
    }
    return NULL;
}

static bool execute_engine_batch(void *opaque, const int32_t *ids,
                                 const size_t *offsets, size_t batch_size,
                                 float *out, char *err, size_t err_len) {
    return ei_engine_embed_tokens_batch(
        opaque, ids, offsets, batch_size, out, err, err_len);
}

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);

    server_opts opts;
    if (!parse_args(argc, argv, &opts)) {
        usage(argv[0]);
        return 2;
    }

    char *model_path = resolve_model_path(&opts);
    ensure_model_available(model_path);

    fprintf(stderr, "loading model: %s\n", model_path);
    ei_engine engine;
    ei_engine_load_backend(&engine, model_path, opts.backend);
    free(model_path);
    char reserve_error[256];
    if (!ei_engine_reserve(&engine, opts.max_batch_tokens,
                           opts.max_batch_requests,
                           reserve_error, sizeof reserve_error)) {
        ei_die("cannot reserve inference workspace: %s", reserve_error);
    }
    ei_inference_service_config service_config = {
        .cache_entries = opts.cache_entries,
        .max_batch_tokens = opts.max_batch_tokens,
        .max_batch_requests = opts.max_batch_requests,
        .max_batch_sequence_tokens = opts.max_batch_sequence_tokens,
        .tokenizer_workers = opts.tokenizer_workers,
        .batch_wait_us = opts.batch_wait_us,
    };
    ei_inference_service *service = ei_inference_service_create(
        &engine.tokenizer, execute_engine_batch, &engine, &service_config);
    if (!service) ei_die("invalid inference service configuration");
    ei_response_cache *response_cache = ei_response_cache_create(
        opts.response_cache_bytes, 4096);

    socket_queue connection_queue;
    socket_queue_init(&connection_queue, opts.max_queue);
    keepalive_limiter keepalive;
    keepalive_limiter_init(&keepalive, opts.keepalive_connections);
    pthread_t *workers = ei_xmalloc(opts.workers * sizeof(*workers));
    server_worker *worker_contexts = ei_xmalloc(
        opts.workers * sizeof(*worker_contexts));
    for (size_t i = 0; i < opts.workers; i++) {
        worker_contexts[i] = (server_worker){
            .queue = &connection_queue, .keepalive = &keepalive,
            .service = service, .response_cache = response_cache,
            .opts = &opts,
        };
        if (pthread_create(&workers[i], NULL, server_worker_main,
                           &worker_contexts[i]) != 0) {
            ei_die("failed to create HTTP worker %zu", i);
        }
    }

    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) ei_die("socket failed: %s", strerror(errno));
    int yes = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)opts.port);
    if (inet_pton(AF_INET, opts.bind_host, &addr.sin_addr) != 1) {
        ei_die("invalid --bind address '%s' (IPv4 only in this build)", opts.bind_host);
    }
    if (bind(s, (struct sockaddr *)&addr, sizeof addr) != 0) {
        ei_die("bind %s:%d failed: %s", opts.bind_host, opts.port, strerror(errno));
    }
    if (listen(s, (int)opts.max_queue) != 0) {
        ei_die("listen failed: %s", strerror(errno));
    }

    fprintf(stderr, "embeddinggemma.c serving %s on http://%s:%d (%s backend, "
                    "%zu workers, %zu batch tokens, %zu-token packing cutoff, "
                    "%zu cache entries, %zu keep-alive connections, "
                    "%zu MiB response cache)\n",
            DEFAULT_MODEL_NAME, opts.bind_host, opts.port, ei_engine_backend(&engine),
            opts.workers, opts.max_batch_tokens,
            opts.max_batch_sequence_tokens, opts.cache_entries,
            opts.keepalive_connections,
            opts.response_cache_bytes / (1024u * 1024u));
    for (;;) {
        int c = accept(s, NULL, NULL);
        if (c < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "accept failed: %s\n", strerror(errno));
            continue;
        }
        if (!socket_queue_try_push(&connection_queue, c)) {
            http_error(c, 503, "Service Unavailable",
                       "server request queue is full", false);
            close(c);
        }
    }
}
