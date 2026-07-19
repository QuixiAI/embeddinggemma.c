#include "engine.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>

typedef struct {
    char  *s;
    size_t len;
} json_string;

typedef struct {
    json_string *strings;
    float       *l2;
    size_t       n;
} embed_goldens;

typedef struct {
    char *data;
    size_t len;
} file_buf;

static void *t_realloc(void *p, size_t n) {
    void *r = realloc(p, n ? n : 1);
    if (!r) ei_die("out of memory");
    return r;
}

static file_buf read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) ei_die("%s: cannot open: %s", path, strerror(errno));
    if (fseek(f, 0, SEEK_END) != 0) ei_die("%s: seek failed", path);
    long n = ftell(f);
    if (n < 0) ei_die("%s: tell failed", path);
    if (fseek(f, 0, SEEK_SET) != 0) ei_die("%s: seek failed", path);
    char *data = malloc((size_t)n + 1u);
    if (!data) ei_die("out of memory");
    if (fread(data, 1, (size_t)n, f) != (size_t)n) ei_die("%s: read failed", path);
    fclose(f);
    data[n] = '\0';
    return (file_buf){ data, (size_t)n };
}

static void skip_ws(const char **p) {
    while (isspace((unsigned char)**p)) (*p)++;
}

static void expect_char(const char **p, char c, const char *ctx) {
    skip_ws(p);
    if (**p != c) ei_die("bad JSON in %s: expected '%c'", ctx, c);
    (*p)++;
}

static bool consume_char(const char **p, char c) {
    skip_ws(p);
    if (**p != c) return false;
    (*p)++;
    return true;
}

static uint32_t hex4(const char *p, const char *ctx) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
        else ei_die("bad JSON in %s: invalid unicode escape", ctx);
    }
    return v;
}

static void append_byte(char **buf, size_t *n, size_t *cap, uint8_t b) {
    if (*n + 2u > *cap) {
        *cap = *cap ? *cap * 2u : 64u;
        *buf = t_realloc(*buf, *cap);
    }
    (*buf)[(*n)++] = (char)b;
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

static json_string parse_string(const char **p, const char *ctx) {
    skip_ws(p);
    if (**p != '"') ei_die("bad JSON in %s: expected string", ctx);
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
        if (!esc) ei_die("bad JSON in %s: unfinished escape", ctx);
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
                uint32_t cp = hex4(*p, ctx);
                *p += 4;
                if (0xD800u <= cp && cp <= 0xDBFFu) {
                    if ((*p)[0] != '\\' || (*p)[1] != 'u') {
                        ei_die("bad JSON in %s: missing low surrogate", ctx);
                    }
                    *p += 2;
                    uint32_t lo = hex4(*p, ctx);
                    *p += 4;
                    if (lo < 0xDC00u || lo > 0xDFFFu) {
                        ei_die("bad JSON in %s: invalid low surrogate", ctx);
                    }
                    cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                }
                append_utf8(&buf, &n, &cap, cp);
            } break;
            default:
                ei_die("bad JSON in %s: unsupported escape \\%c", ctx, esc);
        }
    }
    if (**p != '"') ei_die("bad JSON in %s: unterminated string", ctx);
    (*p)++;
    if (!buf) {
        buf = malloc(1);
        if (!buf) ei_die("out of memory");
        buf[0] = '\0';
    }
    return (json_string){ buf, n };
}

static void parse_strings(const char **p, embed_goldens *g, const char *ctx) {
    expect_char(p, '[', ctx);
    if (consume_char(p, ']')) return;
    for (;;) {
        g->strings = t_realloc(g->strings, sizeof(json_string) * (g->n + 1u));
        g->strings[g->n++] = parse_string(p, ctx);
        if (consume_char(p, ']')) return;
        expect_char(p, ',', ctx);
    }
}

static float parse_float(const char **p, const char *ctx) {
    skip_ws(p);
    char *end = NULL;
    float v = strtof(*p, &end);
    if (end == *p) ei_die("bad JSON in %s: expected float", ctx);
    *p = end;
    return v;
}

static void parse_l2(const char **p, embed_goldens *g, const char *ctx) {
    expect_char(p, '[', ctx);
    g->l2 = malloc(sizeof(float) * g->n * EI_N_EMBD);
    if (!g->l2) ei_die("out of memory");
    for (size_t i = 0; i < g->n; i++) {
        expect_char(p, '[', ctx);
        for (int d = 0; d < EI_N_EMBD; d++) {
            g->l2[i * EI_N_EMBD + (size_t)d] = parse_float(p, ctx);
            if (d + 1 < EI_N_EMBD) expect_char(p, ',', ctx);
        }
        expect_char(p, ']', ctx);
        if (i + 1 < g->n) expect_char(p, ',', ctx);
    }
    expect_char(p, ']', ctx);
}

static const char *field_after_colon(const char *data, const char *name, const char *ctx) {
    char needle[64];
    int n = snprintf(needle, sizeof needle, "\"%s\"", name);
    if (n < 0 || (size_t)n >= sizeof needle) ei_die("field name too long");
    const char *p = strstr(data, needle);
    if (!p) ei_die("bad JSON in %s: missing field %s", ctx, name);
    p += n;
    expect_char(&p, ':', ctx);
    return p;
}

static embed_goldens parse_goldens(const char *path) {
    file_buf f = read_file(path);
    embed_goldens g = {0};
    const char *p = field_after_colon(f.data, "strings", path);
    parse_strings(&p, &g, path);
    p = field_after_colon(f.data, "llama_l2", path);
    parse_l2(&p, &g, path);
    free(f.data);
    return g;
}

static void free_goldens(embed_goldens *g) {
    for (size_t i = 0; i < g->n; i++) free(g->strings[i].s);
    free(g->strings);
    free(g->l2);
    memset(g, 0, sizeof *g);
}

static float cosine(const float *a, const float *b) {
    float dot = 0.0f;
    float aa = 0.0f;
    float bb = 0.0f;
    for (int i = 0; i < EI_N_EMBD; i++) {
        dot += a[i] * b[i];
        aa += a[i] * a[i];
        bb += b[i] * b[i];
    }
    return dot / sqrtf(aa * bb);
}

int main(int argc, char **argv) {
    if (argc != 3) ei_die("usage: %s <model.gguf> <goldens-llamacpp.json>", argv[0]);

    ei_engine engine;
    ei_engine_load(&engine, argv[1]);
    embed_goldens g = parse_goldens(argv[2]);

    float out[EI_N_EMBD];
    float min_cos = 1.0f;
    for (size_t i = 0; i < g.n; i++) {
        char err[256];
        if (!ei_engine_embed(&engine, g.strings[i].s, g.strings[i].len, out, err, sizeof err)) {
            ei_die("embedding sample %zu failed: %s", i, err);
        }
        float c = cosine(out, g.l2 + i * EI_N_EMBD);
        if (c < min_cos) min_cos = c;
        printf("sample %zu cosine %.9f\n", i, c);
        if (c < 0.999f) {
            free_goldens(&g);
            ei_engine_free(&engine);
            return 1;
        }
    }

    printf("embedding parity: %zu/%zu samples, min cosine %.9f\n", g.n, g.n, min_cos);
    free_goldens(&g);
    ei_engine_free(&engine);
    return 0;
}
