#include "tokenizer.h"

#include <ctype.h>
#include <errno.h>

typedef struct {
    char  *s;
    size_t len;
} json_string;

typedef struct {
    int32_t *ids;
    size_t   n;
} golden_ids;

typedef struct {
    json_string *strings;
    golden_ids  *ids;
    size_t        n;
} goldens;

typedef struct {
    char  *data;
    size_t len;
    const char *path;
} file_buf;

static void *t_xrealloc(void *p, size_t n) {
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
    return (file_buf){ data, (size_t)n, path };
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

static void append_utf8(char **buf, size_t *n, size_t *cap, uint32_t cp) {
    char tmp[4];
    size_t m = 0;
    if (cp <= 0x7Fu) {
        tmp[m++] = (char)cp;
    } else if (cp <= 0x7FFu) {
        tmp[m++] = (char)(0xC0u | (cp >> 6));
        tmp[m++] = (char)(0x80u | (cp & 0x3Fu));
    } else if (cp <= 0xFFFFu) {
        tmp[m++] = (char)(0xE0u | (cp >> 12));
        tmp[m++] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
        tmp[m++] = (char)(0x80u | (cp & 0x3Fu));
    } else {
        tmp[m++] = (char)(0xF0u | (cp >> 18));
        tmp[m++] = (char)(0x80u | ((cp >> 12) & 0x3Fu));
        tmp[m++] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
        tmp[m++] = (char)(0x80u | (cp & 0x3Fu));
    }
    if (*n + m + 1u > *cap) {
        while (*n + m + 1u > *cap) *cap = *cap ? *cap * 2u : 64u;
        *buf = t_xrealloc(*buf, *cap);
    }
    memcpy(*buf + *n, tmp, m);
    *n += m;
    (*buf)[*n] = '\0';
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
            append_utf8(&buf, &n, &cap, c);
            (*p)++;
            continue;
        }

        (*p)++;
        char esc = **p;
        if (!esc) ei_die("bad JSON in %s: unfinished escape", ctx);
        (*p)++;
        switch (esc) {
            case '"': append_utf8(&buf, &n, &cap, '"'); break;
            case '\\': append_utf8(&buf, &n, &cap, '\\'); break;
            case '/': append_utf8(&buf, &n, &cap, '/'); break;
            case 'b': append_utf8(&buf, &n, &cap, '\b'); break;
            case 'f': append_utf8(&buf, &n, &cap, '\f'); break;
            case 'n': append_utf8(&buf, &n, &cap, '\n'); break;
            case 'r': append_utf8(&buf, &n, &cap, '\r'); break;
            case 't': append_utf8(&buf, &n, &cap, '\t'); break;
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
                } else if (0xDC00u <= cp && cp <= 0xDFFFu) {
                    ei_die("bad JSON in %s: unexpected low surrogate", ctx);
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

static int32_t parse_i32(const char **p, const char *ctx) {
    skip_ws(p);
    char *end = NULL;
    long v = strtol(*p, &end, 10);
    if (end == *p || v < INT32_MIN || v > INT32_MAX) {
        ei_die("bad JSON in %s: expected i32", ctx);
    }
    *p = end;
    return (int32_t)v;
}

static void append_golden_string(goldens *g, json_string s) {
    g->strings = t_xrealloc(g->strings, sizeof(json_string) * (g->n + 1u));
    g->strings[g->n++] = s;
}

static void append_golden_ids(goldens *g, golden_ids ids, size_t idx) {
    if (idx >= g->n) ei_die("bad JSON: more id arrays than strings");
    g->ids[idx] = ids;
}

static void parse_strings_array(const char **p, goldens *g, const char *ctx) {
    expect_char(p, '[', ctx);
    if (consume_char(p, ']')) return;
    for (;;) {
        append_golden_string(g, parse_string(p, ctx));
        if (consume_char(p, ']')) return;
        expect_char(p, ',', ctx);
    }
}

static golden_ids parse_id_array(const char **p, const char *ctx) {
    golden_ids ids = {0};
    size_t cap = 0;
    expect_char(p, '[', ctx);
    if (consume_char(p, ']')) return ids;
    for (;;) {
        if (ids.n == cap) {
            cap = cap ? cap * 2u : 16u;
            ids.ids = t_xrealloc(ids.ids, sizeof(int32_t) * cap);
        }
        ids.ids[ids.n++] = parse_i32(p, ctx);
        if (consume_char(p, ']')) return ids;
        expect_char(p, ',', ctx);
    }
}

static void parse_ids_array(const char **p, goldens *g, const char *ctx) {
    expect_char(p, '[', ctx);
    g->ids = calloc(g->n ? g->n : 1u, sizeof(golden_ids));
    if (!g->ids) ei_die("out of memory");
    if (consume_char(p, ']')) return;
    size_t idx = 0;
    for (;;) {
        append_golden_ids(g, parse_id_array(p, ctx), idx++);
        if (consume_char(p, ']')) break;
        expect_char(p, ',', ctx);
    }
    if (idx != g->n) ei_die("bad JSON: %zu strings but %zu id arrays", g->n, idx);
}

static goldens parse_goldens(const char *path) {
    file_buf f = read_file(path);
    const char *p = f.data;
    goldens g = {0};

    expect_char(&p, '{', f.path);
    json_string key = parse_string(&p, f.path);
    if (key.len != strlen("strings") || memcmp(key.s, "strings", key.len) != 0) {
        ei_die("bad JSON in %s: expected strings field", f.path);
    }
    free(key.s);
    expect_char(&p, ':', f.path);
    parse_strings_array(&p, &g, f.path);
    expect_char(&p, ',', f.path);
    key = parse_string(&p, f.path);
    if (key.len != strlen("ids") || memcmp(key.s, "ids", key.len) != 0) {
        ei_die("bad JSON in %s: expected ids field", f.path);
    }
    free(key.s);
    expect_char(&p, ':', f.path);
    parse_ids_array(&p, &g, f.path);
    expect_char(&p, '}', f.path);

    free(f.data);
    return g;
}

static void free_goldens(goldens *g) {
    for (size_t i = 0; i < g->n; i++) {
        free(g->strings[i].s);
        free(g->ids[i].ids);
    }
    free(g->strings);
    free(g->ids);
    memset(g, 0, sizeof *g);
}

static void print_ids(const int32_t *ids, size_t n) {
    fputc('[', stderr);
    for (size_t i = 0; i < n; i++) {
        fprintf(stderr, "%s%d", i ? ", " : "", ids[i]);
    }
    fputc(']', stderr);
}

int main(int argc, char **argv) {
    if (argc != 3) ei_die("usage: %s <model.gguf> <goldens-tokens.json>", argv[0]);

    ei_model model;
    ei_model_load(&model, argv[1]);
    ei_tokenizer tok;
    ei_tokenizer_init(&tok, &model);
    goldens g = parse_goldens(argv[2]);

    for (size_t i = 0; i < g.n; i++) {
        ei_tokens got;
        ei_tokenize_spm(&tok, g.strings[i].s, g.strings[i].len, true, false, &got);
        golden_ids *want = &g.ids[i];
        bool ok = got.n == want->n;
        for (size_t j = 0; ok && j < got.n; j++) {
            if (got.ids[j] != want->ids[j]) ok = false;
        }
        if (!ok) {
            fprintf(stderr, "token mismatch at sample %zu: \"%.*s\"\n",
                    i, (int)g.strings[i].len, g.strings[i].s);
            fputs("want: ", stderr);
            print_ids(want->ids, want->n);
            fputs("\n got: ", stderr);
            print_ids(got.ids, got.n);
            fputc('\n', stderr);
            ei_tokens_free(&got);
            free_goldens(&g);
            ei_tokenizer_free(&tok);
            ei_model_free(&model);
            return 1;
        }
        ei_tokens_free(&got);
    }

    printf("tokenizer exact-match: %zu/%zu samples\n", g.n, g.n);
    free_goldens(&g);
    ei_tokenizer_free(&tok);
    ei_model_free(&model);
    return 0;
}
