#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_MAGIC 0x46554747u /* "GGUF" little-endian */

typedef struct {
    const uint8_t *p;
    const uint8_t *end;
    const char    *path;
} cur;

static const uint8_t *take(cur *c, uint64_t n) {
    if ((uint64_t)(c->end - c->p) < n) {
        ei_die("%s: truncated file (need %llu more bytes)", c->path, (unsigned long long)n);
    }
    const uint8_t *r = c->p;
    c->p += n;
    return r;
}

static uint32_t take_u32(cur *c) { uint32_t v; memcpy(&v, take(c, 4), 4); return v; }
static uint64_t take_u64(cur *c) { uint64_t v; memcpy(&v, take(c, 8), 8); return v; }

static ei_str take_str(cur *c) {
    ei_str s;
    s.len = take_u64(c);
    s.str = (const char *)take(c, s.len);
    return s;
}

static uint64_t scalar_size(uint32_t type, const char *path) {
    switch (type) {
        case EI_GGUF_U8: case EI_GGUF_I8: case EI_GGUF_BOOL: return 1;
        case EI_GGUF_U16: case EI_GGUF_I16: return 2;
        case EI_GGUF_U32: case EI_GGUF_I32: case EI_GGUF_F32: return 4;
        case EI_GGUF_U64: case EI_GGUF_I64: case EI_GGUF_F64: return 8;
        default: ei_die("%s: unsupported kv scalar type %u", path, type);
    }
}

/* Reads a scalar of `type` at p into the union. */
static void read_scalar(ei_kv *kv, uint32_t type, const uint8_t *p) {
    switch (type) {
        case EI_GGUF_U8:  kv->v.u64 = *p; break;
        case EI_GGUF_I8:  kv->v.i64 = *(const int8_t *)p; break;
        case EI_GGUF_U16: { uint16_t v; memcpy(&v, p, 2); kv->v.u64 = v; } break;
        case EI_GGUF_I16: { int16_t v; memcpy(&v, p, 2); kv->v.i64 = v; } break;
        case EI_GGUF_U32: { uint32_t v; memcpy(&v, p, 4); kv->v.u64 = v; } break;
        case EI_GGUF_I32: { int32_t v; memcpy(&v, p, 4); kv->v.i64 = v; } break;
        case EI_GGUF_F32: { float v; memcpy(&v, p, 4); kv->v.f64 = v; } break;
        case EI_GGUF_U64: { uint64_t v; memcpy(&v, p, 8); kv->v.u64 = v; } break;
        case EI_GGUF_I64: { int64_t v; memcpy(&v, p, 8); kv->v.i64 = v; } break;
        case EI_GGUF_F64: { double v; memcpy(&v, p, 8); kv->v.f64 = v; } break;
        case EI_GGUF_BOOL: kv->v.b = *p != 0; break;
        default: break;
    }
}

/* For ARRAY of STRING we pre-index every element so lookups are O(1);
 * the vocab is 262k strings and the tokenizer reads them all anyway. */
static void parse_array(ei_kv *kv, cur *c) {
    kv->arr_type = take_u32(c);
    kv->arr_n    = take_u64(c);
    if (kv->arr_type == EI_GGUF_STRING) {
        ei_str *idx = malloc(sizeof(ei_str) * kv->arr_n);
        if (!idx) ei_die("out of memory indexing string array (%llu entries)",
                         (unsigned long long)kv->arr_n);
        for (uint64_t i = 0; i < kv->arr_n; i++) idx[i] = take_str(c);
        kv->arr_data = idx;
    } else if (kv->arr_type == EI_GGUF_ARRAY) {
        ei_die("%s: nested arrays are not supported", c->path);
    } else {
        uint64_t sz = scalar_size(kv->arr_type, c->path);
        kv->arr_data = take(c, sz * kv->arr_n);
    }
}

static uint64_t block_bytes(uint32_t type, const char *name) {
    switch (type) {
        case EI_T_F32:  return 4;   /* per element */
        case EI_T_F16:  return 2;   /* per element */
        case EI_T_Q4_0: return 18;  /* per 32 elements */
        case EI_T_Q8_0: return 34;  /* per 32 elements */
        default: ei_die("tensor %s: unsupported ggml type %u", name, type);
    }
}

static bool is_quant(uint32_t type) { return type == EI_T_Q4_0 || type == EI_T_Q8_0; }

uint64_t ei_tensor_n_elements(const ei_tensor *t) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) n *= t->ne[i];
    return n;
}

uint64_t ei_tensor_row_bytes(const ei_tensor *t) {
    uint64_t bb = block_bytes(t->type, "?");
    return is_quant(t->type) ? t->ne[0] / 32 * bb : t->ne[0] * bb;
}

static uint64_t tensor_bytes(const ei_tensor *t, const char *namebuf) {
    uint64_t n  = ei_tensor_n_elements(t);
    uint64_t bb = block_bytes(t->type, namebuf);
    if (is_quant(t->type)) {
        if (t->ne[0] % 32 != 0) {
            ei_die("tensor %s: row length %llu not a multiple of the quant block (32)",
                   namebuf, (unsigned long long)t->ne[0]);
        }
        return n / 32 * bb;
    }
    return n * bb;
}

void ei_gguf_open(ei_gguf *g, const char *path) {
    memset(g, 0, sizeof *g);

    int fd = open(path, O_RDONLY);
    if (fd < 0) ei_die("%s: cannot open", path);
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) ei_die("%s: cannot stat", path);
    g->map_len = (size_t)st.st_size;
    g->map = mmap(NULL, g->map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (g->map == MAP_FAILED) ei_die("%s: mmap failed", path);

    cur c = { g->map, g->map + g->map_len, path };

    if (take_u32(&c) != GGUF_MAGIC) ei_die("%s: not a GGUF file", path);
    g->version = take_u32(&c);
    if (g->version != 3) ei_die("%s: GGUF version %u (only v3 supported)", path, g->version);
    g->n_tensors = take_u64(&c);
    g->n_kv      = take_u64(&c);

    g->kv = calloc(g->n_kv, sizeof(ei_kv));
    g->tensors = calloc(g->n_tensors, sizeof(ei_tensor));
    if (!g->kv || !g->tensors) ei_die("%s: out of memory", path);

    g->alignment = 32; /* default; general.alignment overrides below */

    for (uint64_t i = 0; i < g->n_kv; i++) {
        ei_kv *kv = &g->kv[i];
        kv->key  = take_str(&c);
        kv->type = take_u32(&c);
        if (kv->type == EI_GGUF_ARRAY) {
            parse_array(kv, &c);
        } else if (kv->type == EI_GGUF_STRING) {
            kv->v.s = take_str(&c);
        } else {
            uint64_t sz = scalar_size(kv->type, path);
            read_scalar(kv, kv->type, take(&c, sz));
        }
    }

    const ei_kv *align = ei_gguf_kv(g, "general.alignment", false);
    if (align) g->alignment = align->v.u64;
    if (g->alignment == 0 || (g->alignment & (g->alignment - 1)) != 0) {
        ei_die("%s: bad alignment %llu", path, (unsigned long long)g->alignment);
    }

    for (uint64_t i = 0; i < g->n_tensors; i++) {
        ei_tensor *t = &g->tensors[i];
        t->name   = take_str(&c);
        t->n_dims = take_u32(&c);
        if (t->n_dims == 0 || t->n_dims > 4) {
            ei_die("%s: tensor %.*s has %u dims", path,
                   (int)t->name.len, t->name.str, t->n_dims);
        }
        for (uint32_t d = 0; d < 4; d++) t->ne[d] = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) t->ne[d] = take_u64(&c);
        t->type   = take_u32(&c);
        t->offset = take_u64(&c);
    }

    uint64_t off = (uint64_t)(c.p - g->map);
    g->data_off = (off + g->alignment - 1) & ~(g->alignment - 1);

    for (uint64_t i = 0; i < g->n_tensors; i++) {
        ei_tensor *t = &g->tensors[i];
        char namebuf[256];
        snprintf(namebuf, sizeof namebuf, "%.*s", (int)t->name.len, t->name.str);
        t->n_bytes = tensor_bytes(t, namebuf);
        if (t->offset % g->alignment != 0) {
            ei_die("%s: tensor %s misaligned offset", path, namebuf);
        }
        uint64_t abs = g->data_off + t->offset;
        if (abs + t->n_bytes > g->map_len) {
            ei_die("%s: tensor %s extends past end of file", path, namebuf);
        }
        t->data = g->map + abs;
    }
}

void ei_gguf_close(ei_gguf *g) {
    for (uint64_t i = 0; i < g->n_kv; i++) {
        if (g->kv[i].type == EI_GGUF_ARRAY && g->kv[i].arr_type == EI_GGUF_STRING) {
            free((void *)g->kv[i].arr_data);
        }
    }
    free(g->kv);
    free(g->tensors);
    if (g->map) munmap((void *)g->map, g->map_len);
    memset(g, 0, sizeof *g);
}

static bool str_eq(ei_str s, const char *z) {
    size_t n = strlen(z);
    return s.len == n && memcmp(s.str, z, n) == 0;
}

const ei_kv *ei_gguf_kv(const ei_gguf *g, const char *key, bool required) {
    for (uint64_t i = 0; i < g->n_kv; i++) {
        if (str_eq(g->kv[i].key, key)) return &g->kv[i];
    }
    if (required) ei_die("missing required GGUF key: %s", key);
    return NULL;
}

uint64_t ei_gguf_kv_u64(const ei_gguf *g, const char *key, uint64_t dflt) {
    const ei_kv *kv = ei_gguf_kv(g, key, false);
    return kv ? kv->v.u64 : dflt;
}

double ei_gguf_kv_f64(const ei_gguf *g, const char *key, double dflt) {
    const ei_kv *kv = ei_gguf_kv(g, key, false);
    return kv ? kv->v.f64 : dflt;
}

bool ei_gguf_kv_bool(const ei_gguf *g, const char *key, bool dflt) {
    const ei_kv *kv = ei_gguf_kv(g, key, false);
    return kv ? kv->v.b : dflt;
}

const ei_tensor *ei_gguf_tensor(const ei_gguf *g, const char *name, bool required) {
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (str_eq(g->tensors[i].name, name)) return &g->tensors[i];
    }
    if (required) ei_die("missing required tensor: %s", name);
    return NULL;
}

void ei_gguf_arr_str(const ei_kv *kv, uint64_t i, ei_str *out) {
    if (kv->type != EI_GGUF_ARRAY || kv->arr_type != EI_GGUF_STRING || i >= kv->arr_n) {
        ei_die("bad string-array access (index %llu of %llu)",
               (unsigned long long)i, (unsigned long long)kv->arr_n);
    }
    *out = ((const ei_str *)kv->arr_data)[i];
}

float ei_gguf_arr_f32(const ei_kv *kv, uint64_t i) {
    if (kv->type != EI_GGUF_ARRAY || kv->arr_type != EI_GGUF_F32 || i >= kv->arr_n) {
        ei_die("bad f32-array access");
    }
    float v;
    memcpy(&v, (const uint8_t *)kv->arr_data + 4 * i, 4);
    return v;
}

int32_t ei_gguf_arr_i32(const ei_kv *kv, uint64_t i) {
    if (kv->type != EI_GGUF_ARRAY || kv->arr_type != EI_GGUF_I32 || i >= kv->arr_n) {
        ei_die("bad i32-array access");
    }
    int32_t v;
    memcpy(&v, (const uint8_t *)kv->arr_data + 4 * i, 4);
    return v;
}
