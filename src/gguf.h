/* Minimal GGUF v3 reader: mmaps the file, indexes KVs and tensors.
 * Exactly what loading embeddinggemma-300M-qat-Q4_0.gguf requires. */
#ifndef EI_GGUF_H
#define EI_GGUF_H

#include "common.h"

enum ei_gguf_kv_type {
    EI_GGUF_U8 = 0, EI_GGUF_I8, EI_GGUF_U16, EI_GGUF_I16,
    EI_GGUF_U32, EI_GGUF_I32, EI_GGUF_F32, EI_GGUF_BOOL,
    EI_GGUF_STRING, EI_GGUF_ARRAY, EI_GGUF_U64, EI_GGUF_I64, EI_GGUF_F64,
};

/* ggml tensor dtypes present in (or relevant to) the model file. */
enum ei_tensor_type {
    EI_T_F32  = 0,
    EI_T_F16  = 1,
    EI_T_Q4_0 = 2,
    EI_T_Q8_0 = 8,
};

typedef struct {
    const char *str; /* NOT null-terminated */
    uint64_t    len;
} ei_str;

typedef struct {
    ei_str   key;
    uint32_t type;          /* ei_gguf_kv_type */
    /* scalar value (valid when type is scalar) */
    union { uint64_t u64; int64_t i64; double f64; bool b; ei_str s; } v;
    /* array value (valid when type == EI_GGUF_ARRAY) */
    uint32_t    arr_type;
    uint64_t    arr_n;
    const void *arr_data;   /* packed scalars, or encoded strings for STRING */
} ei_kv;

typedef struct {
    ei_str      name;
    uint32_t    n_dims;
    uint64_t    ne[4];      /* ne[0] = innermost (row length) */
    uint32_t    type;       /* ei_tensor_type */
    uint64_t    offset;     /* relative to data section */
    const void *data;       /* resolved pointer into the mapping */
    uint64_t    n_bytes;
} ei_tensor;

typedef struct {
    const uint8_t *map;     /* whole-file mmap (read-only) */
    size_t         map_len;
    uint32_t       version;
    uint64_t       n_kv;
    uint64_t       n_tensors;
    uint64_t       alignment;
    uint64_t       data_off;
    ei_kv         *kv;
    ei_tensor     *tensors;
} ei_gguf;

/* Load and index a GGUF file; dies with a message on any malformation. */
void ei_gguf_open(ei_gguf *g, const char *path);
void ei_gguf_close(ei_gguf *g);

/* Lookups (die when `required`, else return NULL/defaults). */
const ei_kv     *ei_gguf_kv(const ei_gguf *g, const char *key, bool required);
uint64_t         ei_gguf_kv_u64(const ei_gguf *g, const char *key, uint64_t dflt);
double           ei_gguf_kv_f64(const ei_gguf *g, const char *key, double dflt);
bool             ei_gguf_kv_bool(const ei_gguf *g, const char *key, bool dflt);
const ei_tensor *ei_gguf_tensor(const ei_gguf *g, const char *name, bool required);

/* Iterate a STRING array KV: returns element i into out (bounds-checked). */
void ei_gguf_arr_str(const ei_kv *kv, uint64_t i, ei_str *out);
/* Typed array element accessors (bounds-checked). */
float   ei_gguf_arr_f32(const ei_kv *kv, uint64_t i);
int32_t ei_gguf_arr_i32(const ei_kv *kv, uint64_t i);

uint64_t ei_tensor_row_bytes(const ei_tensor *t);
uint64_t ei_tensor_n_elements(const ei_tensor *t);

#endif /* EI_GGUF_H */
