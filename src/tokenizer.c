#include "tokenizer.h"

enum {
    EI_TOKEN_UNKNOWN      = 2,
    EI_TOKEN_CONTROL      = 3,
    EI_TOKEN_USER_DEFINED = 4,
    EI_TOKEN_BYTE         = 6,
};

enum {
    EI_ATTR_UNKNOWN      = 1u << 0,
    EI_ATTR_CONTROL      = 1u << 3,
    EI_ATTR_USER_DEFINED = 1u << 4,
    EI_ATTR_BYTE         = 1u << 5,
};

typedef struct {
    int prev;
    int next;
    const char *text;
    size_t n;
} spm_symbol;

typedef struct {
    int left;
    int right;
    float score;
    size_t size;
} spm_bigram;

typedef struct {
    spm_bigram *data;
    size_t n;
    size_t cap;
} bigram_heap;

typedef struct {
    const char *key;
    size_t len;
    uint64_t hash;
    int left;
    int right;
    bool used;
} rev_slot;

typedef struct {
    rev_slot *slots;
    size_t cap;
} rev_map;

typedef struct {
    bool is_token;
    size_t off;
    size_t len;
    int32_t token;
} fragment;

typedef struct {
    fragment *data;
    size_t n;
    size_t cap;
} fragment_vec;

typedef struct {
    int32_t *data;
    size_t n;
    size_t cap;
} id_vec;

typedef struct {
    const ei_tokenizer *tok;
    const char *text;
    size_t len;
    spm_symbol *symbols;
    size_t n_symbols;
    bigram_heap heap;
    rev_map rev;
} spm_session;

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) ei_die("out of memory");
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p) ei_die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    void *r = realloc(p, n ? n : 1);
    if (!r) ei_die("out of memory");
    return r;
}

static uint64_t hash_bytes(const char *p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint8_t)p[i];
        h *= 1099511628211ull;
    }
    return h ? h : 1;
}

static size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void token_piece(const ei_tokenizer *tok, int32_t id, ei_str *out) {
    if (id < 0 || id >= (int32_t)tok->model->tok_tokens->arr_n) {
        ei_die("token id %d out of range", id);
    }
    *out = tok->pieces[id];
}

static float token_score(const ei_tokenizer *tok, int32_t id) {
    return ei_gguf_arr_f32(tok->model->tok_scores, (uint64_t)id);
}

static int32_t token_type(const ei_tokenizer *tok, int32_t id) {
    return ei_gguf_arr_i32(tok->model->tok_types, (uint64_t)id);
}

static uint32_t token_attr(const ei_tokenizer *tok, int32_t id) {
    uint32_t attr = 0;
    switch (token_type(tok, id)) {
        case EI_TOKEN_UNKNOWN:      attr |= EI_ATTR_UNKNOWN; break;
        case EI_TOKEN_CONTROL:      attr |= EI_ATTR_CONTROL; break;
        case EI_TOKEN_USER_DEFINED: attr |= EI_ATTR_USER_DEFINED; break;
        case EI_TOKEN_BYTE:         attr |= EI_ATTR_BYTE; break;
        default: break;
    }
    if (id == tok->model->bos_id || id == tok->model->eos_id || id == tok->model->pad_id) {
        attr |= EI_ATTR_CONTROL;
    }
    if (id == tok->model->unk_id) {
        attr |= EI_ATTR_UNKNOWN;
    }
    return attr;
}

static void owned_key_push(ei_tokenizer *tok, char *key) {
    tok->owned_keys = xrealloc(tok->owned_keys, sizeof(char *) * (tok->owned_keys_n + 1));
    tok->owned_keys[tok->owned_keys_n++] = key;
}

static ei_vocab_slot *vocab_slot(ei_tokenizer *tok, const char *key, size_t len, uint64_t h) {
    size_t mask = tok->vocab_cap - 1;
    size_t i = h & mask;
    for (;;) {
        ei_vocab_slot *slot = &tok->vocab[i];
        if (slot->id < 0) return slot;
        if (slot->hash == h && slot->len == len && memcmp(slot->key, key, len) == 0) return slot;
        i = (i + 1) & mask;
    }
}

static const ei_vocab_slot *vocab_lookup(const ei_tokenizer *tok, const char *key, size_t len) {
    uint64_t h = hash_bytes(key, len);
    size_t mask = tok->vocab_cap - 1;
    size_t i = h & mask;
    for (;;) {
        const ei_vocab_slot *slot = &tok->vocab[i];
        if (slot->id < 0) return NULL;
        if (slot->hash == h && slot->len == len && memcmp(slot->key, key, len) == 0) return slot;
        i = (i + 1) & mask;
    }
}

static void vocab_insert(ei_tokenizer *tok, const char *key, size_t len, int32_t id, float score) {
    uint64_t h = hash_bytes(key, len);
    ei_vocab_slot *slot = vocab_slot(tok, key, len, h);
    slot->key = key;
    slot->len = len;
    slot->hash = h;
    slot->id = id;
    slot->score = score;
}

int32_t ei_tokenizer_text_to_id(const ei_tokenizer *tok, const char *text, size_t len) {
    const ei_vocab_slot *slot = vocab_lookup(tok, text, len);
    return slot ? slot->id : -1;
}

static int special_cmp(const void *pa, const void *pb) {
    const ei_special_token *a = (const ei_special_token *)pa;
    const ei_special_token *b = (const ei_special_token *)pb;
    if (a->len != b->len) return a->len > b->len ? -1 : 1;
    if (a->id == b->id) return 0;
    return a->id < b->id ? -1 : 1;
}

static void special_push(ei_tokenizer *tok, int32_t id, uint32_t attr) {
    ei_str s;
    token_piece(tok, id, &s);
    if (s.len == 0) return;
    tok->specials = xrealloc(tok->specials, sizeof(ei_special_token) * (tok->special_n + 1));
    tok->specials[tok->special_n++] = (ei_special_token){ id, attr, (size_t)s.len };
}

void ei_tokenizer_init(ei_tokenizer *tok, const ei_model *model) {
    memset(tok, 0, sizeof *tok);
    tok->model = model;
    tok->pieces = (const ei_str *)model->tok_tokens->arr_data;
    tok->vocab_cap = next_pow2((size_t)model->tok_tokens->arr_n * 2u);
    tok->vocab = xmalloc(sizeof(ei_vocab_slot) * tok->vocab_cap);
    for (size_t i = 0; i < tok->vocab_cap; i++) tok->vocab[i].id = -1;

    for (int32_t id = 0; id < (int32_t)model->tok_tokens->arr_n; id++) {
        ei_str s;
        token_piece(tok, id, &s);
        const char *key = s.str;
        size_t len = (size_t)s.len;
        if (len == 0) {
            char tmp[64];
            int n = snprintf(tmp, sizeof tmp, "[EMPTY_%d]", id);
            if (n < 0 || (size_t)n >= sizeof tmp) ei_die("empty-token name overflow");
            char *owned = xmalloc((size_t)n);
            memcpy(owned, tmp, (size_t)n);
            owned_key_push(tok, owned);
            key = owned;
            len = (size_t)n;
        }
        vocab_insert(tok, key, len, id, token_score(tok, id));
    }

    static const char hex[] = "0123456789ABCDEF";
    for (int b = 0; b < 256; b++) {
        char buf[6] = { '<', '0', 'x', hex[(uint8_t)b >> 4], hex[b & 15], '>' };
        int32_t id = ei_tokenizer_text_to_id(tok, buf, sizeof buf);
        if (id < 0) {
            char raw = (char)b;
            id = ei_tokenizer_text_to_id(tok, &raw, 1);
        }
        if (id < 0) ei_die("missing byte fallback token for 0x%02X", b);
        tok->byte_id[b] = id;
    }

    for (int32_t id = 0; id < (int32_t)model->tok_tokens->arr_n; id++) {
        uint32_t attr = token_attr(tok, id);
        if (attr & (EI_ATTR_CONTROL | EI_ATTR_USER_DEFINED | EI_ATTR_UNKNOWN)) {
            special_push(tok, id, attr);
        }
    }
    qsort(tok->specials, tok->special_n, sizeof(ei_special_token), special_cmp);
}

void ei_tokenizer_free(ei_tokenizer *tok) {
    for (size_t i = 0; i < tok->owned_keys_n; i++) free(tok->owned_keys[i]);
    free(tok->owned_keys);
    free(tok->specials);
    free(tok->vocab);
    memset(tok, 0, sizeof *tok);
}

static void ids_push(id_vec *v, int32_t id) {
    if (v->n == v->cap) {
        v->cap = v->cap ? v->cap * 2u : 32u;
        v->data = xrealloc(v->data, sizeof(int32_t) * v->cap);
    }
    v->data[v->n++] = id;
}

static void frag_push(fragment_vec *v, fragment f) {
    if (v->n == v->cap) {
        v->cap = v->cap ? v->cap * 2u : 8u;
        v->data = xrealloc(v->data, sizeof(fragment) * v->cap);
    }
    v->data[v->n++] = f;
}

static size_t find_bytes(const char *hay, size_t hlen, const char *needle, size_t nlen, size_t start) {
    if (nlen == 0 || start > hlen || nlen > hlen - start) return SIZE_MAX;
    for (size_t i = start; i <= hlen - nlen; i++) {
        if (hay[i] == needle[0] && memcmp(hay + i, needle, nlen) == 0) return i;
    }
    return SIZE_MAX;
}

static void partition_special(const ei_tokenizer *tok, const char *raw, fragment_vec *frags,
                              bool parse_special) {
    for (size_t si = 0; si < tok->special_n; si++) {
        const ei_special_token *sp = &tok->specials[si];
        if (!parse_special && (sp->attr & (EI_ATTR_CONTROL | EI_ATTR_UNKNOWN))) continue;

        ei_str piece;
        token_piece(tok, sp->id, &piece);
        fragment_vec next = {0};

        for (size_t fi = 0; fi < frags->n; fi++) {
            fragment f = frags->data[fi];
            if (f.is_token) {
                frag_push(&next, f);
                continue;
            }

            size_t pos = 0;
            while (pos < f.len) {
                size_t match = find_bytes(raw + f.off, f.len, piece.str, (size_t)piece.len, pos);
                if (match == SIZE_MAX) break;
                if (match > pos) {
                    frag_push(&next, (fragment){
                        .is_token = false,
                        .off = f.off + pos,
                        .len = match - pos,
                        .token = 0,
                    });
                }
                frag_push(&next, (fragment){
                    .is_token = true,
                    .off = 0,
                    .len = 0,
                    .token = sp->id,
                });
                pos = match + (size_t)piece.len;
            }
            if (pos < f.len) {
                frag_push(&next, (fragment){
                    .is_token = false,
                    .off = f.off + pos,
                    .len = f.len - pos,
                    .token = 0,
                });
            }
        }

        free(frags->data);
        *frags = next;
    }
}

static bool bigram_better(spm_bigram a, spm_bigram b) {
    return a.score > b.score || (a.score == b.score && a.left < b.left);
}

static void heap_push(bigram_heap *h, spm_bigram b) {
    if (h->n == h->cap) {
        h->cap = h->cap ? h->cap * 2u : 32u;
        h->data = xrealloc(h->data, sizeof(spm_bigram) * h->cap);
    }
    size_t i = h->n++;
    h->data[i] = b;
    while (i > 0) {
        size_t parent = (i - 1u) / 2u;
        if (!bigram_better(h->data[i], h->data[parent])) break;
        spm_bigram tmp = h->data[parent];
        h->data[parent] = h->data[i];
        h->data[i] = tmp;
        i = parent;
    }
}

static bool heap_pop(bigram_heap *h, spm_bigram *out) {
    if (h->n == 0) return false;
    *out = h->data[0];
    h->data[0] = h->data[--h->n];
    size_t i = 0;
    for (;;) {
        size_t left = i * 2u + 1u;
        size_t right = left + 1u;
        size_t best = i;
        if (left < h->n && bigram_better(h->data[left], h->data[best])) best = left;
        if (right < h->n && bigram_better(h->data[right], h->data[best])) best = right;
        if (best == i) break;
        spm_bigram tmp = h->data[i];
        h->data[i] = h->data[best];
        h->data[best] = tmp;
        i = best;
    }
    return true;
}

static void rev_init(rev_map *m, size_t expected) {
    m->cap = next_pow2(expected * 4u + 16u);
    m->slots = xcalloc(m->cap, sizeof(rev_slot));
}

static void rev_put(rev_map *m, const char *key, size_t len, int left, int right) {
    uint64_t h = hash_bytes(key, len);
    size_t mask = m->cap - 1u;
    size_t i = h & mask;
    for (;;) {
        rev_slot *slot = &m->slots[i];
        if (!slot->used ||
            (slot->hash == h && slot->len == len && memcmp(slot->key, key, len) == 0)) {
            *slot = (rev_slot){ key, len, h, left, right, true };
            return;
        }
        i = (i + 1u) & mask;
    }
}

static bool rev_get(const rev_map *m, const char *key, size_t len, int *left, int *right) {
    uint64_t h = hash_bytes(key, len);
    size_t mask = m->cap - 1u;
    size_t i = h & mask;
    for (;;) {
        const rev_slot *slot = &m->slots[i];
        if (!slot->used) return false;
        if (slot->hash == h && slot->len == len && memcmp(slot->key, key, len) == 0) {
            *left = slot->left;
            *right = slot->right;
            return true;
        }
        i = (i + 1u) & mask;
    }
}

static size_t utf8_len(uint8_t lead) {
    if ((lead & 0x80u) == 0) return 1;
    if ((lead & 0xE0u) == 0xC0u) return 2;
    if ((lead & 0xF0u) == 0xE0u) return 3;
    if ((lead & 0xF8u) == 0xF0u) return 4;
    return 1;
}

static void try_add_bigram(spm_session *s, int left, int right) {
    if (left < 0 || right < 0) return;
    spm_symbol *ls = &s->symbols[left];
    spm_symbol *rs = &s->symbols[right];
    size_t len = ls->n + rs->n;
    const ei_vocab_slot *slot = vocab_lookup(s->tok, ls->text, len);
    if (!slot) return;
    heap_push(&s->heap, (spm_bigram){ left, right, slot->score, len });
    rev_put(&s->rev, ls->text, len, left, right);
}

static void resegment(spm_session *s, int idx, id_vec *out) {
    if (idx < 0 || idx >= (int)s->n_symbols) return;
    spm_symbol *sym = &s->symbols[idx];
    if (sym->n == 0) return;

    const ei_vocab_slot *slot = vocab_lookup(s->tok, sym->text, sym->n);
    if (slot) {
        ids_push(out, slot->id);
        return;
    }

    int left = -1;
    int right = -1;
    if (rev_get(&s->rev, sym->text, sym->n, &left, &right) &&
        left != idx && right != idx) {
        resegment(s, left, out);
        resegment(s, right, out);
        return;
    }

    for (size_t i = 0; i < sym->n; i++) {
        ids_push(out, s->tok->byte_id[(uint8_t)sym->text[i]]);
    }
}

static void spm_tokenize_escaped(const ei_tokenizer *tok, const char *text, size_t len, id_vec *out) {
    if (len == 0) return;
    if (len > 2147483647u) ei_die("input too large to tokenize");

    spm_session s = {0};
    s.tok = tok;
    s.text = text;
    s.len = len;
    s.symbols = xmalloc(sizeof(spm_symbol) * len);

    size_t off = 0;
    int index = 0;
    while (off < len) {
        size_t n = utf8_len((uint8_t)text[off]);
        if (n > len - off) n = len - off;
        s.symbols[index] = (spm_symbol){
            .prev = index - 1,
            .next = off + n == len ? -1 : index + 1,
            .text = text + off,
            .n = n,
        };
        off += n;
        index++;
    }
    s.n_symbols = (size_t)index;
    rev_init(&s.rev, s.n_symbols);

    for (int i = 1; i < (int)s.n_symbols; i++) {
        try_add_bigram(&s, i - 1, i);
    }

    spm_bigram b;
    while (heap_pop(&s.heap, &b)) {
        spm_symbol *left = &s.symbols[b.left];
        spm_symbol *right = &s.symbols[b.right];
        if (left->n == 0 || right->n == 0 || left->n + right->n != b.size) continue;

        left->n += right->n;
        right->n = 0;
        left->next = right->next;
        if (right->next >= 0) s.symbols[right->next].prev = b.left;

        try_add_bigram(&s, left->prev, b.left);
        try_add_bigram(&s, b.left, left->next);
    }

    for (int i = 0; i != -1; i = s.symbols[i].next) {
        resegment(&s, i, out);
    }

    free(s.rev.slots);
    free(s.heap.data);
    free(s.symbols);
}

static char *escape_whitespace(const char *text, size_t len, bool prefix_space, size_t *out_len) {
    static const char spm_space[3] = { (char)0xE2, (char)0x96, (char)0x81 };
    size_t cap = (prefix_space ? 3u : 0u) + len * 3u + 1u;
    char *buf = xmalloc(cap);
    size_t n = 0;
    if (prefix_space) {
        memcpy(buf + n, spm_space, sizeof spm_space);
        n += sizeof spm_space;
    }
    for (size_t i = 0; i < len; i++) {
        if (text[i] == ' ') {
            memcpy(buf + n, spm_space, sizeof spm_space);
            n += sizeof spm_space;
        } else {
            buf[n++] = text[i];
        }
    }
    buf[n] = '\0';
    *out_len = n;
    return buf;
}

void ei_tokenize_spm(const ei_tokenizer *tok, const char *text, size_t len,
                     bool add_special, bool parse_special, ei_tokens *out) {
    out->ids = NULL;
    out->n = 0;

    fragment_vec frags = {0};
    if (len > 0) {
        frag_push(&frags, (fragment){ .is_token = false, .off = 0, .len = len, .token = 0 });
        partition_special(tok, text, &frags, parse_special);
    }

    id_vec ids = {0};
    bool is_prev_special = true;
    if (add_special && tok->model->add_bos) {
        ids_push(&ids, tok->model->bos_id);
        is_prev_special = true;
    }

    for (size_t i = 0; i < frags.n; i++) {
        fragment f = frags.data[i];
        if (f.is_token) {
            ids_push(&ids, f.token);
            is_prev_special = true;
        } else {
            size_t esc_len = 0;
            char *escaped = escape_whitespace(text + f.off, f.len,
                                              tok->model->add_space_prefix && is_prev_special,
                                              &esc_len);
            spm_tokenize_escaped(tok, escaped, esc_len, &ids);
            free(escaped);
            is_prev_special = false;
        }
    }

    if (add_special && tok->model->add_eos) {
        ids_push(&ids, tok->model->eos_id);
    }

    free(frags.data);
    out->ids = ids.data;
    out->n = ids.n;
}

void ei_tokens_free(ei_tokens *tokens) {
    free(tokens->ids);
    tokens->ids = NULL;
    tokens->n = 0;
}
