/* SentencePiece tokenizer for the single supported embeddinggemma GGUF. */
#ifndef EI_TOKENIZER_H
#define EI_TOKENIZER_H

#include "model.h"

typedef struct {
    const char *key;
    size_t      len;
    uint64_t    hash;
    int32_t     id;
    float       score;
} ei_vocab_slot;

typedef struct {
    int32_t  id;
    uint32_t attr;
    size_t   len;
} ei_special_token;

typedef struct {
    const ei_model *model;
    const ei_str   *pieces;

    ei_vocab_slot *vocab;
    size_t         vocab_cap;

    int32_t byte_id[256];

    ei_special_token *specials;
    size_t            special_n;

    char  **owned_keys;
    size_t  owned_keys_n;
} ei_tokenizer;

typedef struct {
    int32_t *ids;
    size_t   n;
} ei_tokens;

void ei_tokenizer_init(ei_tokenizer *tok, const ei_model *model);
void ei_tokenizer_free(ei_tokenizer *tok);

void ei_tokenize_spm(const ei_tokenizer *tok, const char *text, size_t len,
                     bool add_special, bool parse_special, ei_tokens *out);
void ei_tokens_free(ei_tokens *tokens);

int32_t ei_tokenizer_text_to_id(const ei_tokenizer *tok, const char *text, size_t len);

#endif /* EI_TOKENIZER_H */
