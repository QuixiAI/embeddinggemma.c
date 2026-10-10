#ifndef EI_PROFILES_H
#define EI_PROFILES_H
#include <string.h>
typedef struct {
    const char *name, *file, *url, *projector_file, *projector_url;
} ei_profile;
static const ei_profile ei_profiles[] = {
    {"embeddinggemma-2", "embeddinggemma-2-Q8_0.gguf",
     "https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/"
     "bfcd298762cc34d0357ece5ebdd31791a3a374d8/embeddinggemma-2-Q8_0.gguf",
     "mmproj-embeddinggemma-2-Q8_0.gguf",
     "https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/"
     "bfcd298762cc34d0357ece5ebdd31791a3a374d8/mmproj-embeddinggemma-2-Q8_0.gguf"},
    {"embeddinggemma-300m", "embeddinggemma-300M-qat-Q4_0.gguf",
     "https://huggingface.co/ggml-org/embeddinggemma-300M-qat-q4_0-GGUF/resolve/main/"
     "embeddinggemma-300M-qat-Q4_0.gguf",
     NULL, NULL},
};
static inline const ei_profile *ei_profile_find(const char *name) {
    for (size_t i = 0; i < sizeof(ei_profiles) / sizeof(ei_profiles[0]); ++i)
        if (!strcmp(name, ei_profiles[i].name))
            return &ei_profiles[i];
    return NULL;
}
#endif
