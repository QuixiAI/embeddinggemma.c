/* cpu-max fork: build identity, so an installed binary says what it is.
 * The Makefile defines the three values from CPU_MAX_VERSION, the upstream
 * commit this fork is based on (UPSTREAM_BASE) and the git commit. */
#ifndef EI_CPU_MAX_VERSION_H
#define EI_CPU_MAX_VERSION_H
#ifndef EI_CPU_MAX_VERSION
#define EI_CPU_MAX_VERSION "unknown"
#endif
#ifndef EI_UPSTREAM_VERSION
#define EI_UPSTREAM_VERSION "unknown"
#endif
#ifndef EI_GIT_COMMIT
#define EI_GIT_COMMIT "unknown"
#endif
#define EI_BUILD_IDENTITY                                                                          \
    "EmbeddingGemma-2-CPU-Max " EI_CPU_MAX_VERSION " (commit " EI_GIT_COMMIT                       \
    "; fork of QuixiAI/embeddinggemma.c " EI_UPSTREAM_VERSION ")"
#endif
