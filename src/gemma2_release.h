/* Drop the resident pages of a weight tensor once a kernel holds
 * a repacked copy. The GGUF is mapped PROT_READ | MAP_PRIVATE, so its pages are
 * never dirty: MADV_DONTNEED unmaps them (they are re-read from the file if
 * anything touches them again) and the process keeps one copy of each repacked
 * weight instead of two. Only whole pages inside the tensor are released.
 * EI_GEMMA2_CPU_RELEASE_WEIGHTS=0 keeps the source pages mapped. */
#ifndef EI_GEMMA2_RELEASE_H
#define EI_GEMMA2_RELEASE_H
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "gguf.h"
#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

static inline void g2_release_source(const void *data, size_t bytes) {
#if defined(__linux__) || defined(__APPLE__)
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("EI_GEMMA2_CPU_RELEASE_WEIGHTS");
        on = !(e && !strcmp(e, "0"));
    }
    /* Only pages of a read-only GGUF mapping: on anonymous memory (weights built
     * in memory, e.g. by tests) MADV_DONTNEED would zero the data. */
    if (!on || !ei_gguf_in_mapping(data, bytes))
        return;
    const uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE);
    uintptr_t lo = ((uintptr_t)data + page - 1) & ~(page - 1), hi = ((uintptr_t)data + bytes) & ~(page - 1);
    if (hi > lo)
        madvise((void *)lo, hi - lo, MADV_DONTNEED);
#else
    (void)data, (void)bytes;
#endif
}
#endif
