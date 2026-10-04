/* Native Windows file helpers; MinGW supplies C11 and winpthreads. */
#ifndef EI_WINDOWS_COMPAT_H
#define EI_WINDOWS_COMPAT_H
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Exclusive creation preserves mkstemp's security and binary file semantics. */
static inline int ei_mkstemp(char *pattern) {
    size_t n = strlen(pattern);
    if (n < 6 || strcmp(pattern + n - 6, "XXXXXX") != 0) {
        errno = EINVAL;
        return -1;
    }
    for (unsigned attempt = 0; attempt < 256; attempt++) {
        unsigned int random;
        if (rand_s(&random) != 0) { errno = EIO; return -1; }
        static const char alphabet[] = "0123456789abcdefghijklmnopqrstuvwxyz";
        for (size_t i = n - 6; i < n; i++) {
            pattern[i] = alphabet[random % 36];
            random /= 36;
        }
        int fd = _open(pattern, _O_CREAT | _O_EXCL | _O_RDWR | _O_BINARY,
                       _S_IREAD | _S_IWRITE);
        if (fd >= 0 || errno != EEXIST) return fd;
    }
    errno = EEXIST;
    return -1;
}
#define mkstemp ei_mkstemp
/* Windows rename does not replace a destination; cache publication must. */
static inline int ei_replace_file(const char *from, const char *to) {
    if (MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING)) return 0;
    errno = EIO;
    return -1;
}
#else
#define ei_replace_file rename
#endif
#endif
