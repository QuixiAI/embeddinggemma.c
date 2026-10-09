/* Synthetic UNC roots need no SMB service or access to a real network share. */
#include "common.h"
#include "windows_compat.h"
#include <assert.h>
int main(void) {
    assert(ei_windows_directory_start("//fictional-server/share/cache") ==
           strlen("//fictional-server/share") + 1);
    assert(ei_windows_directory_start("//fictional-server/share") ==
           strlen("//fictional-server/share") + 1);
    assert(ei_windows_directory_start("//fictional-server/") == SIZE_MAX);
    assert(ei_windows_directory_start("//fictional-server//cache") == SIZE_MAX);
    assert(ei_windows_directory_start("C:/fictional directory/cache") == 3);
    assert(ei_windows_directory_start("fictional/cache") == 1);
    char first[] = "build/windows-file-a.XXXXXX";
    char second[] = "build/windows-file-b.XXXXXX";
    int a = mkstemp(first), b = mkstemp(second);
    assert(a >= 0 && b >= 0 && strcmp(first, second) != 0);
    const char bytes[] = {'\r', '\n', 26, 0};
    assert(_write(a, bytes, sizeof bytes) == (int)sizeof bytes);
    assert(_close(a) == 0 && _close(b) == 0);
    assert(ei_replace_file(first, second) == 0);
    FILE *file = fopen(second, "rb");
    assert(file);
    char actual[sizeof bytes];
    assert(fread(actual, 1, sizeof actual, file) == sizeof actual);
    assert(memcmp(actual, bytes, sizeof bytes) == 0);
    assert(fclose(file) == 0 && _unlink(second) == 0);
    puts("Windows UNC roots, binary temporary files and cache replacement: passed");
    return 0;
}
