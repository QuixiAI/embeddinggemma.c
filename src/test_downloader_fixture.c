/* Fictional downloader used only by the isolated process-spawning test. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
    if (getenv("EI_TEST_CURL_FAIL") && strstr(argv[0], "curl")) return 7;
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "-O") == 0) {
            FILE *out = fopen(argv[i + 1], "wb");
            if (!out) return 2;
            if (fwrite("fictional", 1, 9, out) != 9) return 3;
            return fclose(out) == 0 ? 0 : 4;
        }
    }
    return 5;
}
