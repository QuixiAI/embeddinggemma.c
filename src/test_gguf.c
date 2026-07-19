/* Prints the parsed model as JSON for diffing against testdata/model-manifest.json
 * (which was produced by llama.cpp's gguf-py from the same file). */
#include "gguf.h"

int main(int argc, char **argv) {
    if (argc != 2) ei_die("usage: %s <model.gguf>", argv[0]);

    ei_gguf g;
    ei_gguf_open(&g, argv[1]);

    printf("{\n \"version\": %u,\n \"n_kv\": %llu,\n \"n_tensors\": %llu,\n"
           " \"alignment\": %llu,\n \"data_off\": %llu,\n \"tensors\": [\n",
           g.version,
           (unsigned long long)g.n_kv,
           (unsigned long long)g.n_tensors,
           (unsigned long long)g.alignment,
           (unsigned long long)g.data_off);

    for (uint64_t i = 0; i < g.n_tensors; i++) {
        const ei_tensor *t = &g.tensors[i];
        printf("  {\"name\": \"%.*s\", \"type\": %u, \"shape\": [",
               (int)t->name.len, t->name.str, t->type);
        for (uint32_t d = 0; d < t->n_dims; d++) {
            printf("%s%llu", d ? ", " : "", (unsigned long long)t->ne[d]);
        }
        printf("], \"abs_offset\": %llu, \"n_bytes\": %llu}%s\n",
               (unsigned long long)(g.data_off + t->offset),
               (unsigned long long)t->n_bytes,
               i + 1 < g.n_tensors ? "," : "");
    }
    printf(" ]\n}\n");

    ei_gguf_close(&g);
    return 0;
}
