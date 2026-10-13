#pragma once
// EmbeddingGemma 2 kernel parameters and weight decode, shared by the Metal 3
// kernel files.
#include <metal_stdlib>
using namespace metal;
struct G2P {
    uint n, k, m, type, heads, kv, d, window, cols, mode, layer;
    float scale, lo, hi;
};
struct G2Q8 {
    half d;
    char qs[32];
};
inline float g2w(device const uchar *w, uint type, uint i) {
    if (type == 0)
        return ((device const float *)w)[i];
    if (type == 30)
        return as_type<float>(uint(((device const ushort *)w)[i]) << 16);
    if (type == 1)
        return float(((device const half *)w)[i]);
    device const G2Q8 *b = (device const G2Q8 *)w;
    return float(b[i / 32].d) * float(b[i / 32].qs[i % 32]);
}
