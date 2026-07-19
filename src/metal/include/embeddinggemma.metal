#ifndef EI_EMBEDDINGGEMMA_METAL
#define EI_EMBEDDINGGEMMA_METAL

#include <metal_stdlib>

using namespace metal;

constexpr constant static uint EI_METAL_QK = 32;
constexpr constant static uint EI_METAL_N_EMBD = 768;
constexpr constant static uint EI_METAL_N_FF = 1152;
constexpr constant static uint EI_METAL_N_HEAD = 3;
constexpr constant static uint EI_METAL_HEAD_DIM = 256;

struct ei_metal_block_q4_0 {
    half d;
    uchar qs[16];
};

struct ei_metal_block_q8_0 {
    half d;
    char qs[32];
};

static_assert(sizeof(ei_metal_block_q4_0) == 18, "q4_0 block layout mismatch");
static_assert(sizeof(ei_metal_block_q8_0) == 34, "q8_0 block layout mismatch");

inline float ei_metal_gelu_tanh(float x) {
    constexpr float a = 0.044715f;
    constexpr float s = 0.7978845608028654f;
    return 0.5f * x * (1.0f + metal::tanh(s * x * (1.0f + a * x * x)));
}

#endif
