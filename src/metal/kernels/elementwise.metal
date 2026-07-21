#include "embeddinggemma.metal"

kernel void ei_gelu_mul_f32(
    device float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid < count) gate[tid] = ei_metal_gelu_tanh(gate[tid]) * up[tid];
}

kernel void ei_vec_add_f32(
    device float *dst [[buffer(0)]],
    device const float *src [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid < count) dst[tid] += src[tid];
}

kernel void ei_vec_scale_f32(
    device float *x [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    constant float &scale [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid < count) x[tid] *= scale;
}

kernel void ei_kv_f32_to_f16(
    device const float *k [[buffer(0)]],
    device const float *v [[buffer(1)]],
    device half *k_f16 [[buffer(2)]],
    device half *v_f16 [[buffer(3)]],
    constant uint &valid_count [[buffer(4)]],
    constant uint &storage_count [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid < storage_count) {
        k_f16[tid] = tid < valid_count ? half(k[tid]) : 0.0h;
        v_f16[tid] = tid < valid_count ? half(v[tid]) : 0.0h;
    }
}
