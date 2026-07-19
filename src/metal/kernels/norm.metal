#include "embeddinggemma.metal"

// Register-resident, one-simdgroup-per-row RMSNorm as used by QuixiCore-Metal.
kernel void ei_rms_norm_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &n_rows [[buffer(3)]],
    constant uint &n_cols [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (row >= n_rows) return;
    const uint base = row * n_cols;
    float sum = 0.0f;
    for (uint col = lane; col < n_cols; col += 32) {
        const float value = input[base + col];
        sum += value * value;
    }
    sum = metal::simd_sum(sum);
    const float inv = metal::rsqrt(sum / float(n_cols) + eps);
    for (uint col = lane; col < n_cols; col += 32) {
        output[base + col] = input[base + col] * weight[col] * inv;
    }
}

// Normalize a projection and add it directly into the residual stream.
kernel void ei_rms_norm_residual_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *residual [[buffer(2)]],
    constant uint &n_rows [[buffer(3)]],
    constant uint &n_cols [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (row >= n_rows) return;
    const uint base = row * n_cols;
    float sum = 0.0f;
    for (uint col = lane; col < n_cols; col += 32) {
        const float value = input[base + col];
        sum += value * value;
    }
    sum = metal::simd_sum(sum);
    const float inv = metal::rsqrt(sum / float(n_cols) + eps);
    for (uint col = lane; col < n_cols; col += 32) {
        residual[base + col] += input[base + col] * weight[col] * inv;
    }
}
