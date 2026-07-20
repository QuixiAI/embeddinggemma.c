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

// Normalize a projection into the residual stream, then emit the next RMSNorm.
kernel void ei_rms_norm_residual_next_f32(
    device const float *input [[buffer(0)]],
    device const float *post_weight [[buffer(1)]],
    device float *residual [[buffer(2)]],
    device const float *next_weight [[buffer(3)]],
    device float *next_output [[buffer(4)]],
    constant uint &n_rows [[buffer(5)]],
    constant uint &n_cols [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (row >= n_rows) return;
    const uint base = row * n_cols;
    float projected_ss = 0.0f;
    for (uint col = lane; col < n_cols; col += 32) {
        const float value = input[base + col];
        projected_ss += value * value;
    }
    projected_ss = metal::simd_sum(projected_ss);
    const float projected_inv = metal::rsqrt(
        projected_ss / float(n_cols) + eps);

    float residual_ss = 0.0f;
    for (uint col = lane; col < n_cols; col += 32) {
        const float value = residual[base + col] +
            input[base + col] * post_weight[col] * projected_inv;
        residual[base + col] = value;
        residual_ss += value * value;
    }
    residual_ss = metal::simd_sum(residual_ss);
    const float residual_inv = metal::rsqrt(
        residual_ss / float(n_cols) + eps);
    for (uint col = lane; col < n_cols; col += 32) {
        next_output[base + col] = residual[base + col] * next_weight[col] *
                                  residual_inv;
    }
}
