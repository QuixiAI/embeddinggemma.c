#include "embeddinggemma.metal"

// Final RMSNorm, mean pooling, and L2 normalization in one simdgroup. Each
// lane retains 24 of the 768 pooled channels in registers across all tokens.
kernel void ei_mean_pool_rms_l2_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &n_tokens [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint values_per_lane = EI_METAL_N_EMBD / 32;
    float pooled[values_per_lane];
#pragma clang loop unroll(full)
    for (uint i = 0; i < values_per_lane; i++) pooled[i] = 0.0f;

    for (uint token = 0; token < n_tokens; token++) {
        float sum = 0.0f;
#pragma clang loop unroll(full)
        for (uint i = 0; i < values_per_lane; i++) {
            const uint dim = lane + i * 32;
            const float value = input[token * EI_METAL_N_EMBD + dim];
            sum += value * value;
        }
        sum = metal::simd_sum(sum);
        const float inv = metal::rsqrt(sum / float(EI_METAL_N_EMBD) + eps);
#pragma clang loop unroll(full)
        for (uint i = 0; i < values_per_lane; i++) {
            const uint dim = lane + i * 32;
            pooled[i] += input[token * EI_METAL_N_EMBD + dim] * weight[dim] * inv;
        }
    }

    const float inv_tokens = 1.0f / float(n_tokens);
    float sum = 0.0f;
#pragma clang loop unroll(full)
    for (uint i = 0; i < values_per_lane; i++) {
        pooled[i] *= inv_tokens;
        sum += pooled[i] * pooled[i];
    }
    sum = metal::simd_sum(sum);
    const float inv_l2 = sum == 0.0f ? 1.0f : metal::rsqrt(sum);
#pragma clang loop unroll(full)
    for (uint i = 0; i < values_per_lane; i++) {
        const uint dim = lane + i * 32;
        output[dim] = pooled[i] * inv_l2;
    }
}

kernel void ei_mean_pool_rms_l2_f32_batch(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const uint *offsets [[buffer(3)]],
    constant uint &batch_size [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint sequence = group.x;
    if (sequence >= batch_size) return;
    const uint start = offsets[sequence];
    const uint stop = offsets[sequence + 1];
    constexpr uint values_per_lane = EI_METAL_N_EMBD / 32;
    float pooled[values_per_lane];
#pragma clang loop unroll(full)
    for (uint i = 0; i < values_per_lane; i++) pooled[i] = 0.0f;

    for (uint token = start; token < stop; token++) {
        float sum = 0.0f;
#pragma clang loop unroll(full)
        for (uint i = 0; i < values_per_lane; i++) {
            const uint dim = lane + i * 32;
            const float value = input[token * EI_METAL_N_EMBD + dim];
            sum += value * value;
        }
        sum = metal::simd_sum(sum);
        const float inv = metal::rsqrt(sum / float(EI_METAL_N_EMBD) + eps);
#pragma clang loop unroll(full)
        for (uint i = 0; i < values_per_lane; i++) {
            const uint dim = lane + i * 32;
            pooled[i] += input[token * EI_METAL_N_EMBD + dim] * weight[dim] * inv;
        }
    }

    const float inv_tokens = 1.0f / float(stop - start);
    float sum = 0.0f;
#pragma clang loop unroll(full)
    for (uint i = 0; i < values_per_lane; i++) {
        pooled[i] *= inv_tokens;
        sum += pooled[i] * pooled[i];
    }
    sum = metal::simd_sum(sum);
    const float inv_l2 = sum == 0.0f ? 1.0f : metal::rsqrt(sum);
#pragma clang loop unroll(full)
    for (uint i = 0; i < values_per_lane; i++) {
        const uint dim = lane + i * 32;
        output[sequence * EI_METAL_N_EMBD + dim] = pooled[i] * inv_l2;
    }
}
