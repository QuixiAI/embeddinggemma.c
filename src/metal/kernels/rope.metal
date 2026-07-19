#include "embeddinggemma.metal"

// Fuses per-head RMSNorm, NEOX half-split RoPE, and the optional Q scale.
kernel void ei_qk_norm_rope_f32(
    device float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    constant uint &n_tokens [[buffer(2)]],
    constant uint &n_heads [[buffer(3)]],
    constant uint &row_stride [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    device const float2 *rope_table [[buffer(6)]],
    constant float &output_scale [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint head = group.x;
    const uint token = group.y;
    if (head >= n_heads || token >= n_tokens) return;

    const uint base = token * row_stride + head * EI_METAL_HEAD_DIM;
    float sum = 0.0f;
    for (uint col = lane; col < EI_METAL_HEAD_DIM; col += 32) {
        const float value = x[base + col];
        sum += value * value;
    }
    sum = metal::simd_sum(sum);
    const float inv = metal::rsqrt(sum / float(EI_METAL_HEAD_DIM) + eps) * output_scale;

    for (uint k = lane; k < EI_METAL_HEAD_DIM / 2; k += 32) {
        const float2 cs = rope_table[token * (EI_METAL_HEAD_DIM / 2) + k];
        const float c = cs.x;
        const float s = cs.y;
        const float x0 = x[base + k] * weight[k] * inv;
        const float x1 = x[base + k + EI_METAL_HEAD_DIM / 2]
            * weight[k + EI_METAL_HEAD_DIM / 2] * inv;
        x[base + k] = x0 * c - x1 * s;
        x[base + k + EI_METAL_HEAD_DIM / 2] = x0 * s + x1 * c;
    }
}

// Launch-fusion experiment for this model's fixed 3Q:1K head ratio.
kernel void ei_qk_norm_rope_qk_f32(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device const float *q_weight [[buffer(2)]],
    device const float *k_weight [[buffer(3)]],
    constant uint &n_tokens [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    device const float2 *rope_table [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint combined_head = group.x;
    const uint token = group.y;
    if (combined_head >= EI_METAL_N_HEAD + 1 || token >= n_tokens) return;
    const bool key_head = combined_head == EI_METAL_N_HEAD;
    const uint head = key_head ? 0 : combined_head;
    const uint row_stride = key_head ? EI_METAL_HEAD_DIM : EI_METAL_N_EMBD;
    device float *values = key_head ? k : q;
    device const float *weight = key_head ? k_weight : q_weight;
    const uint base = token * row_stride + head * EI_METAL_HEAD_DIM;
    float sum = 0.0f;
    for (uint col = lane; col < EI_METAL_HEAD_DIM; col += 32) {
        const float value = values[base + col];
        sum += value * value;
    }
    sum = metal::simd_sum(sum);
    const float scale = key_head ? 1.0f : 0.0625f;
    const float inv = metal::rsqrt(sum / float(EI_METAL_HEAD_DIM) + eps) * scale;
    for (uint dim = lane; dim < EI_METAL_HEAD_DIM / 2; dim += 32) {
        const float2 cs = rope_table[token * (EI_METAL_HEAD_DIM / 2) + dim];
        const float c = cs.x;
        const float s = cs.y;
        const float x0 = values[base + dim] * weight[dim] * inv;
        const float x1 = values[base + dim + EI_METAL_HEAD_DIM / 2]
            * weight[dim + EI_METAL_HEAD_DIM / 2] * inv;
        values[base + dim] = x0 * c - x1 * s;
        values[base + dim + EI_METAL_HEAD_DIM / 2] = x0 * s + x1 * c;
    }
}

kernel void ei_qk_norm_rope_q_f32_k_f16(
    device float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *q_weight [[buffer(2)]],
    device const float *k_weight [[buffer(3)]],
    constant uint &n_tokens [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    device const float2 *rope_table [[buffer(6)]],
    device half *k_output [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint combined_head = group.x;
    const uint token = group.y;
    if (combined_head >= EI_METAL_N_HEAD + 1 || token >= n_tokens) return;
    const bool key_head = combined_head == EI_METAL_N_HEAD;
    const uint head = key_head ? 0 : combined_head;
    const uint row_stride = key_head ? EI_METAL_HEAD_DIM : EI_METAL_N_EMBD;
    device const float *values = key_head ? k : q;
    device const float *weight = key_head ? k_weight : q_weight;
    const uint base = token * row_stride + head * EI_METAL_HEAD_DIM;
    float sum = 0.0f;
    for (uint col = lane; col < EI_METAL_HEAD_DIM; col += 32) {
        const float value = values[base + col];
        sum += value * value;
    }
    sum = metal::simd_sum(sum);
    const float scale = key_head ? 1.0f : 0.0625f;
    const float inv = metal::rsqrt(sum / float(EI_METAL_HEAD_DIM) + eps) * scale;
    for (uint dim = lane; dim < EI_METAL_HEAD_DIM / 2; dim += 32) {
        const float2 cs = rope_table[token * (EI_METAL_HEAD_DIM / 2) + dim];
        const float x0 = values[base + dim] * weight[dim] * inv;
        const float x1 = values[base + dim + EI_METAL_HEAD_DIM / 2]
            * weight[dim + EI_METAL_HEAD_DIM / 2] * inv;
        const float y0 = x0 * cs.x - x1 * cs.y;
        const float y1 = x0 * cs.y + x1 * cs.x;
        if (key_head) {
            const uint key_base = token * EI_METAL_HEAD_DIM;
            k_output[key_base + dim] = half(y0);
            k_output[key_base + dim + EI_METAL_HEAD_DIM / 2] = half(y1);
        } else {
            q[base + dim] = y0;
            q[base + dim + EI_METAL_HEAD_DIM / 2] = y1;
        }
    }
}

kernel void ei_qk_norm_rope_f32_batch(
    device float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    constant uint &n_tokens [[buffer(2)]],
    constant uint &n_heads [[buffer(3)]],
    constant uint &row_stride [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    device const float2 *rope_table [[buffer(6)]],
    constant float &output_scale [[buffer(7)]],
    device const uint *positions [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint head = group.x;
    const uint token = group.y;
    if (head >= n_heads || token >= n_tokens) return;

    const uint base = token * row_stride + head * EI_METAL_HEAD_DIM;
    float sum = 0.0f;
    for (uint col = lane; col < EI_METAL_HEAD_DIM; col += 32) {
        const float value = x[base + col];
        sum += value * value;
    }
    sum = metal::simd_sum(sum);
    const float inv = metal::rsqrt(sum / float(EI_METAL_HEAD_DIM) + eps) * output_scale;
    const uint position = positions[token];
    for (uint dim = lane; dim < EI_METAL_HEAD_DIM / 2; dim += 32) {
        const float2 cs = rope_table[position * (EI_METAL_HEAD_DIM / 2) + dim];
        const float x0 = x[base + dim] * weight[dim] * inv;
        const float x1 = x[base + dim + EI_METAL_HEAD_DIM / 2]
            * weight[dim + EI_METAL_HEAD_DIM / 2] * inv;
        x[base + dim] = x0 * cs.x - x1 * cs.y;
        x[base + dim + EI_METAL_HEAD_DIM / 2] = x0 * cs.y + x1 * cs.x;
    }
}

kernel void ei_qk_norm_rope_qk_f32_batch(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device const float *q_weight [[buffer(2)]],
    device const float *k_weight [[buffer(3)]],
    constant uint &n_tokens [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    device const float2 *rope_table [[buffer(6)]],
    device const uint *positions [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint combined_head = group.x;
    const uint token = group.y;
    if (combined_head >= EI_METAL_N_HEAD + 1 || token >= n_tokens) return;
    const bool key_head = combined_head == EI_METAL_N_HEAD;
    const uint head = key_head ? 0 : combined_head;
    const uint row_stride = key_head ? EI_METAL_HEAD_DIM : EI_METAL_N_EMBD;
    device float *values = key_head ? k : q;
    device const float *weight = key_head ? k_weight : q_weight;
    const uint base = token * row_stride + head * EI_METAL_HEAD_DIM;
    float sum = 0.0f;
    for (uint col = lane; col < EI_METAL_HEAD_DIM; col += 32) {
        const float value = values[base + col];
        sum += value * value;
    }
    sum = metal::simd_sum(sum);
    const float inv = metal::rsqrt(sum / float(EI_METAL_HEAD_DIM) + eps)
        * (key_head ? 1.0f : 0.0625f);
    const uint position = positions[token];
    for (uint dim = lane; dim < EI_METAL_HEAD_DIM / 2; dim += 32) {
        const float2 cs = rope_table[position * (EI_METAL_HEAD_DIM / 2) + dim];
        const float x0 = values[base + dim] * weight[dim] * inv;
        const float x1 = values[base + dim + EI_METAL_HEAD_DIM / 2]
            * weight[dim + EI_METAL_HEAD_DIM / 2] * inv;
        values[base + dim] = x0 * cs.x - x1 * cs.y;
        values[base + dim + EI_METAL_HEAD_DIM / 2] = x0 * cs.y + x1 * cs.x;
    }
}

kernel void ei_qk_norm_rope_q_f32_k_f16_batch(
    device float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *q_weight [[buffer(2)]],
    device const float *k_weight [[buffer(3)]],
    constant uint &n_tokens [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    device const float2 *rope_table [[buffer(6)]],
    device const uint *positions [[buffer(7)]],
    device half *k_output [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint combined_head = group.x;
    const uint token = group.y;
    if (combined_head >= EI_METAL_N_HEAD + 1 || token >= n_tokens) return;
    const bool key_head = combined_head == EI_METAL_N_HEAD;
    const uint head = key_head ? 0 : combined_head;
    const uint row_stride = key_head ? EI_METAL_HEAD_DIM : EI_METAL_N_EMBD;
    device const float *values = key_head ? k : q;
    device const float *weight = key_head ? k_weight : q_weight;
    const uint base = token * row_stride + head * EI_METAL_HEAD_DIM;
    float sum = 0.0f;
    for (uint col = lane; col < EI_METAL_HEAD_DIM; col += 32) {
        const float value = values[base + col];
        sum += value * value;
    }
    sum = metal::simd_sum(sum);
    const float inv = metal::rsqrt(sum / float(EI_METAL_HEAD_DIM) + eps)
        * (key_head ? 1.0f : 0.0625f);
    const uint position = positions[token];
    for (uint dim = lane; dim < EI_METAL_HEAD_DIM / 2; dim += 32) {
        const float2 cs = rope_table[position * (EI_METAL_HEAD_DIM / 2) + dim];
        const float x0 = values[base + dim] * weight[dim] * inv;
        const float x1 = values[base + dim + EI_METAL_HEAD_DIM / 2]
            * weight[dim + EI_METAL_HEAD_DIM / 2] * inv;
        const float y0 = x0 * cs.x - x1 * cs.y;
        const float y1 = x0 * cs.y + x1 * cs.x;
        if (key_head) {
            const uint key_base = token * EI_METAL_HEAD_DIM;
            k_output[key_base + dim] = half(y0);
            k_output[key_base + dim + EI_METAL_HEAD_DIM / 2] = half(y1);
        } else {
            q[base + dim] = y0;
            q[base + dim + EI_METAL_HEAD_DIM / 2] = y1;
        }
    }
}
