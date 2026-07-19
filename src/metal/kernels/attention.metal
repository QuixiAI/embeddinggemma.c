#include "embeddinggemma.metal"

// Model-specialized online-softmax attention. One simdgroup owns one
// query/head and keeps all 256 output channels in registers (8 per lane).
// K/V use one shared KV head; window=0 selects full bidirectional attention.
kernel void ei_attention_f32(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &n_tokens [[buffer(4)]],
    constant uint &window [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint query = group.x;
    const uint head = group.y;
    if (query >= n_tokens || head >= EI_METAL_N_HEAD) return;

    constexpr uint values_per_lane = EI_METAL_HEAD_DIM / 32;
    float q_values[values_per_lane];
    float out_values[values_per_lane];
#pragma clang loop unroll(full)
    for (uint i = 0; i < values_per_lane; i++) {
        const uint dim = lane + i * 32;
        q_values[i] = q[query * EI_METAL_N_EMBD + head * EI_METAL_HEAD_DIM + dim];
        out_values[i] = 0.0f;
    }

    uint first = 0;
    uint last = n_tokens;
    if (window != 0) {
        const uint half_window = window / 2;
        first = query > half_window ? query - half_window : 0;
        last = metal::min(n_tokens, query + half_window + 1);
    }

    float max_score = -INFINITY;
    float denominator = 0.0f;
    for (uint key = first; key < last; key++) {
        float score = 0.0f;
#pragma clang loop unroll(full)
        for (uint i = 0; i < values_per_lane; i++) {
            const uint dim = lane + i * 32;
            score += q_values[i] * k[key * EI_METAL_HEAD_DIM + dim];
        }
        score = metal::simd_sum(score);

        const float next_max = metal::max(max_score, score);
        const float alpha = metal::exp(max_score - next_max);
        const float beta = metal::exp(score - next_max);
        denominator = denominator * alpha + beta;
#pragma clang loop unroll(full)
        for (uint i = 0; i < values_per_lane; i++) {
            const uint dim = lane + i * 32;
            out_values[i] = out_values[i] * alpha
                + beta * v[key * EI_METAL_HEAD_DIM + dim];
        }
        max_score = next_max;
    }

    const float inv_denominator = 1.0f / denominator;
#pragma clang loop unroll(full)
    for (uint i = 0; i < values_per_lane; i++) {
        const uint dim = lane + i * 32;
        output[query * EI_METAL_N_EMBD + head * EI_METAL_HEAD_DIM + dim]
            = out_values[i] * inv_denominator;
    }
}

// Bandwidth-reduced variant: K/V are stored as FP16 but all score, softmax,
// and output accumulation remains FP32.
kernel void ei_attention_f16_kv(
    device const float *q [[buffer(0)]],
    device const half *k [[buffer(1)]],
    device const half *v [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &n_tokens [[buffer(4)]],
    constant uint &window [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint query = group.x;
    const uint head = group.y;
    if (query >= n_tokens || head >= EI_METAL_N_HEAD) return;

    constexpr uint values_per_lane = EI_METAL_HEAD_DIM / 32;
    float q_values[values_per_lane];
    float out_values[values_per_lane];
#pragma clang loop unroll(full)
    for (uint i = 0; i < values_per_lane; i++) {
        const uint dim = lane + i * 32;
        q_values[i] = q[query * EI_METAL_N_EMBD + head * EI_METAL_HEAD_DIM + dim];
        out_values[i] = 0.0f;
    }

    uint first = 0;
    uint last = n_tokens;
    if (window != 0) {
        const uint half_window = window / 2;
        first = query > half_window ? query - half_window : 0;
        last = metal::min(n_tokens, query + half_window + 1);
    }

    float max_score = -INFINITY;
    float denominator = 0.0f;
    for (uint key = first; key < last; key++) {
        float score = 0.0f;
#pragma clang loop unroll(full)
        for (uint i = 0; i < values_per_lane; i++) {
            const uint dim = lane + i * 32;
            score += q_values[i] * float(k[key * EI_METAL_HEAD_DIM + dim]);
        }
        score = metal::simd_sum(score);

        const float next_max = metal::max(max_score, score);
        const float alpha = metal::exp(max_score - next_max);
        const float beta = metal::exp(score - next_max);
        denominator = denominator * alpha + beta;
#pragma clang loop unroll(full)
        for (uint i = 0; i < values_per_lane; i++) {
            const uint dim = lane + i * 32;
            out_values[i] = out_values[i] * alpha
                + beta * float(v[key * EI_METAL_HEAD_DIM + dim]);
        }
        max_score = next_max;
    }

    const float inv_denominator = 1.0f / denominator;
#pragma clang loop unroll(full)
    for (uint i = 0; i < values_per_lane; i++) {
        const uint dim = lane + i * 32;
        output[query * EI_METAL_N_EMBD + head * EI_METAL_HEAD_DIM + dim]
            = out_values[i] * inv_denominator;
    }
}

// Packed bidirectional attention. Sequence IDs make the flattened token rows
// block diagonal without padding or cross-request attention.
kernel void ei_attention_f32_batch(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *output [[buffer(3)]],
    device const uint *offsets [[buffer(4)]],
    device const uint *sequence_ids [[buffer(5)]],
    constant uint &n_tokens [[buffer(6)]],
    constant uint &window [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint query = group.x;
    const uint head = group.y;
    if (query >= n_tokens || head >= EI_METAL_N_HEAD) return;

    constexpr uint values_per_lane = EI_METAL_HEAD_DIM / 32;
    float q_values[values_per_lane];
    float out_values[values_per_lane];
#pragma clang loop unroll(full)
    for (uint i = 0; i < values_per_lane; i++) {
        const uint dim = lane + i * 32;
        q_values[i] = q[query * EI_METAL_N_EMBD + head * EI_METAL_HEAD_DIM + dim];
        out_values[i] = 0.0f;
    }

    const uint sequence = sequence_ids[query];
    const uint sequence_start = offsets[sequence];
    const uint sequence_stop = offsets[sequence + 1];
    uint first = sequence_start;
    uint last = sequence_stop;
    if (window != 0) {
        const uint half_window = window / 2;
        first = query > sequence_start + half_window
            ? query - half_window : sequence_start;
        last = metal::min(sequence_stop, query + half_window + 1);
    }

    float max_score = -INFINITY;
    float denominator = 0.0f;
    for (uint key = first; key < last; key++) {
        float score = 0.0f;
#pragma clang loop unroll(full)
        for (uint i = 0; i < values_per_lane; i++) {
            const uint dim = lane + i * 32;
            score += q_values[i] * k[key * EI_METAL_HEAD_DIM + dim];
        }
        score = metal::simd_sum(score);
        const float next_max = metal::max(max_score, score);
        const float alpha = metal::exp(max_score - next_max);
        const float beta = metal::exp(score - next_max);
        denominator = denominator * alpha + beta;
#pragma clang loop unroll(full)
        for (uint i = 0; i < values_per_lane; i++) {
            const uint dim = lane + i * 32;
            out_values[i] = out_values[i] * alpha
                + beta * v[key * EI_METAL_HEAD_DIM + dim];
        }
        max_score = next_max;
    }

    const float inv_denominator = 1.0f / denominator;
#pragma clang loop unroll(full)
    for (uint i = 0; i < values_per_lane; i++) {
        const uint dim = lane + i * 32;
        output[query * EI_METAL_N_EMBD + head * EI_METAL_HEAD_DIM + dim]
            = out_values[i] * inv_denominator;
    }
}

kernel void ei_attention_f16_kv_batch(
    device const float *q [[buffer(0)]],
    device const half *k [[buffer(1)]],
    device const half *v [[buffer(2)]],
    device float *output [[buffer(3)]],
    device const uint *offsets [[buffer(4)]],
    device const uint *sequence_ids [[buffer(5)]],
    constant uint &n_tokens [[buffer(6)]],
    constant uint &window [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint query = group.x;
    const uint head = group.y;
    if (query >= n_tokens || head >= EI_METAL_N_HEAD) return;

    constexpr uint values_per_lane = EI_METAL_HEAD_DIM / 32;
    float q_values[values_per_lane];
    float out_values[values_per_lane];
#pragma clang loop unroll(full)
    for (uint i = 0; i < values_per_lane; i++) {
        const uint dim = lane + i * 32;
        q_values[i] = q[query * EI_METAL_N_EMBD + head * EI_METAL_HEAD_DIM + dim];
        out_values[i] = 0.0f;
    }

    const uint sequence = sequence_ids[query];
    const uint sequence_start = offsets[sequence];
    const uint sequence_stop = offsets[sequence + 1];
    uint first = sequence_start;
    uint last = sequence_stop;
    if (window != 0) {
        const uint half_window = window / 2;
        first = query > sequence_start + half_window
            ? query - half_window : sequence_start;
        last = metal::min(sequence_stop, query + half_window + 1);
    }

    float max_score = -INFINITY;
    float denominator = 0.0f;
    for (uint key = first; key < last; key++) {
        float score = 0.0f;
#pragma clang loop unroll(full)
        for (uint i = 0; i < values_per_lane; i++) {
            const uint dim = lane + i * 32;
            score += q_values[i] * float(k[key * EI_METAL_HEAD_DIM + dim]);
        }
        score = metal::simd_sum(score);
        const float next_max = metal::max(max_score, score);
        const float alpha = metal::exp(max_score - next_max);
        const float beta = metal::exp(score - next_max);
        denominator = denominator * alpha + beta;
#pragma clang loop unroll(full)
        for (uint i = 0; i < values_per_lane; i++) {
            const uint dim = lane + i * 32;
            out_values[i] = out_values[i] * alpha
                + beta * float(v[key * EI_METAL_HEAD_DIM + dim]);
        }
        max_score = next_max;
    }

    const float inv_denominator = 1.0f / denominator;
#pragma clang loop unroll(full)
    for (uint i = 0; i < values_per_lane; i++) {
        const uint dim = lane + i * 32;
        output[query * EI_METAL_N_EMBD + head * EI_METAL_HEAD_DIM + dim]
            = out_values[i] * inv_denominator;
    }
}
