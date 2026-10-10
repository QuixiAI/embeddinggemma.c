#include <metal_stdlib>
using namespace metal;
struct G2P {
    uint n, k, m, type, heads, kv, d, window, cols, mode, layer;
    float scale, lo, hi;
};
// Adapted from the original profile's tiled attention. Packed sequence bounds
// mask both full and symmetric sliding attention; softmax/accumulators are FP32.
template <uint head_dim, typename Accumulator>
inline void g2_flash_impl(device const float *q, device const half *k, device const half *v,
                          device float *output, constant G2P &p, device const uint2 *bounds,
                          uint3 group, ushort lane, ushort simdgroup, ushort tid,
                          threadgroup half *q_tile, threadgroup float *scores, Accumulator accum,
                          threadgroup float *row_max, threadgroup float *row_sum) {
    constexpr uint queries_per_group = 8;
    constexpr uint keys_per_group = 32;

    const uint q_stride = p.heads * head_dim;
    const uint kv_stride = p.kv * head_dim;
    const uint n_tokens = p.n, window = p.window * 2;
    constexpr uint simdgroups = 4;

    const uint query_start = group.x * queries_per_group;
    const uint head = group.y;
    const uint kh = head / (p.heads / p.kv);

    for (uint item = tid; item < queries_per_group * head_dim; item += 32 * simdgroups) {
        const uint local_query = item / head_dim;
        const uint dim = item % head_dim;
        const uint query = query_start + local_query;
        q_tile[item] = query < n_tokens ? half(q[query * q_stride + head * head_dim + dim]) : 0.0h;
        accum[item] = 0.0f;
    }
    if (tid < queries_per_group) {
        row_max[tid] = -INFINITY;
        row_sum[tid] = 0.0f;
    }
    threadgroup_barrier(head_dim == 512 ? mem_flags::mem_threadgroup | mem_flags::mem_device
                                        : mem_flags::mem_threadgroup);

    uint key_begin = 0;
    uint key_end = n_tokens;
    if (window != 0) {
        const uint half_window = window / 2;
        key_begin = query_start > half_window ? query_start - half_window : 0;
        key_begin = (key_begin / keys_per_group) * keys_per_group;
        key_end = min(n_tokens, query_start + queries_per_group + half_window);
    }

    for (uint key_start = key_begin; key_start < key_end; key_start += keys_per_group) {
        simdgroup_float8x8 qk = make_filled_simdgroup_matrix<float, 8>(0.0f);
#pragma unroll(8)
        for (ushort pair = 0; pair < head_dim / 16; pair++) {
            simdgroup_half8x8 mq[2];
            simdgroup_half8x8 mk[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mq[0], q_tile + 16 * pair, head_dim);
            simdgroup_load(mq[1], q_tile + 16 * pair + 8, head_dim);
            device const half *key_ptr =
                k + (key_start + 8 * simdgroup) * kv_stride + kh * head_dim + 16 * pair;
            simdgroup_load(mk[0], key_ptr, kv_stride, 0, true);
            simdgroup_load(mk[1], key_ptr + 8, kv_stride, 0, true);
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(qk, mq[0], mk[0], qk);
            simdgroup_multiply_accumulate(qk, mq[1], mk[1], qk);
        }
        simdgroup_store(qk, scores + 8 * simdgroup, keys_per_group, 0, false);
        threadgroup_barrier(head_dim == 512 ? mem_flags::mem_threadgroup | mem_flags::mem_device
                                            : mem_flags::mem_threadgroup);

        // Each simdgroup owns two query rows for online softmax and rescales
        // those rows' running output before the P*V update.
#pragma clang loop unroll(full)
        for (ushort local = 0; local < 2; local++) {
            const ushort row = 2 * simdgroup + local;
            const uint query = query_start + row;
            const uint key_index = key_start + lane;
            const uint half_window = window / 2;
            const bool valid =
                query < n_tokens && key_index < n_tokens &&
                (!p.layer || (key_index >= bounds[query].x && key_index < bounds[query].y)) &&
                (window == 0 ||
                 (key_index + half_window >= query && key_index <= query + half_window));
            const float score = valid ? scores[row * keys_per_group + lane] : -INFINITY;

            if (query < n_tokens) {
                const float previous_max = row_max[row];
                const float next_max = max(previous_max, simd_max(score));
                const float alpha = next_max == -INFINITY ? 0.0f : exp(previous_max - next_max);
                const float probability = valid ? exp(score - next_max) : 0.0f;
                const float next_sum = row_sum[row] * alpha + simd_sum(probability);
                scores[row * keys_per_group + lane] = probability;
                for (uint dim = lane; dim < head_dim; dim += 32) {
                    accum[row * head_dim + dim] *= alpha;
                }
                if (lane == 0) {
                    row_max[row] = next_max;
                    row_sum[row] = next_sum;
                }
            } else {
                scores[row * keys_per_group + lane] = 0.0f;
            }
        }
        threadgroup_barrier(head_dim == 512 ? mem_flags::mem_threadgroup | mem_flags::mem_device
                                            : mem_flags::mem_threadgroup);

        // Four simdgroups split the 256 output dimensions. Each accumulates
        // all four 8-key probability tiles for its 64-column output stripe.
#pragma clang loop unroll(full)
        for (ushort tile = 0; tile < head_dim / 32; tile++) {
            const ushort dim = simdgroup * (head_dim / 4) + tile * 8;
            simdgroup_float8x8 out_matrix;
            simdgroup_load(out_matrix, accum + dim, head_dim, 0, false);
#pragma clang loop unroll(full)
            for (ushort key_tile = 0; key_tile < 4; key_tile++) {
                simdgroup_float8x8 probability_matrix;
                simdgroup_half8x8 value_matrix;
                simdgroup_load(probability_matrix, scores + key_tile * 8, keys_per_group, 0, false);
                device const half *value_ptr =
                    v + (key_start + key_tile * 8) * kv_stride + kh * head_dim + dim;
                simdgroup_load(value_matrix, value_ptr, kv_stride, 0, false);
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_multiply_accumulate(out_matrix, probability_matrix, value_matrix,
                                              out_matrix);
            }
            simdgroup_store(out_matrix, accum + dim, head_dim, 0, false);
        }
        threadgroup_barrier(head_dim == 512 ? mem_flags::mem_threadgroup | mem_flags::mem_device
                                            : mem_flags::mem_threadgroup);
    }

    for (uint item = tid; item < queries_per_group * head_dim; item += 32 * simdgroups) {
        const uint local_query = item / head_dim;
        const uint dim = item % head_dim;
        const uint query = query_start + local_query;
        if (query < n_tokens) {
            const float scale = row_sum[local_query] == 0.0f ? 0.0f : 1.0f / row_sum[local_query];
            output[query * q_stride + head * head_dim + dim] = accum[item] * scale;
        }
    }
}

kernel void g2_flash_64(device const float *q [[buffer(0)]], device const half *k [[buffer(1)]],
                        device const half *v [[buffer(2)]], device float *out [[buffer(3)]],
                        constant G2P &p [[buffer(4)]], device const uint2 *bounds [[buffer(5)]],
                        uint3 group [[threadgroup_position_in_grid]],
                        ushort lane [[thread_index_in_simdgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half qt[8 * 64];
    threadgroup float scores[8 * 32], acc[8 * 64], mx[8], sum[8];
    g2_flash_impl<64>(q, k, v, out, p, bounds, group, lane, sg, tid, qt, scores, acc, mx, sum);
}

kernel void g2_flash_256(device const float *q [[buffer(0)]], device const half *k [[buffer(1)]],
                         device const half *v [[buffer(2)]], device float *out [[buffer(3)]],
                         constant G2P &p [[buffer(4)]], device const uint2 *bounds [[buffer(5)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         ushort lane [[thread_index_in_simdgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half qt[8 * 256];
    threadgroup float scores[8 * 32], acc[8 * 256], mx[8], sum[8];
    g2_flash_impl<256>(q, k, v, out, p, bounds, group, lane, sg, tid, qt, scores, acc, mx, sum);
}

kernel void g2_flash_512(device const float *q [[buffer(0)]], device const half *k [[buffer(1)]],
                         device const half *v [[buffer(2)]], device float *out [[buffer(3)]],
                         constant G2P &p [[buffer(4)]], device const uint2 *bounds [[buffer(5)]],
                         device float *scratch [[buffer(6)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         ushort lane [[thread_index_in_simdgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half qt[8 * 512];
    // Keep threadgroup storage below 16 KiB, including room for shader
    // validation's shadow memory. Each group owns a disjoint accumulator tile.
    threadgroup float scores[8 * 32], mx[8], sum[8];
    device float *acc = scratch + (group.y * ((p.n + 7) / 8) + group.x) * 8 * 512;
    g2_flash_impl<512>(q, k, v, out, p, bounds, group, lane, sg, tid, qt, scores, acc, mx, sum);
}
