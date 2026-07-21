#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// Model-specialized adaptation of llama.cpp's Metal 4 Q4_0 tensor matmul.
// Keeping the fixed layout here avoids the generic graph and quant machinery.
struct ei_metal4_block_q4_0 {
    half d;
    uchar qs[16];
};

static_assert(sizeof(ei_metal4_block_q4_0) == 18,
              "q4_0 block layout mismatch");

static inline half4x4 ei_metal4_dequant_q4_0(
    device const ei_metal4_block_q4_0 *block,
    short half_block) {
    device const ushort *packed =
        reinterpret_cast<device const ushort *>(block) + 1;
    const float d1 = half_block ? float(block->d) / 16.0f : float(block->d);
    const float d2 = d1 / 256.0f;
    const float bias = -8.0f * float(block->d);
    const ushort mask0 = half_block ? 0x00f0 : 0x000f;
    const ushort mask1 = mask0 << 8;
    half4x4 result;
#pragma clang loop unroll(full)
    for (short i = 0; i < 8; i++) {
        result[i / 2][2 * (i % 2)] = half(d1 * float(packed[i] & mask0) + bias);
        result[i / 2][2 * (i % 2) + 1] =
            half(d2 * float(packed[i] & mask1) + bias);
    }
    return result;
}

template <int token_tile>
static inline void ei_q4_0_f32_tensor_mm_impl(
    device const uchar *weights,
    device float *input,
    device float *output,
    uint n_rows,
    uint n_cols,
    uint n_tokens,
    threadgroup half *weight_tile,
    uint token_group,
    uint row_group,
    ushort tid) {
    constexpr int simd_width = 16;
    constexpr int k_chunks = 2;
    constexpr int k_tile = simd_width * k_chunks;
    constexpr int row_tile = simd_width * 2 * 2;
    constexpr int simdgroups = 4;
    constexpr int threads = 32 * simdgroups;

    const int row_start = int(row_group) * row_tile;
    const int token_start = int(token_group) * token_tile;
    const int blocks_per_row = int(n_cols) / 32;
    device const ei_metal4_block_q4_0 *packed_weights =
        reinterpret_cast<device const ei_metal4_block_q4_0 *>(weights);

    auto tensor_a = tensor(weight_tile,
                           dextents<int32_t, 2>(k_tile, row_tile));
    auto tensor_b = tensor(input,
                           dextents<int32_t, 2>(int(n_cols), int(n_tokens)),
                           array<int, 2>({1, int(n_cols)}));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            token_tile, row_tile, k_tile, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<simdgroups>> matmul;
    auto accumulator = matmul.template get_destination_cooperative_tensor<
        decltype(tensor_b), decltype(tensor_a), float>();

    for (int col_start = 0; col_start < int(n_cols); col_start += k_tile) {
        for (int work = int(tid); work < row_tile * k_chunks;
             work += threads) {
            const int local_row = work / k_chunks;
            const int chunk = work % k_chunks;
            const int row = row_start + local_row;
            const int col = col_start + chunk * 16;
            half4x4 values = half4x4(0.0h);
            if (row < int(n_rows)) {
                device const ei_metal4_block_q4_0 *row_weights =
                    packed_weights + row * blocks_per_row;
                values = ei_metal4_dequant_q4_0(
                    row_weights + col / 32, short((col / 16) & 1));
            }
#pragma clang loop unroll(full)
            for (short i = 0; i < 16; i++) {
                weight_tile[local_row * k_tile + chunk * 16 + i] =
                    values[i / 4][i % 4];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto tile_a = tensor_a.slice(0, 0);
        auto tile_b = tensor_b.slice(col_start, token_start);
        matmul.run(tile_b, tile_a, accumulator);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tensor_output = tensor(
        output, dextents<int32_t, 2>(int(n_rows), int(n_tokens)),
        array<int, 2>({1, int(n_rows)}));
    accumulator.store(tensor_output.slice(row_start, token_start));
}

kernel void ei_q4_0_f32_tensor_mm(
    device const uchar *weights [[buffer(0)]],
    device float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &n_rows [[buffer(3)]],
    constant uint &n_cols [[buffer(4)]],
    constant uint &n_tokens [[buffer(5)]],
    threadgroup half *weight_tile [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    ei_q4_0_f32_tensor_mm_impl<128>(weights, input, output,
        n_rows, n_cols, n_tokens, weight_tile, group.x, group.y, tid);
}

kernel void ei_q4_0_f32_tensor_mm_64(
    device const uchar *weights [[buffer(0)]],
    device float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &n_rows [[buffer(3)]],
    constant uint &n_cols [[buffer(4)]],
    constant uint &n_tokens [[buffer(5)]],
    threadgroup half *weight_tile [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    ei_q4_0_f32_tensor_mm_impl<64>(weights, input, output,
        n_rows, n_cols, n_tokens, weight_tile, group.x, group.y, tid);
}

kernel void ei_q4_0_f32_qkv_tensor_mm(
    device const uchar *q_weights [[buffer(0)]],
    device const uchar *k_weights [[buffer(1)]],
    device const uchar *v_weights [[buffer(2)]],
    device float *input [[buffer(3)]],
    device float *q_output [[buffer(4)]],
    device float *k_output [[buffer(5)]],
    device float *v_output [[buffer(6)]],
    constant uint &n_tokens [[buffer(7)]],
    threadgroup half *weight_tile [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    device const uchar *weights;
    device float *output;
    uint rows;
    uint row_group;
    if (group.y < 12) {
        weights = q_weights;
        output = q_output;
        rows = 768;
        row_group = group.y;
    } else if (group.y < 16) {
        weights = k_weights;
        output = k_output;
        rows = 256;
        row_group = group.y - 12;
    } else {
        weights = v_weights;
        output = v_output;
        rows = 256;
        row_group = group.y - 16;
    }
    ei_q4_0_f32_tensor_mm_impl<128>(weights, input, output,
        rows, 768, n_tokens, weight_tile, group.x, row_group, tid);
}

kernel void ei_q4_0_f32_qkv_tensor_mm_64(
    device const uchar *q_weights [[buffer(0)]],
    device const uchar *k_weights [[buffer(1)]],
    device const uchar *v_weights [[buffer(2)]],
    device float *input [[buffer(3)]],
    device float *q_output [[buffer(4)]],
    device float *k_output [[buffer(5)]],
    device float *v_output [[buffer(6)]],
    constant uint &n_tokens [[buffer(7)]],
    threadgroup half *weight_tile [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    device const uchar *weights;
    device float *output;
    uint rows;
    uint row_group;
    if (group.y < 12) {
        weights = q_weights;
        output = q_output;
        rows = 768;
        row_group = group.y;
    } else if (group.y < 16) {
        weights = k_weights;
        output = k_output;
        rows = 256;
        row_group = group.y - 12;
    } else {
        weights = v_weights;
        output = v_output;
        rows = 256;
        row_group = group.y - 16;
    }
    ei_q4_0_f32_tensor_mm_impl<64>(weights, input, output,
        rows, 768, n_tokens, weight_tile, group.x, row_group, tid);
}

kernel void ei_q4_0_f32_up_gate_tensor_mm(
    device const uchar *up_weights [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],
    device float *input [[buffer(2)]],
    device float *up_output [[buffer(3)]],
    device float *gate_output [[buffer(4)]],
    constant uint &n_tokens [[buffer(5)]],
    threadgroup half *weight_tile [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    const bool gate = group.y >= 18;
    ei_q4_0_f32_tensor_mm_impl<128>(
        gate ? gate_weights : up_weights, input,
        gate ? gate_output : up_output, 1152, 768, n_tokens,
        weight_tile, group.x, gate ? group.y - 18 : group.y, tid);
}

kernel void ei_q4_0_f32_up_gate_tensor_mm_64(
    device const uchar *up_weights [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],
    device float *input [[buffer(2)]],
    device float *up_output [[buffer(3)]],
    device float *gate_output [[buffer(4)]],
    constant uint &n_tokens [[buffer(5)]],
    threadgroup half *weight_tile [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    const bool gate = group.y >= 18;
    ei_q4_0_f32_tensor_mm_impl<64>(
        gate ? gate_weights : up_weights, input,
        gate ? gate_output : up_output, 1152, 768, n_tokens,
        weight_tile, group.x, gate ? group.y - 18 : group.y, tid);
}
