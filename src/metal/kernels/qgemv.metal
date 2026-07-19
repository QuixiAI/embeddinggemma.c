#include "embeddinggemma.metal"

// Latency route for very short inputs: one simdgroup per output row/token.
kernel void ei_q4_0_f32_gemv(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &n_rows [[buffer(3)]],
    constant uint &n_cols [[buffer(4)]],
    constant uint &n_tokens [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x;
    const uint token = group.y;
    if (row >= n_rows || token >= n_tokens) return;

    const uint blocks_per_row = n_cols / EI_METAL_QK;
    device const ei_metal_block_q4_0 *row_weights =
        (device const ei_metal_block_q4_0 *)weights + row * blocks_per_row;
    device const float *x = input + token * n_cols;
    const uint block_offset = lane >> 1;
    const uint byte_start = (lane & 1) * 8;
    float sum = 0.0f;
    for (uint block = block_offset; block < blocks_per_row; block += 16) {
        const float d = float(row_weights[block].d);
        device const uchar *qs = row_weights[block].qs + byte_start;
        const uint x0 = block * EI_METAL_QK + byte_start;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            const uchar code = qs[i];
            sum += d * float(int(code & 0x0f) - 8) * x[x0 + i];
            sum += d * float(int(code >> 4) - 8) * x[x0 + i + 16];
        }
    }
    sum = metal::simd_sum(sum);
    if (lane == 0) output[token * n_rows + row] = sum;
}

// llama.cpp-style Q4_0 GEMV: one simdgroup computes several output rows and
// reuses its register-cached activation values across them. Wider variants are
// compiled for tuning; R4 remains the production default.
template <uint rows_per_simdgroup>
static inline void ei_q4_0_f32_gemv_rows(
    device const uchar *weights,
    device const float *input,
    device float *output,
    constant uint &n_rows,
    constant uint &n_cols,
    constant uint &n_tokens,
    uint3 group,
    uint lane) {
    const uint first_row = group.x * rows_per_simdgroup;
    const uint token = group.y;
    if (first_row >= n_rows || token >= n_tokens) return;

    const uint blocks_per_row = n_cols / EI_METAL_QK;
    device const ei_metal_block_q4_0 *packed =
        (device const ei_metal_block_q4_0 *)weights;
    device const float *x = input + token * n_cols;
    const uint block_offset = lane >> 1;
    const uint byte_start = (lane & 1) * 8;

    float sums[rows_per_simdgroup] = { 0.0f };
    for (uint block = block_offset; block < blocks_per_row; block += 16) {
        const uint x0 = block * EI_METAL_QK + byte_start;
        float x_lo[8];
        float x_hi[8];
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            x_lo[i] = x[x0 + i];
            x_hi[i] = x[x0 + i + 16];
        }
#pragma clang loop unroll(full)
        for (uint local_row = 0; local_row < rows_per_simdgroup; local_row++) {
            const uint row = first_row + local_row;
            if (row >= n_rows) continue;
            device const ei_metal_block_q4_0 &q =
                packed[row * blocks_per_row + block];
            const float d = float(q.d);
            device const uchar *qs = q.qs + byte_start;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8; i++) {
                const uchar code = qs[i];
                sums[local_row] += d * float(int(code & 0x0f) - 8) * x_lo[i];
                sums[local_row] += d * float(int(code >> 4) - 8) * x_hi[i];
            }
        }
    }
#pragma clang loop unroll(full)
    for (uint local_row = 0; local_row < rows_per_simdgroup; local_row++) {
        const float sum = metal::simd_sum(sums[local_row]);
        const uint row = first_row + local_row;
        if (lane == 0 && row < n_rows) output[token * n_rows + row] = sum;
    }
}

#define EI_Q4_0_F32_GEMV_ROWS(rows)                                          \
kernel void ei_q4_0_f32_gemv_r##rows(                                       \
    device const uchar *weights [[buffer(0)]],                               \
    device const float *input [[buffer(1)]],                                 \
    device float *output [[buffer(2)]],                                      \
    constant uint &n_rows [[buffer(3)]],                                     \
    constant uint &n_cols [[buffer(4)]],                                     \
    constant uint &n_tokens [[buffer(5)]],                                   \
    uint3 group [[threadgroup_position_in_grid]],                            \
    uint lane [[thread_index_in_simdgroup]]) {                               \
    ei_q4_0_f32_gemv_rows<rows>(weights, input, output, n_rows, n_cols,       \
                                n_tokens, group, lane);                       \
}

EI_Q4_0_F32_GEMV_ROWS(4)
#undef EI_Q4_0_F32_GEMV_ROWS

// Multi-token prefill route. A 256-thread group computes 32 output rows for
// eight tokens and reuses each dequantized weight tile across those tokens.
kernel void ei_q4_0_f32_gemm(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &n_rows [[buffer(3)]],
    constant uint &n_cols [[buffer(4)]],
    constant uint &n_tokens [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint tile_rows = 32;
    constexpr uint tile_tokens = 8;
    threadgroup float weight_tile[tile_rows][EI_METAL_QK];
    threadgroup float input_tile[tile_tokens][EI_METAL_QK];

    const uint local_row = tid & 31;
    const uint local_token = tid >> 5;
    const uint row = group.x * tile_rows + local_row;
    const uint token = group.y * tile_tokens + local_token;
    const uint blocks_per_row = n_cols / EI_METAL_QK;
    device const ei_metal_block_q4_0 *packed =
        (device const ei_metal_block_q4_0 *)weights;
    float sum = 0.0f;

    for (uint block = 0; block < blocks_per_row; block++) {
        for (uint item = tid; item < tile_rows * EI_METAL_QK; item += 256) {
            const uint load_row = item / EI_METAL_QK;
            const uint col = item - load_row * EI_METAL_QK;
            const uint global_row = group.x * tile_rows + load_row;
            float value = 0.0f;
            if (global_row < n_rows) {
                device const ei_metal_block_q4_0 &q =
                    packed[global_row * blocks_per_row + block];
                const uchar byte = q.qs[col < 16 ? col : col - 16];
                const int quant = col < 16
                    ? int(byte & 0x0f) - 8
                    : int(byte >> 4) - 8;
                value = float(q.d) * float(quant);
            }
            weight_tile[load_row][col] = value;
        }

        const uint input_col = local_row;
        input_tile[local_token][input_col] = token < n_tokens
            ? input[token * n_cols + block * EI_METAL_QK + input_col]
            : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < n_rows && token < n_tokens) {
#pragma clang loop unroll(full)
            for (uint col = 0; col < EI_METAL_QK; col++) {
                sum += weight_tile[local_row][col] * input_tile[local_token][col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < n_rows && token < n_tokens) output[token * n_rows + row] = sum;
}

// Integer Q4_0 x Q8_0 projection, matching the CPU activation-quantized path.
kernel void ei_q4_0_q8_0_gemv(
    device const uchar *weights [[buffer(0)]],
    device const uchar *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &n_rows [[buffer(3)]],
    constant uint &n_cols [[buffer(4)]],
    constant uint &n_tokens [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x;
    const uint token = group.y;
    if (row >= n_rows || token >= n_tokens) return;

    const uint blocks_per_row = n_cols / EI_METAL_QK;
    device const ei_metal_block_q4_0 *row_weights =
        (device const ei_metal_block_q4_0 *)weights + row * blocks_per_row;
    device const ei_metal_block_q8_0 *x =
        (device const ei_metal_block_q8_0 *)input + token * blocks_per_row;
    const uint block_offset = lane >> 1;
    const uint byte_start = (lane & 1) * 8;

    float sum = 0.0f;
    for (uint block = block_offset; block < blocks_per_row; block += 16) {
        device const uchar *q4 = row_weights[block].qs + byte_start;
        device const char *q8 = x[block].qs + byte_start;
        int isum = 0;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            const uchar packed = q4[i];
            isum += (int(packed & 0x0f) - 8) * int(q8[i]);
            isum += (int(packed >> 4) - 8) * int(q8[i + 16]);
        }
        sum += float(isum) * float(row_weights[block].d) * float(x[block].d);
    }
    sum = metal::simd_sum(sum);
    if (lane == 0) output[token * n_rows + row] = sum;
}

// One simdgroup quantizes one 32-value activation block to GGUF Q8_0.
kernel void ei_quantize_q8_0_f32(
    device const float *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant uint &n_cols [[buffer(2)]],
    constant uint &n_tokens [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint block_index = group.x;
    const uint token = group.y;
    const uint blocks_per_row = n_cols / EI_METAL_QK;
    if (block_index >= blocks_per_row || token >= n_tokens) return;

    const float value = input[token * n_cols + block_index * EI_METAL_QK + lane];
    const float amax = metal::simd_max(metal::fabs(value));
    const float d = amax / 127.0f;
    const int quant = d == 0.0f
        ? 0
        : metal::clamp(int(metal::round(value / d)), -128, 127);
    device ei_metal_block_q8_0 *blocks = (device ei_metal_block_q8_0 *)output;
    device ei_metal_block_q8_0 &block = blocks[token * blocks_per_row + block_index];
    if (lane == 0) block.d = half(d);
    block.qs[lane] = char(quant);
}

kernel void ei_q4_0_f32_qkv_gemv(
    device const uchar *q_weights [[buffer(0)]],
    device const uchar *k_weights [[buffer(1)]],
    device const uchar *v_weights [[buffer(2)]],
    device const float *input [[buffer(3)]],
    device float *q_output [[buffer(4)]],
    device float *k_output [[buffer(5)]],
    device float *v_output [[buffer(6)]],
    constant uint &n_tokens [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint combined_row = group.x;
    const uint token = group.y;
    if (combined_row >= 1280 || token >= n_tokens) return;

    device const uchar *weight_bytes;
    device float *output;
    uint row;
    uint rows;
    if (combined_row < 768) {
        weight_bytes = q_weights;
        output = q_output;
        row = combined_row;
        rows = 768;
    } else if (combined_row < 1024) {
        weight_bytes = k_weights;
        output = k_output;
        row = combined_row - 768;
        rows = 256;
    } else {
        weight_bytes = v_weights;
        output = v_output;
        row = combined_row - 1024;
        rows = 256;
    }
    constexpr uint blocks_per_row = EI_METAL_N_EMBD / EI_METAL_QK;
    device const ei_metal_block_q4_0 *packed =
        (device const ei_metal_block_q4_0 *)weight_bytes + row * blocks_per_row;
    device const float *x = input + token * EI_METAL_N_EMBD;
    const uint block_offset = lane >> 1;
    const uint byte_start = (lane & 1) * 8;
    float sum = 0.0f;
    for (uint block = block_offset; block < blocks_per_row; block += 16) {
        const float d = float(packed[block].d);
        device const uchar *qs = packed[block].qs + byte_start;
        const uint x0 = block * EI_METAL_QK + byte_start;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            const uchar code = qs[i];
            sum += d * float(int(code & 0x0f) - 8) * x[x0 + i];
            sum += d * float(int(code >> 4) - 8) * x[x0 + i + 16];
        }
    }
    sum = metal::simd_sum(sum);
    if (lane == 0) output[token * rows + row] = sum;
}

// Short-input route: Q, K, and V share the first 256 output rows. Computing
// those matching rows in one simdgroup reuses each activation load three ways.
kernel void ei_q4_0_f32_qkv_gemv_triple(
    device const uchar *q_weights [[buffer(0)]],
    device const uchar *k_weights [[buffer(1)]],
    device const uchar *v_weights [[buffer(2)]],
    device const float *input [[buffer(3)]],
    device float *q_output [[buffer(4)]],
    device float *k_output [[buffer(5)]],
    device float *v_output [[buffer(6)]],
    constant uint &n_tokens [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x;
    const uint token = group.y;
    if (row >= EI_METAL_N_EMBD || token >= n_tokens) return;

    constexpr uint blocks_per_row = EI_METAL_N_EMBD / EI_METAL_QK;
    const bool has_kv = row < EI_METAL_HEAD_DIM;
    const uint kv_row = has_kv ? row : 0;
    device const ei_metal_block_q4_0 *q =
        (device const ei_metal_block_q4_0 *)q_weights + row * blocks_per_row;
    device const ei_metal_block_q4_0 *k =
        (device const ei_metal_block_q4_0 *)k_weights + kv_row * blocks_per_row;
    device const ei_metal_block_q4_0 *v =
        (device const ei_metal_block_q4_0 *)v_weights + kv_row * blocks_per_row;
    device const float *x = input + token * EI_METAL_N_EMBD;
    const uint block_offset = lane >> 1;
    const uint byte_start = (lane & 1) * 8;
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    float v_sum = 0.0f;
    for (uint block = block_offset; block < blocks_per_row; block += 16) {
        const uint x0 = block * EI_METAL_QK + byte_start;
        device const uchar *q_qs = q[block].qs + byte_start;
        const float q_d = float(q[block].d);
        device const uchar *k_qs = has_kv ? k[block].qs + byte_start : q_qs;
        device const uchar *v_qs = has_kv ? v[block].qs + byte_start : q_qs;
        const float k_d = has_kv ? float(k[block].d) : 0.0f;
        const float v_d = has_kv ? float(v[block].d) : 0.0f;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            const float x_lo = x[x0 + i];
            const float x_hi = x[x0 + i + 16];
            const uchar q_code = q_qs[i];
            q_sum += q_d * (float(int(q_code & 0x0f) - 8) * x_lo +
                            float(int(q_code >> 4) - 8) * x_hi);
            if (has_kv) {
                const uchar k_code = k_qs[i];
                const uchar v_code = v_qs[i];
                k_sum += k_d * (float(int(k_code & 0x0f) - 8) * x_lo +
                                float(int(k_code >> 4) - 8) * x_hi);
                v_sum += v_d * (float(int(v_code & 0x0f) - 8) * x_lo +
                                float(int(v_code >> 4) - 8) * x_hi);
            }
        }
    }
    q_sum = metal::simd_sum(q_sum);
    k_sum = metal::simd_sum(k_sum);
    v_sum = metal::simd_sum(v_sum);
    if (lane == 0) {
        q_output[token * EI_METAL_N_EMBD + row] = q_sum;
        if (has_kv) {
            k_output[token * EI_METAL_HEAD_DIM + row] = k_sum;
            v_output[token * EI_METAL_HEAD_DIM + row] = v_sum;
        }
    }
}

template <uint rows_per_simdgroup, typename V>
static inline void ei_q4_0_f32_qkv_gemv_rows(
    device const uchar *q_weights,
    device const uchar *k_weights,
    device const uchar *v_weights,
    device const float *input,
    device float *q_output,
    device float *k_output,
    device V *v_output,
    constant uint &n_tokens,
    uint3 group,
    uint lane) {
    const uint combined_row = group.x * rows_per_simdgroup;
    const uint token = group.y;
    if (combined_row >= 1280 || token >= n_tokens) return;

    device const uchar *weight_bytes;
    uint row;
    if (combined_row < 768) {
        weight_bytes = q_weights;
        row = combined_row;
    } else if (combined_row < 1024) {
        weight_bytes = k_weights;
        row = combined_row - 768;
    } else {
        weight_bytes = v_weights;
        row = combined_row - 1024;
    }

    constexpr uint blocks_per_row = EI_METAL_N_EMBD / EI_METAL_QK;
    device const ei_metal_block_q4_0 *packed =
        (device const ei_metal_block_q4_0 *)weight_bytes + row * blocks_per_row;
    device const float *x = input + token * EI_METAL_N_EMBD;
    const uint block_offset = lane >> 1;
    const uint byte_start = (lane & 1) * 8;
    float sums[rows_per_simdgroup] = { 0.0f };
    for (uint block = block_offset; block < blocks_per_row; block += 16) {
        const uint x0 = block * EI_METAL_QK + byte_start;
        float x_lo[8];
        float x_hi[8];
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            x_lo[i] = x[x0 + i];
            x_hi[i] = x[x0 + i + 16];
        }
#pragma clang loop unroll(full)
        for (uint local_row = 0; local_row < rows_per_simdgroup; local_row++) {
            device const ei_metal_block_q4_0 &q =
                packed[local_row * blocks_per_row + block];
            const float d = float(q.d);
            device const uchar *qs = q.qs + byte_start;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8; i++) {
                const uchar code = qs[i];
                sums[local_row] += d * float(int(code & 0x0f) - 8) * x_lo[i];
                sums[local_row] += d * float(int(code >> 4) - 8) * x_hi[i];
            }
        }
    }
#pragma clang loop unroll(full)
    for (uint local_row = 0; local_row < rows_per_simdgroup; local_row++) {
        const float sum = metal::simd_sum(sums[local_row]);
        if (lane == 0) {
            if (combined_row < 768) {
                q_output[token * EI_METAL_N_EMBD + row + local_row] = sum;
            } else if (combined_row < 1024) {
                k_output[token * EI_METAL_HEAD_DIM + row + local_row] = sum;
            } else {
                v_output[token * EI_METAL_HEAD_DIM + row + local_row] = V(sum);
            }
        }
    }
}

#define EI_Q4_0_F32_QKV_GEMV_ROWS(rows)                                      \
kernel void ei_q4_0_f32_qkv_gemv_r##rows(                                   \
    device const uchar *q_weights [[buffer(0)]],                             \
    device const uchar *k_weights [[buffer(1)]],                             \
    device const uchar *v_weights [[buffer(2)]],                             \
    device const float *input [[buffer(3)]],                                 \
    device float *q_output [[buffer(4)]],                                    \
    device float *k_output [[buffer(5)]],                                    \
    device float *v_output [[buffer(6)]],                                    \
    constant uint &n_tokens [[buffer(7)]],                                   \
    uint3 group [[threadgroup_position_in_grid]],                            \
    uint lane [[thread_index_in_simdgroup]]) {                               \
    ei_q4_0_f32_qkv_gemv_rows<rows, float>(q_weights, k_weights, v_weights, \
        input,                                                               \
        q_output, k_output, v_output, n_tokens, group, lane);                \
}

EI_Q4_0_F32_QKV_GEMV_ROWS(4)
#undef EI_Q4_0_F32_QKV_GEMV_ROWS

kernel void ei_q4_0_f32_qkv_gemv_r4_v_f16(
    device const uchar *q_weights [[buffer(0)]],
    device const uchar *k_weights [[buffer(1)]],
    device const uchar *v_weights [[buffer(2)]],
    device const float *input [[buffer(3)]],
    device float *q_output [[buffer(4)]],
    device float *k_output [[buffer(5)]],
    device half *v_output [[buffer(6)]],
    constant uint &n_tokens [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    ei_q4_0_f32_qkv_gemv_rows<4, half>(q_weights, k_weights, v_weights, input,
        q_output, k_output, v_output, n_tokens, group, lane);
}

kernel void ei_q4_0_f32_qkv_gemm(
    device const uchar *q_weights [[buffer(0)]],
    device const uchar *k_weights [[buffer(1)]],
    device const uchar *v_weights [[buffer(2)]],
    device const float *input [[buffer(3)]],
    device float *q_output [[buffer(4)]],
    device float *k_output [[buffer(5)]],
    device float *v_output [[buffer(6)]],
    constant uint &n_tokens [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint tile_rows = 32;
    constexpr uint tile_tokens = 8;
    threadgroup float weight_tile[tile_rows][EI_METAL_QK];
    threadgroup float input_tile[tile_tokens][EI_METAL_QK];

    device const uchar *weight_bytes;
    device float *output;
    uint matrix_block;
    uint rows;
    if (group.x < 24) {
        weight_bytes = q_weights;
        output = q_output;
        matrix_block = group.x;
        rows = 768;
    } else if (group.x < 32) {
        weight_bytes = k_weights;
        output = k_output;
        matrix_block = group.x - 24;
        rows = 256;
    } else {
        weight_bytes = v_weights;
        output = v_output;
        matrix_block = group.x - 32;
        rows = 256;
    }

    const uint local_row = tid & 31;
    const uint local_token = tid >> 5;
    const uint row = matrix_block * tile_rows + local_row;
    const uint token = group.y * tile_tokens + local_token;
    constexpr uint blocks_per_row = EI_METAL_N_EMBD / EI_METAL_QK;
    device const ei_metal_block_q4_0 *packed =
        (device const ei_metal_block_q4_0 *)weight_bytes;
    float sum = 0.0f;
    for (uint block = 0; block < blocks_per_row; block++) {
        for (uint item = tid; item < tile_rows * EI_METAL_QK; item += 256) {
            const uint load_row = item / EI_METAL_QK;
            const uint col = item - load_row * EI_METAL_QK;
            const uint global_row = matrix_block * tile_rows + load_row;
            device const ei_metal_block_q4_0 &q =
                packed[global_row * blocks_per_row + block];
            const uchar byte = q.qs[col < 16 ? col : col - 16];
            const int quant = col < 16 ? int(byte & 0x0f) - 8 : int(byte >> 4) - 8;
            weight_tile[load_row][col] = float(q.d) * float(quant);
        }
        input_tile[local_token][local_row] = token < n_tokens
            ? input[token * EI_METAL_N_EMBD + block * EI_METAL_QK + local_row]
            : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (token < n_tokens) {
#pragma clang loop unroll(full)
            for (uint col = 0; col < EI_METAL_QK; col++) {
                sum += weight_tile[local_row][col] * input_tile[local_token][col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (token < n_tokens) output[token * rows + row] = sum;
}

kernel void ei_q4_0_f32_up_gate_gemv(
    device const uchar *up_weights [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *up_output [[buffer(3)]],
    device float *gate_output [[buffer(4)]],
    constant uint &n_tokens [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint combined_row = group.x;
    const uint token = group.y;
    if (combined_row >= 2304 || token >= n_tokens) return;
    const bool gate = combined_row >= EI_METAL_N_FF;
    const uint row = gate ? combined_row - EI_METAL_N_FF : combined_row;
    device const uchar *weight_bytes = gate ? gate_weights : up_weights;
    device float *output = gate ? gate_output : up_output;
    constexpr uint blocks_per_row = EI_METAL_N_EMBD / EI_METAL_QK;
    device const ei_metal_block_q4_0 *packed =
        (device const ei_metal_block_q4_0 *)weight_bytes + row * blocks_per_row;
    device const float *x = input + token * EI_METAL_N_EMBD;
    const uint block_offset = lane >> 1;
    const uint byte_start = (lane & 1) * 8;
    float sum = 0.0f;
    for (uint block = block_offset; block < blocks_per_row; block += 16) {
        const float d = float(packed[block].d);
        device const uchar *qs = packed[block].qs + byte_start;
        const uint x0 = block * EI_METAL_QK + byte_start;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            const uchar code = qs[i];
            sum += d * float(int(code & 0x0f) - 8) * x[x0 + i];
            sum += d * float(int(code >> 4) - 8) * x[x0 + i + 16];
        }
    }
    sum = metal::simd_sum(sum);
    if (lane == 0) output[token * EI_METAL_N_FF + row] = sum;
}

template <uint rows_per_simdgroup>
static inline void ei_q4_0_f32_up_gate_gemv_rows(
    device const uchar *up_weights,
    device const uchar *gate_weights,
    device const float *input,
    device float *up_output,
    device float *gate_output,
    constant uint &n_tokens,
    uint3 group,
    uint lane) {
    const uint combined_row = group.x * rows_per_simdgroup;
    const uint token = group.y;
    if (combined_row >= 2304 || token >= n_tokens) return;
    const bool gate = combined_row >= EI_METAL_N_FF;
    const uint row = gate ? combined_row - EI_METAL_N_FF : combined_row;
    device const uchar *weight_bytes = gate ? gate_weights : up_weights;
    device float *output = gate ? gate_output : up_output;
    constexpr uint blocks_per_row = EI_METAL_N_EMBD / EI_METAL_QK;
    device const ei_metal_block_q4_0 *packed =
        (device const ei_metal_block_q4_0 *)weight_bytes + row * blocks_per_row;
    device const float *x = input + token * EI_METAL_N_EMBD;
    const uint block_offset = lane >> 1;
    const uint byte_start = (lane & 1) * 8;
    float sums[rows_per_simdgroup] = { 0.0f };
    for (uint block = block_offset; block < blocks_per_row; block += 16) {
        const uint x0 = block * EI_METAL_QK + byte_start;
        float x_lo[8];
        float x_hi[8];
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            x_lo[i] = x[x0 + i];
            x_hi[i] = x[x0 + i + 16];
        }
#pragma clang loop unroll(full)
        for (uint local_row = 0; local_row < rows_per_simdgroup; local_row++) {
            device const ei_metal_block_q4_0 &q =
                packed[local_row * blocks_per_row + block];
            const float d = float(q.d);
            device const uchar *qs = q.qs + byte_start;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8; i++) {
                const uchar code = qs[i];
                sums[local_row] += d * float(int(code & 0x0f) - 8) * x_lo[i];
                sums[local_row] += d * float(int(code >> 4) - 8) * x_hi[i];
            }
        }
    }
#pragma clang loop unroll(full)
    for (uint local_row = 0; local_row < rows_per_simdgroup; local_row++) {
        const float sum = metal::simd_sum(sums[local_row]);
        if (lane == 0) output[token * EI_METAL_N_FF + row + local_row] = sum;
    }
}

#define EI_Q4_0_F32_UP_GATE_GEMV_ROWS(rows)                                  \
kernel void ei_q4_0_f32_up_gate_gemv_r##rows(                               \
    device const uchar *up_weights [[buffer(0)]],                            \
    device const uchar *gate_weights [[buffer(1)]],                          \
    device const float *input [[buffer(2)]],                                 \
    device float *up_output [[buffer(3)]],                                   \
    device float *gate_output [[buffer(4)]],                                 \
    constant uint &n_tokens [[buffer(5)]],                                   \
    uint3 group [[threadgroup_position_in_grid]],                            \
    uint lane [[thread_index_in_simdgroup]]) {                               \
    ei_q4_0_f32_up_gate_gemv_rows<rows>(up_weights, gate_weights, input,     \
        up_output, gate_output, n_tokens, group, lane);                      \
}

EI_Q4_0_F32_UP_GATE_GEMV_ROWS(4)
#undef EI_Q4_0_F32_UP_GATE_GEMV_ROWS

// Fused gated-FFN projection: matching up/gate rows share activation loads and
// the kernel writes the activated product consumed by the down projection.
kernel void ei_q4_0_f32_up_gate_gelu_gemv(
    device const uchar *up_weights [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &n_tokens [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x;
    const uint token = group.y;
    if (row >= EI_METAL_N_FF || token >= n_tokens) return;

    constexpr uint blocks_per_row = EI_METAL_N_EMBD / EI_METAL_QK;
    device const ei_metal_block_q4_0 *up =
        (device const ei_metal_block_q4_0 *)up_weights + row * blocks_per_row;
    device const ei_metal_block_q4_0 *gate =
        (device const ei_metal_block_q4_0 *)gate_weights + row * blocks_per_row;
    device const float *x = input + token * EI_METAL_N_EMBD;
    const uint block_offset = lane >> 1;
    const uint byte_start = (lane & 1) * 8;
    float up_sum = 0.0f;
    float gate_sum = 0.0f;
    for (uint block = block_offset; block < blocks_per_row; block += 16) {
        const uint x0 = block * EI_METAL_QK + byte_start;
        device const uchar *up_qs = up[block].qs + byte_start;
        device const uchar *gate_qs = gate[block].qs + byte_start;
        const float up_d = float(up[block].d);
        const float gate_d = float(gate[block].d);
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            const float x_lo = x[x0 + i];
            const float x_hi = x[x0 + i + 16];
            const uchar up_code = up_qs[i];
            const uchar gate_code = gate_qs[i];
            up_sum += up_d * (float(int(up_code & 0x0f) - 8) * x_lo +
                              float(int(up_code >> 4) - 8) * x_hi);
            gate_sum += gate_d * (float(int(gate_code & 0x0f) - 8) * x_lo +
                                  float(int(gate_code >> 4) - 8) * x_hi);
        }
    }
    up_sum = metal::simd_sum(up_sum);
    gate_sum = metal::simd_sum(gate_sum);
    if (lane == 0) {
        output[token * EI_METAL_N_FF + row] =
            ei_metal_gelu_tanh(gate_sum) * up_sum;
    }
}

kernel void ei_q4_0_f32_up_gate_gelu_gemv_r2(
    device const uchar *up_weights [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &n_tokens [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows_per_simdgroup = 2;
    const uint first_row = group.x * rows_per_simdgroup;
    const uint token = group.y;
    if (first_row >= EI_METAL_N_FF || token >= n_tokens) return;

    constexpr uint blocks_per_row = EI_METAL_N_EMBD / EI_METAL_QK;
    device const ei_metal_block_q4_0 *up =
        (device const ei_metal_block_q4_0 *)up_weights;
    device const ei_metal_block_q4_0 *gate =
        (device const ei_metal_block_q4_0 *)gate_weights;
    device const float *x = input + token * EI_METAL_N_EMBD;
    const uint block_offset = lane >> 1;
    const uint byte_start = (lane & 1) * 8;
    float up_sums[rows_per_simdgroup] = { 0.0f };
    float gate_sums[rows_per_simdgroup] = { 0.0f };
    for (uint block = block_offset; block < blocks_per_row; block += 16) {
        const uint x0 = block * EI_METAL_QK + byte_start;
        float x_lo[8];
        float x_hi[8];
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            x_lo[i] = x[x0 + i];
            x_hi[i] = x[x0 + i + 16];
        }
#pragma clang loop unroll(full)
        for (uint local_row = 0; local_row < rows_per_simdgroup; local_row++) {
            const uint row = first_row + local_row;
            device const ei_metal_block_q4_0 &up_q =
                up[row * blocks_per_row + block];
            device const ei_metal_block_q4_0 &gate_q =
                gate[row * blocks_per_row + block];
            const float up_d = float(up_q.d);
            const float gate_d = float(gate_q.d);
            device const uchar *up_qs = up_q.qs + byte_start;
            device const uchar *gate_qs = gate_q.qs + byte_start;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8; i++) {
                const uchar up_code = up_qs[i];
                const uchar gate_code = gate_qs[i];
                up_sums[local_row] +=
                    up_d * (float(int(up_code & 0x0f) - 8) * x_lo[i] +
                            float(int(up_code >> 4) - 8) * x_hi[i]);
                gate_sums[local_row] +=
                    gate_d * (float(int(gate_code & 0x0f) - 8) * x_lo[i] +
                              float(int(gate_code >> 4) - 8) * x_hi[i]);
            }
        }
    }
#pragma clang loop unroll(full)
    for (uint local_row = 0; local_row < rows_per_simdgroup; local_row++) {
        const float up_sum = metal::simd_sum(up_sums[local_row]);
        const float gate_sum = metal::simd_sum(gate_sums[local_row]);
        if (lane == 0) {
            output[token * EI_METAL_N_FF + first_row + local_row] =
                ei_metal_gelu_tanh(gate_sum) * up_sum;
        }
    }
}

kernel void ei_q4_0_f32_up_gate_gelu_gemv_r4(
    device const uchar *up_weights [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &n_tokens [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows_per_simdgroup = 4;
    const uint first_row = group.x * rows_per_simdgroup;
    const uint token = group.y;
    if (first_row >= EI_METAL_N_FF || token >= n_tokens) return;

    constexpr uint blocks_per_row = EI_METAL_N_EMBD / EI_METAL_QK;
    device const ei_metal_block_q4_0 *up =
        (device const ei_metal_block_q4_0 *)up_weights;
    device const ei_metal_block_q4_0 *gate =
        (device const ei_metal_block_q4_0 *)gate_weights;
    device const float *x = input + token * EI_METAL_N_EMBD;
    const uint block_offset = lane >> 1;
    const uint byte_start = (lane & 1) * 8;
    float up_sums[rows_per_simdgroup] = { 0.0f };
    float gate_sums[rows_per_simdgroup] = { 0.0f };
    for (uint block = block_offset; block < blocks_per_row; block += 16) {
        const uint x0 = block * EI_METAL_QK + byte_start;
        float x_lo[8];
        float x_hi[8];
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            x_lo[i] = x[x0 + i];
            x_hi[i] = x[x0 + i + 16];
        }
#pragma clang loop unroll(full)
        for (uint local_row = 0; local_row < rows_per_simdgroup; local_row++) {
            const uint row = first_row + local_row;
            device const ei_metal_block_q4_0 &up_q =
                up[row * blocks_per_row + block];
            device const ei_metal_block_q4_0 &gate_q =
                gate[row * blocks_per_row + block];
            const float up_d = float(up_q.d);
            const float gate_d = float(gate_q.d);
            device const uchar *up_qs = up_q.qs + byte_start;
            device const uchar *gate_qs = gate_q.qs + byte_start;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8; i++) {
                const uchar up_code = up_qs[i];
                const uchar gate_code = gate_qs[i];
                up_sums[local_row] +=
                    up_d * (float(int(up_code & 0x0f) - 8) * x_lo[i] +
                            float(int(up_code >> 4) - 8) * x_hi[i]);
                gate_sums[local_row] +=
                    gate_d * (float(int(gate_code & 0x0f) - 8) * x_lo[i] +
                              float(int(gate_code >> 4) - 8) * x_hi[i]);
            }
        }
    }
#pragma clang loop unroll(full)
    for (uint local_row = 0; local_row < rows_per_simdgroup; local_row++) {
        const float up_sum = metal::simd_sum(up_sums[local_row]);
        const float gate_sum = metal::simd_sum(gate_sums[local_row]);
        if (lane == 0) {
            output[token * EI_METAL_N_FF + first_row + local_row] =
                ei_metal_gelu_tanh(gate_sum) * up_sum;
        }
    }
}

kernel void ei_q4_0_f32_up_gate_gemm(
    device const uchar *up_weights [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *up_output [[buffer(3)]],
    device float *gate_output [[buffer(4)]],
    constant uint &n_tokens [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint tile_rows = 32;
    constexpr uint tile_tokens = 8;
    threadgroup float weight_tile[tile_rows][EI_METAL_QK];
    threadgroup float input_tile[tile_tokens][EI_METAL_QK];
    const bool gate = group.x >= 36;
    const uint matrix_block = gate ? group.x - 36 : group.x;
    device const uchar *weight_bytes = gate ? gate_weights : up_weights;
    device float *output = gate ? gate_output : up_output;
    const uint local_row = tid & 31;
    const uint local_token = tid >> 5;
    const uint row = matrix_block * tile_rows + local_row;
    const uint token = group.y * tile_tokens + local_token;
    constexpr uint blocks_per_row = EI_METAL_N_EMBD / EI_METAL_QK;
    device const ei_metal_block_q4_0 *packed =
        (device const ei_metal_block_q4_0 *)weight_bytes;
    float sum = 0.0f;
    for (uint block = 0; block < blocks_per_row; block++) {
        for (uint item = tid; item < tile_rows * EI_METAL_QK; item += 256) {
            const uint load_row = item / EI_METAL_QK;
            const uint col = item - load_row * EI_METAL_QK;
            const uint global_row = matrix_block * tile_rows + load_row;
            device const ei_metal_block_q4_0 &q =
                packed[global_row * blocks_per_row + block];
            const uchar byte = q.qs[col < 16 ? col : col - 16];
            const int quant = col < 16 ? int(byte & 0x0f) - 8 : int(byte >> 4) - 8;
            weight_tile[load_row][col] = float(q.d) * float(quant);
        }
        input_tile[local_token][local_row] = token < n_tokens
            ? input[token * EI_METAL_N_EMBD + block * EI_METAL_QK + local_row]
            : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (token < n_tokens) {
#pragma clang loop unroll(full)
            for (uint col = 0; col < EI_METAL_QK; col++) {
                sum += weight_tile[local_row][col] * input_tile[local_token][col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (token < n_tokens) output[token * EI_METAL_N_FF + row] = sum;
}

// Long-prefill experiment: keep 256 outputs per threadgroup but trade rows for
// tokens. Compared with 32x8 this doubles weight reuse and halves staged weight
// traffic across the token grid.
kernel void ei_q4_0_f32_gemm_16x16(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &n_rows [[buffer(3)]],
    constant uint &n_cols [[buffer(4)]],
    constant uint &n_tokens [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint tile_rows = 16;
    constexpr uint tile_tokens = 16;
    threadgroup float weight_tile[tile_rows][EI_METAL_QK];
    threadgroup float input_tile[tile_tokens][EI_METAL_QK];
    const uint local_row = tid & 15;
    const uint local_token = tid >> 4;
    const uint row = group.x * tile_rows + local_row;
    const uint token = group.y * tile_tokens + local_token;
    const uint blocks_per_row = n_cols / EI_METAL_QK;
    device const ei_metal_block_q4_0 *packed =
        (device const ei_metal_block_q4_0 *)weights;
    float sum = 0.0f;
    for (uint block = 0; block < blocks_per_row; block++) {
        for (uint item = tid; item < tile_rows * EI_METAL_QK; item += 256) {
            const uint load_row = item / EI_METAL_QK;
            const uint col = item - load_row * EI_METAL_QK;
            const uint global_row = group.x * tile_rows + load_row;
            float value = 0.0f;
            if (global_row < n_rows) {
                device const ei_metal_block_q4_0 &q =
                    packed[global_row * blocks_per_row + block];
                const uchar code = q.qs[col < 16 ? col : col - 16];
                const int quant = col < 16
                    ? int(code & 0x0f) - 8 : int(code >> 4) - 8;
                value = float(q.d) * float(quant);
            }
            weight_tile[load_row][col] = value;
        }
        for (uint item = tid; item < tile_tokens * EI_METAL_QK; item += 256) {
            const uint load_token = item / EI_METAL_QK;
            const uint col = item - load_token * EI_METAL_QK;
            const uint global_token = group.y * tile_tokens + load_token;
            input_tile[load_token][col] = global_token < n_tokens
                ? input[global_token * n_cols + block * EI_METAL_QK + col]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < n_rows && token < n_tokens) {
#pragma clang loop unroll(full)
            for (uint col = 0; col < EI_METAL_QK; col++) {
                sum += weight_tile[local_row][col] * input_tile[local_token][col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < n_rows && token < n_tokens) output[token * n_rows + row] = sum;
}

kernel void ei_q4_0_f32_qkv_gemm_16x16(
    device const uchar *q_weights [[buffer(0)]],
    device const uchar *k_weights [[buffer(1)]],
    device const uchar *v_weights [[buffer(2)]],
    device const float *input [[buffer(3)]],
    device float *q_output [[buffer(4)]],
    device float *k_output [[buffer(5)]],
    device float *v_output [[buffer(6)]],
    constant uint &n_tokens [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint tile_rows = 16;
    constexpr uint tile_tokens = 16;
    threadgroup float weight_tile[tile_rows][EI_METAL_QK];
    threadgroup float input_tile[tile_tokens][EI_METAL_QK];
    device const uchar *weight_bytes;
    device float *output;
    uint matrix_block;
    uint rows;
    if (group.x < 48) {
        weight_bytes = q_weights;
        output = q_output;
        matrix_block = group.x;
        rows = EI_METAL_N_EMBD;
    } else if (group.x < 64) {
        weight_bytes = k_weights;
        output = k_output;
        matrix_block = group.x - 48;
        rows = EI_METAL_HEAD_DIM;
    } else {
        weight_bytes = v_weights;
        output = v_output;
        matrix_block = group.x - 64;
        rows = EI_METAL_HEAD_DIM;
    }
    const uint local_row = tid & 15;
    const uint local_token = tid >> 4;
    const uint row = matrix_block * tile_rows + local_row;
    const uint token = group.y * tile_tokens + local_token;
    constexpr uint blocks_per_row = EI_METAL_N_EMBD / EI_METAL_QK;
    device const ei_metal_block_q4_0 *packed =
        (device const ei_metal_block_q4_0 *)weight_bytes;
    float sum = 0.0f;
    for (uint block = 0; block < blocks_per_row; block++) {
        for (uint item = tid; item < tile_rows * EI_METAL_QK; item += 256) {
            const uint load_row = item / EI_METAL_QK;
            const uint col = item - load_row * EI_METAL_QK;
            const uint global_row = matrix_block * tile_rows + load_row;
            device const ei_metal_block_q4_0 &packed_block =
                packed[global_row * blocks_per_row + block];
            const uchar code = packed_block.qs[col < 16 ? col : col - 16];
            const int quant = col < 16
                ? int(code & 0x0f) - 8 : int(code >> 4) - 8;
            weight_tile[load_row][col] = float(packed_block.d) * float(quant);
        }
        for (uint item = tid; item < tile_tokens * EI_METAL_QK; item += 256) {
            const uint load_token = item / EI_METAL_QK;
            const uint col = item - load_token * EI_METAL_QK;
            const uint global_token = group.y * tile_tokens + load_token;
            input_tile[load_token][col] = global_token < n_tokens
                ? input[global_token * EI_METAL_N_EMBD + block * EI_METAL_QK + col]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (token < n_tokens) {
#pragma clang loop unroll(full)
            for (uint col = 0; col < EI_METAL_QK; col++) {
                sum += weight_tile[local_row][col] * input_tile[local_token][col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (token < n_tokens) output[token * rows + row] = sum;
}

kernel void ei_q4_0_f32_up_gate_gemm_16x16(
    device const uchar *up_weights [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *up_output [[buffer(3)]],
    device float *gate_output [[buffer(4)]],
    constant uint &n_tokens [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint tile_rows = 16;
    constexpr uint tile_tokens = 16;
    threadgroup float weight_tile[tile_rows][EI_METAL_QK];
    threadgroup float input_tile[tile_tokens][EI_METAL_QK];
    const bool gate = group.x >= 72;
    const uint matrix_block = gate ? group.x - 72 : group.x;
    device const uchar *weight_bytes = gate ? gate_weights : up_weights;
    device float *output = gate ? gate_output : up_output;
    const uint local_row = tid & 15;
    const uint local_token = tid >> 4;
    const uint row = matrix_block * tile_rows + local_row;
    const uint token = group.y * tile_tokens + local_token;
    constexpr uint blocks_per_row = EI_METAL_N_EMBD / EI_METAL_QK;
    device const ei_metal_block_q4_0 *packed =
        (device const ei_metal_block_q4_0 *)weight_bytes;
    float sum = 0.0f;
    for (uint block = 0; block < blocks_per_row; block++) {
        for (uint item = tid; item < tile_rows * EI_METAL_QK; item += 256) {
            const uint load_row = item / EI_METAL_QK;
            const uint col = item - load_row * EI_METAL_QK;
            const uint global_row = matrix_block * tile_rows + load_row;
            device const ei_metal_block_q4_0 &packed_block =
                packed[global_row * blocks_per_row + block];
            const uchar code = packed_block.qs[col < 16 ? col : col - 16];
            const int quant = col < 16
                ? int(code & 0x0f) - 8 : int(code >> 4) - 8;
            weight_tile[load_row][col] = float(packed_block.d) * float(quant);
        }
        for (uint item = tid; item < tile_tokens * EI_METAL_QK; item += 256) {
            const uint load_token = item / EI_METAL_QK;
            const uint col = item - load_token * EI_METAL_QK;
            const uint global_token = group.y * tile_tokens + load_token;
            input_tile[load_token][col] = global_token < n_tokens
                ? input[global_token * EI_METAL_N_EMBD + block * EI_METAL_QK + col]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (token < n_tokens) {
#pragma clang loop unroll(full)
            for (uint col = 0; col < EI_METAL_QK; col++) {
                sum += weight_tile[local_row][col] * input_tile[local_token][col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (token < n_tokens) output[token * EI_METAL_N_FF + row] = sum;
}
