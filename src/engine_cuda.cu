extern "C" {
#include "engine_cuda.h"
}

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kQK = 32;
constexpr int kThreads = 256;
constexpr int kWarpsPerBlock = kThreads / 32;
constexpr unsigned kWarpMask = 0xffffffffu;

struct __align__(2) block_q4_0 {
    __half d;
    uint8_t qs[16];
};

struct __align__(2) block_q8_0 {
    __half d;
    int8_t qs[32];
};

struct __align__(4) activation_q8 {
    __half d;
    int16_t sum;
    int8_t qs[32];
};

static_assert(sizeof(block_q4_0) == 18, "q4_0 block layout mismatch");
static_assert(sizeof(block_q8_0) == 34, "q8_0 block layout mismatch");
static_assert(sizeof(activation_q8) == 36, "activation q8 block layout mismatch");

struct graph_entry {
    uint32_t tokens;
    uint32_t batch_size;
    uint64_t shape_hash;
    cudaGraphExec_t executable;
};

struct cuda_engine {
    const ei_model *model = nullptr;
    int device = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;

    uint8_t *model_data = nullptr;
    float2 *rope_full = nullptr;
    float2 *rope_swa = nullptr;
    __half *dequantized_weights = nullptr;
    struct {
        __half *attn_q;
        __half *attn_k;
        __half *attn_v;
        __half *attn_output;
        __half *ffn_up;
        __half *ffn_gate;
        __half *ffn_down;
    } layers[EI_N_LAYER] = {};
    uint32_t gemm_min_tokens = 5;

    size_t token_capacity = 0;
    size_t batch_capacity = 0;
    int32_t *ids = nullptr;
    uint32_t *positions = nullptr;
    uint32_t *sequence_ids = nullptr;
    uint32_t *offsets = nullptr;
    float *x = nullptr;
    float *norm = nullptr;
    float *q = nullptr;
    float *k = nullptr;
    float *v = nullptr;
    float *qkv_combined = nullptr;
    __half *q_half = nullptr;
    __half *k_half = nullptr;
    __half *v_half = nullptr;
    float *ctx = nullptr;
    float *up = nullptr;
    float *gate = nullptr;
    float *up_gate_combined = nullptr;
    float *tmp = nullptr;
    __half *half_input = nullptr;
    activation_q8 *q8_input = nullptr;
    uint32_t *flash_query_start = nullptr;
    uint32_t *flash_sequence_start = nullptr;
    uint32_t *flash_sequence_stop = nullptr;
    float *result = nullptr;
    std::vector<graph_entry> graphs;
    uint32_t fp16_attention_min_tokens = 5;
    uint32_t flash_attention_min_tokens = 5;
    uint32_t flash_attention_max_tokens = 192;
    uint32_t tensor_attention_min_tokens = 80;
    bool tensor_attention_tuned = true;
    uint32_t swa_tensor_tile_tokens = 1024;
    uint32_t swa_tensor_banded_min_tokens = 1536;
    bool direct_fp16_qkv = true;
    bool direct_fp16_context = false;
    bool native_q4_gemm = false;
    bool q8_latency = true;
    float *attention_scores = nullptr;
    __half *attention_probs = nullptr;
};

static void set_error(char *err, size_t err_len, const char *message) {
    if (err && err_len) std::snprintf(err, err_len, "%s", message);
}

static bool cuda_check(cudaError_t status, const char *operation,
                       char *err, size_t err_len) {
    if (status == cudaSuccess) return true;
    if (err && err_len) {
        std::snprintf(err, err_len, "%s: %s", operation,
                      cudaGetErrorString(status));
    }
    return false;
}

static bool cublas_check(cublasStatus_t status, const char *operation,
                         char *err, size_t err_len) {
    if (status == CUBLAS_STATUS_SUCCESS) return true;
    if (err && err_len) {
        std::snprintf(err, err_len, "%s: cuBLAS status %d", operation,
                      static_cast<int>(status));
    }
    return false;
}

template <typename T>
static void cuda_release(T *&ptr) {
    if (ptr) cudaFree(ptr);
    ptr = nullptr;
}

static void clear_graphs(cuda_engine *engine) {
    for (const graph_entry &entry : engine->graphs) {
        if (entry.executable) cudaGraphExecDestroy(entry.executable);
    }
    engine->graphs.clear();
}

template <typename T>
static bool cuda_allocate(T **ptr, size_t count, const char *name,
                          char *err, size_t err_len) {
    cudaError_t status = cudaMalloc(reinterpret_cast<void **>(ptr),
                                    count * sizeof(T));
    return cuda_check(status, name, err, err_len);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kWarpMask, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(kWarpMask, value, offset));
    }
    return value;
}

__device__ __forceinline__ int warp_sum_int(int value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kWarpMask, value, offset);
    }
    return value;
}

#if __CUDA_ARCH__ >= 610
__device__ __forceinline__ uint32_t load_u32_unaligned(const uint8_t *bytes) {
    return static_cast<uint32_t>(bytes[0]) |
           (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) |
           (static_cast<uint32_t>(bytes[3]) << 24);
}
#endif

__device__ __forceinline__ float q4_dot(const block_q4_0 *weights,
                                        const float *input, int n_cols,
                                        int lane) {
    const int blocks_per_row = n_cols / kQK;
    const int block_offset = lane >> 1;
    const int byte_start = (lane & 1) * 8;
    float sum = 0.0f;
    for (int block = block_offset; block < blocks_per_row; block += 16) {
        const block_q4_0 &packed = weights[block];
        const float d = __half2float(packed.d);
        const int x0 = block * kQK + byte_start;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint8_t code = packed.qs[byte_start + i];
            sum = fmaf(d * static_cast<float>((code & 0x0f) - 8),
                       input[x0 + i], sum);
            sum = fmaf(d * static_cast<float>((code >> 4) - 8),
                       input[x0 + i + 16], sum);
        }
    }
    return warp_sum(sum);
}

__device__ __forceinline__ float q4_q8_dot(const block_q4_0 *weights,
                                           const activation_q8 *input,
                                           int n_cols, int lane) {
    const int blocks_per_row = n_cols / kQK;
    float sum = 0.0f;
    for (int block = lane; block < blocks_per_row; block += 32) {
        const block_q4_0 &q4 = weights[block];
        const activation_q8 &q8 = input[block];
        int dot = 0;
#pragma unroll
        for (int group = 0; group < 4; group++) {
#if __CUDA_ARCH__ >= 610
            const uint32_t packed = load_u32_unaligned(q4.qs + group * 4);
            const int low = static_cast<int>(packed & 0x0f0f0f0fu);
            const int high = static_cast<int>((packed >> 4) & 0x0f0f0f0fu);
            const int q8_low = *reinterpret_cast<const int *>(q8.qs + group * 4);
            const int q8_high = *reinterpret_cast<const int *>(q8.qs + 16 + group * 4);
            dot = __dp4a(low, q8_low, dot);
            dot = __dp4a(high, q8_high, dot);
#else
#pragma unroll
            for (int item = 0; item < 4; item++) {
                const uint8_t packed = q4.qs[group * 4 + item];
                dot += (static_cast<int>(packed & 0x0f) - 8) *
                       static_cast<int>(q8.qs[group * 4 + item]);
                dot += (static_cast<int>(packed >> 4) - 8) *
                       static_cast<int>(q8.qs[16 + group * 4 + item]);
            }
#endif
        }
#if __CUDA_ARCH__ >= 610
        dot -= 8 * static_cast<int>(q8.sum);
#endif
        sum = fmaf(__half2float(q4.d) * __half2float(q8.d),
                   static_cast<float>(dot), sum);
    }
    return warp_sum(sum);
}

__global__ void quantize_q8_kernel(const float *input, activation_q8 *output,
                                   uint32_t n_rows, uint32_t n_cols) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const uint32_t blocks_per_row = n_cols / kQK;
    const size_t task_count = static_cast<size_t>(n_rows) * blocks_per_row;
    if (task >= task_count) return;
    const uint32_t row = static_cast<uint32_t>(task / blocks_per_row);
    const uint32_t block_index = static_cast<uint32_t>(task -
        static_cast<size_t>(row) * blocks_per_row);
    const float value = input[static_cast<size_t>(row) * n_cols +
                              block_index * kQK + lane];
    float amax = warp_max(fabsf(value));
    amax = __shfl_sync(kWarpMask, amax, 0);
    const float scale = amax / 127.0f;
    int quant = scale == 0.0f ? 0 : __float2int_rn(value / scale);
    quant = max(-128, min(127, quant));
    activation_q8 &block = output[
        static_cast<size_t>(row) * blocks_per_row + block_index];
    block.qs[lane] = static_cast<int8_t>(quant);
    const int sum = warp_sum_int(quant);
    if (lane == 0) {
        block.d = __float2half(scale);
        block.sum = static_cast<int16_t>(sum);
    }
}

__global__ void rms_norm_quantize_q8_kernel(
    const float *input, const float *weight, activation_q8 *output,
    uint32_t n_rows, uint32_t n_cols, float eps) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= n_rows) return;
    const size_t base = static_cast<size_t>(row) * n_cols;
    float ss = 0.0f;
    for (uint32_t col = lane; col < n_cols; col += 32) {
        const float value = input[base + col];
        ss = fmaf(value, value, ss);
    }
    ss = warp_sum(ss);
    ss = __shfl_sync(kWarpMask, ss, 0);
    const float inv = rsqrtf(ss / static_cast<float>(n_cols) + eps);
    const uint32_t blocks_per_row = n_cols / kQK;
    for (uint32_t block_index = 0; block_index < blocks_per_row; block_index++) {
        const uint32_t col = block_index * kQK + lane;
        const float value = input[base + col] * weight[col] * inv;
        float amax = warp_max(fabsf(value));
        amax = __shfl_sync(kWarpMask, amax, 0);
        const float scale = amax / 127.0f;
        int quant = scale == 0.0f ? 0 : __float2int_rn(value / scale);
        quant = max(-128, min(127, quant));
        activation_q8 &block = output[
            static_cast<size_t>(row) * blocks_per_row + block_index];
        block.qs[lane] = static_cast<int8_t>(quant);
        const int sum = warp_sum_int(quant);
        if (lane == 0) {
            block.d = __float2half(scale);
            block.sum = static_cast<int16_t>(sum);
        }
    }
}

__global__ void embedding_kernel(const block_q8_0 *table,
                                 const int32_t *ids, float *output,
                                 uint32_t n_tokens, float scale) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    const size_t count = static_cast<size_t>(n_tokens) * EI_N_EMBD;
    if (index >= count) return;
    const uint32_t token = static_cast<uint32_t>(index / EI_N_EMBD);
    const int col = static_cast<int>(index - static_cast<size_t>(token) * EI_N_EMBD);
    const block_q8_0 *row = table +
        static_cast<size_t>(ids[token]) * (EI_N_EMBD / kQK);
    const block_q8_0 &block = row[col / kQK];
    output[index] = scale * __half2float(block.d) *
                    static_cast<float>(block.qs[col % kQK]);
}

__global__ void dequantize_q4_kernel(const block_q4_0 *input,
                                     __half *output, uint32_t n_rows,
                                     uint32_t n_cols) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    const size_t count = static_cast<size_t>(n_rows) * n_cols;
    if (index >= count) return;
    const uint32_t row = static_cast<uint32_t>(index / n_cols);
    const uint32_t col = static_cast<uint32_t>(index -
        static_cast<size_t>(row) * n_cols);
    const block_q4_0 &block = input[
        static_cast<size_t>(row) * (n_cols / kQK) + col / kQK];
    const uint32_t block_col = col % kQK;
    const uint8_t code = block.qs[block_col < 16 ? block_col : block_col - 16];
    const int quant = block_col < 16 ? (code & 0x0f) - 8 : (code >> 4) - 8;
    output[index] = __float2half(__half2float(block.d) * static_cast<float>(quant));
}

__device__ __forceinline__ void load_q4_mma_fragment(
    __half2 fragment[4], const block_q4_0 *weights, int n_rows, int n_cols,
    int row_start, int col_start) {
    const int lane = threadIdx.x & 31;
    const int row = row_start + lane / 4;
    const int col = (col_start % kQK) + (lane % 4) * 2;
    const int block = col_start / kQK;
    const int blocks_per_row = n_cols / kQK;
#pragma unroll
    for (int item = 0; item < 4; item++) {
        const int fragment_row = row + (item & 1) * 8;
        const int fragment_col = col + (item >> 1) * 8;
        if (fragment_row >= n_rows) {
            fragment[item] = __float2half2_rn(0.0f);
            continue;
        }
        const block_q4_0 &packed = weights[
            static_cast<size_t>(fragment_row) * blocks_per_row + block];
        const int byte = fragment_col < 16 ? fragment_col : fragment_col - 16;
        const uint8_t code0 = packed.qs[byte];
        const uint8_t code1 = packed.qs[byte + 1];
        const int quant0 = fragment_col < 16
            ? (code0 & 0x0f) - 8 : (code0 >> 4) - 8;
        const int quant1 = fragment_col < 16
            ? (code1 & 0x0f) - 8 : (code1 >> 4) - 8;
        const float scale = __half2float(packed.d);
        fragment[item] = __floats2half2_rn(scale * static_cast<float>(quant0),
                                            scale * static_cast<float>(quant1));
    }
}

__device__ __forceinline__ void load_input_mma_fragment(
    __half2 fragment[4], const __half *input, int n_rows, int n_cols,
    int row_start, int col_start) {
    const int lane = threadIdx.x & 31;
    const int row = row_start + lane / 4;
    const int col = col_start + (lane % 4) * 2;
#pragma unroll
    for (int item = 0; item < 4; item++) {
        const int fragment_row = row + (item & 1) * 8;
        const int fragment_col = col + (item >> 1) * 8;
        if (fragment_row < n_rows) {
            const __half *source = input +
                static_cast<size_t>(fragment_row) * n_cols + fragment_col;
            fragment[item] = *reinterpret_cast<const __half2 *>(source);
        } else {
            fragment[item] = __float2half2_rn(0.0f);
        }
    }
}

__device__ __forceinline__ void mma_m16n8k16(
    float accumulator[4], const __half2 input[4], const __half2 weights[2]) {
#if __CUDA_ARCH__ >= 800
    const uint32_t *a = reinterpret_cast<const uint32_t *>(input);
    const uint32_t *b = reinterpret_cast<const uint32_t *>(weights);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(accumulator[0]), "+f"(accumulator[1]),
          "+f"(accumulator[2]), "+f"(accumulator[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
#else
    (void)accumulator;
    (void)input;
    (void)weights;
#endif
}

__global__ void q4_mma_projection_kernel(
    const block_q4_0 *weights, const __half *input, float *output,
    int n_tokens, int n_rows, int n_cols) {
    const int row_start = static_cast<int>(blockIdx.x) * 16;
    const int token_start = static_cast<int>(blockIdx.y) * 16;
    float accumulator0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float accumulator1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int col_start = 0; col_start < n_cols; col_start += 16) {
        __half2 input_fragment[4];
        __half2 weight_fragment[4];
        load_input_mma_fragment(input_fragment, input, n_tokens, n_cols,
                                token_start, col_start);
        load_q4_mma_fragment(weight_fragment, weights, n_rows, n_cols,
                             row_start, col_start);
        const __half2 weight0[2] = {weight_fragment[0], weight_fragment[2]};
        const __half2 weight1[2] = {weight_fragment[1], weight_fragment[3]};
        mma_m16n8k16(accumulator0, input_fragment, weight0);
        mma_m16n8k16(accumulator1, input_fragment, weight1);
    }
    const int lane = threadIdx.x & 31;
    const int token = token_start + lane / 4;
    const int row = row_start + (lane % 4) * 2;
    if (token < n_tokens) {
        if (row < n_rows) output[static_cast<size_t>(token) * n_rows + row] =
            accumulator0[0];
        if (row + 1 < n_rows) output[static_cast<size_t>(token) * n_rows + row + 1] =
            accumulator0[1];
        if (row + 8 < n_rows) output[static_cast<size_t>(token) * n_rows + row + 8] =
            accumulator1[0];
        if (row + 9 < n_rows) output[static_cast<size_t>(token) * n_rows + row + 9] =
            accumulator1[1];
    }
    if (token + 8 < n_tokens) {
        if (row < n_rows) output[static_cast<size_t>(token + 8) * n_rows + row] =
            accumulator0[2];
        if (row + 1 < n_rows) output[static_cast<size_t>(token + 8) * n_rows + row + 1] =
            accumulator0[3];
        if (row + 8 < n_rows) output[static_cast<size_t>(token + 8) * n_rows + row + 8] =
            accumulator1[2];
        if (row + 9 < n_rows) output[static_cast<size_t>(token + 8) * n_rows + row + 9] =
            accumulator1[3];
    }
}

__device__ __forceinline__ float gelu_tanh(float value);

__global__ void f32_to_f16_kernel(const float *input, __half *output,
                                  size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    if (index < count) output[index] = __float2half(input[index]);
}

__global__ void gelu_mul_kernel(float *gate, const float *up, size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    if (index < count) gate[index] = gelu_tanh(gate[index]) * up[index];
}

__global__ void split_qkv_kernel(const float *combined, float *q, float *k,
                                 float *v, uint32_t n_tokens) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    const size_t count = static_cast<size_t>(n_tokens) *
                         (EI_N_EMBD + 2 * EI_HEAD_DIM);
    if (index >= count) return;
    const uint32_t token = static_cast<uint32_t>(index /
        (EI_N_EMBD + 2 * EI_HEAD_DIM));
    const uint32_t col = static_cast<uint32_t>(index -
        static_cast<size_t>(token) * (EI_N_EMBD + 2 * EI_HEAD_DIM));
    if (col < EI_N_EMBD) {
        q[static_cast<size_t>(token) * EI_N_EMBD + col] = combined[index];
    } else if (col < EI_N_EMBD + EI_HEAD_DIM) {
        k[static_cast<size_t>(token) * EI_HEAD_DIM + col - EI_N_EMBD] =
            combined[index];
    } else {
        v[static_cast<size_t>(token) * EI_HEAD_DIM +
          col - EI_N_EMBD - EI_HEAD_DIM] = combined[index];
    }
}

__global__ void qkv_f32_to_f16_kernel(
    const float *q, const float *k, const float *v,
    __half *q_half, __half *k_half, __half *v_half, uint32_t n_tokens) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    const size_t count = static_cast<size_t>(n_tokens) *
                         (EI_N_EMBD + 2 * EI_HEAD_DIM);
    if (index >= count) return;
    const uint32_t token = static_cast<uint32_t>(index /
        (EI_N_EMBD + 2 * EI_HEAD_DIM));
    const uint32_t col = static_cast<uint32_t>(index -
        static_cast<size_t>(token) * (EI_N_EMBD + 2 * EI_HEAD_DIM));
    if (col < EI_N_EMBD) {
        q_half[static_cast<size_t>(token) * EI_N_EMBD + col] =
            __float2half(q[static_cast<size_t>(token) * EI_N_EMBD + col]);
    } else if (col < EI_N_EMBD + EI_HEAD_DIM) {
        const uint32_t dim = col - EI_N_EMBD;
        k_half[static_cast<size_t>(token) * EI_HEAD_DIM + dim] =
            __float2half(k[static_cast<size_t>(token) * EI_HEAD_DIM + dim]);
    } else {
        const uint32_t dim = col - EI_N_EMBD - EI_HEAD_DIM;
        v_half[static_cast<size_t>(token) * EI_HEAD_DIM + dim] =
            __float2half(v[static_cast<size_t>(token) * EI_HEAD_DIM + dim]);
    }
}

__global__ void up_gate_gelu_to_f16_kernel(const float *combined,
                                           __half *output,
                                           uint32_t n_tokens) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    const size_t count = static_cast<size_t>(n_tokens) * EI_N_FF;
    if (index >= count) return;
    const uint32_t token = static_cast<uint32_t>(index / EI_N_FF);
    const uint32_t col = static_cast<uint32_t>(index -
        static_cast<size_t>(token) * EI_N_FF);
    const size_t base = static_cast<size_t>(token) * (2 * EI_N_FF);
    const float up = combined[base + col];
    const float gate = combined[base + EI_N_FF + col];
    output[index] = __float2half(gelu_tanh(gate) * up);
}

__global__ void q4_projection_kernel(const block_q4_0 *weights,
                                     const float *input, float *output,
                                     uint32_t n_rows, uint32_t n_cols,
                                     uint32_t n_tokens) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * n_rows;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / n_rows);
    const uint32_t row = static_cast<uint32_t>(task -
        static_cast<size_t>(token) * n_rows);
    const block_q4_0 *row_weights = weights +
        static_cast<size_t>(row) * (n_cols / kQK);
    const float sum = q4_dot(row_weights,
                             input + static_cast<size_t>(token) * n_cols,
                             static_cast<int>(n_cols), lane);
    if (lane == 0) output[static_cast<size_t>(token) * n_rows + row] = sum;
}

__global__ void q4_q8_projection_kernel(const block_q4_0 *weights,
                                        const activation_q8 *input,
                                        float *output, uint32_t n_rows,
                                        uint32_t n_cols, uint32_t n_tokens) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * n_rows;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / n_rows);
    const uint32_t row = static_cast<uint32_t>(task -
        static_cast<size_t>(token) * n_rows);
    const uint32_t blocks_per_row = n_cols / kQK;
    const float sum = q4_q8_dot(
        weights + static_cast<size_t>(row) * blocks_per_row,
        input + static_cast<size_t>(token) * blocks_per_row,
        static_cast<int>(n_cols), lane);
    if (lane == 0) output[static_cast<size_t>(token) * n_rows + row] = sum;
}

__global__ void qkv_projection_kernel(const block_q4_0 *q_weights,
                                      const block_q4_0 *k_weights,
                                      const block_q4_0 *v_weights,
                                      const float *input, float *q_output,
                                      float *k_output, float *v_output,
                                      uint32_t n_tokens) {
    constexpr uint32_t combined_rows = EI_N_EMBD + 2 * EI_HEAD_DIM;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * combined_rows;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / combined_rows);
    const uint32_t combined_row = static_cast<uint32_t>(task -
        static_cast<size_t>(token) * combined_rows);

    const block_q4_0 *weights;
    float *output;
    uint32_t row;
    uint32_t rows;
    if (combined_row < EI_N_EMBD) {
        weights = q_weights;
        output = q_output;
        row = combined_row;
        rows = EI_N_EMBD;
    } else if (combined_row < EI_N_EMBD + EI_HEAD_DIM) {
        weights = k_weights;
        output = k_output;
        row = combined_row - EI_N_EMBD;
        rows = EI_HEAD_DIM;
    } else {
        weights = v_weights;
        output = v_output;
        row = combined_row - EI_N_EMBD - EI_HEAD_DIM;
        rows = EI_HEAD_DIM;
    }
    const block_q4_0 *row_weights = weights +
        static_cast<size_t>(row) * (EI_N_EMBD / kQK);
    const float sum = q4_dot(row_weights,
                             input + static_cast<size_t>(token) * EI_N_EMBD,
                             EI_N_EMBD, lane);
    if (lane == 0) output[static_cast<size_t>(token) * rows + row] = sum;
}

__global__ void qkv_q8_projection_kernel(
    const block_q4_0 *q_weights, const block_q4_0 *k_weights,
    const block_q4_0 *v_weights, const activation_q8 *input,
    float *q_output, float *k_output, float *v_output, uint32_t n_tokens) {
    constexpr uint32_t combined_rows = EI_N_EMBD + 2 * EI_HEAD_DIM;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * combined_rows;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / combined_rows);
    const uint32_t combined_row = static_cast<uint32_t>(task -
        static_cast<size_t>(token) * combined_rows);
    const block_q4_0 *weights;
    float *output;
    uint32_t row;
    uint32_t rows;
    if (combined_row < EI_N_EMBD) {
        weights = q_weights;
        output = q_output;
        row = combined_row;
        rows = EI_N_EMBD;
    } else if (combined_row < EI_N_EMBD + EI_HEAD_DIM) {
        weights = k_weights;
        output = k_output;
        row = combined_row - EI_N_EMBD;
        rows = EI_HEAD_DIM;
    } else {
        weights = v_weights;
        output = v_output;
        row = combined_row - EI_N_EMBD - EI_HEAD_DIM;
        rows = EI_HEAD_DIM;
    }
    constexpr uint32_t blocks_per_row = EI_N_EMBD / kQK;
    const float sum = q4_q8_dot(
        weights + static_cast<size_t>(row) * blocks_per_row,
        input + static_cast<size_t>(token) * blocks_per_row,
        EI_N_EMBD, lane);
    if (lane == 0) output[static_cast<size_t>(token) * rows + row] = sum;
}

__device__ __forceinline__ float gelu_tanh(float value) {
    constexpr float a = 0.044715f;
    constexpr float s = 0.7978845608028654f;
    return 0.5f * value * (1.0f + tanhf(s * value *
                                        (1.0f + a * value * value)));
}

__global__ void up_gate_gelu_kernel(const block_q4_0 *up_weights,
                                    const block_q4_0 *gate_weights,
                                    const float *input, float *output,
                                    uint32_t n_tokens) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * EI_N_FF;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / EI_N_FF);
    const uint32_t row = static_cast<uint32_t>(task -
        static_cast<size_t>(token) * EI_N_FF);
    constexpr int blocks_per_row = EI_N_EMBD / kQK;
    const float *x = input + static_cast<size_t>(token) * EI_N_EMBD;
    const float up = q4_dot(up_weights + static_cast<size_t>(row) * blocks_per_row,
                            x, EI_N_EMBD, lane);
    const float gate = q4_dot(gate_weights + static_cast<size_t>(row) * blocks_per_row,
                              x, EI_N_EMBD, lane);
    if (lane == 0) {
        output[static_cast<size_t>(token) * EI_N_FF + row] =
            gelu_tanh(gate) * up;
    }
}

__global__ void up_gate_q8_gelu_kernel(
    const block_q4_0 *up_weights, const block_q4_0 *gate_weights,
    const activation_q8 *input, float *output, uint32_t n_tokens) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * EI_N_FF;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / EI_N_FF);
    const uint32_t row = static_cast<uint32_t>(task -
        static_cast<size_t>(token) * EI_N_FF);
    constexpr int blocks_per_row = EI_N_EMBD / kQK;
    const activation_q8 *x = input +
        static_cast<size_t>(token) * blocks_per_row;
    const float up = q4_q8_dot(
        up_weights + static_cast<size_t>(row) * blocks_per_row,
        x, EI_N_EMBD, lane);
    const float gate = q4_q8_dot(
        gate_weights + static_cast<size_t>(row) * blocks_per_row,
        x, EI_N_EMBD, lane);
    if (lane == 0) {
        output[static_cast<size_t>(token) * EI_N_FF + row] =
            gelu_tanh(gate) * up;
    }
}

template <bool AddResidual>
__global__ void rms_norm_kernel(const float *input, const float *weight,
                                float *output, uint32_t n_rows,
                                uint32_t n_cols, float eps) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= n_rows) return;
    const size_t base = static_cast<size_t>(row) * n_cols;
    float sum = 0.0f;
    for (uint32_t col = lane; col < n_cols; col += 32) {
        const float value = input[base + col];
        sum = fmaf(value, value, sum);
    }
    sum = warp_sum(sum);
    sum = __shfl_sync(kWarpMask, sum, 0);
    const float inv = rsqrtf(sum / static_cast<float>(n_cols) + eps);
    for (uint32_t col = lane; col < n_cols; col += 32) {
        const float value = input[base + col] * weight[col] * inv;
        if (AddResidual) output[base + col] += value;
        else output[base + col] = value;
    }
}

__global__ void rms_norm_f16_kernel(const float *input, const float *weight,
                                    __half *output, uint32_t n_rows,
                                    uint32_t n_cols, float eps) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= n_rows) return;
    const size_t base = static_cast<size_t>(row) * n_cols;
    float sum = 0.0f;
    for (uint32_t col = lane; col < n_cols; col += 32) {
        const float value = input[base + col];
        sum = fmaf(value, value, sum);
    }
    sum = warp_sum(sum);
    sum = __shfl_sync(kWarpMask, sum, 0);
    const float inv = rsqrtf(sum / static_cast<float>(n_cols) + eps);
    for (uint32_t col = lane; col < n_cols; col += 32) {
        output[base + col] = __float2half(input[base + col] * weight[col] * inv);
    }
}

__global__ void rms_residual_next_f16_kernel(
    const float *input, const float *post_weight, float *residual,
    const float *next_weight, __half *next_output, uint32_t n_rows,
    uint32_t n_cols, float eps) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= n_rows) return;
    const size_t base = static_cast<size_t>(row) * n_cols;
    float projected_ss = 0.0f;
    for (uint32_t col = lane; col < n_cols; col += 32) {
        const float value = input[base + col];
        projected_ss = fmaf(value, value, projected_ss);
    }
    projected_ss = warp_sum(projected_ss);
    projected_ss = __shfl_sync(kWarpMask, projected_ss, 0);
    const float projected_inv = rsqrtf(
        projected_ss / static_cast<float>(n_cols) + eps);

    float residual_ss = 0.0f;
    for (uint32_t col = lane; col < n_cols; col += 32) {
        const float value = residual[base + col] +
            input[base + col] * post_weight[col] * projected_inv;
        residual[base + col] = value;
        residual_ss = fmaf(value, value, residual_ss);
    }
    residual_ss = warp_sum(residual_ss);
    residual_ss = __shfl_sync(kWarpMask, residual_ss, 0);
    const float residual_inv = rsqrtf(
        residual_ss / static_cast<float>(n_cols) + eps);
    for (uint32_t col = lane; col < n_cols; col += 32) {
        next_output[base + col] = __float2half(
            residual[base + col] * next_weight[col] * residual_inv);
    }
}

__global__ void qk_norm_rope_kernel(float *q, float *k,
                                    const float *q_weight,
                                    const float *k_weight,
                                    const uint32_t *positions,
                                    const float2 *rope_table,
                                    uint32_t n_tokens, float eps) {
    constexpr uint32_t combined_heads = EI_N_HEAD + EI_N_HEAD_KV;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * combined_heads;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / combined_heads);
    const uint32_t combined_head = static_cast<uint32_t>(task -
        static_cast<size_t>(token) * combined_heads);
    const bool key_head = combined_head == EI_N_HEAD;
    const uint32_t head = key_head ? 0 : combined_head;
    const uint32_t stride = key_head ? EI_HEAD_DIM : EI_N_EMBD;
    float *values = key_head ? k : q;
    const float *weight = key_head ? k_weight : q_weight;
    const size_t base = static_cast<size_t>(token) * stride +
                        static_cast<size_t>(head) * EI_HEAD_DIM;
    float sum = 0.0f;
    for (uint32_t dim = lane; dim < EI_HEAD_DIM; dim += 32) {
        const float value = values[base + dim];
        sum = fmaf(value, value, sum);
    }
    sum = warp_sum(sum);
    sum = __shfl_sync(kWarpMask, sum, 0);
    const float scale = key_head ? 1.0f : 0.0625f;
    const float inv = rsqrtf(sum / static_cast<float>(EI_HEAD_DIM) + eps) * scale;
    const size_t rope_base = static_cast<size_t>(positions[token]) *
                             (EI_HEAD_DIM / 2);
    for (uint32_t dim = lane; dim < EI_HEAD_DIM / 2; dim += 32) {
        const float2 cs = rope_table[rope_base + dim];
        const float x0 = values[base + dim] * weight[dim] * inv;
        const float x1 = values[base + dim + EI_HEAD_DIM / 2] *
                         weight[dim + EI_HEAD_DIM / 2] * inv;
        values[base + dim] = fmaf(-x1, cs.y, x0 * cs.x);
        values[base + dim + EI_HEAD_DIM / 2] = fmaf(x0, cs.y, x1 * cs.x);
    }
}

__global__ void qkv_norm_rope_f16_kernel(
    const float *combined, __half *q, __half *k, __half *v,
    const float *q_weight, const float *k_weight,
    const uint32_t *positions, const float2 *rope_table,
    uint32_t n_tokens, float eps) {
    constexpr uint32_t combined_tasks = EI_N_HEAD + EI_N_HEAD_KV + 1;
    constexpr uint32_t combined_stride = EI_N_EMBD + 2 * EI_HEAD_DIM;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * combined_tasks;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / combined_tasks);
    const uint32_t combined_task = static_cast<uint32_t>(task -
        static_cast<size_t>(token) * combined_tasks);
    const size_t input_base = static_cast<size_t>(token) * combined_stride;

    if (combined_task == combined_tasks - 1) {
        const size_t source = input_base + EI_N_EMBD + EI_HEAD_DIM;
        const size_t destination = static_cast<size_t>(token) * EI_HEAD_DIM;
        for (uint32_t dim = lane; dim < EI_HEAD_DIM; dim += 32) {
            v[destination + dim] = __float2half(combined[source + dim]);
        }
        return;
    }

    const bool key_head = combined_task == EI_N_HEAD;
    const uint32_t head = key_head ? 0 : combined_task;
    const size_t source = input_base + (key_head ? EI_N_EMBD :
        static_cast<size_t>(head) * EI_HEAD_DIM);
    const uint32_t output_stride = key_head ? EI_HEAD_DIM : EI_N_EMBD;
    const size_t destination = static_cast<size_t>(token) * output_stride +
                               static_cast<size_t>(head) * EI_HEAD_DIM;
    const float *weight = key_head ? k_weight : q_weight;
    __half *output = key_head ? k : q;
    float sum = 0.0f;
    for (uint32_t dim = lane; dim < EI_HEAD_DIM; dim += 32) {
        const float value = combined[source + dim];
        sum = fmaf(value, value, sum);
    }
    sum = warp_sum(sum);
    sum = __shfl_sync(kWarpMask, sum, 0);
    const float scale = key_head ? 1.0f : 0.0625f;
    const float inv = rsqrtf(sum / static_cast<float>(EI_HEAD_DIM) + eps) * scale;
    const size_t rope_base = static_cast<size_t>(positions[token]) *
                             (EI_HEAD_DIM / 2);
    for (uint32_t dim = lane; dim < EI_HEAD_DIM / 2; dim += 32) {
        const float2 cs = rope_table[rope_base + dim];
        const float x0 = combined[source + dim] * weight[dim] * inv;
        const float x1 = combined[source + dim + EI_HEAD_DIM / 2] *
                         weight[dim + EI_HEAD_DIM / 2] * inv;
        output[destination + dim] =
            __float2half(fmaf(-x1, cs.y, x0 * cs.x));
        output[destination + dim + EI_HEAD_DIM / 2] =
            __float2half(fmaf(x0, cs.y, x1 * cs.x));
    }
}

__global__ void attention_kernel(const float *q, const float *k,
                                 const float *v, float *output,
                                 const uint32_t *offsets,
                                 const uint32_t *sequence_ids,
                                 uint32_t n_tokens, uint32_t window) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * EI_N_HEAD;
    if (task >= task_count) return;
    const uint32_t query = static_cast<uint32_t>(task / EI_N_HEAD);
    const uint32_t head = static_cast<uint32_t>(task -
        static_cast<size_t>(query) * EI_N_HEAD);

    float q_values[EI_HEAD_DIM / 32];
    float out_values[EI_HEAD_DIM / 32];
#pragma unroll
    for (int item = 0; item < EI_HEAD_DIM / 32; item++) {
        const int dim = lane + item * 32;
        q_values[item] = q[static_cast<size_t>(query) * EI_N_EMBD +
                           head * EI_HEAD_DIM + dim];
        out_values[item] = 0.0f;
    }

    const uint32_t sequence = sequence_ids[query];
    const uint32_t sequence_start = offsets[sequence];
    const uint32_t sequence_stop = offsets[sequence + 1];
    uint32_t first = sequence_start;
    uint32_t last = sequence_stop;
    if (window != 0) {
        const uint32_t half_window = window / 2;
        first = query > sequence_start + half_window
            ? query - half_window : sequence_start;
        const uint32_t candidate = query + half_window + 1;
        last = candidate < sequence_stop ? candidate : sequence_stop;
    }

    float max_score = -__int_as_float(0x7f800000);
    float denominator = 0.0f;
    for (uint32_t key = first; key < last; key++) {
        float score = 0.0f;
#pragma unroll
        for (int item = 0; item < EI_HEAD_DIM / 32; item++) {
            const int dim = lane + item * 32;
            score = fmaf(q_values[item],
                         k[static_cast<size_t>(key) * EI_HEAD_DIM + dim], score);
        }
        score = warp_sum(score);
        score = __shfl_sync(kWarpMask, score, 0);
        const float next_max = fmaxf(max_score, score);
        const float alpha = __expf(max_score - next_max);
        const float beta = __expf(score - next_max);
        denominator = denominator * alpha + beta;
#pragma unroll
        for (int item = 0; item < EI_HEAD_DIM / 32; item++) {
            const int dim = lane + item * 32;
            out_values[item] = fmaf(beta,
                v[static_cast<size_t>(key) * EI_HEAD_DIM + dim],
                out_values[item] * alpha);
        }
        max_score = next_max;
    }

    const float inv_denominator = 1.0f / denominator;
#pragma unroll
    for (int item = 0; item < EI_HEAD_DIM / 32; item++) {
        const int dim = lane + item * 32;
        output[static_cast<size_t>(query) * EI_N_EMBD +
               head * EI_HEAD_DIM + dim] = out_values[item] * inv_denominator;
    }
}

__device__ __forceinline__ void store_attention_output(float *output,
                                                        size_t index,
                                                        float value) {
    output[index] = value;
}

__device__ __forceinline__ void store_attention_output(__half *output,
                                                        size_t index,
                                                        float value) {
    output[index] = __float2half(value);
}

template <typename Output>
__global__ void attention_f16_kernel(
    const __half *q, const __half *k, const __half *v, Output *output,
    const uint32_t *offsets, const uint32_t *sequence_ids,
    uint32_t n_tokens, uint32_t window) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * EI_N_HEAD;
    if (task >= task_count) return;
    const uint32_t query = static_cast<uint32_t>(task / EI_N_HEAD);
    const uint32_t head = static_cast<uint32_t>(task -
        static_cast<size_t>(query) * EI_N_HEAD);
    float q_values[EI_HEAD_DIM / 32];
    float out_values[EI_HEAD_DIM / 32];
#pragma unroll
    for (int item = 0; item < EI_HEAD_DIM / 32; item++) {
        const int dim = lane + item * 32;
        q_values[item] = __half2float(q[static_cast<size_t>(query) * EI_N_EMBD +
                                      head * EI_HEAD_DIM + dim]);
        out_values[item] = 0.0f;
    }
    const uint32_t sequence = sequence_ids[query];
    const uint32_t sequence_start = offsets[sequence];
    const uint32_t sequence_stop = offsets[sequence + 1];
    uint32_t first = sequence_start;
    uint32_t last = sequence_stop;
    if (window != 0) {
        const uint32_t half_window = window / 2;
        first = query > sequence_start + half_window
            ? query - half_window : sequence_start;
        last = min(sequence_stop, query + half_window + 1);
    }
    float max_score = -__int_as_float(0x7f800000);
    float denominator = 0.0f;
    for (uint32_t key = first; key < last; key++) {
        float score = 0.0f;
#pragma unroll
        for (int item = 0; item < EI_HEAD_DIM / 32; item++) {
            const int dim = lane + item * 32;
            score = fmaf(q_values[item],
                __half2float(k[static_cast<size_t>(key) * EI_HEAD_DIM + dim]),
                score);
        }
        score = warp_sum(score);
        score = __shfl_sync(kWarpMask, score, 0);
        const float next_max = fmaxf(max_score, score);
        const float alpha = __expf(max_score - next_max);
        const float beta = __expf(score - next_max);
        denominator = denominator * alpha + beta;
#pragma unroll
        for (int item = 0; item < EI_HEAD_DIM / 32; item++) {
            const int dim = lane + item * 32;
            out_values[item] = fmaf(beta,
                __half2float(v[static_cast<size_t>(key) * EI_HEAD_DIM + dim]),
                out_values[item] * alpha);
        }
        max_score = next_max;
    }
    const float inv_denominator = 1.0f / denominator;
#pragma unroll
    for (int item = 0; item < EI_HEAD_DIM / 32; item++) {
        const int dim = lane + item * 32;
        const size_t index = static_cast<size_t>(query) * EI_N_EMBD +
                             head * EI_HEAD_DIM + dim;
        store_attention_output(output, index,
                               out_values[item] * inv_denominator);
    }
}

// Flash-style packed attention: eight query warps share 16-key K/V tiles.
// Full and symmetric-SWA masks use the same online-softmax recurrence.
template <typename Output>
__global__ void flash_attention_f16_kernel(
    const __half *q, const __half *k, const __half *v, Output *output,
    const uint32_t *query_starts, const uint32_t *sequence_starts,
    const uint32_t *sequence_stops, uint32_t tile_count, uint32_t window) {
    constexpr uint32_t query_tile = 8;
    constexpr uint32_t key_tile = 16;
    __shared__ __half shared_k[key_tile * EI_HEAD_DIM];
    __shared__ __half shared_v[key_tile * EI_HEAD_DIM];
    const uint32_t tile = blockIdx.x;
    const uint32_t head = blockIdx.y;
    if (tile >= tile_count || head >= EI_N_HEAD) return;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t query_start = query_starts[tile];
    const uint32_t sequence_start = sequence_starts[tile];
    const uint32_t sequence_stop = sequence_stops[tile];
    const uint32_t query = query_start + static_cast<uint32_t>(warp);
    const bool active = query < sequence_stop;
    const uint32_t query_stop = min(sequence_stop, query_start + query_tile);

    uint32_t first = sequence_start;
    uint32_t last = sequence_stop;
    uint32_t union_first = sequence_start;
    uint32_t union_last = sequence_stop;
    if (window != 0) {
        const uint32_t half_window = window / 2;
        if (active) {
            first = query > sequence_start + half_window
                ? query - half_window : sequence_start;
            last = min(sequence_stop, query + half_window + 1);
        }
        union_first = query_start > sequence_start + half_window
            ? query_start - half_window : sequence_start;
        union_last = min(sequence_stop, query_stop + half_window);
    }

    float q_values[EI_HEAD_DIM / 32];
    float out_values[EI_HEAD_DIM / 32];
#pragma unroll
    for (int item = 0; item < EI_HEAD_DIM / 32; item++) {
        const int dim = lane + item * 32;
        q_values[item] = active
            ? __half2float(q[static_cast<size_t>(query) * EI_N_EMBD +
                             head * EI_HEAD_DIM + dim]) : 0.0f;
        out_values[item] = 0.0f;
    }
    float max_score = -__int_as_float(0x7f800000);
    float denominator = 0.0f;
    for (uint32_t key_base = union_first; key_base < union_last;
         key_base += key_tile) {
        const uint32_t keys = min(key_tile, union_last - key_base);
        for (uint32_t item = threadIdx.x; item < key_tile * EI_HEAD_DIM;
             item += blockDim.x) {
            const uint32_t local_key = item / EI_HEAD_DIM;
            const uint32_t dim = item - local_key * EI_HEAD_DIM;
            if (local_key < keys) {
                const size_t source = static_cast<size_t>(key_base + local_key) *
                                      EI_HEAD_DIM + dim;
                shared_k[item] = k[source];
                shared_v[item] = v[source];
            } else {
                shared_k[item] = __float2half(0.0f);
                shared_v[item] = __float2half(0.0f);
            }
        }
        __syncthreads();
        if (active) {
            for (uint32_t local_key = 0; local_key < keys; local_key++) {
                const uint32_t key = key_base + local_key;
                if (key < first || key >= last) continue;
                float score = 0.0f;
#pragma unroll
                for (int item = 0; item < EI_HEAD_DIM / 32; item++) {
                    const int dim = lane + item * 32;
                    score = fmaf(q_values[item],
                        __half2float(shared_k[local_key * EI_HEAD_DIM + dim]),
                        score);
                }
                score = warp_sum(score);
                score = __shfl_sync(kWarpMask, score, 0);
                const float next_max = fmaxf(max_score, score);
                const float alpha = __expf(max_score - next_max);
                const float beta = __expf(score - next_max);
                denominator = denominator * alpha + beta;
#pragma unroll
                for (int item = 0; item < EI_HEAD_DIM / 32; item++) {
                    const int dim = lane + item * 32;
                    out_values[item] = fmaf(beta,
                        __half2float(shared_v[local_key * EI_HEAD_DIM + dim]),
                        out_values[item] * alpha);
                }
                max_score = next_max;
            }
        }
        __syncthreads();
    }
    if (!active) return;
    const float inv_denominator = 1.0f / denominator;
#pragma unroll
    for (int item = 0; item < EI_HEAD_DIM / 32; item++) {
        const int dim = lane + item * 32;
        const size_t index = static_cast<size_t>(query) * EI_N_EMBD +
                             head * EI_HEAD_DIM + dim;
        store_attention_output(output, index,
                               out_values[item] * inv_denominator);
    }
}

__device__ __forceinline__ float block_max(float value, float *partials) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    value = warp_max(value);
    if (lane == 0) partials[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < (blockDim.x >> 5)
            ? partials[lane] : -__int_as_float(0x7f800000);
        value = warp_max(value);
        if (lane == 0) partials[0] = value;
    }
    __syncthreads();
    return partials[0];
}

__device__ __forceinline__ float block_sum(float value, float *partials) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    value = warp_sum(value);
    if (lane == 0) partials[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < (blockDim.x >> 5) ? partials[lane] : 0.0f;
        value = warp_sum(value);
        if (lane == 0) partials[0] = value;
    }
    __syncthreads();
    return partials[0];
}

__global__ void attention_softmax_f16_kernel(
    const float *scores, __half *probabilities, uint32_t query_start,
    uint32_t key_start, uint32_t query_count, uint32_t key_count,
    uint32_t sequence_tokens, uint32_t window) {
    __shared__ float partials[kWarpsPerBlock];
    const uint32_t local_query = blockIdx.x;
    if (local_query >= query_count) return;
    const uint32_t query = query_start + local_query;
    uint32_t first = 0;
    uint32_t last = sequence_tokens;
    if (window != 0) {
        const uint32_t half_window = window / 2;
        first = query > half_window ? query - half_window : 0;
        last = min(sequence_tokens, query + half_window + 1);
    }
    const uint32_t local_first = first - key_start;
    const uint32_t local_last = last - key_start;
    float local_max = -__int_as_float(0x7f800000);
    for (uint32_t key = local_first + threadIdx.x; key < local_last;
         key += blockDim.x) {
        local_max = fmaxf(local_max,
            scores[static_cast<size_t>(local_query) * key_count + key]);
    }
    const float maximum = block_max(local_max, partials);
    float local_sum = 0.0f;
    for (uint32_t key = local_first + threadIdx.x; key < local_last;
         key += blockDim.x) {
        local_sum += __expf(scores[
            static_cast<size_t>(local_query) * key_count + key] - maximum);
    }
    const float denominator = block_sum(local_sum, partials);
    for (uint32_t key = threadIdx.x; key < key_count; key += blockDim.x) {
        const float probability = key >= local_first && key < local_last
            ? __expf(scores[
                static_cast<size_t>(local_query) * key_count + key] - maximum) /
              denominator : 0.0f;
        probabilities[static_cast<size_t>(local_query) * key_count + key] =
            __float2half(probability);
    }
}

__global__ void pool_kernel(const float *input, const float *weight,
                            float *output, const uint32_t *offsets,
                            uint32_t batch_size, float eps) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t sequence = blockIdx.x * kWarpsPerBlock + warp;
    if (sequence >= batch_size) return;
    const uint32_t start = offsets[sequence];
    const uint32_t stop = offsets[sequence + 1];
    float pooled[EI_N_EMBD / 32];
#pragma unroll
    for (int item = 0; item < EI_N_EMBD / 32; item++) pooled[item] = 0.0f;

    for (uint32_t token = start; token < stop; token++) {
        float sum = 0.0f;
#pragma unroll
        for (int item = 0; item < EI_N_EMBD / 32; item++) {
            const int dim = lane + item * 32;
            const float value = input[static_cast<size_t>(token) * EI_N_EMBD + dim];
            sum = fmaf(value, value, sum);
        }
        sum = warp_sum(sum);
        sum = __shfl_sync(kWarpMask, sum, 0);
        const float inv = rsqrtf(sum / static_cast<float>(EI_N_EMBD) + eps);
#pragma unroll
        for (int item = 0; item < EI_N_EMBD / 32; item++) {
            const int dim = lane + item * 32;
            pooled[item] += input[static_cast<size_t>(token) * EI_N_EMBD + dim] *
                            weight[dim] * inv;
        }
    }

    const float inv_tokens = 1.0f / static_cast<float>(stop - start);
    float sum = 0.0f;
#pragma unroll
    for (int item = 0; item < EI_N_EMBD / 32; item++) {
        pooled[item] *= inv_tokens;
        sum = fmaf(pooled[item], pooled[item], sum);
    }
    sum = warp_sum(sum);
    sum = __shfl_sync(kWarpMask, sum, 0);
    const float inv_l2 = sum == 0.0f ? 1.0f : rsqrtf(sum);
#pragma unroll
    for (int item = 0; item < EI_N_EMBD / 32; item++) {
        const int dim = lane + item * 32;
        output[static_cast<size_t>(sequence) * EI_N_EMBD + dim] =
            pooled[item] * inv_l2;
    }
}

static size_t grow_capacity(size_t current, size_t required) {
    size_t capacity = current ? current : 16;
    while (capacity < required) capacity *= 2;
    return capacity;
}

static void release_token_workspace(cuda_engine *engine) {
    clear_graphs(engine);
    cuda_release(engine->ids);
    cuda_release(engine->positions);
    cuda_release(engine->sequence_ids);
    cuda_release(engine->x);
    cuda_release(engine->norm);
    cuda_release(engine->q);
    cuda_release(engine->k);
    cuda_release(engine->v);
    cuda_release(engine->qkv_combined);
    cuda_release(engine->q_half);
    cuda_release(engine->k_half);
    cuda_release(engine->v_half);
    cuda_release(engine->ctx);
    cuda_release(engine->up);
    cuda_release(engine->gate);
    cuda_release(engine->up_gate_combined);
    cuda_release(engine->tmp);
    cuda_release(engine->half_input);
    cuda_release(engine->q8_input);
    cuda_release(engine->flash_query_start);
    cuda_release(engine->flash_sequence_start);
    cuda_release(engine->flash_sequence_stop);
    engine->token_capacity = 0;
}

static void release_batch_workspace(cuda_engine *engine) {
    clear_graphs(engine);
    cuda_release(engine->offsets);
    cuda_release(engine->result);
    engine->batch_capacity = 0;
}

static bool ensure_capacity(cuda_engine *engine, size_t token_count,
                            size_t batch_size, char *err, size_t err_len) {
    if (token_count > engine->token_capacity) {
        const size_t capacity = grow_capacity(engine->token_capacity, token_count);
        release_token_workspace(engine);
        if (!cuda_allocate(&engine->ids, capacity, "allocate CUDA token IDs", err, err_len) ||
            !cuda_allocate(&engine->positions, capacity, "allocate CUDA positions", err, err_len) ||
            !cuda_allocate(&engine->sequence_ids, capacity, "allocate CUDA sequence IDs", err, err_len) ||
            !cuda_allocate(&engine->x, capacity * EI_N_EMBD, "allocate CUDA residual", err, err_len) ||
            !cuda_allocate(&engine->norm, capacity * EI_N_EMBD, "allocate CUDA norm", err, err_len) ||
            !cuda_allocate(&engine->q, capacity * EI_N_EMBD, "allocate CUDA Q", err, err_len) ||
            !cuda_allocate(&engine->k, capacity * EI_HEAD_DIM, "allocate CUDA K", err, err_len) ||
            !cuda_allocate(&engine->v, capacity * EI_HEAD_DIM, "allocate CUDA V", err, err_len) ||
            !cuda_allocate(&engine->qkv_combined,
                           capacity * (EI_N_EMBD + 2 * EI_HEAD_DIM),
                           "allocate CUDA combined QKV", err, err_len) ||
            !cuda_allocate(&engine->q_half, capacity * EI_N_EMBD,
                           "allocate CUDA FP16 Q", err, err_len) ||
            !cuda_allocate(&engine->k_half, capacity * EI_HEAD_DIM,
                           "allocate CUDA FP16 K", err, err_len) ||
            !cuda_allocate(&engine->v_half, capacity * EI_HEAD_DIM,
                           "allocate CUDA FP16 V", err, err_len) ||
            !cuda_allocate(&engine->ctx, capacity * EI_N_EMBD, "allocate CUDA attention", err, err_len) ||
            !cuda_allocate(&engine->up, capacity * EI_N_FF, "allocate CUDA FFN up", err, err_len) ||
            !cuda_allocate(&engine->gate, capacity * EI_N_FF, "allocate CUDA FFN", err, err_len) ||
            !cuda_allocate(&engine->up_gate_combined, capacity * (2 * EI_N_FF),
                           "allocate CUDA combined up/gate", err, err_len) ||
            !cuda_allocate(&engine->tmp, capacity * EI_N_EMBD, "allocate CUDA projection", err, err_len) ||
            !cuda_allocate(&engine->half_input, capacity * EI_N_FF,
                           "allocate CUDA tensor-core input", err, err_len) ||
            !cuda_allocate(&engine->q8_input, capacity * (EI_N_FF / kQK),
                           "allocate CUDA q8 activations", err, err_len) ||
            !cuda_allocate(&engine->flash_query_start, capacity,
                           "allocate CUDA attention query tiles", err, err_len) ||
            !cuda_allocate(&engine->flash_sequence_start, capacity,
                           "allocate CUDA attention sequence starts", err, err_len) ||
            !cuda_allocate(&engine->flash_sequence_stop, capacity,
                           "allocate CUDA attention sequence stops", err, err_len)) {
            release_token_workspace(engine);
            return false;
        }
        engine->token_capacity = capacity;
    }
    if (batch_size > engine->batch_capacity) {
        const size_t capacity = grow_capacity(engine->batch_capacity, batch_size);
        release_batch_workspace(engine);
        if (!cuda_allocate(&engine->offsets, capacity + 1, "allocate CUDA offsets", err, err_len) ||
            !cuda_allocate(&engine->result, capacity * EI_N_EMBD, "allocate CUDA result", err, err_len)) {
            release_batch_workspace(engine);
            return false;
        }
        engine->batch_capacity = capacity;
    }
    return true;
}

static const uint8_t *model_host_base(const cuda_engine *engine) {
    return engine->model->gguf.map + engine->model->gguf.data_off;
}

static const block_q4_0 *q4_weights(const cuda_engine *engine,
                                    const ei_tensor *tensor) {
    return reinterpret_cast<const block_q4_0 *>(engine->model_data + tensor->offset);
}

static const float *float_weights(const cuda_engine *engine,
                                  const float *host_pointer) {
    const ptrdiff_t offset = reinterpret_cast<const uint8_t *>(host_pointer) -
                             model_host_base(engine);
    return reinterpret_cast<const float *>(engine->model_data + offset);
}

static size_t tensor_elements(const ei_tensor *tensor) {
    return static_cast<size_t>(tensor->ne[0]) * tensor->ne[1];
}

static bool initialize_dequantized_weights(cuda_engine *engine,
                                           char *err, size_t err_len) {
    size_t total = 0;
    for (int layer_index = 0; layer_index < EI_N_LAYER; layer_index++) {
        const ei_layer *layer = &engine->model->layers[layer_index];
        total += tensor_elements(layer->attn_q);
        total += tensor_elements(layer->attn_k);
        total += tensor_elements(layer->attn_v);
        total += tensor_elements(layer->attn_output);
        total += tensor_elements(layer->ffn_up);
        total += tensor_elements(layer->ffn_gate);
        total += tensor_elements(layer->ffn_down);
    }
    if (!cuda_allocate(&engine->dequantized_weights, total,
                       "allocate CUDA tensor-core weights", err, err_len)) {
        return false;
    }

    __half *cursor = engine->dequantized_weights;
    for (int layer_index = 0; layer_index < EI_N_LAYER; layer_index++) {
        const ei_layer *layer = &engine->model->layers[layer_index];
        const ei_tensor *tensors[] = {
            layer->attn_q, layer->attn_k, layer->attn_v, layer->attn_output,
            layer->ffn_up, layer->ffn_gate, layer->ffn_down,
        };
        __half **outputs[] = {
            &engine->layers[layer_index].attn_q,
            &engine->layers[layer_index].attn_k,
            &engine->layers[layer_index].attn_v,
            &engine->layers[layer_index].attn_output,
            &engine->layers[layer_index].ffn_up,
            &engine->layers[layer_index].ffn_gate,
            &engine->layers[layer_index].ffn_down,
        };
        for (size_t tensor_index = 0;
             tensor_index < sizeof(tensors) / sizeof(tensors[0]); tensor_index++) {
            const ei_tensor *tensor = tensors[tensor_index];
            *outputs[tensor_index] = cursor;
            const size_t count = tensor_elements(tensor);
            dequantize_q4_kernel<<<static_cast<unsigned>((count + kThreads - 1) /
                kThreads), kThreads, 0, engine->stream>>>(
                    q4_weights(engine, tensor), cursor,
                    static_cast<uint32_t>(tensor->ne[1]),
                    static_cast<uint32_t>(tensor->ne[0]));
            cursor += count;
        }
    }
    return cuda_check(cudaGetLastError(), "dequantize CUDA tensor-core weights",
                      err, err_len);
}

static void convert_to_half(cuda_engine *engine, const float *input,
                            size_t count) {
    f32_to_f16_kernel<<<static_cast<unsigned>((count + kThreads - 1) / kThreads),
                        kThreads, 0, engine->stream>>>(
        input, engine->half_input, count);
}

static bool tensor_core_projection(cuda_engine *engine, const __half *weights,
                                   const __half *input, float *output,
                                   int rows, int cols, int tokens,
                                   char *err, size_t err_len) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublas_check(cublasGemmEx(
        engine->blas, CUBLAS_OP_T, CUBLAS_OP_N,
        rows, tokens, cols, &alpha,
        weights, CUDA_R_16F, cols,
        input, CUDA_R_16F, cols,
        &beta, output, CUDA_R_32F, rows,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "execute CUDA tensor-core projection", err, err_len);
}

static void native_q4_projection(cuda_engine *engine,
                                 const block_q4_0 *weights,
                                 const __half *input, float *output,
                                 int rows, int cols, int tokens) {
    const dim3 grid(static_cast<unsigned>((rows + 15) / 16),
                    static_cast<unsigned>((tokens + 15) / 16));
    q4_mma_projection_kernel<<<grid, 32, 0, engine->stream>>>(
        weights, input, output, tokens, rows, cols);
}

static bool tensor_core_attention(
    cuda_engine *engine, const std::vector<uint32_t> &offsets,
    uint32_t minimum_tokens, uint32_t window, bool output_f16,
    char *err, size_t err_len) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    for (size_t sequence = 0; sequence + 1 < offsets.size(); sequence++) {
        const uint32_t start = offsets[sequence];
        const int tokens = static_cast<int>(offsets[sequence + 1] - start);
        if (tokens < static_cast<int>(minimum_tokens)) continue;
        for (int head = 0; head < EI_N_HEAD; head++) {
            const uint32_t query_tile = window != 0 &&
                static_cast<uint32_t>(tokens) >=
                    engine->swa_tensor_banded_min_tokens &&
                engine->swa_tensor_tile_tokens != 0
                ? engine->swa_tensor_tile_tokens
                : static_cast<uint32_t>(tokens);
            for (uint32_t query_start = 0;
                 query_start < static_cast<uint32_t>(tokens);
                 query_start += query_tile) {
                const uint32_t query_count = min(query_tile,
                    static_cast<uint32_t>(tokens) - query_start);
                uint32_t key_start = 0;
                uint32_t key_stop = static_cast<uint32_t>(tokens);
                if (window != 0 && query_tile < static_cast<uint32_t>(tokens)) {
                    const uint32_t half_window = window / 2;
                    key_start = query_start > half_window
                        ? query_start - half_window : 0;
                    key_stop = min(static_cast<uint32_t>(tokens),
                                   query_start + query_count + half_window);
                }
                const uint32_t key_count = key_stop - key_start;
                const size_t output_offset =
                    static_cast<size_t>(start + query_start) * EI_N_EMBD +
                    head * EI_HEAD_DIM;
                void *output = output_f16
                    ? static_cast<void *>(engine->half_input + output_offset)
                    : static_cast<void *>(engine->ctx + output_offset);
                const cudaDataType_t output_type = output_f16
                    ? CUDA_R_16F : CUDA_R_32F;
                if (!cublas_check(cublasGemmEx(
                        engine->blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        static_cast<int>(key_count),
                        static_cast<int>(query_count), EI_HEAD_DIM, &alpha,
                        engine->k_half + static_cast<size_t>(start + key_start) *
                            EI_HEAD_DIM,
                        CUDA_R_16F, EI_HEAD_DIM,
                        engine->q_half +
                            static_cast<size_t>(start + query_start) * EI_N_EMBD +
                            head * EI_HEAD_DIM,
                        CUDA_R_16F, EI_N_EMBD,
                        &beta, engine->attention_scores, CUDA_R_32F,
                        static_cast<int>(key_count),
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                        "execute CUDA tensor-core QK", err, err_len)) {
                    return false;
                }
                attention_softmax_f16_kernel<<<query_count, kThreads, 0,
                    engine->stream>>>(engine->attention_scores,
                        engine->attention_probs, query_start, key_start,
                        query_count, key_count, static_cast<uint32_t>(tokens),
                        window);
                if (!cublas_check(cublasGemmEx(
                        engine->blas, CUBLAS_OP_N, CUBLAS_OP_N,
                        EI_HEAD_DIM, static_cast<int>(query_count),
                        static_cast<int>(key_count), &alpha,
                        engine->v_half + static_cast<size_t>(start + key_start) *
                            EI_HEAD_DIM,
                        CUDA_R_16F, EI_HEAD_DIM,
                        engine->attention_probs, CUDA_R_16F,
                        static_cast<int>(key_count),
                        &beta, output, output_type, EI_N_EMBD,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                        "execute CUDA tensor-core PV", err, err_len)) {
                    return false;
                }
            }
        }
    }
    return true;
}

static uint32_t tuned_tensor_attention_threshold(const cuda_engine *engine,
                                                 uint32_t batch_size) {
    uint32_t threshold = engine->tensor_attention_min_tokens;
    if (!engine->tensor_attention_tuned) return threshold;
    const uint32_t batch_threshold = batch_size == 1 ? 80
        : batch_size <= 4 ? 128 : 192;
    return threshold > batch_threshold ? threshold : batch_threshold;
}

static int projection_blocks(size_t tokens, uint32_t rows) {
    const size_t tasks = tokens * rows;
    return static_cast<int>((tasks + kWarpsPerBlock - 1) / kWarpsPerBlock);
}

static int row_blocks(size_t rows) {
    return static_cast<int>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
}

static bool launch_forward(cuda_engine *engine, uint32_t tokens,
                           uint32_t batch_size, uint32_t attention_tiles,
                           uint32_t max_sequence_tokens,
                           const std::vector<uint32_t> &host_offsets,
                           char *err, size_t err_len) {
    const ei_model *model = engine->model;
    const bool use_gemm = tokens >= engine->gemm_min_tokens;
    const bool use_fp16_attention =
        max_sequence_tokens >= engine->fp16_attention_min_tokens;
    const bool direct_fp16_qkv = use_gemm && use_fp16_attention &&
        !engine->native_q4_gemm && engine->direct_fp16_qkv;
    const bool direct_fp16_context = use_gemm && use_fp16_attention &&
        engine->direct_fp16_context;
    const uint32_t tensor_attention_threshold =
        tuned_tensor_attention_threshold(engine, batch_size);
    const block_q8_0 *embedding = reinterpret_cast<const block_q8_0 *>(
        engine->model_data + model->token_embd->offset);
    const size_t embedding_count = static_cast<size_t>(tokens) * EI_N_EMBD;
    embedding_kernel<<<static_cast<unsigned>((embedding_count + kThreads - 1) / kThreads),
                       kThreads, 0, engine->stream>>>(
        embedding, engine->ids, engine->x, tokens,
        std::sqrt(static_cast<float>(EI_N_EMBD)));

    for (int layer_index = 0; layer_index < EI_N_LAYER; layer_index++) {
        const ei_layer *layer = &model->layers[layer_index];
        if (use_gemm) {
            if (layer_index == 0) {
                rms_norm_f16_kernel<<<row_blocks(tokens), kThreads, 0,
                    engine->stream>>>(engine->x,
                        float_weights(engine, layer->attn_norm),
                        engine->half_input, tokens, EI_N_EMBD, model->rms_eps);
            }
            if (engine->native_q4_gemm) {
                native_q4_projection(engine, q4_weights(engine, layer->attn_q),
                                     engine->half_input, engine->q,
                                     EI_N_EMBD, EI_N_EMBD, tokens);
                native_q4_projection(engine, q4_weights(engine, layer->attn_k),
                                     engine->half_input, engine->k,
                                     EI_HEAD_DIM, EI_N_EMBD, tokens);
                native_q4_projection(engine, q4_weights(engine, layer->attn_v),
                                     engine->half_input, engine->v,
                                     EI_HEAD_DIM, EI_N_EMBD, tokens);
            } else {
                if (!tensor_core_projection(engine,
                        engine->layers[layer_index].attn_q,
                        engine->half_input, engine->qkv_combined,
                        EI_N_EMBD + 2 * EI_HEAD_DIM, EI_N_EMBD,
                        tokens, err, err_len)) return false;
                if (!direct_fp16_qkv) {
                    const size_t qkv_count = static_cast<size_t>(tokens) *
                        (EI_N_EMBD + 2 * EI_HEAD_DIM);
                    split_qkv_kernel<<<static_cast<unsigned>(
                        (qkv_count + kThreads - 1) / kThreads), kThreads, 0,
                        engine->stream>>>(engine->qkv_combined, engine->q,
                                          engine->k, engine->v, tokens);
                }
            }
        } else if (engine->q8_latency) {
            rms_norm_quantize_q8_kernel<<<row_blocks(tokens), kThreads, 0,
                engine->stream>>>(engine->x,
                    float_weights(engine, layer->attn_norm), engine->q8_input,
                    tokens, EI_N_EMBD, model->rms_eps);
            qkv_q8_projection_kernel<<<projection_blocks(tokens,
                EI_N_EMBD + 2 * EI_HEAD_DIM), kThreads, 0, engine->stream>>>(
                    q4_weights(engine, layer->attn_q),
                    q4_weights(engine, layer->attn_k),
                    q4_weights(engine, layer->attn_v), engine->q8_input,
                    engine->q, engine->k, engine->v, tokens);
        } else {
            rms_norm_kernel<false><<<row_blocks(tokens), kThreads, 0,
                engine->stream>>>(engine->x,
                    float_weights(engine, layer->attn_norm), engine->norm,
                    tokens, EI_N_EMBD, model->rms_eps);
            qkv_projection_kernel<<<projection_blocks(tokens,
                EI_N_EMBD + 2 * EI_HEAD_DIM), kThreads, 0, engine->stream>>>(
                    q4_weights(engine, layer->attn_q),
                    q4_weights(engine, layer->attn_k),
                    q4_weights(engine, layer->attn_v), engine->norm,
                    engine->q, engine->k, engine->v, tokens);
        }
        const float2 *rope = ei_layer_is_swa(model, layer_index)
            ? engine->rope_swa : engine->rope_full;
        if (direct_fp16_qkv) {
            qkv_norm_rope_f16_kernel<<<projection_blocks(tokens,
                EI_N_HEAD + EI_N_HEAD_KV + 1), kThreads, 0, engine->stream>>>(
                    engine->qkv_combined, engine->q_half, engine->k_half,
                    engine->v_half,
                    float_weights(engine, layer->attn_q_norm),
                    float_weights(engine, layer->attn_k_norm),
                    engine->positions, rope, tokens, model->rms_eps);
        } else {
            qk_norm_rope_kernel<<<projection_blocks(tokens,
                EI_N_HEAD + EI_N_HEAD_KV), kThreads, 0, engine->stream>>>(
                    engine->q, engine->k,
                    float_weights(engine, layer->attn_q_norm),
                    float_weights(engine, layer->attn_k_norm),
                    engine->positions, rope, tokens, model->rms_eps);
        }
        const uint32_t window = ei_layer_is_swa(model, layer_index)
            ? model->swa_window : 0;
        if (use_fp16_attention) {
            if (!direct_fp16_qkv) {
                const size_t qkv_count = static_cast<size_t>(tokens) *
                    (EI_N_EMBD + 2 * EI_HEAD_DIM);
                qkv_f32_to_f16_kernel<<<static_cast<unsigned>(
                    (qkv_count + kThreads - 1) / kThreads), kThreads, 0,
                    engine->stream>>>(engine->q, engine->k, engine->v,
                        engine->q_half, engine->k_half, engine->v_half, tokens);
            }
            if (engine->tensor_attention_tuned) {
                if (max_sequence_tokens >= tensor_attention_threshold &&
                    !tensor_core_attention(engine, host_offsets,
                        tensor_attention_threshold, window, direct_fp16_context,
                        err, err_len)) {
                    return false;
                }
                if (attention_tiles != 0) {
                    if (direct_fp16_context) {
                        flash_attention_f16_kernel<__half><<<dim3(
                            attention_tiles, EI_N_HEAD), kThreads, 0,
                            engine->stream>>>(engine->q_half, engine->k_half,
                                engine->v_half, engine->half_input,
                                engine->flash_query_start,
                                engine->flash_sequence_start,
                                engine->flash_sequence_stop, attention_tiles,
                                window);
                    } else {
                        flash_attention_f16_kernel<float><<<dim3(
                            attention_tiles, EI_N_HEAD), kThreads, 0,
                            engine->stream>>>(engine->q_half, engine->k_half,
                                engine->v_half, engine->ctx,
                                engine->flash_query_start,
                                engine->flash_sequence_start,
                                engine->flash_sequence_stop, attention_tiles,
                                window);
                    }
                }
            } else if (max_sequence_tokens >=
                           engine->tensor_attention_min_tokens) {
                if (!tensor_core_attention(engine, host_offsets,
                        engine->tensor_attention_min_tokens, window,
                        direct_fp16_context, err, err_len)) {
                    return false;
                }
            } else if (max_sequence_tokens >=
                           engine->flash_attention_min_tokens &&
                       max_sequence_tokens <=
                           engine->flash_attention_max_tokens) {
                if (direct_fp16_context) {
                    flash_attention_f16_kernel<__half><<<dim3(attention_tiles,
                        EI_N_HEAD), kThreads, 0, engine->stream>>>(
                            engine->q_half, engine->k_half, engine->v_half,
                            engine->half_input, engine->flash_query_start,
                            engine->flash_sequence_start,
                            engine->flash_sequence_stop, attention_tiles,
                            window);
                } else {
                    flash_attention_f16_kernel<float><<<dim3(attention_tiles,
                        EI_N_HEAD), kThreads, 0, engine->stream>>>(
                            engine->q_half, engine->k_half, engine->v_half,
                            engine->ctx, engine->flash_query_start,
                            engine->flash_sequence_start,
                            engine->flash_sequence_stop, attention_tiles,
                            window);
                }
            } else {
                if (direct_fp16_context) {
                    attention_f16_kernel<__half><<<projection_blocks(tokens,
                        EI_N_HEAD), kThreads, 0, engine->stream>>>(
                            engine->q_half, engine->k_half, engine->v_half,
                            engine->half_input, engine->offsets,
                            engine->sequence_ids, tokens, window);
                } else {
                    attention_f16_kernel<float><<<projection_blocks(tokens,
                        EI_N_HEAD), kThreads, 0, engine->stream>>>(
                            engine->q_half, engine->k_half, engine->v_half,
                            engine->ctx, engine->offsets,
                            engine->sequence_ids, tokens, window);
                }
            }
        } else {
            attention_kernel<<<projection_blocks(tokens, EI_N_HEAD),
                               kThreads, 0, engine->stream>>>(
                engine->q, engine->k, engine->v, engine->ctx,
                engine->offsets, engine->sequence_ids, tokens, window);
        }
        if (use_gemm) {
            if (!direct_fp16_context) {
                convert_to_half(engine, engine->ctx,
                                static_cast<size_t>(tokens) * EI_N_EMBD);
            }
            if (engine->native_q4_gemm) {
                native_q4_projection(engine,
                    q4_weights(engine, layer->attn_output), engine->half_input,
                    engine->tmp, EI_N_EMBD, EI_N_EMBD, tokens);
            } else if (!tensor_core_projection(engine,
                           engine->layers[layer_index].attn_output,
                           engine->half_input, engine->tmp, EI_N_EMBD,
                           EI_N_EMBD, tokens, err, err_len)) {
                return false;
            }
        } else if (engine->q8_latency) {
            quantize_q8_kernel<<<row_blocks(static_cast<size_t>(tokens) *
                (EI_N_EMBD / kQK)), kThreads, 0, engine->stream>>>(
                    engine->ctx, engine->q8_input, tokens, EI_N_EMBD);
            q4_q8_projection_kernel<<<projection_blocks(tokens, EI_N_EMBD),
                                      kThreads, 0, engine->stream>>>(
                q4_weights(engine, layer->attn_output), engine->q8_input,
                engine->tmp, EI_N_EMBD, EI_N_EMBD, tokens);
        } else {
            q4_projection_kernel<<<projection_blocks(tokens, EI_N_EMBD),
                                    kThreads, 0, engine->stream>>>(
                q4_weights(engine, layer->attn_output), engine->ctx,
                engine->tmp, EI_N_EMBD, EI_N_EMBD, tokens);
        }
        if (use_gemm) {
            rms_residual_next_f16_kernel<<<row_blocks(tokens), kThreads, 0,
                engine->stream>>>(engine->tmp,
                    float_weights(engine, layer->post_attention_norm),
                    engine->x, float_weights(engine, layer->ffn_norm),
                    engine->half_input, tokens, EI_N_EMBD, model->rms_eps);
            const size_t ffn_count = static_cast<size_t>(tokens) * EI_N_FF;
            if (engine->native_q4_gemm) {
                native_q4_projection(engine, q4_weights(engine, layer->ffn_up),
                                     engine->half_input, engine->up,
                                     EI_N_FF, EI_N_EMBD, tokens);
                native_q4_projection(engine, q4_weights(engine, layer->ffn_gate),
                                     engine->half_input, engine->gate,
                                     EI_N_FF, EI_N_EMBD, tokens);
                gelu_mul_kernel<<<static_cast<unsigned>(
                    (ffn_count + kThreads - 1) / kThreads), kThreads, 0,
                    engine->stream>>>(engine->gate, engine->up, ffn_count);
                convert_to_half(engine, engine->gate, ffn_count);
                native_q4_projection(engine, q4_weights(engine, layer->ffn_down),
                                     engine->half_input, engine->tmp,
                                     EI_N_EMBD, EI_N_FF, tokens);
            } else {
                if (!tensor_core_projection(engine,
                        engine->layers[layer_index].ffn_up,
                        engine->half_input, engine->up_gate_combined,
                        2 * EI_N_FF, EI_N_EMBD, tokens, err, err_len)) {
                    return false;
                }
                up_gate_gelu_to_f16_kernel<<<static_cast<unsigned>(
                    (ffn_count + kThreads - 1) / kThreads), kThreads, 0,
                    engine->stream>>>(engine->up_gate_combined,
                                      engine->half_input, tokens);
                if (!tensor_core_projection(engine,
                        engine->layers[layer_index].ffn_down,
                        engine->half_input, engine->tmp, EI_N_EMBD, EI_N_FF,
                        tokens, err, err_len)) return false;
            }
            if (layer_index + 1 < EI_N_LAYER) {
                rms_residual_next_f16_kernel<<<row_blocks(tokens), kThreads, 0,
                    engine->stream>>>(engine->tmp,
                        float_weights(engine, layer->post_ffw_norm), engine->x,
                        float_weights(engine,
                            model->layers[layer_index + 1].attn_norm),
                        engine->half_input, tokens, EI_N_EMBD, model->rms_eps);
            } else {
                rms_norm_kernel<true><<<row_blocks(tokens), kThreads, 0,
                    engine->stream>>>(engine->tmp,
                        float_weights(engine, layer->post_ffw_norm), engine->x,
                        tokens, EI_N_EMBD, model->rms_eps);
            }
        } else if (engine->q8_latency) {
            rms_norm_kernel<true><<<row_blocks(tokens), kThreads, 0,
                engine->stream>>>(engine->tmp,
                    float_weights(engine, layer->post_attention_norm),
                    engine->x, tokens, EI_N_EMBD, model->rms_eps);
            rms_norm_quantize_q8_kernel<<<row_blocks(tokens), kThreads, 0,
                engine->stream>>>(engine->x,
                    float_weights(engine, layer->ffn_norm), engine->q8_input,
                    tokens, EI_N_EMBD, model->rms_eps);
            up_gate_q8_gelu_kernel<<<projection_blocks(tokens, EI_N_FF),
                                     kThreads, 0, engine->stream>>>(
                q4_weights(engine, layer->ffn_up),
                q4_weights(engine, layer->ffn_gate), engine->q8_input,
                engine->gate, tokens);
            quantize_q8_kernel<<<row_blocks(static_cast<size_t>(tokens) *
                (EI_N_FF / kQK)), kThreads, 0, engine->stream>>>(
                    engine->gate, engine->q8_input, tokens, EI_N_FF);
            q4_q8_projection_kernel<<<projection_blocks(tokens, EI_N_EMBD),
                                      kThreads, 0, engine->stream>>>(
                q4_weights(engine, layer->ffn_down), engine->q8_input,
                engine->tmp, EI_N_EMBD, EI_N_FF, tokens);
            rms_norm_kernel<true><<<row_blocks(tokens), kThreads, 0,
                engine->stream>>>(engine->tmp,
                    float_weights(engine, layer->post_ffw_norm), engine->x,
                    tokens, EI_N_EMBD, model->rms_eps);
        } else {
            rms_norm_kernel<true><<<row_blocks(tokens), kThreads, 0,
                engine->stream>>>(engine->tmp,
                    float_weights(engine, layer->post_attention_norm),
                    engine->x, tokens, EI_N_EMBD, model->rms_eps);
            rms_norm_kernel<false><<<row_blocks(tokens), kThreads, 0,
                engine->stream>>>(engine->x,
                    float_weights(engine, layer->ffn_norm), engine->norm,
                    tokens, EI_N_EMBD, model->rms_eps);
            up_gate_gelu_kernel<<<projection_blocks(tokens, EI_N_FF),
                                  kThreads, 0, engine->stream>>>(
                q4_weights(engine, layer->ffn_up),
                q4_weights(engine, layer->ffn_gate), engine->norm,
                engine->gate, tokens);
            q4_projection_kernel<<<projection_blocks(tokens, EI_N_EMBD),
                                    kThreads, 0, engine->stream>>>(
                q4_weights(engine, layer->ffn_down), engine->gate,
                engine->tmp, EI_N_EMBD, EI_N_FF, tokens);
            rms_norm_kernel<true><<<row_blocks(tokens), kThreads, 0,
                engine->stream>>>(engine->tmp,
                    float_weights(engine, layer->post_ffw_norm), engine->x,
                    tokens, EI_N_EMBD, model->rms_eps);
        }
    }

    pool_kernel<<<row_blocks(batch_size), kThreads, 0, engine->stream>>>(
        engine->x, float_weights(engine, model->output_norm), engine->result,
        engine->offsets, batch_size, model->rms_eps);
    return cuda_check(cudaGetLastError(), "launch CUDA forward kernels", err, err_len);
}

static bool execute_forward(cuda_engine *engine, uint32_t tokens,
                            uint32_t batch_size, uint32_t attention_tiles,
                            uint32_t max_sequence_tokens, uint64_t shape_hash,
                            const std::vector<uint32_t> &host_offsets,
                            char *err, size_t err_len) {
    graph_entry *entry = nullptr;
    for (graph_entry &candidate : engine->graphs) {
        if (candidate.tokens == tokens && candidate.batch_size == batch_size &&
            candidate.shape_hash == shape_hash) {
            entry = &candidate;
            break;
        }
    }
    if (!entry) {
        if (engine->graphs.size() >= 16) {
            if (engine->graphs.front().executable) {
                cudaGraphExecDestroy(engine->graphs.front().executable);
            }
            engine->graphs.erase(engine->graphs.begin());
        }
        engine->graphs.push_back({tokens, batch_size, shape_hash, nullptr});
        return launch_forward(engine, tokens, batch_size, attention_tiles,
                              max_sequence_tokens, host_offsets, err, err_len);
    }
    if (entry->executable) {
        return cuda_check(cudaGraphLaunch(entry->executable, engine->stream),
                          "launch CUDA inference graph", err, err_len);
    }

    if (!cuda_check(cudaStreamBeginCapture(engine->stream,
                    cudaStreamCaptureModeThreadLocal),
                    "begin CUDA inference graph capture", err, err_len)) {
        return false;
    }
    if (!launch_forward(engine, tokens, batch_size, attention_tiles,
                        max_sequence_tokens, host_offsets, err, err_len)) {
        cudaGraph_t abandoned = nullptr;
        if (cudaStreamEndCapture(engine->stream, &abandoned) == cudaSuccess && abandoned) {
            cudaGraphDestroy(abandoned);
        }
        return false;
    }
    cudaGraph_t graph = nullptr;
    if (!cuda_check(cudaStreamEndCapture(engine->stream, &graph),
                    "finish CUDA inference graph capture", err, err_len)) {
        return false;
    }
    cudaGraphExec_t executable = nullptr;
    cudaError_t status = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    if (!cuda_check(status, "instantiate CUDA inference graph", err, err_len)) {
        return false;
    }
    entry->executable = executable;
    return cuda_check(cudaGraphLaunch(entry->executable, engine->stream),
                      "launch captured CUDA inference graph", err, err_len);
}

static bool embed_batch(cuda_engine *engine, const int32_t *ids,
                        const size_t *offsets, size_t batch_size, float *out,
                        char *err, size_t err_len) {
    const size_t token_count = offsets[batch_size];
    if (!ensure_capacity(engine, token_count, batch_size, err, err_len)) return false;

    std::vector<uint32_t> device_offsets(batch_size + 1);
    std::vector<uint32_t> positions(token_count);
    std::vector<uint32_t> sequence_ids(token_count);
    std::vector<uint32_t> flash_query_start;
    std::vector<uint32_t> flash_sequence_start;
    std::vector<uint32_t> flash_sequence_stop;
    uint32_t max_sequence_tokens = 0;
    const uint32_t tensor_attention_threshold =
        tuned_tensor_attention_threshold(engine,
                                          static_cast<uint32_t>(batch_size));
    uint64_t shape_hash = 1469598103934665603ull;
    for (size_t sequence = 0; sequence <= batch_size; sequence++) {
        device_offsets[sequence] = static_cast<uint32_t>(offsets[sequence]);
        shape_hash ^= static_cast<uint64_t>(offsets[sequence]);
        shape_hash *= 1099511628211ull;
    }
    for (size_t sequence = 0; sequence < batch_size; sequence++) {
        const uint32_t start = static_cast<uint32_t>(offsets[sequence]);
        const uint32_t stop = static_cast<uint32_t>(offsets[sequence + 1]);
        if (stop - start > max_sequence_tokens) {
            max_sequence_tokens = stop - start;
        }
        if (!engine->tensor_attention_tuned ||
            stop - start < tensor_attention_threshold) {
            for (uint32_t query = start; query < stop; query += kWarpsPerBlock) {
                flash_query_start.push_back(query);
                flash_sequence_start.push_back(start);
                flash_sequence_stop.push_back(stop);
            }
        }
        for (size_t token = offsets[sequence]; token < offsets[sequence + 1]; token++) {
            positions[token] = static_cast<uint32_t>(token - offsets[sequence]);
            sequence_ids[token] = static_cast<uint32_t>(sequence);
        }
    }

    if (!cuda_check(cudaMemcpyAsync(engine->ids, ids,
                    token_count * sizeof(*ids), cudaMemcpyHostToDevice,
                    engine->stream), "copy CUDA token IDs", err, err_len) ||
        !cuda_check(cudaMemcpyAsync(engine->offsets, device_offsets.data(),
                    (batch_size + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice,
                    engine->stream), "copy CUDA offsets", err, err_len) ||
        !cuda_check(cudaMemcpyAsync(engine->positions, positions.data(),
                    token_count * sizeof(uint32_t), cudaMemcpyHostToDevice,
                    engine->stream), "copy CUDA positions", err, err_len) ||
        !cuda_check(cudaMemcpyAsync(engine->sequence_ids, sequence_ids.data(),
                    token_count * sizeof(uint32_t), cudaMemcpyHostToDevice,
                    engine->stream), "copy CUDA sequence IDs", err, err_len) ||
        !cuda_check(cudaMemcpyAsync(engine->flash_query_start,
                    flash_query_start.data(),
                    flash_query_start.size() * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, engine->stream),
                    "copy CUDA attention query tiles", err, err_len) ||
        !cuda_check(cudaMemcpyAsync(engine->flash_sequence_start,
                    flash_sequence_start.data(),
                    flash_sequence_start.size() * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, engine->stream),
                    "copy CUDA attention sequence starts", err, err_len) ||
        !cuda_check(cudaMemcpyAsync(engine->flash_sequence_stop,
                    flash_sequence_stop.data(),
                    flash_sequence_stop.size() * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, engine->stream),
                    "copy CUDA attention sequence stops", err, err_len)) {
        return false;
    }
    if (!execute_forward(engine, static_cast<uint32_t>(token_count),
                         static_cast<uint32_t>(batch_size),
                         static_cast<uint32_t>(flash_query_start.size()),
                         max_sequence_tokens, shape_hash, device_offsets,
                         err, err_len)) {
        return false;
    }
    if (!cuda_check(cudaMemcpyAsync(out, engine->result,
                    batch_size * EI_N_EMBD * sizeof(float),
                    cudaMemcpyDeviceToHost, engine->stream),
                    "copy CUDA embeddings", err, err_len) ||
        !cuda_check(cudaStreamSynchronize(engine->stream),
                    "execute CUDA forward pass", err, err_len)) {
        return false;
    }
    if (err && err_len) err[0] = '\0';
    return true;
}

}  // namespace

extern "C" void *ei_cuda_engine_create(const ei_model *model,
                                         char *err, size_t err_len) {
    int device_count = 0;
    if (!cuda_check(cudaGetDeviceCount(&device_count), "enumerate CUDA devices",
                    err, err_len)) return nullptr;
    if (device_count == 0) {
        set_error(err, err_len, "no CUDA devices are available");
        return nullptr;
    }
    int device = 0;
    const char *device_env = std::getenv("EI_CUDA_DEVICE");
    if (device_env && *device_env) {
        char *end = nullptr;
        long parsed = std::strtol(device_env, &end, 10);
        if (*end != '\0' || parsed < 0 || parsed >= device_count) {
            set_error(err, err_len, "EI_CUDA_DEVICE is outside the available device range");
            return nullptr;
        }
        device = static_cast<int>(parsed);
    }
    if (!cuda_check(cudaSetDevice(device), "select CUDA device", err, err_len)) {
        return nullptr;
    }
    cudaDeviceProp device_properties;
    if (!cuda_check(cudaGetDeviceProperties(&device_properties, device),
                    "inspect CUDA device", err, err_len)) {
        return nullptr;
    }

    cuda_engine *engine = new cuda_engine;
    engine->model = model;
    engine->device = device;
    const char *gemm_min_env = std::getenv("EI_CUDA_GEMM_MIN_TOKENS");
    if (gemm_min_env && *gemm_min_env) {
        char *end = nullptr;
        long parsed = std::strtol(gemm_min_env, &end, 10);
        if (*end != '\0' || parsed < 1 || parsed > 65536) {
            set_error(err, err_len,
                      "EI_CUDA_GEMM_MIN_TOKENS must be an integer from 1 to 65536");
            delete engine;
            return nullptr;
        }
        engine->gemm_min_tokens = static_cast<uint32_t>(parsed);
    }
    const char *fp16_attention_env = std::getenv(
        "EI_CUDA_FP16_ATTENTION_MIN_TOKENS");
    if (fp16_attention_env && *fp16_attention_env) {
        char *end = nullptr;
        long parsed = std::strtol(fp16_attention_env, &end, 10);
        if (*end != '\0' || parsed < 1 || parsed > 65536) {
            set_error(err, err_len,
                "EI_CUDA_FP16_ATTENTION_MIN_TOKENS must be an integer from 1 to 65536");
            delete engine;
            return nullptr;
        }
        engine->fp16_attention_min_tokens = static_cast<uint32_t>(parsed);
    }
    const char *direct_fp16_qkv_env = std::getenv("EI_CUDA_DIRECT_FP16_QKV");
    if (direct_fp16_qkv_env && *direct_fp16_qkv_env) {
        if (std::strcmp(direct_fp16_qkv_env, "0") != 0 &&
            std::strcmp(direct_fp16_qkv_env, "1") != 0) {
            set_error(err, err_len, "EI_CUDA_DIRECT_FP16_QKV must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->direct_fp16_qkv = direct_fp16_qkv_env[0] == '1';
    }
    const char *direct_fp16_context_env = std::getenv(
        "EI_CUDA_DIRECT_FP16_CONTEXT");
    if (direct_fp16_context_env && *direct_fp16_context_env) {
        if (std::strcmp(direct_fp16_context_env, "0") != 0 &&
            std::strcmp(direct_fp16_context_env, "1") != 0) {
            set_error(err, err_len,
                "EI_CUDA_DIRECT_FP16_CONTEXT must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->direct_fp16_context = direct_fp16_context_env[0] == '1';
    }
    const char *flash_attention_env = std::getenv(
        "EI_CUDA_FLASH_ATTENTION_MIN_TOKENS");
    if (flash_attention_env && *flash_attention_env) {
        char *end = nullptr;
        long parsed = std::strtol(flash_attention_env, &end, 10);
        if (*end != '\0' || parsed < 1 || parsed > 65536) {
            set_error(err, err_len,
                "EI_CUDA_FLASH_ATTENTION_MIN_TOKENS must be an integer from 1 to 65536");
            delete engine;
            return nullptr;
        }
        engine->flash_attention_min_tokens = static_cast<uint32_t>(parsed);
    }
    const char *flash_attention_max_env = std::getenv(
        "EI_CUDA_FLASH_ATTENTION_MAX_TOKENS");
    if (flash_attention_max_env && *flash_attention_max_env) {
        char *end = nullptr;
        long parsed = std::strtol(flash_attention_max_env, &end, 10);
        if (*end != '\0' || parsed < 1 || parsed > 65536) {
            set_error(err, err_len,
                "EI_CUDA_FLASH_ATTENTION_MAX_TOKENS must be an integer from 1 to 65536");
            delete engine;
            return nullptr;
        }
        engine->flash_attention_max_tokens = static_cast<uint32_t>(parsed);
    }
    const char *tensor_attention_env = std::getenv(
        "EI_CUDA_TENSOR_ATTENTION_MIN_TOKENS");
    if (tensor_attention_env && *tensor_attention_env) {
        char *end = nullptr;
        long parsed = std::strtol(tensor_attention_env, &end, 10);
        if (*end != '\0' || parsed < 1 || parsed > 65536) {
            set_error(err, err_len,
                "EI_CUDA_TENSOR_ATTENTION_MIN_TOKENS must be an integer from 1 to 65536");
            delete engine;
            return nullptr;
        }
        engine->tensor_attention_min_tokens = static_cast<uint32_t>(parsed);
        engine->tensor_attention_tuned = false;
    }
    const char *swa_tile_env = std::getenv("EI_CUDA_SWA_TENSOR_TILE_TOKENS");
    if (swa_tile_env && *swa_tile_env) {
        char *end = nullptr;
        long parsed = std::strtol(swa_tile_env, &end, 10);
        if (*end != '\0' || (parsed != 0 && parsed != 128 && parsed != 256 &&
                             parsed != 512 && parsed != 1024)) {
            set_error(err, err_len,
                "EI_CUDA_SWA_TENSOR_TILE_TOKENS must be 0, 128, 256, 512, or 1024");
            delete engine;
            return nullptr;
        }
        engine->swa_tensor_tile_tokens = static_cast<uint32_t>(parsed);
    }
    const char *swa_min_env = std::getenv(
        "EI_CUDA_SWA_TENSOR_MIN_TOKENS");
    if (swa_min_env && *swa_min_env) {
        char *end = nullptr;
        long parsed = std::strtol(swa_min_env, &end, 10);
        if (*end != '\0' || parsed < 1 || parsed > 65536) {
            set_error(err, err_len,
                "EI_CUDA_SWA_TENSOR_MIN_TOKENS must be an integer from 1 to 65536");
            delete engine;
            return nullptr;
        }
        engine->swa_tensor_banded_min_tokens = static_cast<uint32_t>(parsed);
    }
    const char *native_q4_env = std::getenv("EI_CUDA_NATIVE_Q4_GEMM");
    if (native_q4_env && *native_q4_env) {
        if (std::strcmp(native_q4_env, "0") != 0 &&
            std::strcmp(native_q4_env, "1") != 0) {
            set_error(err, err_len,
                      "EI_CUDA_NATIVE_Q4_GEMM must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->native_q4_gemm = std::strcmp(native_q4_env, "1") == 0;
    }
    if (engine->native_q4_gemm && device_properties.major < 8) {
        set_error(err, err_len,
                  "EI_CUDA_NATIVE_Q4_GEMM requires compute capability 8.0 or newer");
        delete engine;
        return nullptr;
    }
    const char *q8_latency_env = std::getenv("EI_CUDA_Q8_LATENCY");
    if (q8_latency_env && *q8_latency_env) {
        if (std::strcmp(q8_latency_env, "0") != 0 &&
            std::strcmp(q8_latency_env, "1") != 0) {
            set_error(err, err_len, "EI_CUDA_Q8_LATENCY must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->q8_latency = std::strcmp(q8_latency_env, "1") == 0;
    }
    if (!cuda_check(cudaStreamCreateWithFlags(&engine->stream, cudaStreamNonBlocking),
                    "create CUDA stream", err, err_len)) {
        delete engine;
        return nullptr;
    }
    if (!cublas_check(cublasCreate(&engine->blas), "create cuBLAS handle",
                      err, err_len) ||
        !cublas_check(cublasSetStream(engine->blas, engine->stream),
                      "set cuBLAS stream", err, err_len) ||
        !cublas_check(cublasSetMathMode(engine->blas, CUBLAS_TENSOR_OP_MATH),
                      "enable cuBLAS tensor cores", err, err_len)) {
        ei_cuda_engine_free(engine);
        return nullptr;
    }
    const uint8_t *model_bytes = model->gguf.map + model->gguf.data_off;
    const size_t model_length = model->gguf.map_len - model->gguf.data_off;
    if (!cuda_allocate(&engine->model_data, model_length,
                       "allocate CUDA model", err, err_len) ||
        !cuda_check(cudaMemcpyAsync(engine->model_data, model_bytes, model_length,
                    cudaMemcpyHostToDevice, engine->stream),
                    "copy CUDA model", err, err_len) ||
        (!engine->native_q4_gemm &&
         !initialize_dequantized_weights(engine, err, err_len))) {
        ei_cuda_engine_free(engine);
        return nullptr;
    }

    const size_t rope_count = static_cast<size_t>(EI_N_CTX) * (EI_HEAD_DIM / 2);
    std::vector<float2> full(rope_count);
    std::vector<float2> swa(rope_count);
    for (uint32_t position = 0; position < EI_N_CTX; position++) {
        for (uint32_t dim = 0; dim < EI_HEAD_DIM / 2; dim++) {
            const float exponent = -2.0f * static_cast<float>(dim) /
                                   static_cast<float>(EI_HEAD_DIM);
            const float full_theta = static_cast<float>(position) *
                                     std::pow(model->rope_base_full, exponent);
            const float swa_theta = static_cast<float>(position) *
                                    std::pow(model->rope_base_swa, exponent);
            full[static_cast<size_t>(position) * (EI_HEAD_DIM / 2) + dim] =
                make_float2(std::cos(full_theta), std::sin(full_theta));
            swa[static_cast<size_t>(position) * (EI_HEAD_DIM / 2) + dim] =
                make_float2(std::cos(swa_theta), std::sin(swa_theta));
        }
    }
    if (!cuda_allocate(&engine->rope_full, rope_count,
                       "allocate CUDA full RoPE table", err, err_len) ||
        !cuda_allocate(&engine->rope_swa, rope_count,
                       "allocate CUDA SWA RoPE table", err, err_len) ||
        !cuda_allocate(&engine->attention_scores,
                       static_cast<size_t>(EI_N_CTX) * EI_N_CTX,
                       "allocate CUDA attention scores", err, err_len) ||
        !cuda_allocate(&engine->attention_probs,
                       static_cast<size_t>(EI_N_CTX) * EI_N_CTX,
                       "allocate CUDA attention probabilities", err, err_len) ||
        !cuda_check(cudaMemcpyAsync(engine->rope_full, full.data(),
                    rope_count * sizeof(float2), cudaMemcpyHostToDevice,
                    engine->stream), "copy CUDA full RoPE table", err, err_len) ||
        !cuda_check(cudaMemcpyAsync(engine->rope_swa, swa.data(),
                    rope_count * sizeof(float2), cudaMemcpyHostToDevice,
                    engine->stream), "copy CUDA SWA RoPE table", err, err_len) ||
        !cuda_check(cudaStreamSynchronize(engine->stream),
                    "initialize CUDA backend", err, err_len)) {
        ei_cuda_engine_free(engine);
        return nullptr;
    }
    if (err && err_len) err[0] = '\0';
    return engine;
}

extern "C" void ei_cuda_engine_free(void *handle) {
    cuda_engine *engine = static_cast<cuda_engine *>(handle);
    if (!engine) return;
    cudaSetDevice(engine->device);
    if (engine->stream) cudaStreamSynchronize(engine->stream);
    release_token_workspace(engine);
    release_batch_workspace(engine);
    cuda_release(engine->rope_swa);
    cuda_release(engine->rope_full);
    cuda_release(engine->attention_probs);
    cuda_release(engine->attention_scores);
    cuda_release(engine->dequantized_weights);
    cuda_release(engine->model_data);
    if (engine->blas) cublasDestroy(engine->blas);
    if (engine->stream) cudaStreamDestroy(engine->stream);
    delete engine;
}

extern "C" bool ei_cuda_engine_reserve(void *handle, size_t total_tokens,
                                        size_t batch_size, char *err,
                                        size_t err_len) {
    cuda_engine *engine = static_cast<cuda_engine *>(handle);
    if (!engine) {
        set_error(err, err_len, "CUDA backend is not initialized");
        return false;
    }
    if (!cuda_check(cudaSetDevice(engine->device), "select CUDA device",
                    err, err_len)) return false;
    const bool ok = ensure_capacity(engine, total_tokens, batch_size, err, err_len);
    if (ok && err && err_len) err[0] = '\0';
    return ok;
}

extern "C" bool ei_cuda_engine_embed_tokens(void *handle,
                                             const int32_t *ids,
                                             size_t n_tokens,
                                             float out[EI_N_EMBD],
                                             char *err, size_t err_len) {
    const size_t offsets[2] = {0, n_tokens};
    return ei_cuda_engine_embed_tokens_batch(handle, ids, offsets, 1, out,
                                             err, err_len);
}

extern "C" bool ei_cuda_engine_embed_tokens_batch(void *handle,
                                                    const int32_t *ids,
                                                    const size_t *offsets,
                                                    size_t batch_size,
                                                    float *out, char *err,
                                                    size_t err_len) {
    cuda_engine *engine = static_cast<cuda_engine *>(handle);
    if (!engine) {
        set_error(err, err_len, "CUDA backend is not initialized");
        return false;
    }
    if (!cuda_check(cudaSetDevice(engine->device), "select CUDA device",
                    err, err_len)) return false;
    return embed_batch(engine, ids, offsets, batch_size, out, err, err_len);
}
