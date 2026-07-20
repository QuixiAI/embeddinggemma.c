extern "C" {
#include "engine_rocm.h"
}

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

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
    hipGraphExec_t executable;
};

struct rocm_engine {
    const ei_model *model = nullptr;
    int device = 0;
    hipStream_t stream = nullptr;
    hipblasHandle_t blas = nullptr;

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
    uint32_t gemm_min_tokens = 32;

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
    uint32_t *mfma_query_start = nullptr;
    uint32_t *mfma_sequence_start = nullptr;
    uint32_t *mfma_sequence_stop = nullptr;
    float *result = nullptr;
    int32_t *host_ids = nullptr;
    uint32_t *host_offsets = nullptr;
    float *host_result = nullptr;
    std::vector<graph_entry> graphs;
    bool command_graph = false;
    uint32_t fp16_attention_min_tokens = 5;
    uint32_t flash_attention_min_tokens = 5;
    uint32_t flash_attention_max_tokens = 192;
    uint32_t tensor_attention_min_tokens = 96;
    bool tensor_attention_tuned = true;
    uint32_t swa_tensor_tile_tokens = 0;
    uint32_t swa_tensor_banded_min_tokens = 1536;
    bool direct_fp16_qkv = true;
    bool direct_fp16_context = true;
    bool mfma_attention = false;
    bool batched_tensor_attention = true;
    bool fp16_attention_scores = true;
    uint32_t fp16_attention_scores_max_tokens = 896;
    uint32_t tensor_softmax_threads = 256;
    uint32_t mfma_attention_min_tokens = 16;
    uint32_t mfma_attention_max_tokens = 255;
    bool native_q4_gemm = true;
    uint32_t native_q4_max_tokens = 368;
    bool native_q4_fused = true;
    bool native_q4_fused_activation = true;
    uint32_t native_q4_fused_activation_min_tokens = 320;
    bool native_q4_wide = false;
    bool native_q4_direct_fp16_qkv = false;
    bool rms_register_cache = true;
    int compute_units = 1;
    bool q8_latency = false;
    bool q8_two_row = false;
    bool single_token_v_only = true;
    bool direct_rms_fusion = true;
    bool direct_q4_pair = true;
    bool fused_embedding_rms = true;
    bool singleton_metadata_elision = true;
    bool final_singleton_pool = true;
    uint32_t singleton_direct_max_tokens = 72;
    bool direct_q4_quad = false;
    bool pinned_io_staging = false;
    float *attention_scores = nullptr;
    __half *attention_probs = nullptr;
};

static void set_error(char *err, size_t err_len, const char *message) {
    if (err && err_len) std::snprintf(err, err_len, "%s", message);
}

static bool hip_check(hipError_t status, const char *operation,
                       char *err, size_t err_len) {
    if (status == hipSuccess) return true;
    if (err && err_len) {
        std::snprintf(err, err_len, "%s: %s", operation,
                      hipGetErrorString(status));
    }
    return false;
}

static bool hipblas_check(hipblasStatus_t status, const char *operation,
                         char *err, size_t err_len) {
    if (status == HIPBLAS_STATUS_SUCCESS) return true;
    if (err && err_len) {
        std::snprintf(err, err_len, "%s: hipBLAS status %d", operation,
                      static_cast<int>(status));
    }
    return false;
}

template <typename T>
static void hip_release(T *&ptr) {
    if (ptr) (void)hipFree(ptr);
    ptr = nullptr;
}

template <typename T>
static void hip_host_release(T *&ptr) {
    if (ptr) (void)hipHostFree(ptr);
    ptr = nullptr;
}

template <typename T>
static bool hip_host_allocate(T **ptr, size_t count, const char *name,
                              char *err, size_t err_len) {
    return hip_check(hipHostMalloc(reinterpret_cast<void **>(ptr),
                                   count * sizeof(T), hipHostMallocDefault),
                     name, err, err_len);
}

static void clear_graphs(rocm_engine *engine) {
    for (const graph_entry &entry : engine->graphs) {
        if (entry.executable) (void)hipGraphExecDestroy(entry.executable);
    }
    engine->graphs.clear();
}

template <typename T>
static bool hip_allocate(T **ptr, size_t count, const char *name,
                          char *err, size_t err_len) {
    hipError_t status = hipMalloc(reinterpret_cast<void **>(ptr),
                                    count * sizeof(T));
    return hip_check(status, name, err, err_len);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, 32);
    }
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down(value, offset, 32));
    }
    return value;
}

__device__ __forceinline__ int warp_sum_int(int value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, 32);
    }
    return value;
}

__device__ __forceinline__ uint32_t load_u32_unaligned(const uint8_t *bytes) {
    return static_cast<uint32_t>(bytes[0]) |
           (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) |
           (static_cast<uint32_t>(bytes[3]) << 24);
}

__device__ __forceinline__ int sdot4(int a, int b, int accumulator) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    return __builtin_amdgcn_sdot4(a, b, accumulator, false);
#else
    const int8_t *av = reinterpret_cast<const int8_t *>(&a);
    const int8_t *bv = reinterpret_cast<const int8_t *>(&b);
    for (int i = 0; i < 4; i++) accumulator += av[i] * bv[i];
    return accumulator;
#endif
}

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

__device__ __forceinline__ void q4_dot_pair(
    const block_q4_0 *first_weights, const block_q4_0 *second_weights,
    const float *input, int n_cols, int lane,
    float &first_sum, float &second_sum) {
    const int blocks_per_row = n_cols / kQK;
    const int block_offset = lane >> 1;
    const int byte_start = (lane & 1) * 8;
    first_sum = 0.0f;
    second_sum = 0.0f;
    for (int block = block_offset; block < blocks_per_row; block += 16) {
        const block_q4_0 &first = first_weights[block];
        const block_q4_0 &second = second_weights[block];
        const float first_d = __half2float(first.d);
        const float second_d = __half2float(second.d);
        const int x0 = block * kQK + byte_start;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float low_input = input[x0 + i];
            const float high_input = input[x0 + i + 16];
            const uint8_t first_code = first.qs[byte_start + i];
            const uint8_t second_code = second.qs[byte_start + i];
            first_sum = fmaf(first_d * static_cast<float>(
                (first_code & 0x0f) - 8), low_input, first_sum);
            first_sum = fmaf(first_d * static_cast<float>(
                (first_code >> 4) - 8), high_input, first_sum);
            second_sum = fmaf(second_d * static_cast<float>(
                (second_code & 0x0f) - 8), low_input, second_sum);
            second_sum = fmaf(second_d * static_cast<float>(
                (second_code >> 4) - 8), high_input, second_sum);
        }
    }
    first_sum = warp_sum(first_sum);
    second_sum = warp_sum(second_sum);
}

__device__ __forceinline__ void q4_dot_quad(
    const block_q4_0 *weights0, const block_q4_0 *weights1,
    const block_q4_0 *weights2, const block_q4_0 *weights3,
    const float *input, int n_cols, int lane,
    float &sum0, float &sum1, float &sum2, float &sum3) {
    const int blocks_per_row = n_cols / kQK;
    const int block_offset = lane >> 1;
    const int byte_start = (lane & 1) * 8;
    sum0 = 0.0f;
    sum1 = 0.0f;
    sum2 = 0.0f;
    sum3 = 0.0f;
    for (int block = block_offset; block < blocks_per_row; block += 16) {
        const block_q4_0 &packed0 = weights0[block];
        const block_q4_0 &packed1 = weights1[block];
        const block_q4_0 &packed2 = weights2[block];
        const block_q4_0 &packed3 = weights3[block];
        const float d0 = __half2float(packed0.d);
        const float d1 = __half2float(packed1.d);
        const float d2 = __half2float(packed2.d);
        const float d3 = __half2float(packed3.d);
        const int x0 = block * kQK + byte_start;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float low = input[x0 + i];
            const float high = input[x0 + i + 16];
            const uint8_t code0 = packed0.qs[byte_start + i];
            const uint8_t code1 = packed1.qs[byte_start + i];
            const uint8_t code2 = packed2.qs[byte_start + i];
            const uint8_t code3 = packed3.qs[byte_start + i];
            sum0 = fmaf(d0 * static_cast<float>((code0 & 15) - 8), low, sum0);
            sum0 = fmaf(d0 * static_cast<float>((code0 >> 4) - 8), high, sum0);
            sum1 = fmaf(d1 * static_cast<float>((code1 & 15) - 8), low, sum1);
            sum1 = fmaf(d1 * static_cast<float>((code1 >> 4) - 8), high, sum1);
            sum2 = fmaf(d2 * static_cast<float>((code2 & 15) - 8), low, sum2);
            sum2 = fmaf(d2 * static_cast<float>((code2 >> 4) - 8), high, sum2);
            sum3 = fmaf(d3 * static_cast<float>((code3 & 15) - 8), low, sum3);
            sum3 = fmaf(d3 * static_cast<float>((code3 >> 4) - 8), high, sum3);
        }
    }
    sum0 = warp_sum(sum0);
    sum1 = warp_sum(sum1);
    sum2 = warp_sum(sum2);
    sum3 = warp_sum(sum3);
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
            const uint32_t packed = load_u32_unaligned(q4.qs + group * 4);
            const int low = static_cast<int>(packed & 0x0f0f0f0fu);
            const int high = static_cast<int>((packed >> 4) & 0x0f0f0f0fu);
            const int q8_low = *reinterpret_cast<const int *>(q8.qs + group * 4);
            const int q8_high = *reinterpret_cast<const int *>(q8.qs + 16 + group * 4);
            dot = sdot4(low, q8_low, dot);
            dot = sdot4(high, q8_high, dot);
        }
        dot -= 8 * static_cast<int>(q8.sum);
        sum = fmaf(__half2float(q4.d) * __half2float(q8.d),
                   static_cast<float>(dot), sum);
    }
    return warp_sum(sum);
}

__device__ __forceinline__ void q4_q8_dot_pair(
    const block_q4_0 *first_weights, const block_q4_0 *second_weights,
    const activation_q8 *input, int n_cols, int lane,
    float &first_sum, float &second_sum) {
    const int blocks_per_row = n_cols / kQK;
    first_sum = 0.0f;
    second_sum = 0.0f;
    for (int block = lane; block < blocks_per_row; block += 32) {
        const block_q4_0 &first_q4 = first_weights[block];
        const block_q4_0 &second_q4 = second_weights[block];
        const activation_q8 &q8 = input[block];
        int first_dot = 0;
        int second_dot = 0;
#pragma unroll
        for (int group = 0; group < 4; group++) {
            const uint32_t first_packed = load_u32_unaligned(
                first_q4.qs + group * 4);
            const uint32_t second_packed = load_u32_unaligned(
                second_q4.qs + group * 4);
            const int q8_low = *reinterpret_cast<const int *>(
                q8.qs + group * 4);
            const int q8_high = *reinterpret_cast<const int *>(
                q8.qs + 16 + group * 4);
            first_dot = sdot4(
                static_cast<int>(first_packed & 0x0f0f0f0fu), q8_low,
                first_dot);
            first_dot = sdot4(
                static_cast<int>((first_packed >> 4) & 0x0f0f0f0fu),
                q8_high, first_dot);
            second_dot = sdot4(
                static_cast<int>(second_packed & 0x0f0f0f0fu), q8_low,
                second_dot);
            second_dot = sdot4(
                static_cast<int>((second_packed >> 4) & 0x0f0f0f0fu),
                q8_high, second_dot);
        }
        const int bias = 8 * static_cast<int>(q8.sum);
        first_dot -= bias;
        second_dot -= bias;
        const float input_scale = __half2float(q8.d);
        first_sum = fmaf(__half2float(first_q4.d) * input_scale,
                         static_cast<float>(first_dot), first_sum);
        second_sum = fmaf(__half2float(second_q4.d) * input_scale,
                          static_cast<float>(second_dot), second_sum);
    }
    first_sum = warp_sum(first_sum);
    second_sum = warp_sum(second_sum);
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
    amax = __shfl(amax, 0, 32);
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
    ss = __shfl(ss, 0, 32);
    const float inv = rsqrtf(ss / static_cast<float>(n_cols) + eps);
    const uint32_t blocks_per_row = n_cols / kQK;
    for (uint32_t block_index = 0; block_index < blocks_per_row; block_index++) {
        const uint32_t col = block_index * kQK + lane;
        const float value = input[base + col] * weight[col] * inv;
        float amax = warp_max(fabsf(value));
        amax = __shfl(amax, 0, 32);
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

template <bool OutputF16>
__global__ void embedding_rms_kernel(
    const block_q8_0 *table, const int32_t *ids, const float *weight,
    float *residual, float *f32_output, __half *f16_output,
    uint32_t n_tokens, float scale, float eps) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t token = blockIdx.x * kWarpsPerBlock + warp;
    if (token >= n_tokens) return;
    constexpr int values_per_lane = EI_N_EMBD / 32;
    constexpr int blocks_per_row = EI_N_EMBD / kQK;
    const block_q8_0 *row = table +
        static_cast<size_t>(ids[token]) * blocks_per_row;
    float values[values_per_lane];
    float sum = 0.0f;
#pragma unroll
    for (int item = 0; item < values_per_lane; item++) {
        const int col = lane + item * 32;
        const block_q8_0 &block = row[col / kQK];
        const float value = scale * __half2float(block.d) *
            static_cast<float>(block.qs[col % kQK]);
        values[item] = value;
        sum = fmaf(value, value, sum);
    }
    sum = warp_sum(sum);
    const float inv = rsqrtf(__shfl(sum, 0, 32) /
        static_cast<float>(EI_N_EMBD) + eps);
    const size_t base = static_cast<size_t>(token) * EI_N_EMBD;
#pragma unroll
    for (int item = 0; item < values_per_lane; item++) {
        const int col = lane + item * 32;
        const float value = values[item];
        residual[base + col] = value;
        const float normalized = value * weight[col] * inv;
        if constexpr (OutputF16) {
            f16_output[base + col] = __float2half(normalized);
        } else {
            f32_output[base + col] = normalized;
        }
    }
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

typedef __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16 half4_t;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float float4_t;

__device__ __forceinline__ half4_t load_input_mfma_fragment(
    const __half *input, int n_tokens, int n_cols,
    int token_start, int col_start) {
    const int lane = threadIdx.x & 63;
    const int token = token_start + (lane & 15);
    const int col = col_start + (lane >> 4) * 4;
    if (token < n_tokens) {
        return *reinterpret_cast<const half4_t *>(
            input + static_cast<size_t>(token) * n_cols + col);
    }
    return half4_t{0, 0, 0, 0};
}

__device__ __forceinline__ half4_t load_q4_mfma_fragment(
    const block_q4_0 *weights, int n_rows, int n_cols,
    int row_start, int col_start) {
    const int lane = threadIdx.x & 63;
    const int row = row_start + (lane & 15);
    const int col = col_start + (lane >> 4) * 4;
    if (row >= n_rows) return half4_t{0, 0, 0, 0};
    const int blocks_per_row = n_cols / kQK;
    const block_q4_0 &packed = weights[
        static_cast<size_t>(row) * blocks_per_row + col / kQK];
    const int block_col = col % kQK;
    const float scale = __half2float(packed.d);
    half4_t fragment;
#pragma unroll
    for (int item = 0; item < 4; item++) {
        const int quant_col = block_col + item;
        const uint8_t code = packed.qs[
            quant_col < 16 ? quant_col : quant_col - 16];
        const int quant = quant_col < 16
            ? (code & 0x0f) - 8 : (code >> 4) - 8;
        fragment[item] = static_cast<__fp16>(scale * static_cast<float>(quant));
    }
    return fragment;
}

__device__ __forceinline__ float4_t mfma_m16n16k16(
    half4_t input, half4_t weights, float4_t accumulator) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(
        input, weights, accumulator, 0, 0, 0);
}

__global__ void q4_mma_projection_kernel(
    const block_q4_0 *weights, const __half *input, float *output,
    int n_tokens, int n_rows, int n_cols) {
    const int row_start = static_cast<int>(blockIdx.x) * 16;
    const int token_start = static_cast<int>(blockIdx.y) * 16;
    float4_t accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int col_start = 0; col_start < n_cols; col_start += 16) {
        const half4_t input_fragment = load_input_mfma_fragment(
            input, n_tokens, n_cols, token_start, col_start);
        const half4_t weight_fragment = load_q4_mfma_fragment(
            weights, n_rows, n_cols, row_start, col_start);
        accumulator = mfma_m16n16k16(
            input_fragment, weight_fragment, accumulator);
    }
    const int lane = threadIdx.x & 63;
    const int row = row_start + (lane & 15);
    const int token = token_start + (lane >> 4) * 4;
#pragma unroll
    for (int item = 0; item < 4; item++) {
        if (row < n_rows && token + item < n_tokens) {
            output[static_cast<size_t>(token + item) * n_rows + row] =
                accumulator[item];
        }
    }
}

__global__ void q4_mfma_v_only_projection_kernel(
    const block_q4_0 *weights, const __half *input, __half *output,
    int n_tokens) {
    const int row_start = static_cast<int>(blockIdx.x) * 16;
    const int token_start = static_cast<int>(blockIdx.y) * 16;
    float4_t accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int col_start = 0; col_start < EI_N_EMBD; col_start += 16) {
        const half4_t input_fragment = load_input_mfma_fragment(
            input, n_tokens, EI_N_EMBD, token_start, col_start);
        const half4_t weight_fragment = load_q4_mfma_fragment(
            weights, EI_HEAD_DIM, EI_N_EMBD, row_start, col_start);
        accumulator = mfma_m16n16k16(
            input_fragment, weight_fragment, accumulator);
    }
    const int lane = threadIdx.x & 63;
    const int row = row_start + (lane & 15);
    const int token = token_start + (lane >> 4) * 4;
#pragma unroll
    for (int item = 0; item < 4; item++) {
        if (row < EI_HEAD_DIM && token + item < n_tokens) {
            const __half value = __float2half(accumulator[item]);
#pragma unroll
            for (int head = 0; head < EI_N_HEAD; head++) {
                output[static_cast<size_t>(token + item) * EI_N_EMBD +
                       head * EI_HEAD_DIM + row] = value;
            }
        }
    }
}

template <int OutputTiles>
__global__ void q4_mfma_projection_wide_kernel(
    const block_q4_0 *weights, const __half *input, float *output,
    int n_tokens, int n_rows, int n_cols) {
    const int row_start = static_cast<int>(blockIdx.x) * 16 * OutputTiles;
    const int token_start = static_cast<int>(blockIdx.y) * 16;
    float4_t accumulators[OutputTiles];
#pragma unroll
    for (int tile = 0; tile < OutputTiles; tile++) {
        accumulators[tile] = float4_t{0.0f, 0.0f, 0.0f, 0.0f};
    }
    for (int col_start = 0; col_start < n_cols; col_start += 16) {
        const half4_t input_fragment = load_input_mfma_fragment(
            input, n_tokens, n_cols, token_start, col_start);
#pragma unroll
        for (int tile = 0; tile < OutputTiles; tile++) {
            const half4_t weight_fragment = load_q4_mfma_fragment(
                weights, n_rows, n_cols, row_start + tile * 16, col_start);
            accumulators[tile] = mfma_m16n16k16(
                input_fragment, weight_fragment, accumulators[tile]);
        }
    }
    const int lane = threadIdx.x & 63;
    const int token = token_start + (lane >> 4) * 4;
#pragma unroll
    for (int tile = 0; tile < OutputTiles; tile++) {
        const int row = row_start + tile * 16 + (lane & 15);
#pragma unroll
        for (int item = 0; item < 4; item++) {
            if (row < n_rows && token + item < n_tokens) {
                output[static_cast<size_t>(token + item) * n_rows + row] =
                    accumulators[tile][item];
            }
        }
    }
}

__device__ __forceinline__ void qkv_mfma_target(
    int combined_row_start, const block_q4_0 *q_weights,
    const block_q4_0 *k_weights, const block_q4_0 *v_weights,
    float *q_output, float *k_output, float *v_output,
    const block_q4_0 *&weights, float *&output,
    int &row_start, int &output_rows) {
    if (combined_row_start < EI_N_EMBD) {
        weights = q_weights;
        output = q_output;
        row_start = combined_row_start;
        output_rows = EI_N_EMBD;
    } else if (combined_row_start < EI_N_EMBD + EI_HEAD_DIM) {
        weights = k_weights;
        output = k_output;
        row_start = combined_row_start - EI_N_EMBD;
        output_rows = EI_HEAD_DIM;
    } else {
        weights = v_weights;
        output = v_output;
        row_start = combined_row_start - EI_N_EMBD - EI_HEAD_DIM;
        output_rows = EI_HEAD_DIM;
    }
}

template <int OutputTiles>
__global__ void q4_mfma_qkv_projection_wide_kernel(
    const block_q4_0 *q_weights, const block_q4_0 *k_weights,
    const block_q4_0 *v_weights, const __half *input,
    float *q_output, float *k_output, float *v_output,
    int n_tokens) {
    const int combined_row_start =
        static_cast<int>(blockIdx.x) * 16 * OutputTiles;
    const int token_start = static_cast<int>(blockIdx.y) * 16;
    float4_t accumulators[OutputTiles];
#pragma unroll
    for (int tile = 0; tile < OutputTiles; tile++) {
        accumulators[tile] = float4_t{0.0f, 0.0f, 0.0f, 0.0f};
    }
    for (int col_start = 0; col_start < EI_N_EMBD; col_start += 16) {
        const half4_t input_fragment = load_input_mfma_fragment(
            input, n_tokens, EI_N_EMBD, token_start, col_start);
#pragma unroll
        for (int tile = 0; tile < OutputTiles; tile++) {
            const int combined = combined_row_start + tile * 16;
            const block_q4_0 *weights;
            float *output;
            int row_start;
            int output_rows;
            qkv_mfma_target(combined, q_weights, k_weights, v_weights,
                q_output, k_output, v_output, weights, output,
                row_start, output_rows);
            const half4_t weight_fragment = load_q4_mfma_fragment(
                weights, output_rows, EI_N_EMBD, row_start, col_start);
            accumulators[tile] = mfma_m16n16k16(
                input_fragment, weight_fragment, accumulators[tile]);
        }
    }
    const int lane = threadIdx.x & 63;
    const int token = token_start + (lane >> 4) * 4;
#pragma unroll
    for (int tile = 0; tile < OutputTiles; tile++) {
        const int combined = combined_row_start + tile * 16;
        const block_q4_0 *weights;
        float *output;
        int row_start;
        int output_rows;
        qkv_mfma_target(combined, q_weights, k_weights, v_weights,
            q_output, k_output, v_output, weights, output,
            row_start, output_rows);
        (void)weights;
        const int row = row_start + (lane & 15);
#pragma unroll
        for (int item = 0; item < 4; item++) {
            if (row < output_rows && token + item < n_tokens) {
                output[static_cast<size_t>(token + item) * output_rows + row] =
                    accumulators[tile][item];
            }
        }
    }
}

template <int OutputTiles>
__global__ void q4_mfma_up_gate_projection_wide_kernel(
    const block_q4_0 *up_weights, const block_q4_0 *gate_weights,
    const __half *input, float *combined_output, int n_tokens) {
    constexpr int combined_rows = 2 * EI_N_FF;
    const int combined_row_start =
        static_cast<int>(blockIdx.x) * 16 * OutputTiles;
    const int token_start = static_cast<int>(blockIdx.y) * 16;
    float4_t accumulators[OutputTiles];
#pragma unroll
    for (int tile = 0; tile < OutputTiles; tile++) {
        accumulators[tile] = float4_t{0.0f, 0.0f, 0.0f, 0.0f};
    }
    for (int col_start = 0; col_start < EI_N_EMBD; col_start += 16) {
        const half4_t input_fragment = load_input_mfma_fragment(
            input, n_tokens, EI_N_EMBD, token_start, col_start);
#pragma unroll
        for (int tile = 0; tile < OutputTiles; tile++) {
            const int combined = combined_row_start + tile * 16;
            const bool gate = combined >= EI_N_FF;
            const int row_start = gate ? combined - EI_N_FF : combined;
            const half4_t weight_fragment = load_q4_mfma_fragment(
                gate ? gate_weights : up_weights, EI_N_FF, EI_N_EMBD,
                row_start, col_start);
            accumulators[tile] = mfma_m16n16k16(
                input_fragment, weight_fragment, accumulators[tile]);
        }
    }
    const int lane = threadIdx.x & 63;
    const int token = token_start + (lane >> 4) * 4;
#pragma unroll
    for (int tile = 0; tile < OutputTiles; tile++) {
        const int combined = combined_row_start + tile * 16;
        const bool gate = combined >= EI_N_FF;
        const int row = (gate ? combined - EI_N_FF : combined) + (lane & 15);
        const int combined_row = (gate ? EI_N_FF : 0) + row;
#pragma unroll
        for (int item = 0; item < 4; item++) {
            if (combined_row < combined_rows && row < EI_N_FF &&
                token + item < n_tokens) {
                combined_output[
                    static_cast<size_t>(token + item) * combined_rows +
                    combined_row] = accumulators[tile][item];
            }
        }
    }
}

__global__ void q4_mfma_qkv_projection_kernel(
    const block_q4_0 *q_weights, const block_q4_0 *k_weights,
    const block_q4_0 *v_weights, const __half *input,
    float *q_output, float *k_output, float *v_output,
    int n_tokens) {
    constexpr int combined_rows = EI_N_EMBD + 2 * EI_HEAD_DIM;
    const int combined_row_start = static_cast<int>(blockIdx.x) * 16;
    const int token_start = static_cast<int>(blockIdx.y) * 16;
    const block_q4_0 *weights;
    float *output;
    int row_start;
    int output_rows;
    if (combined_row_start < EI_N_EMBD) {
        weights = q_weights;
        output = q_output;
        row_start = combined_row_start;
        output_rows = EI_N_EMBD;
    } else if (combined_row_start < EI_N_EMBD + EI_HEAD_DIM) {
        weights = k_weights;
        output = k_output;
        row_start = combined_row_start - EI_N_EMBD;
        output_rows = EI_HEAD_DIM;
    } else {
        weights = v_weights;
        output = v_output;
        row_start = combined_row_start - EI_N_EMBD - EI_HEAD_DIM;
        output_rows = EI_HEAD_DIM;
    }
    float4_t accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int col_start = 0; col_start < EI_N_EMBD; col_start += 16) {
        const half4_t input_fragment = load_input_mfma_fragment(
            input, n_tokens, EI_N_EMBD, token_start, col_start);
        const half4_t weight_fragment = load_q4_mfma_fragment(
            weights, output_rows, EI_N_EMBD, row_start, col_start);
        accumulator = mfma_m16n16k16(
            input_fragment, weight_fragment, accumulator);
    }
    const int lane = threadIdx.x & 63;
    const int row = row_start + (lane & 15);
    const int token = token_start + (lane >> 4) * 4;
#pragma unroll
    for (int item = 0; item < 4; item++) {
        if (combined_row_start < combined_rows && row < output_rows &&
            token + item < n_tokens) {
            output[static_cast<size_t>(token + item) * output_rows + row] =
                accumulator[item];
        }
    }
}

__global__ void q4_mfma_up_gate_projection_kernel(
    const block_q4_0 *up_weights, const block_q4_0 *gate_weights,
    const __half *input, float *combined_output, int n_tokens) {
    constexpr int combined_rows = 2 * EI_N_FF;
    const int combined_row_start = static_cast<int>(blockIdx.x) * 16;
    const int token_start = static_cast<int>(blockIdx.y) * 16;
    const bool gate = combined_row_start >= EI_N_FF;
    const block_q4_0 *weights = gate ? gate_weights : up_weights;
    const int row_start = gate
        ? combined_row_start - EI_N_FF : combined_row_start;
    float4_t accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int col_start = 0; col_start < EI_N_EMBD; col_start += 16) {
        const half4_t input_fragment = load_input_mfma_fragment(
            input, n_tokens, EI_N_EMBD, token_start, col_start);
        const half4_t weight_fragment = load_q4_mfma_fragment(
            weights, EI_N_FF, EI_N_EMBD, row_start, col_start);
        accumulator = mfma_m16n16k16(
            input_fragment, weight_fragment, accumulator);
    }
    const int lane = threadIdx.x & 63;
    const int row = row_start + (lane & 15);
    const int combined_row = (gate ? EI_N_FF : 0) + row;
    const int token = token_start + (lane >> 4) * 4;
#pragma unroll
    for (int item = 0; item < 4; item++) {
        if (combined_row < combined_rows && row < EI_N_FF &&
            token + item < n_tokens) {
            combined_output[
                static_cast<size_t>(token + item) * combined_rows +
                combined_row] = accumulator[item];
        }
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

__global__ void q4_projection_pair_kernel(
    const block_q4_0 *weights, const float *input, float *output,
    uint32_t n_rows, uint32_t n_cols, uint32_t n_tokens) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t row_pairs = (n_rows + 1) / 2;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * row_pairs;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / row_pairs);
    const uint32_t first_row = 2 * static_cast<uint32_t>(
        task - static_cast<size_t>(token) * row_pairs);
    const uint32_t second_row = min(first_row + 1, n_rows - 1);
    const uint32_t blocks_per_row = n_cols / kQK;
    float first_sum;
    float second_sum;
    q4_dot_pair(
        weights + static_cast<size_t>(first_row) * blocks_per_row,
        weights + static_cast<size_t>(second_row) * blocks_per_row,
        input + static_cast<size_t>(token) * n_cols,
        static_cast<int>(n_cols), lane, first_sum, second_sum);
    if (lane == 0) {
        output[static_cast<size_t>(token) * n_rows + first_row] = first_sum;
        if (second_row != first_row) {
            output[static_cast<size_t>(token) * n_rows + second_row] = second_sum;
        }
    }
}

__global__ void q4_projection_quad_kernel(
    const block_q4_0 *weights, const float *input, float *output,
    uint32_t n_rows, uint32_t n_cols, uint32_t n_tokens) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t row_quads = (n_rows + 3) / 4;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * row_quads;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / row_quads);
    const uint32_t row = 4 * static_cast<uint32_t>(
        task - static_cast<size_t>(token) * row_quads);
    const uint32_t blocks_per_row = n_cols / kQK;
    float sum0;
    float sum1;
    float sum2;
    float sum3;
    q4_dot_quad(
        weights + static_cast<size_t>(row) * blocks_per_row,
        weights + static_cast<size_t>(min(row + 1, n_rows - 1)) * blocks_per_row,
        weights + static_cast<size_t>(min(row + 2, n_rows - 1)) * blocks_per_row,
        weights + static_cast<size_t>(min(row + 3, n_rows - 1)) * blocks_per_row,
        input + static_cast<size_t>(token) * n_cols,
        static_cast<int>(n_cols), lane, sum0, sum1, sum2, sum3);
    if (lane == 0) {
        const size_t base = static_cast<size_t>(token) * n_rows + row;
        output[base] = sum0;
        if (row + 1 < n_rows) output[base + 1] = sum1;
        if (row + 2 < n_rows) output[base + 2] = sum2;
        if (row + 3 < n_rows) output[base + 3] = sum3;
    }
}

__global__ void v_only_projection_kernel(const block_q4_0 *weights,
                                          const float *input, float *output,
                                          uint32_t n_tokens) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * EI_HEAD_DIM;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / EI_HEAD_DIM);
    const uint32_t row = static_cast<uint32_t>(
        task - static_cast<size_t>(token) * EI_HEAD_DIM);
    constexpr int blocks_per_row = EI_N_EMBD / kQK;
    const float value = q4_dot(
        weights + static_cast<size_t>(row) * blocks_per_row,
        input + static_cast<size_t>(token) * EI_N_EMBD, EI_N_EMBD, lane);
    if (lane == 0) {
#pragma unroll
        for (int head = 0; head < EI_N_HEAD; head++) {
            output[static_cast<size_t>(token) * EI_N_EMBD +
                   head * EI_HEAD_DIM + row] = value;
        }
    }
}

__global__ void v_only_projection_pair_kernel(
    const block_q4_0 *weights, const float *input, float *output,
    uint32_t n_tokens) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr uint32_t row_pairs = EI_HEAD_DIM / 2;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * row_pairs;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / row_pairs);
    const uint32_t first_row = 2 * static_cast<uint32_t>(
        task - static_cast<size_t>(token) * row_pairs);
    const uint32_t second_row = first_row + 1;
    constexpr int blocks_per_row = EI_N_EMBD / kQK;
    float first_value;
    float second_value;
    q4_dot_pair(
        weights + static_cast<size_t>(first_row) * blocks_per_row,
        weights + static_cast<size_t>(second_row) * blocks_per_row,
        input + static_cast<size_t>(token) * EI_N_EMBD, EI_N_EMBD, lane,
        first_value, second_value);
    if (lane == 0) {
#pragma unroll
        for (int head = 0; head < EI_N_HEAD; head++) {
            const size_t base = static_cast<size_t>(token) * EI_N_EMBD +
                head * EI_HEAD_DIM;
            output[base + first_row] = first_value;
            output[base + second_row] = second_value;
        }
    }
}

__global__ void v_only_projection_quad_kernel(
    const block_q4_0 *weights, const float *input, float *output,
    uint32_t n_tokens) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr uint32_t row_quads = EI_HEAD_DIM / 4;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * row_quads;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / row_quads);
    const uint32_t row = 4 * static_cast<uint32_t>(
        task - static_cast<size_t>(token) * row_quads);
    constexpr int blocks_per_row = EI_N_EMBD / kQK;
    float values[4];
    q4_dot_quad(
        weights + static_cast<size_t>(row) * blocks_per_row,
        weights + static_cast<size_t>(row + 1) * blocks_per_row,
        weights + static_cast<size_t>(row + 2) * blocks_per_row,
        weights + static_cast<size_t>(row + 3) * blocks_per_row,
        input + static_cast<size_t>(token) * EI_N_EMBD, EI_N_EMBD, lane,
        values[0], values[1], values[2], values[3]);
    if (lane == 0) {
#pragma unroll
        for (int head = 0; head < EI_N_HEAD; head++) {
            const size_t base = static_cast<size_t>(token) * EI_N_EMBD +
                head * EI_HEAD_DIM + row;
#pragma unroll
            for (int item = 0; item < 4; item++) output[base + item] = values[item];
        }
    }
}

__global__ void repeat_value_f32_to_f16_kernel(const float *value,
                                                __half *output,
                                                uint32_t n_tokens) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    const size_t count = static_cast<size_t>(n_tokens) * EI_N_EMBD;
    if (index >= count) return;
    const uint32_t token = static_cast<uint32_t>(index / EI_N_EMBD);
    const uint32_t dim = static_cast<uint32_t>(
        index - static_cast<size_t>(token) * EI_N_EMBD);
    output[index] = __float2half(
        value[static_cast<size_t>(token) * EI_HEAD_DIM + dim % EI_HEAD_DIM]);
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

__global__ void q4_q8_projection_pair_kernel(
    const block_q4_0 *weights, const activation_q8 *input, float *output,
    uint32_t n_rows, uint32_t n_cols, uint32_t n_tokens) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t row_pairs = (n_rows + 1) / 2;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * row_pairs;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / row_pairs);
    const uint32_t first_row = 2 * static_cast<uint32_t>(
        task - static_cast<size_t>(token) * row_pairs);
    const uint32_t second_row = min(first_row + 1, n_rows - 1);
    const uint32_t blocks_per_row = n_cols / kQK;
    float first_sum;
    float second_sum;
    q4_q8_dot_pair(
        weights + static_cast<size_t>(first_row) * blocks_per_row,
        weights + static_cast<size_t>(second_row) * blocks_per_row,
        input + static_cast<size_t>(token) * blocks_per_row,
        static_cast<int>(n_cols), lane, first_sum, second_sum);
    if (lane == 0) {
        output[static_cast<size_t>(token) * n_rows + first_row] = first_sum;
        if (second_row != first_row) {
            output[static_cast<size_t>(token) * n_rows + second_row] = second_sum;
        }
    }
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

__global__ void qkv_projection_pair_kernel(
    const block_q4_0 *q_weights, const block_q4_0 *k_weights,
    const block_q4_0 *v_weights, const float *input,
    float *q_output, float *k_output, float *v_output, uint32_t n_tokens) {
    constexpr uint32_t combined_rows = EI_N_EMBD + 2 * EI_HEAD_DIM;
    constexpr uint32_t combined_pairs = combined_rows / 2;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * combined_pairs;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / combined_pairs);
    const uint32_t combined_row = 2 * static_cast<uint32_t>(
        task - static_cast<size_t>(token) * combined_pairs);

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
    float first_sum;
    float second_sum;
    q4_dot_pair(
        weights + static_cast<size_t>(row) * blocks_per_row,
        weights + static_cast<size_t>(row + 1) * blocks_per_row,
        input + static_cast<size_t>(token) * EI_N_EMBD, EI_N_EMBD, lane,
        first_sum, second_sum);
    if (lane == 0) {
        const size_t base = static_cast<size_t>(token) * rows + row;
        output[base] = first_sum;
        output[base + 1] = second_sum;
    }
}

__global__ void qkv_projection_quad_kernel(
    const block_q4_0 *q_weights, const block_q4_0 *k_weights,
    const block_q4_0 *v_weights, const float *input,
    float *q_output, float *k_output, float *v_output, uint32_t n_tokens) {
    constexpr uint32_t combined_rows = EI_N_EMBD + 2 * EI_HEAD_DIM;
    constexpr uint32_t combined_quads = combined_rows / 4;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * combined_quads;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / combined_quads);
    const uint32_t combined_row = 4 * static_cast<uint32_t>(
        task - static_cast<size_t>(token) * combined_quads);

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
    float sums[4];
    q4_dot_quad(
        weights + static_cast<size_t>(row) * blocks_per_row,
        weights + static_cast<size_t>(row + 1) * blocks_per_row,
        weights + static_cast<size_t>(row + 2) * blocks_per_row,
        weights + static_cast<size_t>(row + 3) * blocks_per_row,
        input + static_cast<size_t>(token) * EI_N_EMBD, EI_N_EMBD, lane,
        sums[0], sums[1], sums[2], sums[3]);
    if (lane == 0) {
        const size_t base = static_cast<size_t>(token) * rows + row;
#pragma unroll
        for (int item = 0; item < 4; item++) output[base + item] = sums[item];
    }
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

__global__ void qkv_q8_projection_pair_kernel(
    const block_q4_0 *q_weights, const block_q4_0 *k_weights,
    const block_q4_0 *v_weights, const activation_q8 *input,
    float *q_output, float *k_output, float *v_output, uint32_t n_tokens) {
    constexpr uint32_t combined_rows = EI_N_EMBD + 2 * EI_HEAD_DIM;
    constexpr uint32_t row_pairs = combined_rows / 2;
    constexpr uint32_t blocks_per_row = EI_N_EMBD / kQK;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * row_pairs;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / row_pairs);
    const uint32_t combined_row = 2 * static_cast<uint32_t>(
        task - static_cast<size_t>(token) * row_pairs);
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
    float first_sum;
    float second_sum;
    q4_q8_dot_pair(
        weights + static_cast<size_t>(row) * blocks_per_row,
        weights + static_cast<size_t>(row + 1) * blocks_per_row,
        input + static_cast<size_t>(token) * blocks_per_row,
        EI_N_EMBD, lane, first_sum, second_sum);
    if (lane == 0) {
        const size_t base = static_cast<size_t>(token) * rows + row;
        output[base] = first_sum;
        output[base + 1] = second_sum;
    }
}

__device__ __forceinline__ float gelu_tanh(float value) {
    constexpr float a = 0.044715f;
    constexpr float s = 0.7978845608028654f;
    return 0.5f * value * (1.0f + tanhf(s * value *
                                        (1.0f + a * value * value)));
}

__global__ void q4_mfma_up_gate_gelu_f16_kernel(
    const block_q4_0 *up_weights, const block_q4_0 *gate_weights,
    const __half *input, __half *output, int n_tokens) {
    const int row_start = static_cast<int>(blockIdx.x) * 16;
    const int token_start = static_cast<int>(blockIdx.y) * 16;
    float4_t up_accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
    float4_t gate_accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int col_start = 0; col_start < EI_N_EMBD; col_start += 16) {
        const half4_t input_fragment = load_input_mfma_fragment(
            input, n_tokens, EI_N_EMBD, token_start, col_start);
        const half4_t up_fragment = load_q4_mfma_fragment(
            up_weights, EI_N_FF, EI_N_EMBD, row_start, col_start);
        const half4_t gate_fragment = load_q4_mfma_fragment(
            gate_weights, EI_N_FF, EI_N_EMBD, row_start, col_start);
        up_accumulator = mfma_m16n16k16(
            input_fragment, up_fragment, up_accumulator);
        gate_accumulator = mfma_m16n16k16(
            input_fragment, gate_fragment, gate_accumulator);
    }
    const int lane = threadIdx.x & 63;
    const int row = row_start + (lane & 15);
    const int token = token_start + (lane >> 4) * 4;
#pragma unroll
    for (int item = 0; item < 4; item++) {
        if (row < EI_N_FF && token + item < n_tokens) {
            output[static_cast<size_t>(token + item) * EI_N_FF + row] =
                __float2half(gelu_tanh(gate_accumulator[item]) *
                             up_accumulator[item]);
        }
    }
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

__global__ void up_gate_gelu_pair_kernel(
    const block_q4_0 *up_weights, const block_q4_0 *gate_weights,
    const float *input, float *output, uint32_t n_tokens) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * EI_N_FF;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / EI_N_FF);
    const uint32_t row = static_cast<uint32_t>(task -
        static_cast<size_t>(token) * EI_N_FF);
    constexpr int blocks_per_row = EI_N_EMBD / kQK;
    float up;
    float gate;
    q4_dot_pair(
        up_weights + static_cast<size_t>(row) * blocks_per_row,
        gate_weights + static_cast<size_t>(row) * blocks_per_row,
        input + static_cast<size_t>(token) * EI_N_EMBD, EI_N_EMBD, lane,
        up, gate);
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

__global__ void up_gate_q8_gelu_pair_kernel(
    const block_q4_0 *up_weights, const block_q4_0 *gate_weights,
    const activation_q8 *input, float *output, uint32_t n_tokens) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * EI_N_FF;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / EI_N_FF);
    const uint32_t row = static_cast<uint32_t>(
        task - static_cast<size_t>(token) * EI_N_FF);
    constexpr int blocks_per_row = EI_N_EMBD / kQK;
    float up;
    float gate;
    q4_q8_dot_pair(
        up_weights + static_cast<size_t>(row) * blocks_per_row,
        gate_weights + static_cast<size_t>(row) * blocks_per_row,
        input + static_cast<size_t>(token) * blocks_per_row,
        EI_N_EMBD, lane, up, gate);
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
    sum = __shfl(sum, 0, 32);
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
    sum = __shfl(sum, 0, 32);
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
    projected_ss = __shfl(projected_ss, 0, 32);
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
    residual_ss = __shfl(residual_ss, 0, 32);
    const float residual_inv = rsqrtf(
        residual_ss / static_cast<float>(n_cols) + eps);
    for (uint32_t col = lane; col < n_cols; col += 32) {
        next_output[base + col] = __float2half(
            residual[base + col] * next_weight[col] * residual_inv);
    }
}

__global__ void rms_residual_next_f16_cached_kernel(
    const float *input, const float *post_weight, float *residual,
    const float *next_weight, __half *next_output, uint32_t n_rows,
    float eps) {
    constexpr int items_per_lane = EI_N_EMBD / 32;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= n_rows) return;
    const size_t base = static_cast<size_t>(row) * EI_N_EMBD;
    float projected_ss = 0.0f;
#pragma unroll
    for (int item = 0; item < items_per_lane; item++) {
        const int col = lane + item * 32;
        const float value = input[base + col];
        projected_ss = fmaf(value, value, projected_ss);
    }
    projected_ss = warp_sum(projected_ss);
    projected_ss = __shfl(projected_ss, 0, 32);
    const float projected_inv = rsqrtf(
        projected_ss / static_cast<float>(EI_N_EMBD) + eps);

    float residual_values[items_per_lane];
    float residual_ss = 0.0f;
#pragma unroll
    for (int item = 0; item < items_per_lane; item++) {
        const int col = lane + item * 32;
        const float value = residual[base + col] +
            input[base + col] * post_weight[col] * projected_inv;
        residual_values[item] = value;
        residual[base + col] = value;
        residual_ss = fmaf(value, value, residual_ss);
    }
    residual_ss = warp_sum(residual_ss);
    residual_ss = __shfl(residual_ss, 0, 32);
    const float residual_inv = rsqrtf(
        residual_ss / static_cast<float>(EI_N_EMBD) + eps);
#pragma unroll
    for (int item = 0; item < items_per_lane; item++) {
        const int col = lane + item * 32;
        next_output[base + col] = __float2half(
            residual_values[item] * next_weight[col] * residual_inv);
    }
}

__global__ void rms_residual_next_f32_cached_kernel(
    const float *input, const float *post_weight, float *residual,
    const float *next_weight, float *next_output, uint32_t n_rows,
    float eps) {
    constexpr int items_per_lane = EI_N_EMBD / 32;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= n_rows) return;
    const size_t base = static_cast<size_t>(row) * EI_N_EMBD;
    float projected_ss = 0.0f;
#pragma unroll
    for (int item = 0; item < items_per_lane; item++) {
        const int col = lane + item * 32;
        const float value = input[base + col];
        projected_ss = fmaf(value, value, projected_ss);
    }
    projected_ss = warp_sum(projected_ss);
    projected_ss = __shfl(projected_ss, 0, 32);
    const float projected_inv = rsqrtf(
        projected_ss / static_cast<float>(EI_N_EMBD) + eps);

    float residual_values[items_per_lane];
    float residual_ss = 0.0f;
#pragma unroll
    for (int item = 0; item < items_per_lane; item++) {
        const int col = lane + item * 32;
        const float value = residual[base + col] +
            input[base + col] * post_weight[col] * projected_inv;
        residual_values[item] = value;
        residual[base + col] = value;
        residual_ss = fmaf(value, value, residual_ss);
    }
    residual_ss = warp_sum(residual_ss);
    residual_ss = __shfl(residual_ss, 0, 32);
    const float residual_inv = rsqrtf(
        residual_ss / static_cast<float>(EI_N_EMBD) + eps);
#pragma unroll
    for (int item = 0; item < items_per_lane; item++) {
        const int col = lane + item * 32;
        next_output[base + col] =
            residual_values[item] * next_weight[col] * residual_inv;
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
    sum = __shfl(sum, 0, 32);
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
    sum = __shfl(sum, 0, 32);
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

__global__ void qkv_separate_norm_rope_f16_kernel(
    const float *q_input, const float *k_input, const float *v_input,
    __half *q, __half *k, __half *v,
    const float *q_weight, const float *k_weight,
    const uint32_t *positions, const float2 *rope_table,
    uint32_t n_tokens, float eps) {
    constexpr uint32_t combined_tasks = EI_N_HEAD + EI_N_HEAD_KV + 1;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * combined_tasks;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / combined_tasks);
    const uint32_t combined_task = static_cast<uint32_t>(
        task - static_cast<size_t>(token) * combined_tasks);

    if (combined_task == combined_tasks - 1) {
        const size_t base = static_cast<size_t>(token) * EI_HEAD_DIM;
        for (uint32_t dim = lane; dim < EI_HEAD_DIM; dim += 32) {
            v[base + dim] = __float2half(v_input[base + dim]);
        }
        return;
    }

    const bool key_head = combined_task == EI_N_HEAD;
    const uint32_t head = key_head ? 0 : combined_task;
    const uint32_t stride = key_head ? EI_HEAD_DIM : EI_N_EMBD;
    const size_t base = static_cast<size_t>(token) * stride +
                        static_cast<size_t>(head) * EI_HEAD_DIM;
    const float *input = key_head ? k_input : q_input;
    const float *weight = key_head ? k_weight : q_weight;
    __half *output = key_head ? k : q;
    float sum = 0.0f;
    for (uint32_t dim = lane; dim < EI_HEAD_DIM; dim += 32) {
        const float value = input[base + dim];
        sum = fmaf(value, value, sum);
    }
    sum = warp_sum(sum);
    sum = __shfl(sum, 0, 32);
    const float scale = key_head ? 1.0f : 0.0625f;
    const float inv = rsqrtf(sum / static_cast<float>(EI_HEAD_DIM) + eps) * scale;
    const size_t rope_base = static_cast<size_t>(positions[token]) *
                             (EI_HEAD_DIM / 2);
    for (uint32_t dim = lane; dim < EI_HEAD_DIM / 2; dim += 32) {
        const float2 cs = rope_table[rope_base + dim];
        const float x0 = input[base + dim] * weight[dim] * inv;
        const float x1 = input[base + dim + EI_HEAD_DIM / 2] *
                         weight[dim + EI_HEAD_DIM / 2] * inv;
        output[base + dim] = __float2half(fmaf(-x1, cs.y, x0 * cs.x));
        output[base + dim + EI_HEAD_DIM / 2] =
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
        score = __shfl(score, 0, 32);
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
        score = __shfl(score, 0, 32);
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
                score = __shfl(score, 0, 32);
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

// CDNA MFMA flash attention adapted from QuixiCore's BQ=BK=16 GQA kernel.
// One wave64 reuses each K/V tile across 16 queries and keeps softmax online.
template <typename Output>
__global__ void mfma_attention_f16_kernel(
    const __half *q_half, const __half *k_half, const __half *v_half,
    Output *output, const uint32_t *query_starts,
    const uint32_t *sequence_starts, const uint32_t *sequence_stops,
    uint32_t tile_count, uint32_t window) {
    constexpr uint32_t query_tile = 16;
    constexpr uint32_t key_tile = 16;
    constexpr int dim_tiles = EI_HEAD_DIM / 16;
    __shared__ float scores[query_tile][key_tile];
    __shared__ __half probabilities[query_tile][key_tile];
    __shared__ float running_max[query_tile];
    __shared__ float running_sum[query_tile];
    __shared__ float correction[query_tile];

    const uint32_t tile = blockIdx.x;
    const uint32_t head = blockIdx.y;
    if (tile >= tile_count || head >= EI_N_HEAD) return;
    const int lane = threadIdx.x & 63;
    const int lane_group = lane >> 4;
    const int lane_item = lane & 15;
    const uint32_t query_start = query_starts[tile];
    const uint32_t sequence_start = sequence_starts[tile];
    const uint32_t sequence_stop = sequence_stops[tile];
    const uint32_t query_stop = min(sequence_stop, query_start + query_tile);
    const __fp16 *q = reinterpret_cast<const __fp16 *>(q_half);
    const __fp16 *k = reinterpret_cast<const __fp16 *>(k_half);
    const __fp16 *v = reinterpret_cast<const __fp16 *>(v_half);

    half4_t query_fragments[dim_tiles];
#pragma unroll
    for (int dim_tile = 0; dim_tile < dim_tiles; dim_tile++) {
        const uint32_t query_index = query_start + lane_item;
        const uint32_t dim = dim_tile * 16 + lane_group * 4;
        if (query_index < query_stop) {
            query_fragments[dim_tile] = *reinterpret_cast<const half4_t *>(
                q + static_cast<size_t>(query_index) * EI_N_EMBD +
                head * EI_HEAD_DIM + dim);
        } else {
            query_fragments[dim_tile] = half4_t{0, 0, 0, 0};
        }
    }
    float4_t output_accumulators[dim_tiles];
#pragma unroll
    for (int dim_tile = 0; dim_tile < dim_tiles; dim_tile++) {
        output_accumulators[dim_tile] = float4_t{0, 0, 0, 0};
    }
    if (lane < static_cast<int>(query_tile)) {
        running_max[lane] = -__int_as_float(0x7f800000);
        running_sum[lane] = 0.0f;
    }
    __syncthreads();

    uint32_t union_first = sequence_start;
    uint32_t union_last = sequence_stop;
    if (window != 0) {
        const uint32_t half_window = window / 2;
        union_first = query_start > sequence_start + half_window
            ? query_start - half_window : sequence_start;
        union_last = min(sequence_stop, query_stop + half_window);
    }

    for (uint32_t key_start = union_first; key_start < union_last;
         key_start += key_tile) {
        float4_t score = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int dim_tile = 0; dim_tile < dim_tiles; dim_tile++) {
            const uint32_t key_index = key_start + lane_item;
            const uint32_t dim = dim_tile * 16 + lane_group * 4;
            half4_t key_fragment = {0, 0, 0, 0};
            if (key_index < union_last) {
                key_fragment = *reinterpret_cast<const half4_t *>(
                    k + static_cast<size_t>(key_index) * EI_HEAD_DIM + dim);
            }
            score = mfma_m16n16k16(
                query_fragments[dim_tile], key_fragment, score);
        }
#pragma unroll
        for (int item = 0; item < 4; item++) {
            scores[4 * lane_group + item][lane_item] = score[item];
        }
        __syncthreads();

        if (lane < static_cast<int>(query_tile)) {
            const uint32_t query_index = query_start + lane;
            if (query_index < query_stop) {
                uint32_t first = sequence_start;
                uint32_t last = sequence_stop;
                if (window != 0) {
                    const uint32_t half_window = window / 2;
                    first = query_index > sequence_start + half_window
                        ? query_index - half_window : sequence_start;
                    last = min(sequence_stop, query_index + half_window + 1);
                }
                const float previous_max = running_max[lane];
                float next_max = previous_max;
#pragma unroll
                for (uint32_t item = 0; item < key_tile; item++) {
                    const uint32_t key_index = key_start + item;
                    if (key_index < union_last && key_index >= first &&
                        key_index < last) {
                        next_max = fmaxf(next_max, scores[lane][item]);
                    }
                }
                const float alpha = __expf(previous_max - next_max);
                float next_sum = running_sum[lane] * alpha;
#pragma unroll
                for (uint32_t item = 0; item < key_tile; item++) {
                    const uint32_t key_index = key_start + item;
                    const bool active = key_index < union_last &&
                        key_index >= first && key_index < last;
                    const float probability = active
                        ? __expf(scores[lane][item] - next_max) : 0.0f;
                    probabilities[lane][item] = __float2half(probability);
                    next_sum += probability;
                }
                running_max[lane] = next_max;
                running_sum[lane] = next_sum;
                correction[lane] = alpha;
            } else {
#pragma unroll
                for (uint32_t item = 0; item < key_tile; item++) {
                    probabilities[lane][item] = __float2half(0.0f);
                }
                correction[lane] = 1.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int dim_tile = 0; dim_tile < dim_tiles; dim_tile++) {
#pragma unroll
            for (int item = 0; item < 4; item++) {
                output_accumulators[dim_tile][item] *=
                    correction[4 * lane_group + item];
            }
        }
        const half4_t probability_fragment =
            *reinterpret_cast<const half4_t *>(
                &probabilities[lane_item][lane_group * 4]);
#pragma unroll
        for (int dim_tile = 0; dim_tile < dim_tiles; dim_tile++) {
            const uint32_t dim = dim_tile * 16 + lane_item;
            half4_t value_fragment = {0, 0, 0, 0};
#pragma unroll
            for (int item = 0; item < 4; item++) {
                const uint32_t key_index = key_start + lane_group * 4 + item;
                if (key_index < union_last) {
                    value_fragment[item] = v[
                        static_cast<size_t>(key_index) * EI_HEAD_DIM + dim];
                }
            }
            output_accumulators[dim_tile] = mfma_m16n16k16(
                probability_fragment, value_fragment,
                output_accumulators[dim_tile]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int dim_tile = 0; dim_tile < dim_tiles; dim_tile++) {
        const uint32_t dim = dim_tile * 16 + lane_item;
#pragma unroll
        for (int item = 0; item < 4; item++) {
            const uint32_t query_index = query_start + 4 * lane_group + item;
            if (query_index < query_stop) {
                const size_t output_index =
                    static_cast<size_t>(query_index) * EI_N_EMBD +
                    head * EI_HEAD_DIM + dim;
                store_attention_output(output, output_index,
                    output_accumulators[dim_tile][item] /
                    running_sum[4 * lane_group + item]);
            }
        }
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

template <typename Score>
__device__ __forceinline__ float attention_score(Score value);

template <>
__device__ __forceinline__ float attention_score(float value) {
    return value;
}

template <>
__device__ __forceinline__ float attention_score(__half value) {
    return __half2float(value);
}

template <typename Score>
__global__ void attention_softmax_f16_kernel(
    const Score *scores, __half *probabilities, uint32_t query_start,
    uint32_t key_start, uint32_t query_count, uint32_t key_count,
    uint32_t sequence_tokens, uint32_t window) {
    __shared__ float partials[kWarpsPerBlock];
    const uint32_t head = blockIdx.x / query_count;
    const uint32_t local_query = blockIdx.x - head * query_count;
    const size_t matrix_stride = static_cast<size_t>(query_count) * key_count;
    scores += static_cast<size_t>(head) * matrix_stride;
    probabilities += static_cast<size_t>(head) * matrix_stride;
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
        local_max = fmaxf(local_max, attention_score(
            scores[static_cast<size_t>(local_query) * key_count + key]));
    }
    const float maximum = block_max(local_max, partials);
    float local_sum = 0.0f;
    for (uint32_t key = local_first + threadIdx.x; key < local_last;
         key += blockDim.x) {
        local_sum += __expf(attention_score(scores[
            static_cast<size_t>(local_query) * key_count + key]) - maximum);
    }
    const float denominator = block_sum(local_sum, partials);
    for (uint32_t key = threadIdx.x; key < key_count; key += blockDim.x) {
        const float probability = key >= local_first && key < local_last
            ? __expf(attention_score(scores[
                static_cast<size_t>(local_query) * key_count + key]) - maximum) /
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
        sum = __shfl(sum, 0, 32);
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
    sum = __shfl(sum, 0, 32);
    const float inv_l2 = sum == 0.0f ? 1.0f : rsqrtf(sum);
#pragma unroll
    for (int item = 0; item < EI_N_EMBD / 32; item++) {
        const int dim = lane + item * 32;
        output[static_cast<size_t>(sequence) * EI_N_EMBD + dim] =
            pooled[item] * inv_l2;
    }
}

__global__ void final_singleton_pool_kernel(
    const float *projected, const float *post_weight, const float *residual,
    const float *output_weight, float *output, uint32_t batch_size, float eps) {
    constexpr int items_per_lane = EI_N_EMBD / 32;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t sequence = blockIdx.x * kWarpsPerBlock + warp;
    if (sequence >= batch_size) return;
    const size_t base = static_cast<size_t>(sequence) * EI_N_EMBD;
    float values[items_per_lane];
    float projected_ss = 0.0f;
#pragma unroll
    for (int item = 0; item < items_per_lane; item++) {
        const int dim = lane + item * 32;
        const float value = projected[base + dim];
        values[item] = value;
        projected_ss = fmaf(value, value, projected_ss);
    }
    projected_ss = warp_sum(projected_ss);
    const float projected_inv = rsqrtf(__shfl(projected_ss, 0, 32) /
        static_cast<float>(EI_N_EMBD) + eps);

    float residual_ss = 0.0f;
#pragma unroll
    for (int item = 0; item < items_per_lane; item++) {
        const int dim = lane + item * 32;
        const float value = residual[base + dim] +
            values[item] * post_weight[dim] * projected_inv;
        values[item] = value;
        residual_ss = fmaf(value, value, residual_ss);
    }
    residual_ss = warp_sum(residual_ss);
    const float residual_inv = rsqrtf(__shfl(residual_ss, 0, 32) /
        static_cast<float>(EI_N_EMBD) + eps);

    float l2_sum = 0.0f;
#pragma unroll
    for (int item = 0; item < items_per_lane; item++) {
        const int dim = lane + item * 32;
        const float value = values[item] * output_weight[dim] * residual_inv;
        values[item] = value;
        l2_sum = fmaf(value, value, l2_sum);
    }
    l2_sum = warp_sum(l2_sum);
    const float norm = __shfl(l2_sum, 0, 32);
    const float l2_inv = norm == 0.0f ? 1.0f : rsqrtf(norm);
#pragma unroll
    for (int item = 0; item < items_per_lane; item++) {
        const int dim = lane + item * 32;
        output[base + dim] = values[item] * l2_inv;
    }
}

static size_t grow_capacity(size_t current, size_t required) {
    size_t capacity = current ? current : 16;
    while (capacity < required) capacity *= 2;
    return capacity;
}

static void release_token_workspace(rocm_engine *engine) {
    clear_graphs(engine);
    hip_release(engine->ids);
    hip_release(engine->positions);
    hip_release(engine->sequence_ids);
    hip_release(engine->x);
    hip_release(engine->norm);
    hip_release(engine->q);
    hip_release(engine->k);
    hip_release(engine->v);
    hip_release(engine->qkv_combined);
    hip_release(engine->q_half);
    hip_release(engine->k_half);
    hip_release(engine->v_half);
    hip_release(engine->ctx);
    hip_release(engine->up);
    hip_release(engine->gate);
    hip_release(engine->up_gate_combined);
    hip_release(engine->tmp);
    hip_release(engine->half_input);
    hip_release(engine->q8_input);
    hip_release(engine->flash_query_start);
    hip_release(engine->flash_sequence_start);
    hip_release(engine->flash_sequence_stop);
    hip_release(engine->mfma_query_start);
    hip_release(engine->mfma_sequence_start);
    hip_release(engine->mfma_sequence_stop);
    hip_host_release(engine->host_ids);
    engine->token_capacity = 0;
}

static void release_batch_workspace(rocm_engine *engine) {
    clear_graphs(engine);
    hip_release(engine->offsets);
    hip_release(engine->result);
    hip_host_release(engine->host_offsets);
    hip_host_release(engine->host_result);
    engine->batch_capacity = 0;
}

static bool ensure_capacity(rocm_engine *engine, size_t token_count,
                            size_t batch_size, char *err, size_t err_len) {
    if (token_count > engine->token_capacity) {
        const size_t capacity = grow_capacity(engine->token_capacity, token_count);
        release_token_workspace(engine);
        if (!hip_allocate(&engine->ids, capacity, "allocate HIP token IDs", err, err_len) ||
            !hip_allocate(&engine->positions, capacity, "allocate HIP positions", err, err_len) ||
            !hip_allocate(&engine->sequence_ids, capacity, "allocate HIP sequence IDs", err, err_len) ||
            !hip_allocate(&engine->x, capacity * EI_N_EMBD, "allocate HIP residual", err, err_len) ||
            !hip_allocate(&engine->norm, capacity * EI_N_EMBD, "allocate HIP norm", err, err_len) ||
            !hip_allocate(&engine->q, capacity * EI_N_EMBD, "allocate HIP Q", err, err_len) ||
            !hip_allocate(&engine->k, capacity * EI_HEAD_DIM, "allocate HIP K", err, err_len) ||
            !hip_allocate(&engine->v, capacity * EI_HEAD_DIM, "allocate HIP V", err, err_len) ||
            !hip_allocate(&engine->qkv_combined,
                           capacity * (EI_N_EMBD + 2 * EI_HEAD_DIM),
                           "allocate HIP combined QKV", err, err_len) ||
            !hip_allocate(&engine->q_half, capacity * EI_N_EMBD,
                           "allocate HIP FP16 Q", err, err_len) ||
            !hip_allocate(&engine->k_half, capacity * EI_HEAD_DIM,
                           "allocate HIP FP16 K", err, err_len) ||
            !hip_allocate(&engine->v_half, capacity * EI_HEAD_DIM,
                           "allocate HIP FP16 V", err, err_len) ||
            !hip_allocate(&engine->ctx, capacity * EI_N_EMBD, "allocate HIP attention", err, err_len) ||
            !hip_allocate(&engine->up, capacity * EI_N_FF, "allocate HIP FFN up", err, err_len) ||
            !hip_allocate(&engine->gate, capacity * EI_N_FF, "allocate HIP FFN", err, err_len) ||
            !hip_allocate(&engine->up_gate_combined, capacity * (2 * EI_N_FF),
                           "allocate HIP combined up/gate", err, err_len) ||
            !hip_allocate(&engine->tmp, capacity * EI_N_EMBD, "allocate HIP projection", err, err_len) ||
            !hip_allocate(&engine->half_input, capacity * EI_N_FF,
                           "allocate HIP tensor-core input", err, err_len) ||
            !hip_allocate(&engine->q8_input, capacity * (EI_N_FF / kQK),
                           "allocate HIP q8 activations", err, err_len) ||
            !hip_allocate(&engine->flash_query_start, capacity,
                           "allocate HIP attention query tiles", err, err_len) ||
            !hip_allocate(&engine->flash_sequence_start, capacity,
                           "allocate HIP attention sequence starts", err, err_len) ||
            !hip_allocate(&engine->flash_sequence_stop, capacity,
                           "allocate HIP attention sequence stops", err, err_len) ||
            !hip_allocate(&engine->mfma_query_start, capacity,
                           "allocate HIP MFMA attention query tiles", err, err_len) ||
            !hip_allocate(&engine->mfma_sequence_start, capacity,
                           "allocate HIP MFMA attention sequence starts", err, err_len) ||
            !hip_allocate(&engine->mfma_sequence_stop, capacity,
                           "allocate HIP MFMA attention sequence stops", err, err_len) ||
            (engine->pinned_io_staging &&
             !hip_host_allocate(&engine->host_ids, capacity,
                                "allocate pinned HIP token IDs", err, err_len))) {
            release_token_workspace(engine);
            return false;
        }
        engine->token_capacity = capacity;
    }
    if (batch_size > engine->batch_capacity) {
        const size_t capacity = grow_capacity(engine->batch_capacity, batch_size);
        release_batch_workspace(engine);
        if (!hip_allocate(&engine->offsets, capacity + 1, "allocate HIP offsets", err, err_len) ||
            !hip_allocate(&engine->result, capacity * EI_N_EMBD, "allocate HIP result", err, err_len) ||
            (engine->pinned_io_staging &&
             (!hip_host_allocate(&engine->host_offsets, capacity + 1,
                                 "allocate pinned HIP offsets", err, err_len) ||
              !hip_host_allocate(&engine->host_result, capacity * EI_N_EMBD,
                                 "allocate pinned HIP result", err, err_len)))) {
            release_batch_workspace(engine);
            return false;
        }
        engine->batch_capacity = capacity;
    }
    return true;
}

static const uint8_t *model_host_base(const rocm_engine *engine) {
    return engine->model->gguf.map + engine->model->gguf.data_off;
}

static const block_q4_0 *q4_weights(const rocm_engine *engine,
                                    const ei_tensor *tensor) {
    return reinterpret_cast<const block_q4_0 *>(engine->model_data + tensor->offset);
}

static const float *float_weights(const rocm_engine *engine,
                                  const float *host_pointer) {
    const ptrdiff_t offset = reinterpret_cast<const uint8_t *>(host_pointer) -
                             model_host_base(engine);
    return reinterpret_cast<const float *>(engine->model_data + offset);
}

static size_t tensor_elements(const ei_tensor *tensor) {
    return static_cast<size_t>(tensor->ne[0]) * tensor->ne[1];
}

static bool initialize_dequantized_weights(rocm_engine *engine,
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
    if (!hip_allocate(&engine->dequantized_weights, total,
                       "allocate HIP tensor-core weights", err, err_len)) {
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
    return hip_check(hipGetLastError(), "dequantize HIP tensor-core weights",
                      err, err_len);
}

static void convert_to_half(rocm_engine *engine, const float *input,
                            size_t count) {
    f32_to_f16_kernel<<<static_cast<unsigned>((count + kThreads - 1) / kThreads),
                        kThreads, 0, engine->stream>>>(
        input, engine->half_input, count);
}

static bool tensor_core_projection(rocm_engine *engine, const __half *weights,
                                   const __half *input, float *output,
                                   int rows, int cols, int tokens,
                                   char *err, size_t err_len) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return hipblas_check(hipblasGemmEx(
        engine->blas, HIPBLAS_OP_T, HIPBLAS_OP_N,
        rows, tokens, cols, &alpha,
        weights, HIP_R_16F, cols,
        input, HIP_R_16F, cols,
        &beta, output, HIP_R_32F, rows,
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
        "execute HIP tensor-core projection", err, err_len);
}

static int native_q4_output_tiles(const rocm_engine *engine,
                                  int tokens, int rows) {
    if (!engine->native_q4_wide) return 1;
    const int token_tiles = (tokens + 15) / 16;
    const int target_waves = 2 * engine->compute_units;
    if (rows % 64 == 0 && (rows / 64) * token_tiles >= target_waves) return 4;
    if (rows % 32 == 0 && (rows / 32) * token_tiles >= target_waves) return 2;
    return 1;
}

static void native_q4_projection(rocm_engine *engine,
                                 const block_q4_0 *weights,
                                 const __half *input, float *output,
                                 int rows, int cols, int tokens) {
    const int output_tiles = native_q4_output_tiles(
        engine, tokens, rows);
    const dim3 grid(static_cast<unsigned>(
                        (rows + 16 * output_tiles - 1) /
                        (16 * output_tiles)),
                    static_cast<unsigned>((tokens + 15) / 16));
    if (output_tiles == 4) {
        q4_mfma_projection_wide_kernel<4><<<grid, 64, 0, engine->stream>>>(
            weights, input, output, tokens, rows, cols);
    } else if (output_tiles == 2) {
        q4_mfma_projection_wide_kernel<2><<<grid, 64, 0, engine->stream>>>(
            weights, input, output, tokens, rows, cols);
    } else {
        q4_mma_projection_kernel<<<grid, 64, 0, engine->stream>>>(
            weights, input, output, tokens, rows, cols);
    }
}

static void native_q4_v_only_projection(
    rocm_engine *engine, const block_q4_0 *weights, const __half *input,
    __half *output, int tokens) {
    const dim3 grid(EI_HEAD_DIM / 16,
                    static_cast<unsigned>((tokens + 15) / 16));
    q4_mfma_v_only_projection_kernel<<<grid, 64, 0, engine->stream>>>(
        weights, input, output, tokens);
}

static void native_q4_qkv_projection(
    rocm_engine *engine, const block_q4_0 *q_weights,
    const block_q4_0 *k_weights, const block_q4_0 *v_weights,
    const __half *input, float *q_output, float *k_output, float *v_output,
    int tokens) {
    constexpr int rows = EI_N_EMBD + 2 * EI_HEAD_DIM;
    const int output_tiles = native_q4_output_tiles(engine, tokens, rows);
    const dim3 grid(rows / (16 * output_tiles),
                    static_cast<unsigned>((tokens + 15) / 16));
    if (output_tiles == 4) {
        q4_mfma_qkv_projection_wide_kernel<4><<<grid, 64, 0,
            engine->stream>>>(q_weights, k_weights, v_weights, input,
                q_output, k_output, v_output, tokens);
    } else if (output_tiles == 2) {
        q4_mfma_qkv_projection_wide_kernel<2><<<grid, 64, 0,
            engine->stream>>>(q_weights, k_weights, v_weights, input,
                q_output, k_output, v_output, tokens);
    } else {
        q4_mfma_qkv_projection_kernel<<<grid, 64, 0, engine->stream>>>(
            q_weights, k_weights, v_weights, input, q_output, k_output,
            v_output, tokens);
    }
}

static void native_q4_up_gate_projection(
    rocm_engine *engine, const block_q4_0 *up_weights,
    const block_q4_0 *gate_weights, const __half *input,
    float *combined_output, int tokens) {
    constexpr int rows = 2 * EI_N_FF;
    const int output_tiles = native_q4_output_tiles(engine, tokens, rows);
    const dim3 grid(rows / (16 * output_tiles),
                    static_cast<unsigned>((tokens + 15) / 16));
    if (output_tiles == 4) {
        q4_mfma_up_gate_projection_wide_kernel<4><<<grid, 64, 0,
            engine->stream>>>(up_weights, gate_weights, input,
                combined_output, tokens);
    } else if (output_tiles == 2) {
        q4_mfma_up_gate_projection_wide_kernel<2><<<grid, 64, 0,
            engine->stream>>>(up_weights, gate_weights, input,
                combined_output, tokens);
    } else {
        q4_mfma_up_gate_projection_kernel<<<grid, 64, 0, engine->stream>>>(
            up_weights, gate_weights, input, combined_output, tokens);
    }
}

static void native_q4_up_gate_activation(
    rocm_engine *engine, const block_q4_0 *up_weights,
    const block_q4_0 *gate_weights, const __half *input,
    __half *output, int tokens) {
    const dim3 grid(EI_N_FF / 16,
                    static_cast<unsigned>((tokens + 15) / 16));
    q4_mfma_up_gate_gelu_f16_kernel<<<grid, 64, 0, engine->stream>>>(
        up_weights, gate_weights, input, output, tokens);
}

static bool tensor_core_attention(
    rocm_engine *engine, const std::vector<uint32_t> &offsets,
    uint32_t minimum_tokens, uint32_t window, bool output_f16,
    char *err, size_t err_len) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    for (size_t sequence = 0; sequence + 1 < offsets.size(); sequence++) {
        const uint32_t start = offsets[sequence];
        const int tokens = static_cast<int>(offsets[sequence + 1] - start);
        if (tokens < static_cast<int>(minimum_tokens)) continue;
        const bool use_fp16_scores = engine->fp16_attention_scores &&
            tokens <= static_cast<int>(engine->fp16_attention_scores_max_tokens);
        if (engine->batched_tensor_attention) {
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
                const hipblasStride matrix_stride =
                    static_cast<hipblasStride>(query_count) * key_count;
                const size_t output_offset =
                    static_cast<size_t>(start + query_start) * EI_N_EMBD;
                void *output = output_f16
                    ? static_cast<void *>(engine->half_input + output_offset)
                    : static_cast<void *>(engine->ctx + output_offset);
                const hipDataType output_type = output_f16
                    ? HIP_R_16F : HIP_R_32F;
                void *scores = use_fp16_scores
                    ? static_cast<void *>(reinterpret_cast<__half *>(
                        engine->attention_scores))
                    : static_cast<void *>(engine->attention_scores);
                const hipDataType score_type = use_fp16_scores
                    ? HIP_R_16F : HIP_R_32F;
                if (!hipblas_check(hipblasGemmStridedBatchedEx(
                        engine->blas, HIPBLAS_OP_T, HIPBLAS_OP_N,
                        static_cast<int>(key_count),
                        static_cast<int>(query_count), EI_HEAD_DIM, &alpha,
                        engine->k_half + static_cast<size_t>(start + key_start) *
                            EI_HEAD_DIM,
                        HIP_R_16F, EI_HEAD_DIM, 0,
                        engine->q_half +
                            static_cast<size_t>(start + query_start) * EI_N_EMBD,
                        HIP_R_16F, EI_N_EMBD, EI_HEAD_DIM,
                        &beta, scores, score_type,
                        static_cast<int>(key_count), matrix_stride, EI_N_HEAD,
                        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
                        "execute batched HIP tensor-core QK", err, err_len)) {
                    return false;
                }
                if (use_fp16_scores) {
                    attention_softmax_f16_kernel<<<query_count * EI_N_HEAD,
                        engine->tensor_softmax_threads, 0, engine->stream>>>(
                            reinterpret_cast<const __half *>(
                                engine->attention_scores),
                            engine->attention_probs, query_start, key_start,
                            query_count, key_count,
                            static_cast<uint32_t>(tokens), window);
                } else {
                    attention_softmax_f16_kernel<<<query_count * EI_N_HEAD,
                        engine->tensor_softmax_threads, 0, engine->stream>>>(
                            engine->attention_scores,
                            engine->attention_probs, query_start, key_start,
                            query_count, key_count,
                            static_cast<uint32_t>(tokens), window);
                }
                if (!hipblas_check(hipblasGemmStridedBatchedEx(
                        engine->blas, HIPBLAS_OP_N, HIPBLAS_OP_N,
                        EI_HEAD_DIM, static_cast<int>(query_count),
                        static_cast<int>(key_count), &alpha,
                        engine->v_half + static_cast<size_t>(start + key_start) *
                            EI_HEAD_DIM,
                        HIP_R_16F, EI_HEAD_DIM, 0,
                        engine->attention_probs, HIP_R_16F,
                        static_cast<int>(key_count), matrix_stride,
                        &beta, output, output_type, EI_N_EMBD, EI_HEAD_DIM,
                        EI_N_HEAD, HIPBLAS_COMPUTE_32F,
                        HIPBLAS_GEMM_DEFAULT),
                        "execute batched HIP tensor-core PV", err, err_len)) {
                    return false;
                }
            }
            continue;
        }
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
                const hipDataType output_type = output_f16
                    ? HIP_R_16F : HIP_R_32F;
                void *scores = use_fp16_scores
                    ? static_cast<void *>(reinterpret_cast<__half *>(
                        engine->attention_scores))
                    : static_cast<void *>(engine->attention_scores);
                const hipDataType score_type = use_fp16_scores
                    ? HIP_R_16F : HIP_R_32F;
                if (!hipblas_check(hipblasGemmEx(
                        engine->blas, HIPBLAS_OP_T, HIPBLAS_OP_N,
                        static_cast<int>(key_count),
                        static_cast<int>(query_count), EI_HEAD_DIM, &alpha,
                        engine->k_half + static_cast<size_t>(start + key_start) *
                            EI_HEAD_DIM,
                        HIP_R_16F, EI_HEAD_DIM,
                        engine->q_half +
                            static_cast<size_t>(start + query_start) * EI_N_EMBD +
                            head * EI_HEAD_DIM,
                        HIP_R_16F, EI_N_EMBD,
                        &beta, scores, score_type,
                        static_cast<int>(key_count),
                        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
                        "execute HIP tensor-core QK", err, err_len)) {
                    return false;
                }
                if (use_fp16_scores) {
                    attention_softmax_f16_kernel<<<query_count,
                        engine->tensor_softmax_threads, 0, engine->stream>>>(
                            reinterpret_cast<const __half *>(
                            engine->attention_scores), engine->attention_probs,
                            query_start, key_start, query_count, key_count,
                            static_cast<uint32_t>(tokens), window);
                } else {
                    attention_softmax_f16_kernel<<<query_count,
                        engine->tensor_softmax_threads, 0, engine->stream>>>(
                            engine->attention_scores,
                            engine->attention_probs, query_start, key_start,
                            query_count, key_count,
                            static_cast<uint32_t>(tokens), window);
                }
                if (!hipblas_check(hipblasGemmEx(
                        engine->blas, HIPBLAS_OP_N, HIPBLAS_OP_N,
                        EI_HEAD_DIM, static_cast<int>(query_count),
                        static_cast<int>(key_count), &alpha,
                        engine->v_half + static_cast<size_t>(start + key_start) *
                            EI_HEAD_DIM,
                        HIP_R_16F, EI_HEAD_DIM,
                        engine->attention_probs, HIP_R_16F,
                        static_cast<int>(key_count),
                        &beta, output, output_type, EI_N_EMBD,
                        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
                        "execute HIP tensor-core PV", err, err_len)) {
                    return false;
                }
            }
        }
    }
    return true;
}

static uint32_t tuned_tensor_attention_threshold(const rocm_engine *engine,
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

static void launch_rms_residual_next(
    rocm_engine *engine, const float *input, const float *post_weight,
    float *residual, const float *next_weight, __half *next_output,
    uint32_t tokens, float eps) {
    if (engine->rms_register_cache) {
        rms_residual_next_f16_cached_kernel<<<row_blocks(tokens), kThreads, 0,
            engine->stream>>>(input, post_weight, residual, next_weight,
                next_output, tokens, eps);
    } else {
        rms_residual_next_f16_kernel<<<row_blocks(tokens), kThreads, 0,
            engine->stream>>>(input, post_weight, residual, next_weight,
                next_output, tokens, EI_N_EMBD, eps);
    }
}

static void launch_rms_residual_next_f32(
    rocm_engine *engine, const float *input, const float *post_weight,
    float *residual, const float *next_weight, float *next_output,
    uint32_t tokens, float eps) {
    rms_residual_next_f32_cached_kernel<<<row_blocks(tokens), kThreads, 0,
        engine->stream>>>(input, post_weight, residual, next_weight,
            next_output, tokens, eps);
}

static bool launch_forward(rocm_engine *engine, uint32_t tokens,
                           uint32_t batch_size, uint32_t attention_tiles,
                           uint32_t mfma_attention_tiles,
                           uint32_t max_sequence_tokens,
                           const std::vector<uint32_t> &host_offsets,
                           char *err, size_t err_len) {
    const ei_model *model = engine->model;
    const bool use_singleton_direct = max_sequence_tokens == 1 &&
        tokens <= engine->singleton_direct_max_tokens;
    const bool use_gemm = tokens >= engine->gemm_min_tokens &&
        !use_singleton_direct;
    const bool use_native_q4 = use_gemm && engine->native_q4_gemm &&
        tokens <= engine->native_q4_max_tokens;
    const bool use_fp16_attention =
        max_sequence_tokens >= engine->fp16_attention_min_tokens;
    const bool use_v_only = engine->single_token_v_only &&
        max_sequence_tokens == 1;
    const bool use_direct_rms_fusion = !use_gemm && !engine->q8_latency &&
        engine->direct_rms_fusion;
    const bool use_direct_q4_pair = !use_gemm && !engine->q8_latency &&
        engine->direct_q4_pair && tokens >= 16;
    const bool use_direct_q4_quad = use_direct_q4_pair &&
        engine->direct_q4_quad;
    const bool use_fused_embedding_rms = engine->fused_embedding_rms &&
        !use_gemm && !engine->q8_latency;
    const bool use_final_singleton_pool = engine->final_singleton_pool &&
        max_sequence_tokens == 1 && !use_gemm;
    const bool direct_fp16_qkv = use_gemm && use_fp16_attention &&
        !use_native_q4 && engine->direct_fp16_qkv;
    const bool native_direct_fp16_qkv = use_native_q4 && use_fp16_attention &&
        engine->native_q4_direct_fp16_qkv;
    const bool direct_fp16_context = use_gemm && use_fp16_attention &&
        engine->direct_fp16_context;
    const uint32_t tensor_attention_threshold =
        tuned_tensor_attention_threshold(engine, batch_size);
    const block_q8_0 *embedding = reinterpret_cast<const block_q8_0 *>(
        engine->model_data + model->token_embd->offset);
    const float embedding_scale = std::sqrt(static_cast<float>(EI_N_EMBD));
    if (use_fused_embedding_rms) {
        const float *first_weight = float_weights(
            engine, model->layers[0].attn_norm);
        if (use_gemm) {
            embedding_rms_kernel<true><<<row_blocks(tokens), kThreads, 0,
                engine->stream>>>(embedding, engine->ids, first_weight,
                    engine->x, nullptr, engine->half_input, tokens,
                    embedding_scale, model->rms_eps);
        } else {
            embedding_rms_kernel<false><<<row_blocks(tokens), kThreads, 0,
                engine->stream>>>(embedding, engine->ids, first_weight,
                    engine->x, engine->norm, nullptr, tokens,
                    embedding_scale, model->rms_eps);
        }
    } else {
        const size_t embedding_count = static_cast<size_t>(tokens) * EI_N_EMBD;
        embedding_kernel<<<static_cast<unsigned>(
            (embedding_count + kThreads - 1) / kThreads), kThreads, 0,
            engine->stream>>>(embedding, engine->ids, engine->x, tokens,
                              embedding_scale);
    }

    for (int layer_index = 0; layer_index < EI_N_LAYER; layer_index++) {
        const ei_layer *layer = &model->layers[layer_index];
        if (use_gemm) {
            if (layer_index == 0 && !use_fused_embedding_rms) {
                rms_norm_f16_kernel<<<row_blocks(tokens), kThreads, 0,
                    engine->stream>>>(engine->x,
                        float_weights(engine, layer->attn_norm),
                        engine->half_input, tokens, EI_N_EMBD, model->rms_eps);
            }
            if (use_v_only && use_native_q4) {
                native_q4_v_only_projection(engine,
                    q4_weights(engine, layer->attn_v), engine->half_input,
                    engine->q_half, tokens);
            } else if (use_v_only) {
                if (!tensor_core_projection(engine,
                        engine->layers[layer_index].attn_v,
                        engine->half_input, engine->v, EI_HEAD_DIM,
                        EI_N_EMBD, tokens, err, err_len)) return false;
                repeat_value_f32_to_f16_kernel<<<static_cast<unsigned>(
                    (static_cast<size_t>(tokens) * EI_N_EMBD + kThreads - 1) /
                        kThreads), kThreads, 0, engine->stream>>>(
                            engine->v, engine->q_half, tokens);
            } else if (use_native_q4) {
                if (engine->native_q4_fused) {
                    native_q4_qkv_projection(engine,
                        q4_weights(engine, layer->attn_q),
                        q4_weights(engine, layer->attn_k),
                        q4_weights(engine, layer->attn_v), engine->half_input,
                        engine->q, engine->k, engine->v, tokens);
                } else {
                    native_q4_projection(engine,
                        q4_weights(engine, layer->attn_q), engine->half_input,
                        engine->q, EI_N_EMBD, EI_N_EMBD, tokens);
                    native_q4_projection(engine,
                        q4_weights(engine, layer->attn_k), engine->half_input,
                        engine->k, EI_HEAD_DIM, EI_N_EMBD, tokens);
                    native_q4_projection(engine,
                        q4_weights(engine, layer->attn_v), engine->half_input,
                        engine->v, EI_HEAD_DIM, EI_N_EMBD, tokens);
                }
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
        } else if (use_v_only) {
            if ((layer_index == 0 && !use_fused_embedding_rms) ||
                (layer_index > 0 && !use_direct_rms_fusion)) {
                rms_norm_kernel<false><<<row_blocks(tokens), kThreads, 0,
                    engine->stream>>>(engine->x,
                        float_weights(engine, layer->attn_norm), engine->norm,
                        tokens, EI_N_EMBD, model->rms_eps);
            }
            if (use_direct_q4_quad) {
                v_only_projection_quad_kernel<<<projection_blocks(tokens,
                    EI_HEAD_DIM / 4), kThreads, 0, engine->stream>>>(
                        q4_weights(engine, layer->attn_v), engine->norm,
                        engine->ctx, tokens);
            } else if (use_direct_q4_pair) {
                v_only_projection_pair_kernel<<<projection_blocks(tokens,
                    EI_HEAD_DIM / 2), kThreads, 0, engine->stream>>>(
                        q4_weights(engine, layer->attn_v), engine->norm,
                        engine->ctx, tokens);
            } else {
                v_only_projection_kernel<<<projection_blocks(tokens,
                    EI_HEAD_DIM), kThreads, 0, engine->stream>>>(
                        q4_weights(engine, layer->attn_v), engine->norm,
                        engine->ctx, tokens);
            }
        } else if (engine->q8_latency) {
            rms_norm_quantize_q8_kernel<<<row_blocks(tokens), kThreads, 0,
                engine->stream>>>(engine->x,
                    float_weights(engine, layer->attn_norm), engine->q8_input,
                    tokens, EI_N_EMBD, model->rms_eps);
            if (engine->q8_two_row) {
                qkv_q8_projection_pair_kernel<<<projection_blocks(tokens,
                    (EI_N_EMBD + 2 * EI_HEAD_DIM) / 2), kThreads, 0,
                    engine->stream>>>(q4_weights(engine, layer->attn_q),
                        q4_weights(engine, layer->attn_k),
                        q4_weights(engine, layer->attn_v), engine->q8_input,
                        engine->q, engine->k, engine->v, tokens);
            } else {
                qkv_q8_projection_kernel<<<projection_blocks(tokens,
                    EI_N_EMBD + 2 * EI_HEAD_DIM), kThreads, 0,
                    engine->stream>>>(q4_weights(engine, layer->attn_q),
                        q4_weights(engine, layer->attn_k),
                        q4_weights(engine, layer->attn_v), engine->q8_input,
                        engine->q, engine->k, engine->v, tokens);
            }
        } else {
            if ((layer_index == 0 && !use_fused_embedding_rms) ||
                (layer_index > 0 && !use_direct_rms_fusion)) {
                rms_norm_kernel<false><<<row_blocks(tokens), kThreads, 0,
                    engine->stream>>>(engine->x,
                        float_weights(engine, layer->attn_norm), engine->norm,
                        tokens, EI_N_EMBD, model->rms_eps);
            }
            if (use_direct_q4_quad) {
                qkv_projection_quad_kernel<<<projection_blocks(tokens,
                    (EI_N_EMBD + 2 * EI_HEAD_DIM) / 4), kThreads, 0,
                    engine->stream>>>(q4_weights(engine, layer->attn_q),
                        q4_weights(engine, layer->attn_k),
                        q4_weights(engine, layer->attn_v), engine->norm,
                        engine->q, engine->k, engine->v, tokens);
            } else if (use_direct_q4_pair) {
                qkv_projection_pair_kernel<<<projection_blocks(tokens,
                    (EI_N_EMBD + 2 * EI_HEAD_DIM) / 2), kThreads, 0,
                    engine->stream>>>(q4_weights(engine, layer->attn_q),
                        q4_weights(engine, layer->attn_k),
                        q4_weights(engine, layer->attn_v), engine->norm,
                        engine->q, engine->k, engine->v, tokens);
            } else {
                qkv_projection_kernel<<<projection_blocks(tokens,
                    EI_N_EMBD + 2 * EI_HEAD_DIM), kThreads, 0,
                    engine->stream>>>(q4_weights(engine, layer->attn_q),
                        q4_weights(engine, layer->attn_k),
                        q4_weights(engine, layer->attn_v), engine->norm,
                        engine->q, engine->k, engine->v, tokens);
            }
        }
        if (!use_v_only) {
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
            } else if (native_direct_fp16_qkv) {
            qkv_separate_norm_rope_f16_kernel<<<projection_blocks(tokens,
                EI_N_HEAD + EI_N_HEAD_KV + 1), kThreads, 0, engine->stream>>>(
                    engine->q, engine->k, engine->v,
                    engine->q_half, engine->k_half, engine->v_half,
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
            if (!direct_fp16_qkv && !native_direct_fp16_qkv) {
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
                if (mfma_attention_tiles != 0) {
                    if (direct_fp16_context) {
                        mfma_attention_f16_kernel<__half><<<dim3(
                            mfma_attention_tiles, EI_N_HEAD), 64, 0,
                            engine->stream>>>(engine->q_half, engine->k_half,
                                engine->v_half, engine->half_input,
                                engine->mfma_query_start,
                                engine->mfma_sequence_start,
                                engine->mfma_sequence_stop,
                                mfma_attention_tiles, window);
                    } else {
                        mfma_attention_f16_kernel<float><<<dim3(
                            mfma_attention_tiles, EI_N_HEAD), 64, 0,
                            engine->stream>>>(engine->q_half, engine->k_half,
                                engine->v_half, engine->ctx,
                                engine->mfma_query_start,
                                engine->mfma_sequence_start,
                                engine->mfma_sequence_stop,
                                mfma_attention_tiles, window);
                    }
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
        }
        if (use_gemm) {
            if (!direct_fp16_context && !use_v_only) {
                convert_to_half(engine, engine->ctx,
                                static_cast<size_t>(tokens) * EI_N_EMBD);
            }
            const __half *attention_input = use_v_only
                ? engine->q_half : engine->half_input;
            if (use_native_q4) {
                native_q4_projection(engine,
                    q4_weights(engine, layer->attn_output), attention_input,
                    engine->tmp, EI_N_EMBD, EI_N_EMBD, tokens);
            } else if (!tensor_core_projection(engine,
                           engine->layers[layer_index].attn_output,
                           attention_input, engine->tmp, EI_N_EMBD,
                           EI_N_EMBD, tokens, err, err_len)) {
                return false;
            }
        } else if (engine->q8_latency) {
            quantize_q8_kernel<<<row_blocks(static_cast<size_t>(tokens) *
                (EI_N_EMBD / kQK)), kThreads, 0, engine->stream>>>(
                    engine->ctx, engine->q8_input, tokens, EI_N_EMBD);
            if (engine->q8_two_row) {
                q4_q8_projection_pair_kernel<<<projection_blocks(tokens,
                    EI_N_EMBD / 2), kThreads, 0, engine->stream>>>(
                        q4_weights(engine, layer->attn_output),
                        engine->q8_input, engine->tmp, EI_N_EMBD, EI_N_EMBD,
                        tokens);
            } else {
                q4_q8_projection_kernel<<<projection_blocks(tokens, EI_N_EMBD),
                                          kThreads, 0, engine->stream>>>(
                    q4_weights(engine, layer->attn_output), engine->q8_input,
                    engine->tmp, EI_N_EMBD, EI_N_EMBD, tokens);
            }
        } else if (use_direct_q4_quad) {
            q4_projection_quad_kernel<<<projection_blocks(tokens,
                EI_N_EMBD / 4), kThreads, 0, engine->stream>>>(
                    q4_weights(engine, layer->attn_output), engine->ctx,
                    engine->tmp, EI_N_EMBD, EI_N_EMBD, tokens);
        } else if (use_direct_q4_pair) {
            q4_projection_pair_kernel<<<projection_blocks(tokens,
                EI_N_EMBD / 2), kThreads, 0, engine->stream>>>(
                    q4_weights(engine, layer->attn_output), engine->ctx,
                    engine->tmp, EI_N_EMBD, EI_N_EMBD, tokens);
        } else {
            q4_projection_kernel<<<projection_blocks(tokens, EI_N_EMBD),
                                    kThreads, 0, engine->stream>>>(
                q4_weights(engine, layer->attn_output), engine->ctx,
                engine->tmp, EI_N_EMBD, EI_N_EMBD, tokens);
        }
        if (use_gemm) {
            launch_rms_residual_next(engine, engine->tmp,
                float_weights(engine, layer->post_attention_norm), engine->x,
                float_weights(engine, layer->ffn_norm), engine->half_input,
                tokens, model->rms_eps);
            const size_t ffn_count = static_cast<size_t>(tokens) * EI_N_FF;
            if (use_native_q4) {
                if (engine->native_q4_fused) {
                    if (engine->native_q4_fused_activation &&
                        tokens >= engine->native_q4_fused_activation_min_tokens) {
                        native_q4_up_gate_activation(engine,
                            q4_weights(engine, layer->ffn_up),
                            q4_weights(engine, layer->ffn_gate),
                            engine->half_input,
                            reinterpret_cast<__half *>(engine->up), tokens);
                    } else {
                        native_q4_up_gate_projection(engine,
                            q4_weights(engine, layer->ffn_up),
                            q4_weights(engine, layer->ffn_gate),
                            engine->half_input, engine->up_gate_combined,
                            tokens);
                        up_gate_gelu_to_f16_kernel<<<static_cast<unsigned>(
                            (ffn_count + kThreads - 1) / kThreads), kThreads,
                            0, engine->stream>>>(engine->up_gate_combined,
                                                engine->half_input, tokens);
                    }
                } else {
                    native_q4_projection(engine,
                        q4_weights(engine, layer->ffn_up), engine->half_input,
                        engine->up, EI_N_FF, EI_N_EMBD, tokens);
                    native_q4_projection(engine,
                        q4_weights(engine, layer->ffn_gate), engine->half_input,
                        engine->gate, EI_N_FF, EI_N_EMBD, tokens);
                    gelu_mul_kernel<<<static_cast<unsigned>(
                        (ffn_count + kThreads - 1) / kThreads), kThreads, 0,
                        engine->stream>>>(engine->gate, engine->up, ffn_count);
                    convert_to_half(engine, engine->gate, ffn_count);
                }
                const __half *ffn_input = engine->native_q4_fused &&
                    engine->native_q4_fused_activation &&
                    tokens >= engine->native_q4_fused_activation_min_tokens
                    ? reinterpret_cast<const __half *>(engine->up)
                    : engine->half_input;
                native_q4_projection(engine, q4_weights(engine, layer->ffn_down),
                                     ffn_input, engine->tmp,
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
                launch_rms_residual_next(engine, engine->tmp,
                    float_weights(engine, layer->post_ffw_norm), engine->x,
                    float_weights(engine,
                        model->layers[layer_index + 1].attn_norm),
                    engine->half_input, tokens, model->rms_eps);
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
            if (engine->q8_two_row) {
                up_gate_q8_gelu_pair_kernel<<<projection_blocks(tokens,
                    EI_N_FF), kThreads, 0, engine->stream>>>(
                        q4_weights(engine, layer->ffn_up),
                        q4_weights(engine, layer->ffn_gate), engine->q8_input,
                        engine->gate, tokens);
            } else {
                up_gate_q8_gelu_kernel<<<projection_blocks(tokens, EI_N_FF),
                                         kThreads, 0, engine->stream>>>(
                    q4_weights(engine, layer->ffn_up),
                    q4_weights(engine, layer->ffn_gate), engine->q8_input,
                    engine->gate, tokens);
            }
            quantize_q8_kernel<<<row_blocks(static_cast<size_t>(tokens) *
                (EI_N_FF / kQK)), kThreads, 0, engine->stream>>>(
                    engine->gate, engine->q8_input, tokens, EI_N_FF);
            if (engine->q8_two_row) {
                q4_q8_projection_pair_kernel<<<projection_blocks(tokens,
                    EI_N_EMBD / 2), kThreads, 0, engine->stream>>>(
                        q4_weights(engine, layer->ffn_down), engine->q8_input,
                        engine->tmp, EI_N_EMBD, EI_N_FF, tokens);
            } else {
                q4_q8_projection_kernel<<<projection_blocks(tokens, EI_N_EMBD),
                                          kThreads, 0, engine->stream>>>(
                    q4_weights(engine, layer->ffn_down), engine->q8_input,
                    engine->tmp, EI_N_EMBD, EI_N_FF, tokens);
            }
            if (layer_index + 1 < EI_N_LAYER || !use_final_singleton_pool) {
                rms_norm_kernel<true><<<row_blocks(tokens), kThreads, 0,
                    engine->stream>>>(engine->tmp,
                        float_weights(engine, layer->post_ffw_norm), engine->x,
                        tokens, EI_N_EMBD, model->rms_eps);
            }
        } else {
            if (use_direct_rms_fusion) {
                launch_rms_residual_next_f32(engine, engine->tmp,
                    float_weights(engine, layer->post_attention_norm),
                    engine->x, float_weights(engine, layer->ffn_norm),
                    engine->norm, tokens, model->rms_eps);
            } else {
                rms_norm_kernel<true><<<row_blocks(tokens), kThreads, 0,
                    engine->stream>>>(engine->tmp,
                        float_weights(engine, layer->post_attention_norm),
                        engine->x, tokens, EI_N_EMBD, model->rms_eps);
                rms_norm_kernel<false><<<row_blocks(tokens), kThreads, 0,
                    engine->stream>>>(engine->x,
                        float_weights(engine, layer->ffn_norm), engine->norm,
                        tokens, EI_N_EMBD, model->rms_eps);
            }
            if (use_direct_q4_pair) {
                up_gate_gelu_pair_kernel<<<projection_blocks(tokens, EI_N_FF),
                    kThreads, 0, engine->stream>>>(
                        q4_weights(engine, layer->ffn_up),
                        q4_weights(engine, layer->ffn_gate), engine->norm,
                        engine->gate, tokens);
                if (use_direct_q4_quad) {
                    q4_projection_quad_kernel<<<projection_blocks(tokens,
                        EI_N_EMBD / 4), kThreads, 0, engine->stream>>>(
                            q4_weights(engine, layer->ffn_down), engine->gate,
                            engine->tmp, EI_N_EMBD, EI_N_FF, tokens);
                } else {
                    q4_projection_pair_kernel<<<projection_blocks(tokens,
                        EI_N_EMBD / 2), kThreads, 0, engine->stream>>>(
                            q4_weights(engine, layer->ffn_down), engine->gate,
                            engine->tmp, EI_N_EMBD, EI_N_FF, tokens);
                }
            } else {
                up_gate_gelu_kernel<<<projection_blocks(tokens, EI_N_FF),
                                      kThreads, 0, engine->stream>>>(
                    q4_weights(engine, layer->ffn_up),
                    q4_weights(engine, layer->ffn_gate), engine->norm,
                    engine->gate, tokens);
                q4_projection_kernel<<<projection_blocks(tokens, EI_N_EMBD),
                                        kThreads, 0, engine->stream>>>(
                    q4_weights(engine, layer->ffn_down), engine->gate,
                    engine->tmp, EI_N_EMBD, EI_N_FF, tokens);
            }
            if (use_direct_rms_fusion && layer_index + 1 < EI_N_LAYER) {
                launch_rms_residual_next_f32(engine, engine->tmp,
                    float_weights(engine, layer->post_ffw_norm), engine->x,
                    float_weights(engine,
                        model->layers[layer_index + 1].attn_norm),
                    engine->norm, tokens, model->rms_eps);
            } else if (layer_index + 1 < EI_N_LAYER ||
                       !use_final_singleton_pool) {
                rms_norm_kernel<true><<<row_blocks(tokens), kThreads, 0,
                    engine->stream>>>(engine->tmp,
                        float_weights(engine, layer->post_ffw_norm), engine->x,
                        tokens, EI_N_EMBD, model->rms_eps);
            }
        }
    }

    if (use_final_singleton_pool) {
        const ei_layer *last = &model->layers[EI_N_LAYER - 1];
        final_singleton_pool_kernel<<<row_blocks(batch_size), kThreads, 0,
            engine->stream>>>(engine->tmp,
                float_weights(engine, last->post_ffw_norm), engine->x,
                float_weights(engine, model->output_norm), engine->result,
                batch_size, model->rms_eps);
    } else {
        pool_kernel<<<row_blocks(batch_size), kThreads, 0, engine->stream>>>(
            engine->x, float_weights(engine, model->output_norm), engine->result,
            engine->offsets, batch_size, model->rms_eps);
    }
    return hip_check(hipGetLastError(), "launch HIP forward kernels", err, err_len);
}

static bool execute_forward(rocm_engine *engine, uint32_t tokens,
                            uint32_t batch_size, uint32_t attention_tiles,
                            uint32_t mfma_attention_tiles,
                            uint32_t max_sequence_tokens, uint64_t shape_hash,
                            const std::vector<uint32_t> &host_offsets,
                            char *err, size_t err_len) {
    if (!engine->command_graph) {
        return launch_forward(engine, tokens, batch_size, attention_tiles,
                              mfma_attention_tiles, max_sequence_tokens,
                              host_offsets, err, err_len);
    }
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
                (void)hipGraphExecDestroy(engine->graphs.front().executable);
            }
            engine->graphs.erase(engine->graphs.begin());
        }
        engine->graphs.push_back({tokens, batch_size, shape_hash, nullptr});
        return launch_forward(engine, tokens, batch_size, attention_tiles,
                              mfma_attention_tiles, max_sequence_tokens,
                              host_offsets, err, err_len);
    }
    if (entry->executable) {
        return hip_check(hipGraphLaunch(entry->executable, engine->stream),
                          "launch HIP inference graph", err, err_len);
    }

    if (!hip_check(hipStreamBeginCapture(engine->stream,
                    hipStreamCaptureModeThreadLocal),
                    "begin HIP inference graph capture", err, err_len)) {
        return false;
    }
    if (!launch_forward(engine, tokens, batch_size, attention_tiles,
                        mfma_attention_tiles, max_sequence_tokens,
                        host_offsets, err, err_len)) {
        hipGraph_t abandoned = nullptr;
        if (hipStreamEndCapture(engine->stream, &abandoned) == hipSuccess && abandoned) {
            (void)hipGraphDestroy(abandoned);
        }
        return false;
    }
    hipGraph_t graph = nullptr;
    if (!hip_check(hipStreamEndCapture(engine->stream, &graph),
                    "finish HIP inference graph capture", err, err_len)) {
        return false;
    }
    hipGraphExec_t executable = nullptr;
    hipError_t status = hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
    (void)hipGraphDestroy(graph);
    if (!hip_check(status, "instantiate HIP inference graph", err, err_len)) {
        return false;
    }
    entry->executable = executable;
    return hip_check(hipGraphLaunch(entry->executable, engine->stream),
                      "launch captured HIP inference graph", err, err_len);
}

static bool embed_batch(rocm_engine *engine, const int32_t *ids,
                        const size_t *offsets, size_t batch_size, float *out,
                        char *err, size_t err_len) {
    const size_t token_count = offsets[batch_size];
    if (!ensure_capacity(engine, token_count, batch_size, err, err_len)) return false;

    std::vector<uint32_t> device_offsets(batch_size + 1);
    uint32_t max_sequence_tokens = 0;
    const uint32_t tensor_attention_threshold =
        tuned_tensor_attention_threshold(engine,
                                          static_cast<uint32_t>(batch_size));
    uint64_t shape_hash = 1469598103934665603ull;
    bool all_singleton = true;
    for (size_t sequence = 0; sequence <= batch_size; sequence++) {
        device_offsets[sequence] = static_cast<uint32_t>(offsets[sequence]);
        shape_hash ^= static_cast<uint64_t>(offsets[sequence]);
        shape_hash *= 1099511628211ull;
        if (sequence < batch_size) {
            const uint32_t sequence_tokens = static_cast<uint32_t>(
                offsets[sequence + 1] - offsets[sequence]);
            if (sequence_tokens > max_sequence_tokens) {
                max_sequence_tokens = sequence_tokens;
            }
            all_singleton = all_singleton && sequence_tokens == 1;
        }
    }
    const bool elide_singleton_metadata =
        engine->singleton_metadata_elision && engine->single_token_v_only &&
        all_singleton;
    std::vector<uint32_t> positions(
        elide_singleton_metadata ? 0 : token_count);
    std::vector<uint32_t> sequence_ids(
        elide_singleton_metadata ? 0 : token_count);
    std::vector<uint32_t> flash_query_start;
    std::vector<uint32_t> flash_sequence_start;
    std::vector<uint32_t> flash_sequence_stop;
    std::vector<uint32_t> mfma_query_start;
    std::vector<uint32_t> mfma_sequence_start;
    std::vector<uint32_t> mfma_sequence_stop;
    for (size_t sequence = 0; sequence < batch_size; sequence++) {
        const uint32_t start = static_cast<uint32_t>(offsets[sequence]);
        const uint32_t stop = static_cast<uint32_t>(offsets[sequence + 1]);
        if (!elide_singleton_metadata &&
            (!engine->tensor_attention_tuned ||
             stop - start < tensor_attention_threshold)) {
            const uint32_t sequence_tokens = stop - start;
            if (engine->mfma_attention &&
                sequence_tokens >= engine->mfma_attention_min_tokens &&
                sequence_tokens <= engine->mfma_attention_max_tokens) {
                for (uint32_t query = start; query < stop; query += 16) {
                    mfma_query_start.push_back(query);
                    mfma_sequence_start.push_back(start);
                    mfma_sequence_stop.push_back(stop);
                }
            } else {
                for (uint32_t query = start; query < stop;
                     query += kWarpsPerBlock) {
                    flash_query_start.push_back(query);
                    flash_sequence_start.push_back(start);
                    flash_sequence_stop.push_back(stop);
                }
            }
        }
        if (!elide_singleton_metadata) {
            for (size_t token = offsets[sequence];
                 token < offsets[sequence + 1]; token++) {
                positions[token] = static_cast<uint32_t>(
                    token - offsets[sequence]);
                sequence_ids[token] = static_cast<uint32_t>(sequence);
            }
        }
    }

    const int32_t *ids_source = ids;
    const uint32_t *offsets_source = device_offsets.data();
    if (engine->pinned_io_staging) {
        std::memcpy(engine->host_ids, ids, token_count * sizeof(*ids));
        std::memcpy(engine->host_offsets, device_offsets.data(),
                    (batch_size + 1) * sizeof(uint32_t));
        ids_source = engine->host_ids;
        offsets_source = engine->host_offsets;
    }
    if (!hip_check(hipMemcpyAsync(engine->ids, ids_source,
                    token_count * sizeof(*ids), hipMemcpyHostToDevice,
                    engine->stream), "copy HIP token IDs", err, err_len) ||
        !hip_check(hipMemcpyAsync(engine->offsets, offsets_source,
                    (batch_size + 1) * sizeof(uint32_t), hipMemcpyHostToDevice,
                    engine->stream), "copy HIP offsets", err, err_len)) {
        return false;
    }
    if (!elide_singleton_metadata &&
        (!hip_check(hipMemcpyAsync(engine->positions, positions.data(),
                    token_count * sizeof(uint32_t), hipMemcpyHostToDevice,
                    engine->stream), "copy HIP positions", err, err_len) ||
        !hip_check(hipMemcpyAsync(engine->sequence_ids, sequence_ids.data(),
                    token_count * sizeof(uint32_t), hipMemcpyHostToDevice,
                    engine->stream), "copy HIP sequence IDs", err, err_len) ||
        !hip_check(hipMemcpyAsync(engine->flash_query_start,
                    flash_query_start.data(),
                    flash_query_start.size() * sizeof(uint32_t),
                    hipMemcpyHostToDevice, engine->stream),
                    "copy HIP attention query tiles", err, err_len) ||
        !hip_check(hipMemcpyAsync(engine->flash_sequence_start,
                    flash_sequence_start.data(),
                    flash_sequence_start.size() * sizeof(uint32_t),
                    hipMemcpyHostToDevice, engine->stream),
                    "copy HIP attention sequence starts", err, err_len) ||
        !hip_check(hipMemcpyAsync(engine->flash_sequence_stop,
                    flash_sequence_stop.data(),
                    flash_sequence_stop.size() * sizeof(uint32_t),
                    hipMemcpyHostToDevice, engine->stream),
                    "copy HIP attention sequence stops", err, err_len) ||
        !hip_check(hipMemcpyAsync(engine->mfma_query_start,
                    mfma_query_start.data(),
                    mfma_query_start.size() * sizeof(uint32_t),
                    hipMemcpyHostToDevice, engine->stream),
                    "copy HIP MFMA attention query tiles", err, err_len) ||
        !hip_check(hipMemcpyAsync(engine->mfma_sequence_start,
                    mfma_sequence_start.data(),
                    mfma_sequence_start.size() * sizeof(uint32_t),
                    hipMemcpyHostToDevice, engine->stream),
                    "copy HIP MFMA attention sequence starts", err, err_len) ||
        !hip_check(hipMemcpyAsync(engine->mfma_sequence_stop,
                    mfma_sequence_stop.data(),
                    mfma_sequence_stop.size() * sizeof(uint32_t),
                    hipMemcpyHostToDevice, engine->stream),
                    "copy HIP MFMA attention sequence stops", err, err_len))) {
        return false;
    }
    if (!execute_forward(engine, static_cast<uint32_t>(token_count),
                         static_cast<uint32_t>(batch_size),
                         static_cast<uint32_t>(flash_query_start.size()),
                         static_cast<uint32_t>(mfma_query_start.size()),
                         max_sequence_tokens, shape_hash, device_offsets,
                         err, err_len)) {
        return false;
    }
    float *result_destination = engine->pinned_io_staging
        ? engine->host_result : out;
    if (!hip_check(hipMemcpyAsync(result_destination, engine->result,
                    batch_size * EI_N_EMBD * sizeof(float),
                    hipMemcpyDeviceToHost, engine->stream),
                    "copy HIP embeddings", err, err_len) ||
        !hip_check(hipStreamSynchronize(engine->stream),
                    "execute HIP forward pass", err, err_len)) {
        return false;
    }
    if (engine->pinned_io_staging) {
        std::memcpy(out, engine->host_result,
                    batch_size * EI_N_EMBD * sizeof(float));
    }
    if (err && err_len) err[0] = '\0';
    return true;
}

}  // namespace

extern "C" void *ei_rocm_engine_create(const ei_model *model,
                                         char *err, size_t err_len) {
    int device_count = 0;
    if (!hip_check(hipGetDeviceCount(&device_count), "enumerate HIP devices",
                    err, err_len)) return nullptr;
    if (device_count == 0) {
        set_error(err, err_len, "no HIP devices are available");
        return nullptr;
    }
    int device = 0;
    const char *device_env = std::getenv("EI_ROCM_DEVICE");
    if (device_env && *device_env) {
        char *end = nullptr;
        long parsed = std::strtol(device_env, &end, 10);
        if (*end != '\0' || parsed < 0 || parsed >= device_count) {
            set_error(err, err_len, "EI_ROCM_DEVICE is outside the available device range");
            return nullptr;
        }
        device = static_cast<int>(parsed);
    }
    if (!hip_check(hipSetDevice(device), "select HIP device", err, err_len)) {
        return nullptr;
    }
    hipDeviceProp_t device_properties;
    if (!hip_check(hipGetDeviceProperties(&device_properties, device),
                    "inspect HIP device", err, err_len)) {
        return nullptr;
    }

    rocm_engine *engine = new rocm_engine;
    engine->model = model;
    engine->device = device;
    engine->compute_units = device_properties.multiProcessorCount;
    const char *command_graph_env = std::getenv("EI_ROCM_COMMAND_GRAPH");
    if (command_graph_env && *command_graph_env) {
        if (std::strcmp(command_graph_env, "0") != 0 &&
            std::strcmp(command_graph_env, "1") != 0) {
            set_error(err, err_len, "EI_ROCM_COMMAND_GRAPH must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->command_graph = command_graph_env[0] == '1';
    }
    const char *gemm_min_env = std::getenv("EI_ROCM_GEMM_MIN_TOKENS");
    if (gemm_min_env && *gemm_min_env) {
        char *end = nullptr;
        long parsed = std::strtol(gemm_min_env, &end, 10);
        if (*end != '\0' || parsed < 1 || parsed > 65536) {
            set_error(err, err_len,
                      "EI_ROCM_GEMM_MIN_TOKENS must be an integer from 1 to 65536");
            delete engine;
            return nullptr;
        }
        engine->gemm_min_tokens = static_cast<uint32_t>(parsed);
    }
    const char *singleton_direct_env = std::getenv(
        "EI_ROCM_SINGLETON_DIRECT_MAX_TOKENS");
    if (singleton_direct_env && *singleton_direct_env) {
        char *end = nullptr;
        long parsed = std::strtol(singleton_direct_env, &end, 10);
        if (*end != '\0' || parsed < 0 || parsed > 65536) {
            set_error(err, err_len,
                "EI_ROCM_SINGLETON_DIRECT_MAX_TOKENS must be an integer from 0 to 65536");
            delete engine;
            return nullptr;
        }
        engine->singleton_direct_max_tokens = static_cast<uint32_t>(parsed);
    }
    const char *fp16_attention_env = std::getenv(
        "EI_ROCM_FP16_ATTENTION_MIN_TOKENS");
    if (fp16_attention_env && *fp16_attention_env) {
        char *end = nullptr;
        long parsed = std::strtol(fp16_attention_env, &end, 10);
        if (*end != '\0' || parsed < 1 || parsed > 65536) {
            set_error(err, err_len,
                "EI_ROCM_FP16_ATTENTION_MIN_TOKENS must be an integer from 1 to 65536");
            delete engine;
            return nullptr;
        }
        engine->fp16_attention_min_tokens = static_cast<uint32_t>(parsed);
    }
    const char *direct_fp16_qkv_env = std::getenv("EI_ROCM_DIRECT_FP16_QKV");
    if (direct_fp16_qkv_env && *direct_fp16_qkv_env) {
        if (std::strcmp(direct_fp16_qkv_env, "0") != 0 &&
            std::strcmp(direct_fp16_qkv_env, "1") != 0) {
            set_error(err, err_len, "EI_ROCM_DIRECT_FP16_QKV must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->direct_fp16_qkv = direct_fp16_qkv_env[0] == '1';
    }
    const char *direct_fp16_context_env = std::getenv(
        "EI_ROCM_DIRECT_FP16_CONTEXT");
    if (direct_fp16_context_env && *direct_fp16_context_env) {
        if (std::strcmp(direct_fp16_context_env, "0") != 0 &&
            std::strcmp(direct_fp16_context_env, "1") != 0) {
            set_error(err, err_len,
                "EI_ROCM_DIRECT_FP16_CONTEXT must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->direct_fp16_context = direct_fp16_context_env[0] == '1';
    }
    const char *mfma_attention_env = std::getenv("EI_ROCM_MFMA_ATTENTION");
    if (mfma_attention_env && *mfma_attention_env) {
        if (std::strcmp(mfma_attention_env, "0") != 0 &&
            std::strcmp(mfma_attention_env, "1") != 0) {
            set_error(err, err_len, "EI_ROCM_MFMA_ATTENTION must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->mfma_attention = mfma_attention_env[0] == '1';
    }
    const char *flash_attention_env = std::getenv(
        "EI_ROCM_FLASH_ATTENTION_MIN_TOKENS");
    if (flash_attention_env && *flash_attention_env) {
        char *end = nullptr;
        long parsed = std::strtol(flash_attention_env, &end, 10);
        if (*end != '\0' || parsed < 1 || parsed > 65536) {
            set_error(err, err_len,
                "EI_ROCM_FLASH_ATTENTION_MIN_TOKENS must be an integer from 1 to 65536");
            delete engine;
            return nullptr;
        }
        engine->flash_attention_min_tokens = static_cast<uint32_t>(parsed);
    }
    const char *flash_attention_max_env = std::getenv(
        "EI_ROCM_FLASH_ATTENTION_MAX_TOKENS");
    if (flash_attention_max_env && *flash_attention_max_env) {
        char *end = nullptr;
        long parsed = std::strtol(flash_attention_max_env, &end, 10);
        if (*end != '\0' || parsed < 1 || parsed > 65536) {
            set_error(err, err_len,
                "EI_ROCM_FLASH_ATTENTION_MAX_TOKENS must be an integer from 1 to 65536");
            delete engine;
            return nullptr;
        }
        engine->flash_attention_max_tokens = static_cast<uint32_t>(parsed);
    }
    const char *tensor_attention_env = std::getenv(
        "EI_ROCM_TENSOR_ATTENTION_MIN_TOKENS");
    if (tensor_attention_env && *tensor_attention_env) {
        char *end = nullptr;
        long parsed = std::strtol(tensor_attention_env, &end, 10);
        if (*end != '\0' || parsed < 1 || parsed > 65536) {
            set_error(err, err_len,
                "EI_ROCM_TENSOR_ATTENTION_MIN_TOKENS must be an integer from 1 to 65536");
            delete engine;
            return nullptr;
        }
        engine->tensor_attention_min_tokens = static_cast<uint32_t>(parsed);
        engine->tensor_attention_tuned = false;
    }
    const char *batched_attention_env = std::getenv(
        "EI_ROCM_BATCHED_TENSOR_ATTENTION");
    if (batched_attention_env && *batched_attention_env) {
        if (std::strcmp(batched_attention_env, "0") != 0 &&
            std::strcmp(batched_attention_env, "1") != 0) {
            set_error(err, err_len,
                "EI_ROCM_BATCHED_TENSOR_ATTENTION must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->batched_tensor_attention = batched_attention_env[0] == '1';
    }
    const char *fp16_attention_scores_env = std::getenv(
        "EI_ROCM_FP16_ATTENTION_SCORES");
    if (fp16_attention_scores_env && *fp16_attention_scores_env) {
        if (std::strcmp(fp16_attention_scores_env, "0") != 0 &&
            std::strcmp(fp16_attention_scores_env, "1") != 0) {
            set_error(err, err_len,
                "EI_ROCM_FP16_ATTENTION_SCORES must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->fp16_attention_scores = fp16_attention_scores_env[0] == '1';
    }
    const char *softmax_threads_env = std::getenv(
        "EI_ROCM_TENSOR_SOFTMAX_THREADS");
    if (softmax_threads_env && *softmax_threads_env) {
        char *end = nullptr;
        long parsed = std::strtol(softmax_threads_env, &end, 10);
        if (*end != '\0' || (parsed != 64 && parsed != 128 && parsed != 256)) {
            set_error(err, err_len,
                "EI_ROCM_TENSOR_SOFTMAX_THREADS must be 64, 128, or 256");
            delete engine;
            return nullptr;
        }
        engine->tensor_softmax_threads = static_cast<uint32_t>(parsed);
    }
    const char *swa_tile_env = std::getenv("EI_ROCM_SWA_TENSOR_TILE_TOKENS");
    if (swa_tile_env && *swa_tile_env) {
        char *end = nullptr;
        long parsed = std::strtol(swa_tile_env, &end, 10);
        if (*end != '\0' || (parsed != 0 && parsed != 128 && parsed != 256 &&
                             parsed != 512 && parsed != 1024)) {
            set_error(err, err_len,
                "EI_ROCM_SWA_TENSOR_TILE_TOKENS must be 0, 128, 256, 512, or 1024");
            delete engine;
            return nullptr;
        }
        engine->swa_tensor_tile_tokens = static_cast<uint32_t>(parsed);
    }
    const char *swa_min_env = std::getenv(
        "EI_ROCM_SWA_TENSOR_MIN_TOKENS");
    if (swa_min_env && *swa_min_env) {
        char *end = nullptr;
        long parsed = std::strtol(swa_min_env, &end, 10);
        if (*end != '\0' || parsed < 1 || parsed > 65536) {
            set_error(err, err_len,
                "EI_ROCM_SWA_TENSOR_MIN_TOKENS must be an integer from 1 to 65536");
            delete engine;
            return nullptr;
        }
        engine->swa_tensor_banded_min_tokens = static_cast<uint32_t>(parsed);
    }
    const char *native_q4_env = std::getenv("EI_ROCM_NATIVE_Q4_GEMM");
    if (native_q4_env && *native_q4_env) {
        if (std::strcmp(native_q4_env, "0") != 0 &&
            std::strcmp(native_q4_env, "1") != 0) {
            set_error(err, err_len,
                      "EI_ROCM_NATIVE_Q4_GEMM must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->native_q4_gemm = std::strcmp(native_q4_env, "1") == 0;
    }
    const char *native_q4_max_env = std::getenv(
        "EI_ROCM_NATIVE_Q4_MAX_TOKENS");
    if (native_q4_max_env && *native_q4_max_env) {
        char *end = nullptr;
        long parsed = std::strtol(native_q4_max_env, &end, 10);
        if (*end != '\0' || parsed < 1 || parsed > 65536) {
            set_error(err, err_len,
                "EI_ROCM_NATIVE_Q4_MAX_TOKENS must be an integer from 1 to 65536");
            delete engine;
            return nullptr;
        }
        engine->native_q4_max_tokens = static_cast<uint32_t>(parsed);
    }
    const char *native_q4_fused_env = std::getenv("EI_ROCM_NATIVE_Q4_FUSED");
    if (native_q4_fused_env && *native_q4_fused_env) {
        if (std::strcmp(native_q4_fused_env, "0") != 0 &&
            std::strcmp(native_q4_fused_env, "1") != 0) {
            set_error(err, err_len, "EI_ROCM_NATIVE_Q4_FUSED must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->native_q4_fused = native_q4_fused_env[0] == '1';
    }
    const char *native_q4_fused_activation_env = std::getenv(
        "EI_ROCM_NATIVE_Q4_FUSED_ACTIVATION");
    if (native_q4_fused_activation_env && *native_q4_fused_activation_env) {
        if (std::strcmp(native_q4_fused_activation_env, "0") != 0 &&
            std::strcmp(native_q4_fused_activation_env, "1") != 0) {
            set_error(err, err_len,
                "EI_ROCM_NATIVE_Q4_FUSED_ACTIVATION must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->native_q4_fused_activation =
            native_q4_fused_activation_env[0] == '1';
    }
    const char *native_q4_wide_env = std::getenv("EI_ROCM_NATIVE_Q4_WIDE");
    if (native_q4_wide_env && *native_q4_wide_env) {
        if (std::strcmp(native_q4_wide_env, "0") != 0 &&
            std::strcmp(native_q4_wide_env, "1") != 0) {
            set_error(err, err_len, "EI_ROCM_NATIVE_Q4_WIDE must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->native_q4_wide = native_q4_wide_env[0] == '1';
    }
    const char *native_q4_direct_qkv_env = std::getenv(
        "EI_ROCM_NATIVE_Q4_DIRECT_FP16_QKV");
    if (native_q4_direct_qkv_env && *native_q4_direct_qkv_env) {
        if (std::strcmp(native_q4_direct_qkv_env, "0") != 0 &&
            std::strcmp(native_q4_direct_qkv_env, "1") != 0) {
            set_error(err, err_len,
                "EI_ROCM_NATIVE_Q4_DIRECT_FP16_QKV must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->native_q4_direct_fp16_qkv = native_q4_direct_qkv_env[0] == '1';
    }
    const char *rms_register_cache_env = std::getenv(
        "EI_ROCM_RMS_REGISTER_CACHE");
    if (rms_register_cache_env && *rms_register_cache_env) {
        if (std::strcmp(rms_register_cache_env, "0") != 0 &&
            std::strcmp(rms_register_cache_env, "1") != 0) {
            set_error(err, err_len,
                "EI_ROCM_RMS_REGISTER_CACHE must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->rms_register_cache = rms_register_cache_env[0] == '1';
    }
    const char *q8_latency_env = std::getenv("EI_ROCM_Q8_LATENCY");
    if (q8_latency_env && *q8_latency_env) {
        if (std::strcmp(q8_latency_env, "0") != 0 &&
            std::strcmp(q8_latency_env, "1") != 0) {
            set_error(err, err_len, "EI_ROCM_Q8_LATENCY must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->q8_latency = std::strcmp(q8_latency_env, "1") == 0;
    }
    const char *q8_two_row_env = std::getenv("EI_ROCM_Q8_TWO_ROW");
    if (q8_two_row_env && *q8_two_row_env) {
        if (std::strcmp(q8_two_row_env, "0") != 0 &&
            std::strcmp(q8_two_row_env, "1") != 0) {
            set_error(err, err_len, "EI_ROCM_Q8_TWO_ROW must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->q8_two_row = std::strcmp(q8_two_row_env, "1") == 0;
    }
    const char *single_token_v_only_env = std::getenv(
        "EI_ROCM_SINGLE_TOKEN_V_ONLY");
    if (single_token_v_only_env && *single_token_v_only_env) {
        if (std::strcmp(single_token_v_only_env, "0") != 0 &&
            std::strcmp(single_token_v_only_env, "1") != 0) {
            set_error(err, err_len,
                "EI_ROCM_SINGLE_TOKEN_V_ONLY must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->single_token_v_only = single_token_v_only_env[0] == '1';
    }
    const char *direct_rms_fusion_env = std::getenv(
        "EI_ROCM_DIRECT_RMS_FUSION");
    if (direct_rms_fusion_env && *direct_rms_fusion_env) {
        if (std::strcmp(direct_rms_fusion_env, "0") != 0 &&
            std::strcmp(direct_rms_fusion_env, "1") != 0) {
            set_error(err, err_len, "EI_ROCM_DIRECT_RMS_FUSION must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->direct_rms_fusion = direct_rms_fusion_env[0] == '1';
    }
    const char *direct_q4_pair_env = std::getenv("EI_ROCM_DIRECT_Q4_PAIR");
    if (direct_q4_pair_env && *direct_q4_pair_env) {
        if (std::strcmp(direct_q4_pair_env, "0") != 0 &&
            std::strcmp(direct_q4_pair_env, "1") != 0) {
            set_error(err, err_len, "EI_ROCM_DIRECT_Q4_PAIR must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->direct_q4_pair = direct_q4_pair_env[0] == '1';
    }
    const char *direct_q4_quad_env = std::getenv("EI_ROCM_DIRECT_Q4_QUAD");
    if (direct_q4_quad_env && *direct_q4_quad_env) {
        if (std::strcmp(direct_q4_quad_env, "0") != 0 &&
            std::strcmp(direct_q4_quad_env, "1") != 0) {
            set_error(err, err_len, "EI_ROCM_DIRECT_Q4_QUAD must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->direct_q4_quad = direct_q4_quad_env[0] == '1';
    }
    const char *pinned_io_env = std::getenv("EI_ROCM_PINNED_IO_STAGING");
    if (pinned_io_env && *pinned_io_env) {
        if (std::strcmp(pinned_io_env, "0") != 0 &&
            std::strcmp(pinned_io_env, "1") != 0) {
            set_error(err, err_len, "EI_ROCM_PINNED_IO_STAGING must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->pinned_io_staging = pinned_io_env[0] == '1';
    }
    const char *fused_embedding_rms_env = std::getenv(
        "EI_ROCM_FUSED_EMBEDDING_RMS");
    if (fused_embedding_rms_env && *fused_embedding_rms_env) {
        if (std::strcmp(fused_embedding_rms_env, "0") != 0 &&
            std::strcmp(fused_embedding_rms_env, "1") != 0) {
            set_error(err, err_len,
                "EI_ROCM_FUSED_EMBEDDING_RMS must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->fused_embedding_rms = fused_embedding_rms_env[0] == '1';
    }
    const char *singleton_metadata_env = std::getenv(
        "EI_ROCM_SINGLETON_METADATA_ELISION");
    if (singleton_metadata_env && *singleton_metadata_env) {
        if (std::strcmp(singleton_metadata_env, "0") != 0 &&
            std::strcmp(singleton_metadata_env, "1") != 0) {
            set_error(err, err_len,
                "EI_ROCM_SINGLETON_METADATA_ELISION must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->singleton_metadata_elision = singleton_metadata_env[0] == '1';
    }
    const char *final_singleton_pool_env = std::getenv(
        "EI_ROCM_FINAL_SINGLETON_POOL");
    if (final_singleton_pool_env && *final_singleton_pool_env) {
        if (std::strcmp(final_singleton_pool_env, "0") != 0 &&
            std::strcmp(final_singleton_pool_env, "1") != 0) {
            set_error(err, err_len,
                "EI_ROCM_FINAL_SINGLETON_POOL must be 0 or 1");
            delete engine;
            return nullptr;
        }
        engine->final_singleton_pool = final_singleton_pool_env[0] == '1';
    }
    if (!hip_check(hipStreamCreateWithFlags(&engine->stream, hipStreamNonBlocking),
                    "create HIP stream", err, err_len)) {
        delete engine;
        return nullptr;
    }
    if (!hipblas_check(hipblasCreate(&engine->blas), "create hipBLAS handle",
                      err, err_len) ||
        !hipblas_check(hipblasSetStream(engine->blas, engine->stream),
                      "set hipBLAS stream", err, err_len)) {
        ei_rocm_engine_free(engine);
        return nullptr;
    }
    const uint8_t *model_bytes = model->gguf.map + model->gguf.data_off;
    const size_t model_length = model->gguf.map_len - model->gguf.data_off;
    if (!hip_allocate(&engine->model_data, model_length,
                       "allocate HIP model", err, err_len) ||
        !hip_check(hipMemcpyAsync(engine->model_data, model_bytes, model_length,
                    hipMemcpyHostToDevice, engine->stream),
                    "copy HIP model", err, err_len) ||
        !initialize_dequantized_weights(engine, err, err_len)) {
        ei_rocm_engine_free(engine);
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
    const size_t attention_matrix_count =
        engine->batched_tensor_attention ? EI_N_HEAD : 1;
    if (!hip_allocate(&engine->rope_full, rope_count,
                       "allocate HIP full RoPE table", err, err_len) ||
        !hip_allocate(&engine->rope_swa, rope_count,
                       "allocate HIP SWA RoPE table", err, err_len) ||
        !hip_allocate(&engine->attention_scores,
                       static_cast<size_t>(EI_N_CTX) * EI_N_CTX *
                           attention_matrix_count,
                       "allocate HIP attention scores", err, err_len) ||
        !hip_allocate(&engine->attention_probs,
                       static_cast<size_t>(EI_N_CTX) * EI_N_CTX *
                           attention_matrix_count,
                       "allocate HIP attention probabilities", err, err_len) ||
        !hip_check(hipMemcpyAsync(engine->rope_full, full.data(),
                    rope_count * sizeof(float2), hipMemcpyHostToDevice,
                    engine->stream), "copy HIP full RoPE table", err, err_len) ||
        !hip_check(hipMemcpyAsync(engine->rope_swa, swa.data(),
                    rope_count * sizeof(float2), hipMemcpyHostToDevice,
                    engine->stream), "copy HIP SWA RoPE table", err, err_len) ||
        !hip_check(hipStreamSynchronize(engine->stream),
                    "initialize HIP backend", err, err_len)) {
        ei_rocm_engine_free(engine);
        return nullptr;
    }
    if (err && err_len) err[0] = '\0';
    return engine;
}

extern "C" void ei_rocm_engine_free(void *handle) {
    rocm_engine *engine = static_cast<rocm_engine *>(handle);
    if (!engine) return;
    (void)hipSetDevice(engine->device);
    if (engine->stream) (void)hipStreamSynchronize(engine->stream);
    release_token_workspace(engine);
    release_batch_workspace(engine);
    hip_release(engine->rope_swa);
    hip_release(engine->rope_full);
    hip_release(engine->attention_probs);
    hip_release(engine->attention_scores);
    hip_release(engine->dequantized_weights);
    hip_release(engine->model_data);
    if (engine->blas) (void)hipblasDestroy(engine->blas);
    if (engine->stream) (void)hipStreamDestroy(engine->stream);
    delete engine;
}

extern "C" bool ei_rocm_engine_reserve(void *handle, size_t total_tokens,
                                        size_t batch_size, char *err,
                                        size_t err_len) {
    rocm_engine *engine = static_cast<rocm_engine *>(handle);
    if (!engine) {
        set_error(err, err_len, "HIP backend is not initialized");
        return false;
    }
    if (!hip_check(hipSetDevice(engine->device), "select HIP device",
                    err, err_len)) return false;
    const bool ok = ensure_capacity(engine, total_tokens, batch_size, err, err_len);
    if (ok && err && err_len) err[0] = '\0';
    return ok;
}

extern "C" bool ei_rocm_engine_embed_tokens(void *handle,
                                             const int32_t *ids,
                                             size_t n_tokens,
                                             float out[EI_N_EMBD],
                                             char *err, size_t err_len) {
    const size_t offsets[2] = {0, n_tokens};
    return ei_rocm_engine_embed_tokens_batch(handle, ids, offsets, 1, out,
                                             err, err_len);
}

extern "C" bool ei_rocm_engine_embed_tokens_batch(void *handle,
                                                    const int32_t *ids,
                                                    const size_t *offsets,
                                                    size_t batch_size,
                                                    float *out, char *err,
                                                    size_t err_len) {
    rocm_engine *engine = static_cast<rocm_engine *>(handle);
    if (!engine) {
        set_error(err, err_len, "HIP backend is not initialized");
        return false;
    }
    if (!hip_check(hipSetDevice(engine->device), "select HIP device",
                    err, err_len)) return false;
    return embed_batch(engine, ids, offsets, batch_size, out, err, err_len);
}
