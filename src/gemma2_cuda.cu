// EmbeddingGemma 2 text tower on CUDA: packed multi-sequence waves.
// Mirrors the Metal fused route (perf/gemma2_kernel_design.md): fp16 row-major
// copies of every text GEMM weight (Q/K/V and up/gate concatenated) for cuBLAS
// tensor-core GEMMs with fp32 accumulation, fused row kernels writing fp16
// GEMM inputs, varlen flash attention with per-token sequence bounds and the
// sliding window, and CUDA graphs keyed by the wave shape.
#include "gemma2_cuda.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <string>

extern "C" void ei_l2_normalize(float *x, int32_t n);
#include <unordered_map>
#include <vector>

namespace {

constexpr int kModel = 512, kLayers = 24, kPle = 512 * 24, kFfn = 2048, kOut = 768;
constexpr unsigned kWarpMask = 0xffffffffu;

struct G2Q8 {
    __half d;
    int8_t qs[32];
};
static_assert(sizeof(G2Q8) == 34, "q8_0 block layout");

struct graph_entry {
    uint32_t tokens, batch;
    cudaGraphExec_t exec;
};

struct layer_weights {
    __half *qkv, *o, *upgate, *down, *inp_gate, *proj;
    float *attn_norm, *q_norm, *k_norm, *post_attn, *ffn_norm, *post_ffw, *post_norm;
    float output_scale;
};

struct engine {
    const ei_gguf *g = nullptr;
    int device = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    void *blas_workspace = nullptr;
    // Weights.
    G2Q8 *token_embd = nullptr;
    __half *ple_proj = nullptr, *output = nullptr;
    float *ple_norm = nullptr, *output_norm = nullptr;
    float2 *rope[2] = {nullptr, nullptr}; // (cos, sin) [pos][d/2]: base 1e4/d 256, 1e6/d 512
    layer_weights layers[kLayers];
    std::vector<void *> owned;
    // Workspace (capacity tokens / sequences).
    size_t cap = 0, cap_batch = 0;
    int32_t *ids = nullptr;
    uint32_t *lo = nullptr, *hi = nullptr, *pos = nullptr, *offsets = nullptr;
    float *x = nullptr, *ple = nullptr, *proj = nullptr, *wide = nullptr, *pooled = nullptr,
          *inputs = nullptr;
    __half *h16 = nullptr, *q16 = nullptr, *k16 = nullptr, *v16 = nullptr, *a16 = nullptr,
           *g16 = nullptr, *x16 = nullptr;
    std::vector<graph_entry> graphs;
    // Media towers: projector weights converted on first use, scratch by name.
    const ei_gguf *projector = nullptr;
    std::unordered_map<std::string, __half *> half_w;
    std::unordered_map<std::string, float *> float_w;
    std::unordered_map<std::string, std::pair<void *, size_t>> scratch;
    float *rpe[12] = {};
    bool use_graphs = true;
    bool mma_attention = true; // EI_GEMMA2_CUDA_MMA_ATTN=0 selects the CUDA-core kernel
};

bool check(cudaError_t status, const char *what, char *err, size_t len) {
    if (status == cudaSuccess)
        return true;
    snprintf(err, len, "EmbeddingGemma 2 CUDA %s: %s", what, cudaGetErrorString(status));
    return false;
}
bool check_blas(cublasStatus_t status, const char *what, char *err, size_t len) {
    if (status == CUBLAS_STATUS_SUCCESS)
        return true;
    snprintf(err, len, "EmbeddingGemma 2 cuBLAS %s: status %d", what, (int)status);
    return false;
}

// ---- host-side weight conversion ------------------------------------------

float host_element(const ei_tensor *t, size_t i) {
    switch (t->type) {
    case EI_T_F32:
        return ((const float *)t->data)[i];
    case EI_T_F16: {
        __half_raw raw;
        raw.x = ((const uint16_t *)t->data)[i];
        return __half2float(__half(raw));
    }
    case EI_T_BF16: {
        uint32_t bits = (uint32_t)((const uint16_t *)t->data)[i] << 16;
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }
    default: {
        const G2Q8 *b = (const G2Q8 *)t->data + i / 32;
        return __half2float(b->d) * (float)b->qs[i % 32];
    }
    }
}

template <typename T>
bool upload(engine *e, T **dst, const void *src, size_t bytes, char *err, size_t len) {
    if (!check(cudaMalloc((void **)dst, bytes), "allocate weights", err, len))
        return false;
    e->owned.push_back(*dst);
    // On the engine stream: a pageable cudaMemcpy may return before its DMA lands,
    // and the legacy default stream does not order with our non-blocking
    // stream, so lazily uploaded projector weights raced the kernels using them.
    return check(cudaMemcpyAsync(*dst, src, bytes, cudaMemcpyHostToDevice, e->stream),
                 "upload weights", err, len) &&
           check(cudaStreamSynchronize(e->stream), "upload weights", err, len);
}

// fp16 row-major [rows][k] from one or more [m_i][k] tensors stacked by rows.
bool upload_half(engine *e, __half **dst, const std::vector<const ei_tensor *> &parts, char *err,
                 size_t len) {
    size_t k = (size_t)parts[0]->ne[0], rows = 0;
    for (const ei_tensor *t : parts)
        rows += (size_t)t->ne[1];
    std::vector<__half> host(rows * k);
    size_t at = 0;
    for (const ei_tensor *t : parts) {
        size_t n = (size_t)t->ne[0] * (size_t)t->ne[1];
        for (size_t i = 0; i < n; i++)
            host[at + i] = __float2half(host_element(t, i));
        at += n;
    }
    return upload(e, dst, host.data(), host.size() * sizeof(__half), err, len);
}
bool upload_float(engine *e, float **dst, const ei_tensor *t, char *err, size_t len) {
    size_t n = (size_t)t->ne[0];
    std::vector<float> host(n);
    for (size_t i = 0; i < n; i++)
        host[i] = host_element(t, i);
    return upload(e, dst, host.data(), n * sizeof(float), err, len);
}
const ei_tensor *tensor(const ei_gguf *g, const char *fmt, int layer = -1) {
    char name[128];
    snprintf(name, sizeof name, fmt, layer);
    return ei_gguf_tensor(g, name, true);
}

// ---- device helpers ---------------------------------------------------------

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(kWarpMask, v, o);
    return v;
}
// Block-wide sum for blocks of up to 32 warps; all threads get the result.
__device__ float block_sum(float v) {
    __shared__ float partial[32];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0)
        partial[warp] = v;
    __syncthreads();
    float total = lane < warps ? partial[lane] : 0.0f;
    return warp_sum(total);
}
__device__ __forceinline__ float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * x * (1.0f + 0.044715f * x * x)));
}

// ---- kernels -----------------------------------------------------------------

// x[t] = dequant(token_embd[id]) * sqrt(512), (d * q) * s like the reference.
__global__ void embed_kernel(const G2Q8 *table, const int32_t *ids, float *x, __half *x16) {
    int t = blockIdx.x, z = threadIdx.x * 4;
    const G2Q8 *row = table + (size_t)ids[t] * (kModel / 32);
    for (int j = 0; j < 4; j++) {
        const G2Q8 &b = row[(z + j) / 32];
        float v = __half2float(b.d) * (float)b.qs[(z + j) % 32] * sqrtf(512.0f);
        x[(size_t)t * kModel + z + j] = v;
        x16[(size_t)t * kModel + z + j] = __float2half(v);
    }
}
__global__ void to_half_kernel(const float *x, __half *y, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = __float2half(x[i]);
}
// PLE rows (token, layer): v / sqrt(512), then weighted RMS. Block 128 x 4.
__global__ void ple_norm_kernel(float *ple, const float *w) {
    float *r = ple + (size_t)blockIdx.x * kModel;
    int z = threadIdx.x * 4;
    float v[4], ss = 0;
    for (int j = 0; j < 4; j++) {
        v[j] = r[z + j] * (1.0f / sqrtf(512.0f));
        ss += v[j] * v[j];
    }
    float inv = rsqrtf(block_sum(ss) / kModel + 1e-6f);
    for (int j = 0; j < 4; j++)
        r[z + j] = v[j] * inv * w[z + j];
}
// out16 = rms(x) * w over rows of width W (block W / 4 threads x 4 values).
__global__ void rms_half_kernel(const float *x, const float *w, __half *out, int width) {
    const float *r = x + (size_t)blockIdx.x * width;
    int z = threadIdx.x * 4;
    float v[4], ss = 0;
    for (int j = 0; j < 4; j++) {
        v[j] = r[z + j];
        ss += v[j] * v[j];
    }
    float inv = rsqrtf(block_sum(ss) / width + 1e-6f);
    for (int j = 0; j < 4; j++)
        out[(size_t)blockIdx.x * width + z + j] = __float2half(v[j] * inv * (w ? w[z + j] : 1.0f));
}
// Per (token, slot): slots [0,4) q heads, [4,4+kv) k, [4+kv,4+2kv) v from the
// concatenated QKV output (row stride 4d + 1024). Weighted RMS for q/k, plain
// for v, NEOX RoPE from the table for q/k. Block d/2 threads: thread i owns
// elements i and i + d/2 (a RoPE pair).
__global__ void qkv_post_kernel(const float *qkv, int d, int kv, const float *qn,
                                const float *kn, const float2 *rope, const uint32_t *pos,
                                __half *q16, __half *k16, __half *v16) {
    int t = blockIdx.x, slot = blockIdx.y, i = threadIdx.x, half_d = d >> 1;
    size_t stride = 4 * (size_t)d + 1024, col;
    const float *w = nullptr;
    __half *dst;
    if (slot < 4) {
        col = (size_t)slot * d, w = qn, dst = q16 + ((size_t)t * 4 + slot) * d;
    } else if (slot < 4 + kv) {
        col = 4 * (size_t)d + (size_t)(slot - 4) * d, w = kn;
        dst = k16 + ((size_t)t * kv + slot - 4) * d;
    } else {
        col = 4 * (size_t)d + 512 + (size_t)(slot - 4 - kv) * d;
        dst = v16 + ((size_t)t * kv + slot - 4 - kv) * d;
    }
    const float *r = qkv + (size_t)t * stride + col;
    float a = r[i], b = r[i + half_d];
    float inv = rsqrtf(block_sum(a * a + b * b) / d + 1e-6f);
    a *= inv, b *= inv;
    if (w) {
        a *= w[i], b *= w[i + half_d];
        float2 cs = rope[(size_t)pos[t] * half_d + i];
        float ra = a * cs.x - b * cs.y, rb = a * cs.y + b * cs.x;
        a = ra, b = rb;
    }
    dst[i] = __float2half(a);
    dst[i + half_d] = __float2half(b);
}
// CUDA-core varlen flash attention (any GPU), one block = 8 queries (one warp
// each) x one head. K/V tiles of 16 keys staged in shared memory; online
// softmax in fp32. Each query sees keys in [lo, hi) of its sequence (lo/hi
// NULL: one sequence of n tokens) and, if window != 0, |t - s| <= window.
// q/out are [t][heads][D], k/v [t][kv][D].
template <int D>
__global__ void attention_kernel(const __half *q16, const __half *k16, const __half *v16,
                                 __half *out, const uint32_t *lo, const uint32_t *hi, int n,
                                 int heads, int kv, int window) {
    constexpr int QB = 8, KT = 16, PER = D / 32;
    __shared__ __half sk[KT * D], sv[KT * D];
    int head = blockIdx.y, kh = head / (heads / kv), warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int t0 = blockIdx.x * QB, t = t0 + warp;
    bool active = t < n;
    int tl = min(t0 + QB, n) - 1;
    int qlo = 0, qhi = 0;
    if (active) {
        qlo = lo ? lo[t] : 0, qhi = hi ? hi[t] : n;
        if (window) {
            qlo = max(qlo, t - window);
            qhi = min(qhi, t + window + 1);
        }
    }
    int ulo = lo ? lo[t0] : 0, uhi = hi ? hi[tl] : n;
    if (window) {
        ulo = max(ulo, t0 - window);
        uhi = min(uhi, tl + window + 1);
    }
    float qv[PER], ov[PER];
#pragma unroll
    for (int j = 0; j < PER; j++) {
        qv[j] = active ? __half2float(q16[((size_t)t * heads + head) * D + lane + 32 * j]) : 0.0f;
        ov[j] = 0.0f;
    }
    float mx = -INFINITY, den = 0.0f;
    for (int base = ulo; base < uhi; base += KT) {
        int keys = min(KT, uhi - base);
        for (int it = threadIdx.x; it < KT * D; it += blockDim.x) {
            int lk = it / D, dim = it - lk * D;
            bool in = lk < keys;
            size_t src = ((size_t)(base + lk) * kv + kh) * D + dim;
            sk[it] = in ? k16[src] : __float2half(0.0f);
            sv[it] = in ? v16[src] : __float2half(0.0f);
        }
        __syncthreads();
        if (active)
            for (int lk = 0; lk < keys; lk++) {
                int s = base + lk;
                if (s < qlo || s >= qhi)
                    continue;
                float sc = 0.0f;
#pragma unroll
                for (int j = 0; j < PER; j++)
                    sc = fmaf(qv[j], __half2float(sk[lk * D + lane + 32 * j]), sc);
                sc = warp_sum(sc);
                float next = fmaxf(mx, sc), alpha = __expf(mx - next), beta = __expf(sc - next);
                den = den * alpha + beta;
#pragma unroll
                for (int j = 0; j < PER; j++)
                    ov[j] = fmaf(beta, __half2float(sv[lk * D + lane + 32 * j]), ov[j] * alpha);
                mx = next;
            }
        __syncthreads();
    }
    if (!active)
        return;
    float inv = 1.0f / den;
#pragma unroll
    for (int j = 0; j < PER; j++)
        out[((size_t)t * heads + head) * D + lane + 32 * j] = __float2half(ov[j] * inv);
}
// Tensor-core varlen flash attention (mma.sync m16n8k16, fp16 in, fp32 acc).
// Block = W warps x 16 query rows for one head and DO output dims (blockIdx.z
// picks the DO slice of D). Shared: Q [W*16][D], K [KT][D] (row-major), V^T
// [DO][KT]; rows padded by 8 halves. Each thread holds query rows g and g+8
// (g = lane / 4) of its warp's S and O tiles, so the online-softmax rescale
// stays in registers; S accumulators become the A operand of P.V directly.
// Needs sm_80+ (Ampere and later: 3090, 4090, 5090, ...). Fat builds include
// older targets; the host never launches tensor-core attention there.
__device__ __forceinline__ void mma16816(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    (void)c, (void)a, (void)b0, (void)b1;
    __trap();
#endif
}
__device__ __forceinline__ uint32_t pack_half2(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *(uint32_t *)&h;
}
template <int D, int DO, int W, int KT, bool QS>
__global__ void __launch_bounds__(W * 32)
    flash_mma_kernel(const __half *q16, const __half *k16, const __half *v16, __half *out,
                     const uint32_t *lo, const uint32_t *hi, int n, int heads, int kv,
                     int window) {
    constexpr int QR = W * 16, DP = D + 8, KP = KT + 8, NT = KT / 8, OT = DO / 8;
    extern __shared__ __align__(16) __half smem[];
    // QS: stage Q in shared memory; otherwise A fragments load from global (L1).
    __half *qs = smem, *ks = qs + (QS ? QR * DP : 0), *vt = ks + KT * DP;
    const int head = blockIdx.y, kh = head / (heads / kv), d0 = blockIdx.z * DO;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
    const int t0 = blockIdx.x * QR, tl = min(t0 + QR, n) - 1;
    // Stage Q rows (zero past n).
    for (int i = threadIdx.x; QS && i < QR * (D / 8); i += W * 32) {
        int r = i / (D / 8), c = (i % (D / 8)) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (t0 + r < n)
            v = *(const uint4 *)(q16 + ((size_t)(t0 + r) * heads + head) * D + c);
        *(uint4 *)(qs + r * DP + c) = v;
    }
    // Per-row bounds for this thread's rows g and g + 8.
    int rlo[2], rhi[2];
    for (int h2 = 0; h2 < 2; h2++) {
        int t = t0 + warp * 16 + g + 8 * h2, tc = min(t, n - 1);
        rlo[h2] = lo ? lo[tc] : 0, rhi[h2] = hi ? hi[tc] : n; // no bounds: one sequence
        if (window) {
            rlo[h2] = max(rlo[h2], tc - window);
            rhi[h2] = min(rhi[h2], tc + window + 1);
        }
        if (t >= n)
            rhi[h2] = rlo[h2]; // padded row: no keys
    }
    int ulo = lo ? lo[t0] : 0, uhi = hi ? hi[tl] : n;
    if (window) {
        ulo = max(ulo, t0 - window);
        uhi = min(uhi, tl + window + 1);
    }
    float o[OT][4], mx[2] = {-INFINITY, -INFINITY}, den[2] = {0, 0};
#pragma unroll
    for (int j = 0; j < OT; j++)
        o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0;
    for (int base = ulo; base < uhi; base += KT) {
        __syncthreads();
        for (int i = threadIdx.x; i < KT * (D / 8); i += W * 32) {
            int r = i / (D / 8), c = (i % (D / 8)) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (base + r < uhi)
                v = *(const uint4 *)(k16 + ((size_t)(base + r) * kv + kh) * D + c);
            *(uint4 *)(ks + r * DP + c) = v;
        }
        for (int i = threadIdx.x; i < KT * DO; i += W * 32) {
            int r = i / DO, c = i % DO;
            vt[c * KP + r] = base + r < uhi ? v16[((size_t)(base + r) * kv + kh) * D + d0 + c]
                                            : __float2half(0.0f);
        }
        __syncthreads();
        // S = Q K^T for this warp's 16 rows.
        float sc[NT][4];
#pragma unroll
        for (int j = 0; j < NT; j++)
            sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0;
        const __half *qw = qs + (warp * 16) * DP;
        // Global Q rows for QS = false (clamped past n; their rows are masked).
        const int r0 = min(t0 + warp * 16 + g, n - 1), r1 = min(t0 + warp * 16 + g + 8, n - 1);
        const __half *qg0 = q16 + ((size_t)r0 * heads + head) * D,
                     *qg1 = q16 + ((size_t)r1 * heads + head) * D;
#pragma unroll 4
        for (int c = 0; c < D; c += 16) {
            uint32_t a[4];
            if (QS) {
                a[0] = *(const uint32_t *)(qw + g * DP + c + 2 * tq);
                a[1] = *(const uint32_t *)(qw + (g + 8) * DP + c + 2 * tq);
                a[2] = *(const uint32_t *)(qw + g * DP + c + 8 + 2 * tq);
                a[3] = *(const uint32_t *)(qw + (g + 8) * DP + c + 8 + 2 * tq);
            } else {
                a[0] = __ldg((const unsigned int *)(qg0 + c + 2 * tq));
                a[1] = __ldg((const unsigned int *)(qg1 + c + 2 * tq));
                a[2] = __ldg((const unsigned int *)(qg0 + c + 8 + 2 * tq));
                a[3] = __ldg((const unsigned int *)(qg1 + c + 8 + 2 * tq));
            }
#pragma unroll
            for (int j = 0; j < NT; j++) {
                const __half *kr = ks + (j * 8 + g) * DP + c + 2 * tq;
                mma16816(sc[j], a, *(const uint32_t *)kr, *(const uint32_t *)(kr + 8));
            }
        }
        // Mask, online softmax per row (rows g: elements 0,1; g + 8: 2,3).
        float tmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int r = e >> 1, s = base + j * 8 + 2 * tq + (e & 1);
                if (s < rlo[r] || s >= rhi[r])
                    sc[j][e] = -INFINITY;
                tmax[r] = fmaxf(tmax[r], sc[j][e]);
            }
        float alpha[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(kWarpMask, tmax[r], 1));
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(kWarpMask, tmax[r], 2));
            float next = fmaxf(mx[r], tmax[r]);
            alpha[r] = next == -INFINITY ? 1.0f : __expf(mx[r] - next);
            mx[r] = next;
        }
        float rsum[2] = {0, 0};
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int r = e >> 1;
                float p = mx[r] == -INFINITY ? 0.0f : __expf(sc[j][e] - mx[r]);
                sc[j][e] = p;
                rsum[r] += p;
            }
#pragma unroll
        for (int r = 0; r < 2; r++) {
            rsum[r] += __shfl_xor_sync(kWarpMask, rsum[r], 1);
            rsum[r] += __shfl_xor_sync(kWarpMask, rsum[r], 2);
            den[r] = den[r] * alpha[r] + rsum[r];
        }
#pragma unroll
        for (int j = 0; j < OT; j++) {
            o[j][0] *= alpha[0], o[j][1] *= alpha[0];
            o[j][2] *= alpha[1], o[j][3] *= alpha[1];
        }
        // O += P V: two adjacent 16x8 S tiles form one 16x16 A fragment.
#pragma unroll
        for (int kc = 0; kc < KT / 16; kc++) {
            uint32_t a[4];
            a[0] = pack_half2(sc[2 * kc][0], sc[2 * kc][1]);
            a[1] = pack_half2(sc[2 * kc][2], sc[2 * kc][3]);
            a[2] = pack_half2(sc[2 * kc + 1][0], sc[2 * kc + 1][1]);
            a[3] = pack_half2(sc[2 * kc + 1][2], sc[2 * kc + 1][3]);
#pragma unroll
            for (int j = 0; j < OT; j++) {
                const __half *vr = vt + (j * 8 + g) * KP + kc * 16 + 2 * tq;
                mma16816(o[j], a, *(const uint32_t *)vr, *(const uint32_t *)(vr + 8));
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        int t = t0 + warp * 16 + g + 8 * r;
        if (t >= n)
            continue;
        float inv = den[r] > 0 ? 1.0f / den[r] : 0.0f;
        __half *dst = out + ((size_t)t * heads + head) * D + d0;
#pragma unroll
        for (int j = 0; j < OT; j++)
            *(uint32_t *)(dst + j * 8 + 2 * tq) = pack_half2(o[j][2 * r] * inv, o[j][2 * r + 1] * inv);
    }
}
template <int D, int DO, int W, int KT, bool QS>
constexpr size_t flash_smem() {
    return ((QS ? (size_t)W * 16 * (D + 8) : 0) + (size_t)KT * (D + 8) + (size_t)DO * (KT + 8)) * 2;
}
// Called once at create: cudaFuncSetAttribute is not legal inside graph capture.
template <int D, int DO, int W, int KT, bool QS>
cudaError_t configure_flash() {
    return cudaFuncSetAttribute(flash_mma_kernel<D, DO, W, KT, QS>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)flash_smem<D, DO, W, KT, QS>());
}
template <int D, int DO, int W, int KT, bool QS>
void launch_flash(const __half *q16, const __half *k16, const __half *v16, __half *out,
                  const uint32_t *lo, const uint32_t *hi, int n, int kv, int window,
                  cudaStream_t st, int heads = 4) {
    constexpr int QR = W * 16;
    flash_mma_kernel<D, DO, W, KT, QS><<<dim3((n + QR - 1) / QR, heads, D / DO), W * 32,
                                         flash_smem<D, DO, W, KT, QS>(), st>>>(
        q16, k16, v16, out, lo, hi, n, heads, kv, window);
}
// 8 warps (128 queries per K/V tile) x 32 keys, Q fragments from global/L1;
// width 512 splits the output into two 256-wide halves. The 2026-10-09 sweep
// (perf/optimization_status.md): 512x16 32.9 -> 24.7 ms and 8192x1 153 -> 59.5
// ms over 4 warps with staged Q (width 256) / 2 warps (width 512).
#define G2_FLASH_256 256, 256, 8, 32, false
#define G2_FLASH_512 512, 256, 8, 32, false
#define G2_FLASH_64 64, 64, 8, 32, false
constexpr size_t kFlashSmemMax = flash_smem<G2_FLASH_512>() > flash_smem<G2_FLASH_256>()
                                     ? flash_smem<G2_FLASH_512>()
                                     : flash_smem<G2_FLASH_256>();
// Sandwich over rows of width W (block W / 4 threads): x = (x + hs * rms(h) *
// w_post) * scale; optional next16 = rms(x) * w_next and raw16 = x (fp16
// copies feeding the next GEMMs). w_post NULL: x = (x + hs * h) * scale.
__global__ void post_add_kernel(const float *h, float *x, const float *wp, float scale,
                                const float *wn, __half *next16, __half *raw16, int width,
                                float hs) {
    size_t row = (size_t)blockIdx.x * width;
    int z = threadIdx.x * 4;
    float a[4], ss = 0;
    for (int j = 0; j < 4; j++) {
        a[j] = h[row + z + j];
        ss += a[j] * a[j];
    }
    float inv = wp ? rsqrtf(block_sum(ss) / width + 1e-6f) : 1.0f, ss2 = 0;
    for (int j = 0; j < 4; j++) {
        a[j] = (x[row + z + j] + hs * a[j] * inv * (wp ? wp[z + j] : 1.0f)) * scale;
        x[row + z + j] = a[j];
        ss2 += a[j] * a[j];
        if (raw16)
            raw16[row + z + j] = __float2half(a[j]);
    }
    if (next16) {
        float inv2 = rsqrtf(block_sum(ss2) / width + 1e-6f);
        for (int j = 0; j < 4; j++)
            next16[row + z + j] = __float2half(a[j] * inv2 * wn[z + j]);
    }
}
// g16 = gelu(gate) * up from the concatenated [up | gate] GEMM output.
__global__ void geglu_kernel(const float *ug, __half *g16, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n * kFfn)
        return;
    size_t t = i / kFfn, c = i % kFfn;
    const float *r = ug + t * 2 * kFfn;
    g16[i] = __float2half(gelu_tanh(r[kFfn + c]) * r[c]);
}
// pg16 = gelu(inp_gate(x)) * ple[t, layer].
__global__ void ple_gate_kernel(const float *ig, const float *ple, int layer, __half *pg16,
                                size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n * kModel)
        return;
    size_t t = i / kModel, c = i % kModel;
    pg16[i] = __float2half(gelu_tanh(ig[i]) * ple[(t * kLayers + layer) * kModel + c]);
}
// Mean pool of the projected rows per sequence.
__global__ void pool_kernel(const float *y, const uint32_t *offsets, float *pooled) {
    int s = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= kOut)
        return;
    uint32_t a = offsets[s], b = offsets[s + 1];
    float sum = 0;
    for (uint32_t t = a; t < b; t++)
        sum += y[(size_t)t * kOut + c];
    pooled[(size_t)s * kOut + c] = sum / (float)(b - a);
}

// ---- host forward ----------------------------------------------------------

bool gemm(engine *e, const __half *w, const __half *x, float *y, int m, int k, int n, int ldy,
          char *err, size_t len) {
    const float one = 1.0f, zero = 0.0f;
    return check_blas(cublasGemmEx(e->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, w,
                                   CUDA_R_16F, k, x, CUDA_R_16F, k, &zero, y, CUDA_R_32F, ldy,
                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                      "GEMM", err, len);
}

template <typename T>
bool grow(engine *e, T **p, size_t count, char *err, size_t len) {
    cudaFree(*p);
    *p = nullptr;
    return check(cudaMalloc((void **)p, count * sizeof(T) + 256), "allocate workspace", err, len);
}

bool reserve(engine *e, size_t tokens, size_t batch, char *err, size_t len) {
    if (tokens <= e->cap && batch <= e->cap_batch)
        return true;
    size_t cap = e->cap ? e->cap : 64, cb = e->cap_batch ? e->cap_batch : 16;
    while (cap < tokens)
        cap *= 2;
    while (cb < batch)
        cb *= 2;
    if (cap > 8192)
        cap = 8192;
    for (graph_entry &g : e->graphs)
        cudaGraphExecDestroy(g.exec);
    e->graphs.clear();
    bool ok = grow(e, &e->ids, cap, err, len) && grow(e, &e->lo, cap, err, len) &&
              grow(e, &e->hi, cap, err, len) && grow(e, &e->pos, cap, err, len) &&
              grow(e, &e->offsets, cb + 1, err, len) &&
              grow(e, &e->x, cap * kModel, err, len) && grow(e, &e->ple, cap * kPle, err, len) &&
              grow(e, &e->proj, cap * kModel, err, len) &&
              grow(e, &e->wide, cap * 2 * kFfn, err, len) &&
              grow(e, &e->pooled, cb * kOut, err, len) &&
              grow(e, &e->inputs, cap * kModel, err, len) &&
              grow(e, &e->h16, cap * kModel, err, len) && grow(e, &e->q16, cap * 2048, err, len) &&
              grow(e, &e->k16, cap * 512, err, len) && grow(e, &e->v16, cap * 512, err, len) &&
              grow(e, &e->a16, cap * 2048, err, len) && grow(e, &e->g16, cap * kFfn, err, len) &&
              grow(e, &e->x16, cap * kModel, err, len);
    e->cap = cap, e->cap_batch = cb;
    return ok;
}

// Enqueue one forward for n tokens / batch sequences (inputs already uploaded).
// Attention for head widths 64/256/512: tensor-core flash where the device
// supports it (sm_80+, enough opt-in shared memory), the CUDA-core kernel
// otherwise. lo/hi NULL: one sequence of n tokens.
void attend(engine *e, int d, const __half *q, const __half *k, const __half *v, __half *out,
            const uint32_t *lo, const uint32_t *hi, int n, int heads, int kv, int window) {
    cudaStream_t st = e->stream;
    if (e->mma_attention) {
        if (d == 64)
            launch_flash<G2_FLASH_64>(q, k, v, out, lo, hi, n, kv, window, st, heads);
        else if (d == 256)
            launch_flash<G2_FLASH_256>(q, k, v, out, lo, hi, n, kv, window, st, heads);
        else
            launch_flash<G2_FLASH_512>(q, k, v, out, lo, hi, n, kv, window, st, heads);
        return;
    }
    dim3 grid((n + 7) / 8, heads);
    if (d == 64)
        attention_kernel<64><<<grid, 256, 0, st>>>(q, k, v, out, lo, hi, n, heads, kv, window);
    else if (d == 256)
        attention_kernel<256><<<grid, 256, 0, st>>>(q, k, v, out, lo, hi, n, heads, kv, window);
    else
        attention_kernel<512><<<grid, 256, 0, st>>>(q, k, v, out, lo, hi, n, heads, kv, window);
}

bool forward(engine *e, int n, int batch, bool from_inputs, char *err, size_t len) {
    cudaStream_t st = e->stream;
    if (from_inputs) {
        if (!check(cudaMemcpyAsync(e->x, e->inputs, (size_t)n * kModel * 4,
                                   cudaMemcpyDeviceToDevice, st),
                   "copy inputs", err, len))
            return false;
        to_half_kernel<<<(n * kModel + 255) / 256, 256, 0, st>>>(e->x, e->x16, (size_t)n * kModel);
    } else
        embed_kernel<<<n, 128, 0, st>>>(e->token_embd, e->ids, e->x, e->x16);
    if (!gemm(e, e->ple_proj, e->x16, e->ple, kPle, kModel, n, kPle, err, len))
        return false;
    ple_norm_kernel<<<n * kLayers, 128, 0, st>>>(e->ple, e->ple_norm);
    rms_half_kernel<<<n, 128, 0, st>>>(e->x, e->layers[0].attn_norm, e->h16, kModel);
    for (int l = 0; l < kLayers; l++) {
        const layer_weights &w = e->layers[l];
        int d = l % 6 == 5 ? 512 : 256, kv = 512 / d, rows = 4 * d + 1024;
        if (!gemm(e, w.qkv, e->h16, e->wide, rows, kModel, n, rows, err, len))
            return false;
        qkv_post_kernel<<<dim3(n, 4 + 2 * kv), d / 2, 0, st>>>(
            e->wide, d, kv, w.q_norm, w.k_norm, e->rope[l % 6 == 5], e->pos, e->q16, e->k16,
            e->v16);
        attend(e, d, e->q16, e->k16, e->v16, e->a16, e->lo, e->hi, n, 4, kv, d == 512 ? 0 : 512);
        if (!gemm(e, w.o, e->a16, e->proj, kModel, 4 * d, n, kModel, err, len))
            return false;
        post_add_kernel<<<n, 128, 0, st>>>(e->proj, e->x, w.post_attn, 1.0f, w.ffn_norm, e->h16,
                                           nullptr, kModel, 1.0f);
        if (!gemm(e, w.upgate, e->h16, e->wide, 2 * kFfn, kModel, n, 2 * kFfn, err, len))
            return false;
        geglu_kernel<<<(n * kFfn + 255) / 256, 256, 0, st>>>(e->wide, e->g16, n);
        if (!gemm(e, w.down, e->g16, e->proj, kModel, kFfn, n, kModel, err, len))
            return false;
        post_add_kernel<<<n, 128, 0, st>>>(e->proj, e->x, w.post_ffw, 1.0f, nullptr, nullptr,
                                           e->x16, kModel, 1.0f);
        if (!gemm(e, w.inp_gate, e->x16, e->proj, kModel, kModel, n, kModel, err, len))
            return false;
        ple_gate_kernel<<<(n * kModel + 255) / 256, 256, 0, st>>>(e->proj, e->ple, l, e->h16, n);
        if (!gemm(e, w.proj, e->h16, e->wide, kModel, kModel, n, kModel, err, len))
            return false;
        const float *next = l + 1 < kLayers ? e->layers[l + 1].attn_norm : e->output_norm;
        post_add_kernel<<<n, 128, 0, st>>>(e->wide, e->x, w.post_norm, w.output_scale, next,
                                           e->h16, nullptr, kModel, 1.0f);
    }
    if (!gemm(e, e->output, e->h16, e->wide, kOut, kModel, n, kOut, err, len))
        return false;
    pool_kernel<<<dim3(batch, (kOut + 255) / 256), 256, 0, st>>>(e->wide, e->offsets, e->pooled);
    return check(cudaGetLastError(), "launch", err, len);
}

bool run(engine *e, const int32_t *ids, const float *inputs, const size_t *offsets, size_t batch,
         float *out, char *err, size_t len) {
    if (!check(cudaSetDevice(e->device), "select device", err, len))
        return false;
    size_t n = offsets[batch];
    if (!n || n > 8192) {
        snprintf(err, len, "EmbeddingGemma 2 accepts 1..8192 tokens per wave");
        return false;
    }
    if (!reserve(e, n, batch, err, len))
        return false;
    std::vector<uint32_t> lo(n), hi(n), pos(n), off(batch + 1);
    for (size_t s = 0; s < batch; s++) {
        off[s] = (uint32_t)offsets[s];
        for (size_t t = offsets[s]; t < offsets[s + 1]; t++)
            lo[t] = (uint32_t)offsets[s], hi[t] = (uint32_t)offsets[s + 1],
            pos[t] = (uint32_t)(t - offsets[s]);
    }
    off[batch] = (uint32_t)n;
    cudaStream_t st = e->stream;
    bool ok = check(cudaMemcpyAsync(e->lo, lo.data(), n * 4, cudaMemcpyHostToDevice, st), "upload",
                    err, len) &&
              check(cudaMemcpyAsync(e->hi, hi.data(), n * 4, cudaMemcpyHostToDevice, st), "upload",
                    err, len) &&
              check(cudaMemcpyAsync(e->pos, pos.data(), n * 4, cudaMemcpyHostToDevice, st),
                    "upload", err, len) &&
              check(cudaMemcpyAsync(e->offsets, off.data(), (batch + 1) * 4,
                                    cudaMemcpyHostToDevice, st),
                    "upload", err, len);
    if (ok && ids)
        ok = check(cudaMemcpyAsync(e->ids, ids, n * 4, cudaMemcpyHostToDevice, st), "upload", err,
                   len);
    if (ok && inputs)
        ok = check(cudaMemcpyAsync(e->inputs, inputs, n * kModel * 4, cudaMemcpyHostToDevice, st),
                   "upload", err, len);
    if (!ok)
        return false;
    // Graphs depend only on (n, batch, ids-or-inputs): bounds live in device buffers.
    uint32_t key_batch = (uint32_t)batch | (inputs ? 0x80000000u : 0u);
    graph_entry *entry = nullptr;
    if (e->use_graphs)
        for (graph_entry &g : e->graphs)
            if (g.tokens == n && g.batch == key_batch)
                entry = &g;
    if (entry) {
        ok = check(cudaGraphLaunch(entry->exec, st), "launch graph", err, len);
    } else if (e->use_graphs) {
        if (e->graphs.size() >= 64) {
            cudaGraphExecDestroy(e->graphs.front().exec);
            e->graphs.erase(e->graphs.begin());
        }
        cudaGraph_t graph = nullptr;
        ok = check(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "begin capture",
                   err, len);
        bool fw = ok && forward(e, (int)n, (int)batch, inputs != nullptr, err, len);
        cudaError_t end = cudaStreamEndCapture(st, &graph);
        ok = fw && check(end, "end capture", err, len);
        cudaGraphExec_t exec = nullptr;
        if (ok)
            ok = check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate",
                       err, len);
        if (graph)
            cudaGraphDestroy(graph);
        if (ok) {
            e->graphs.push_back({(uint32_t)n, key_batch, exec});
            ok = check(cudaGraphLaunch(exec, st), "launch graph", err, len);
        }
    } else
        ok = forward(e, (int)n, (int)batch, inputs != nullptr, err, len);
    ok = ok && check(cudaMemcpyAsync(out, e->pooled, batch * kOut * 4, cudaMemcpyDeviceToHost, st),
                     "download", err, len) &&
         check(cudaStreamSynchronize(st), "synchronize", err, len);
    if (!ok)
        return false;
    for (size_t s = 0; s < batch; s++)
        ei_l2_normalize(out + s * kOut, kOut);
    return true;
}


// ---- media towers ---------------------------------------------------------

// fp16 = clamp(act(x), lo, hi); act 1 = SiLU.
__global__ void to_half_clamp_kernel(const float *x, __half *y, size_t n, float lo, float hi,
                                     int act) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float v = x[i];
    if (act == 1)
        v = v / (1.0f + __expf(-v));
    y[i] = __float2half(fminf(fmaxf(v, lo), hi));
}
__global__ void clamp_kernel(float *x, size_t n, float lo, float hi) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = fminf(fmaxf(x[i], lo), hi);
}
// 16x16 RGB patches scaled to [-1, 1], rounded through fp16 like the reference.
__global__ void patch_kernel(const uint8_t *rgb, __half *patch, int n, int cols) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)n * 768)
        return;
    int t = i / 768, z = i % 768, c = z / 256, y = (z % 256) / 16, x = z % 16;
    float v = 2.0f * (float)rgb[((t / cols * 16 + y) * cols * 16 + t % cols * 16 + x) * 3 + c] /
                  255.0f - 1.0f;
    patch[i] = __float2half(v);
}
__global__ void position_kernel(float *x, const float *pos, int n, int cols) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)n * 768)
        return;
    int t = i / 768, z = i % 768;
    x[i] += pos[(size_t)(t % cols) * 768 + z] + pos[(size_t)(10240 + t / cols) * 768 + z];
}
// Vision q/k/v (12 heads x 64): weighted RMS for q/k, plain for v, axial RoPE
// (base 100) on q/k: dims 0..31 by column (pairs z, z + 16), 32..63 by row.
// Block 64 threads = one (token, slot); slots [0,12) q, [12,24) k, [24,36) v.
__global__ void vision_qkv_kernel(const float *qkv, const float *qn, const float *kn, int cols,
                                  __half *q16, __half *k16, __half *v16) {
    int t = blockIdx.x, slot = blockIdx.y, i = threadIdx.x;
    int part = slot / 12, head = slot % 12;
    float v = qkv[(size_t)t * 2304 + part * 768 + head * 64 + i];
    __shared__ float row[64];
    float inv = rsqrtf(block_sum(v * v) / 64 + 1e-6f);
    v *= inv;
    __half *dst = part == 0 ? q16 : part == 1 ? k16 : v16;
    if (part < 2) {
        v *= (part == 0 ? qn : kn)[i];
        row[i] = v;
        __syncthreads();
        int axis = i / 32, zz = i % 16, base = axis * 32;
        bool first = (i % 32) < 16;
        float theta = (float)(axis ? t / cols : t % cols) * powf(100.0f, -(float)zz / 16.0f);
        float co = cosf(theta), si = sinf(theta);
        float a = row[base + zz], b = row[base + zz + 16];
        v = first ? a * co - b * si : a * si + b * co;
    }
    dst[((size_t)t * 12 + head) * 64 + i] = __float2half(v);
}
// gelu(gate) * up (or QuickGELU) from the concatenated [up | gate] output.
__global__ void glu_kernel(const float *ug, __half *out, size_t n, int width, int quick) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n * width)
        return;
    size_t t = i / width, c = i % width;
    const float *r = ug + t * 2 * width;
    float g = r[width + c];
    g = quick ? g / (1.0f + __expf(-1.702f * g)) : gelu_tanh(g);
    out[i] = __float2half(g * r[c]);
}
// 3x3 mean pool over the patch grid, times sqrt(768).
__global__ void spatial_pool_kernel(const float *x, float *y, int nt, int cols) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)nt * 768)
        return;
    int t = i / 768, z = i % 768, c3 = cols / 3;
    float sum = 0;
    for (int yy = 0; yy < 3; yy++)
        for (int xx = 0; xx < 3; xx++)
            sum += x[((size_t)(t / c3 * 3 + yy) * cols + t % c3 * 3 + xx) * 768 + z] / 9.0f *
                   sqrtf(768.0f);
    y[i] = sum;
}
// Stride-2 3x3 conv, input [t][f][c], weight [o][c][3][3], output [t][f][o].
__global__ void conv_kernel(const float *x, const float *w, float *y, int nt, int nf, int nc,
                            int oc) {
    int of = (nf + 1) / 2, ot = (nt + 1) / 2;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)ot * of * oc)
        return;
    int o = i % oc, f = (i / oc) % of, t = i / ((size_t)oc * of);
    float sum = 0;
    for (int ky = 0; ky < 3; ky++)
        for (int kx = 0; kx < 3; kx++) {
            int ti = t * 2 + ky - 1, fi = f * 2 + kx - 1;
            if (ti < 0 || ti >= nt || fi < 0 || fi >= nf)
                continue;
            for (int c = 0; c < nc; c++)
                sum += x[((size_t)ti * nf + fi) * nc + c] * w[((size_t)o * nc + c) * 9 + ky * 3 + kx];
        }
    y[i] = sum;
}
// Mean-subtracted LayerNorm (weight, no bias) then ReLU, rows of width W <= 128.
__global__ void layer_norm_relu_kernel(float *x, const float *w, int width) {
    float *r = x + (size_t)blockIdx.x * width;
    int z = threadIdx.x;
    float v = z < width ? r[z] : 0.0f;
    float mean = block_sum(v) / width, d = z < width ? v - mean : 0.0f;
    float inv = rsqrtf(block_sum(d * d) / width + 1e-6f);
    if (z < width)
        r[z] = fmaxf(d * inv * w[z], 0.0f);
}
// rms over rows of width W into fp32 (in place allowed), optional weight.
__global__ void rms_kernel(const float *x, const float *w, float *out, int width) {
    const float *r = x + (size_t)blockIdx.x * width;
    float ss = 0;
    for (int z = threadIdx.x; z < width; z += blockDim.x)
        ss += r[z] * r[z];
    float inv = rsqrtf(block_sum(ss) / width + 1e-6f);
    for (int z = threadIdx.x; z < width; z += blockDim.x)
        out[(size_t)blockIdx.x * width + z] = r[z] * inv * (w ? w[z] : 1.0f);
}
__global__ void audio_scale_kernel(float *q, float *k, const float *w, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n * 1024) {
        q[i] *= w[i % 128] / (sqrtf(128.0f) * logf(2.0f));
        k[i] *= logf(1.0f + expf(1.0f)) / logf(2.0f);
    }
}
// Conformer relative attention: 8 heads x 128, causal window of 12 keys,
// score = q . (k + rpe[12 - t + s]), soft-capped 50 tanh(s / 50). Warp per
// (token, head); fp32.
__global__ void audio_attention_kernel(const float *q, const float *k, const float *v,
                                       const float *rpe, float *out, int n) {
    int t = blockIdx.x, h = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float qv[4], acc[4] = {0, 0, 0, 0};
    for (int j = 0; j < 4; j++)
        qv[j] = q[((size_t)t * 8 + h) * 128 + lane + 32 * j];
    int lo = t > 11 ? t - 11 : 0;
    float mx = -INFINITY, total = 0;
    for (int s = lo; s <= t; s++) {
        float dot = 0;
        for (int j = 0; j < 4; j++) {
            int z = lane + 32 * j;
            dot += qv[j] * (k[((size_t)s * 8 + h) * 128 + z] + rpe[((12 - t + s) * 8 + h) * 128 + z]);
        }
        dot = warp_sum(dot);
        dot = 50.0f * tanhf(dot / 50.0f);
        float next = fmaxf(mx, dot), a = __expf(mx - next), b = __expf(dot - next);
        total = total * a + b;
        for (int j = 0; j < 4; j++)
            acc[j] = acc[j] * a + b * v[((size_t)s * 8 + h) * 128 + lane + 32 * j];
        mx = next;
    }
    for (int j = 0; j < 4; j++)
        out[((size_t)t * 8 + h) * 128 + lane + 32 * j] = acc[j] / total;
}
// GLU: out = a * sigmoid(b) from [a | b] rows of 2 x 1024.
__global__ void conv_glu_kernel(const float *ab, float *out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n * 1024)
        return;
    size_t t = i / 1024, c = i % 1024;
    out[i] = ab[t * 2048 + c] / (1.0f + __expf(-ab[t * 2048 + 1024 + c]));
}
// Causal depthwise conv, kernel 5, weight [1024][5].
__global__ void depthwise_kernel(const float *x, const float *w, float *y, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n * 1024)
        return;
    size_t t = i / 1024, z = i % 1024;
    float sum = 0;
    for (int j = 0; j < 5; j++)
        if (t + j >= 4)
            sum += x[(t + j - 4) * 1024 + z] * w[z * 5 + j];
    y[i] = sum;
}
__global__ void silu_kernel(float *x, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = x[i] / (1.0f + __expf(-x[i]));
}
__global__ void bias_kernel(float *x, const float *b, size_t n, int width) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n * width)
        x[i] += b[i % width];
}

unsigned blocks(size_t n, int per = 256) { return (unsigned)((n + per - 1) / per); }

void *scratch(engine *e, const char *name, size_t bytes) {
    auto &slot = e->scratch[name];
    if (slot.second < bytes) {
        cudaFree(slot.first);
        slot.first = nullptr;
        cudaMalloc(&slot.first, bytes + 256);
        slot.second = slot.first ? bytes : 0;
    }
    return slot.first;
}
// Projector weights: fp16 GEMM copies and fp32 vectors, converted on first use.
__half *proj_half(engine *e, const std::vector<std::string> &names, char *err, size_t len) {
    std::string key = names[0];
    for (size_t i = 1; i < names.size(); i++)
        key += "+" + names[i];
    auto it = e->half_w.find(key);
    if (it != e->half_w.end())
        return it->second;
    std::vector<const ei_tensor *> parts;
    for (const std::string &n : names)
        parts.push_back(ei_gguf_tensor(e->projector, n.c_str(), true));
    __half *w = nullptr;
    if (!upload_half(e, &w, parts, err, len))
        return nullptr;
    e->half_w[key] = w;
    return w;
}
float *proj_float(engine *e, const char *name, char *err, size_t len) {
    auto it = e->float_w.find(name);
    if (it != e->float_w.end())
        return it->second;
    const ei_tensor *t = ei_gguf_tensor(e->projector, name, true);
    size_t n = 1;
    for (uint32_t d = 0; d < t->n_dims; d++)
        n *= (size_t)t->ne[d];
    std::vector<float> host(n);
    for (size_t i = 0; i < n; i++)
        host[i] = host_element(t, i);
    float *w = nullptr;
    if (!upload(e, &w, host.data(), n * sizeof(float), err, len))
        return nullptr;
    e->float_w[name] = w;
    return w;
}
float proj_scalar(engine *e, const char *name, float fallback) {
    const ei_tensor *t = ei_gguf_tensor(e->projector, name, false);
    return t ? host_element(t, 0) : fallback;
}
// y[n][m] = clamp_out(W . clamp_in(x)) for a projector weight (Gemma 3n clipped linear).
bool clipped(engine *e, const char *name, const float *x, int n, int k, int m, float *y, int act,
             char *err, size_t len) {
    std::string stem(name, strlen(name) - 7); // strip ".weight"
    float ilo = proj_scalar(e, (stem + ".input_min").c_str(), -INFINITY);
    float ihi = proj_scalar(e, (stem + ".input_max").c_str(), INFINITY);
    float olo = proj_scalar(e, (stem + ".output_min").c_str(), -INFINITY);
    float ohi = proj_scalar(e, (stem + ".output_max").c_str(), INFINITY);
    __half *w = proj_half(e, {name}, err, len);
    __half *x16 = (__half *)scratch(e, "clip_in", (size_t)n * k * 2);
    if (!w || !x16)
        return false;
    to_half_clamp_kernel<<<blocks((size_t)n * k), 256, 0, e->stream>>>(x, x16, (size_t)n * k, ilo,
                                                                        ihi, act);
    if (!gemm(e, w, x16, y, m, k, n, m, err, len))
        return false;
    if (olo > -INFINITY || ohi < INFINITY)
        clamp_kernel<<<blocks((size_t)n * m), 256, 0, e->stream>>>(y, (size_t)n * m, olo, ohi);
    return true;
}
bool finish_media(engine *e, const float *result, size_t count, float **out, char *err,
                  size_t len) {
    *out = (float *)malloc(count * sizeof(float));
    if (!*out) {
        snprintf(err, len, "out of memory");
        return false;
    }
    return check(cudaMemcpyAsync(*out, result, count * 4, cudaMemcpyDeviceToHost, e->stream),
                 "download", err, len) &&
           check(cudaStreamSynchronize(e->stream), "synchronize", err, len) &&
           check(cudaGetLastError(), "media kernels", err, len);
}

bool vision(engine *e, const uint8_t *rgb, int width, int height, float **out, size_t *tokens,
            char *err, size_t len) {
    if (width < 48 || height < 48 || width % 48 || height % 48 ||
        (size_t)width * height > 1120 * 48 * 48) {
        snprintf(err, len, "vision input must be aligned to 48 pixels and at most 1120 soft tokens");
        return false;
    }
    cudaStream_t st = e->stream;
    int cols = width / 16, n = cols * (height / 16), nt = n / 9;
    size_t count = (size_t)n * 768;
    uint8_t *pix = (uint8_t *)scratch(e, "pixels", (size_t)width * height * 3);
    float *x = (float *)scratch(e, "vx", count * 4), *wide = (float *)scratch(e, "vwide", count * 8 * 4);
    __half *h16 = (__half *)scratch(e, "vh16", count * 2), *q16 = (__half *)scratch(e, "vq16", count * 2),
           *k16 = (__half *)scratch(e, "vk16", count * 2), *v16 = (__half *)scratch(e, "vv16", count * 2),
           *a16 = (__half *)scratch(e, "va16", count * 2), *g16 = (__half *)scratch(e, "vg16", count * 4 * 2);
    float *pooled = (float *)scratch(e, "vpool", (size_t)nt * 768 * 4),
          *result = (float *)scratch(e, "vresult", (size_t)nt * 512 * 4);
    if (!pix || !x || !wide || !h16 || !q16 || !k16 || !v16 || !a16 || !g16 || !pooled || !result) {
        snprintf(err, len, "EmbeddingGemma 2 CUDA vision workspace allocation failed");
        return false;
    }
    if (!check(cudaMemcpyAsync(pix, rgb, (size_t)width * height * 3, cudaMemcpyHostToDevice, st),
               "upload pixels", err, len))
        return false;
    // Patch embedding: the GGUF stores it as [16][16][3][768]; it is used as 768 x 768.
    auto it = e->half_w.find("v.patch_embd");
    __half *pw = it != e->half_w.end() ? it->second : nullptr;
    if (!pw) {
        const ei_tensor *t = ei_gguf_tensor(e->projector, "v.patch_embd.weight", true);
        std::vector<__half> host(768 * 768);
        for (size_t i = 0; i < host.size(); i++)
            host[i] = __float2half(host_element(t, i));
        if (!upload(e, &pw, host.data(), host.size() * 2, err, len))
            return false;
        e->half_w["v.patch_embd"] = pw;
    }
    patch_kernel<<<blocks(count), 256, 0, st>>>(pix, h16, n, cols);
    float *pos = proj_float(e, "v.position_embd.weight", err, len);
    if (!pos || !gemm(e, pw, h16, x, 768, 768, n, 768, err, len))
        return false;
    position_kernel<<<blocks(count), 256, 0, st>>>(x, pos, n, cols);
    bool quick = !ei_gguf_kv_bool(e->projector, "clip.use_gelu", false);
    char a[128], b[128], c[128];
#define VN(s, l) (snprintf(a, sizeof a, "v.blk.%d." s ".weight", l), a)
    float *ln1 = proj_float(e, VN("ln1", 0), err, len);
    if (!ln1)
        return false;
    rms_half_kernel<<<n, 192, 0, st>>>(x, ln1, h16, 768);
    for (int l = 0; l < 16; l++) {
        snprintf(a, sizeof a, "v.blk.%d.attn_q.weight", l);
        snprintf(b, sizeof b, "v.blk.%d.attn_k.weight", l);
        snprintf(c, sizeof c, "v.blk.%d.attn_v.weight", l);
        __half *wqkv = proj_half(e, {a, b, c}, err, len);
        float *qn = proj_float(e, VN("attn_q_norm", l), err, len);
        float *kn = proj_float(e, VN("attn_k_norm", l), err, len);
        if (!wqkv || !qn || !kn || !gemm(e, wqkv, h16, wide, 2304, 768, n, 2304, err, len))
            return false;
        vision_qkv_kernel<<<dim3(n, 36), 64, 0, st>>>(wide, qn, kn, cols, q16, k16, v16);
        attend(e, 64, q16, k16, v16, a16, nullptr, nullptr, n, 12, 12, 0);
        __half *wo = proj_half(e, {VN("attn_out", l)}, err, len);
        float *pa = proj_float(e, VN("attn_post_norm", l), err, len);
        float *ln2 = proj_float(e, VN("ln2", l), err, len);
        if (!wo || !pa || !ln2 || !gemm(e, wo, a16, wide, 768, 768, n, 768, err, len))
            return false;
        post_add_kernel<<<n, 192, 0, st>>>(wide, x, pa, 1.0f, ln2, h16, nullptr, 768, 1.0f);
        snprintf(a, sizeof a, "v.blk.%d.ffn_up.weight", l);
        snprintf(b, sizeof b, "v.blk.%d.ffn_gate.weight", l);
        __half *wug = proj_half(e, {a, b}, err, len);
        if (!wug || !gemm(e, wug, h16, wide, 6144, 768, n, 6144, err, len))
            return false;
        glu_kernel<<<blocks(count * 4), 256, 0, st>>>(wide, g16, n, 3072, quick);
        __half *wd = proj_half(e, {VN("ffn_down", l)}, err, len);
        float *pf = proj_float(e, VN("ffn_post_norm", l), err, len);
        float *next = l < 15 ? proj_float(e, (snprintf(b, sizeof b, "v.blk.%d.ln1.weight", l + 1), b),
                                          err, len)
                             : nullptr;
        if (!wd || !pf || (l < 15 && !next) || !gemm(e, wd, g16, wide, 768, 3072, n, 768, err, len))
            return false;
        post_add_kernel<<<n, 192, 0, st>>>(wide, x, pf, 1.0f, next, next ? h16 : nullptr, nullptr,
                                           768, 1.0f);
    }
#undef VN
    spatial_pool_kernel<<<blocks((size_t)nt * 768), 256, 0, st>>>(x, pooled, nt, cols);
    rms_half_kernel<<<nt, 192, 0, st>>>(pooled, nullptr, h16, 768);
    __half *wp = proj_half(e, {"mm.input_projection.weight"}, err, len);
    if (!wp || !gemm(e, wp, h16, result, 512, 768, nt, 512, err, len))
        return false;
    *tokens = nt;
    return finish_media(e, result, (size_t)nt * 512, out, err, len);
}

bool audio(engine *e, const float *mel, int frames, float **out, size_t *tokens, char *err,
           size_t len) {
    if (frames < 1 || frames > 3000) {
        snprintf(err, len, "audio chunk must contain 1..3000 mel frames");
        return false;
    }
    cudaStream_t st = e->stream;
    int n = (frames + 3) / 4;
    size_t count = (size_t)n * 1024;
    float *cur = (float *)scratch(e, "mel", (size_t)frames * 128 * 4);
    float *c0 = (float *)scratch(e, "conv0", (size_t)((frames + 1) / 2) * 64 * 128 * 4);
    float *x = (float *)scratch(e, "ax", count * 4), *h = (float *)scratch(e, "ah", count * 4),
          *q = (float *)scratch(e, "aq", count * 4), *k = (float *)scratch(e, "ak", count * 4),
          *v = (float *)scratch(e, "av", count * 4), *att = (float *)scratch(e, "aa", count * 4),
          *up = (float *)scratch(e, "aup", count * 4 * 4),
          *result = (float *)scratch(e, "aresult", (size_t)n * 512 * 4);
    __half *x16 = (__half *)scratch(e, "ax16", count * 2);
    if (!cur || !c0 || !x || !h || !q || !k || !v || !att || !up || !result || !x16) {
        snprintf(err, len, "EmbeddingGemma 2 CUDA audio workspace allocation failed");
        return false;
    }
    if (!check(cudaMemcpyAsync(cur, mel, (size_t)frames * 128 * 4, cudaMemcpyHostToDevice, st),
               "upload mel", err, len))
        return false;
    // Two stride-2 convs, each followed by LayerNorm + ReLU: 128 mel bins -> 32 x 32.
    int nt = frames, nf = 128, nc = 1;
    float *src = cur;
    for (int l = 0; l < 2; l++) {
        char name[64];
        int ot = (nt + 1) / 2, of = (nf + 1) / 2, oc = l ? 32 : 128;
        float *dst = l ? h : c0;
        snprintf(name, sizeof name, "a.conv1d.%d.weight", l);
        float *w = proj_float(e, name, err, len);
        snprintf(name, sizeof name, "a.conv1d.%d.norm.weight", l);
        float *nw = proj_float(e, name, err, len);
        if (!w || !nw)
            return false;
        conv_kernel<<<blocks((size_t)ot * of * oc), 256, 0, st>>>(src, w, dst, nt, nf, nc, oc);
        layer_norm_relu_kernel<<<ot * of, 128, 0, st>>>(dst, nw, oc);
        src = dst, nt = ot, nf = of, nc = oc;
    }
    __half *win = proj_half(e, {"a.input_projection.weight"}, err, len);
    if (!win)
        return false;
    to_half_kernel<<<blocks(count), 256, 0, st>>>(h, x16, count);
    if (!gemm(e, win, x16, x, 1024, 1024, n, 1024, err, len))
        return false;
    char nm[128];
#define AN(s) (snprintf(nm, sizeof nm, "a.blk.%d." s ".weight", l), nm)
    for (int l = 0; l < 12; l++) {
        auto ANS = [&](const char *stem) {
            snprintf(nm, sizeof nm, "a.blk.%d.%s.weight", l, stem);
            return nm;
        };
        if (!e->rpe[l]) {
            // Relative position embeddings are a constant per layer: sinusoid table
            // (13 x 1024) through attn_k_rel, computed once.
            std::vector<__half> pos(13 * 1024);
            for (int p = 0; p < 13; p++)
                for (int z = 0; z < 512; z++) {
                    float theta = (12 - p) * expf(-z * logf(10000.0f) / 511.0f);
                    pos[p * 1024 + z] = __float2half(sinf(theta));
                    pos[p * 1024 + z + 512] = __float2half(cosf(theta));
                }
            __half *pos16 = (__half *)scratch(e, "rpe_pos", 13 * 1024 * 2);
            __half *wk = proj_half(e, {AN("attn_k_rel")}, err, len);
            if (!wk || !check(cudaMalloc((void **)&e->rpe[l], 13 * 1024 * 4), "allocate rpe", err, len) ||
                !check(cudaMemcpyAsync(pos16, pos.data(), 13 * 1024 * 2, cudaMemcpyHostToDevice,
                                       e->stream),
                       "upload positions", err, len) ||
                !check(cudaStreamSynchronize(e->stream), "upload positions", err, len) ||
                !gemm(e, wk, pos16, e->rpe[l], 1024, 1024, 13, 1024, err, len))
                return false;
        }
        const char *ffn[2][4] = {{"ffn_norm", "ffn_up", "ffn_down", "ffn_post_norm"},
                                 {"ffn_norm_1", "ffn_up_1", "ffn_down_1", "ffn_post_norm_1"}};
        for (int half = 0; half < 2; half++) {
            if (half == 1) {
                // Attention and the convolution module sit between the two FFNs.
                float *pre = proj_float(e, AN("attn_pre_norm"), err, len);
                if (!pre)
                    return false;
                rms_kernel<<<n, 256, 0, st>>>(x, pre, h, 1024);
                if (!clipped(e, AN("attn_q"), h, n, 1024, 1024, q, 0, err, len) ||
                    !clipped(e, AN("attn_k"), h, n, 1024, 1024, k, 0, err, len) ||
                    !clipped(e, AN("attn_v"), h, n, 1024, 1024, v, 0, err, len))
                    return false;
                float *pds = proj_float(e, AN("per_dim_scale"), err, len);
                if (!pds)
                    return false;
                audio_scale_kernel<<<blocks(count), 256, 0, st>>>(q, k, pds, n);
                audio_attention_kernel<<<n, 256, 0, st>>>(q, k, v, e->rpe[l], att, n);
                float *pan = proj_float(e, AN("attn_post_norm"), err, len);
                if (!pan || !clipped(e, AN("attn_out"), att, n, 1024, 1024, h, 0, err, len))
                    return false;
                post_add_kernel<<<n, 256, 0, st>>>(h, x, pan, 1.0f, nullptr, nullptr, nullptr, 1024,
                                                   1.0f);
                float *cn = proj_float(e, AN("conv_norm"), err, len);
                if (!cn)
                    return false;
                rms_kernel<<<n, 256, 0, st>>>(x, cn, h, 1024);
                if (!clipped(e, AN("conv_pw1"), h, n, 1024, 2048, up, 0, err, len))
                    return false;
                conv_glu_kernel<<<blocks(count), 256, 0, st>>>(up, h, n);
                float *dw = proj_float(e, AN("conv_dw"), err, len);
                float *nc2 = proj_float(e, AN("norm_conv"), err, len);
                if (!dw || !nc2)
                    return false;
                depthwise_kernel<<<blocks(count), 256, 0, st>>>(h, dw, att, n);
                rms_kernel<<<n, 256, 0, st>>>(att, nc2, att, 1024);
                silu_kernel<<<blocks(count), 256, 0, st>>>(att, count);
                if (!clipped(e, AN("conv_pw2"), att, n, 1024, 1024, h, 0, err, len))
                    return false;
                post_add_kernel<<<n, 256, 0, st>>>(h, x, nullptr, 1.0f, nullptr, nullptr, nullptr,
                                                   1024, 1.0f);
            }
            float *fn = proj_float(e, ANS(ffn[half][0]), err, len);
            if (!fn)
                return false;
            rms_kernel<<<n, 256, 0, st>>>(x, fn, h, 1024);
            if (!clipped(e, ANS(ffn[half][1]), h, n, 1024, 4096, up, 0, err, len))
                return false;
            // SiLU is applied to up before the down projection's input clamp.
            if (!clipped(e, ANS(ffn[half][2]), up, n, 4096, 1024, h, 1, err, len))
                return false;
            float *pn = proj_float(e, ANS(ffn[half][3]), err, len);
            if (!pn)
                return false;
            post_add_kernel<<<n, 256, 0, st>>>(h, x, pn, 1.0f, nullptr, nullptr, nullptr, 1024,
                                               0.5f);
        }
        float *ln2 = proj_float(e, AN("ln2"), err, len);
        if (!ln2)
            return false;
        rms_kernel<<<n, 256, 0, st>>>(x, ln2, x, 1024);
    }
#undef AN
    if (!clipped(e, "a.pre_encode.out.weight", x, n, 1024, 1536, up, 0, err, len))
        return false;
    float *bias = proj_float(e, "a.pre_encode.out.bias", err, len);
    if (!bias)
        return false;
    bias_kernel<<<blocks((size_t)n * 1536), 256, 0, st>>>(up, bias, n, 1536);
    rms_kernel<<<n, 256, 0, st>>>(up, nullptr, up, 1536);
    if (!clipped(e, "mm.a.input_projection.weight", up, n, 1536, 512, result, 0, err, len))
        return false;
    *tokens = n;
    return finish_media(e, result, (size_t)n * 512, out, err, len);
}

bool use_projector(engine *e, const ei_gguf *g, char *err, size_t len) {
    if (e->projector && e->projector != g) {
        snprintf(err, len, "EmbeddingGemma 2 CUDA supports one projector per engine");
        return false;
    }
    e->projector = g;
    return check(cudaSetDevice(e->device), "select device", err, len);
}

} // namespace

extern "C" int ei_gemma2_cuda_device_count(void) {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0;
}

extern "C" int ei_gemma2_cuda_device(void *ctx) {
    return ((engine *)ctx)->device;
}

extern "C" void *ei_gemma2_cuda_create(const ei_gguf *g, char *err, size_t len) {
    const char *dev = getenv("EI_CUDA_DEVICE");
    return ei_gemma2_cuda_create_on(g, dev ? atoi(dev) : 0, err, len);
}

extern "C" void *ei_gemma2_cuda_create_on(const ei_gguf *g, int device, char *err, size_t len) {
    int count = ei_gemma2_cuda_device_count();
    if (count == 0) {
        snprintf(err, len, "no CUDA device");
        return nullptr;
    }
    if (device < 0 || device >= count) {
        snprintf(err, len, "CUDA device %d does not exist (%d visible)", device, count);
        return nullptr;
    }
    engine *e = new engine();
    e->g = g;
    e->device = device;
    const char *graphs = getenv("EI_GEMMA2_CUDA_GRAPHS");
    e->use_graphs = !graphs || strcmp(graphs, "0");
    const char *mma = getenv("EI_GEMMA2_CUDA_MMA_ATTN");
    e->mma_attention = !mma || strcmp(mma, "0");
    cudaSetDevice(e->device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, e->device);
    // Tensor-core attention needs mma.sync m16n8k16 (sm_80+) and opt-in shared
    // memory for the largest tile; decided from the device, not assumed.
    if (e->mma_attention && (prop.major < 8 || prop.sharedMemPerBlockOptin < kFlashSmemMax)) {
        fprintf(stderr,
                "EmbeddingGemma 2 CUDA: sm_%d%d / %zu B opt-in shared memory, using CUDA-core "
                "attention\n",
                prop.major, prop.minor, (size_t)prop.sharedMemPerBlockOptin);
        e->mma_attention = false;
    }
    if (e->mma_attention &&
        (configure_flash<G2_FLASH_256>() != cudaSuccess ||
         configure_flash<G2_FLASH_512>() != cudaSuccess ||
         configure_flash<G2_FLASH_64>() != cudaSuccess)) {
        fprintf(stderr, "EmbeddingGemma 2 CUDA: tensor-core attention unavailable, using CUDA cores\n");
        e->mma_attention = false;
    }
    bool ok = check(cudaSetDevice(e->device), "select device", err, len) &&
              check(cudaStreamCreateWithFlags(&e->stream, cudaStreamNonBlocking), "stream", err,
                    len) &&
              check_blas(cublasCreate(&e->blas), "create", err, len) &&
              check_blas(cublasSetStream(e->blas, e->stream), "stream", err, len) &&
              check(cudaMalloc(&e->blas_workspace, 32 << 20), "cuBLAS workspace", err, len) &&
              check_blas(cublasSetWorkspace(e->blas, e->blas_workspace, 32 << 20), "workspace",
                         err, len);
    const ei_tensor *emb = tensor(g, "token_embd.weight");
    ok = ok && upload(e, &e->token_embd, emb->data, emb->n_bytes, err, len) &&
         upload_half(e, &e->ple_proj, {tensor(g, "per_layer_model_proj.weight")}, err, len) &&
         upload_half(e, &e->output, {tensor(g, "output.weight")}, err, len) &&
         upload_float(e, &e->ple_norm, tensor(g, "per_layer_proj_norm.weight"), err, len) &&
         upload_float(e, &e->output_norm, tensor(g, "output_norm.weight"), err, len);
    for (int l = 0; ok && l < kLayers; l++) {
        layer_weights &w = e->layers[l];
        ok = upload_half(e, &w.qkv,
                         {tensor(g, "blk.%d.attn_q.weight", l), tensor(g, "blk.%d.attn_k.weight", l),
                          tensor(g, "blk.%d.attn_v.weight", l)},
                         err, len) &&
             upload_half(e, &w.o, {tensor(g, "blk.%d.attn_output.weight", l)}, err, len) &&
             upload_half(e, &w.upgate,
                         {tensor(g, "blk.%d.ffn_up.weight", l), tensor(g, "blk.%d.ffn_gate.weight", l)},
                         err, len) &&
             upload_half(e, &w.down, {tensor(g, "blk.%d.ffn_down.weight", l)}, err, len) &&
             upload_half(e, &w.inp_gate, {tensor(g, "blk.%d.inp_gate.weight", l)}, err, len) &&
             upload_half(e, &w.proj, {tensor(g, "blk.%d.proj.weight", l)}, err, len) &&
             upload_float(e, &w.attn_norm, tensor(g, "blk.%d.attn_norm.weight", l), err, len) &&
             upload_float(e, &w.q_norm, tensor(g, "blk.%d.attn_q_norm.weight", l), err, len) &&
             upload_float(e, &w.k_norm, tensor(g, "blk.%d.attn_k_norm.weight", l), err, len) &&
             upload_float(e, &w.post_attn, tensor(g, "blk.%d.post_attention_norm.weight", l), err,
                          len) &&
             upload_float(e, &w.ffn_norm, tensor(g, "blk.%d.ffn_norm.weight", l), err, len) &&
             upload_float(e, &w.post_ffw, tensor(g, "blk.%d.post_ffw_norm.weight", l), err, len) &&
             upload_float(e, &w.post_norm, tensor(g, "blk.%d.post_norm.weight", l), err, len);
        char name[96];
        snprintf(name, sizeof name, "blk.%d.layer_output_scale.weight", l);
        const ei_tensor *scale = ei_gguf_tensor(g, name, false);
        w.output_scale = scale ? host_element(scale, 0) : 1.0f;
    }
    for (int b = 0; ok && b < 2; b++) {
        int d = b ? 512 : 256;
        double base = b ? 1e6 : 1e4;
        std::vector<float2> host((size_t)8192 * d / 2);
        for (int t = 0; t < 8192; t++)
            for (int z = 0; z < d / 2; z++) {
                double theta = t * pow(base, -2.0 * z / d);
                host[(size_t)t * d / 2 + z] = make_float2((float)cos(theta), (float)sin(theta));
            }
        ok = upload(e, &e->rope[b], host.data(), host.size() * sizeof(float2), err, len);
    }
    if (!ok) {
        ei_gemma2_cuda_free(e);
        return nullptr;
    }
    fprintf(stderr, "EmbeddingGemma 2 CUDA: %s (sm_%d%d, text, vision, audio)\n", prop.name, prop.major,
            prop.minor);
    return e;
}

extern "C" void ei_gemma2_cuda_free(void *ctx) {
    engine *e = (engine *)ctx;
    if (!e)
        return;
    cudaSetDevice(e->device);
    for (graph_entry &g : e->graphs)
        cudaGraphExecDestroy(g.exec);
    for (void *p : e->owned)
        cudaFree(p);
    for (float *p : e->rpe)
        cudaFree(p);
    for (auto &kv : e->scratch)
        cudaFree(kv.second.first);
    void *ws[] = {e->ids, e->lo, e->hi, e->pos, e->offsets, e->x, e->ple, e->proj, e->wide,
                  e->pooled, e->inputs, e->h16, e->q16, e->k16, e->v16, e->a16, e->g16, e->x16};
    for (void *p : ws)
        cudaFree(p);
    if (e->blas)
        cublasDestroy(e->blas);
    cudaFree(e->blas_workspace);
    if (e->stream)
        cudaStreamDestroy(e->stream);
    delete e;
}

extern "C" bool ei_gemma2_cuda_batch(void *ctx, const int32_t *ids, const size_t *offsets,
                                     size_t batch, float *out, char *err, size_t len) {
    engine *e = (engine *)ctx;
    // Waves above 8192 tokens are split at sequence boundaries.
    size_t first = 0;
    while (first < batch) {
        size_t last = first + 1;
        while (last < batch && offsets[last + 1] - offsets[first] <= 8192)
            last++;
        std::vector<size_t> local(last - first + 1);
        for (size_t s = first; s <= last; s++)
            local[s - first] = offsets[s] - offsets[first];
        if (!run(e, ids + offsets[first], nullptr, local.data(), last - first, out + first * kOut,
                 err, len))
            return false;
        first = last;
    }
    return true;
}

extern "C" bool ei_gemma2_cuda_embed_inputs(void *ctx, const float *inputs, size_t n,
                                            float out[768], char *err, size_t len) {
    size_t offsets[2] = {0, n};
    return run((engine *)ctx, nullptr, inputs, offsets, 1, out, err, len);
}

extern "C" bool ei_gemma2_cuda_vision(void *ctx, const ei_gguf *g, const uint8_t *rgb, int width,
                                      int height, float **out, size_t *tokens, char *err,
                                      size_t len) {
    engine *e = (engine *)ctx;
    return use_projector(e, g, err, len) && vision(e, rgb, width, height, out, tokens, err, len);
}

extern "C" bool ei_gemma2_cuda_audio(void *ctx, const ei_gguf *g, const float *mel, int frames,
                                     float **out, size_t *tokens, char *err, size_t len) {
    engine *e = (engine *)ctx;
    return use_projector(e, g, err, len) && audio(e, mel, frames, out, tokens, err, len);
}

extern "C" void *ei_gemma2_cuda_clone(void *ctx, bool high_priority, char *err, size_t len) {
    engine *p = (engine *)ctx, *e = new engine();
    e->g = p->g;
    e->device = p->device;
    e->token_embd = p->token_embd;
    e->ple_proj = p->ple_proj;
    e->output = p->output;
    e->ple_norm = p->ple_norm;
    e->output_norm = p->output_norm;
    e->rope[0] = p->rope[0];
    e->rope[1] = p->rope[1];
    for (int l = 0; l < kLayers; l++)
        e->layers[l] = p->layers[l];
    e->use_graphs = p->use_graphs;
    e->mma_attention = p->mma_attention;
    // Shared weights stay owned by the parent; the clone owns its stream,
    // cuBLAS handle, workspace, graphs, and any projector weights it loads.
    int least = 0, greatest = 0;
    bool ok = check(cudaSetDevice(e->device), "select device", err, len) &&
              check(cudaDeviceGetStreamPriorityRange(&least, &greatest), "stream priorities", err,
                    len) &&
              check(cudaStreamCreateWithPriority(&e->stream, cudaStreamNonBlocking,
                                                 high_priority ? greatest : least),
                    "stream", err, len) &&
              check_blas(cublasCreate(&e->blas), "create", err, len) &&
              check_blas(cublasSetStream(e->blas, e->stream), "stream", err, len) &&
              check(cudaMalloc(&e->blas_workspace, 32 << 20), "cuBLAS workspace", err, len) &&
              check_blas(cublasSetWorkspace(e->blas, e->blas_workspace, 32 << 20), "workspace",
                         err, len);
    if (!ok) {
        ei_gemma2_cuda_free(e);
        return nullptr;
    }
    return e;
}
