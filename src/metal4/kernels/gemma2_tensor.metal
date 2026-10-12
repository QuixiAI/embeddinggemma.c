#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include <metal_tensor>
using namespace metal;
struct G2P {
    uint n, k, m, type, heads, kv, d, window, cols, mode, layer;
    float scale, lo, hi;
};
struct G2Q8 {
    half d;
    char qs[32];
};
inline float g2gelu_t(float x) {
    return 0.5f * x * (1 + tanh(0.7978845608f * x * (1 + 0.044715f * x * x)));
}
inline half g2_weight(device const uchar *w, uint type, uint i) {
    if (type == 0)
        return half(((device const float *)w)[i]);
    if (type == 30)
        return half(as_type<float>(uint(((device const ushort *)w)[i]) << 16));
    device const G2Q8 *b = (device const G2Q8 *)w;
    return half(float(b[i / 32].d) * float(b[i / 32].qs[i % 32]));
}
// Q8/BF16/F32 -> half weight tiles, FP32 activations and accumulators. Fixed
// K multiples of 32 and output multiples of 64 for all three Gemma 2 graphs.
kernel void g2_tensor_mm(device const uchar *w [[buffer(0)]], device float *x [[buffer(1)]],
                         device float *y [[buffer(2)]], constant G2P &p [[buffer(3)]],
                         uint2 g [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half wt[64 * 32];
    auto ta = tensor(wt, dextents<int32_t, 2>(32, 64));
    auto tx = tensor(x, dextents<int32_t, 2>(int(p.k), int(p.n)), array<int, 2>({1, int(p.k)}));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(
                                  64, 64, 32, false, true, true,
                                  mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                              execution_simdgroups<4>>
        op;
    auto acc = op.template get_destination_cooperative_tensor<decltype(tx), decltype(ta), float>();
    for (uint k = 0; k < p.k; k += 32) {
        for (uint i = tid; i < 64 * 32; i += 128)
            wt[i] = g2_weight(w, p.type, (g.y * 64 + i / 32) * p.k + k + i % 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto left = tx.slice(k, g.x * 64);
        auto right = ta.slice(0, 0);
        op.run(left, right, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto ty = tensor(y, dextents<int32_t, 2>(int(p.m), int(p.n)), array<int, 2>({1, int(p.m)}));
    acc.store(ty.slice(g.y * 64, g.x * 64));
}
// Dense GEMM for wide waves: row-major fp16 weights W[m][k] (Q8_0/BF16
// dequantized once at load) read as a transposed right operand, fp32
// activations, relaxed-precision matmul2d with fp32 accumulation. TM tokens x
// 64 rows per threadgroup; rows past p.n read buffer slack and are not stored.
// Measured alternatives (int8 operand + per-group scales, tile-major fp16,
// strict precision, fp16 activations) are in perf/optimization_status.md.
// EPI 0 stores y[n][m]. EPI 1 (GeGLU): W interleaves 32 up rows then the 32
// matching gate rows per 64-row tile; stores gelu(gate) * up into y[n][m / 2].
// EPI 2 (PLE gate): stores gelu(y) * ple[token][p.layer][row] (buffer 4).
// EPI 3: as EPI 1 with the vision tower's quick GELU, x * sigmoid(1.702 x).
// EPI 4: stores clamp(y, p.lo, p.hi) (the audio tower's output clamps).
template <int TM, int EPI>
kernel void g2t_gemm_r(device const half *w [[buffer(0)]], device const float *x [[buffer(1)]],
                       device float *y [[buffer(2)]], constant G2P &p [[buffer(3)]],
                       device const float *ple [[buffer(4)]],
                       uint2 g [[threadgroup_position_in_grid]],
                       ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int KS = 64;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        TM, 64, KS, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    using ta = tensor<device float, extents<int32_t, KS, TM>, tensor_inline>;
    using tb = tensor<device half, extents<int32_t, KS, 64>, tensor_inline>;
    const int k = int(p.k), m = int(p.m), t0 = int(g.x) * TM, r0 = int(g.y) * 64;
    auto acc = op.template get_destination_cooperative_tensor<ta, tb, float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i))
            acc[i] = 0;
    for (int k0 = 0; k0 < k; k0 += KS) {
        ta A((device float *)(x + (size_t)t0 * k + k0), extents<int32_t, KS, TM>(),
             array<int32_t, 2>{1, k});
        tb B((device half *)(w + (size_t)r0 * k + k0), extents<int32_t, KS, 64>(),
             array<int32_t, 2>{1, k});
        op.run(A, B, acc);
    }
    if (EPI == 1 || EPI == 3) {
        threadgroup float st[TM * 64];
        for (uint16_t i = 0; i < acc.get_capacity(); ++i)
            if (acc.is_valid_element(i)) {
                auto idx = acc.get_multidimensional_index(i);
                st[idx[1] * 64 + idx[0]] = acc[i];
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < TM * 32; i += 128) {
            int t = i / 32, c = i % 32;
            if (t0 + t < int(p.n)) {
                float gate = st[t * 64 + 32 + c];
                y[(size_t)(t0 + t) * (m / 2) + r0 / 2 + c] =
                    (EPI == 3 ? gate / (1 + exp(-1.702f * gate)) : g2gelu_t(gate)) *
                    st[t * 64 + c];
            }
        }
        return;
    }
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            auto idx = acc.get_multidimensional_index(i);
            int t = t0 + idx[1], r = r0 + idx[0];
            if (t < int(p.n)) {
                float v = acc[i];
                if (EPI == 2)
                    v = g2gelu_t(v) * ple[((size_t)t * 24 + p.layer) * 512 + r];
                if (EPI == 4)
                    v = clamp(v, p.lo, p.hi);
                y[(size_t)t * m + r] = v;
            }
        }
}
#define G2T_GEMM_R(TM, EPI, name)                                                              \
    template [[host_name(name)]] kernel void g2t_gemm_r<TM, EPI>(                             \
        device const half *, device const float *, device float *, constant G2P &,            \
        device const float *, uint2, ushort);
G2T_GEMM_R(32, 0, "g2t_gemm_32")
G2T_GEMM_R(64, 0, "g2t_gemm_64")
G2T_GEMM_R(32, 1, "g2t_gemm_geglu_32")
G2T_GEMM_R(64, 1, "g2t_gemm_geglu_64")
G2T_GEMM_R(32, 2, "g2t_gemm_ple_32")
G2T_GEMM_R(64, 2, "g2t_gemm_ple_64")
G2T_GEMM_R(32, 3, "g2t_gemm_qgeglu_32")
G2T_GEMM_R(64, 3, "g2t_gemm_qgeglu_64")
G2T_GEMM_R(32, 4, "g2t_gemm_clamp_32")
G2T_GEMM_R(64, 4, "g2t_gemm_clamp_64")
// Tensor-op flash attention for packed waves. One threadgroup = TQ queries x
// one head; 64-key blocks. S = Q K^T and O += P V run on matmul2d; the score
// tile goes through threadgroup memory for the online softmax (one simdgroup
// per 8 rows). Rows mask keys outside their sequence bounds and, when
// p.window != 0, outside |t - s| <= window. K/V are fp16, zero-padded to a
// 64-key multiple past p.n. Q is fp32 [t][heads][D], output fp32 likewise.
// DQ is the real head width of Q and the output; D is the K/V row width in
// memory. D > DQ pads K/V rows with zeros (the g2_half_pad layout) so a head
// width the driver cannot compile directly (64) runs as a 128-wide tile: the
// pad columns add exact zeros to S and P.V and are never stored.
template <int D, int DQ, int TQ, int TK, int NSG>
kernel void g2t_flash(device const float *q [[buffer(0)]], device const half *k [[buffer(1)]],
                      device const half *v [[buffer(2)]], device float *out [[buffer(3)]],
                      constant G2P &p [[buffer(4)]], device const uint2 *bounds [[buffer(5)]],
                      uint2 g [[threadgroup_position_in_grid]],
                      ushort tid [[thread_index_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int DK = 64, ROWS = TQ / NSG, KPL = TK / 32;
    threadgroup float S[TQ * TK];
    threadgroup half P[TQ * TK];
    threadgroup float row_max[TQ], row_sum[TQ], row_alpha[TQ];
    const int n = int(p.n), heads = int(p.heads), kvh = int(p.kv), t0 = int(g.x) * TQ;
    const int h = int(g.y), kh = h / (heads / kvh), w = int(p.window);
    const int qs = heads * DQ, ks = kvh * D;
    // Key range of the tile: bounds and windows are monotone in t.
    const int tl = min(t0 + TQ, n) - 1;
    // p.layer == 0: one unpacked sequence (vision), no bounds buffer.
    const bool packed = p.layer != 0;
    int lo = packed ? int(bounds[t0].x) : 0, hi = packed ? int(bounds[tl].y) : n;
    if (w) {
        lo = max(lo, t0 - w);
        hi = min(hi, tl + w + 1);
    }
    if (tid < TQ) {
        row_max[tid] = -INFINITY;
        row_sum[tid] = 0;
    }
    constexpr auto sdesc = mpp::tensor_ops::matmul2d_descriptor(
        TQ, TK, DK, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<sdesc, execution_simdgroups<NSG>> sop;
    constexpr auto odesc = mpp::tensor_ops::matmul2d_descriptor(
        TQ, D, TK, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<odesc, execution_simdgroups<NSG>> oop;
    using tq = tensor<device float, extents<int32_t, DK, TQ>, tensor_inline>;
    using tk = tensor<device half, extents<int32_t, DK, TK>, tensor_inline>;
    using tp = tensor<threadgroup half, extents<int32_t, TK, TQ>, tensor_inline>;
    using tv = tensor<device half, extents<int32_t, D, TK>, tensor_inline>;
    auto o = oop.template get_destination_cooperative_tensor<tp, tv, float>();
    for (uint16_t i = 0; i < o.get_capacity(); ++i)
        if (o.is_valid_element(i))
            o[i] = 0;
    tp tP(P, extents<int32_t, TK, TQ>());
    for (int s0 = lo; s0 < hi; s0 += TK) {
        auto sc = sop.template get_destination_cooperative_tensor<tq, tk, float>();
        for (uint16_t i = 0; i < sc.get_capacity(); ++i)
            if (sc.is_valid_element(i))
                sc[i] = 0;
        for (int d0 = 0; d0 < DQ; d0 += DK) {
            tq A((device float *)(q + (size_t)t0 * qs + h * DQ + d0), extents<int32_t, DK, TQ>(),
                 array<int32_t, 2>{1, qs});
            tk B((device half *)(k + (size_t)s0 * ks + kh * D + d0), extents<int32_t, DK, TK>(),
                 array<int32_t, 2>{1, ks});
            sop.run(A, B, sc);
        }
        for (uint16_t i = 0; i < sc.get_capacity(); ++i)
            if (sc.is_valid_element(i)) {
                auto idx = sc.get_multidimensional_index(i);
                S[idx[1] * TK + idx[0]] = sc[i];
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Online softmax: simdgroup sg owns ROWS rows; lanes own KPL keys each.
        for (int r = sg * ROWS; r < sg * ROWS + ROWS; ++r) {
            int t = min(t0 + r, n - 1);
            int rlo = packed ? int(bounds[t].x) : 0, rhi = packed ? int(bounds[t].y) : n;
            if (w) {
                rlo = max(rlo, t - w);
                rhi = min(rhi, t + w + 1);
            }
            float xs[KPL], local = -INFINITY;
            for (int j = 0; j < KPL; ++j) {
                int s = s0 + lane + 32 * j;
                xs[j] = s >= rlo && s < rhi ? S[r * TK + lane + 32 * j] : -INFINITY;
                local = max(local, xs[j]);
            }
            float old = row_max[r], mx = max(old, simd_max(local));
            float sum = 0;
            for (int j = 0; j < KPL; ++j) {
                float e = mx == -INFINITY ? 0 : exp(xs[j] - mx);
                sum += e;
                P[r * TK + lane + 32 * j] = half(e);
            }
            sum = simd_sum(sum);
            if (lane == 0) {
                float alpha = mx == -INFINITY ? 1 : exp(old - mx);
                row_max[r] = mx;
                row_sum[r] = row_sum[r] * alpha + sum;
                row_alpha[r] = alpha;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint16_t i = 0; i < o.get_capacity(); ++i)
            if (o.is_valid_element(i))
                o[i] *= row_alpha[o.get_multidimensional_index(i)[1]];
        tv B((device half *)(v + (size_t)s0 * ks + kh * D), extents<int32_t, D, TK>(),
             array<int32_t, 2>{1, ks});
        oop.run(tP, B, o);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < o.get_capacity(); ++i)
        if (o.is_valid_element(i)) {
            auto idx = o.get_multidimensional_index(i);
            int t = t0 + idx[1];
            if (t < n && idx[0] < DQ)
                out[(size_t)t * qs + h * DQ + idx[0]] = o[i] / row_sum[idx[1]];
        }
}
#define G2T_FLASH(D, DQ, TQ, TK, NSG, name)                                                    \
    template [[host_name(name)]] kernel void g2t_flash<D, DQ, TQ, TK, NSG>(                   \
        device const float *, device const half *, device const half *, device float *,       \
        constant G2P &, device const uint2 *, uint2, ushort, ushort, ushort);
// 32 queries x 128 keys won the 2026-10-09 sweep (1.06-1.16x over 32x64);
// 64x64 on 8 simdgroups was neutral; 64x128 exceeds 32 KB threadgroup memory.
// No D=64 instantiation: any g2t_flash<64, ...> crashes the M5 driver at first
// dispatch (AGX RayPayloadSizeAlignImpl::merge, macOS 26.6.2); vision attention
// uses g2_flash_64 instead.
G2T_FLASH(256, 256, 32, 128, 4, "g2t_flash_256")
G2T_FLASH(512, 512, 32, 128, 4, "g2t_flash_512")
G2T_FLASH(256, 256, 32, 64, 4, "g2t_flash_256_k64")
G2T_FLASH(512, 512, 32, 64, 4, "g2t_flash_512_k64")
// Vision: 64-wide heads on 128-wide zero-padded K/V rows (see g2_half_pad).
G2T_FLASH(128, 64, 32, 128, 4, "g2t_flash_64p")
// Long-context tile experiments (EI_GEMMA2_METAL_FLASH2_TILE=q16 | q16k256 |
// n8): fewer rows per threadgroup (smaller O accumulators per lane at head
// width 512), 256-key blocks (half the softmax passes), 8 simdgroups.
G2T_FLASH(256, 256, 16, 128, 4, "g2t_flash_256_q16")
G2T_FLASH(512, 512, 16, 128, 4, "g2t_flash_512_q16")
G2T_FLASH(256, 256, 16, 256, 4, "g2t_flash_256_q16k256")
G2T_FLASH(512, 512, 16, 256, 4, "g2t_flash_512_q16k256")
G2T_FLASH(256, 256, 32, 128, 8, "g2t_flash_256_n8")
G2T_FLASH(512, 512, 32, 128, 8, "g2t_flash_512_n8")
G2T_FLASH(256, 256, 16, 128, 8, "g2t_flash_256_q16n8")
G2T_FLASH(512, 512, 16, 128, 8, "g2t_flash_512_q16n8")
G2T_FLASH(256, 256, 16, 256, 8, "g2t_flash_256_q16k256n8")
G2T_FLASH(512, 512, 16, 256, 8, "g2t_flash_512_q16k256n8")
G2T_FLASH(256, 256, 32, 128, 16, "g2t_flash_256_n16")
G2T_FLASH(512, 512, 32, 128, 16, "g2t_flash_512_n16")
// 64 queries x 64 keys: twice the FLOPs per K/V byte of the 32x128 tile.
G2T_FLASH(256, 256, 64, 64, 8, "g2t_flash_256_q64k64n8")
G2T_FLASH(512, 512, 64, 64, 8, "g2t_flash_512_q64k64n8")
G2T_FLASH(256, 256, 64, 64, 16, "g2t_flash_256_q64k64n16")
G2T_FLASH(512, 512, 64, 64, 16, "g2t_flash_512_q64k64n16")
