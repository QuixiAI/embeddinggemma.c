#include "../include/gemma2_common.metal"
#include <metal_simdgroup_matrix>
// Packed-wave GEMM for GPUs without the Metal 4 tensor route: y = x W^T on the
// simdgroup 8x8 matrix units Metal 3 exposes (Apple7+). Each threadgroup reuses
// one 64-row x 32-k weight tile across 32 tokens, which the per-token GEMVs and
// g2_mm cannot; operands are fp16 in threadgroup memory and every sum
// accumulates in fp32. The model card says the activations exceed fp16's range
// (measured operands stay below 740, see README), so g2s_rowscale gives every
// token a scale: 1 while its row stays within SG_XMAX, which leaves the
// ordinary wave bit-identical, else |x|max / SG_XMAX so the staged fp16 row is
// finite; the epilogue multiplies it back.
constant constexpr uint SG_TOKENS = 32, SG_ROWS = 64, SG_K = 32;
constant constexpr float SG_XMAX = 16384.f;

// One simdgroup per token: scale[t] for row t of x (p.k floats, k % 32 == 0).
kernel void g2s_rowscale(device const float *x [[buffer(0)]], device float *scale [[buffer(1)]],
                         constant G2P &p [[buffer(2)]], uint t [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    device const float *row = x + (size_t)t * p.k;
    float peak = 0;
    for (uint i = lane; i < p.k; i += 32)
        peak = max(peak, abs(row[i]));
    peak = simd_max(peak);
    if (lane == 0)
        scale[t] = peak > SG_XMAX ? peak / SG_XMAX : 1.f;
}

// 128 threads: thread t fills 16 k of weight row t / 2 and 8 k of token t / 4,
// and simdgroup s owns all 32 tokens x rows 16s..16s+15 of the tile.
kernel void g2s_gemm(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]],
                     device float *y [[buffer(2)]], constant G2P &p [[buffer(3)]],
                     device const float *scale [[buffer(4)]],
                     uint2 g [[threadgroup_position_in_grid]],
                     uint t [[thread_index_in_threadgroup]],
                     uint s [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sw[SG_ROWS * SG_K];
    threadgroup half sx[SG_TOKENS * SG_K];
    threadgroup float sy[SG_TOKENS * SG_ROWS];
    const uint row0 = g.x * SG_ROWS, token0 = g.y * SG_TOKENS;
    const uint wr = t / 2, wk = (t % 2) * 16, xr = t / 4, xk = (t % 4) * 8;
    const bool live = token0 + xr < p.n;
    // 1 / scale is exact for the common scale of 1; a scaled row is already
    // beyond fp16's precision at its magnitude, so the division costs nothing.
    const float inv = live ? 1.f / scale[token0 + xr] : 0.f;
    simdgroup_float8x8 acc[8];
    for (uint i = 0; i < 8; i++)
        acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    for (uint k0 = 0; k0 < p.k; k0 += SG_K) {
        const uint wi = (row0 + wr) * p.k + k0 + wk;
        threadgroup half *dw = sw + wr * SG_K + wk;
        if (p.type == 8) {
            device const G2Q8 *b = (device const G2Q8 *)w + wi / 32;
            const float scale = float(b->d);
            for (uint j = 0; j < 16; j++)
                dw[j] = half(scale * float(b->qs[wk + j]));
        } else {
            for (uint j = 0; j < 16; j++)
                dw[j] = half(g2w(w, p.type, wi + j));
        }
        device const float *src = x + (size_t)(token0 + xr) * p.k + k0 + xk;
        threadgroup half *dx = sx + xr * SG_K + xk;
        for (uint j = 0; j < 8; j++)
            dx[j] = live ? half(src[j] * inv) : half(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < SG_K; kk += 8) {
            simdgroup_half8x8 a[4], b[2];
            for (uint i = 0; i < 4; i++)
                simdgroup_load(a[i], sx + i * 8 * SG_K + kk, SG_K);
            for (uint i = 0; i < 2; i++)
                simdgroup_load(b[i], sw + (s * 16 + i * 8) * SG_K + kk, SG_K, ulong2(0, 0), true);
            for (uint i = 0; i < 4; i++)
                for (uint j = 0; j < 2; j++)
                    simdgroup_multiply_accumulate(acc[i * 2 + j], a[i], b[j], acc[i * 2 + j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 2; j++)
            simdgroup_store(acc[i * 2 + j], sy + i * 8 * SG_ROWS + s * 16 + j * 8, SG_ROWS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool clamped = p.lo < p.hi;
    for (uint e = t; e < SG_TOKENS * SG_ROWS; e += 128) {
        const uint token = token0 + e / SG_ROWS;
        if (token >= p.n)
            continue;
        const float v = sy[e] * scale[token];
        y[(size_t)token * p.m + row0 + e % SG_ROWS] = clamped ? clamp(v, p.lo, p.hi) : v;
    }
}

// Flash attention for packed waves without Metal 4, matching g2_attention's
// masks (symmetric window, keys inside the token's own sequence) and its
// unscaled scores. One simdgroup takes 8 consecutive queries of one head
// against fp16 K/V (rows zero-padded past p.n by at least 32) in blocks of 32
// keys: scores and P.V run on the 8x8 matrix units, the softmax is online, and
// the 8 x D output stays in registers.
constant constexpr uint FA_QUERIES = 8, FA_KEYS = 32;

template <uint D>
inline void g2s_flash(device const float *q, device const half *kh, device const half *vh,
                      device float *out, constant G2P &p, device const uint2 *bounds,
                      threadgroup half *sq, threadgroup float *ss, threadgroup half *sp,
                      threadgroup float *sd, uint q0, uint h, uint lane) {
    const uint kvh = h / (p.heads / p.kv), stride = p.kv * D;
    const uint last = min(q0 + FA_QUERIES, p.n) - 1;
    for (uint i = lane; i < FA_QUERIES * D; i += 32) {
        const uint t = q0 + i / D;
        sq[i] = t < p.n ? half(q[(t * p.heads + h) * D + i % D]) : half(0);
    }
    // Lane l scores row l / 4, keys (l % 4) * 8 .. +7 of each block.
    const uint row = lane / 4, col = (lane % 4) * 8, t = min(q0 + row, last);
    uint lo = p.window && t > p.window ? t - p.window : 0,
         hi = p.window ? min(p.n, t + p.window + 1) : p.n;
    uint first = p.window && q0 > p.window ? q0 - p.window : 0,
         end = p.window ? min(p.n, last + p.window + 1) : p.n;
    if (p.layer) {
        lo = max(lo, bounds[t].x);
        hi = min(hi, bounds[t].y);
        first = max(first, bounds[q0].x);
        end = min(end, bounds[last].y);
    }
    simdgroup_float8x8 o[D / 8];
    for (uint i = 0; i < D / 8; i++)
        o[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    float mx = -INFINITY, total = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key0 = first; key0 < end; key0 += FA_KEYS) {
        simdgroup_float8x8 s[4];
        for (uint j = 0; j < 4; j++)
            s[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
        for (uint d0 = 0; d0 < D; d0 += 8) {
            simdgroup_half8x8 a;
            simdgroup_load(a, sq + d0, D);
            for (uint j = 0; j < 4; j++) {
                simdgroup_half8x8 b;
                simdgroup_load(b, kh + (size_t)(key0 + j * 8) * stride + kvh * D + d0, stride,
                               ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s[j], a, b, s[j]);
            }
        }
        for (uint j = 0; j < 4; j++)
            simdgroup_store(s[j], ss + j * 8, FA_KEYS);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float v[8], peak = -INFINITY;
        for (uint j = 0; j < 8; j++) {
            const uint key = key0 + col + j;
            v[j] = key >= lo && key < hi ? ss[row * FA_KEYS + col + j] : -INFINITY;
            peak = max(peak, v[j]);
        }
        peak = max(peak, simd_shuffle_xor(peak, 1));
        peak = max(peak, simd_shuffle_xor(peak, 2));
        const float next = max(mx, peak);
        const float scale = next == -INFINITY ? 1 : exp(mx - next);
        float sum = 0;
        for (uint j = 0; j < 8; j++) {
            const float e = v[j] == -INFINITY ? 0 : exp(v[j] - next);
            sum += e;
            sp[row * FA_KEYS + col + j] = half(e);
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        total = total * scale + sum;
        mx = next;
        // Once the running maxima settle every scale is exactly 1 (exp(0)),
        // and the D / 8 diagonal multiplies are a quarter of the P.V work;
        // the test is simdgroup-uniform, so skipping them changes no bit.
        const bool rescaled = !simd_all(scale == 1.f);
        if (rescaled && lane % 4 == 0)
            for (uint j = 0; j < 8; j++)
                sd[row * 8 + j] = j == row ? scale : 0;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 rescale;
        if (rescaled)
            simdgroup_load(rescale, sd, 8);
        simdgroup_half8x8 pj[4];
        for (uint j = 0; j < 4; j++)
            simdgroup_load(pj[j], sp + j * 8, FA_KEYS);
        for (uint i = 0; i < D / 8; i++) {
            if (rescaled)
                simdgroup_multiply(o[i], rescale, o[i]);
            for (uint j = 0; j < 4; j++) {
                simdgroup_half8x8 b;
                simdgroup_load(b, vh + (size_t)(key0 + j * 8) * stride + kvh * D + i * 8, stride);
                simdgroup_multiply_accumulate(o[i], pj[j], b, o[i]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane % 4 == 0)
        for (uint j = 0; j < 8; j++)
            sd[row * 8 + j] = j == row ? 1 / total : 0;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 norm;
    simdgroup_load(norm, sd, 8);
    const bool clamped = p.lo < p.hi;
    for (uint i = 0; i < D / 8; i++) {
        simdgroup_multiply(o[i], norm, o[i]);
        simdgroup_store(o[i], ss, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 64; e += 32) {
            const uint tt = q0 + e / 8;
            if (tt < p.n) {
                const float r = ss[e];
                out[(tt * p.heads + h) * D + i * 8 + e % 8] = clamped ? clamp(r, p.lo, p.hi) : r;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define G2S_FLASH(D)                                                                              \
    kernel void g2s_flash_##D(device const float *q [[buffer(0)]],                                \
                              device const half *kh [[buffer(1)]],                                \
                              device const half *vh [[buffer(2)]], device float *out [[buffer(3)]], \
                              constant G2P &p [[buffer(4)]],                                      \
                              device const uint2 *bounds [[buffer(5)]],                           \
                              uint2 g [[threadgroup_position_in_grid]],                           \
                              uint lane [[thread_index_in_simdgroup]]) {                          \
        threadgroup half sq[FA_QUERIES * D];                                                      \
        threadgroup float ss[FA_QUERIES * FA_KEYS];                                               \
        threadgroup half sp[FA_QUERIES * FA_KEYS];                                                \
        threadgroup float sd[64];                                                                 \
        g2s_flash<D>(q, kh, vh, out, p, bounds, sq, ss, sp, sd, g.x * FA_QUERIES, g.y, lane);     \
    }
G2S_FLASH(256)
G2S_FLASH(512)
