#include <metal_stdlib>
using namespace metal;
// Fixed-profile kernels. Activations stay FP32; GGUF Q8/BF16 weights stay packed.
struct G2P {
    uint n, k, m, type, heads, kv, d, window, cols, mode, layer;
    float scale, lo, hi;
};
struct G2Q8 {
    half d;
    char qs[32];
};
inline float g2w(device const uchar *w, uint type, uint i) {
    if (type == 0)
        return ((device const float *)w)[i];
    if (type == 30)
        return as_type<float>(uint(((device const ushort *)w)[i]) << 16);
    if (type == 1)
        return float(((device const half *)w)[i]);
    device const G2Q8 *b = (device const G2Q8 *)w;
    return float(b[i / 32].d) * float(b[i / 32].qs[i % 32]);
}
inline float g2gelu(float x) {
    return 0.5f * x * (1 + tanh(0.7978845608f * x * (1 + 0.044715f * x * x)));
}
kernel void g2_mm(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]],
                  device float *y [[buffer(2)]], constant G2P &p [[buffer(3)]],
                  uint3 g [[threadgroup_position_in_grid]],
                  uint lane [[thread_index_in_simdgroup]]) {
    float sum = 0;
    for (uint z = lane; z < p.k; z += 32)
        sum += g2w(w, p.type, g.x * p.k + z) * x[g.y * p.k + z];
    sum = simd_sum(sum);
    if (lane == 0)
        y[g.y * p.m + g.x] = sum;
}
// Weight-stationary GEMV for short packed waves: each lane decodes 16 weights of
// R rows once and applies them to NT tokens. K is a multiple of 32, M of 16.
struct G2WQ8 {
    packed_char4 q[4];
    float s;
    void load(device const uchar *w, uint i) {
        device const G2Q8 *b = (device const G2Q8 *)w + i / 32;
        device const packed_char4 *qs = (device const packed_char4 *)(b->qs + i % 32);
        s = float(b->d);
        for (uint j = 0; j < 4; ++j)
            q[j] = qs[j];
    }
    float dot4(float4 x0, float4 x1, float4 x2, float4 x3) const {
        return s * (dot(float4(char4(q[0])), x0) + dot(float4(char4(q[1])), x1) +
                    dot(float4(char4(q[2])), x2) + dot(float4(char4(q[3])), x3));
    }
};
struct G2WBF16 {
    ushort4 q[4];
    void load(device const uchar *w, uint i) {
        device const ushort4 *v = (device const ushort4 *)((device const ushort *)w + i);
        for (uint j = 0; j < 4; ++j)
            q[j] = v[j];
    }
    static float4 f(ushort4 v) { return as_type<float4>(uint4(v) << 16); }
    float dot4(float4 x0, float4 x1, float4 x2, float4 x3) const {
        return dot(f(q[0]), x0) + dot(f(q[1]), x1) + dot(f(q[2]), x2) + dot(f(q[3]), x3);
    }
};
struct G2WF16 {
    half4 q[4];
    void load(device const uchar *w, uint i) {
        device const half4 *v = (device const half4 *)((device const half *)w + i);
        for (uint j = 0; j < 4; ++j)
            q[j] = v[j];
    }
    float dot4(float4 x0, float4 x1, float4 x2, float4 x3) const {
        return dot(float4(q[0]), x0) + dot(float4(q[1]), x1) + dot(float4(q[2]), x2) +
               dot(float4(q[3]), x3);
    }
};
struct G2WF32 {
    float4 q[4];
    void load(device const uchar *w, uint i) {
        device const float4 *v = (device const float4 *)((device const float *)w + i);
        for (uint j = 0; j < 4; ++j)
            q[j] = v[j];
    }
    float dot4(float4 x0, float4 x1, float4 x2, float4 x3) const {
        return dot(q[0], x0) + dot(q[1], x1) + dot(q[2], x2) + dot(q[3], x3);
    }
};
// GEMV input rows: plain fp32 activations.
struct G2InPlain {
    device const float *x;
    uint k;
    void prepare(uint, uint, uint) {}
    void load(uint, uint token, uint e, thread float4 &x0, thread float4 &x1, thread float4 &x2,
              thread float4 &x3) const {
        device const float4 *xp = (device const float4 *)(x + token * k + e);
        x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
    }
};
// MODE 0 stores y; MODE 1 stores gelu(y) * ple[token, layer, row] (PLE input gate).
template <typename W, uint R, uint NT, uint MODE, typename IN>
inline void g2_gemv_core(device const uchar *w, thread IN &in, device float *y, uint k, uint m,
                         uint n, uint row0, uint t0, uint lane, device const float *ple,
                         uint layer) {
    const uint nt = min(NT, n - t0);
    in.prepare(t0, nt, lane);
    float acc[R][NT];
    for (uint r = 0; r < R; ++r)
        for (uint t = 0; t < NT; ++t)
            acc[r][t] = 0;
    for (uint e = lane * 16; e < k; e += 512) {
        W wr[R];
        for (uint r = 0; r < R; ++r)
            wr[r].load(w, (row0 + r) * k + e);
        for (uint t = 0; t < NT; ++t) {
            if (t >= nt)
                break;
            float4 x0, x1, x2, x3;
            in.load(t, t0 + t, e, x0, x1, x2, x3);
            for (uint r = 0; r < R; ++r)
                acc[r][t] += wr[r].dot4(x0, x1, x2, x3);
        }
    }
    for (uint r = 0; r < R; ++r)
        for (uint t = 0; t < NT; ++t) {
            float v = simd_sum(acc[r][t]);
            if (lane == (r * NT + t) % 32 && t < nt) {
                uint row = row0 + r, token = t0 + t;
                if (MODE == 1)
                    v = g2gelu(v) * ple[(token * 24 + layer) * 512 + row];
                y[token * m + row] = v;
            }
        }
}
template <typename W, uint R, uint NT>
kernel void g2_gemv(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]],
                    device float *y [[buffer(2)]], constant G2P &p [[buffer(3)]],
                    uint2 g [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    G2InPlain in{x, p.k};
    g2_gemv_core<W, R, NT, 0>(w, in, y, p.k, p.m, p.n, (g.x * 4 + sg) * R, g.y * NT, lane, x, 0);
}
#define G2_GEMV(T, name)                                                                        \
    template [[host_name(name)]] kernel void g2_gemv<T, 4, 8>(                                  \
        device const uchar *, device const float *, device float *, constant G2P &, uint2,     \
        uint, uint);
G2_GEMV(G2WQ8, "g2_gemv_q8")
G2_GEMV(G2WBF16, "g2_gemv_bf16")
G2_GEMV(G2WF16, "g2_gemv_f16")
G2_GEMV(G2WF32, "g2_gemv_f32")
// inp_gate with the PLE gate in the epilogue (buffer 4 = ple, p.layer = layer).
kernel void g2_gemv_ple(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]], constant G2P &p [[buffer(3)]],
                        device const float *ple [[buffer(4)]],
                        uint2 g [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    G2InPlain in{x, p.k};
    g2_gemv_core<G2WQ8, 4, 8, 1>(w, in, y, p.k, p.m, p.n, (g.x * 4 + sg) * 4, g.y * 8, lane, ple,
                                 p.layer);
}
// Q, K and V in one dispatch over the concatenated row space: p.m = Q rows,
// p.d = K rows (= V rows). Weights in buffers 0/4/5, outputs in 2/6/7.
template <typename IN>
inline void g2_qkv(device const uchar *wq, device const uchar *wk, device const uchar *wv,
                   thread IN &in, device float *q, device float *k, device float *v,
                   constant G2P &p, uint2 g, uint sg, uint lane) {
    uint row = (g.x * 4 + sg) * 4;
    device const uchar *w = wq;
    device float *y = q;
    uint m = p.m;
    if (row >= p.m + p.d) {
        w = wv, y = v, m = p.d, row -= p.m + p.d;
    } else if (row >= p.m) {
        w = wk, y = k, m = p.d, row -= p.m;
    }
    g2_gemv_core<G2WQ8, 4, 8, 0>(w, in, y, p.k, m, p.n, row, g.y * 8, lane, q, 0);
}
kernel void g2_gemv_qkv(device const uchar *wq [[buffer(0)]], device const float *x [[buffer(1)]],
                        device float *q [[buffer(2)]], constant G2P &p [[buffer(3)]],
                        device const uchar *wk [[buffer(4)]], device const uchar *wv [[buffer(5)]],
                        device float *k [[buffer(6)]], device float *v [[buffer(7)]],
                        uint2 g [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    G2InPlain in{x, p.k};
    g2_qkv(wq, wk, wv, in, q, k, v, p, g, sg, lane);
}
// Up and gate rows together; stores gelu_tanh(gate) * up. Weights 0 (up) / 4 (gate).
template <typename IN>
inline void g2_geglu(device const uchar *wu, device const uchar *wg, thread IN &in, device float *y,
                     constant G2P &p, uint2 g, uint sg, uint lane) {
    const uint row0 = (g.x * 4 + sg) * 2, t0 = g.y * 8, nt = min(8u, p.n - t0);
    in.prepare(t0, nt, lane);
    float au[2][8], ag[2][8];
    for (uint r = 0; r < 2; ++r)
        for (uint t = 0; t < 8; ++t)
            au[r][t] = ag[r][t] = 0;
    for (uint e = lane * 16; e < p.k; e += 512) {
        G2WQ8 u[2], gt[2];
        for (uint r = 0; r < 2; ++r) {
            u[r].load(wu, (row0 + r) * p.k + e);
            gt[r].load(wg, (row0 + r) * p.k + e);
        }
        for (uint t = 0; t < 8; ++t) {
            if (t >= nt)
                break;
            float4 x0, x1, x2, x3;
            in.load(t, t0 + t, e, x0, x1, x2, x3);
            for (uint r = 0; r < 2; ++r) {
                au[r][t] += u[r].dot4(x0, x1, x2, x3);
                ag[r][t] += gt[r].dot4(x0, x1, x2, x3);
            }
        }
    }
    for (uint r = 0; r < 2; ++r)
        for (uint t = 0; t < 8; ++t) {
            float a = simd_sum(au[r][t]), b = simd_sum(ag[r][t]);
            if (lane == (r * 8 + t) % 32 && t < nt)
                y[(t0 + t) * p.m + row0 + r] = g2gelu(b) * a;
        }
}
kernel void g2_gemv_geglu(device const uchar *wu [[buffer(0)]], device const float *x [[buffer(1)]],
                          device float *y [[buffer(2)]], constant G2P &p [[buffer(3)]],
                          device const uchar *wg [[buffer(4)]],
                          uint2 g [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    G2InPlain in{x, p.k};
    g2_geglu(wu, wg, in, y, p, g, sg, lane);
}
kernel void g2_norm(device const float *x [[buffer(0)]], device const uchar *w [[buffer(1)]],
                    device float *y [[buffer(2)]], constant G2P &p [[buffer(3)]],
                    uint row [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]]) {
    float mean = 0, ss = 0;
    if (p.mode == 2) {
        for (uint z = lane; z < p.d; z += 32)
            mean += x[row * p.d + z];
        mean = simd_sum(mean) / p.d;
    }
    for (uint z = lane; z < p.d; z += 32) {
        float v = x[row * p.d + z] - mean;
        ss += v * v;
    }
    float inv = rsqrt(simd_sum(ss) / p.d + 1e-6f);
    for (uint z = lane; z < p.d; z += 32) {
        float v = (x[row * p.d + z] - mean) * inv * (p.mode ? g2w(w, p.type, z) : 1.0f);
        y[row * p.d + z] = p.mode == 2 ? max(v, 0.0f) : v;
    }
}
kernel void g2_element(device const float *x [[buffer(0)]], device const float *b [[buffer(1)]],
                       device float *y [[buffer(2)]], constant G2P &p [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= p.n)
        return;
    float v = x[i];
    switch (p.mode) {
    case 0:
        v = (v + b[i]) * p.scale;
        break;
    case 1:
        v = g2gelu(v) * b[i];
        break;
    case 2:
        v = v / (1 + exp(-1.702f * v)) * b[i];
        break;
    case 3:
        v = v / (1 + exp(-v));
        break;
    case 4:
        v = g2gelu(v) * b[(i / 512 * 24 + p.layer) * 512 + i % 512];
        break;
    case 5:
        v = clamp(v, p.lo, p.hi);
        break;
    case 6:
        v = v + b[i] * p.scale;
        break;
    case 7:
        v *= p.scale;
        break;
    case 8:
        v = b[(i / 1024) * 2048 + i % 1024] / (1 + exp(-b[(i / 1024) * 2048 + 1024 + i % 1024]));
        break;
    }
    y[i] = v;
}
kernel void g2_embed(device const uchar *w [[buffer(0)]], device const int *ids [[buffer(1)]],
                     device float *x [[buffer(2)]], constant G2P &p [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < p.n * 512)
        x[i] = g2w(w, 8, uint(ids[i / 512]) * 512 + i % 512) * sqrt(512.0f);
}
kernel void g2_rope(device float *x [[buffer(0)]], constant G2P &p [[buffer(1)]],
                    device const uint2 *bounds [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n * p.heads * p.d / 2)
        return;
    uint z = i % (p.d / 2), h = i / (p.d / 2), t = h / p.heads, off = h * p.d;
    if (p.layer)
        t -= bounds[t].x;
    float theta;
    if (p.cols) {
        uint axis = z / 16;
        z %= 16;
        off += axis * 32;
        theta = float(axis ? t / p.cols : t % p.cols) * pow(100.0f, -float(z) / 16);
    } else
        theta = float(t) * pow(p.scale, -2.0f * float(z) / p.d);
    uint halfd = p.cols ? 16 : p.d / 2;
    float co = cos(theta), si = sin(theta), a = x[off + z], b = x[off + z + halfd];
    x[off + z] = a * co - b * si;
    x[off + z + halfd] = a * si + b * co;
}
// One SIMD group per query/head, online softmax, no quadratic score allocation.
kernel void g2_attention(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
                         device const float *v [[buffer(2)]], device float *out [[buffer(3)]],
                         device const float *rpe [[buffer(4)]], constant G2P &p [[buffer(5)]],
                         device const uint2 *bounds [[buffer(6)]],
                         uint2 g [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint t = g.x, h = g.y, kh = h / (p.heads / p.kv);
    float qs[16], acc[16];
    for (uint j = 0; j < p.d / 32; j++) {
        qs[j] = q[(t * p.heads + h) * p.d + lane + j * 32];
        acc[j] = 0;
    }
    uint lo = p.window && t > p.window ? t - p.window : 0,
         hi = p.window ? min(p.n, t + p.window + 1) : p.n;
    if (p.layer) {
        lo = max(lo, bounds[t].x);
        hi = min(hi, bounds[t].y);
    }
    if (p.mode) {
        lo = t > 11 ? t - 11 : 0;
        hi = t + 1;
    }
    float mx = -INFINITY, total = 0;
    for (uint s = lo; s < hi; s++) {
        float dot = 0;
        for (uint j = 0; j < p.d / 32; j++) {
            uint z = lane + j * 32;
            float kval = k[(s * p.kv + kh) * p.d + z];
            if (p.mode)
                kval += rpe[((12 - t + s) * p.heads + h) * p.d + z];
            dot += qs[j] * kval;
        }
        dot = simd_sum(dot);
        if (p.mode)
            dot = 50 * tanh(dot / 50);
        float next = max(mx, dot), a = exp(mx - next), b = exp(dot - next);
        total = total * a + b;
        for (uint j = 0; j < p.d / 32; j++)
            acc[j] = acc[j] * a + b * v[(s * p.kv + kh) * p.d + lane + j * 32];
        mx = next;
    }
    for (uint j = 0; j < p.d / 32; j++)
        out[(t * p.heads + h) * p.d + lane + j * 32] = acc[j] / total;
}
kernel void g2_pool(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                    device const uint2 *bounds [[buffer(2)]], constant G2P &p [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= p.n * 768)
        return;
    uint z = i % 768;
    uint2 range = bounds[i / 768];
    float sum = 0;
    for (uint t = range.x; t < range.y; t++)
        sum += x[t * 768 + z] / float(range.y - range.x);
    out[i] = sum;
}
kernel void g2_patch(device const uchar *rgb [[buffer(0)]], device float *patch [[buffer(1)]],
                     constant G2P &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n * 768)
        return;
    uint t = i / 768, z = i % 768, c = z / 256, y = (z % 256) / 16, x = z % 16;
    patch[i] = float(half(
        2.0f * float(rgb[((t / p.cols * 16 + y) * p.cols * 16 + t % p.cols * 16 + x) * 3 + c]) /
            255 -
        1));
}
kernel void g2_position(device float *x [[buffer(0)]], device const float *pos [[buffer(1)]],
                        constant G2P &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n * 768) {
        uint t = i / 768, z = i % 768;
        x[i] += pos[(t % p.cols) * 768 + z] + pos[(10240 + t / p.cols) * 768 + z];
    }
}
kernel void g2_spatial_pool(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                            constant G2P &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n * 768)
        return;
    uint t = i / 768, z = i % 768, cols = p.cols / 3;
    float sum = 0;
    for (uint yy = 0; yy < 3; yy++)
        for (uint xx = 0; xx < 3; xx++)
            sum +=
                x[((t / cols * 3 + yy) * p.cols + t % cols * 3 + xx) * 768 + z] / 9 * sqrt(768.0f);
    y[i] = sum;
}
kernel void g2_conv(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                    device float *y [[buffer(2)]], constant G2P &p [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    uint of = (p.k + 1) / 2, ot = (p.n + 1) / 2;
    if (i >= ot * of * p.m)
        return;
    uint o = i % p.m, f = (i / p.m) % of, t = i / (p.m * of);
    float sum = 0;
    for (int ky = 0; ky < 3; ky++)
        for (int kx = 0; kx < 3; kx++) {
            int ti = int(t * 2) + ky - 1, fi = int(f * 2) + kx - 1;
            if (ti < 0 || ti >= int(p.n) || fi < 0 || fi >= int(p.k))
                continue;
            for (uint c = 0; c < p.d; c++)
                sum += x[(ti * p.k + fi) * p.d + c] * w[(o * p.d + c) * 9 + ky * 3 + kx];
        }
    y[i] = sum;
}
kernel void g2_audio_scale(device float *q [[buffer(0)]], device float *k [[buffer(1)]],
                           device const float *w [[buffer(2)]], constant G2P &p [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
    if (i < p.n * 1024) {
        q[i] *= w[i % 128] / (sqrt(128.0f) * log(2.0f));
        k[i] *= log(1 + exp(1.0f)) / log(2.0f);
    }
}
kernel void g2_depthwise(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                         device float *y [[buffer(2)]], constant G2P &p [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= p.n * 1024)
        return;
    uint t = i / 1024, z = i % 1024;
    float sum = 0;
    for (uint j = 0; j < 5; j++)
        if (t + j >= 4)
            sum += x[(t + j - 4) * 1024 + z] * w[z * 5 + j];
    y[i] = sum;
}
kernel void g2_bias(device float *x [[buffer(0)]], device const float *b [[buffer(1)]],
                    constant G2P &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n * p.d)
        x[i] += b[i % p.d];
}
kernel void g2_half(device const float *x [[buffer(0)]], device half *y [[buffer(1)]],
                    constant G2P &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.m * p.d)
        y[i] = i < p.n * p.d ? half(x[i]) : 0.0h;
}
// Per (token, head slot): weighted RMS of q/k heads, unweighted RMS of v heads,
// then NEOX RoPE of q/k in registers (pair z, z + d/2 lives in the same lane).
// Slots: [0, heads) q, [heads, heads + kv) k, [heads + kv, heads + 2kv) v.
// Weights: 4 = q norm, 5 = k norm (type p.type). p.scale = RoPE base.
// p.mode != 0 also writes fp16 K/V copies (buffers 7/8) for tensor flash; the
// grid then covers 64 pad rows past p.n, which are zeroed.
kernel void g2_qkv_post(device float *q [[buffer(0)]], device float *k [[buffer(1)]],
                        device float *v [[buffer(2)]], constant G2P &p [[buffer(3)]],
                        device const uchar *wq [[buffer(4)]], device const uchar *wk [[buffer(5)]],
                        device const uint2 *bounds [[buffer(6)]], device half *kh [[buffer(7)]],
                        device half *vh [[buffer(8)]], device const float2 *rope [[buffer(9)]],
                        uint2 g [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    uint t = g.x, slot = g.y, nd = p.d / 32;
    if (t >= p.n) {
        if (slot >= p.heads) {
            device half *dst = slot < p.heads + p.kv ? kh + (t * p.kv + slot - p.heads) * p.d
                                                     : vh + (t * p.kv + slot - p.heads - p.kv) * p.d;
            for (uint j = 0; j < nd; ++j)
                dst[lane + j * 32] = 0.0h;
        }
        return;
    }
    device float *row;
    device const uchar *w = nullptr;
    if (slot < p.heads) {
        row = q + (t * p.heads + slot) * p.d;
        w = wq;
    } else if (slot < p.heads + p.kv) {
        row = k + (t * p.kv + slot - p.heads) * p.d;
        w = wk;
    } else
        row = v + (t * p.kv + slot - p.heads - p.kv) * p.d;
    float r[16], ss = 0;
    for (uint j = 0; j < nd; ++j) {
        r[j] = row[lane + j * 32];
        ss += r[j] * r[j];
    }
    float inv = rsqrt(simd_sum(ss) / p.d + 1e-6f);
    for (uint j = 0; j < nd; ++j)
        r[j] *= inv * (w ? g2w(w, p.type, lane + j * 32) : 1.0f);
    if (w) {
        uint pos = t - bounds[t].x, half_j = nd / 2;
        for (uint j = 0; j < half_j; ++j) {
            uint z = lane + j * 32;
            // p.cols: precomputed (cos, sin) table [pos][d / 2] for this base.
            float co, si;
            if (p.cols) {
                float2 cs = rope[pos * (p.d / 2) + z];
                co = cs.x, si = cs.y;
            } else {
                float theta = float(pos) * pow(p.scale, -2.0f * float(z) / p.d);
                co = cos(theta), si = sin(theta);
            }
            float a = r[j], b = r[j + half_j];
            r[j] = a * co - b * si;
            r[j + half_j] = a * si + b * co;
        }
    }
    for (uint j = 0; j < nd; ++j)
        row[lane + j * 32] = r[j];
    if (p.mode && slot >= p.heads) {
        device half *dst = slot < p.heads + p.kv ? kh + (t * p.kv + slot - p.heads) * p.d
                                                 : vh + (t * p.kv + slot - p.heads - p.kv) * p.d;
        for (uint j = 0; j < nd; ++j)
            dst[lane + j * 32] = half(r[j]);
    }
}
// Gemma sandwich per token (width 512): x = (x + rms(h) * w_post) * p.scale.
// If p.mode != 0 also y = rms(x) * w_next (type p.mode - 1). Buffers: 0 h,
// 1 x, 2 y, 3 params, 4 w_post (type p.type), 5 w_next.
kernel void g2_post_add(device const float *h [[buffer(0)]], device float *x [[buffer(1)]],
                        device float *y [[buffer(2)]], constant G2P &p [[buffer(3)]],
                        device const uchar *wp [[buffer(4)]], device const uchar *wn [[buffer(5)]],
                        uint t [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    device const float *hr = h + t * 512;
    device float *xr = x + t * 512;
    float a[16], ss = 0;
    for (uint j = 0; j < 16; ++j) {
        a[j] = hr[lane + j * 32];
        ss += a[j] * a[j];
    }
    float inv = rsqrt(simd_sum(ss) / 512 + 1e-6f), ss2 = 0;
    for (uint j = 0; j < 16; ++j) {
        uint z = lane + j * 32;
        a[j] = (xr[z] + a[j] * inv * g2w(wp, p.type, z)) * p.scale;
        xr[z] = a[j];
        ss2 += a[j] * a[j];
    }
    if (p.mode) {
        float inv2 = rsqrt(simd_sum(ss2) / 512 + 1e-6f);
        for (uint j = 0; j < 16; ++j) {
            uint z = lane + j * 32;
            y[t * 512 + z] = a[j] * inv2 * g2w(wn, p.mode - 1, z);
        }
    }
}
