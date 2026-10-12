// Persistent text layer kernel for EmbeddingGemma 2 (Metal 4 library: the
// device-scope atomic fence needs -std=metal3.2 or later, and the base library
// stays at 3.1). The stage bodies come from gemma2.metal so the result is
// bit-identical to the GEMV route; the duplicate kernels this include brings
// into this library are never looked up here.
#include "../../metal/kernels/gemma2.metal"
// ---------------------------------------------------------------------------
// Persistent text layer (EI_GEMMA2_METAL_LAYER): the eleven dependent short-wave
// stages of one layer run in one dispatch of L.groups co-resident threadgroups
// (128 threads) separated by a device-atomic grid barrier instead of eleven
// dispatches. Every stage calls the same bodies as the standalone kernels with
// the same lane assignment, so the output is bit-identical to the GEMV route
// with scalar attention. The spin is capped: a group that cannot rendezvous
// (not co-resident) raises `fail` and every group returns early; the host then
// re-runs the forward on the dispatch route and turns the route off.
struct G2L {
    uint n, d, kv, groups, ntg, layer;
    uint wq, wk, wv, wo, wu, wg, wd, wig, wpr; // byte offsets of the Q8_0 projections
    uint wqn, wkn, wpa, wfn, wpf, wpn, wnext;  // byte offsets of the norm weights
};
#define G2_SPIN_CAP 200000u
inline bool g2_grid_sync(device atomic_uint *bar, device atomic_uint *fail, uint target, uint tid,
                         threadgroup uint *ok) {
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        atomic_fetch_add_explicit(bar, 1u, memory_order_relaxed);
        uint spins = 0, good = 1;
        while (atomic_load_explicit(bar, memory_order_relaxed) < target) {
            if (atomic_load_explicit(fail, memory_order_relaxed)) {
                good = 0;
                break;
            }
            if (++spins > G2_SPIN_CAP) {
                atomic_store_explicit(fail, 1u, memory_order_relaxed);
                good = 0;
                break;
            }
        }
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        *ok = good;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    return *ok != 0;
}
// Startup probe: can `groups` threadgroups rendezvous p.n times? Sets fail if not.
kernel void g2_layer_probe(device atomic_uint *bar [[buffer(0)]], device atomic_uint *fail [[buffer(1)]],
                           constant G2P &p [[buffer(2)]], uint gid [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint ok;
    (void)gid;
    for (uint i = 1; i <= p.n; ++i)
        if (!g2_grid_sync(bar, fail, i * p.m, tid, &ok))
            return;
}
// Stage parameter blocks S (constant G2P): 0 qkv_post, 1 attention, 2 post_add
// after attention (next = ffn_norm), 3 post_add after the FFN, 4 the layer's
// final post_add (scale, next = the next layer's attn_norm or output_norm).
kernel void g2_layer(device const uchar *W [[buffer(0)]], device float *x [[buffer(1)]],
                     device float *h [[buffer(2)]], device float *q [[buffer(3)]],
                     device float *k [[buffer(4)]], device float *v [[buffer(5)]],
                     device float *a [[buffer(6)]], device float *gate [[buffer(7)]],
                     device const float *ple [[buffer(8)]], device const uint2 *bounds [[buffer(9)]],
                     device const float2 *rope [[buffer(10)]],
                     device atomic_uint *bar [[buffer(11)]], device atomic_uint *fail [[buffer(12)]],
                     constant G2L &L [[buffer(13)]], constant G2P *S [[buffer(14)]],
                     uint gid [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint ok;
    const uint G = L.groups, n = L.n, d = L.d, kv = L.kv, ntg = L.ntg, slots = 4 + 2 * kv;
    uint stage = 0;
#define G2_SYNC()                                                                                  \
    if (!g2_grid_sync(bar, fail, (++stage) * G, tid, &ok))                                         \
        return;
    { // 1. q, k, v projections of h (16 rows x 8 tokens per group)
        G2InPlain in{h, 512};
        const uint rows = (4 * d + 2 * kv * d) / 16;
        for (uint vg = gid; vg < rows * ntg; vg += G)
            g2_qkv<G2InPlain, 8>(W + L.wq, W + L.wk, W + L.wv, in, q, k, v, 512, 4 * d, kv * d, n,
                                 uint2(vg % rows, vg / rows), sg, lane);
    }
    G2_SYNC();
    // 2. q/k norms, v norm and RoPE: one (token, head slot) per simdgroup
    for (uint it = gid * 4 + sg; it < n * slots; it += G * 4)
        g2_qkv_post_body(q, k, v, S[0], W + L.wqn, W + L.wkn, bounds, (device half *)q,
                         (device half *)q, rope, it / slots, it % slots, lane);
    G2_SYNC();
    // 3. attention: one (token, head) per simdgroup
    for (uint it = gid * 4 + sg; it < n * 4; it += G * 4)
        g2_attention_body(q, k, v, a, q, S[1], bounds, it / 4, it % 4, lane);
    G2_SYNC();
    { // 4. attention output projection (1 row per simdgroup: 128 row groups)
        G2InPlain in{a, 4 * d};
        for (uint vg = gid; vg < 128 * ntg; vg += G)
            g2_gemv_core<G2WQ8, 1, 8, 0>(W + L.wo, in, h, 4 * d, 512, n, (vg % 128) * 4 + sg,
                                         vg / 128 * 8, lane, a, 0);
    }
    G2_SYNC();
    // 5. x += rms(h) post_attention_norm; h = rms(x) ffn_norm
    for (uint it = gid * 4 + sg; it < n; it += G * 4)
        g2_post_add_body<512>(h, x, h, S[2], W + L.wpa, W + L.wfn, it, lane);
    G2_SYNC();
    { // 6. GeGLU: gate = gelu(h Wg) * (h Wu), 8 rows x 8 tokens per group
        G2InPlain in{h, 512};
        for (uint vg = gid; vg < 256 * ntg; vg += G)
            g2_geglu<G2InPlain, 8>(W + L.wu, W + L.wg, in, gate, 512, 2048, n,
                                   uint2(vg % 256, vg / 256), sg, lane);
    }
    G2_SYNC();
    { // 7. down projection
        G2InPlain in{gate, 2048};
        for (uint vg = gid; vg < 128 * ntg; vg += G)
            g2_gemv_core<G2WQ8, 1, 8, 0>(W + L.wd, in, h, 2048, 512, n, (vg % 128) * 4 + sg,
                                         vg / 128 * 8, lane, a, 0);
    }
    G2_SYNC();
    // 8. x += rms(h) post_ffw_norm
    for (uint it = gid * 4 + sg; it < n; it += G * 4)
        g2_post_add_body<512>(h, x, h, S[3], W + L.wpf, W + L.wpf, it, lane);
    G2_SYNC();
    { // 9. PLE input gate: h = gelu(x Wig) * ple[token, layer]
        G2InPlain in{x, 512};
        for (uint vg = gid; vg < 128 * ntg; vg += G)
            g2_gemv_core<G2WQ8, 1, 8, 1>(W + L.wig, in, h, 512, 512, n, (vg % 128) * 4 + sg,
                                         vg / 128 * 8, lane, ple, L.layer);
    }
    G2_SYNC();
    { // 10. PLE projection: a = h Wpr
        G2InPlain in{h, 512};
        for (uint vg = gid; vg < 128 * ntg; vg += G)
            g2_gemv_core<G2WQ8, 1, 8, 0>(W + L.wpr, in, a, 512, 512, n, (vg % 128) * 4 + sg,
                                         vg / 128 * 8, lane, h, 0);
    }
    G2_SYNC();
    // 11. x = (x + rms(a) post_norm) * scale; h = rms(x) next_norm
    for (uint it = gid * 4 + sg; it < n; it += G * 4)
        g2_post_add_body<512>(a, x, h, S[4], W + L.wpn, W + L.wnext, it, lane);
#undef G2_SYNC
}
