#include "gemma2_metal.h"
#include "kernels.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach-o/getsect.h>
#import <mach-o/ldsyms.h>
#include <math.h>
#include <time.h>

typedef struct {
    uint32_t n, k, m, type, heads, kv, d, window, cols, mode, layer;
    float scale, lo, hi;
} G2P;
_Static_assert(sizeof(G2P) == 56, "Metal parameter layout mismatch");
// Persistent layer kernel parameters (g2_layer): byte offsets into textWeights.
typedef struct {
    uint32_t n, d, kv, groups, ntg, layer;
    uint32_t wq, wk, wv, wo, wu, wg, wd, wig, wpr;
    uint32_t wqn, wkn, wpa, wfn, wpf, wpn, wnext;
} G2L;
_Static_assert(sizeof(G2L) == 88, "Metal layer parameter layout mismatch");
@class EIGemma2Metal;
static bool layer_probe(EIGemma2Metal *c);
@interface EIGemma2Metal : NSObject {
  @public
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
    NSMutableDictionary<NSString *, id<MTLBuffer>> *workspace;
    id<MTLBuffer> textWeights, mediaWeights;
    const ei_gguf *textModel, *mediaModel;
    bool tensorMM, flash, gemv, fused, tensor2;
    bool flash2;
    bool flash64; // EI_GEMMA2_METAL_FLASH64: padded tensor flash for 64-wide (vision) heads
    bool vfused;  // EI_GEMMA2_METAL_VFUSED: fused vision layer (qkv post, GeGLU epilogue, post_add)
    bool afused;  // EI_GEMMA2_METAL_AFUSED: audio clamps folded into producers/GEMM epilogues
    int tensor2Min, tensor2Wide, flash2Min;
    NSString *flashTile; // EI_GEMMA2_METAL_FLASH2_TILE=k64 selects the 64-key tile
    bool epi; // fused GeGLU / PLE epilogues on the wide GEMM (EI_GEMMA2_METAL_NOEPI=1 disables)
    bool kvHalfReady; // fp16 K/V for this layer already written by g2_qkv_post
    bool layer;       // EI_GEMMA2_METAL_LAYER: persistent per-layer kernel on short waves
    int layerGroups;  // EI_GEMMA2_METAL_LAYER_GROUPS: co-resident threadgroups per layer
    id<MTLBuffer> rope[2]; // fused route: (cos, sin) for base 1e4/d 256 and 1e6/d 512
    // fp16 row-major copies of Q8_0/BF16 text weights for g2t_gemm: name -> buffer.
    NSMutableDictionary<NSString *, id<MTLBuffer>> *packed;
    double encodeStart;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> enc;
}
@end
@implementation EIGemma2Metal
@end
static id<MTLBuffer> buffer(EIGemma2Metal *c, NSString *key, size_t floats) {
    id<MTLBuffer> b = c->workspace[key];
    if (!b || b.length < floats * 4) {
        // Slack lets 64-token tensor tiles read past the last token row.
        b = [c->device newBufferWithLength:floats * 4 + 64 * 4096 * 4
                                   options:MTLResourceStorageModeShared];
        if (!b)
            ei_die("EmbeddingGemma 2 Metal workspace allocation failed (%zu bytes)", floats * 4);
        c->workspace[key] = b;
    }
    return b;
}
static void pipeline(EIGemma2Metal *c, NSString *name) {
    [c->enc setComputePipelineState:c->pipelines[name]];
}
static void bind(EIGemma2Metal *c, id<MTLBuffer> b, int i) {
    [c->enc setBuffer:b offset:0 atIndex:i];
}
static void bind_at(EIGemma2Metal *c, id<MTLBuffer> b, size_t floats, int i) {
    [c->enc setBuffer:b offset:floats * 4 atIndex:i];
}
static void params(EIGemma2Metal *c, G2P p, int i) {
    [c->enc setBytes:&p length:sizeof p atIndex:i];
}
static const ei_tensor *weight(EIGemma2Metal *c, const ei_gguf *g, const char *name, int slot) {
    const ei_tensor *w = ei_gguf_tensor(g, name, true);
    [c->enc setBuffer:g == c->textModel ? c->textWeights : c->mediaWeights
               offset:w->offset
              atIndex:slot];
    return w;
}
static void threads(EIGemma2Metal *c, size_t n) {
    [c->enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
static void groups(EIGemma2Metal *c, size_t x, size_t y) {
    [c->enc dispatchThreadgroups:MTLSizeMake(x, y, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}
// Diagnostic only: EI_GEMMA2_METAL_SKIP=attn,mm,qkv,geglu,ple,post,qknorm drops
// those dispatches so their share of GPU time can be measured. Output is invalid.
static bool skipped(const char *kind) {
    const char *list = getenv("EI_GEMMA2_METAL_SKIP");
    return list && strstr(list, kind);
}
static void begin(EIGemma2Metal *c) {
    c->encodeStart = now_ms();
    c->command = [c->queue commandBuffer];
    c->enc = [c->command computeCommandEncoder];
}
static bool finish(EIGemma2Metal *c, char *err, size_t len) {
    [c->enc endEncoding];
    c->enc = nil;
    double encoded = now_ms();
    [c->command commit];
    [c->command waitUntilCompleted];
    if (getenv("EI_GEMMA2_METAL_PROFILE"))
        fprintf(stderr, "gemma2 Metal encode %.3f ms, GPU %.3f ms, total %.3f ms\n",
                encoded - c->encodeStart, (c->command.GPUEndTime - c->command.GPUStartTime) * 1000,
                now_ms() - c->encodeStart);
    bool ok = c->command.status == MTLCommandBufferStatusCompleted;
    if (!ok)
        snprintf(err, len, "EmbeddingGemma 2 Metal: %s",
                 c->command.error.localizedDescription.UTF8String);
    c->command = nil;
    return ok;
}
// y = x W^T; with lo < hi the result is clamped to [lo, hi] in the wide GEMM's
// epilogue (returns true) or left for the caller to clamp (false).
static bool mm_clamp(EIGemma2Metal *c, const ei_gguf *g, const char *name, id<MTLBuffer> x, int n,
                     id<MTLBuffer> y, float lo, float hi) {
    if (skipped("mm"))
        return true;
    pipeline(c, @"g2_mm");
    const ei_tensor *w = weight(c, g, name, 0);
    G2P p = {.n = n, .k = (uint32_t)w->ne[0], .m = (uint32_t)w->ne[1], .type = w->type};
    if (!strcmp(name, "v.patch_embd.weight"))
        p.k = p.m = 768;
    bool wide = c->tensor2 && n >= c->tensor2Min;
    id<MTLBuffer> packed = wide ? c->packed[@(name)] : nil;
    bool clamped = packed && lo < hi;
    if (clamped) {
        p.lo = lo;
        p.hi = hi;
    }
    bind(c, x, 1);
    bind(c, y, 2);
    params(c, p, 3);
    if (packed) {
        int tm = n >= c->tensor2Wide ? 64 : 32;
        pipeline(c, clamped ? (tm == 64 ? @"g2t_gemm_clamp_64" : @"g2t_gemm_clamp_32")
                            : (tm == 64 ? @"g2t_gemm_64" : @"g2t_gemm_32"));
        [c->enc setBuffer:packed offset:0 atIndex:0];
        [c->enc dispatchThreadgroups:MTLSizeMake((n + tm - 1) / tm, p.m / 64, 1)
               threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else if ((c->tensorMM && n >= 32) || wide) {
        // Unpacked weights (BF16 PLE projection, F32 media) keep the tile route.
        pipeline(c, @"g2_tensor_mm");
        [c->enc dispatchThreadgroups:MTLSizeMake((n + 63) / 64, p.m / 64, 1)
               threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else if (c->gemv && p.m % 16 == 0 && p.k % 32 == 0) {
        // 4 simdgroups x 4 rows per threadgroup, 8 tokens per weight decode
        // (16 tokens per decode measured 1.3x slower at 16-39 tokens, 2026-10-10).
        pipeline(c, p.type == EI_T_Q8_0  ? @"g2_gemv_q8"
                    : p.type == EI_T_BF16 ? @"g2_gemv_bf16"
                    : p.type == EI_T_F16  ? @"g2_gemv_f16"
                                          : @"g2_gemv_f32");
        [c->enc dispatchThreadgroups:MTLSizeMake(p.m / 16, (n + 7) / 8, 1)
               threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else
        groups(c, p.m, n);
    return clamped;
}
static void mm(EIGemma2Metal *c, const ei_gguf *g, const char *name, id<MTLBuffer> x, int n,
               id<MTLBuffer> y) {
    mm_clamp(c, g, name, x, n, y, 0, 0);
}
// RMS norm rows; silu applies SiLU after the weight, lo < hi clamps the result.
static void norm_c(EIGemma2Metal *c, const ei_gguf *g, const char *name, id<MTLBuffer> x, int n,
                   int d, id<MTLBuffer> y, int mode, bool silu, float lo, float hi) {
    pipeline(c, @"g2_norm");
    bind(c, x, 0);
    bind(c, y, 2);
    G2P p = {.n = n, .d = d, .mode = name ? (uint32_t)mode : 0, .cols = silu, .lo = lo, .hi = hi};
    if (name)
        p.type = weight(c, g, name, 1)->type;
    else
        bind(c, x, 1);
    params(c, p, 3);
    groups(c, n, 1);
}
static void norm(EIGemma2Metal *c, const ei_gguf *g, const char *name, id<MTLBuffer> x, int n,
                 int d, id<MTLBuffer> y, int mode) {
    norm_c(c, g, name, x, n, d, y, mode, false, 0, 0);
}
static void element_c(EIGemma2Metal *c, id<MTLBuffer> x, id<MTLBuffer> b, id<MTLBuffer> y,
                      size_t n, int mode, float scale, int layer, float lo, float hi) {
    pipeline(c, @"g2_element");
    bind(c, x, 0);
    bind(c, b, 1);
    bind(c, y, 2);
    params(c,
           (G2P){.n = (uint32_t)n, .mode = mode, .scale = scale, .layer = layer, .lo = lo, .hi = hi},
           3);
    threads(c, n);
}
static void element(EIGemma2Metal *c, id<MTLBuffer> x, id<MTLBuffer> b, id<MTLBuffer> y, size_t n,
                    int mode, float scale, int layer) {
    element_c(c, x, b, y, n, mode, scale, layer, 0, 0);
}
static void rotary(EIGemma2Metal *c, id<MTLBuffer> x, int n, int heads, int d, float base, int cols,
                   id<MTLBuffer> bounds) {
    pipeline(c, @"g2_rope");
    bind(c, x, 0);
    bind(c, bounds ? bounds : x, 2);
    params(
        c,
        (G2P){.n = n, .heads = heads, .d = d, .scale = base, .cols = cols, .layer = bounds ? 1 : 0},
        1);
    threads(c, (size_t)n * heads * d / 2);
}
static void attention_c(EIGemma2Metal *c, id<MTLBuffer> q, id<MTLBuffer> k, id<MTLBuffer> v,
                        id<MTLBuffer> a, int n, int heads, int kv, int d, int window,
                        id<MTLBuffer> rpe, id<MTLBuffer> bounds, float lo, float hi) {
    if (skipped("attn"))
        return;
    if (c->flash2 && n >= c->flash2Min && !rpe && (d == 256 || d == 512)) {
        // fp16 K/V with 128 zero rows past the last token for whole key blocks.
        int padded = (n + 63) / 64 * 64 + 128;
        id<MTLBuffer> kh = buffer(c, @"kh", (size_t)padded * kv * d / 2),
                      vh = buffer(c, @"vh_half", (size_t)padded * kv * d / 2);
        bool ready = c->kvHalfReady;
        c->kvHalfReady = false;
        if (!ready) {
        pipeline(c, @"g2_half");
        bind(c, k, 0);
        bind(c, kh, 1);
        params(c, (G2P){.n = n, .m = padded, .d = kv * d}, 2);
        threads(c, (size_t)padded * kv * d);
        bind(c, v, 0);
        bind(c, vh, 1);
        threads(c, (size_t)padded * kv * d);
        }
        NSString *flash = [NSString stringWithFormat:@"g2t_flash_%d%@", d, c->flashTile];
        if (!c->pipelines[flash])
            flash = [NSString stringWithFormat:@"g2t_flash_%d", d];
        // Tile geometry follows the kernel name: _q16 = 16 queries, _n8 = 8 simdgroups.
        int tq = [flash containsString:@"_q16"] ? 16 : [flash containsString:@"_q64"] ? 64 : 32;
        int nsg = [flash containsString:@"n16"] ? 16 : [flash containsString:@"n8"] ? 8 : 4;
        pipeline(c, flash);
        bind(c, q, 0);
        bind(c, kh, 1);
        bind(c, vh, 2);
        bind(c, a, 3);
        params(c,
               (G2P){.n = n, .heads = heads, .kv = kv, .d = d, .window = window,
                     .layer = bounds ? 1 : 0},
               4);
        bind(c, bounds ? bounds : q, 5);
        [c->enc dispatchThreadgroups:MTLSizeMake((n + tq - 1) / tq, heads, 1)
               threadsPerThreadgroup:MTLSizeMake(32 * nsg, 1, 1)];
        return;
    }
    // Vision (d = 64): the tensor flash kernel over 128-wide zero-padded fp16
    // K/V rows (no 64-wide instantiation compiles on the M5 driver).
    if (c->flash2 && c->flash64 && d == 64 && !rpe && n >= c->flash2Min) {
        int padded = (n + 63) / 64 * 64 + 128;
        id<MTLBuffer> kh = buffer(c, @"kh", (size_t)padded * kv * 128 / 2),
                      vh = buffer(c, @"vh_half", (size_t)padded * kv * 128 / 2);
        bool ready = c->kvHalfReady;
        c->kvHalfReady = false;
        if (!ready) {
            pipeline(c, @"g2_half_pad");
            bind(c, k, 0);
            bind(c, kh, 1);
            params(c, (G2P){.n = n * kv, .m = padded * kv, .d = 64, .k = 128}, 2);
            threads(c, (size_t)padded * kv * 128);
            bind(c, v, 0);
            bind(c, vh, 1);
            threads(c, (size_t)padded * kv * 128);
        }
        pipeline(c, @"g2t_flash_64p");
        bind(c, q, 0);
        bind(c, kh, 1);
        bind(c, vh, 2);
        bind(c, a, 3);
        params(c,
               (G2P){.n = n, .heads = heads, .kv = kv, .d = d, .window = window,
                     .layer = bounds ? 1 : 0},
               4);
        bind(c, bounds ? bounds : q, 5);
        [c->enc dispatchThreadgroups:MTLSizeMake((n + 31) / 32, heads, 1)
               threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return;
    }
    // Vision (d = 64) takes the simdgroup flash kernel under FLASH2 as well.
    if ((c->flash || (c->flash2 && d == 64)) && !rpe && n >= 128) {
        int padded = (n + 31) / 32 * 32;
        id<MTLBuffer> kh = buffer(c, @"kh", (size_t)padded * kv * d / 2),
                      vh = buffer(c, @"vh_half", (size_t)padded * kv * d / 2);
        pipeline(c, @"g2_half");
        bind(c, k, 0);
        bind(c, kh, 1);
        params(c, (G2P){.n = n, .m = padded, .d = kv * d}, 2);
        threads(c, (size_t)padded * kv * d);
        bind(c, v, 0);
        bind(c, vh, 1);
        threads(c, (size_t)padded * kv * d);
        pipeline(c, [NSString stringWithFormat:@"g2_flash_%d", d]);
        if (d == 512)
            bind(c, buffer(c, @"flash_accum", (size_t)((n + 7) / 8) * 8 * heads * d), 6);
        bind(c, q, 0);
        bind(c, kh, 1);
        bind(c, vh, 2);
        bind(c, a, 3);
        params(c,
               (G2P){.n = n,
                     .heads = heads,
                     .kv = kv,
                     .d = d,
                     .window = window,
                     .layer = bounds ? 1 : 0},
               4);
        bind(c, bounds ? bounds : q, 5);
        [c->enc dispatchThreadgroups:MTLSizeMake((n + 7) / 8, heads, 1)
               threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return;
    }
    pipeline(c, @"g2_attention");
    bind(c, q, 0);
    bind(c, k, 1);
    bind(c, v, 2);
    bind(c, a, 3);
    bind(c, rpe ? rpe : q, 4);
    bind(c, bounds ? bounds : q, 6);
    params(c,
           (G2P){.n = n,
                 .heads = heads,
                 .kv = kv,
                 .d = d,
                 .window = window,
                 .mode = rpe ? 1 : 0,
                 .layer = bounds ? 1 : 0,
                 .lo = lo,
                 .hi = hi},
           5);
    groups(c, n, heads);
}
static void attention(EIGemma2Metal *c, id<MTLBuffer> q, id<MTLBuffer> k, id<MTLBuffer> v,
                      id<MTLBuffer> a, int n, int heads, int kv, int d, int window,
                      id<MTLBuffer> rpe, id<MTLBuffer> bounds) {
    attention_c(c, q, k, v, a, n, heads, kv, d, window, rpe, bounds, 0, 0);
}
static float scalar(const ei_gguf *g, const char *name, float fallback) {
    const ei_tensor *w = ei_gguf_tensor(g, name, false);
    return w ? *(const float *)w->data : fallback;
}
static void clamp_buffer(EIGemma2Metal *c, id<MTLBuffer> x, id<MTLBuffer> y, size_t count, float lo,
                         float hi) {
    pipeline(c, @"g2_element");
    bind(c, x, 0);
    bind(c, x, 1);
    bind(c, y, 2);
    params(c, (G2P){.n = (uint32_t)count, .mode = 5, .lo = lo, .hi = hi}, 3);
    threads(c, count);
}
// The audio tower's per-GEMM clamp bounds (<stem>.input_min/max, output_min/max).
static void clamp_bounds(const ei_gguf *g, const char *name, float *ilo, float *ihi, float *olo,
                         float *ohi) {
    char key[160];
    int stem = (int)strlen(name) - 7;
    snprintf(key, sizeof key, "%.*s.input_min", stem, name);
    *ilo = scalar(g, key, -INFINITY);
    snprintf(key, sizeof key, "%.*s.input_max", stem, name);
    *ihi = scalar(g, key, INFINITY);
    snprintf(key, sizeof key, "%.*s.output_min", stem, name);
    *olo = scalar(g, key, -INFINITY);
    snprintf(key, sizeof key, "%.*s.output_max", stem, name);
    *ohi = scalar(g, key, INFINITY);
}
// clamp(x) W^T then clamp: input_ready says the producer already applied the
// input clamp (AFUSED); the output clamp folds into the wide GEMM where it can.
static void clipped_c(EIGemma2Metal *c, const ei_gguf *g, const char *name, id<MTLBuffer> x,
                      int n, id<MTLBuffer> y, bool input_ready) {
    const ei_tensor *w = ei_gguf_tensor(g, name, true);
    float ilo, ihi, olo, ohi;
    clamp_bounds(g, name, &ilo, &ihi, &olo, &ohi);
    id<MTLBuffer> in = x;
    if (!input_ready) {
        in = buffer(c, @"clamp", (size_t)n * w->ne[0]);
        clamp_buffer(c, x, in, (size_t)n * w->ne[0], ilo, ihi);
    }
    if (!mm_clamp(c, g, name, in, n, y, c->afused ? olo : 0, c->afused ? ohi : 0))
        clamp_buffer(c, y, y, (size_t)n * w->ne[1], olo, ohi);
}
static void clipped(EIGemma2Metal *c, const ei_gguf *g, const char *name, id<MTLBuffer> x, int n,
                    id<MTLBuffer> y) {
    clipped_c(c, g, name, x, n, y, false);
}
// The input clamp of GEMM `name`, for folding into its producer.
static void input_clamp(const ei_gguf *g, const char *name, float *lo, float *hi) {
    float olo, ohi;
    clamp_bounds(g, name, lo, hi, &olo, &ohi);
}
// Dequantize a Q8_0/BF16 [m][k] text weight once into row-major fp16 for the
// wide-wave GEMM. BF16 rounds to fp16, as the tile route always has.
static bool repack_f16(EIGemma2Metal *c, const ei_tensor *t, NSString *name) {
    size_t k = (size_t)t->ne[0], m = (size_t)t->ne[1];
    if ((t->type != EI_T_Q8_0 && t->type != EI_T_BF16 && t->type != EI_T_F32 &&
         t->type != EI_T_F16) ||
        k % 64 || m % 64 || t->n_dims != 2)
        return true;
    id<MTLBuffer> hb = [c->device newBufferWithLength:k * m * 2
                                              options:MTLResourceStorageModeShared];
    if (!hb)
        return false;
    ei_fp16 *h = hb.contents;
    if (t->type == EI_T_F32) {
        const float *src = t->data;
        for (size_t i = 0; i < k * m; i++)
            h[i] = ei_fp32_to_fp16(src[i]);
    } else if (t->type == EI_T_F16) {
        memcpy(h, t->data, k * m * 2);
    } else if (t->type == EI_T_BF16) {
        const uint16_t *src = t->data;
        for (size_t i = 0; i < k * m; i++) {
            uint32_t bits = (uint32_t)src[i] << 16;
            float f;
            memcpy(&f, &bits, 4);
            h[i] = ei_fp32_to_fp16(f);
        }
    } else {
        const ei_block_q8_0 *blocks = t->data;
        for (size_t b = 0; b < k * m / 32; b++) {
            float d = ei_fp16_to_fp32(blocks[b].d);
            for (size_t z = 0; z < 32; z++)
                h[b * 32 + z] = ei_fp32_to_fp16(d * (float)blocks[b].qs[z]);
        }
    }
    c->packed[name] = hb;
    return true;
}
// FFN up/gate interleaved per 64-row tile (32 up rows, then the matching 32
// gate rows), fp16 row-major, for the GeGLU epilogue of g2t_gemm.
static bool repack_upgate(EIGemma2Metal *c, const ei_gguf *g, const char *prefix, int layer) {
    char un[64], gn[64];
    snprintf(un, sizeof un, "%s%d.ffn_up.weight", prefix, layer);
    snprintf(gn, sizeof gn, "%s%d.ffn_gate.weight", prefix, layer);
    id<MTLBuffer> up = c->packed[@(un)], gate = c->packed[@(gn)];
    const ei_tensor *t = ei_gguf_tensor(g, un, true);
    size_t k = (size_t)t->ne[0], m = (size_t)t->ne[1];
    if (!up || !gate || m % 32)
        return true;
    id<MTLBuffer> b = [c->device newBufferWithLength:k * m * 4 options:MTLResourceStorageModeShared];
    if (!b)
        return false;
    uint16_t *dst = b.contents;
    const uint16_t *u = up.contents, *gt = gate.contents;
    for (size_t blk = 0; blk < m / 32; blk++) {
        memcpy(dst + (blk * 64) * k, u + blk * 32 * k, 32 * k * 2);
        memcpy(dst + (blk * 64 + 32) * k, gt + blk * 32 * k, 32 * k * 2);
    }
    char name[64];
    snprintf(name, sizeof name, "%s%d.ffn_upgate", prefix, layer);
    c->packed[@(name)] = b;
    return true;
}
void *ei_gemma2_metal_create(const ei_gguf *g, char *err, size_t len) {
    @autoreleasepool {
        EIGemma2Metal *c = [EIGemma2Metal new];
        c->device = MTLCreateSystemDefaultDevice();
        if (!c->device) {
            snprintf(err, len, "no Metal device");
            return NULL;
        }
        NSError *error = nil;
        id<MTLLibrary> lib = nil;
        const char *path = getenv("EI_METALLIB_PATH");
        if (path && *path)
            lib = [c->device newLibraryWithURL:[NSURL fileURLWithPath:@(path)] error:&error];
        else {
            unsigned long bytes = 0;
            const uint8_t *data =
                getsectiondata(&_mh_execute_header, "__DATA", "__metallib", &bytes);
            if (data && bytes)
                lib = [c->device
                    newLibraryWithData:dispatch_data_create(data, bytes, NULL,
                                                            DISPATCH_DATA_DESTRUCTOR_DEFAULT)
                                 error:&error];
        }
        if (!lib) {
            snprintf(err, len, "Gemma 2 Metal library: %s",
                     error ? error.localizedDescription.UTF8String : "missing embedded library");
            return NULL;
        }
        c->pipelines = [NSMutableDictionary new];
        c->workspace = [NSMutableDictionary new];
        for (NSString *name in @[
                 @"g2_half", @"g2_half_pad", @"g2_mm", @"g2_norm", @"g2_element", @"g2_embed", @"g2_rope",
                 @"g2_attention", @"g2_pool", @"g2_patch", @"g2_position", @"g2_spatial_pool",
                 @"g2_conv", @"g2_audio_scale", @"g2_depthwise", @"g2_bias", @"g2_gemv_q8",
                 @"g2_gemv_bf16", @"g2_gemv_f16", @"g2_gemv_f32", @"g2_gemv_ple", @"g2_gemv_qkv",
                 @"g2_gemv_geglu", @"g2_qkv_post", @"g2_post_add", @"g2_post_add_768",
                 @"g2_post_add_1024", @"g2_vqkv_post"
             ]) {
            id<MTLFunction> f = [lib newFunctionWithName:name];
            id<MTLComputePipelineState> p =
                f ? [c->device newComputePipelineStateWithFunction:f error:&error] : nil;
            if (!p) {
                snprintf(err, len, "Gemma 2 Metal pipeline %s: %s", name.UTF8String,
                         error ? error.localizedDescription.UTF8String : "missing kernel");
                return NULL;
            }
            c->pipelines[name] = p;
        }
        const char *tensorFlag = getenv("EI_GEMMA2_METAL_TENSOR");
        const char *flashFlag = getenv("EI_GEMMA2_METAL_FLASH");
        c->tensorMM = tensorFlag && !strcmp(tensorFlag, "1");
        c->flash = flashFlag && !strcmp(flashFlag, "1");
        // The batched-serving stack (FUSED, TENSOR2, FLASH2) is the default;
        // "0" turns a route off, "1" requires it. TENSOR2/FLASH2 default on only
        // where Metal 4 is available. The 10-07 TENSOR/FLASH routes stay opt-in.
        bool metal4 = false;
        if (@available(macOS 26.0, *))
            metal4 = [c->device supportsFamily:MTLGPUFamilyMetal4];
        const char *gemvFlag = getenv("EI_GEMMA2_METAL_GEMV");
        c->gemv = gemvFlag && !strcmp(gemvFlag, "1");
        const char *fusedFlag = getenv("EI_GEMMA2_METAL_FUSED");
        c->fused = !fusedFlag || strcmp(fusedFlag, "0");
        if (c->fused) {
            // The fused GEMVs decode Q8_0 only; every text layer projection is Q8_0.
            const char *kinds[] = {"attn_q", "attn_k", "attn_v", "ffn_up", "ffn_gate", "inp_gate"};
            for (int l = 0; l < 24 && c->fused; l++)
                for (size_t i = 0; i < sizeof kinds / sizeof *kinds; i++) {
                    char name[64];
                    snprintf(name, sizeof name, "blk.%d.%s.weight", l, kinds[i]);
                    if (ei_gguf_tensor(g, name, true)->type != EI_T_Q8_0)
                        c->fused = false;
                }
            c->gemv |= c->fused;
        }
        if (c->fused)
            // RoPE (cos, sin) tables for all 8192 positions: theta = t * base^(-2z/d).
            for (int b = 0; b < 2; b++) {
                int d = b ? 512 : 256;
                double base = b ? 1e6 : 1e4;
                c->rope[b] = [c->device newBufferWithLength:(size_t)8192 * d / 2 * 8
                                                    options:MTLResourceStorageModeShared];
                float *cs = c->rope[b].contents;
                for (int t = 0; t < 8192; t++)
                    for (int z = 0; z < d / 2; z++) {
                        double theta = t * pow(base, -2.0 * z / d);
                        cs[((size_t)t * d / 2 + z) * 2] = (float)cos(theta);
                        cs[((size_t)t * d / 2 + z) * 2 + 1] = (float)sin(theta);
                    }
            }
        const char *tensor2Flag = getenv("EI_GEMMA2_METAL_TENSOR2");
        c->tensor2 = tensor2Flag ? !strcmp(tensor2Flag, "1") : metal4;
        const char *t2min = getenv("EI_GEMMA2_METAL_TENSOR2_MIN_TOKENS");
        const char *t2wide = getenv("EI_GEMMA2_METAL_TENSOR2_WIDE_TOKENS");
        c->tensor2Min = t2min ? atoi(t2min) : 40;
        const char *noepi = getenv("EI_GEMMA2_METAL_NOEPI");
        c->epi = !(noepi && !strcmp(noepi, "1"));
        // Flash tile: 32 queries x 128 keys on 8 simdgroups by default (n8, same
        // arithmetic as the 4-simdgroup tile, 1.1-1.27x at 2k-8k tokens);
        // EI_GEMMA2_METAL_FLASH2_TILE names another (k64, q16, q16k256, n16,
        // q16n8, q16k256n8) or "n4" for the 4-simdgroup tile.
        const char *tile = getenv("EI_GEMMA2_METAL_FLASH2_TILE");
        if (!tile || !*tile)
            tile = "n8";
        c->flashTile = strcmp(tile, "n4") ? [@"_" stringByAppendingString:@(tile)] : @"";
        c->tensor2Wide = t2wide ? atoi(t2wide) : 256;
        const char *flash2Flag = getenv("EI_GEMMA2_METAL_FLASH2");
        c->flash2 = flash2Flag ? !strcmp(flash2Flag, "1") : metal4;
        const char *f2min = getenv("EI_GEMMA2_METAL_FLASH2_MIN_TOKENS");
        c->flash2Min = f2min ? atoi(f2min) : 16;
        // Media stack (2026-10-10): default on with Metal 4, "0" turns a route off.
        const char *f64 = getenv("EI_GEMMA2_METAL_FLASH64");
        c->flash64 = f64 ? !strcmp(f64, "1") : metal4;
        const char *vf = getenv("EI_GEMMA2_METAL_VFUSED");
        c->vfused = vf ? !strcmp(vf, "1") : metal4;
        const char *af = getenv("EI_GEMMA2_METAL_AFUSED");
        c->afused = af ? !strcmp(af, "1") : true;
        // Persistent layer kernel (Metal 4 library), opt-in: "1" turns it on where
        // the fused route runs and the startup probe shows the groups rendezvous.
        // Measured 0.99-1.13x at 1-8 tokens and 0.5-0.8x at 20-39 (2026-10-11):
        // a grid barrier costs what a dispatch boundary costs on the M5.
        const char *layerFlag = getenv("EI_GEMMA2_METAL_LAYER");
        const char *layerGroups = getenv("EI_GEMMA2_METAL_LAYER_GROUPS");
        c->layerGroups = layerGroups ? atoi(layerGroups) : 128;
        if (c->layerGroups < 8 || c->layerGroups > 1024)
            c->layerGroups = 128;
        bool layerWanted = metal4 && c->fused && layerFlag && !strcmp(layerFlag, "1");
        if (c->tensorMM || c->flash || c->tensor2 || c->flash2 || layerWanted) {
            bool supported = false;
            if (@available(macOS 26.0, *))
                supported = [c->device supportsFamily:MTLGPUFamilyMetal4];
            if (!supported) {
                snprintf(err, len, "Gemma 2 tensor/flash routes require a Metal 4 GPU");
                return NULL;
            }
            id<MTLLibrary> tensorLib = nil;
            const char *tensorPath = getenv("EI_METAL4LIB_PATH");
            if (tensorPath && *tensorPath)
                tensorLib = [c->device newLibraryWithURL:[NSURL fileURLWithPath:@(tensorPath)]
                                                   error:&error];
            else {
                unsigned long bytes = 0;
                const uint8_t *data =
                    getsectiondata(&_mh_execute_header, "__DATA", "__metal4lib", &bytes);
                if (data && bytes)
                    tensorLib = [c->device
                        newLibraryWithData:dispatch_data_create(data, bytes, NULL,
                                                                DISPATCH_DATA_DESTRUCTOR_DEFAULT)
                                     error:&error];
            }
            for (NSString *name in @[
                     @"g2_tensor_mm", @"g2_flash_64", @"g2_flash_256", @"g2_flash_512",
                     @"g2t_gemm_32", @"g2t_gemm_64", @"g2t_gemm_geglu_32",
                     @"g2t_gemm_geglu_64", @"g2t_gemm_ple_32", @"g2t_gemm_ple_64",
                     @"g2t_gemm_qgeglu_32", @"g2t_gemm_qgeglu_64", @"g2t_gemm_clamp_32",
                     @"g2t_gemm_clamp_64", @"g2t_flash_256", @"g2t_flash_512",
                     @"g2t_flash_256_k64", @"g2t_flash_512_k64", @"g2t_flash_64p",
                     @"g2t_flash_256_q16", @"g2t_flash_512_q16", @"g2t_flash_256_q16k256",
                     @"g2t_flash_512_q16k256", @"g2t_flash_256_n8", @"g2t_flash_512_n8",
                     @"g2t_flash_256_q16n8", @"g2t_flash_512_q16n8", @"g2t_flash_256_q16k256n8",
                     @"g2t_flash_512_q16k256n8", @"g2t_flash_256_n16", @"g2t_flash_512_n16",
                     @"g2t_flash_256_q64k64n8", @"g2t_flash_512_q64k64n8",
                     @"g2t_flash_256_q64k64n16", @"g2t_flash_512_q64k64n16", @"g2_layer",
                     @"g2_layer_probe"
                 ]) {
                id<MTLFunction> f = [tensorLib newFunctionWithName:name];
                id<MTLComputePipelineState> p =
                    f ? [c->device newComputePipelineStateWithFunction:f error:&error] : nil;
                if (!p && ([name hasSuffix:@"_k64"] || [name containsString:@"_q16"] ||
                           [name containsString:@"_q64"] || [name containsString:@"_n8"] ||
                           [name hasSuffix:@"_n16"] || [name hasPrefix:@"g2_layer"])) {
                    // Optional flash tile experiments: report and fall back.
                    if (getenv("EI_GEMMA2_METAL_PROFILE"))
                        fprintf(stderr, "%s unavailable: %s\n", name.UTF8String,
                                error ? error.localizedDescription.UTF8String : "missing");
                    continue;
                }
                if (!p) {
                    snprintf(err, len, "Gemma 2 Metal tensor pipeline %s unavailable: %s",
                             name.UTF8String,
                             error ? error.localizedDescription.UTF8String : "missing kernel");
                    return NULL;
                }
                if (getenv("EI_GEMMA2_METAL_PROFILE"))
                    fprintf(stderr, "%s threadgroup memory %lu\n", name.UTF8String,
                            (unsigned long)p.staticThreadgroupMemoryLength);
                c->pipelines[name] = p;
            }
        }
        c->queue = [c->device newCommandQueue];
        c->textModel = g;
        c->textWeights = [c->device newBufferWithBytes:g->map + g->data_off
                                                length:g->map_len - g->data_off
                                               options:MTLResourceStorageModeShared];
        if (!c->queue || !c->textWeights) {
            snprintf(err, len, "Gemma 2 Metal weight allocation failed");
            return NULL;
        }
        c->layer = layerWanted && c->pipelines[@"g2_layer"] && c->pipelines[@"g2_layer_probe"];
        if (c->layer) {
            // The layer kernel decodes Q8_0 only (the fused check covers the rest).
            const char *kinds[] = {"attn_output", "ffn_down", "proj"};
            for (int l = 0; l < 24 && c->layer; l++)
                for (size_t i = 0; i < sizeof kinds / sizeof *kinds; i++) {
                    char name[64];
                    snprintf(name, sizeof name, "blk.%d.%s.weight", l, kinds[i]);
                    const ei_tensor *t = ei_gguf_tensor(g, name, true);
                    if (t->type != EI_T_Q8_0 || t->offset > UINT32_MAX)
                        c->layer = false;
                }
        }
        if (c->layer && !layer_probe(c)) {
            c->layer = false;
            fprintf(stderr,
                    "EmbeddingGemma 2 Metal: %d persistent layer groups did not rendezvous; "
                    "using the dispatch route\n",
                    c->layerGroups);
        }
        c->packed = [NSMutableDictionary new];
        if (c->tensor2)
            for (uint64_t i = 0; i < g->n_tensors; i++) {
                const ei_tensor *t = &g->tensors[i];
                NSString *name = [[NSString alloc] initWithBytes:t->name.str
                                                          length:t->name.len
                                                        encoding:NSUTF8StringEncoding];
                if (([name hasPrefix:@"blk."] || [name isEqualToString:@"output.weight"] ||
                     [name isEqualToString:@"per_layer_model_proj.weight"]) &&
                    !repack_f16(c, t, name)) {
                    snprintf(err, len, "Gemma 2 Metal repack allocation failed");
                    return NULL;
                }
            }
        if (c->tensor2)
            for (int l = 0; l < 24; l++)
                if (!repack_upgate(c, g, "blk.", l)) {
                    snprintf(err, len, "Gemma 2 Metal repack allocation failed");
                    return NULL;
                }
        fprintf(stderr, "EmbeddingGemma 2 Metal: %s (native text, vision, audio)\n",
                c->device.name.UTF8String);
        return (__bridge_retained void *)c;
    }
}
void ei_gemma2_metal_free(void *ctx) {
    if (ctx) {
        EIGemma2Metal *c = (__bridge_transfer EIGemma2Metal *)ctx;
        (void)c;
    }
}
void *ei_gemma2_metal_clone(void *ctx, char *err, size_t len) {
    @autoreleasepool {
        EIGemma2Metal *p = (__bridge EIGemma2Metal *)ctx, *c = [EIGemma2Metal new];
        c->device = p->device;
        c->pipelines = p->pipelines;
        c->textWeights = p->textWeights;
        c->textModel = p->textModel;
        c->packed = p->packed;
        c->tensorMM = p->tensorMM;
        c->flash = p->flash;
        c->gemv = p->gemv;
        c->fused = p->fused;
        c->tensor2 = p->tensor2;
        c->flash2 = p->flash2;
        c->flash64 = p->flash64;
        c->vfused = p->vfused;
        c->afused = p->afused;
        c->tensor2Min = p->tensor2Min;
        c->tensor2Wide = p->tensor2Wide;
        c->flash2Min = p->flash2Min;
        c->epi = p->epi;
        c->flashTile = p->flashTile;
        c->layer = p->layer;
        c->layerGroups = p->layerGroups;
        c->rope[0] = p->rope[0];
        c->rope[1] = p->rope[1];
        c->workspace = [NSMutableDictionary new];
        c->queue = [c->device newCommandQueue];
        if (!c->queue) {
            snprintf(err, len, "Gemma 2 Metal slot queue allocation failed");
            return NULL;
        }
        return (__bridge_retained void *)c;
    }
}
static bool projector(EIGemma2Metal *c, const ei_gguf *g, char *err, size_t len) {
    if (c->mediaModel == g && c->mediaWeights)
        return true;
    c->mediaWeights = [c->device newBufferWithBytes:g->map + g->data_off
                                             length:g->map_len - g->data_off
                                            options:MTLResourceStorageModeShared];
    if (!c->mediaWeights) {
        snprintf(err, len, "Gemma 2 Metal projector allocation failed");
        return false;
    }
    c->mediaModel = g;
    if (c->tensor2) {
        // fp16 copies for wide media waves; the patch embedding keeps its own
        // [768 x 768] view in mm() and stays on the tile route.
        NSMutableDictionary *packed = [c->packed mutableCopy];
        c->packed = packed;
        for (uint64_t i = 0; i < g->n_tensors; i++) {
            const ei_tensor *t = &g->tensors[i];
            NSString *name = [[NSString alloc] initWithBytes:t->name.str
                                                      length:t->name.len
                                                    encoding:NSUTF8StringEncoding];
            ei_tensor view = *t;
            if ([name isEqualToString:@"v.patch_embd.weight"]) {
                // [16][16][3][768] read as the [768][768] matrix mm() uses.
                view.n_dims = 2;
                view.ne[0] = view.ne[1] = 768;
            }
            if (([name hasPrefix:@"v."] || [name hasPrefix:@"a."] || [name hasPrefix:@"mm."]) &&
                !repack_f16(c, &view, name)) {
                snprintf(err, len, "Gemma 2 Metal projector repack allocation failed");
                return false;
            }
        }
        for (int l = 0; l < 16; l++)
            if (!repack_upgate(c, g, "v.blk.", l)) {
                snprintf(err, len, "Gemma 2 Metal projector repack allocation failed");
                return false;
            }
    }
    return true;
}
// Fused text-layer helpers (EI_GEMMA2_METAL_FUSED=1). Short waves use fused
// GEMVs; waves on the tensor route keep separate projections.
static bool short_route(EIGemma2Metal *c, int n) {
    return !(c->tensorMM && n >= 32) && !(c->tensor2 && n >= c->tensor2Min);
}
static void short_groups(EIGemma2Metal *c, size_t rows_per_group, size_t rows, int n) {
    [c->enc dispatchThreadgroups:MTLSizeMake(rows / rows_per_group, (n + 7) / 8, 1)
           threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}
// x = (x + rms(h) w_post) * scale, or x += rms(h) w_post * scale when hscale;
// y = rms(x) w_next (clamped to [lo, hi] when lo < hi) if next is given.
static void post_add_a(EIGemma2Metal *c, const ei_gguf *g, const char *post, const char *next,
                       id<MTLBuffer> h, id<MTLBuffer> x, id<MTLBuffer> y, int n, float scale,
                       int d, bool hscale, float lo, float hi) {
    if (skipped("post"))
        return;
    pipeline(c, d == 1024 ? @"g2_post_add_1024" : d == 768 ? @"g2_post_add_768" : @"g2_post_add");
    bind(c, h, 0);
    bind(c, x, 1);
    bind(c, y, 2);
    G2P p = {.n = n, .scale = scale, .type = weight(c, g, post, 4)->type, .cols = hscale,
             .lo = lo, .hi = hi};
    if (next)
        p.mode = weight(c, g, next, 5)->type + 1;
    params(c, p, 3);
    groups(c, n, 1);
}
static void post_add_d(EIGemma2Metal *c, const ei_gguf *g, const char *post, const char *next,
                       id<MTLBuffer> h, id<MTLBuffer> x, id<MTLBuffer> y, int n, float scale,
                       int d) {
    post_add_a(c, g, post, next, h, x, y, n, scale, d, false, 0, 0);
}
static void post_add(EIGemma2Metal *c, const ei_gguf *g, const char *post, const char *next,
                     id<MTLBuffer> h, id<MTLBuffer> x, id<MTLBuffer> y, int n, float scale) {
    post_add_d(c, g, post, next, h, x, y, n, scale, 512);
}
// Wide-wave GEMM with a fused epilogue on a packed fp16 weight: "geglu" over
// an interleaved up/gate weight (k=512, 2 x m rows), or "ple" for inp_gate.
static bool wide_epi(EIGemma2Metal *c, NSString *key, NSString *kind, id<MTLBuffer> x, int n,
                     int k, int m, id<MTLBuffer> y, id<MTLBuffer> ple, int layer) {
    id<MTLBuffer> w = c->packed[key];
    if (!w || !c->epi || !c->tensor2 || n < c->tensor2Min)
        return false;
    int tm = n >= c->tensor2Wide ? 64 : 32;
    pipeline(c, [NSString stringWithFormat:@"g2t_gemm_%@_%d", kind, tm]);
    [c->enc setBuffer:w offset:0 atIndex:0];
    bind(c, x, 1);
    bind(c, y, 2);
    params(c, (G2P){.n = n, .k = k, .m = m, .layer = layer}, 3);
    bind(c, ple ? ple : y, 4);
    [c->enc dispatchThreadgroups:MTLSizeMake((n + tm - 1) / tm, m / 64, 1)
           threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return true;
}
static void text_layer_fused(EIGemma2Metal *c, const ei_gguf *g, int l, int n, size_t count,
                             id<MTLBuffer> x, id<MTLBuffer> h, id<MTLBuffer> q, id<MTLBuffer> k,
                             id<MTLBuffer> v, id<MTLBuffer> a, id<MTLBuffer> up,
                             id<MTLBuffer> gate, id<MTLBuffer> ple, id<MTLBuffer> bounds) {
    char name[128], next[128];
#define T(s) (snprintf(name, sizeof name, "blk.%d." s ".weight", l), name)
    int d = l % 6 == 5 ? 512 : 256, kv = 512 / d;
    bool fast = short_route(c, n);
    // h holds rms(x) * attn_norm, written by the previous post_add (or the prologue).
    if (fast && skipped("qkv")) {
    } else if (fast) {
        pipeline(c, @"g2_gemv_qkv");
        weight(c, g, T("attn_q"), 0);
        weight(c, g, T("attn_k"), 4);
        weight(c, g, T("attn_v"), 5);
        bind(c, h, 1);
        bind(c, q, 2);
        bind(c, k, 6);
        bind(c, v, 7);
        params(c, (G2P){.n = n, .k = 512, .m = 4 * d, .d = 512}, 3);
        short_groups(c, 16, 4 * d + 1024, n);
    } else {
        mm(c, g, T("attn_q"), h, n, q);
        mm(c, g, T("attn_k"), h, n, k);
        mm(c, g, T("attn_v"), h, n, v);
    }
    if (!skipped("qknorm")) {
    pipeline(c, @"g2_qkv_post");
    bind(c, q, 0);
    bind(c, k, 1);
    bind(c, v, 2);
    uint32_t norm_type = weight(c, g, T("attn_q_norm"), 4)->type;
    weight(c, g, T("attn_k_norm"), 5);
    bind(c, bounds, 6);
    // Tensor flash reads fp16 K/V with 64 zero pad rows; write them here.
    bool half_kv = c->flash2 && n >= c->flash2Min;
    int rows = half_kv ? (n + 63) / 64 * 64 + 128 : n;
    if (half_kv) {
        bind(c, buffer(c, @"kh", (size_t)rows * kv * d / 2), 7);
        bind(c, buffer(c, @"vh_half", (size_t)rows * kv * d / 2), 8);
    } else {
        bind(c, q, 7);
        bind(c, q, 8);
    }
    bind(c, c->rope[l % 6 == 5], 9);
    params(c,
           (G2P){.n = n, .heads = 4, .kv = kv, .d = d, .type = norm_type,
                 .scale = l % 6 == 5 ? 1e6f : 1e4f, .mode = half_kv, .cols = 1},
           3);
    groups(c, rows, 4 + 2 * kv);
    c->kvHalfReady = half_kv;
    }
    attention(c, q, k, v, a, n, 4, kv, d, l % 6 == 5 ? 0 : 512, nil, bounds);
    mm(c, g, T("attn_output"), a, n, h);
    post_add(c, g, T("post_attention_norm"),
             (snprintf(next, sizeof next, "blk.%d.ffn_norm.weight", l), next), h, x, h, n, 1);
    if (fast && skipped("geglu")) {
    } else if (fast) {
        pipeline(c, @"g2_gemv_geglu");
        weight(c, g, T("ffn_up"), 0);
        weight(c, g, T("ffn_gate"), 4);
        bind(c, h, 1);
        bind(c, gate, 2);
        params(c, (G2P){.n = n, .k = 512, .m = 2048}, 3);
        short_groups(c, 8, 2048, n);
    } else if (!wide_epi(c, [NSString stringWithFormat:@"blk.%d.ffn_upgate", l], @"geglu", h, n,
                         512, 4096, gate, nil, 0)) {
        mm(c, g, T("ffn_up"), h, n, up);
        mm(c, g, T("ffn_gate"), h, n, gate);
        element(c, gate, up, gate, count * 2048, 1, 1, 0);
    }
    mm(c, g, T("ffn_down"), gate, n, h);
    post_add(c, g, T("post_ffw_norm"), NULL, h, x, h, n, 1);
    if (fast && skipped("ple")) {
    } else if (fast) {
        pipeline(c, @"g2_gemv_ple");
        weight(c, g, T("inp_gate"), 0);
        bind(c, x, 1);
        bind(c, h, 2);
        bind(c, ple, 4);
        params(c, (G2P){.n = n, .k = 512, .m = 512, .layer = l}, 3);
        short_groups(c, 16, 512, n);
    } else if (!wide_epi(c, @(T("inp_gate")), @"ple", x, n, 512, 512, h, ple, l)) {
        mm(c, g, T("inp_gate"), x, n, h);
        element(c, h, ple, h, count * 512, 4, 1, l);
    }
    mm(c, g, T("proj"), h, n, a);
    float scale = scalar(g, T("layer_output_scale"), 1);
    if (l < 23)
        snprintf(next, sizeof next, "blk.%d.attn_norm.weight", l + 1);
    else
        snprintf(next, sizeof next, "output_norm.weight");
    post_add(c, g, T("post_norm"), next, a, x, h, n, scale);
#undef T
}
// Startup probe for the persistent layer kernel: a forward's worth of grid
// barriers (24 layers x 11 stages) across layerGroups threadgroups. The spin is
// capped in the kernel, so a group count the GPU cannot keep co-resident fails
// in milliseconds instead of hanging.
static bool layer_probe(EIGemma2Metal *c) {
    @autoreleasepool {
        id<MTLBuffer> bar = buffer(c, @"layer_bar", 24), fail = buffer(c, @"layer_fail", 1);
        memset(bar.contents, 0, 24 * 4);
        memset(fail.contents, 0, 4);
        begin(c);
        pipeline(c, @"g2_layer_probe");
        bind(c, bar, 0);
        bind(c, fail, 1);
        params(c, (G2P){.n = 264, .m = (uint32_t)c->layerGroups}, 2);
        [c->enc dispatchThreadgroups:MTLSizeMake(c->layerGroups, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        char err[256];
        double start = now_ms();
        bool ok = finish(c, err, sizeof err) && !*(uint32_t *)fail.contents;
        if (getenv("EI_GEMMA2_METAL_PROFILE"))
            fprintf(stderr, "persistent layer probe: %d groups x 264 barriers %s in %.3f ms\n",
                    c->layerGroups, ok ? "ok" : "FAILED", now_ms() - start);
        return ok;
    }
}
static uint32_t woff(const ei_gguf *g, const char *name, uint32_t *type) {
    const ei_tensor *w = ei_gguf_tensor(g, name, true);
    if (type)
        *type = w->type;
    return (uint32_t)w->offset;
}
// One text layer as a single persistent dispatch (g2_layer): the short-wave
// stages of text_layer_fused with grid barriers between them. h holds
// rms(x) * attn_norm on entry and rms(x) * next_norm on exit, as there.
static void text_layer_persistent(EIGemma2Metal *c, const ei_gguf *g, int l, int n,
                                  id<MTLBuffer> x, id<MTLBuffer> h, id<MTLBuffer> q,
                                  id<MTLBuffer> k, id<MTLBuffer> v, id<MTLBuffer> a,
                                  id<MTLBuffer> gate, id<MTLBuffer> ple, id<MTLBuffer> bounds) {
    char name[128], next[128];
#define T(s) (snprintf(name, sizeof name, "blk.%d." s ".weight", l), name)
    int d = l % 6 == 5 ? 512 : 256, kv = 512 / d;
    uint32_t tqn = 0, tpa = 0, tfn = 0, tpf = 0, tpn = 0, tnext = 0;
    G2L L = {.n = (uint32_t)n, .d = (uint32_t)d, .kv = (uint32_t)kv,
             .groups = (uint32_t)c->layerGroups, .ntg = (uint32_t)(n + 7) / 8, .layer = (uint32_t)l};
    L.wq = woff(g, T("attn_q"), NULL);
    L.wk = woff(g, T("attn_k"), NULL);
    L.wv = woff(g, T("attn_v"), NULL);
    L.wo = woff(g, T("attn_output"), NULL);
    L.wu = woff(g, T("ffn_up"), NULL);
    L.wg = woff(g, T("ffn_gate"), NULL);
    L.wd = woff(g, T("ffn_down"), NULL);
    L.wig = woff(g, T("inp_gate"), NULL);
    L.wpr = woff(g, T("proj"), NULL);
    L.wqn = woff(g, T("attn_q_norm"), &tqn);
    L.wkn = woff(g, T("attn_k_norm"), NULL);
    L.wpa = woff(g, T("post_attention_norm"), &tpa);
    L.wfn = woff(g, T("ffn_norm"), &tfn);
    L.wpf = woff(g, T("post_ffw_norm"), &tpf);
    L.wpn = woff(g, T("post_norm"), &tpn);
    if (l < 23)
        snprintf(next, sizeof next, "blk.%d.attn_norm.weight", l + 1);
    else
        snprintf(next, sizeof next, "output_norm.weight");
    L.wnext = woff(g, next, &tnext);
    float scale = scalar(g, T("layer_output_scale"), 1);
    // Stage parameters exactly as the standalone dispatches would receive them.
    G2P S[5] = {
        {.n = n, .heads = 4, .kv = kv, .d = d, .type = tqn, .scale = l % 6 == 5 ? 1e6f : 1e4f,
         .cols = 1},
        {.n = n, .heads = 4, .kv = kv, .d = d, .window = l % 6 == 5 ? 0 : 512, .layer = 1},
        {.n = n, .scale = 1, .type = tpa, .mode = tfn + 1},
        {.n = n, .scale = 1, .type = tpf},
        {.n = n, .scale = scale, .type = tpn, .mode = tnext + 1},
    };
    pipeline(c, @"g2_layer");
    [c->enc setBuffer:c->textWeights offset:0 atIndex:0];
    bind(c, x, 1);
    bind(c, h, 2);
    bind(c, q, 3);
    bind(c, k, 4);
    bind(c, v, 5);
    bind(c, a, 6);
    bind(c, gate, 7);
    bind(c, ple, 8);
    bind(c, bounds, 9);
    bind(c, c->rope[l % 6 == 5], 10);
    [c->enc setBuffer:buffer(c, @"layer_bar", 24) offset:(NSUInteger)l * 4 atIndex:11];
    bind(c, buffer(c, @"layer_fail", 1), 12);
    [c->enc setBytes:&L length:sizeof L atIndex:13];
    [c->enc setBytes:S length:sizeof S atIndex:14];
    [c->enc dispatchThreadgroups:MTLSizeMake(c->layerGroups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
#undef T
}
static bool embed(void *ctx, const int32_t *ids, const float *inputs, size_t count,
                  const size_t *offsets, size_t batch, float *out, char *err, size_t len) {
    @autoreleasepool {
        if (!count || count > 8192 || (!ids && !inputs)) {
            snprintf(err, len, "EmbeddingGemma 2 accepts 1..8192 input tokens");
            return false;
        }
        EIGemma2Metal *c = (__bridge EIGemma2Metal *)ctx;
        const ei_gguf *g = c->textModel;
        int n = (int)count;
        id<MTLBuffer> x = buffer(c, @"x", count * 512), h = buffer(c, @"h", count * 512),
                      q = buffer(c, @"q", count * 2048), k = buffer(c, @"k", count * 512),
                      v = buffer(c, @"v", count * 512), a = buffer(c, @"a", count * 2048),
                      up = buffer(c, @"up", count * 2048), gate = buffer(c, @"gate", count * 2048),
                      ple = buffer(c, @"ple", count * 512 * 24),
                      result = buffer(c, @"result", batch * 768);
        id<MTLBuffer> bounds = buffer(c, @"bounds", count * 2),
                      ranges = buffer(c, @"ranges", batch * 2);
        uint32_t *bp = bounds.contents, *rp = ranges.contents;
        for (size_t s = 0; s < batch; s++) {
            uint32_t first = (uint32_t)offsets[s], last = (uint32_t)offsets[s + 1];
            rp[2 * s] = first;
            rp[2 * s + 1] = last;
            for (uint32_t t = first; t < last; t++) {
                bp[2 * t] = first;
                bp[2 * t + 1] = last;
            }
        }
        if (inputs)
            memcpy(x.contents, inputs, count * 512 * 4);
        begin(c);
        if (!inputs) {
            id<MTLBuffer> tokens = buffer(c, @"ids", count);
            memcpy(tokens.contents, ids, count * 4);
            pipeline(c, @"g2_embed");
            weight(c, g, "token_embd.weight", 0);
            bind(c, tokens, 1);
            bind(c, x, 2);
            params(c, (G2P){.n = n}, 3);
            threads(c, count * 512);
        }
        mm(c, g, "per_layer_model_proj.weight", x, n, ple);
        element(c, ple, ple, ple, count * 512 * 24, 7, 1 / sqrtf(512), 0);
        norm(c, g, "per_layer_proj_norm.weight", ple, n * 24, 512, ple, 1);
        bool persistent = c->fused && c->layer && short_route(c, n);
        if (c->fused) {
            norm(c, g, "blk.0.attn_norm.weight", x, n, 512, h, 1);
            if (persistent) {
                memset(buffer(c, @"layer_bar", 24).contents, 0, 24 * 4);
                memset(buffer(c, @"layer_fail", 1).contents, 0, 4);
            }
            for (int l = 0; l < 24; l++)
                if (persistent)
                    text_layer_persistent(c, g, l, n, x, h, q, k, v, a, gate, ple, bounds);
                else
                    text_layer_fused(c, g, l, n, count, x, h, q, k, v, a, up, gate, ple, bounds);
            // h = rms(x) * output_norm from the last layer's post_add.
            mm(c, g, "output.weight", h, n, a);
        } else {
        for (int l = 0; l < 24; l++) {
            char name[128];
#define T(s) (snprintf(name, sizeof name, "blk.%d." s ".weight", l), name)
            int d = l % 6 == 5 ? 512 : 256, kv = 512 / d;
            norm(c, g, T("attn_norm"), x, n, 512, h, 1);
            mm(c, g, T("attn_q"), h, n, q);
            mm(c, g, T("attn_k"), h, n, k);
            mm(c, g, T("attn_v"), h, n, v);
            norm(c, g, T("attn_q_norm"), q, n * 4, d, q, 1);
            norm(c, g, T("attn_k_norm"), k, n * kv, d, k, 1);
            norm(c, g, NULL, v, n * kv, d, v, 0);
            rotary(c, q, n, 4, d, l % 6 == 5 ? 1e6f : 1e4f, 0, bounds);
            rotary(c, k, n, kv, d, l % 6 == 5 ? 1e6f : 1e4f, 0, bounds);
            attention(c, q, k, v, a, n, 4, kv, d, l % 6 == 5 ? 0 : 512, nil, bounds);
            mm(c, g, T("attn_output"), a, n, h);
            norm(c, g, T("post_attention_norm"), h, n, 512, h, 1);
            element(c, x, h, x, count * 512, 0, 1, 0);
            norm(c, g, T("ffn_norm"), x, n, 512, h, 1);
            mm(c, g, T("ffn_up"), h, n, up);
            mm(c, g, T("ffn_gate"), h, n, gate);
            element(c, gate, up, gate, count * 2048, 1, 1, 0);
            mm(c, g, T("ffn_down"), gate, n, h);
            norm(c, g, T("post_ffw_norm"), h, n, 512, h, 1);
            element(c, x, h, x, count * 512, 0, 1, 0);
            mm(c, g, T("inp_gate"), x, n, h);
            element(c, h, ple, h, count * 512, 4, 1, l);
            mm(c, g, T("proj"), h, n, a);
            norm(c, g, T("post_norm"), a, n, 512, a, 1);
            float scale = scalar(g, T("layer_output_scale"), 1);
            element(c, x, a, x, count * 512, 0, scale, 0);
#undef T
        }
        norm(c, g, "output_norm.weight", x, n, 512, x, 1);
        mm(c, g, "output.weight", x, n, a);
        }
        pipeline(c, @"g2_pool");
        bind(c, a, 0);
        bind(c, result, 1);
        bind(c, ranges, 2);
        params(c, (G2P){.n = (uint32_t)batch}, 3);
        threads(c, batch * 768);
        if (!finish(c, err, len))
            return false;
        if (persistent && *(uint32_t *)buffer(c, @"layer_fail", 1).contents) {
            // The groups did not rendezvous (GPU shared with other work): the
            // forward is invalid. Redo it on the dispatch route and stay there.
            fprintf(stderr, "EmbeddingGemma 2 Metal: persistent layer barrier timed out; "
                            "using the dispatch route\n");
            c->layer = false;
            return embed(ctx, ids, inputs, count, offsets, batch, out, err, len);
        }
        memcpy(out, result.contents, batch * 768 * 4);
        for (size_t s = 0; s < batch; s++)
            ei_l2_normalize(out + s * 768, 768);
        return true;
    }
}
bool ei_gemma2_metal_vision(void *ctx, const ei_gguf *g, const uint8_t *rgb, int width, int height,
                            float **out, size_t *tokens, char *err, size_t len) {
    return ei_gemma2_metal_vision_frames(ctx, g, &rgb, 1, width, height, out, tokens, err, len);
}
// Frames of one size run as one packed wave: tokens are laid out frame after
// frame (per-frame kernels index t % fn, attention and RoPE take per-token
// sequence bounds), so every GEMM sees frames x fn rows and the tower runs
// once instead of once per frame.
bool ei_gemma2_metal_vision_frames(void *ctx, const ei_gguf *g, const uint8_t *const *rgb,
                                   int frames, int width, int height, float **out, size_t *tokens,
                                   char *err, size_t len) {
    @autoreleasepool {
        if (width < 48 || height < 48 || width % 48 || height % 48 ||
            (size_t)width * height > 1120 * 48 * 48) {
            snprintf(err, len,
                     "vision input must be aligned to 48 pixels and at most 1120 soft tokens");
            return false;
        }
        if (frames < 1 || (size_t)frames * width * height > EI_GEMMA2_VISION_WAVE_TOKENS * 9 * 256) {
            snprintf(err, len, "vision wave exceeds %d patch tokens", EI_GEMMA2_VISION_WAVE_TOKENS);
            return false;
        }
        EIGemma2Metal *c = (__bridge EIGemma2Metal *)ctx;
        if (!projector(c, g, err, len))
            return false;
        int cols = width / 16, fn = cols * (height / 16), n = fn * frames, nt = n / 9;
        size_t count = (size_t)n * 768, pixel_bytes = (size_t)width * height * 3;
        id<MTLBuffer> x = buffer(c, @"vx", count), h = buffer(c, @"vh", count),
                      q = buffer(c, @"vq", count), k = buffer(c, @"vk", count),
                      v = buffer(c, @"vv", count), a = buffer(c, @"va", count),
                      up = buffer(c, @"vup", count * 4), gate = buffer(c, @"vgate", count * 4),
                      pixels = buffer(c, @"pixels", (pixel_bytes * frames + 3) / 4),
                      result = buffer(c, @"vresult", (size_t)nt * 512), bounds = nil;
        for (int f = 0; f < frames; f++)
            memcpy((uint8_t *)pixels.contents + pixel_bytes * f, rgb[f], pixel_bytes);
        if (frames > 1) {
            // Per-token [start, end) of its frame for attention and RoPE.
            bounds = buffer(c, @"vbounds", (size_t)n * 2);
            uint32_t *b = bounds.contents;
            for (int t = 0; t < n; t++) {
                b[2 * t] = (uint32_t)(t / fn * fn);
                b[2 * t + 1] = b[2 * t] + (uint32_t)fn;
            }
        }
        begin(c);
        pipeline(c, @"g2_patch");
        bind(c, pixels, 0);
        bind(c, h, 1);
        params(c, (G2P){.n = n, .k = fn, .cols = cols}, 2);
        threads(c, count);
        mm(c, g, "v.patch_embd.weight", h, n, x);
        pipeline(c, @"g2_position");
        bind(c, x, 0);
        weight(c, g, "v.position_embd.weight", 1);
        params(c, (G2P){.n = n, .k = fn, .cols = cols}, 2);
        threads(c, count);
        bool vfused = c->vfused && c->flash2 && c->flash64;
        if (vfused)
            norm(c, g, "v.blk.0.ln1.weight", x, n, 768, h, 1);
        for (int l = 0; l < 16 && vfused; l++) {
            char name[128], next[128];
#define V(s) (snprintf(name, sizeof name, "v.blk.%d." s ".weight", l), name)
            // h = rms(x) * ln1 from the prologue or the previous post_add.
            mm(c, g, V("attn_q"), h, n, q);
            mm(c, g, V("attn_k"), h, n, k);
            mm(c, g, V("attn_v"), h, n, v);
            bool half_kv = n >= c->flash2Min;
            int rows = half_kv ? (n + 63) / 64 * 64 + 128 : n;
            pipeline(c, @"g2_vqkv_post");
            bind(c, q, 0);
            bind(c, k, 1);
            bind(c, v, 2);
            weight(c, g, V("attn_q_norm"), 4);
            weight(c, g, V("attn_k_norm"), 5);
            if (half_kv) {
                bind(c, buffer(c, @"kh", (size_t)rows * 12 * 128 / 2), 7);
                bind(c, buffer(c, @"vh_half", (size_t)rows * 12 * 128 / 2), 8);
            } else {
                bind(c, q, 7);
                bind(c, q, 8);
            }
            params(c, (G2P){.n = n, .k = fn, .m = rows, .heads = 12, .cols = cols, .mode = half_kv},
                   3);
            groups(c, rows, 36);
            c->kvHalfReady = half_kv;
            attention(c, q, k, v, a, n, 12, 12, 64, 0, nil, bounds);
            mm(c, g, V("attn_out"), a, n, h);
            post_add_d(c, g, V("attn_post_norm"),
                       (snprintf(next, sizeof next, "v.blk.%d.ln2.weight", l), next), h, x, h, n,
                       1, 768);
            if (!wide_epi(c, [NSString stringWithFormat:@"v.blk.%d.ffn_upgate", l], @"qgeglu", h,
                          n, 768, 6144, gate, nil, 0)) {
                mm(c, g, V("ffn_up"), h, n, up);
                mm(c, g, V("ffn_gate"), h, n, gate);
                element(c, gate, up, gate, count * 4, 2, 1, 0);
            }
            mm(c, g, V("ffn_down"), gate, n, h);
            if (l < 15)
                snprintf(next, sizeof next, "v.blk.%d.ln1.weight", l + 1);
            post_add_d(c, g, V("ffn_post_norm"), l < 15 ? next : NULL, h, x, h, n, 1, 768);
#undef V
        }
        for (int l = 0; l < 16 && !vfused; l++) {
            char name[128];
#define V(s) (snprintf(name, sizeof name, "v.blk.%d." s ".weight", l), name)
            norm(c, g, V("ln1"), x, n, 768, h, 1);
            mm(c, g, V("attn_q"), h, n, q);
            mm(c, g, V("attn_k"), h, n, k);
            mm(c, g, V("attn_v"), h, n, v);
            norm(c, g, V("attn_q_norm"), q, n * 12, 64, q, 1);
            norm(c, g, V("attn_k_norm"), k, n * 12, 64, k, 1);
            norm(c, g, NULL, v, n * 12, 64, v, 0);
            rotary(c, q, n, 12, 64, 100, cols, bounds);
            rotary(c, k, n, 12, 64, 100, cols, bounds);
            attention(c, q, k, v, a, n, 12, 12, 64, 0, nil, bounds);
            mm(c, g, V("attn_out"), a, n, h);
            norm(c, g, V("attn_post_norm"), h, n, 768, h, 1);
            element(c, x, h, x, count, 0, 1, 0);
            norm(c, g, V("ln2"), x, n, 768, h, 1);
            mm(c, g, V("ffn_up"), h, n, up);
            mm(c, g, V("ffn_gate"), h, n, gate);
            element(c, gate, up, gate, count * 4,
                    ei_gguf_kv_bool(g, "clip.use_gelu", false) ? 1 : 2, 1, 0);
            mm(c, g, V("ffn_down"), gate, n, h);
            norm(c, g, V("ffn_post_norm"), h, n, 768, h, 1);
            element(c, x, h, x, count, 0, 1, 0);
#undef V
        }
        pipeline(c, @"g2_spatial_pool");
        bind(c, x, 0);
        bind(c, h, 1);
        params(c, (G2P){.n = nt, .k = fn / 9, .cols = cols}, 2);
        threads(c, (size_t)nt * 768);
        norm(c, g, NULL, h, nt, 768, h, 0);
        mm(c, g, "mm.input_projection.weight", h, nt, result);
        if (!finish(c, err, len))
            return false;
        *tokens = nt;
        *out = ei_xmalloc((size_t)nt * 512 * 4);
        memcpy(*out, result.contents, (size_t)nt * 512 * 4);
        return true;
    }
}
bool ei_gemma2_metal_audio(void *ctx, const ei_gguf *g, const float *mel, int frames, float **out,
                           size_t *tokens, char *err, size_t len) {
    return ei_gemma2_metal_audio_clips(ctx, g, &mel, &frames, 1, out, tokens, err, len);
}
// Clips run as one packed wave: the subsampling convs run per clip (their
// outputs land clip after clip, (frames + 3) / 4 rows of 1024 each), then the
// conformer layers see the sum of the rows with per-token clip bounds for the
// depthwise conv and attention. One clip encodes exactly as before.
bool ei_gemma2_metal_audio_clips(void *ctx, const ei_gguf *g, const float *const *mel,
                                 const int *frames, int clips, float **out, size_t *tokens,
                                 char *err, size_t len) {
    @autoreleasepool {
        if (clips < 1 || clips > 64) {
            snprintf(err, len, "audio wave must contain 1..64 clips");
            return false;
        }
        size_t total_frames = 0;
        int n = 0;
        for (int i = 0; i < clips; i++) {
            if (frames[i] < 1 || frames[i] > 3000) {
                snprintf(err, len, "audio chunk must contain 1..3000 mel frames");
                return false;
            }
            total_frames += (size_t)frames[i];
            n += (frames[i] + 3) / 4;
        }
        if (total_frames > EI_GEMMA2_AUDIO_WAVE_FRAMES) {
            snprintf(err, len, "audio wave exceeds %d mel frames", EI_GEMMA2_AUDIO_WAVE_FRAMES);
            return false;
        }
        EIGemma2Metal *c = (__bridge EIGemma2Metal *)ctx;
        if (!projector(c, g, err, len))
            return false;
        size_t count = (size_t)n * 1024, conv0_floats = 0;
        for (int i = 0; i < clips; i++)
            conv0_floats += (size_t)((frames[i] + 1) / 2) * 64 * 128;
        id<MTLBuffer> x = buffer(c, @"ax", count), h = buffer(c, @"ah", count),
                      q = buffer(c, @"aq", count), k = buffer(c, @"ak", count),
                      v = buffer(c, @"av", count), a = buffer(c, @"aa", count),
                      up = buffer(c, @"aup", count * 4),
                      melbuf = buffer(c, @"mel", total_frames * 128),
                      conv0 = buffer(c, @"conv0", conv0_floats), conv1 = buffer(c, @"conv1", count),
                      pos = buffer(c, @"pos", 13 * 1024), rpe = buffer(c, @"rpe", 13 * 1024),
                      result = buffer(c, @"aresult", (size_t)n * 512), bounds = nil;
        for (int i = 0, at = 0; i < clips; at += frames[i++])
            memcpy((float *)melbuf.contents + (size_t)at * 128, mel[i], (size_t)frames[i] * 128 * 4);
        if (clips > 1) {
            bounds = buffer(c, @"abounds", (size_t)n * 2);
            uint32_t *b = bounds.contents;
            for (int i = 0, t = 0; i < clips; i++) {
                int rows = (frames[i] + 3) / 4;
                for (int r = 0; r < rows; r++, t++) {
                    b[2 * t] = (uint32_t)(t - r);
                    b[2 * t + 1] = (uint32_t)(t - r + rows);
                }
            }
        }
        float *pv = pos.contents;
        for (int p = 0; p < 13; p++)
            for (int z = 0; z < 512; z++) {
                float theta = (12 - p) * expf(-z * logf(10000) / 511);
                pv[p * 1024 + z] = sinf(theta);
                pv[p * 1024 + z + 512] = cosf(theta);
            }
        begin(c);
        for (int i = 0, mel_at = 0, c0_at = 0, row_at = 0; i < clips; i++) {
            int nt = frames[i], nf = 128, nc = 1;
            id<MTLBuffer> cur = melbuf;
            size_t cur_at = (size_t)mel_at * 128;
            for (int l = 0; l < 2; l++) {
                char name[96];
                int ot = (nt + 1) / 2, of = (nf + 1) / 2, oc = l ? 32 : 128;
                id<MTLBuffer> next = l ? conv1 : conv0;
                size_t next_at = l ? (size_t)row_at * 1024 : (size_t)c0_at;
                pipeline(c, @"g2_conv");
                bind_at(c, cur, cur_at, 0);
                snprintf(name, sizeof name, "a.conv1d.%d.weight", l);
                weight(c, g, name, 1);
                bind_at(c, next, next_at, 2);
                params(c, (G2P){.n = nt, .k = nf, .d = nc, .m = oc}, 3);
                threads(c, (size_t)ot * of * oc);
                snprintf(name, sizeof name, "a.conv1d.%d.norm.weight", l);
                pipeline(c, @"g2_norm");
                bind_at(c, next, next_at, 0);
                bind_at(c, next, next_at, 2);
                params(c, (G2P){.n = ot * of, .d = oc, .mode = 2, .type = weight(c, g, name, 1)->type},
                       3);
                groups(c, (size_t)ot * of, 1);
                cur = next;
                cur_at = next_at;
                nt = ot;
                nf = of;
                nc = oc;
            }
            mel_at += frames[i];
            c0_at += ((frames[i] + 1) / 2) * 64 * 128;
            row_at += (frames[i] + 3) / 4;
        }
        mm(c, g, "a.input_projection.weight", conv1, n, x);
        for (int l = 0; l < 12 && c->afused; l++) {
            // Clamps fold into producers (norm/SiLU/attention/post_add) and
            // GEMM epilogues; the three Gemma sandwiches become post_add_1024.
            char name[128], next[128];
            float lo, hi, lo2, hi2, lo3, hi3;
#define A(s) (snprintf(name, sizeof name, "a.blk.%d." s ".weight", l), name)
#define AN(s) (snprintf(next, sizeof next, "a.blk.%d." s ".weight", l), next)
#define AMF(s, in, outbuf) clipped_c(c, g, A(s), in, n, outbuf, true)
            input_clamp(g, A("ffn_up"), &lo, &hi);
            norm_c(c, g, A("ffn_norm"), x, n, 1024, h, 1, false, lo, hi);
            AMF("ffn_up", h, up);
            input_clamp(g, A("ffn_down"), &lo, &hi);
            element_c(c, up, up, up, count * 4, 3, 1, 0, lo, hi);
            AMF("ffn_down", up, h);
            input_clamp(g, A("attn_q"), &lo, &hi);
            input_clamp(g, A("attn_k"), &lo2, &hi2);
            input_clamp(g, A("attn_v"), &lo3, &hi3);
            bool same = lo == lo2 && hi == hi2 && lo == lo3 && hi == hi3;
            post_add_a(c, g, A("ffn_post_norm"), AN("attn_pre_norm"), h, x, h, n, .5f, 1024, true,
                       same ? lo : 0, same ? hi : 0);
            clipped_c(c, g, A("attn_q"), h, n, q, same);
            clipped_c(c, g, A("attn_k"), h, n, k, same);
            clipped_c(c, g, A("attn_v"), h, n, v, same);
            pipeline(c, @"g2_audio_scale");
            bind(c, q, 0);
            bind(c, k, 1);
            weight(c, g, A("per_dim_scale"), 2);
            params(c, (G2P){.n = n}, 3);
            threads(c, count);
            mm(c, g, A("attn_k_rel"), pos, 13, rpe);
            input_clamp(g, A("attn_out"), &lo, &hi);
            attention_c(c, q, k, v, a, n, 8, 8, 128, 0, rpe, bounds, lo, hi);
            AMF("attn_out", a, h);
            input_clamp(g, A("conv_pw1"), &lo, &hi);
            post_add_a(c, g, A("attn_post_norm"), AN("conv_norm"), h, x, h, n, 1, 1024, true, lo,
                       hi);
            AMF("conv_pw1", h, up);
            element(c, h, up, h, count, 8, 1, 0);
            pipeline(c, @"g2_depthwise");
            bind(c, h, 0);
            weight(c, g, A("conv_dw"), 1);
            bind(c, a, 2);
            params(c, (G2P){.n = n, .layer = bounds ? 1 : 0}, 3);
            bind(c, bounds ? bounds : h, 4);
            threads(c, count);
            input_clamp(g, A("conv_pw2"), &lo, &hi);
            norm_c(c, g, A("norm_conv"), a, n, 1024, a, 1, true, lo, hi);
            AMF("conv_pw2", a, h);
            element(c, x, h, x, count, 0, 1, 0);
            input_clamp(g, A("ffn_up_1"), &lo, &hi);
            norm_c(c, g, A("ffn_norm_1"), x, n, 1024, h, 1, false, lo, hi);
            AMF("ffn_up_1", h, up);
            input_clamp(g, A("ffn_down_1"), &lo, &hi);
            element_c(c, up, up, up, count * 4, 3, 1, 0, lo, hi);
            AMF("ffn_down_1", up, h);
            // ln2 normalizes the residual in place (y = x).
            post_add_a(c, g, A("ffn_post_norm_1"), AN("ln2"), h, x, x, n, .5f, 1024, true, 0, 0);
#undef AMF
#undef AN
#undef A
        }
        for (int l = 0; l < 12 && !c->afused; l++) {
            char name[128];
#define A(s) (snprintf(name, sizeof name, "a.blk.%d." s ".weight", l), name)
#define AM(s, in, outbuf) clipped(c, g, A(s), in, n, outbuf)
            norm(c, g, A("ffn_norm"), x, n, 1024, h, 1);
            AM("ffn_up", h, up);
            element(c, up, up, up, count * 4, 3, 1, 0);
            AM("ffn_down", up, h);
            norm(c, g, A("ffn_post_norm"), h, n, 1024, h, 1);
            element(c, x, h, x, count, 6, .5f, 0);
            norm(c, g, A("attn_pre_norm"), x, n, 1024, h, 1);
            AM("attn_q", h, q);
            AM("attn_k", h, k);
            AM("attn_v", h, v);
            pipeline(c, @"g2_audio_scale");
            bind(c, q, 0);
            bind(c, k, 1);
            weight(c, g, A("per_dim_scale"), 2);
            params(c, (G2P){.n = n}, 3);
            threads(c, count);
            mm(c, g, A("attn_k_rel"), pos, 13, rpe);
            attention(c, q, k, v, a, n, 8, 8, 128, 0, rpe, bounds);
            AM("attn_out", a, h);
            norm(c, g, A("attn_post_norm"), h, n, 1024, h, 1);
            element(c, x, h, x, count, 0, 1, 0);
            norm(c, g, A("conv_norm"), x, n, 1024, h, 1);
            AM("conv_pw1", h, up);
            element(c, h, up, h, count, 8, 1, 0);
            pipeline(c, @"g2_depthwise");
            bind(c, h, 0);
            weight(c, g, A("conv_dw"), 1);
            bind(c, a, 2);
            params(c, (G2P){.n = n, .layer = bounds ? 1 : 0}, 3);
            bind(c, bounds ? bounds : h, 4);
            threads(c, count);
            norm(c, g, A("norm_conv"), a, n, 1024, a, 1);
            element(c, a, a, a, count, 3, 1, 0);
            AM("conv_pw2", a, h);
            element(c, x, h, x, count, 0, 1, 0);
            norm(c, g, A("ffn_norm_1"), x, n, 1024, h, 1);
            AM("ffn_up_1", h, up);
            element(c, up, up, up, count * 4, 3, 1, 0);
            AM("ffn_down_1", up, h);
            norm(c, g, A("ffn_post_norm_1"), h, n, 1024, h, 1);
            element(c, x, h, x, count, 6, .5f, 0);
            norm(c, g, A("ln2"), x, n, 1024, x, 1);
#undef AM
#undef A
        }
        clipped(c, g, "a.pre_encode.out.weight", x, n, up);
        pipeline(c, @"g2_bias");
        bind(c, up, 0);
        weight(c, g, "a.pre_encode.out.bias", 1);
        params(c, (G2P){.n = n, .d = 1536}, 2);
        threads(c, (size_t)n * 1536);
        float plo = 0, phi = 0;
        if (c->afused)
            input_clamp(g, "mm.a.input_projection.weight", &plo, &phi);
        norm_c(c, g, NULL, up, n, 1536, up, 0, false, plo, phi);
        clipped_c(c, g, "mm.a.input_projection.weight", up, n, result, c->afused);
        if (!finish(c, err, len))
            return false;
        *tokens = n;
        *out = ei_xmalloc((size_t)n * 512 * 4);
        memcpy(*out, result.contents, (size_t)n * 512 * 4);
        return true;
    }
}

bool ei_gemma2_metal_embed(void *ctx, const int32_t *ids, const float *inputs, size_t n,
                           float out[768], char *err, size_t len) {
    size_t offsets[] = {0, n};
    return embed(ctx, ids, inputs, n, offsets, 1, out, err, len);
}
static bool batch_waves(void *ctx, const int32_t *ids, const float *inputs, const size_t *offsets,
                        size_t batch, float *out, char *err, size_t len) {
    // Bound PLE workspace to 8192 tokens while retaining packed short requests.
    size_t first = 0;
    while (first < batch) {
        size_t last = first + 1;
        while (last < batch && offsets[last + 1] - offsets[first] <= 8192)
            last++;
        size_t *local = ei_xmalloc((last - first + 1) * sizeof(size_t));
        for (size_t s = first; s <= last; s++)
            local[s - first] = offsets[s] - offsets[first];
        bool ok = embed(ctx, ids ? ids + offsets[first] : NULL,
                        inputs ? inputs + offsets[first] * 512 : NULL, local[last - first], local,
                        last - first, out + first * 768, err, len);
        free(local);
        if (!ok)
            return false;
        first = last;
    }
    return true;
}
bool ei_gemma2_metal_batch(void *ctx, const int32_t *ids, const size_t *offsets, size_t batch,
                           float *out, char *err, size_t len) {
    return batch_waves(ctx, ids, NULL, offsets, batch, out, err, len);
}
bool ei_gemma2_metal_batch_inputs(void *ctx, const float *inputs, const size_t *offsets,
                                  size_t batch, float *out, char *err, size_t len) {
    return batch_waves(ctx, NULL, inputs, offsets, batch, out, err, len);
}
