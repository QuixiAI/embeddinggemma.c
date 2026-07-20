#import "engine_metal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach-o/getsect.h>
#import <mach-o/ldsyms.h>

#include <math.h>

static void metal_set_error(char *err, size_t err_len, NSString *message) {
    if (err && err_len) snprintf(err, err_len, "%s", message.UTF8String);
}

static id<MTLLibrary> new_metal_library(id<MTLDevice> device,
                                        const char *library_path,
                                        NSError **error) {
    const char *override = library_path && *library_path
        ? library_path : getenv("EI_METALLIB_PATH");
    if (override && *override) {
        NSString *path = [NSString stringWithUTF8String:override];
        return [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:error];
    }

    unsigned long size = 0;
    const uint8_t *bytes = getsectiondata(&_mh_execute_header, "__DATA",
                                          "__metallib", &size);
    if (!bytes || size == 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"embeddinggemma.c"
                                         code:1
                                     userInfo:@{
                NSLocalizedDescriptionKey: @"embedded Metal library is missing"
            }];
        }
        return nil;
    }
    dispatch_data_t data = dispatch_data_create(
        bytes, size, NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    return [device newLibraryWithData:data error:error];
}

static id<MTLBuffer> new_rope_table(id<MTLDevice> device, float base) {
    const NSUInteger value_count = EI_N_CTX * EI_HEAD_DIM;
    float *values = ei_xmalloc(sizeof(float) * value_count);
    for (uint32_t pos = 0; pos < EI_N_CTX; pos++) {
        float *row = values + (size_t)pos * EI_HEAD_DIM;
        for (uint32_t dim = 0; dim < EI_HEAD_DIM / 2; dim++) {
            float theta = (float)pos
                * powf(base, -2.0f * (float)dim / (float)EI_HEAD_DIM);
            row[2 * dim] = cosf(theta);
            row[2 * dim + 1] = sinf(theta);
        }
    }
    id<MTLBuffer> buffer = [device newBufferWithBytes:values
                                               length:sizeof(float) * value_count
                                              options:MTLResourceStorageModeShared];
    free(values);
    return buffer;
}

@interface EIMetalEngine : NSObject {
@public
    const ei_model *_model;
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;
    id<MTLBuffer> _model_buffer;
    id<MTLBuffer> _rope_full;
    id<MTLBuffer> _rope_swa;
    NSDictionary<NSString *, id<MTLComputePipelineState>> *_pipelines;

    NSUInteger _capacity;
    NSUInteger _batch_capacity;
    id<MTLBuffer> _ids;
    id<MTLBuffer> _positions;
    id<MTLBuffer> _sequence_ids;
    id<MTLBuffer> _offsets;
    id<MTLBuffer> _x;
    id<MTLBuffer> _norm;
    id<MTLBuffer> _q;
    id<MTLBuffer> _k;
    id<MTLBuffer> _v;
    id<MTLBuffer> _k_f16;
    id<MTLBuffer> _v_f16;
    id<MTLBuffer> _ctx;
    id<MTLBuffer> _up;
    id<MTLBuffer> _gate;
    id<MTLBuffer> _tmp;
    id<MTLBuffer> _result;
    uint32_t _gemm_min_tokens;
    uint32_t _gemm_tile_tokens;
    uint32_t _gemv_r4_min_tokens;
    uint32_t _fp16_kv_min_tokens;
    bool _fp16_kv_active;
    bool _fused_qk_rope;
    bool _fused_up_gate_gelu;
    bool _fused_residual_next_norm;
    uint32_t _fused_residual_next_max_tokens;
    uint32_t _fused_up_gate_rows;
    bool _triple_qkv_gemv;
}

- (instancetype)initWithModel:(const ei_model *)model
                   libraryPath:(const char *)libraryPath
                         error:(NSString **)errorMessage;
- (BOOL)embedTokens:(const int32_t *)ids
               count:(NSUInteger)count
              output:(float *)output
               error:(NSString **)errorMessage;
- (BOOL)embedTokensBatch:(const int32_t *)ids
                 offsets:(const size_t *)offsets
               batchSize:(NSUInteger)batchSize
                  output:(float *)output
                   error:(NSString **)errorMessage;
- (BOOL)ensureCapacity:(NSUInteger)count
             batchSize:(NSUInteger)batchSize
                 error:(NSString **)errorMessage;
@end

static id<MTLComputePipelineState> metal_pipeline(EIMetalEngine *engine, NSString *name) {
    return engine->_pipelines[name];
}

static bool metal_direct_fp16_kv(EIMetalEngine *engine, uint32_t tokens) {
    return engine->_fp16_kv_active && engine->_fused_qk_rope &&
        tokens < engine->_gemm_min_tokens &&
        tokens >= engine->_gemv_r4_min_tokens;
}

static void metal_dispatch_groups(id<MTLComputeCommandEncoder> encoder,
                                  NSUInteger gx, NSUInteger gy, NSUInteger gz,
                                  NSUInteger tx, NSUInteger ty, NSUInteger tz) {
    [encoder dispatchThreadgroups:MTLSizeMake(gx, gy, gz)
            threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
}

static void metal_dispatch_threads(id<MTLComputeCommandEncoder> encoder,
                                   NSUInteger count, NSUInteger width) {
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}

static void encode_embedding(EIMetalEngine *engine, id<MTLComputeCommandEncoder> encoder,
                             NSUInteger tokens) {
    uint32_t count = (uint32_t)tokens;
    float scale = sqrtf((float)EI_N_EMBD);
    [encoder setComputePipelineState:metal_pipeline(engine, @"ei_embedding_q8_0_f32")];
    [encoder setBuffer:engine->_model_buffer offset:engine->_model->token_embd->offset atIndex:0];
    [encoder setBuffer:engine->_ids offset:0 atIndex:1];
    [encoder setBuffer:engine->_x offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof count atIndex:3];
    [encoder setBytes:&scale length:sizeof scale atIndex:4];
    metal_dispatch_threads(encoder, tokens * EI_N_EMBD, 256);
}

static void encode_rms(EIMetalEngine *engine, id<MTLComputeCommandEncoder> encoder,
                       id<MTLBuffer> input, const float *weight, id<MTLBuffer> output,
                       uint32_t rows, uint32_t cols, float eps) {
    NSUInteger weight_offset = (const uint8_t *)weight
        - (engine->_model->gguf.map + engine->_model->gguf.data_off);
    [encoder setComputePipelineState:metal_pipeline(engine, @"ei_rms_norm_f32")];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:engine->_model_buffer offset:weight_offset atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&rows length:sizeof rows atIndex:3];
    [encoder setBytes:&cols length:sizeof cols atIndex:4];
    [encoder setBytes:&eps length:sizeof eps atIndex:5];
    metal_dispatch_groups(encoder, rows, 1, 1, 32, 1, 1);
}

static void encode_q4_projection(EIMetalEngine *engine,
                                 id<MTLComputeCommandEncoder> encoder,
                                 const ei_tensor *weight, id<MTLBuffer> input,
                                 id<MTLBuffer> output, uint32_t tokens) {
    uint32_t rows = (uint32_t)weight->ne[1];
    uint32_t cols = (uint32_t)weight->ne[0];
    bool use_gemm = tokens >= engine->_gemm_min_tokens;
    bool tile16 = use_gemm && engine->_gemm_tile_tokens == 16;
    bool rows4 = !use_gemm && tokens >= engine->_gemv_r4_min_tokens;
    NSString *name = use_gemm
        ? (tile16 ? @"ei_q4_0_f32_gemm_16x16" : @"ei_q4_0_f32_gemm")
        : (rows4 ? @"ei_q4_0_f32_gemv_r4" : @"ei_q4_0_f32_gemv");
    [encoder setComputePipelineState:metal_pipeline(engine, name)];
    [encoder setBuffer:engine->_model_buffer offset:weight->offset atIndex:0];
    [encoder setBuffer:input offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&rows length:sizeof rows atIndex:3];
    [encoder setBytes:&cols length:sizeof cols atIndex:4];
    [encoder setBytes:&tokens length:sizeof tokens atIndex:5];
    if (use_gemm) {
        uint32_t tile = tile16 ? 16 : 8;
        uint32_t row_tile = tile16 ? 16 : 32;
        metal_dispatch_groups(encoder, (rows + row_tile - 1) / row_tile,
                              (tokens + tile - 1) / tile, 1,
                              256, 1, 1);
    } else {
        metal_dispatch_groups(encoder, rows4 ? (rows + 3) / 4 : rows,
                              tokens, 1, 32, 1, 1);
    }
}

static void encode_qkv_projection(EIMetalEngine *engine,
                                  id<MTLComputeCommandEncoder> encoder,
                                  const ei_layer *layer, uint32_t tokens) {
    bool use_gemm = tokens >= engine->_gemm_min_tokens;
    bool tile16 = use_gemm && engine->_gemm_tile_tokens == 16;
    bool rows4 = !use_gemm && tokens >= engine->_gemv_r4_min_tokens;
    bool triple = !use_gemm && !rows4 && engine->_triple_qkv_gemv;
    bool direct_fp16_kv = metal_direct_fp16_kv(engine, tokens);
    NSString *name = direct_fp16_kv
        ? @"ei_q4_0_f32_qkv_gemv_r4_v_f16"
        : (triple
        ? @"ei_q4_0_f32_qkv_gemv_triple"
        : (use_gemm
        ? (tile16 ? @"ei_q4_0_f32_qkv_gemm_16x16" : @"ei_q4_0_f32_qkv_gemm")
        : (rows4 ? @"ei_q4_0_f32_qkv_gemv_r4"
                 : @"ei_q4_0_f32_qkv_gemv")));
    [encoder setComputePipelineState:metal_pipeline(engine, name)];
    [encoder setBuffer:engine->_model_buffer offset:layer->attn_q->offset atIndex:0];
    [encoder setBuffer:engine->_model_buffer offset:layer->attn_k->offset atIndex:1];
    [encoder setBuffer:engine->_model_buffer offset:layer->attn_v->offset atIndex:2];
    [encoder setBuffer:engine->_norm offset:0 atIndex:3];
    [encoder setBuffer:engine->_q offset:0 atIndex:4];
    [encoder setBuffer:engine->_k offset:0 atIndex:5];
    [encoder setBuffer:(direct_fp16_kv ? engine->_v_f16 : engine->_v)
                 offset:0 atIndex:6];
    [encoder setBytes:&tokens length:sizeof tokens atIndex:7];
    if (use_gemm) {
        metal_dispatch_groups(encoder, tile16 ? 80 : 40,
                              (tokens + (tile16 ? 15 : 7)) / (tile16 ? 16 : 8),
                              1, 256, 1, 1);
    } else {
        metal_dispatch_groups(encoder, triple ? EI_N_EMBD
                                              : (rows4 ? 1280 / 4 : 1280),
                              tokens, 1, 32, 1, 1);
    }
}

static bool encode_up_gate_projection(EIMetalEngine *engine,
                                      id<MTLComputeCommandEncoder> encoder,
                                      const ei_layer *layer, uint32_t tokens) {
    bool use_gemm = tokens >= engine->_gemm_min_tokens;
    bool tile16 = use_gemm && engine->_gemm_tile_tokens == 16;
    bool rows4 = !use_gemm && tokens >= engine->_gemv_r4_min_tokens;
    bool fused = !use_gemm && engine->_fused_up_gate_gelu;
    NSString *name = fused
        ? (rows4
            ? (engine->_fused_up_gate_rows == 4
                ? @"ei_q4_0_f32_up_gate_gelu_gemv_r4"
                : @"ei_q4_0_f32_up_gate_gelu_gemv_r2")
            : @"ei_q4_0_f32_up_gate_gelu_gemv")
        : (use_gemm
            ? (tile16 ? @"ei_q4_0_f32_up_gate_gemm_16x16"
                      : @"ei_q4_0_f32_up_gate_gemm")
            : (rows4 ? @"ei_q4_0_f32_up_gate_gemv_r4"
                     : @"ei_q4_0_f32_up_gate_gemv"));
    [encoder setComputePipelineState:metal_pipeline(engine, name)];
    [encoder setBuffer:engine->_model_buffer offset:layer->ffn_up->offset atIndex:0];
    [encoder setBuffer:engine->_model_buffer offset:layer->ffn_gate->offset atIndex:1];
    [encoder setBuffer:engine->_norm offset:0 atIndex:2];
    if (fused) {
        [encoder setBuffer:engine->_gate offset:0 atIndex:3];
        [encoder setBytes:&tokens length:sizeof tokens atIndex:4];
        metal_dispatch_groups(encoder,
                              rows4 ? EI_N_FF / engine->_fused_up_gate_rows
                                    : EI_N_FF,
                              tokens, 1, 32, 1, 1);
    } else if (use_gemm) {
        [encoder setBuffer:engine->_up offset:0 atIndex:3];
        [encoder setBuffer:engine->_gate offset:0 atIndex:4];
        [encoder setBytes:&tokens length:sizeof tokens atIndex:5];
        metal_dispatch_groups(encoder, tile16 ? 144 : 72,
                              (tokens + (tile16 ? 15 : 7)) / (tile16 ? 16 : 8),
                              1, 256, 1, 1);
    } else {
        [encoder setBuffer:engine->_up offset:0 atIndex:3];
        [encoder setBuffer:engine->_gate offset:0 atIndex:4];
        [encoder setBytes:&tokens length:sizeof tokens atIndex:5];
        metal_dispatch_groups(encoder, rows4 ? 2 * EI_N_FF / 4 : 2 * EI_N_FF,
                              tokens, 1, 32, 1, 1);
    }
    return fused;
}

static void encode_qk_norm_rope(EIMetalEngine *engine,
                                id<MTLComputeCommandEncoder> encoder,
                                id<MTLBuffer> x, const float *weight,
                                uint32_t tokens, uint32_t heads, uint32_t row_stride,
                                float eps, id<MTLBuffer> rope_table, float output_scale) {
    NSUInteger weight_offset = (const uint8_t *)weight
        - (engine->_model->gguf.map + engine->_model->gguf.data_off);
    [encoder setComputePipelineState:metal_pipeline(engine, @"ei_qk_norm_rope_f32")];
    [encoder setBuffer:x offset:0 atIndex:0];
    [encoder setBuffer:engine->_model_buffer offset:weight_offset atIndex:1];
    [encoder setBytes:&tokens length:sizeof tokens atIndex:2];
    [encoder setBytes:&heads length:sizeof heads atIndex:3];
    [encoder setBytes:&row_stride length:sizeof row_stride atIndex:4];
    [encoder setBytes:&eps length:sizeof eps atIndex:5];
    [encoder setBuffer:rope_table offset:0 atIndex:6];
    [encoder setBytes:&output_scale length:sizeof output_scale atIndex:7];
    metal_dispatch_groups(encoder, heads, tokens, 1, 32, 1, 1);
}

static void encode_qk_norm_rope_fused(EIMetalEngine *engine,
                                      id<MTLComputeCommandEncoder> encoder,
                                      const ei_layer *layer, uint32_t tokens,
                                      float eps, id<MTLBuffer> rope_table) {
    const uint8_t *model_base = engine->_model->gguf.map + engine->_model->gguf.data_off;
    NSUInteger q_weight_offset = (const uint8_t *)layer->attn_q_norm - model_base;
    NSUInteger k_weight_offset = (const uint8_t *)layer->attn_k_norm - model_base;
    bool direct_fp16_kv = metal_direct_fp16_kv(engine, tokens);
    NSString *name = direct_fp16_kv ? @"ei_qk_norm_rope_q_f32_k_f16"
                                    : @"ei_qk_norm_rope_qk_f32";
    [encoder setComputePipelineState:metal_pipeline(engine, name)];
    [encoder setBuffer:engine->_q offset:0 atIndex:0];
    [encoder setBuffer:engine->_k offset:0 atIndex:1];
    [encoder setBuffer:engine->_model_buffer offset:q_weight_offset atIndex:2];
    [encoder setBuffer:engine->_model_buffer offset:k_weight_offset atIndex:3];
    [encoder setBytes:&tokens length:sizeof tokens atIndex:4];
    [encoder setBytes:&eps length:sizeof eps atIndex:5];
    [encoder setBuffer:rope_table offset:0 atIndex:6];
    if (direct_fp16_kv) [encoder setBuffer:engine->_k_f16 offset:0 atIndex:7];
    metal_dispatch_groups(encoder, EI_N_HEAD + EI_N_HEAD_KV, tokens, 1, 32, 1, 1);
}

static void encode_qk_norm_rope_batch(EIMetalEngine *engine,
                                      id<MTLComputeCommandEncoder> encoder,
                                      id<MTLBuffer> x, const float *weight,
                                      uint32_t tokens, uint32_t heads,
                                      uint32_t row_stride, float eps,
                                      id<MTLBuffer> rope_table, float output_scale) {
    NSUInteger weight_offset = (const uint8_t *)weight
        - (engine->_model->gguf.map + engine->_model->gguf.data_off);
    [encoder setComputePipelineState:metal_pipeline(engine, @"ei_qk_norm_rope_f32_batch")];
    [encoder setBuffer:x offset:0 atIndex:0];
    [encoder setBuffer:engine->_model_buffer offset:weight_offset atIndex:1];
    [encoder setBytes:&tokens length:sizeof tokens atIndex:2];
    [encoder setBytes:&heads length:sizeof heads atIndex:3];
    [encoder setBytes:&row_stride length:sizeof row_stride atIndex:4];
    [encoder setBytes:&eps length:sizeof eps atIndex:5];
    [encoder setBuffer:rope_table offset:0 atIndex:6];
    [encoder setBytes:&output_scale length:sizeof output_scale atIndex:7];
    [encoder setBuffer:engine->_positions offset:0 atIndex:8];
    metal_dispatch_groups(encoder, heads, tokens, 1, 32, 1, 1);
}

static void encode_qk_norm_rope_fused_batch(EIMetalEngine *engine,
                                            id<MTLComputeCommandEncoder> encoder,
                                            const ei_layer *layer, uint32_t tokens,
                                            float eps, id<MTLBuffer> rope_table) {
    const uint8_t *model_base = engine->_model->gguf.map + engine->_model->gguf.data_off;
    NSUInteger q_weight_offset = (const uint8_t *)layer->attn_q_norm - model_base;
    NSUInteger k_weight_offset = (const uint8_t *)layer->attn_k_norm - model_base;
    bool direct_fp16_kv = metal_direct_fp16_kv(engine, tokens);
    NSString *name = direct_fp16_kv ? @"ei_qk_norm_rope_q_f32_k_f16_batch"
                                    : @"ei_qk_norm_rope_qk_f32_batch";
    [encoder setComputePipelineState:metal_pipeline(engine, name)];
    [encoder setBuffer:engine->_q offset:0 atIndex:0];
    [encoder setBuffer:engine->_k offset:0 atIndex:1];
    [encoder setBuffer:engine->_model_buffer offset:q_weight_offset atIndex:2];
    [encoder setBuffer:engine->_model_buffer offset:k_weight_offset atIndex:3];
    [encoder setBytes:&tokens length:sizeof tokens atIndex:4];
    [encoder setBytes:&eps length:sizeof eps atIndex:5];
    [encoder setBuffer:rope_table offset:0 atIndex:6];
    [encoder setBuffer:engine->_positions offset:0 atIndex:7];
    if (direct_fp16_kv) [encoder setBuffer:engine->_k_f16 offset:0 atIndex:8];
    metal_dispatch_groups(encoder, EI_N_HEAD + EI_N_HEAD_KV, tokens, 1, 32, 1, 1);
}

static void encode_gelu_mul(EIMetalEngine *engine, id<MTLComputeCommandEncoder> encoder,
                            uint32_t count) {
    [encoder setComputePipelineState:metal_pipeline(engine, @"ei_gelu_mul_f32")];
    [encoder setBuffer:engine->_gate offset:0 atIndex:0];
    [encoder setBuffer:engine->_up offset:0 atIndex:1];
    [encoder setBytes:&count length:sizeof count atIndex:2];
    metal_dispatch_threads(encoder, count, 256);
}

static void encode_kv_f16(EIMetalEngine *engine, id<MTLComputeCommandEncoder> encoder,
                          uint32_t tokens) {
    uint32_t count = tokens * EI_HEAD_DIM;
    [encoder setComputePipelineState:metal_pipeline(engine, @"ei_kv_f32_to_f16")];
    [encoder setBuffer:engine->_k offset:0 atIndex:0];
    [encoder setBuffer:engine->_v offset:0 atIndex:1];
    [encoder setBuffer:engine->_k_f16 offset:0 atIndex:2];
    [encoder setBuffer:engine->_v_f16 offset:0 atIndex:3];
    [encoder setBytes:&count length:sizeof count atIndex:4];
    metal_dispatch_threads(encoder, count, 256);
}

static void encode_attention(EIMetalEngine *engine, id<MTLComputeCommandEncoder> encoder,
                             uint32_t tokens, uint32_t window) {
    bool fp16_kv = engine->_fp16_kv_active;
    NSString *name = fp16_kv ? @"ei_attention_f16_kv" : @"ei_attention_f32";
    [encoder setComputePipelineState:metal_pipeline(engine, name)];
    [encoder setBuffer:engine->_q offset:0 atIndex:0];
    [encoder setBuffer:fp16_kv ? engine->_k_f16 : engine->_k offset:0 atIndex:1];
    [encoder setBuffer:fp16_kv ? engine->_v_f16 : engine->_v offset:0 atIndex:2];
    [encoder setBuffer:engine->_ctx offset:0 atIndex:3];
    [encoder setBytes:&tokens length:sizeof tokens atIndex:4];
    [encoder setBytes:&window length:sizeof window atIndex:5];
    metal_dispatch_groups(encoder, tokens, EI_N_HEAD, 1, 32, 1, 1);
}

static void encode_attention_batch(EIMetalEngine *engine,
                                   id<MTLComputeCommandEncoder> encoder,
                                   uint32_t tokens, uint32_t window) {
    bool fp16_kv = engine->_fp16_kv_active;
    NSString *name = fp16_kv ? @"ei_attention_f16_kv_batch"
                             : @"ei_attention_f32_batch";
    [encoder setComputePipelineState:metal_pipeline(engine, name)];
    [encoder setBuffer:engine->_q offset:0 atIndex:0];
    [encoder setBuffer:fp16_kv ? engine->_k_f16 : engine->_k offset:0 atIndex:1];
    [encoder setBuffer:fp16_kv ? engine->_v_f16 : engine->_v offset:0 atIndex:2];
    [encoder setBuffer:engine->_ctx offset:0 atIndex:3];
    [encoder setBuffer:engine->_offsets offset:0 atIndex:4];
    [encoder setBuffer:engine->_sequence_ids offset:0 atIndex:5];
    [encoder setBytes:&tokens length:sizeof tokens atIndex:6];
    [encoder setBytes:&window length:sizeof window atIndex:7];
    metal_dispatch_groups(encoder, tokens, EI_N_HEAD, 1, 32, 1, 1);
}

static void encode_rms_residual(EIMetalEngine *engine,
                                id<MTLComputeCommandEncoder> encoder,
                                id<MTLBuffer> input, const float *weight,
                                id<MTLBuffer> residual, uint32_t rows,
                                uint32_t cols, float eps) {
    NSUInteger weight_offset = (const uint8_t *)weight
        - (engine->_model->gguf.map + engine->_model->gguf.data_off);
    [encoder setComputePipelineState:metal_pipeline(engine, @"ei_rms_norm_residual_f32")];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:engine->_model_buffer offset:weight_offset atIndex:1];
    [encoder setBuffer:residual offset:0 atIndex:2];
    [encoder setBytes:&rows length:sizeof rows atIndex:3];
    [encoder setBytes:&cols length:sizeof cols atIndex:4];
    [encoder setBytes:&eps length:sizeof eps atIndex:5];
    metal_dispatch_groups(encoder, rows, 1, 1, 32, 1, 1);
}

static void encode_rms_residual_next(EIMetalEngine *engine,
                                     id<MTLComputeCommandEncoder> encoder,
                                     id<MTLBuffer> input,
                                     const float *post_weight,
                                     id<MTLBuffer> residual,
                                     const float *next_weight,
                                     id<MTLBuffer> next_output,
                                     uint32_t rows, uint32_t cols, float eps) {
    const uint8_t *model_base = engine->_model->gguf.map +
                                engine->_model->gguf.data_off;
    NSUInteger post_offset = (const uint8_t *)post_weight - model_base;
    NSUInteger next_offset = (const uint8_t *)next_weight - model_base;
    [encoder setComputePipelineState:metal_pipeline(
        engine, @"ei_rms_norm_residual_next_f32")];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:engine->_model_buffer offset:post_offset atIndex:1];
    [encoder setBuffer:residual offset:0 atIndex:2];
    [encoder setBuffer:engine->_model_buffer offset:next_offset atIndex:3];
    [encoder setBuffer:next_output offset:0 atIndex:4];
    [encoder setBytes:&rows length:sizeof rows atIndex:5];
    [encoder setBytes:&cols length:sizeof cols atIndex:6];
    [encoder setBytes:&eps length:sizeof eps atIndex:7];
    metal_dispatch_groups(encoder, rows, 1, 1, 32, 1, 1);
}

static void encode_pool(EIMetalEngine *engine, id<MTLComputeCommandEncoder> encoder,
                        uint32_t tokens, float eps) {
    NSUInteger weight_offset = (const uint8_t *)engine->_model->output_norm
        - (engine->_model->gguf.map + engine->_model->gguf.data_off);
    [encoder setComputePipelineState:metal_pipeline(engine, @"ei_mean_pool_rms_l2_f32")];
    [encoder setBuffer:engine->_x offset:0 atIndex:0];
    [encoder setBuffer:engine->_model_buffer offset:weight_offset atIndex:1];
    [encoder setBuffer:engine->_result offset:0 atIndex:2];
    [encoder setBytes:&tokens length:sizeof tokens atIndex:3];
    [encoder setBytes:&eps length:sizeof eps atIndex:4];
    metal_dispatch_groups(encoder, 1, 1, 1, 32, 1, 1);
}

static void encode_pool_batch(EIMetalEngine *engine,
                              id<MTLComputeCommandEncoder> encoder,
                              uint32_t batch_size, float eps) {
    NSUInteger weight_offset = (const uint8_t *)engine->_model->output_norm
        - (engine->_model->gguf.map + engine->_model->gguf.data_off);
    [encoder setComputePipelineState:metal_pipeline(engine, @"ei_mean_pool_rms_l2_f32_batch")];
    [encoder setBuffer:engine->_x offset:0 atIndex:0];
    [encoder setBuffer:engine->_model_buffer offset:weight_offset atIndex:1];
    [encoder setBuffer:engine->_result offset:0 atIndex:2];
    [encoder setBuffer:engine->_offsets offset:0 atIndex:3];
    [encoder setBytes:&batch_size length:sizeof batch_size atIndex:4];
    [encoder setBytes:&eps length:sizeof eps atIndex:5];
    metal_dispatch_groups(encoder, batch_size, 1, 1, 32, 1, 1);
}

@implementation EIMetalEngine

- (instancetype)initWithModel:(const ei_model *)model
                   libraryPath:(const char *)libraryPath
                         error:(NSString **)errorMessage {
    self = [super init];
    if (!self) return nil;
    _model = model;
    _gemm_min_tokens = UINT32_MAX;
    _gemm_tile_tokens = 16;
    _gemv_r4_min_tokens = 7;
    _fp16_kv_min_tokens = 1024;
    _fused_residual_next_max_tokens = 128;
    const char *fused_residual_next = getenv(
        "EI_METAL_FUSED_RESIDUAL_NEXT_NORM");
    _fused_residual_next_norm = !fused_residual_next ||
        strcmp(fused_residual_next, "0") != 0;
    const char *fused_residual_max = getenv(
        "EI_METAL_FUSED_RESIDUAL_NEXT_MAX_TOKENS");
    if (fused_residual_max && *fused_residual_max) {
        char *end = NULL;
        long parsed = strtol(fused_residual_max, &end, 10);
        if (*end != '\0' || parsed < 1 || parsed > 65536) {
            if (errorMessage) {
                *errorMessage =
                    @"EI_METAL_FUSED_RESIDUAL_NEXT_MAX_TOKENS must be 1..65536";
            }
            return nil;
        }
        _fused_residual_next_max_tokens = (uint32_t)parsed;
    }
    const char *fused_qk_rope = getenv("EI_METAL_FUSED_QK_ROPE");
    _fused_qk_rope = !fused_qk_rope || strcmp(fused_qk_rope, "0") != 0;
    const char *fused_up_gate_gelu = getenv("EI_METAL_FUSED_UP_GATE_GELU");
    _fused_up_gate_gelu = fused_up_gate_gelu &&
        strcmp(fused_up_gate_gelu, "0") != 0;
    const char *triple_qkv_gemv = getenv("EI_METAL_TRIPLE_QKV_GEMV");
    _triple_qkv_gemv = triple_qkv_gemv &&
        strcmp(triple_qkv_gemv, "0") != 0;
    _fused_up_gate_rows = 2;
    const char *fused_up_gate_rows = getenv("EI_METAL_FUSED_UP_GATE_ROWS");
    if (fused_up_gate_rows && *fused_up_gate_rows) {
        char *end = NULL;
        long parsed = strtol(fused_up_gate_rows, &end, 10);
        if (*end != '\0' || (parsed != 2 && parsed != 4)) {
            if (errorMessage) *errorMessage =
                @"EI_METAL_FUSED_UP_GATE_ROWS must be 2 or 4";
            return nil;
        }
        _fused_up_gate_rows = (uint32_t)parsed;
    }
    const char *gemm_min = getenv("EI_METAL_GEMM_MIN_TOKENS");
    if (gemm_min && *gemm_min) {
        char *end = NULL;
        long parsed = strtol(gemm_min, &end, 10);
        if (*end != '\0' || parsed < 1 || parsed > 65536) {
            if (errorMessage) *errorMessage = @"EI_METAL_GEMM_MIN_TOKENS must be 1..65536";
            return nil;
        }
        _gemm_min_tokens = (uint32_t)parsed;
    }
    const char *gemv_r4_min = getenv("EI_METAL_GEMV_R4_MIN_TOKENS");
    if (gemv_r4_min && *gemv_r4_min) {
        char *end = NULL;
        long parsed = strtol(gemv_r4_min, &end, 10);
        if (*end != '\0' || parsed < 1 || parsed > 64) {
            if (errorMessage) *errorMessage = @"EI_METAL_GEMV_R4_MIN_TOKENS must be 1..64";
            return nil;
        }
        _gemv_r4_min_tokens = (uint32_t)parsed;
    }
    const char *fp16_kv_min = getenv("EI_METAL_FP16_KV_MIN_TOKENS");
    if (fp16_kv_min && *fp16_kv_min) {
        char *end = NULL;
        long parsed = strtol(fp16_kv_min, &end, 10);
        if (*end != '\0' || parsed < 1 || parsed > 65536) {
            if (errorMessage) {
                *errorMessage = @"EI_METAL_FP16_KV_MIN_TOKENS must be 1..65536";
            }
            return nil;
        }
        _fp16_kv_min_tokens = (uint32_t)parsed;
    }
    const char *gemm_tile = getenv("EI_METAL_GEMM_TILE_TOKENS");
    if (gemm_tile && *gemm_tile) {
        char *end = NULL;
        long parsed = strtol(gemm_tile, &end, 10);
        if (*end != '\0' || (parsed != 8 && parsed != 16)) {
            if (errorMessage) *errorMessage = @"EI_METAL_GEMM_TILE_TOKENS must be 8 or 16";
            return nil;
        }
        _gemm_tile_tokens = (uint32_t)parsed;
    }
    _device = MTLCreateSystemDefaultDevice();
    if (!_device) {
        if (errorMessage) *errorMessage = @"Metal device is unavailable";
        return nil;
    }
    _queue = [_device newCommandQueue];
    if (!_queue) {
        if (errorMessage) *errorMessage = @"failed to create Metal command queue";
        return nil;
    }

    NSError *error = nil;
    id<MTLLibrary> library = new_metal_library(_device, libraryPath, &error);
    if (!library) {
        if (errorMessage) {
            *errorMessage = [NSString stringWithFormat:
                @"failed to load Metal library: %@", error];
        }
        return nil;
    }

    NSArray<NSString *> *names = @[
        @"ei_embedding_q8_0_f32", @"ei_q4_0_f32_gemv",
        @"ei_q4_0_f32_gemv_r4",
        @"ei_q4_0_f32_gemm",
        @"ei_q4_0_f32_gemm_16x16",
        @"ei_q4_0_f32_qkv_gemv", @"ei_q4_0_f32_qkv_gemv_r4",
        @"ei_q4_0_f32_qkv_gemv_r4_v_f16",
        @"ei_q4_0_f32_qkv_gemv_triple",
        @"ei_q4_0_f32_qkv_gemm",
        @"ei_q4_0_f32_qkv_gemm_16x16",
        @"ei_q4_0_f32_up_gate_gemv", @"ei_q4_0_f32_up_gate_gemv_r4",
        @"ei_q4_0_f32_up_gate_gelu_gemv",
        @"ei_q4_0_f32_up_gate_gelu_gemv_r2",
        @"ei_q4_0_f32_up_gate_gelu_gemv_r4",
        @"ei_q4_0_f32_up_gate_gemm",
        @"ei_q4_0_f32_up_gate_gemm_16x16",
        @"ei_q4_0_q8_0_gemv", @"ei_quantize_q8_0_f32", @"ei_rms_norm_f32",
        @"ei_rms_norm_residual_f32", @"ei_rms_norm_residual_next_f32",
        @"ei_qk_norm_rope_f32", @"ei_attention_f32",
        @"ei_attention_f16_kv",
        @"ei_qk_norm_rope_qk_f32",
        @"ei_qk_norm_rope_q_f32_k_f16",
        @"ei_qk_norm_rope_f32_batch", @"ei_qk_norm_rope_qk_f32_batch",
        @"ei_qk_norm_rope_q_f32_k_f16_batch",
        @"ei_attention_f32_batch", @"ei_attention_f16_kv_batch",
        @"ei_mean_pool_rms_l2_f32_batch",
        @"ei_gelu_mul_f32", @"ei_vec_add_f32", @"ei_vec_scale_f32",
        @"ei_kv_f32_to_f16",
        @"ei_mean_pool_rms_l2_f32",
    ];
    NSMutableDictionary *pipelines = [NSMutableDictionary dictionaryWithCapacity:names.count];
    for (NSString *name in names) {
        id<MTLFunction> function = [library newFunctionWithName:name];
        if (!function) {
            if (errorMessage) *errorMessage = [NSString stringWithFormat:@"missing kernel %@", name];
            return nil;
        }
        id<MTLComputePipelineState> pipeline =
            [_device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            if (errorMessage) {
                *errorMessage = [NSString stringWithFormat:@"pipeline %@ failed: %@", name, error];
            }
            return nil;
        }
        pipelines[name] = pipeline;
    }
    _pipelines = [pipelines copy];

    const uint8_t *model_bytes = model->gguf.map + model->gguf.data_off;
    NSUInteger model_length = model->gguf.map_len - model->gguf.data_off;
    _model_buffer = [_device newBufferWithBytes:model_bytes
                                         length:model_length
                                        options:MTLResourceStorageModeShared];
    if (!_model_buffer) {
        if (errorMessage) *errorMessage = @"failed to allocate Metal model buffer";
        return nil;
    }
    _model_buffer.label = @"embeddinggemma model";
    _rope_full = new_rope_table(_device, model->rope_base_full);
    _rope_swa = new_rope_table(_device, model->rope_base_swa);
    if (!_rope_full || !_rope_swa) {
        if (errorMessage) *errorMessage = @"failed to allocate Metal RoPE tables";
        return nil;
    }
    _result = [_device newBufferWithLength:sizeof(float) * EI_N_EMBD
                                   options:MTLResourceStorageModeShared];
    if (!_result) {
        if (errorMessage) *errorMessage = @"failed to allocate Metal result buffer";
        return nil;
    }
    _batch_capacity = 1;
    return self;
}

- (BOOL)ensureCapacity:(NSUInteger)count
             batchSize:(NSUInteger)batchSize
                 error:(NSString **)errorMessage {
    if (_capacity < count) {
    NSUInteger capacity = _capacity ? _capacity : 16;
    while (capacity < count) capacity *= 2;

    MTLResourceOptions private_options = MTLResourceStorageModePrivate;
    _ids = [_device newBufferWithLength:capacity * sizeof(int32_t)
                                options:MTLResourceStorageModeShared];
    _positions = [_device newBufferWithLength:capacity * sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared];
    _sequence_ids = [_device newBufferWithLength:capacity * sizeof(uint32_t)
                                         options:MTLResourceStorageModeShared];
    _x = [_device newBufferWithLength:capacity * EI_N_EMBD * sizeof(float) options:private_options];
    _norm = [_device newBufferWithLength:capacity * EI_N_EMBD * sizeof(float) options:private_options];
    _q = [_device newBufferWithLength:capacity * EI_N_EMBD * sizeof(float) options:private_options];
    _k = [_device newBufferWithLength:capacity * EI_HEAD_DIM * sizeof(float) options:private_options];
    _v = [_device newBufferWithLength:capacity * EI_HEAD_DIM * sizeof(float) options:private_options];
    if (_fp16_kv_min_tokens <= EI_N_CTX) {
        _k_f16 = [_device newBufferWithLength:capacity * EI_HEAD_DIM * sizeof(uint16_t)
                                      options:private_options];
        _v_f16 = [_device newBufferWithLength:capacity * EI_HEAD_DIM * sizeof(uint16_t)
                                      options:private_options];
    }
    _ctx = [_device newBufferWithLength:capacity * EI_N_EMBD * sizeof(float) options:private_options];
    _up = [_device newBufferWithLength:capacity * EI_N_FF * sizeof(float) options:private_options];
    _gate = [_device newBufferWithLength:capacity * EI_N_FF * sizeof(float) options:private_options];
    _tmp = [_device newBufferWithLength:capacity * EI_N_EMBD * sizeof(float) options:private_options];
    if (!_ids || !_positions || !_sequence_ids || !_x || !_norm || !_q || !_k ||
        !_v || (_fp16_kv_min_tokens <= EI_N_CTX && (!_k_f16 || !_v_f16)) ||
        !_ctx || !_up || !_gate || !_tmp) {
        if (errorMessage) *errorMessage = @"failed to allocate Metal activation workspace";
        return NO;
    }
    _capacity = capacity;
    }
    if (_batch_capacity < batchSize) {
        NSUInteger capacity = _batch_capacity ? _batch_capacity : 1;
        while (capacity < batchSize) capacity *= 2;
        _offsets = [_device newBufferWithLength:(capacity + 1) * sizeof(uint32_t)
                                       options:MTLResourceStorageModeShared];
        _result = [_device newBufferWithLength:capacity * EI_N_EMBD * sizeof(float)
                                      options:MTLResourceStorageModeShared];
        if (!_offsets || !_result) {
            if (errorMessage) *errorMessage = @"failed to allocate Metal batch metadata";
            return NO;
        }
        _batch_capacity = capacity;
    } else if (!_offsets) {
        _offsets = [_device newBufferWithLength:2 * sizeof(uint32_t)
                                       options:MTLResourceStorageModeShared];
        if (!_offsets) {
            if (errorMessage) *errorMessage = @"failed to allocate Metal batch offsets";
            return NO;
        }
    }
    return YES;
}

- (BOOL)embedTokens:(const int32_t *)ids
               count:(NSUInteger)count
              output:(float *)output
               error:(NSString **)errorMessage {
    const size_t offsets[2] = { 0, count };
    return [self embedTokensBatch:ids offsets:offsets batchSize:1
                           output:output error:errorMessage];
}

- (BOOL)embedTokensBatch:(const int32_t *)ids
                 offsets:(const size_t *)offsets
               batchSize:(NSUInteger)batchSize
                  output:(float *)output
                   error:(NSString **)errorMessage {
    NSUInteger count = offsets[batchSize];
    if (![self ensureCapacity:count batchSize:batchSize error:errorMessage]) return NO;
    NSUInteger max_sequence_tokens = 0;
    for (NSUInteger sequence = 0; sequence < batchSize; sequence++) {
        NSUInteger sequence_tokens = offsets[sequence + 1] - offsets[sequence];
        if (sequence_tokens > max_sequence_tokens) max_sequence_tokens = sequence_tokens;
    }
    _fp16_kv_active = max_sequence_tokens >= _fp16_kv_min_tokens;
    memcpy(_ids.contents, ids, count * sizeof(int32_t));
    uint32_t *metal_offsets = _offsets.contents;
    uint32_t *positions = _positions.contents;
    uint32_t *sequence_ids = _sequence_ids.contents;
    for (NSUInteger sequence = 0; sequence <= batchSize; sequence++) {
        metal_offsets[sequence] = (uint32_t)offsets[sequence];
    }
    for (NSUInteger sequence = 0; sequence < batchSize; sequence++) {
        for (NSUInteger token = offsets[sequence]; token < offsets[sequence + 1]; token++) {
            positions[token] = (uint32_t)(token - offsets[sequence]);
            sequence_ids[token] = (uint32_t)sequence;
        }
    }

    id<MTLCommandBuffer> command = [_queue commandBuffer];
    command.label = @"embeddinggemma forward";
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder) {
        if (errorMessage) *errorMessage = @"failed to create Metal compute encoder";
        return NO;
    }

    uint32_t tokens = (uint32_t)count;
    const float eps = _model->rms_eps;
    const bool fused_residual_next = _fused_residual_next_norm &&
        tokens <= _fused_residual_next_max_tokens;
    encode_embedding(self, encoder, count);
    for (int layer_index = 0; layer_index < EI_N_LAYER; layer_index++) {
        const ei_layer *layer = &_model->layers[layer_index];
        const bool swa = ei_layer_is_swa(_model, layer_index);
        id<MTLBuffer> rope_table = swa ? _rope_swa : _rope_full;

        if (layer_index == 0 || !fused_residual_next) {
            encode_rms(self, encoder, _x, layer->attn_norm, _norm,
                       tokens, EI_N_EMBD, eps);
        }
        encode_qkv_projection(self, encoder, layer, tokens);
        if (batchSize > 1 && _fused_qk_rope) {
            encode_qk_norm_rope_fused_batch(self, encoder, layer, tokens, eps, rope_table);
        } else if (batchSize > 1) {
            encode_qk_norm_rope_batch(self, encoder, _q, layer->attn_q_norm,
                                      tokens, EI_N_HEAD, EI_N_EMBD, eps, rope_table, 0.0625f);
            encode_qk_norm_rope_batch(self, encoder, _k, layer->attn_k_norm,
                                      tokens, EI_N_HEAD_KV, EI_HEAD_DIM, eps, rope_table, 1.0f);
        } else if (_fused_qk_rope) {
            encode_qk_norm_rope_fused(self, encoder, layer, tokens, eps, rope_table);
        } else {
            encode_qk_norm_rope(self, encoder, _q, layer->attn_q_norm,
                                tokens, EI_N_HEAD, EI_N_EMBD, eps, rope_table, 0.0625f);
            encode_qk_norm_rope(self, encoder, _k, layer->attn_k_norm,
                                tokens, EI_N_HEAD_KV, EI_HEAD_DIM, eps, rope_table, 1.0f);
        }
        if (_fp16_kv_active && !metal_direct_fp16_kv(self, tokens)) {
            encode_kv_f16(self, encoder, tokens);
        }
        if (batchSize > 1) {
            encode_attention_batch(self, encoder, tokens, swa ? _model->swa_window : 0);
        } else {
            encode_attention(self, encoder, tokens, swa ? _model->swa_window : 0);
        }
        encode_q4_projection(self, encoder, layer->attn_output, _ctx, _tmp, tokens);
        if (fused_residual_next) {
            encode_rms_residual_next(self, encoder, _tmp,
                layer->post_attention_norm, _x, layer->ffn_norm, _norm,
                tokens, EI_N_EMBD, eps);
        } else {
            encode_rms_residual(self, encoder, _tmp,
                layer->post_attention_norm, _x, tokens, EI_N_EMBD, eps);
            encode_rms(self, encoder, _x, layer->ffn_norm, _norm,
                       tokens, EI_N_EMBD, eps);
        }
        bool fused_up_gate = encode_up_gate_projection(self, encoder, layer, tokens);
        if (!fused_up_gate) encode_gelu_mul(self, encoder, tokens * EI_N_FF);
        encode_q4_projection(self, encoder, layer->ffn_down, _gate, _tmp, tokens);
        if (fused_residual_next && layer_index + 1 < EI_N_LAYER) {
            encode_rms_residual_next(self, encoder, _tmp,
                layer->post_ffw_norm, _x,
                _model->layers[layer_index + 1].attn_norm, _norm,
                tokens, EI_N_EMBD, eps);
        } else {
            encode_rms_residual(self, encoder, _tmp, layer->post_ffw_norm,
                                _x, tokens, EI_N_EMBD, eps);
        }
    }
    if (batchSize > 1) encode_pool_batch(self, encoder, (uint32_t)batchSize, eps);
    else encode_pool(self, encoder, tokens, eps);
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        if (errorMessage) {
            *errorMessage = [NSString stringWithFormat:@"Metal forward failed: %@", command.error];
        }
        return NO;
    }
    memcpy(output, _result.contents, batchSize * sizeof(float) * EI_N_EMBD);
    return YES;
}

@end

void *ei_metal_engine_create(const ei_model *model, const char *library_path,
                             char *err, size_t err_len) {
    @autoreleasepool {
        NSString *message = nil;
        EIMetalEngine *engine = [[EIMetalEngine alloc] initWithModel:model
                                                        libraryPath:library_path
                                                              error:&message];
        if (!engine) {
            metal_set_error(err, err_len, message ?: @"failed to initialize Metal backend");
            return NULL;
        }
        if (err && err_len) err[0] = '\0';
        return (__bridge_retained void *)engine;
    }
}

void ei_metal_engine_free(void *handle) {
    if (handle) CFBridgingRelease(handle);
}

bool ei_metal_engine_reserve(void *handle, size_t total_tokens,
                             size_t batch_size, char *err, size_t err_len) {
    @autoreleasepool {
        EIMetalEngine *engine = (__bridge EIMetalEngine *)handle;
        NSString *message = nil;
        BOOL ok = [engine ensureCapacity:total_tokens
                               batchSize:batch_size error:&message];
        if (!ok) {
            metal_set_error(err, err_len,
                            message ?: @"Metal workspace reservation failed");
            return false;
        }
        if (err && err_len) err[0] = '\0';
        return true;
    }
}

bool ei_metal_engine_embed_tokens(void *handle, const int32_t *ids, size_t n_tokens,
                                  float out[EI_N_EMBD], char *err, size_t err_len) {
    @autoreleasepool {
        EIMetalEngine *engine = (__bridge EIMetalEngine *)handle;
        NSString *message = nil;
        BOOL ok = [engine embedTokens:ids count:n_tokens output:out error:&message];
        if (!ok) {
            metal_set_error(err, err_len, message ?: @"Metal forward failed");
            return false;
        }
        if (err && err_len) err[0] = '\0';
        return true;
    }
}

bool ei_metal_engine_embed_tokens_batch(void *handle, const int32_t *ids,
                                        const size_t *offsets, size_t batch_size,
                                        float *out, char *err, size_t err_len) {
    @autoreleasepool {
        EIMetalEngine *engine = (__bridge EIMetalEngine *)handle;
        NSString *message = nil;
        BOOL ok = [engine embedTokensBatch:ids offsets:offsets batchSize:batch_size
                                    output:out error:&message];
        if (!ok) {
            metal_set_error(err, err_len, message ?: @"Metal batch forward failed");
            return false;
        }
        if (err && err_len) err[0] = '\0';
        return true;
    }
}
