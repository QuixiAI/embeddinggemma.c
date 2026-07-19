#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static void fail(NSString *message) {
    fprintf(stderr, "%s\n", message.UTF8String);
    exit(1);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) fail(@"usage: test_metal <metallib>");

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) fail(@"Metal device is unavailable");

        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
        if (library == nil) {
            fail([NSString stringWithFormat:@"failed to load metallib: %@", error]);
        }

        NSArray<NSString *> *expected = @[
            @"ei_embedding_q8_0_f32",
            @"ei_q4_0_f32_gemv",
            @"ei_q4_0_f32_gemv_r4",
            @"ei_q4_0_f32_gemm",
            @"ei_q4_0_f32_gemm_16x16",
            @"ei_q4_0_f32_qkv_gemv",
            @"ei_q4_0_f32_qkv_gemv_r4",
            @"ei_q4_0_f32_qkv_gemv_triple",
            @"ei_q4_0_f32_qkv_gemm",
            @"ei_q4_0_f32_qkv_gemm_16x16",
            @"ei_q4_0_f32_up_gate_gemv",
            @"ei_q4_0_f32_up_gate_gemv_r4",
            @"ei_q4_0_f32_up_gate_gelu_gemv",
            @"ei_q4_0_f32_up_gate_gelu_gemv_r2",
            @"ei_q4_0_f32_up_gate_gelu_gemv_r4",
            @"ei_q4_0_f32_up_gate_gemm",
            @"ei_q4_0_f32_up_gate_gemm_16x16",
            @"ei_q4_0_q8_0_gemv",
            @"ei_quantize_q8_0_f32",
            @"ei_rms_norm_f32",
            @"ei_rms_norm_residual_f32",
            @"ei_qk_norm_rope_f32",
            @"ei_qk_norm_rope_qk_f32",
            @"ei_attention_f32",
            @"ei_gelu_mul_f32",
            @"ei_vec_add_f32",
            @"ei_vec_scale_f32",
            @"ei_mean_pool_rms_l2_f32",
        ];

        for (NSString *name in expected) {
            id<MTLFunction> function = [library newFunctionWithName:name];
            if (function == nil) {
                fail([NSString stringWithFormat:@"missing Metal function %@", name]);
            }
            error = nil;
            id<MTLComputePipelineState> pipeline =
                [device newComputePipelineStateWithFunction:function error:&error];
            if (pipeline == nil) {
                fail([NSString stringWithFormat:@"failed to create pipeline %@: %@", name, error]);
            }
        }

        printf("Metal library: %lu/%lu functions and pipelines ready on %s\n",
               (unsigned long)expected.count, (unsigned long)library.functionNames.count,
               device.name.UTF8String);
    }
    return 0;
}
