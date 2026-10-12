// Grid barrier probe: can G threadgroups of 128 threads rendezvous on a device
// atomic, and what does one barrier cost (GPU timestamps) with and without the
// device-scope fence? Spins are capped so a non-resident group count fails fast
// instead of hanging the GPU (an orphaned spinning kernel survives its process).
// build: clang -fobjc-arc -framework Metal -framework Foundation -o grid_barrier_probe perf/grid_barrier_probe.m
// usage: grid_barrier_probe [groups] [barriers]   (default 128 groups, 264 barriers)
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
int main(int argc, char **argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        uint groups = argc > 1 ? atoi(argv[1]) : 128, iters = argc > 2 ? atoi(argv[2]) : 264;
        const char *variants[] = {"none", "seqcst", "relacq", "relaxed_fence"};
        const char *fence_pre[] = {"", "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);",
            "atomic_thread_fence(mem_flags::mem_device, memory_order_release, thread_scope_device);",
            "atomic_thread_fence(mem_flags::mem_device, memory_order_relaxed, thread_scope_device);"};
        const char *fence_post[] = {"", "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);",
            "atomic_thread_fence(mem_flags::mem_device, memory_order_acquire, thread_scope_device);",
            "atomic_thread_fence(mem_flags::mem_device, memory_order_relaxed, thread_scope_device);"};
        for (int vi = 0; vi < 4; vi++) {
            NSString *src = [NSString stringWithFormat:@"#include <metal_stdlib>\nusing namespace metal;\n"
                "kernel void bar(device atomic_uint *ctr [[buffer(0)]], device atomic_uint *fail [[buffer(1)]], constant uint &groups [[buffer(2)]], constant uint &iters [[buffer(3)]], device float *work [[buffer(4)]],"
                " uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {\n"
                "  threadgroup uint ok;\n"
                "  for (uint it = 0; it < iters; ++it) {\n"
                "    work[g * 128 + tid] += 1.0f;\n"
                "    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);\n"
                "    if (tid == 0) {\n"
                "      %s\n"
                "      atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);\n"
                "      uint target = (it + 1) * groups, spins = 0, good = 1;\n"
                "      while (atomic_load_explicit(ctr, memory_order_relaxed) < target) { if (++spins > 200000u) { atomic_store_explicit(fail, 1u, memory_order_relaxed); good = 0; break; } }\n"
                "      %s\n"
                "      ok = good;\n"
                "    }\n"
                "    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);\n"
                "    if (!ok) return;\n"
                "  }\n"
                "}\n", fence_pre[vi], fence_post[vi]];
            NSError *err = nil;
            MTLCompileOptions *opts = [MTLCompileOptions new];
            opts.languageVersion = MTLLanguageVersion3_2;
            id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
            if (!lib) { printf("%s compile: %s\n", variants[vi], err.localizedDescription.UTF8String); continue; }
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bar"] error:&err];
            id<MTLCommandQueue> q = [dev newCommandQueue];
            id<MTLBuffer> ctr = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> fail = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> work = [dev newBufferWithLength:groups * 128 * 4 options:MTLResourceStorageModeShared];
            double best = 1e9; int failed = 0;
            for (int rep = 0; rep < 5; rep++) {
                *(uint *)ctr.contents = 0; *(uint *)fail.contents = 0;
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:ctr offset:0 atIndex:0]; [enc setBuffer:fail offset:0 atIndex:1];
                [enc setBytes:&groups length:4 atIndex:2]; [enc setBytes:&iters length:4 atIndex:3];
                [enc setBuffer:work offset:0 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                double gpu = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
                if (*(uint *)fail.contents) failed++; else if (gpu < best) best = gpu;
            }
            printf("%-14s %4u groups x 128: %u barriers best %.3f ms = %.2f us/barrier, %d/5 failed\n", variants[vi], groups, iters, best, best * 1e3 / iters, failed);
            fflush(stdout);
        }
    }
    return 0;
}
