# Metal kernels

The Metal backend follows QuixiCore-Metal's standalone metallib pattern. Each
kernel family is a separate `.metal` translation unit, every source is listed
explicitly in the Makefile, and included headers are dependencies of one output
library:

```sh
make metal-kernels
```

The build is one direct compiler invocation:

```sh
xcrun -sdk macosx metal -std=metal3.1 -O2 -Wall -Wextra \
  -fno-fast-math -Isrc/metal/include <sources...> \
  -o build/embeddinggemma.metallib
```

`engine_metal.m` loads the metallib next to the executable unless
`EI_METALLIB_PATH` overrides it, creates all 24 compute pipelines at engine
initialization, owns persistent model/RoPE/workspace buffers, encodes the full
24-layer graph into one command buffer, and synchronizes once per embedding.

## Kernel Inventory

| Family | Functions | Purpose |
|---|---:|---|
| embedding | 1 | Q8_0 token gather, dequantization, and embedding scale |
| generic Q4 projection | 4 | R1/R4 direct GEMV and staged 32x8/16x16 GEMM |
| Q4 x Q8 projection | 1 | CPU-parity activation-quantized diagnostic route |
| Q8 quantization | 1 | F32 to GGUF Q8_0 blocks |
| fused QKV projection | 4 | R1/R4 direct and staged 32x8/16x16 variants |
| fused up/gate projection | 4 | R1/R4 direct and staged 32x8/16x16 variants |
| normalization | 2 | RMSNorm and fused projection RMSNorm plus residual |
| Q/K plus RoPE | 2 | single-tensor and fused Q+K norm/NEOX RoPE/Q scale |
| attention | 1 | online-softmax GQA with full or symmetric SWA masking |
| elementwise | 3 | gated GELU, fallback add, and fallback scale |
| pooling | 1 | output RMSNorm, mean pooling, and L2 normalization |

All quantized kernels consume GGUF blocks directly. Q4_0 blocks are 18 bytes
and Q8_0 blocks are 34 bytes; the backend has no weight repacking step.

## Runtime Routes

Production projection dispatch is model-specialized:

- T=1..6: one output row per simdgroup (`*_gemv`).
- T=7..2048: four output rows per simdgroup (`*_gemv_r4`).
- Staged GEMM: disabled by the default threshold of 2049, retained for
  diagnostics and future GPU generations.

The three projection families use the same boundary. The fused QKV kernels
write Q/K/V in one dispatch and fused up/gate kernels write both FFN branches
in one dispatch. A 32-thread simdgroup is the reduction unit for direct
projection, norm/RoPE, attention, and pooling.

Run `make test-metal` to load every function, create every pipeline, check
llama.cpp embedding goldens, compare the production R1/R4 route with CPU, and
execute both staged GEMM tile variants for route parity.
