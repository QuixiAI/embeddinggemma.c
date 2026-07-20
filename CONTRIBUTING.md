# Contributing

Contributions should preserve the project's narrow scope: standalone inference
and serving for `embeddinggemma-300M-qat-Q4_0.gguf` on CPU, Metal, CUDA, ROCm,
and Intel XPU SYCL.

## Before Starting

- Open an issue before changing the HTTP contract, model graph, quantization
  format, dependency policy, supported platform matrix, or release artifacts.
- Keep changes scoped. Do not combine kernel work with unrelated server or
  formatting changes.
- Do not commit GGUF files, model weights, generated benchmark runs, build
  output, or managed XPU dependencies.
- Read `EXTRACTION.md` for llama.cpp algorithm provenance and `PORT_SPEC.md` for
  the fixed graph and backend contract.
- For performance work, read `perf/perf.md` and the current entries in
  `perf/optimization_status.md` before changing a kernel.

The project is MIT licensed. Model acquisition and use are separate from this
repository and remain subject to the model provider's terms.

## Development Setup

The default model path is:

```text
${XDG_CACHE_HOME:-$HOME/.cache}/embeddinggemma.c/embeddinggemma-300M-qat-Q4_0.gguf
```

The server can populate that cache with `curl` or `wget`. Tests use `MODEL`, so
an existing file can be selected explicitly:

```sh
make test MODEL=/path/to/embeddinggemma-300M-qat-Q4_0.gguf
```

Common requirements are a C11 compiler, `make`, POSIX threads, Python 3, and a
local model. Backend toolchains are platform-specific:

| backend | required toolchain/runtime |
|---|---|
| CPU | Clang or GCC |
| Metal | full Xcode with the Metal Toolchain component |
| CUDA | CUDA toolkit, cuBLAS, and a supported NVIDIA driver |
| ROCm | ROCm HIP, hipBLAS, and an AMDGPU runtime |
| XPU SYCL | Intel oneAPI DPC++, oneMKL, and Level Zero |

Run `make help` for the current build, test, performance, and release targets.

## Code Map

- `src/engine.c` and `src/kernels.c`: portable CPU graph and kernels.
- `src/engine_metal.m` and `src/metal/`: Metal host code and kernel families.
- `src/engine_cuda.cu`: CUDA engine and kernels.
- `src/engine_rocm.hip`: ROCm engine and kernels.
- `src/engine_xpu*.cpp`: portable SYCL and specialized Xe2 routes.
- `src/inference_service.c`: batching, tokenization, singleflight, and result
  caching.
- `src/server.c`: HTTP transport, request parsing, and response caching.
- `perf/`: native harnesses, Python runners, retained results, and experiment
  decisions.

Prefer the existing backend boundary and model-specialized shapes over generic
abstractions. Shared behavior belongs in the portable engine only when all
backends can preserve the same contract.

## Build And Test

### Build targets

Build the portable CPU server:

```sh
make
```

The output is `build/embeddinggemma`. The CPU route includes scalar C, ARM64
NEON, AVX2, and SSSE3 kernels and selects the best available implementation at
runtime.

Build the self-contained Metal server:

```sh
make metal
```

The output is `build/embeddinggemma-metal`. Every `.metal` translation unit is
compiled in one direct `xcrun metal` invocation, and the resulting metallib is
embedded in the executable's Mach-O `__DATA,__metallib` section. The
`build/embeddinggemma.metallib` file is an inspection intermediate, not a
runtime sidecar. Darwin builds default to `MACOSX_DEPLOYMENT_TARGET=14.0`.

Build the portable CUDA server:

```sh
make cuda NVCC=/usr/local/cuda/bin/nvcc
```

The output is `build/embeddinggemma-cuda`. The default release-style build
contains native code for every architecture reported by the installed CUDA
compiler plus PTX for its newest target. Use `CUDA_ARCHS=86` only for a focused
developer build.

Build the portable ROCm server:

```sh
make rocm HIPCC=/opt/rocm/bin/hipcc
```

The output is `build/embeddinggemma-rocm`. The default fat binary covers
`gfx908`, `gfx90a`, `gfx942`, and `gfx950`. `ROCM_ARCHS="gfx90a gfx942"` or the
legacy `ROCM_ARCH=gfx942` may reduce local build time but must not be used for a
release.

Build the Intel XPU SYCL server:

```sh
source /opt/intel/oneapi/setvars.sh
make xpu XPU_XE2_FLASH=1 SYCL_CXX=icpx
```

The output is `build/embeddinggemma-xpu`. `XPU_XE2_FLASH=1` fetches the pinned
`vllm-xpu-kernels` and SYCL-TLA revisions into `.xpu-deps` and enables the Xe2
FlashAttention and packed-W4 routes. `make xpu-deps` fetches those dependencies
without building.

Each accelerator binary defaults to its compiled backend. `--backend cpu` is a
diagnostic override; `--backend auto` requests fallback behavior.

### Xcode and Metal

Metal compilation requires full Xcode and the separately installed Metal
Toolchain component. Command Line Tools alone are insufficient.

```sh
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -runFirstLaunch
xcodebuild -downloadComponent MetalToolchain
```

Verify the selected toolchain before diagnosing the Makefile:

```sh
xcode-select -p
xcodebuild -version
xcodebuild -showComponent MetalToolchain -json
xcrun --find metal
xcrun metal --version
```

The component status must be `installed`. If the command-line download cannot
reach Apple's catalog, install **Metal Toolchain** from **Xcode > Settings >
Components**, then ensure `xcode-select` points at that Xcode installation. The
Makefile prefers `/Applications/Xcode.app/Contents/Developer` through
`DEVELOPER_DIR` when no explicit value is supplied.

### Test matrix

CPU changes must pass:

```sh
make clean-cpu
make test
```

Backend changes must pass the complete suite on matching hardware:

```sh
make test-metal
make test-cuda NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCHS=86
make test-rocm HIPCC=/opt/rocm/bin/hipcc ROCM_ARCHS=gfx942
source /opt/intel/oneapi/setvars.sh
make test-xpu XPU_XE2_FLASH=1 SYCL_CXX=icpx
```

Architecture overrides above are for fast local validation. Release builds
must use the portable architecture sets in `RELEASE.md`.

Every inference change must retain:

- Exact tokenizer output against checked-in vectors.
- Cosine similarity of at least `0.999` against llama.cpp embedding goldens.
- Backend drift and packed-batch thresholds enforced by the backend suite.
- Correct HTTP output at 768, 512, 256, and 128 Matryoshka dimensions.
- Bounded queue, workspace, and client-batch behavior.

Run the shell checks after changing scripts or release tooling:

```sh
make check
```

The suites validate the GGUF manifest, tokenizer vectors, quantized kernels,
full and sliding-window attention, fused routes, ten llama.cpp embedding
goldens, packed batches, and all Matryoshka dimensions. Release acceptance is
cosine similarity of at least `0.999` against llama.cpp. Backend-specific
synthetic drift thresholds are additional regression guards, not replacements
for that acceptance test.

## Backend Implementation Notes

### CPU

The CPU engine uses Q4_0 x Q8_0 projection kernels with scalar, NEON, AVX2, and
SSSE3 implementations; SIMD quantization, normalization, elementwise, and
attention kernels; and a persistent pthread pool. Retained model-specific
fusions include Q/K norm plus RoPE, triple QKV projection, adjacent three-row
projection, RMS norm plus Q8 activation quantization, and a four-activation-row
projection route for large flattened batches.

Diagnostic controls:

- `EI_THREADS`: override the CPU worker count.
- `EI_CPU_SHORT_THREADS`: override the short-projection worker width.
- `EI_CPU_DUAL_PROJECTION`: control paired projection accumulation.
- `EI_CPU_FUSED_RMS_QUANT=0`: materialize normalized activations before Q8.
- `EI_CPU_MULTIROW_MIN_TOKENS=0..65536`: disable or move the multirow boundary.
- `EI_CPU_FUSED_GELU_QUANT=1`: enable the rejected fused activation experiment.

### Metal

Metal uses one-row direct Q4 GEMV at T=1..6 and four-row direct Q4 GEMV at
T=7..2048. Retained fusions cover QKV and up/gate projections, Q/K norm plus
RoPE, residual plus next RMS norm, and final norm/pooling. RoPE tables are
precomputed. Attention uses online GQA, with FP16 K/V at sequence lengths of
1024 and above while accumulation remains FP32.

Diagnostic controls:

- `EI_METALLIB_PATH`: load an alternate metallib for kernel experiments.
- `EI_METAL_FUSED_QK_ROPE=0`: restore separate Q and K dispatches.
- `EI_METAL_GEMV_R4_MIN_TOKENS=1..64`: move the R1/R4 boundary.
- `EI_METAL_GEMM_MIN_TOKENS=1..65536`: enable staged GEMM at a threshold.
- `EI_METAL_GEMM_TILE_TOKENS=8|16`: select the staged tile.
- `EI_METAL_FP16_KV_MIN_TOKENS=1..65536`: move the FP16 K/V boundary.
- `EI_METAL_FUSED_RESIDUAL_NEXT_NORM=0`: disable residual/next-norm fusion.
- `EI_METAL_FUSED_RESIDUAL_NEXT_MAX_TOKENS=1..65536`: move its upper boundary.
- `EI_METAL_FUSED_UP_GATE_GELU=1`: enable the rejected fused FFN experiment.
- `EI_METAL_FUSED_UP_GATE_ROWS=2|4`: select its row grouping.
- `EI_METAL_TRIPLE_QKV_GEMV=1`: enable the rejected short QKV experiment.

### CUDA

CUDA uses fused RMS plus Q8 activation quantization and packed Q4_0 x Q8_0
`dp4a` projections at T=1..4. Larger batches use weights expanded once at model
load and cuBLAS FP16 tensor-core GEMM with FP32 accumulation. Combined QKV and
up/gate GEMMs, direct FP16 QKV epilogues, shared-tile online attention,
tensor-core QK/PV, symmetric-SWA query bands, and CUDA graph replay are retained.

Diagnostic controls:

- `EI_CUDA_DEVICE=0..N`: select the CUDA device.
- `EI_CUDA_GEMM_MIN_TOKENS=1..65536`: move the direct/tensor-core boundary.
- `EI_CUDA_Q8_LATENCY=0|1`: select FP32 Q4 or fused Q8/DP4A at short lengths.
- `EI_CUDA_FP16_ATTENTION_MIN_TOKENS=1..65536`: move FP16 attention storage.
- `EI_CUDA_DIRECT_FP16_QKV=0|1`: control the combined direct QKV epilogue.
- `EI_CUDA_DIRECT_FP16_CONTEXT=0|1`: control direct FP16 context output.
- `EI_CUDA_FLASH_ATTENTION_MIN_TOKENS` and
  `EI_CUDA_FLASH_ATTENTION_MAX_TOKENS`: bound shared-tile attention.
- `EI_CUDA_TENSOR_ATTENTION_MIN_TOKENS`: override the batch-aware crossover.
- `EI_CUDA_SWA_TENSOR_TILE_TOKENS=0|128|256|512|1024`: select SWA banding.
- `EI_CUDA_SWA_TENSOR_MIN_TOKENS`: move the banded-SWA boundary.
- `EI_CUDA_NATIVE_Q4_GEMM=0|1`: enable the experimental packed-Q4 MMA route.

### ROCm

ROCm keeps packed Q4 weights resident and also expands them to FP16 for large
hipBLAS GEMMs. Direct packed-wave kernels serve short requests, native Q4 MFMA
serves the middle range, and hipBLAS serves large batches. Retained routes
include combined projections, residual/next-RMS fusion, a fused MFMA FFN
epilogue, FP16 attention storage, batched GQA tensor attention, and exact V-only
attention for independent one-token sequences.

Diagnostic controls:

- `EI_ROCM_DEVICE=0..N`: select the HIP device.
- `EI_ROCM_GEMM_MIN_TOKENS`: move the direct/native boundary.
- `EI_ROCM_SINGLETON_DIRECT_MAX_TOKENS`: move the singleton direct ceiling.
- `EI_ROCM_NATIVE_Q4_GEMM` and `EI_ROCM_NATIVE_Q4_MAX_TOKENS`: control MFMA.
- `EI_ROCM_NATIVE_Q4_FUSED` and `EI_ROCM_NATIVE_Q4_FUSED_ACTIVATION`: control
  combined MFMA projections and their FFN epilogue.
- `EI_ROCM_FP16_ATTENTION_MIN_TOKENS`: move FP16 attention storage.
- `EI_ROCM_TENSOR_ATTENTION_MIN_TOKENS`: override dense attention routing.
- `EI_ROCM_BATCHED_TENSOR_ATTENTION`: control batched GQA tensor calls.
- `EI_ROCM_FP16_ATTENTION_SCORES`: control FP16 dense score storage.
- `EI_ROCM_SWA_TENSOR_TILE_TOKENS`: select or disable SWA query banding.
- `EI_ROCM_DIRECT_FP16_CONTEXT` and `EI_ROCM_RMS_REGISTER_CACHE`: control
  retained conversion and residual/RMS fusions.
- `EI_ROCM_SINGLE_TOKEN_V_ONLY`, `EI_ROCM_SINGLETON_METADATA_ELISION`, and
  `EI_ROCM_FINAL_SINGLETON_POOL`: control singleton fast paths.
- `EI_ROCM_DIRECT_RMS_FUSION`, `EI_ROCM_DIRECT_Q4_PAIR`, and
  `EI_ROCM_FUSED_EMBEDDING_RMS`: control direct-route launch reductions.

Rejected ROCm experiments remain behind `EI_ROCM_COMMAND_GRAPH`,
`EI_ROCM_MFMA_ATTENTION`, `EI_ROCM_NATIVE_Q4_WIDE`,
`EI_ROCM_NATIVE_Q4_DIRECT_FP16_QKV`, `EI_ROCM_DIRECT_Q4_QUAD`, and
`EI_ROCM_PINNED_IO_STAGING` for architecture-specific retesting.

### XPU SYCL

XPU keeps model/workspace allocations resident, expands Q4_0 weights to FP16
once, and uses oneMKL XMX GEMM with combined projections. It includes fused
FP16 QKV norm/RoPE, batch-aware online/dense attention, one-token V-only
attention, cooperative RMS and pooling, Xe2 FlashAttention, and a specialized
Xe2 W4 route.

Diagnostic controls:

- `EI_XPU_DEVICE=0..N`: select the Level Zero device.
- `EI_XPU_TENSOR_ATTENTION_MIN_TOKENS`: override dense attention routing.
- `EI_XPU_FP16_ATTENTION=0|1|auto`: control FP16 Q/K/V storage.
- `EI_XPU_SINGLE_TOKEN_V_ONLY=0|1`: control singleton V-only attention.
- `EI_XPU_COOPERATIVE_RMS_MAX_ROWS`: move the cooperative RMS boundary.
- `EI_XPU_RMS_REGISTER_CACHE=0|1`: control register-cached residual/RMS.
- `EI_XPU_COOPERATIVE_POOL=0|1|auto`: control cooperative final pooling.
- `EI_XPU_XE2_FLASH=0|1|auto`: control the Xe2 Flash route.
- `EI_XPU_XE2_FLASH_MIN_TOKENS` and `EI_XPU_XE2_FLASH_BATCH_MIN_TOKENS`: move
  its single and packed-batch thresholds.
- `EI_XPU_Q4_M_TILED=1`: enable direct-Q4 M2-M8 weight reuse diagnostics.
- `EI_XPU_COMMAND_GRAPH=0|1|auto`: control exact-shape graph replay.
- `EI_XPU_XE2_W4=0|1`: control the Xe2 W4A16 up/gate route.
- `XPU_ONEDNN=1`, `EI_XPU_ONEDNN_F16=1`, and `EI_XPU_ONEDNN_W4=1`: build and
  select rejected oneDNN experiments.

Route defaults were selected on the hardware recorded in
`perf/optimization_status.md`. Re-benchmark before changing them for another
GPU generation.

## Runtime And Serving

The canonical HTTP port is `42666`. `--bind`, `--port`, and `--model` override
the listener and model path. The default model path is documented above;
resolution order is `--model`, `EI_MODEL_PATH`, then that cache path.

Important production controls:

- `--workers N` and `--max-queue N`: bound HTTP concurrency and backlog.
- `--cache-entries N`: set final-embedding LRU capacity.
- `--max-batch-tokens N` and `--max-batch-requests N`: set packed batch limits.
- `--max-batch-sequence-tokens N`: cap sequences eligible for packing.
- `--max-client-batch-size N`: bound one HTTP request's input array.
- `--tokenizer-workers N`: set persistent tokenizer workers.
- `--batch-wait-us N`: bound microbatch collection delay.
- `--keepalive-connections N`, `--keepalive-max-requests N`, and
  `--keepalive-timeout-ms N`: control persistent HTTP connections.
- `--response-cache-mb N`: size the exact float-JSON response LRU.
- `EI_BATCH_LOOKAHEAD=0`: restore strict FIFO batching for diagnostics.
- `EI_ADAPTIVE_BATCH_WAIT=0`: always apply the configured collection delay.
- `EI_HTTP_KEEPALIVE=0`: disable persistent HTTP connections.

The inference owner preallocates token/output buffers, and accelerator engines
reserve production workspace before listening. HTTP admission, client batch
size, and backend batch size are deliberately separate bounds.

The canonical embedding cache key is the exact token-ID sequence. Concurrent
duplicates coalesce onto one execution; completed 768-dimensional embeddings
are retained and all Matryoshka dimensions reuse that entry. The server-level
response LRU additionally caches exact float-JSON response bodies. Transformer
prefix KV caching is invalid for EmbeddingGemma's bidirectional attention: a
prefix state changes when later tokens are present.

## Performance Changes

Treat optimization as an experiment, not an intuition-only change:

1. Record the machine, compiler, driver, commit, shapes, warmups, and iteration
   count.
2. Establish a same-process or alternating-order baseline.
3. Change one bottleneck hypothesis at a time.
4. Run correctness before trusting timing data.
5. Retain low-risk changes only when the representative gain is stable and at
   least 3%; document useful neutral and rejected experiments too.
6. Measure single requests, packed batches, and service concurrency. Include
   token lengths 1, 32, 512, and 2048 where the route applies.
7. Append each pass and its decision to `perf/optimization_status.md`.

Raw machine-specific runs belong under ignored `perf/results/`. Summaries that
justify retained behavior belong in the tracked optimization log.

Useful entry points:

```sh
make perf
make perf-engine
make perf-engine-metal
make perf-engine-cuda
make perf-engine-rocm
make perf-engine-xpu XPU_XE2_FLASH=1
make perf-concurrency
make perf-concurrency-cuda
make perf-concurrency-rocm
make perf-concurrency-xpu XPU_XE2_FLASH=1
make perf-batch
make perf-tokenization

python3 perf/bench_engine.py --backend both --tokens 1,7,32,128,512,2048
python3 perf/bench_concurrency.py --backend both --tokens 32 \
  --concurrency 1,2,4,8,16,32
python3 perf/bench_dimensions.py --backend metal --encoding-format both
python3 perf/bench_http.py --backend metal --keepalive on --response-cache-mb 64
```

The detailed measurement protocol, shape priorities, concurrency rules, and
Matryoshka methodology are in `perf/perf.md`. Every run writes structured data
under `perf/results/`; append only durable decisions and representative numbers
to `perf/optimization_status.md`.

## Style

- Keep portable code at C11. Backend translation units may use Objective-C,
  CUDA C++, HIP C++, or SYCL C++ as required.
- Preserve `-Wall -Wextra -Werror` cleanliness for host C code.
- Use ASCII unless an existing file requires otherwise.
- Add comments only for non-obvious invariants, synchronization, numerical
  choices, or architecture constraints.
- Avoid new runtime dependencies unless they are available on every supported
  deployment or isolated to an optional backend.

## Pull Requests

Describe the behavioral change, hardware and toolchain used, exact test
commands, correctness result, and before/after performance for kernel work.
Call out untested backends explicitly. Do not update `VERSION`, create a tag,
or change pinned install examples unless the release owner has approved the
exact next version.

Release procedure and artifact acceptance are defined in `RELEASE.md`.
