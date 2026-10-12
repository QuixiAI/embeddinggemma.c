# Contributing

Contributions should preserve the project's narrow scope: standalone inference
and serving for two fixed profiles: native CPU/Metal EmbeddingGemma 2 with text,
vision and audio towers, and EmbeddingGemma 300M on CPU, Metal, CUDA, ROCm,
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
${XDG_CACHE_HOME:-$HOME/.cache}/embeddinggemma.c/embeddinggemma-2-Q8_0.gguf
```

The combined `mmproj-embeddinggemma-2-Q8_0.gguf` projector is resolved beside
the new text model, or selected with `--mmproj`. `--profile embeddinggemma-300m`
selects the original model. The Metal binary selects Metal for both profiles. New-profile CUDA/ROCm/XPU
graphs are not implemented yet; select CPU explicitly on those builds.

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

- `src/gemma2.c`, `src/media.c`: native second-profile text/vision/audio graphs,
  image decoding/resizing, and WAV/log-mel preprocessing.
- `src/gemma2_metal.m`, `src/metal/kernels/gemma2.metal`, and
  `src/metal4/kernels/gemma2_*.metal`: second-profile Metal graphs and kernels.
- `src/profiles.h`: startup profiles and pinned new-model download URLs.
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

The output is `build/embeddinggemma-metal`. The build compiles a portable Metal
3.1 library and a Metal 4 tensor/flash library, then embeds them in Mach-O
`__DATA,__metallib` and `__DATA,__metal4lib` sections. The corresponding files
under `build/` are inspection intermediates, not runtime sidecars. The runtime
selects Metal 4 on supported GPUs and keeps the Metal 3.1 path for older Apple
Silicon. Building the fast path requires Xcode 26 or newer; Darwin executables
still default to `MACOSX_DEPLOYMENT_TARGET=14.0`.

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
make test-gemma2  # requires both new GGUF files in model/
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

New-profile validation uses `testdata/gemma2-goldens.json`; reference provenance
and regeneration are recorded in `EXTRACTION.md`. The new CPU path must meet
cosine >= 0.999 for both encoder outputs and final text/media embeddings.

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

- `EI_THREADS` (1..1024; the server's `--threads` sets it): override the CPU
  thread count. The default is every core up to 32.
- `EI_CPU_SHORT_THREADS`: override the short-projection worker width.
- `EI_CPU_DUAL_PROJECTION`: control paired projection accumulation.
- `EI_CPU_FUSED_RMS_QUANT=0`: materialize normalized activations before Q8.
- `EI_CPU_MULTIROW_MIN_TOKENS=0..65536`: disable or move the multirow boundary.
- `EI_CPU_FUSED_GELU_QUANT=1`: enable the rejected fused activation experiment.

### Metal

Metal 3.1 uses one-row direct Q4 GEMV at T=1..6 and four-row direct Q4 GEMV from
T=7. On Metal 4 GPUs, Q4 tensor matmul takes over at 64 flattened tokens with
combined QKV and up/gate dispatches. Model-specific flash attention starts at
128 tokens for one sequence and 192 tokens in packed batches; it uses FP16 K/V
with FP32 softmax and output accumulation. Retained portable fusions cover Q/K
norm plus RoPE, residual plus next RMS norm, and final norm/pooling. RoPE tables
are precomputed.

Diagnostic controls:

- `EI_METALLIB_PATH`: load an alternate metallib for kernel experiments.
- `EI_METAL4LIB_PATH`: load an alternate Metal 4 fast-path library.
- `EI_METAL_TENSOR_MM_MIN_TOKENS=1..65536`: move the tensor-matmul boundary.
- `EI_METAL_FLASH_ATTN_MIN_TOKENS=1..65536`: move single-sequence flash attention.
- `EI_METAL_FLASH_ATTN_BATCH_MIN_TOKENS=1..65536`: move packed flash attention.
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

EmbeddingGemma 2 uses resident Q8/BF16/F32 weights, reusable activation buffers,
GPU text/vision/audio graphs, and block-diagonal packed text batches. Its direct
FP32 path runs on Metal 3.1. Optional routes require a Metal 4 GPU:

- `EI_GEMMA2_METAL_TENSOR=1`: 64x64 matrix tiles from 32 tokens; packed Q8/BF16
  weights are decoded to FP16 tiles, with FP32 activations and accumulation.
- `EI_GEMMA2_METAL_FLASH=1`: 8-query/32-key tiled attention from 128 tokens;
  separate 64/256/512-wide kernels, FP16 Q/K/V, FP32 softmax/accumulation.
- `EI_GEMMA2_METAL_PROFILE=1`: print CPU encoding, GPU, and total graph times.

The batched-serving stack (2026-10-09, `perf/gemma2_kernel_design.md`). Since
the end of that day FUSED, TENSOR2 and FLASH2 are on by default (TENSOR2 and
FLASH2 only where Metal 4 is available); set a flag to `0` to turn a route
off, `1` to require it. All three at `0` reproduce the 10-07 direct path
exactly (goldens identical to 9 digits):

- `EI_GEMMA2_METAL_GEMV=1`: weight-stationary GEMV; each lane decodes 16 weights
  of 4 rows once and applies them to 8 tokens (Q8/BF16/F16/F32). Metal 3.1.
- `EI_GEMMA2_METAL_FUSED=1`: fused text layer, 11 dispatches instead of 25:
  QKV and up/gate+GeGLU and inp_gate+PLE GEMVs, q/k/v norm + RoPE in one pass,
  post-norm + residual + next pre-norm in one pass. Implies `GEMV`. Metal 3.1.
- `EI_GEMMA2_METAL_TENSOR2=1`: from 40 tokens, `matmul2d` against fp16
  row-major copies of the Q8/BF16 text weights (and the F32/BF16 projector
  weights) made at load, relaxed precision, FP32 activations/accumulation;
  32-token tiles, 64 from 256. FFN up/gate are interleaved per tile so GeGLU,
  and the PLE gate, run in the GEMM epilogue (`EI_GEMMA2_METAL_NOEPI=1`
  disables the epilogues).
  `EI_GEMMA2_METAL_TENSOR2_MIN_TOKENS` / `_WIDE_TOKENS` move the crossovers.
- `EI_GEMMA2_METAL_FLASH2=1`: from 16 tokens, `matmul2d` flash attention for
  head widths 256/512: 32 queries x 128 keys on 8 simdgroups
  (`EI_GEMMA2_METAL_FLASH2_TILE`: `n4` is the former 4-simdgroup tile, same
  arithmetic, 0.8-0.9x at 2k-8k tokens; `k64`, `q16`, `q16k256`, `n16`,
  `q16n8`, `q16k256n8` are the other instantiations), per-row packed-sequence
  and window masks, tile key ranges clipped to the visible span; fp16 K/V written by the q/k/v norm pass. Vision
  (width 64) uses the padded tensor kernel below (`FLASH64`), else the
  simdgroup flash kernel: any `matmul2d` flash at width 64 crashes the M5
  driver. `EI_GEMMA2_METAL_FLASH2_MIN_TOKENS` moves the threshold.
- `EI_GEMMA2_METAL_SKIP=attn,mm,qkv,geglu,ple,post,qknorm`: diagnostic only;
  drops those dispatches to size their GPU share. Output is invalid (`attn`,
  `mm` and `post` also apply to the media towers).

The media stack (2026-10-10, `perf/optimization_status.md`), on by default
(FLASH64 and VFUSED where Metal 4 is available); `0` turns a route off:

- `EI_GEMMA2_METAL_LAYER=1` (opt-in): persistent per-layer kernel for waves
  under 40 tokens. The eleven dependent short-wave stages of a text layer run
  as one dispatch of `EI_GEMMA2_METAL_LAYER_GROUPS` (default 128) threadgroups
  separated by a device-atomic grid barrier; bit-identical to the GEMV route
  with scalar attention (`make test-gemma2-metal-routes` checks it). A startup
  probe (264 barriers, capped spin) enables it only where the groups
  rendezvous, and a barrier timeout inside a forward fails that wave over to
  the dispatch route and turns the route off. It stays off because a grid
  barrier costs what a dispatch boundary costs on the M5 (2.2-4.7 us for
  64-512 groups): 0.99-1.13x at 1-8 tokens, 0.5-0.8x at 20-39.
- `EI_GEMMA2_METAL_FLASH64=1`: vision attention (64-wide heads) on the
  `matmul2d` flash kernel over 128-wide zero-padded fp16 K/V rows
  (`g2t_flash_64p`, `g2_half_pad`): the pad columns add exact zeros to Q.K^T
  and P.V and are never stored, which sidesteps the driver crash of every
  64-wide instantiation. Attention FLOPs are 1.5x but run on tensor ops.
- `EI_GEMMA2_METAL_VFUSED=1`: fused vision layer: q/k/v norms + 2-D RoPE +
  padded fp16 K/V in one pass (`g2_vqkv_post`), up/gate interleaved per tile
  with the quick-GELU in the GEMM epilogue (`g2t_gemm_qgeglu_*`), post-norm +
  residual + next pre-norm in one pass (`g2_post_add_768`). 10 dispatches per
  layer instead of 17.
- `EI_GEMMA2_METAL_AFUSED=1`: audio clamps folded into their producers (norm,
  SiLU, attention, post-add) and the GEMM epilogue (`g2t_gemm_clamp_*`), and
  the Gemma sandwiches in `g2_post_add_1024`; the 18 clamp passes per layer
  are gone. The patch embedding (as [768][768]) and every 2-D `a.*` projector
  weight now have fp16 copies too (the audio pre-encode and input projection
  were on the tile route).
- `--media-cache-mb N` (server, default 256, 0 disables): media cache for
  content-object requests. Image and video-frame tower outputs are cached by
  (token budget, payload hash), audio by (samples used, payload hash), and
  whole content objects by their JSON text, so a repeated image, frame or
  clip skips base64, decode, resize and the tower (image768 46.9 -> 13 ms,
  audio10s 21 -> 10 ms, video4 44 -> 14 ms at c1, byte-identical responses).
  One LRU shared by every media slot (`ei_response_cache`), 128-bit keys.
- `--media-slots N` (server, 1..8, default 2): GPU media contexts. Each media
  request runs its towers on the least-loaded slot under that slot's lock, so
  media requests overlap on the GPU and never take the global inference lock
  (the CPU backend still shares the engine pool under it). 1 is the old
  behaviour minus the global lock.
- Media finishes (the text pass over a media request's assembled rows) queue
  to one thread that packs every waiting finish into a wave on its own
  context (`ei_gemma2_metal_batch_inputs`); `EI_MEDIA_FINISH_BATCH=0` runs
  them one by one as before. Metal only.
- Audio towers queue the same way: one thread packs every waiting clip into
  one wave on its own context (`ei_gemma2_metal_audio_clips`: the subsampling
  convs run per clip, the conformer layers see the packed rows with per-token
  clip bounds for the depthwise conv and attention; at most 12,000 mel frames
  and 64 clips per wave). The tower is 1.35x at two 10 s clips and 1.48x from
  four up; `EI_MEDIA_AUDIO_BATCH=0` runs each clip on its request's slot as
  before. Metal only. Images are not packed across requests: the vision tower
  gains under 10% at 432x432 and 3.5% at 768x768 from packing.
- Video frames decode and resize in parallel (up to 8 threads), then on Metal
  runs of same-sized frames go through one packed vision wave
  (`ei_gemma2_metal_vision_frames`: tokens laid out frame after frame, per-frame
  kernels index `t % p.k`, attention and RoPE take per-token bounds; at most
  10,080 patch tokens per wave) and commit in stream order.
  `EI_GEMMA2_VIDEO_BATCH=0` runs one frame per wave. On macOS PNG IDAT
  streams inflate through libcompression (`EI_PNG_INFLATE=0` restores stb's
  inflate; output is identical) and base64 decodes through a table (18x).
  `EI_MEDIA_TIMING=1`
  prints each media request's stage times (base64, decode, resize, tower,
  log-mel, finish) on stderr. Parity with single-frame
  towers >= 0.999999 on every attention route (exact on `g2_attention`); the
  wave is 1.08x at 4 and 1.11x at 8 432x432 frames, the tower being
  compute-bound already at 729 patches. The log-mel front end tabulates FFT
  twiddles and uses the sparse filter ranges (identical values, 3-4x less CPU).
- `make bench-gemma2-media [ITEMS="v768 a30 v960x960 video4 v432x432x8"]
  [EI_BENCH_STREAMS=N] [EI_BENCH_REPEAT=N]` times the towers (`vWxHxF` is a
  packed F-frame wave, `aFxC` a packed wave of C audio clips of F mel frames)
  and the mixed-input text pass, repeating the list
  round-robin with `EI_BENCH_REPEAT` so configurations interleave on this
  noisy host; `perf/loadgen_media.c` posts a content-object body
  (`perf/media_fixtures.py` writes image/audio/video fixtures) at a given
  concurrency.
- `--pipeline-slots N` (server, 1..8): N text waves in flight on separate
  queues/workspaces sharing the weights; media gets its own context, so text
  waves no longer take the global inference lock. 1 is the old behaviour.
  Measured no gain on uniform loads (0.67-0.85x for tiny requests at c4-c32,
  neutral above ~128 tokens): one packed wave beats overlapping small ones.
  `EI_WAVE_LOG=1` prints each wave (slot, sequences, tokens, execute time,
  idle gap) to stderr.
- `--short-bypass-tokens N` (server, Metal or CUDA; implies 2 slots): requests
  of at most N tokens are short. Waves never mix short and long requests, and
  while every in-flight wave is long a free slot starts a short-only wave, so
  short requests bypass long ones. On CUDA bypass waves run on high-priority
  streams. Off (0) by default; on the CPU backend it needs `--cpu-workers` >= 2.
- Executors (EmbeddingGemma 2; `ei_engine_configure_executors`): every GPU
  pipeline slot and CPU worker is a slot of the inference service, tagged with
  a group (its device). `pick_slot_locked` starts a wave on an idle device
  first, else (without bypass) on the least-loaded device with a free slot.
  `--cuda-devices all|I,J` loads weights per device (`ei_gemma2_cuda_create_on`;
  the engine's context serves the first device). `--cpu-workers N` creates N
  `ei_gemma2_fast` contexts (own workspace; shared weights and repack caches,
  guarded by a mutex) with private spin pools splitting `--threads`; on Linux
  pools are pinned along `ei_cpu_placement` (one hardware thread per core,
  grouped by L3, SMT siblings last; `--cpu-pin off`), and the calling backend
  thread is pinned to the pool's first CPU for each wave. CPU waves are capped
  at `--cpu-max-batch-tokens` (`slot_max_batch_tokens`). Beside a GPU, CPU
  workers are overflow slots: a wave starts only when no device slot can and
  the queue holds requests of at most `--cpu-max-sequence-tokens` behind the
  `max_batch_tokens x devices` tokens the devices' next waves absorb (the same
  `in_backlog` rule decides eligibility and selection; a mismatch once spun
  with the service mutex held). Unless `--workers` is given, HTTP workers grow
  to 32 per executor. `test_inference_service` covers overflow budgets, the
  backlog boundary, and spreading across devices.

EmbeddingGemma 2 CPU (default on since 2026-10-09; `EI_GEMMA2_CPU_FAST=0`
selects the unchanged reference graph):

- `EI_GEMMA2_CPU_FAST`: packed-batch text graph (`src/gemma2_fast.c`):
  per-sequence RoPE/attention bounds, activations quantized once per
  normalized input, 4-row x 4-token NEON `sdot` Q8 tiles, threaded SIMD row
  kernels with a vector GELU, cached RoPE tables, persistent workspace. Media
  soft tokens use it too, so content-object text matches plain text.
- `EI_GEMMA2_CPU_I8MM` (default on where `FEAT_I8MM`): `smmla` Q8 GEMM on
  row-pair-interleaved weights (repacked at first use) for waves of 4+ tokens.
- `EI_GEMMA2_CPU_SHORT_THREADS=N` (default 2/3 of the pool): tasks per GEMM
  for waves of at most 8 tokens; short waves are wake-up and preemption bound.
- x86-64 (`src/gemma2_fast_x86.c`, runtime-dispatched, generic builds): AVX2 +
  FMA, VNNI 256-bit, and on AVX-512 VNNI CPUs a repacked 512-bit GEMM (u8
  activations with a precomputed -128*sum(w) correction).
  `EI_GEMMA2_CPU_X86=scalar|avx2|vnni256` forces a tier.
- `EI_GEMMA2_CPU_PROFILE=1`: per-call GEMM/row/attention wall time.
- `EI_POOL_SPIN_US=N`: pool workers spin N us on a published
  generation/participant ticket before parking, so back-to-back parallel
  calls skip the condition-variable wake. The EmbeddingGemma 2 CPU pool
  defaults to 50; the 300M pool keeps 0 (the original condvar pool).

`make bench-gemma2-cpu CASES="1x1 32x16"` times the fast graph against the
reference on natural-text token windows (`EI_BENCH_RANDOM_IDS=1` for random
ids, which exaggerate Q8 activation rounding; `EI_BENCH_FAST_ONLY=1` skips the
slow reference).

Validate the default and each stack you change:

```sh
make test-gemma2-metal
EI_GEMMA2_METAL_FUSED=0 EI_GEMMA2_METAL_TENSOR2=0 EI_GEMMA2_METAL_FLASH2=0 EI_GEMMA2_METAL_TENSOR=1 EI_GEMMA2_METAL_FLASH=1 make test-gemma2-metal
EI_GEMMA2_METAL_FUSED=0 EI_GEMMA2_METAL_TENSOR2=0 EI_GEMMA2_METAL_FLASH2=0 make test-gemma2-metal
make test-gemma2-metal-routes   # default stack vs every route off
EI_GEMMA2_ROUTE_FLAGS=TENSOR,FLASH make test-gemma2-metal-routes
```

EmbeddingGemma 2 on CUDA (`src/gemma2_cuda.cu`: text, vision and audio towers): build with `make CUDA_ARCHS=120 cuda` (86 for the 3090),
validate with `make test-gemma2-cuda`, time with `build/bench_gemma2_cuda
MODEL [TxB ...]` (reports packed-vs-single and mma-vs-CUDA-core parity).
`EI_GEMMA2_CUDA_MMA_ATTN=0` selects the CUDA-core attention kernel,
`EI_GEMMA2_CUDA_GRAPHS=0` disables graph capture (for nsys per-kernel
profiles), `EI_CUDA_DEVICE` picks the GPU. `perf/loadgen.c` is a minimal
keep-alive HTTP load generator for hosts without wrk/hey.

`make bench-gemma2-metal ROUTE=FUSED=1,TENSOR2=1,FLASH2=1 CASES="1x1 32x16"`
times a route against the default path (`EI_BENCH_BASE=` names another
baseline; `EI_BENCH_STREAMS=N` measures N concurrent waves on separate queues).
Cases are `TOKENSxSEQUENCES` packed into one wave; `T+TxB+...` packs a mixed
wave in that order (`2000+20x15`), for ordering and straddling experiments.

The route test requires Metal 4. It covers thresholds, 8192 tokens, packed
sequence isolation, larger images, and audio spanning the projection threshold.
The ordinary Metal suite supports older GPUs without selecting these routes.

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
- `EI_CUDA_W4A8_GEMM=0|1|2`: opt-in hand-written per-block int8 projection GEMM
  (Q4 weight x int8 activation on the IMMA tensor cores; keeps native Q4 nibbles
  and per-32-block scales via a cp.async double-buffered kernel). Beats cuBLAS
  fp16 1.0-1.41x on the projections but ~3% end-to-end (T>=1024); `2` limits it
  to the two decisive projections. Default off; batch self-parity ~0.997.
- `EI_CUDA_INT8_ATTN=0|1|2`: opt-in int8 flash attention. `1` = int8 Q.K^T
  (accuracy-safe, ~3% at T>=1024, route parity tightens to ~0.998); `2` also
  int8 P.V, which breaks accuracy and is not for use. Default off.

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
- `EI_XPU_FUSE_ATTN_OUTPUT_HALF=0|1`: opt-in fusion that writes the online-
  attention context as half inline, skipping a separate float-to-half
  submission before the attn-output GEMM (+2-5% p50, submission-bound regime).
  Default off; bit-identical on its active path.
- `EI_XPU_XE2_W4=0|1`: control the Xe2 W4A16 up/gate route.
- `XPU_ONEDNN=1`, `EI_XPU_ONEDNN_F16=1`, and `EI_XPU_ONEDNN_W4=1`: build and
  select rejected oneDNN experiments.

Route defaults were selected on the hardware recorded in
`perf/optimization_status.md`. Re-benchmark before changing them for another
GPU generation.

## Runtime And Serving

The canonical HTTP port is `42666`. The original `/api/embed` route and the
additive OpenAI-compatible `/v1/embeddings` route share one inference service;
their response-cache keys are format-namespaced. `--bind`, `--port`, and
`--model` override the listener and model path. The default model path is
documented above; resolution order is `--model`, `EI_MODEL_PATH`, then that
cache path.

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
- `--persistent-cache-path PATH`: persist the exact-embedding cache to disk.
  The service loads matching entries at startup and flushes ready entries on
  graceful shutdown (SIGTERM/SIGINT), so re-indexing and eval workloads survive
  restarts as cache hits instead of recomputing from cold. The file is guarded
  by a model fingerprint (size + first 64 KiB); a cache from a different model
  or build is ignored, never served. Writes are atomic (temp file + rename);
  a hard kill (SIGKILL) simply loses the current session's new entries.
- `EI_BATCH_LOOKAHEAD=0`: restore strict FIFO batching for diagnostics.
- `EI_ADAPTIVE_BATCH_WAIT=0`: always apply the configured collection delay.
- `EI_BATCH_WAVE=0`: disable demand-aware wave collection. By default the
  scheduler tracks the peak number of concurrently in-flight requests since
  the previous dispatch and both keeps collecting while more clients are in
  flight than queued and dispatches the moment the queue reaches that peak.
- `EI_BATCH_ARRIVAL_WAIT_US=N` (default 5000, 0 disables): while accepted
  requests are still tokenizing, the collector keeps extending its deadline
  (one `--batch-wait-us` at a time, at most N us from the start of the
  collection) and solo inline execution yields, so a wave is not cut just
  ahead of requests about to queue. At 512 tokens x c16 the 16 clients
  otherwise phase-lock into a 15-request wave plus a lone one; with it every
  wave carries 16 (wave log) and the c16 p99 drops from ~2 waves to ~1.
- `EI_BATCH_SORT=0` keeps queue order inside a wave. By default a wave is
  packed longest sequence first (results still land in request order):
  sequences pack ragged, so there is no padding to save, but the packed
  attention tiles then start aligned to the long sequences, 2-4% on mixed
  waves in-process (seven ordered pairs, all the same direction).
- `EI_BATCH_WAIT_MAX_US=N`: cap the extended collection deadline. New arrivals
  during collection extend the base `--batch-wait-us` deadline, bounded by
  this cap; the default is four times the base delay.
- `EI_SOLO_INLINE=0`: disable inline execution of solo requests. By default a
  request that arrives with an idle backend, an empty queue, and an expected
  wave of one executes on the submitting worker thread, skipping two thread
  handoffs; concurrent traffic still routes through the batch collector.
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
