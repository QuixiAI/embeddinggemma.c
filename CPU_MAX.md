# cpu-max: CPU media fast paths and AVX-VNNI kernels for EmbeddingGemma 2

This repository is a fork of
[QuixiAI/embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c) at
upstream commit `d8f9e33` (MIT license, retained in `LICENSE`; copyright
QuixiAI). Every engine change is a separate commit on top of upstream, so
`git log d8f9e33..HEAD -- src Makefile` and `git diff d8f9e33 HEAD -- src Makefile`
show exactly what differs in the engine. The benchmark suite, evaluations and
reports are in [`cpu-max/`](cpu-max/); the project overview is `README.md`
(upstream's README is kept as `README.upstream.md`). The model graph, GGUF files and HTTP API are upstream's; the changes are CPU
execution paths, kernels, tests, flags, and three semantic defaults that now
follow the official implementation (see Flags).

## What upstream already had (not claimed here)

The SIMD packed-batch *text* graph (`gemma2_fast.c`), its x86 kernels
(AVX2 / AVX-VNNI 256-bit Q8 tiles, the AVX-512 VNNI repacked GEMM), NEON /
i8mm, the spin thread pool, CPU workers with L3-aware pinning, batching and
caching, and all GPU backends.

## What this fork adds

| commit | change | files |
|---|---|---|
| `8120079` | `EI_GEMMA2_REF_PROFILE=1`: op-class timers in the reference vision/audio graphs | `src/gemma2.c` |
| `3ec5d16` | CPU **vision** fast path (upstream ran it in the scalar reference graph); repacked **256-bit AVX-VNNI Q8_0 GEMM** for CPUs without AVX-512 | `src/gemma2_vision_fast.c`, `src/gemma2_media_x86.c`, `src/gemma2_vnni256.c`, hooks in `src/gemma2.c`, `src/gemma2_fast.c` |
| `78a761a` | CPU **audio** fast path | `src/gemma2_audio_fast.c` |
| `d5f92ff` | tiled online-softmax attention for head widths 64-512 (GQA, windows, packed bounds); the text graph uses it on x86 | `src/gemma2_media_x86.c`, `src/gemma2_fast.c` |
| `b80e57a` | packed fp32 GEMM for the BF16/F32-weight projections | `src/gemma2_sgemm_x86.c`, `src/gemma2_fast.c` |
| `eb60d91` | fidelity flags (official activation, log-mel, video layout) | `src/media.c`, `src/http_media.h` |
| `1a3e206` | VNNI256 for waves of <= 8 tokens; release GGUF pages of repacked weights | `src/gemma2_fast.c`, `src/gemma2_release.h`, `src/gguf.c` |
| `fcb176f` | official semantics by default (`EI_GEMMA2_SEMANTICS=upstream` restores upstream) | `src/gemma2_vision_fast.c`, `src/gemma2.c`, `src/media.c`, `src/http_media.h` |
| `97fdcc7` | hybrid P/E-core aware CPU pool (P-cores first, caller on a P-core, small waves capped to P-cores) | `src/parallel.c`, `src/engine.c`, `src/gemma2_fast.c`, towers |
| `aaf445d` | `make test-gemma2` runs the llama.cpp goldens with upstream semantics | `Makefile` |
| `2cd784b` | attention block-size hooks for tuning (`EI_G2M_QB`, `-DG2M_KB`); defaults unchanged | `src/gemma2_media_x86.c` |
| `a77b187` | fp16 scales / int16 sums in the VNNI256 repack; streamed audio subsampling (less resident memory, bit-identical) | `src/gemma2_vnni256.c`, `src/gemma2_audio_fast.c` |
| `d9da7bf` | vision attention reads head-major K/V planes (bit-identical, faster at large images) | `src/gemma2_vision_fast.c`, `src/test_gemma2_attn.c` |
| `e3dfb02` | `EI_GEMMA2_CPU_PREPACK=1`: repack CPU weights at startup | `src/server.c` |
| `f665036` | vision attention writes into the free V buffer (one workspace buffer less) | `src/gemma2_vision_fast.c` |
| `6d1ce48` | log-mel frames on the engine pool, Hann window table (bit-identical) | `src/media.c`, `src/gemma2_audio_fast.c` |
| `e114de5` | CPU media towers yield the engine to waiting text waves between layers (no head-of-line blocking) | `src/server.c`, `src/media.c`, `src/media.h`, `src/http_media.h`, towers |
| `d74e061` | weight repack/packing runs on the caller's pool (cold first requests near warm speed) | `src/gemma2_vnni256.c`, `src/gemma2_sgemm_x86.c`, `src/gemma2_fast.c` |

All numerics keep upstream's choices: Q8_0 weights with Q8_0 activations
for the integer GEMMs, fp32 attention, norms, softmax and accumulation.
Only fp32 summation order differs from the reference graphs.

## Flags

Defaults are on where the hardware supports them; `0` restores upstream.

| variable | default | effect |
|---|---|---|
| `EI_GEMMA2_CPU_MEDIA_FAST` | 1 | vision/audio fast towers (`0`: reference graph) |
| `EI_GEMMA2_CPU_VNNI256` | 1 on AVX-VNNI without AVX-512 | repacked VNNI GEMM |
| `EI_GEMMA2_CPU_VNNI256_SHORT` | 1 | also for waves of <= 8 tokens |
| `EI_GEMMA2_CPU_FLASH` | 1 on x86 AVX2 | tiled attention in the text graph |
| `EI_GEMMA2_CPU_SGEMM` | 1 on x86 AVX2 | packed fp32 GEMM |
| `EI_GEMMA2_CPU_RELEASE_WEIGHTS` | 1 | drop GGUF pages of repacked weights |
| `EI_CPU_HYBRID` | 1 on hybrid CPUs | pin the gemma2 CPU pool P-cores first (`0`: upstream's unpinned pool) |
| `EI_GEMMA2_CPU_HYBRID_TOKENS` | 512 | text waves below this run on the P-core threads only |
| `EI_GEMMA2_CPU_PREPACK` | 0 | repack text/vision/audio weights before listening (all towers resident from startup; the remaining first-request cost is small since `d74e061`) |
| `EI_GEMMA2_CPU_YIELD` | 1 | CPU media towers hand the engine to waiting text waves between layers (`0`: a text wave waits for the whole tower) |
| `EI_GEMMA2_CPU_PROFILE` | 0 | per-stage timers (text, vision, audio) |
| `EI_G2M_QB` | per head width | attention query block override (tuning only) |
| `EI_GEMMA2_REF_PROFILE` | 0 | op-class timers of the reference graphs |
| `EI_GEMMA2_SEMANTICS` | `official` | `upstream` restores upstream's three deviations below (embeddings identical in kind to upstream's, e.g. for existing indexes) |
| `EI_GEMMA2_VISION_GELU` | official: `tanh` | vision MLP activation (`gelu_pytorch_tanh` per the official config; upstream: QuickGELU, the GGUF lacks `clip.use_gelu`) |
| `EI_GEMMA2_AUDIO_LOGMEL` | official: `hf` | `log(mel + 0.001)` as the Transformers extractor; `max`: upstream's `log(max(mel, 0.001))` |
| `EI_GEMMA2_VIDEO_TIMESTAMPS` | official: 0 | `1`: `MM:SS` text before each frame (upstream; the processor's add_timestamps=True layout) |

Why the default changed: against the official Sentence Transformers FP32
embeddings on the benchmark workloads, upstream semantics reach cosine
0.8746 (video with timestamps) to 0.9998; official semantics reach
>= 0.99914 on every workload (mean 0.99970).  Measurements:
`cpu-max/reports/quality_embeddings.*`.

## Tests

```sh
make test-gemma2 GEMMA2_MODEL=... GEMMA2_MMPROJ=...      # upstream suite (passes; runs with upstream semantics)
make build/test_gemma2_vnni256 && ./build/test_gemma2_vnni256        # GEMM routes vs fp64 scalar
make build/test_gemma2_attn && ./build/test_gemma2_attn              # attention vs scalar fp32 vs fp64
./build/test_gemma2_attn bench 10080 64 12                           # throughput; EI_ATTN_BENCH_HEADMAJOR=1, EI_ATTN_BENCH_THP=1
make build/test_gemma2_media && ./build/test_gemma2_media T.gguf P.gguf vision-synthetic   # fast vs reference towers
./build/test_gemma2_media T.gguf P.gguf audio-synthetic
```

`test_gemma2_media` gates on flattened cosine >= 0.999 and every soft token
>= 0.95 between the fast and reference towers (both quantize activations to
Q8_0, so they differ by rounding noise; against the HF FP32 towers each
graph's worst token is ~0.95).
