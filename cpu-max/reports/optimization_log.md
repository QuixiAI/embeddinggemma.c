# Optimization log — EmbeddingGemma 2 CPU (fork of QuixiAI/embeddinggemma.c)

Every attempted change, in order, with the measurement that justified keeping
or rejecting it. Fork: this repository (every engine change a commit on top
of upstream `d8f9e33`); baseline: unmodified upstream at `d8f9e33`, built by
`cpu-max/scripts/setup_upstream.sh`. Paths: engine sources are relative to the
repository root (`src/…`), tools and results to `cpu-max/` (`scripts/…`,
`bench/…`, `reports/…`).
Host: Intel Core i7-14700HX (8 P-cores + 12 E-cores, AVX2 + AVX-VNNI, no
AVX-512/AMX), 16 GB, Ubuntu 24.04, GCC 13.3, upstream build flags
(`-O2 -g`, no `-march`, runtime ISA dispatch). "P8" = 8 threads pinned to
one hardware thread per P-core (cpus 0,2,…,14). Hardware counters are not
available (perf_event_paranoid=4, no sudo), so profiles are wall-clock
instrumentation.

## 0. Baseline profile (upstream reference media towers)

Commit `8120079` adds `EI_GEMMA2_REF_PROFILE=1` op-class timers to the
reference vision/audio graphs (computation unchanged). P8, one request each:

| tower / input | total | attention | Q8 GEMM | act. quant (serial) | conv | other |
|---|---:|---:|---:|---:|---:|---:|
| vision, 280-token image (2,520 patches) | 31.6 s | 67.9% | 26.3% | 3.3% | – | 2.5% |
| vision, 70-token image (630 patches) | 3.47 s | 28.9% | 58.5% | 7.4% | – | 5.2% |
| audio, 15 s (1,499 mel frames) | 3.32 s | 0.8% | 68.6% | 8.2% | 11.8% | 10.6% |

Effective rates: vision attention 7 GMAC/s, Q8 GEMM ~46 GMAC/s on 8 cores
(scalar loops; F32 weights through a per-element type switch). Upstream's
SIMD "fast" CPU graph exists only for the text backbone.

## 1. Vision fast path (kept) — `3ec5d16`

`src/gemma2_vision_fast.c`, `src/gemma2_media_x86.c`. Same math as the
reference (Q8_0 activations for Q8_0 weights, fp32 attention/norms, the
fp16-rounded patch input, the GGUF's GELU variant); threaded fused row
passes; tiled online-softmax attention (transposed S^T/P^T/O^T form, K/V
read in place); FFN in 256-token chunks; persistent workspace.
`EI_GEMMA2_CPU_MEDIA_FAST=0` restores the reference.

* First version reused upstream's x86 Q8 tile for GEMMs: 4-5.6x on the
  tower (12 E-cores) — GEMM was 61% of the new path.
* Correctness: initial parity gate (cosine 0.9999 vs reference) failed at
  0.9993. Diagnosed against the HF FP32 tower on the same pixels
  (`scripts/compare_vision_towers.py`): reference 0.99827, fast 0.99863 vs
  same-activation FP32 (i70) → the gap is two independent Q8 rounding
  errors, not a bug. Gate changed to flattened cosine ≥ 0.999.

## 2. Repacked 256-bit AVX-VNNI Q8_0 GEMM (kept) — `3ec5d16`

`src/gemma2_vnni256.c`. Upstream has a repacked GEMM only for AVX-512 VNNI;
on AVX-VNNI-only CPUs its tile spends ~6 vector ops per 32 MACs (abs, sign,
zero-seeded vpdpbusd, convert, scalar scale product, FMA) plus horizontal
sums. New: 8-row weight groups (one 32-byte load = 4 k-bytes of 8 rows),
u8 activations with a −128·Σw seeded accumulator, one convert/mul/FMA per
8 rows per 32-block, outputs contiguous.

| attempt | 1 P-core GMAC/s | P8 GMAC/s | note |
|---|---:|---:|---|
| upstream tile (`EI_GEMMA2_CPU_VNNI256=0`) | 28-30 | 196-213 | |
| v1 (rolled loops) | 47.5-50 | 329 | GCC -O2 kept `dot[]`/`acc[]` on the stack (load+store per vpdpbusd, seen in objdump) |
| v2 explicit `#pragma GCC unroll`, 2 int32 chains/token | **211-213** | **1,073-1,251** | kept |

Max relative error vs a float64 scalar Q8_0 dot: 4.8e-8 (exact integer
dots). Bug found during testing: the repack cache is keyed by tensor
address and the test reused one stack tensor across shapes (crash); the
cache now also checks the shape.

The same unrolling problem existed in my attention kernel (19 FMAs in the
whole function); fixed with unroll pragmas before any measurement was kept.

Vision tower after 1+2 (same process/pixels, P8): i70 3,565 → 125 ms
(28.4x), i280 31,459 → 977 ms (32.2x). Tower cosine vs HF FP32 (same
activation): reference 0.99907 / fast 0.99894 (i280).

## 3. Audio fast path (kept) — `78a761a`

`src/gemma2_audio_fast.c`: reference math (clipped-linear clamps, Q8
activations, fp32 convs/norms/local attention); AVX2 subsampling convs
with fused LayerNorm+ReLU, Q8 GEMMs on VNNI256, fused row passes,
relative-position keys cached per layer.

* Bug during development: the GLU pass read the 2048-wide `conv_pw1`
  output with a 4096 stride (cosine 0.45 vs reference). Fixed; then
  0.99987-0.99995 (synthetic) and, on a real Clotho clip with HF-extractor
  mel, tower cosine vs HF FP32: reference 0.99893, fast 0.99907.
* A plain residual add after `conv_pw2` was briefly routed through an
  RMS-normalizing op; caught in review before testing.

## 4. Tiled attention for all head widths; text path uses it (kept) — `d5f92ff`

`g2m_attn`: head widths 64-512, GQA/MQA strides, per-query key bounds
(packed sequences, ±512 window), exact zeros for masked keys. Text profile
before (12 E-cores, 2,048 tokens): attention 2.45 s of 3.5 s (70%).
`bench_gemma2_cpu` 12 E-cores: 600x2 1,298 → 867 ms, 1100x1 1,498 →
908 ms; min cosine vs the reference graph unchanged. `test_gemma2_attn`:
tiled kernel as accurate as scalar fp32 against a float64 truth.

## 5. Packed fp32 GEMM for float-weight projections (kept) — `b80e57a`

`src/gemma2_sgemm_x86.c` replaces upstream `gemm_f` on x86 (BF16
per-layer-input projection, F32 patch embedding, F32 audio input
projection): 12 E-cores 250 vs 53 GMAC/s (4.7x), rel err 1.8e-7; +31 MB
resident for fp32 weight copies. 600x2 text: 867 → 735 ms.

## Observations not yet acted on

* Upstream `make test-gemma2` fails its HTTP stage on this host
  (`400 failed to read request body` on a long-text + media request); the
  fork passes the same suite. Not root-caused.
* An upstream server whose client died mid-request kept ~1.6 cores busy
  for 30 min (found as a stray process during baseline runs; those ST/ONNX
  runs are being repeated). Not root-caused.
* The mmproj GGUF lacks `clip.use_gelu`, so upstream (and llama.cpp) run
  the vision MLP with QuickGELU; the official model uses `gelu_pytorch_tanh`.
  Measured in FP32: QuickGELU vs tanh tower cosine 0.998 (i70/i280); fast
  path with tanh vs HF official 0.99876/0.99899 vs reference 0.99699/0.99714.
* QuixiAI's native video part inserts `MM:SS` timestamp text before every
  frame (597 vs 570 tokens for 4 frames); the official processor default is
  `add_timestamps=False`.

## 6. Rejected: larger attention register tiles (not kept)

48-query x 48-key blocks with 6-key x 16-query S tiles and 4-dim x 24-query
O tiles (12 accumulators). `test_gemma2_attn bench`, P-cores: 1 thread
52.1 -> 52.8 GMAC/s, P8 n=2520 305 -> 316 GMAC/s (+3.5%), P8 n=10080
271 -> 254 GMAC/s (-6.5%). Below the 3% stable-gain bar and a regression at
the largest shape; code removed. While integrating it, vision was found to
size attention tasks with a fixed 64-query constant — fixed (tasks now use
`g2m_attn_qblock`) before any result was kept.

## 7. Short waves on VNNI256; source-page release (kept) — see `CPU_MAX.md`

P8 `bench_gemma2_cpu`: 1-token 4.65 -> 3.37 ms, 8-token 11.25 -> 5.37 ms.
Weight-page release is implemented (MADV_DONTNEED on GGUF-mapped tensors
after repack, guarded by a mapping registry); its end-to-end memory effect
has not been measured yet.

## Status at the end of this session (P8, end-to-end over HTTP)

Upstream (`reports/raw/m1_upstream_P8_t8*.json`) vs fork `3ec5d16`
(`reports/raw/m1_fork_3ec5d16_P8_t8.json`; later commits are faster and not
yet benchmarked end to end): i70 3.91 s -> 0.157 s, i280 32.1 s -> 1.00 s,
i1120 552 s -> 12.1 s, v4 42.7 s -> 1.76 s, v8 87.6 s -> 3.54 s, v16 161 s ->
7.7 s, t128 117 -> 41 ms. In-process towers at the latest commit (P8): i70
vision 125 ms (29.8x vs reference), a15 audio 141 ms (23.6x).

## 8. Weight-page release: measured (kept) — `1a3e206`

Fresh server per workload, P8, fork `1a3e206` with
`EI_GEMMA2_CPU_RELEASE_WEIGHTS=1` (default) vs `=0`, peak RSS (VmHWM):
i1120 807 vs 1,106 MB (-27%), a30 842 vs 1,298 MB (-35%), v16 806 vs
1,103 MB (-27%), m_tia (all three towers) 1,033 vs 1,643 MB (-37%).
Raw: `reports/raw/final_mem_fork_1a3e206*_P8_t8.json`.

## 9. Official semantics as the default (kept) — `fcb176f`

Upstream deviates from the official implementation in three places
(vision QuickGELU, log-mel floor, per-frame timestamp text). Final-embedding
cosine vs the official ST FP32 embeddings, 16 workloads, P8:

| backend | min | mean | video | images | audio |
|---|---:|---:|---:|---:|---:|
| upstream | 0.8756 | 0.983 | 0.876-0.963 | 0.9987-0.9996 | 0.9986-0.9990 |
| fork, upstream semantics | 0.8746 | 0.983 | 0.875-0.963 | 0.9992-0.9997 | 0.9988-0.9993 |
| fork, official semantics | **0.99914** | **0.99970** | 0.99987-0.99989 | 0.9996-0.9999 | 0.9991-0.9995 |
| ONNX Community int8 | 0.99850 | 0.99938 | 0.9985-0.9988 | 0.9985-0.9996 | 0.9998 |

Fork (upstream semantics) vs upstream itself: 0.9992-0.99995 on every
workload (Q8 rounding noise), so it is used as upstream's stand-in where
upstream's scalar towers are too slow to evaluate (retrieval on Flickr /
MSR-VTT). `EI_GEMMA2_SEMANTICS=upstream` restores upstream's behaviour; the
upstream test suite's llama.cpp goldens encode upstream semantics and must
run with it.

## 10. Hybrid P/E-core aware CPU pool (kept) — `97fdcc7`

`ei_cpu_hybrid_order` (P-cores first, one thread per core before SMT
siblings, then E-cores), the gemma2 CPU pool pinned in that order, the
calling thread bound to a P-core for the duration of a tower, and text
waves below `EI_GEMMA2_CPU_HYBRID_TOKENS` (512) capped to the P-core
threads. Motivation (fork `1a3e206`, HTTP p50 relative to P8): at 20-28
threads short text got 1.3-1.8x *slower* (t16 8.7 → 13.6-13.8 ms at ALL;
E-cores finishing their share late) while large media gained 12-22%.

* With the hybrid pool, ALL (28 threads): t16 8.3, t128 28.6, t512 132 ms
  — short text no longer loses to P8 — and large media keep the E-core
  gain: i1120 8,269 ms (P8 11,529), v16 5,367 (P8 6,481), m_tia 1,120
  (P8 1,313). Scaling chart: `reports/plots/scaling.png`.
* At P8 (pool already all P-cores) pinning is neutral. Interleaved A/B,
  two rounds each, `EI_CPU_HYBRID=0` vs `1`: t128 29.3/28.5 vs 28.6/28.3,
  t512 149/140 vs 140/142, i280 978/1019 vs 1006/1029, a15 245/252 vs
  250/254 ms. The 3-11% slower P8 numbers of the first `97fdcc7` matrix
  run were host noise, not pinning (`reports/raw/ab_hybrid_P8_*`).

## 11. Final benchmark matrix (round 2)

Every backend at P8 and at ALL 28 threads, warm p50 over HTTP (native) or
in-process (ST, ONNX); `reports/benchmarks.{csv,json}`. Best configuration
of each backend per workload (`reports/best_config_comparison.json`), ms:

| workload | fork | ST FP32 | ONNX Q8 | upstream | fork vs best other |
|---|---:|---:|---:|---:|---:|
| text 16 tok | 8.3 | 20.4 | 36.8 | 17.4 | 2.11x (upstream) |
| text 128 | 28.6 | 77.5 | 84.8 | 107.5 | 2.71x (ST) |
| text 512 | 132 | 250 | 303 | 398 | 1.89x (ST) |
| image 70 tok | 146 | 341 | 1,393 | 2,548 | 2.33x (ST) |
| image 280 | 814 | 1,829 | 3,028 | 32,082 | 2.25x (ST) |
| image 1120 | 8,269 | 13,646 | 13,749 | 552,466 | 1.65x (ST) |
| audio 5 s | 84 | 361 | 1,297 | 987 | 4.31x (ST) |
| audio 15 s | 247 | 654 | 2,014 | 2,911 | 2.65x (ST) |
| audio 30 s | 470 | 1,169 | 2,448 | 8,202 | 2.49x (ST) |
| video 4 frames | 1,296 | 3,265 | 4,346 | 42,652 | 2.52x (ST) |
| video 8 | 2,564 | 6,946 | 7,465 | 87,577 | 2.71x (ST) |
| video 16 | 5,367 | 16,418 | 14,397 | 161,199 | 2.68x (ONNX) |
| text+image | 872 | 1,910 | 3,050 | 33,015 | 2.19x (ST) |
| text+audio | 265 | 670 | 1,997 | 2,901 | 2.53x (ST) |
| text+image+audio | 1,120 | 2,594 | 4,454 | 36,394 | 2.32x (ST) |
| text+video | 2,657 | 7,276 | 7,142 | 87,103 | 2.69x (ONNX) |

Geometric mean 2.45x over the fastest other backend per workload (minimum
1.65x, image 1120). Fork vs upstream at P8: 2.1-3.7x text, 25-48x images,
16-17x audio, 25-29x video.

* Sentence Transformers is *slower* at 28 threads than at P8 on every
  media workload (a30 18.3 s vs 1.17 s, v4 12.7 vs 3.3 s, i1120 25.4 vs
  13.6 s); its best numbers are P8. ONNX Runtime gains only on i1120/m_tv.
* Throughput, P8, concurrent HTTP clients with distinct inputs
  (`reports/throughput.*`): t16 c16 468 vs 131 req/s (upstream), t128 c16
  77.5 vs 16.4, i70 c2 6.45 vs 0.25, a15 c2 3.99 vs 0.26.
* Cold start: the first request per tower includes the weight repack
  (P8: t512 307 vs 150 ms warm, i280 1,279 vs 956, a30 896 vs 529) — see 15.

## 12. Attention block-size sweep (rejected; hooks kept) — `2cd784b`

`EI_G2M_QB` (query block, 16-128) and `-DG2M_KB` (key block) for
`test_gemma2_attn bench`, P8, vision shapes (12 heads x 64), GMAC/s:

| KB \ QB | 16 | 32 | 48 | 64 | 96 | 128 |
|---|---:|---:|---:|---:|---:|---:|
| n=2520, KB 32 | 272 | 296 | 304 | 307-308 | 308 | 301 |
| n=2520, KB 64 (default) | 266 | 294 | 303 | **305-306** | 305 | 297 |
| n=2520, KB 128 | 268 | 293 | 300 | 301 | | |
| n=10080, KB 32 | 219 | 251 | 274 | 270-284 | 275 | 271 |
| n=10080, KB 64 (default) | 180 | 226 | 252 | **264-272** | 289 | 278 |
| n=10080, KB 128 | 199 | 227 | 255 | 271 | | |

Small query blocks lose (K/V re-streaming); from QB 48 up everything is
within ±5% of the default, the same spread as repeating one configuration.
No default changed. The sweep did show that the large-n loss is about the
K/V stream, which led to 14.

## 13. Resident memory: compact GEMM metadata, streamed audio subsampling (kept) — `a77b187`

* VNNI256 repacked weights stored fp32 block scales and int32 −128·Σw per
  row block: 40 bytes per 32 weights of a row. Now fp16 scales (exact:
  the GGUF stores fp16) and int16 sums (|Σw| ≤ 4,064), expanded in
  registers: 36 bytes (−10% of every repacked weight).
* The audio tower materialized the first subsampling conv for the whole
  clip (t0 x 64 x 128 floats, 49 MB at 30 s). Each task now computes only
  the conv0 rows its conv1 rows read, into a reused thread-local buffer
  (boundary rows recomputed: +6% of conv0, ~1% of the subsampling).
  A first version allocated the buffer per task: subsample stage 10.5 →
  11.5 ms (1,499 frames, 12 E-cores); the thread-local buffer brought it
  back to 10.2 vs 10.2 ms.

Bit-identical outputs (fast and reference tower dumps compared with
`cmp`). Fresh-server peak RSS, 12 E-core threads: t512 367 → 352, i280
582 → 548, i1120 810 → 776, a30 845 → 753, v16 798 → 764, m_tia 1,036 →
948 MB. Where the rest goes (a30 after one request, `smaps_rollup`, measured
before the conv streaming): fork 792 MB anonymous (repacked text + audio
weights, workspaces) + 10 MB of GGUF pages vs upstream 183 MB anonymous +
476 MB of mapped GGUF pages.

## 14. Vision attention with head-major K/V (kept) — `d9da7bf`

The tiled kernel read K and V in place from the token-major [n][768]
buffers: consecutive keys of a head are 3 KB apart, each on its own 4 KB
page, and at 10,080 patches the per-head K/V stream of one query block
(5.2 MB) spills L2 with the prefetchers unable to follow. `test_gemma2_attn
bench` (new `EI_ATTN_BENCH_HEADMAJOR`, `EI_ATTN_BENCH_THP`), 12 E-cores,
two rounds, GMAC/s:

| layout | n=2520 | n=10080 |
|---|---:|---:|
| token-major (before) | 143 / 143 | 128 / 126 |
| token-major + transparent huge pages | 141 / 142 | 142 / 127 |
| head-major [12][n][64] | 157 / 153 | 159 / 145 |

Huge pages alone do not explain it (TLB reach); the contiguous stream does.
The QKV post pass now writes normed+RoPE'd K head-major into `h` (unused
until the output projection) and a second pass writes normed V head-major
into the `k` buffer (free once K moved): no added memory. Vision tower,
12 E-cores, interleaved: attention i1120 20.3-20.6 → 17.4-17.8 s, i280
1.18 → 1.07 s, i70 74 → 70 ms; the V pass adds 45 ms of row work at i1120;
tower −11% (i1120), −6% (i280), −2.5% (i70). Bit-identical (i280 and
i1120 dumps compared with `cmp`).

## 15. Startup prepack (kept, opt-in) — `e3dfb02`

`EI_GEMMA2_CPU_PREPACK=1` runs a one-token text, a 48x48 image and a
16-frame silent clip through the CPU graphs before the server listens.
First request on a fresh server, 12 E-cores, two rounds: t512 647/639 →
400/394 ms, i280 2,857/3,032 → 2,407/2,441 ms, a30 2,091/2,179 →
1,530/1,448 ms (≈ warm). Startup +0.7 s. Off by default: all towers'
repacked weights are then resident from the start (t512-only server peak
352 → 852 MB).

## 16. Vision attention output into the V buffer (kept) — `f665036`

After 14 the token-major V buffer is free during attention, so attention
writes its output there; one [n][768] workspace buffer less (31 MB at
10,080 patches). Bit-identical. Fresh-server peak RSS, 12 E-cores: i280
548 → 541, i1120 776 → 747, v16 764 → 761, m_tia 948 → 941 MB.

## 17. Rejected: larger query blocks for text attention (head width 256/512)

Text attention is 31% (512 tokens) to 45% (1,130 tokens) of the text pass
on 12 E-cores, and its query block is capped at 8,192 / d (32 queries at
d=256, 16 at d=512) by the per-thread Q^T/O^T scratch. With a 16,384-float
scratch (`-DG2M_ATTN_QT`) and `EI_G2M_QB`, `bench_gemma2_cpu`, 12 E-cores,
two rounds, ms (512 / 1,130 tokens): default 362/940, 361/944; QB 32 at
d=512 418/930, 370/940; QB 64 at d=256 385/1,015, 373/963. No gain (the
E-core L1D is 32 KB; d x QB floats of Q^T alone fill it). Reverted. Text K/V
already have 1-2 KV heads, so the head-major change of 14 does not apply.

## 18. Audio front end: log-mel on the engine pool (kept) — `6d1ce48`

The log-mel ran single-threaded in the HTTP worker, evaluating the
320-point Hann window with `cosf` for every frame: 35 ms of a 30 s request
on 12 E-cores (2.4% of its 1.49 s there; 2.3% of a5). The window is now a table (same expression, computed
once), and on the CPU tower path the frames run on the engine pool with
the engine lock taken before the mel instead of before the tower. Server
stage medians, 12 E-cores: logmel a5 5.9 → 0.7 ms, a15 17.4 → 1.6, a30
35.2 → 3.0. End to end, four rounds with alternating order (a first
fixed-order run had the new build always third and showed it 6% slower on
a5 in stages it does not touch — an ordering artifact): a5 274 → 261 ms,
a15 712 → 698 ms. Embeddings bit-identical (`numpy.array_equal`).

The streamed subsampling (13) used 8-row tasks, 16 tasks for a 5 s clip
on 12 threads: subsample stage 3.9 → 4.5 ms median; 4-row tasks restore
3.9 ms (1,499 frames: 10-12 ms in all variants).

## 19. Head-of-line blocking: towers yield the engine to text waves (kept) — `e114de5`

On the CPU backend every tower call held `inference_mutex`, so a text
request queued behind a media request waited for the whole tower. Probe
(`bench/hol_blocking.py`): one closed-loop client sending 16-token texts
while a single media request runs, 12 E-cores (t16 alone: 22 ms):

| media request | t16 p50 during it | t16 max | texts done meanwhile | media request |
|---|---:|---:|---:|---:|
| i280, before | 1,133 ms | 2,082 ms | 2 | 2,292 ms |
| i280, yield | 24.7 ms | 183 ms | 177 | 5,722 ms |
| a30, before | 659 ms | 766 ms | 2 | 1,345 ms |
| a30, yield | 89 ms | 555 ms | 13 | 1,625 ms |

Media engine sections now take `cpu_media_mutex` before `inference_mutex`
(one media request in the towers at a time); the towers call
`ei_gemma2_cpu_yield()` between layers, after vision attention and between
vision FFN chunks; the server hook releases `inference_mutex` only when text
waves wait and blocks on a condition variable until one holds it (a plain
unlock/lock would let the tower barge back in). The media request slows by
the text work done in between (177 x ~20 ms ≈ 3.5 s for i280): the engine is
shared, not lost. The a30 maximum is the request's own final text pass.
`bench/stress_mixed.py` (6 clients, 90 s, all modalities): no deadlock, all
responses bit-identical to the same request alone; `make test-gemma2`
passes. `EI_GEMMA2_CPU_YIELD=0` restores the plain lock.

Starvation check, eight closed-loop t16 clients during one i280 request
(12 E-cores): without the yield 9 texts finish (p50 2,171 ms) and the image
takes 2,150 ms; with it 750 texts finish (p50 89 ms, max 257 ms) and the
image takes 9,522 ms. The image still completes — the engine is shared
between the text waves and the tower's layers, not handed to text for good.

## 20. Weight repack on the caller's pool (kept) — `d74e061`

The VNNI256 repack and the SGEMM panel packing ran on the calling thread
alone at a tensor's first use, inside the first request of each modality
(P8: t512 cold 307 vs 150 ms warm, i280 1,279 vs 956, a30 896 vs 529).
Both now split their work over the caller's pool; same bytes (i280 tower
dump compared with `cmp`). First request on a fresh server, 12 E-cores,
two interleaved rounds: t512 581/591 → 415/479 ms, i280 2,552/2,602 →
2,318/2,292 ms, a30 1,868/1,860 → 1,484/1,509 ms (warm ≈ 370, 2,300-2,500,
1,470). This removes most of what `EI_GEMMA2_CPU_PREPACK` (15) was for,
without its resident-memory cost.

## 21. Retrieval quality (measured, `reports/retrieval.*`)

`scripts/run_retrieval.sh`, embeddings cached per backend, metrics and
agreement with the FP32 model's rankings by `eval_retrieval.py --report`.
Fork runs used `2cd784b` (later commits are bit-identical in output).

| task | fork R@1 / nDCG@10 | ST FP32 | fork, upstream semantics | upstream | fork top-1 = FP32 top-1 |
|---|---:|---:|---:|---:|---:|
| Flickr30k 1k, text → image (5,000 q) | 0.8630 / 0.9307 | 0.8636 / 0.9309 | 0.8634 / 0.9307 | – | 99.3% |
| MSR-VTT 1k-A subset, text → video (200 q) | 0.635 / 0.762 | 0.620 / 0.756 | 0.610 / 0.746 | – | 98.0% |
| SciFact, text → text (300 q, 5,183 docs) | 0.770 / 0.852 | 0.773 / 0.851 | 0.770 / 0.852 | 0.773 / 0.854 | 98.3% |
| Clotho subset, text → audio (521 q, 120 clips) | 0.131 / 0.303 | 0.132 / 0.303 | 0.134 / 0.313 | 0.131 / 0.312 | 80.2% |

The fork matches the FP32 model's retrieval quality on every task (all
differences within the sampling noise of these query counts). Upstream's
video layout (timestamp text per frame) is visible in rankings, not just
in cosine: with upstream semantics only 75% of MSR-VTT top-1 results and
67% of top-10 sets agree with FP32 (fork: 98% / 97.6%). Clotho's low
agreement is a property of the task (near-tied audio clips, R@1 ≈ 0.13 for
every backend; upstream itself agrees 81%).

The first `quant_study.py` run reported cosine 1.00000 for every scheme,
including 4-bit weights everywhere: its module-name filter
(`startswith("language_model")`) matched nothing inside the Sentence
Transformers wrapper (`0.auto_model.model.…`), so no weight was changed.
The result was discarded (`reports/raw/excluded/quant_study_invalid_empty_match.json`);
the script now matches name components and refuses to run if any tower
matches no layer.

## 22. Measurement integrity: a contaminated final run, the quiet-host guard

The first full run of `5dd7dbf` (12:41-12:52) overlapped restarts of
other services on the host (container workers, a browser renderer): the
unchanged upstream binary measured t16 23.3 ms (17.4 before), the fork's
ALL t512 629 ms (132 before) with 0.23 CPU utilization, i70 160-181 ms in
runs minutes apart. Those files are kept in `reports/raw/excluded/` and not
used. `bench/run_bench.py` now (a) waits before each item, bounded, until
other processes use ≤ 1.5 cores (`--quiet-cores`), measured from
`/proc/stat` while the backend is idle, and (b) records per item the CPU
used by everything except the backend and the client during the
measurement (`foreign_cores`). The final matrix was re-run with both, the
strongest baselines too (ST and ONNX at P8, ONNX at ALL on its best media
items, upstream's fast items); for baselines the better of the two runs is
used, which can only favour them.

Two checks prompted by the contaminated run:

* GEMM layout of 13 (fp16 scales / int16 sums vs fp32 / int32 vs fp16 /
  int32): a three-round `test_gemma2_vnni256` A/B at P8 suggested −10-22% at
  k = 3072; eight interleaved rounds put all three layouts within ±5% (best
  of 8 at 630x768x3072: 1,216 / 1,265 / 1,230 GMAC/s; medians 1,180 /
  1,195 / 1,170). No change; the three-round result was interference.
* `97fdcc7` vs `5dd7dbf` over HTTP at P8, four interleaved rounds with
  alternating order and the guard (foreign load ≈ 0.1 core in clean
  rounds), p50 ms: t128 29.3-29.5 vs 29.5-29.7, t512 136.0-136.5 vs
  132.2-134.0, i280 963-986 vs 944-969, a15 238-242 vs 237-240, i70
  146-152 vs 152.4-153.5. HEAD is 1-3% faster on t512/i280 and ~2%
  slower on i70 (the extra V-plane pass of 14 at 630 patches outweighs the
  attention gain there); the large-image gain of 14 is in the final matrix.

## 23. Final P8 confirmations (`5dd7dbf`)

* Attention layout (14) on the P-cores, `test_gemma2_attn bench`, 8 P-core
  threads, best of 5 per run, two runs: n=2520 274.5 / 288.1 → 283.5 /
  293.9 GMAC/s (+2-3%), n=10080 242.5 / 251.7 → 277.0 / 278.6 GMAC/s
  (+11-14%) — `reports/raw/attn_layout_P8.txt`.
* Head-of-line blocking (19) at P8 (`reports/raw/hol_*_P8.json`; t16 alone
  ≈ 11 ms): during an i280 request t16 p50 525 → 9.0 ms (max 971 → 79 ms,
  2 → 177 served); during an i1120 request 5,487 → 8.9 ms (max 10,569 →
  588 ms, 2 → 657 served; the image 11.0 → 15.7 s, i.e. by the text work
  done meanwhile); during a30 263 → 31.5 ms.
* Final matrix with the quiet-host guard: `reports/benchmarks.*` (fork
  `5dd7dbf`, one run per budget; baselines best of two runs). Embeddings of
  all 16 workloads are bit-identical to `97fdcc7` and between P8 and ALL,
  so the quality and retrieval results apply unchanged.
* Quantization study (fixed, `reports/quant_study.json`; linear layers:
  218 text, 114 vision, 135 audio): Q8_0 weights everywhere cost at most
  0.0007 cosine (min 0.99932, a5); 4-bit weights cost 0.036 on text
  (min 0.964), 0.012 on vision, 0.067 on audio, 0.084 overall (min 0.916).
  4-bit weights are not an acceptable trade for this model; Q8 stays.
