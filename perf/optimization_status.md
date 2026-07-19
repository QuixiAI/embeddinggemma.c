# Optimization Status

Raw benchmark output belongs under `perf/results/`; durable conclusions belong
here.

## Entry Template

```text
## YYYY-MM-DD: <kernel or pass name>

Status: not started | baselining | experimenting | candidate | landed | deferred.
Current implementation:
Current public route:
References inspected:
Correctness:
Baseline:
Experiments:
Decision:
Open questions:
Raw results:
```

Record enough context to reproduce the run: machine, OS, compiler/toolchain,
backend, command, git label, dtype, shape, quant format, warmups, iterations,
median, variance, correctness tolerance, observed error, and raw result path.

## 2026-07-19: Perf Scaffold

Status: landed.

Current implementation: added a C-native benchmark harness and Python runner for
the active CPU kernel surface.

Current public route: `python3 perf/bench_kernels.py --preset quick`.

References inspected:

- `/Users/eric/QuixiCore/QuixiCore-Metal/perf/README.md`
- `/Users/eric/QuixiCore/QuixiCore-Metal/perf/perf.md`
- `/Users/eric/QuixiCore/QuixiCore-Metal/perf/bench_kernels.py`
- `/Users/eric/QuixiCore/QuixiCore-Metal/perf/baseline_status.md`
- `/Users/eric/QuixiCore/QuixiCore-Metal/perf/optimization_status.md`

Correctness: harness includes scalar checks for q4/q8 dot and matmul rows.

Baseline: not recorded yet.

Decision: mirror QuixiCore's result layout and notebook discipline, but keep the
first harness native C because this repository currently exposes C kernels.

Open questions: add accelerator-backed cases once Metal/CUDA/ROCm/SYCL launch
APIs are wired into the repo.

Raw results: none.

## 2026-07-19: CPU And Metal Kernel Completion

Status: landed.

Current implementation: NEON/AVX2/SSE CPU row kernels, vectorized full/SWA
attention, and fused QK norm+RoPE, norm+residual, and final norm+pool paths.
Metal has family-separated translation units compiled into one metallib using
the QuixiCore-Metal direct `xcrun metal` build pattern.

References inspected:

- `llama.cpp/ggml/src/ggml-cpu/*/quants.c`
- `llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`
- `QuixiCore-CPU/kernels/quantization/qgemv_neon.cpp`
- `QuixiCore-Metal/kernels/quantization/qgemv/qgemv.metal`
- `QuixiCore-Metal/kernels/norms/rms_norm/rms_norm.metal`
- `QuixiCore-Metal/kernels/norms/add_norm/add_norm.metal`
- `QuixiCore-Metal/kernels/norms/qk_norm_rope/qk_norm_rope.metal`
- `QuixiCore-Metal/kernels/attention/attn_multiwarp/attn_multiwarp.metal`
- `QuixiCore-Metal/bindings/mlx/cmake/extension.cmake`

Correctness: CPU kernel reference tests pass with maximum attention error
`1.61e-6`; model parity passes all 10 samples with minimum cosine
`0.999862850` against llama.cpp.

Baseline: see `perf/baseline_status.md`. Q4 GEMV is currently about
`31 GFLOP/s` and `8.7` packed-weight GB/s; T=128 attention is about
`35 GFLOP/s`.

Decision: retain direct GGUF block consumption on both backends. Metal offers
both direct Q4_0 x F32 and CPU-parity Q4_0 x Q8_0 projection routes. Add the
model-specific online-softmax attention and final pooling kernels because the
generic decomposed forms add avoidable launches and global-memory traffic.

Metal build validation: Xcode 26.6 with Metal Toolchain `17F109` compiles the
seven translation units into `embeddinggemma.metallib`. At this stage, the
native smoke test loaded the library and created all 12 expected compute
pipelines on Apple M5 Max; the final inventory is 24.

Open questions: benchmark direct F32 versus activation-quantized Metal
projection and add dispatch timings before tuning simdgroups/output rows.

Raw results: `perf/results/2026-07-19/000431-cpu-quick/`.

## 2026-07-19: Optimization Loop Pass 1 - Projection Accumulation And FFN Fusion

Status: CPU landed; Metal experiment rejected. Significant-gain streak reset;
consecutive no-gain passes: 0.

Current implementation: CPU Q4_0 x Q8_0 keeps NEON/AVX2/SSSE3 SIMD lane
accumulators live across the complete K dimension and horizontally reduces once.
For one-token inference, bounded-row Q4 projections use a four-way schedule on
the persistent worker pool; per-worker conditions wake only the workers assigned
to a job. Metal retains the separate up/gate projection plus GELU dispatch.

Current public route: CPU SIMD projection for all Q4 matmuls, with row-parallel
projection at T=1 and token-parallel projection otherwise. Metal uses fused QKV
and up/gate dispatches, but not the experimental GELU epilogue fusion.

References inspected:

- `llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c`
- `llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c`
- `QuixiCore-Metal/perf/perf.md` (decision threshold and fusion rules)
- `QuixiCore-Metal/kernels/quantization/qgemv/qgemv.metal`

Correctness: CPU model parity 10/10, minimum cosine `0.999860704`; experimental
Metal fusion parity 10/10, minimum cosine `0.999910414`. Q4 microbench maximum
relative error after reassociation was `4.24e-5`, within model parity tolerance.

Baseline and retained CPU result on Apple M5 Max, `-O2`, 5 warmups and 30 timed
samples for kernels:

| shape | before ms | after ms | improvement |
|---|---:|---:|---:|
| Q4 GEMV N768 K768 | 0.04365 | 0.03871 | 11.3% |
| Q4 GEMV N1152 K768 | 0.06395 | 0.05693 | 11.0% |
| Q4 GEMV N768 K1152 | 0.06567 | 0.05389 | 17.9% |
| full graph, T=1 | 6.43 | 2.81 | 56.3% |

Metal experiment: compute up and gate for the same row in one kernel and apply
GELU in registers, removing the up intermediate and one dispatch per layer.
Controlled same-binary A/B with 5-8 warmups and 30-50 iterations found only
1-3% at T=1..3, regressions of 2-3% at T=4/8, and less than 1.2% at T=32/128.

Decision: retain CPU SIMD accumulation, bounded-row API, exact worker wakeups,
and short-input row scheduling. Remove the Metal FFN epilogue kernels: their
small-shape result is below the 8-10% threshold warranted by the added kernel
complexity and does not hold at the GEMM crossover.

Open questions: vectorize CPU Q8 quantization; sweep Metal GEMV/GEMM crossover
and tile-token geometry with the durable engine harness.

Raw results: `perf/results/2026-07-19/023732-pass0-baseline/` and
`perf/results/2026-07-19/023900-pass1-ffn-fusion/`.

## 2026-07-19: Optimization Loop Pass 2 - Q8 Packing And Metal Route Crossover

Status: CPU and Metal landed. Significant-gain streak reset; consecutive
no-gain passes: 0.

Current implementation: CPU Q8_0 activation quantization uses ties-away NEON
conversion plus saturating vector narrows (and an equivalent AVX2 pack path).
Metal selects direct Q4 GEMV below 160 tokens and the staged 32-row x 8-token
Q4 GEMM at 160 tokens and above. `EI_METAL_GEMM_MIN_TOKENS` remains an
experimental override for repeatable route sweeps.

References inspected:

- `llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c` (`quantize_row_q8_0`)
- `llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c`
- `QuixiCore-Metal/perf/perf.md` (routing and Apple staging guidance)
- `QuixiCore-Metal/perf/optimization_status.md` (direct-vs-staged findings)

Correctness: CPU and Metal model parity each pass 10/10. Minimum cosine is
`0.999860704` for CPU and `0.999910235` for Metal. Q8 quantization remains exact
against the scalar ties-away oracle in the native harness.

CPU experiment, 5 warmups and 30 samples:

| shape | scalar-store ms | SIMD-pack ms | improvement |
|---|---:|---:|---:|
| Q8 quant K768 | 0.0001216 | 0.0000973 | 20.0% |
| Q8 quant K1152 | 0.0001802 | 0.0001636 | 9.2% |

Metal experiment: sweep `gemm_min_tokens` through
`1,2,4,8,16,32,64,128,256,512,2049` with 4-5 warmups and 15-30 iterations.
The staged GEMM's threadgroup-memory traffic and two barriers per K block lose
to direct cache-resident loads at small and medium M.

| tokens | old route ms | retained route ms | improvement |
|---|---:|---:|---:|
| 4 | 5.60 GEMM | 2.06 GEMV | 63.2% |
| 8 | 6.04 GEMM | 2.43 GEMV | 59.8% |
| 32 | 7.48 GEMM | 4.78 GEMV | 36.1% |
| 64 | 10.83 GEMM | 8.32 GEMV | 23.2% |
| 128 | 18.86 GEMM | 17.24 GEMV | 8.6% |
| 224 | 33.91 GEMV | 30.71 GEMM | 9.4% |
| 256 | 39.59 GEMV | 34.65 GEMM | 12.5% |
| 512 | 81.09 GEMV | 66.84 GEMM | 17.6% |

Decision: retain both changes. A threshold of 160 is conservative at the noisy
crossover (T=160 was approximately equal); it captures the large direct-load
wins below that point and the staged reuse wins above it.

Open questions: test Q4 dual-row CPU accumulation; tune or remove staging inside
the long-M Metal GEMM now that its useful shape range is known.

Raw results: `perf/results/2026-07-19/024309-pass2-metal-threshold-1/` through
`024315-pass2-metal-threshold-2049/`, `024331-pass2-metal-long-threshold-32/`
through `024345-pass2-metal-long-threshold-2049/`, and
`perf/results/2026-07-19/024454-pass2-retained/`.

## 2026-07-19: Optimization Loop Pass 3 - CPU Width And Metal 16x16 Tile

Status: CPU and Metal landed. Significant-gain streak reset; consecutive
no-gain passes: 0.

Current implementation: the one-token CPU projection path uses six total
threads. Long-prefill Metal Q4 GEMM uses a 16-output-row x 16-token tile with
256 threads; direct GEMV remains selected below 64 tokens.

References inspected:

- `QuixiCore-Metal/perf/perf.md` (tile sweeps, staging/reuse, route retesting)
- `QuixiCore-Metal/kernels/quantization/qgemm/qgemm.metal`
- `llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`

Correctness: CPU and Metal llama.cpp parity remain 10/10 with minimum cosines
`0.999860704` and `0.999910235`. A new route-boundary test compares T=
`1,4,63,64,96,160,192,512`: old Metal 32x8 and new 16x16 output cosine is
`1.000000000` for every shape. CPU/Metal synthetic-token cosine is reported as
a diagnostic (minimum `0.997155726`) rather than used as an oracle because the
CPU route quantizes activations and the Metal route intentionally consumes F32.

CPU experiment: sweep `EI_CPU_SHORT_THREADS=1,2,3,4,5,6,7,8`, 5 warmups and
30-50 iterations at T=1. Six threads measured `2.52-2.55 ms` versus
`2.79-2.85 ms` for four threads, a repeatable 9-11% improvement. Seven and
eight threads regressed to `3.34-3.68 ms` on the heterogeneous M5 Max.

Metal experiment: compare 32x8 and 16x16 in the same metallib, 4 warmups and 12
iterations. The 16x16 tile halves staged weight traffic over the token grid
while preserving 256 outputs per threadgroup.

| tokens | 32x8 ms | 16x16 ms | improvement |
|---|---:|---:|---:|
| 160 | 23.09 | 17.96 | 22.2% |
| 192 | 26.85 | 20.85 | 22.4% |
| 256 | 34.94 | 27.05 | 22.6% |
| 512 | 67.34 | 52.21 | 22.5% |
| 1024 | 137.24 | 107.59 | 21.6% |

The crossover was re-swept after changing the tile: GEMV wins 15-18% at T=32,
the routes are effectively equal at T=64, and 16x16 GEMM wins 9% at T=96 and
15-20% at T=128/160. The retained default threshold is therefore 64.

Decision: retain six-way CPU projection, the 16x16 Metal QKV/general/up-gate
GEMMs, and the 64-token crossover. Keep the old 32x8 kernels as a tested route
and A/B control for now; `EI_METAL_GEMM_TILE_TOKENS=8|16` selects them.

Open questions: test CPU dual-output Q4 accumulation and Metal tile rows/tokens
beyond the two balanced 256-output geometries.

Raw results: `perf/results/2026-07-19/025144-pass3-retained-rebuilt/` (the
earlier `025102` run is explicitly stale and must not be used).

## 2026-07-19: Optimization Loop Pass 4 - Dual CPU Dot And Metal Storage Mode

Status: CPU landed; Metal experiment rejected. Significant-gain streak reset;
consecutive no-gain passes: 0.

Current implementation: paired CPU projections use a dual-output Q4_0 x Q8_0
dot. K/V and up/gate share each Q8 activation load and scale conversion while
maintaining independent SIMD accumulators. Metal model weights remain in a
shared unified-memory buffer.

References inspected:

- `llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c` (multi-result dot paths)
- `llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c`
- `QuixiCore-Metal/perf/perf.md` (memory-byte hypothesis and A/B rules)

Correctness: CPU llama.cpp parity remains 10/10, minimum cosine
`0.999860704`; controlled dual on/off output checksums are identical.

CPU controlled same-binary A/B, 4 warmups and 20 iterations, two alternating
runs per route:

| tokens | separate ms | dual ms | improvement |
|---|---:|---:|---:|
| 4 | 8.22 | 7.51 | 8.6% |
| 8 | 13.59 | 12.45 | 8.4% |
| 32 | 32.17 | 29.83 | 7.3% |
| 128 | 85.03 | 75.64 | 11.0% |

T=1 was noisy and only about 3%, but the realistic multi-token shapes all clear
the local-change threshold. The retained T=512 median is `298.96 ms` versus
`368.48 ms` in the pre-dual Pass 3 run, though that cross-run delta also includes
system variance and is not used as the acceptance metric.

Metal experiment: copy the 271 MiB tensor data section to
`MTLResourceStorageModePrivate` once at initialization, then compare with the
shared buffer. Two alternating runs at T=`1,32,128,512` differed by less than
1% at every shape (`52.11-52.19 ms` at T=512). Unified-memory caches erase the
expected storage-mode distinction.

Decision: retain dual CPU projections and the scalar fallback/AVX2 equivalent.
Remove the Metal private-storage branch because it adds upload time, memory, and
code without a runtime gain.

Open questions: prefetching and output-row unrolling in the CPU dot; Metal
attention/GQA reuse now that Q4 projection is substantially faster.

Raw results: `perf/results/2026-07-19/025746-pass4-retained/`.

## 2026-07-19: Optimization Loop Pass 5 - Online CPU Attention And Shared GQA

Status: experiments rejected. Consecutive no-significant-gain passes: 1 of 3.

Current implementation: unchanged retained three-pass CPU attention and
single-simdgroup-per-query/head Metal online attention.

References inspected:

- `QuixiCore-Metal/kernels/attention/attn_multiwarp/attn_multiwarp.metal`
- `QuixiCore-Metal/perf/optimization_status.md` (GQA staging rejection)
- `llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` attention kernels

Correctness: both experiments produced finite outputs. Metal shared-GQA output
checksums were identical to the retained route. CPU online accumulation changed
sampled full-graph output values, so it would require a fresh tolerance audit
even if it were faster.

CPU experiment: fuse max tracking, exponentiation, and V accumulation into one
online pass, eliminating the score buffer and separate softmax pass. Two
alternating runs at T=128/512 (3 warmups, 10 iterations) regressed roughly
6-8%: `78.7-83.6 ms` retained versus `83.6-90.5 ms` online at T=128, and
`316.5-344.5 ms` versus `339.1-365.3 ms` at T=512. Repeated output rescaling
when the running maximum changes outweighs the removed scalar score pass.

Metal experiment: one 96-thread group per query, one simdgroup per Q head,
staging the shared GQA K/V head once. At T=`128,512,1024,2048` it regressed
approximately 3-6% in the first alternating pair and more under sustained load.
The two barriers per key and 3x smaller threadgroup grid cost more than the
reduced K/V reads, reproducing QuixiCore's prior Apple result.

Decision: remove both experimental implementations. This is the first pass
without a retained >=3% realistic-shape gain.

Open questions: CPU prefetch/unroll and Metal command-buffer/dispatch overhead.

Raw results: focused same-binary command output only; no retained result set.

## 2026-07-19: Optimization Loop Pass 6 - Compiler Optimization Levels

Status: experiments rejected. Consecutive no-significant-gain passes: 2 of 3.

Current implementation: unchanged C `-O2` and Metal `-O2 -fno-fast-math`.

References inspected:

- `QuixiCore-Metal/perf/perf.md` (variance, warmup, and numerical decision rules)
- Xcode Metal compiler help for supported optimization flags

Correctness: O2/O3 output checksums were identical on both backends. Fast math
was not enabled, so this pass did not relax numerical semantics.

CPU experiment: separately linked O2 and O3 full-engine binaries, alternated at
T=`1,4,32,128,512`, 4 warmups and 15 iterations. In the stabilized second pair,
O3 was neutral at T=4 (`8.67` versus `8.74 ms`) and 1-2% slower at T=128/512
(`86.82/351.53` versus `85.16/346.84 ms`). The earlier pair favored O2 more
strongly and confirms there is no hidden O3 win.

Metal experiment: separately compiled O2 and O3 metallibs, selected through
`EI_METALLIB_PATH`, alternated at T=`32,128,512,2048`. Results were order- and
temperature-sensitive: O3 was roughly 20% slower than the first cool O2 run,
then about 7% faster than a heat-soaked O2 run. This is variance, not a compiler
effect; no consistent shape-local advantage survived both pairs.

Decision: retain O2. O3 adds no reproducible benefit and is not accepted based
on a thermally confounded run.

Open questions: launch overhead and small row-kernel fusion.

Raw results: focused alternate-binary command output only; experimental
artifacts remain under ignored `build/` until the final clean rebuild.

## 2026-07-19: Optimization Loop Pass 7 - Embedding Parallelism And QK Launch Fusion

Status: Metal landed; CPU experiment rejected. Significant-gain streak reset;
consecutive no-gain passes: 0.

Current implementation: CPU embedding gather remains sequential. Metal combines
Q and K RMSNorm, NEOX RoPE, and Q scaling in one four-head dispatch per layer.

References inspected:

- `QuixiCore-Metal/kernels/norms/qk_norm_rope/qk_norm_rope.metal`
- `QuixiCore-Metal/perf/perf.md` (launch-bound fusion criteria)
- `llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` RoPE specialization

Correctness: CPU and Metal llama.cpp parity remain 10/10 at minimum cosines
`0.999860704` and `0.999910235`. Fused/unfused Metal output checksums are
identical for every benchmark shape.

CPU experiment: parallelize Q8 embedding row gathers with grain 4. The isolated
row cost is too small to amortize a pool launch. Alternating runs at
T=`4,32,128,512` show the parallel route consistently slower, roughly 4-11% in
the first pair, with no output difference.

Metal experiment: replace separate Q and K norm/RoPE dispatches with one
model-specialized dispatch over the fixed 3Q+1K heads. Two alternating runs,
5 warmups and 20 iterations:

| tokens | separate ms | fused ms | improvement |
|---|---:|---:|---:|
| 1 | 1.90-1.96 | 1.74-1.75 | 8-11% |
| 4 | 2.11-2.13 | 1.98-1.99 | 6-7% |
| 32 | 5.58-5.61 | 5.38-5.44 | 3-4% |
| 128 | 15.92-15.96 | 15.80 | <1% |
| 512 | 56.98-57.13 | 56.87-57.07 | neutral |

Decision: remove CPU embedding parallelism. Retain fused QK norm/RoPE for all
Metal shapes because it clears the local-change threshold on latency-sensitive
short inputs and does not regress long inputs. `EI_METAL_FUSED_QK_ROPE=0`
keeps a reproducible diagnostic route.

Open questions: fuse other tiny Metal row operations only where dispatch count
can fall without repeating projection work.

Raw results: `perf/results/2026-07-19/030744-pass7-retained/`.

## 2026-07-19: Optimization Loop Pass 8 - CPU Row Prefetch And Metal Fast Math

Status: experiments rejected. Consecutive no-significant-gain passes: 1 of 3.

Current implementation: unchanged compiler semantics and hardware-managed CPU
cache behavior.

References inspected:

- `QuixiCore-Metal/perf/perf.md` (minimum-win and numerical acceptance rules)
- Apple Metal compiler help for `-ffast-math` and `-fno-fast-math`

Correctness: the fast-math metallib passed all 10 llama.cpp golden vectors at
minimum cosine `0.999910474`. CPU prefetch did not change output checksums.

CPU experiment: prefetch each packed Q4 row two output rows ahead in single and
dual projections. A second stabilized alternating pair at T=`1,4,32,128`
measured retained/prefetch times of `3.865/3.987`, `9.465/9.566`,
`32.707/33.005`, and `89.024/87.198 ms`. The only improvement was 2.1% at
T=128, below the local-change threshold, while short inputs regressed. Apple
hardware prefetching already covers this sequential access pattern.

Metal experiment: compile the same source with `-O2 -ffast-math`. Two
alternating pairs at T=`1,4,32,128,512` found realistic-shape improvements of
0-2.7%. The apparent 5-8% T=1/4 result occurred in only one pair at the GPU
launch floor and did not reproduce consistently; T=32 was neutral in the
second pair and T=128/512 remained below 2%.

Decision: remove explicit Q4 prefetch and retain Metal `-fno-fast-math`. Neither
experiment provides a stable >=3% gain, and relaxed floating-point semantics
are not justified by sub-threshold throughput changes.

Open questions: CPU attention job grain and redundant exponentials in Metal
online softmax.

Raw results: focused alternate-binary and alternate-metallib command output;
experimental artifacts remain under ignored `build/` until the final clean.

## 2026-07-19: Optimization Loop Pass 9 - CPU Attention Grain And One-Exp Metal Softmax

Status: experiments rejected. Consecutive no-significant-gain passes: 2 of 3.

Current implementation: unchanged one-query CPU scheduling and branchless
two-exponential Metal online-softmax recurrence.

References inspected:

- `QuixiCore-Metal/kernels/attention/attn_fwd/`
- `QuixiCore-Metal/perf/perf.md` (latency, divergence, and variance guidance)
- `llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` online attention recurrence

Correctness: CPU grain variants produced identical checksums. The one-exp
Metal variant passed all 10 golden vectors at minimum cosine `0.999910235` and
also produced identical benchmark checksums.

CPU experiment: group 2 or 4 adjacent attention queries into each persistent
pool job instead of one query per job. In the first alternating run at
T=`32,128,512`, grain 1 measured `30.14/79.66/319.80 ms`, grain 2
`31.33/84.02/338.65 ms`, and grain 4 `31.24/85.62/352.09 ms`. A reverse-order
run moved the apparent winner with temperature and did not reproduce a stable
advantage for either coarser grain. Fewer atomic job claims do not compensate
for reduced load balancing across queries with different attention spans.

Metal experiment: replace the branchless `max` update and two exponentials
with a simdgroup-uniform branch and exactly one exponential per key. In the
first pair at T=`32,128,512,1024,2048`, it regressed roughly 4-7%. The reverse
pair was approximately tied through T=1024, while T=2048 became thermally
unstable. No shape showed a reproducible gain. The compiler/SFU can overlap the
branchless formulation more effectively than the explicit control dependency.

Decision: retain grain-1 CPU attention and branchless Metal online softmax.
Neither change clears the handbook threshold in both A/B orderings.

Open questions: eliminate repeated RoPE `pow` work, which is pure scalar side
work on both backends.

Raw results: focused same-binary CPU and alternate-metallib Metal command
output; no retained result set.

## 2026-07-19: Optimization Loop Pass 10 - Precomputed Metal RoPE

Status: Metal landed; CPU experiment rejected. Significant-gain streak reset;
consecutive no-gain passes: 0.

Current implementation: CPU retains direct scalar `powf/cosf/sinf`. The Metal
engine builds model-specific full/SWA `[cos,sin]` tables once and fused QK
norm/RoPE kernels consume them as `float2` values.

References inspected:

- `llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp` RoPE argument handling
- `QuixiCore-Metal/kernels/norms/qk_norm_rope/qk_norm_rope.metal`
- `QuixiCore-Metal/perf/perf.md` scalar-side-work and realistic-shape rules

Correctness: CPU kernel tests remain exact and llama.cpp parity remains 10/10
at minimum cosine `0.999860704`. Metal parity is 10/10 at minimum cosine
`0.999910295`; output differences from device transcendental approximations are
well inside the existing tolerance.

CPU experiment: use the same 2 MiB full/SWA table in `build_qkv`. The first
baseline/table pair showed 2-10% table regressions, while the reverse pair
showed 2-5% apparent gains. Run order, not the variant, controlled the result.
The CPU path therefore keeps its existing math and does not pay table startup
or memory costs.

Metal experiment: replace per-layer, per-head `powr/cos/sin` with a cached
`float2` load. In the first baseline/table pair the table improved T=1 by 4.0%,
T=1024 by 1.7%, and other shapes by 0-1%. In the reverse pair it improved
T=`4,32,128,1024,2048` by approximately `3.5,7.3,4.7,2.1,4.3%`; T=1 was
1.6% slower. Across both orders there is no long-shape regression and several
latency-sensitive shapes clear the 3% local-change threshold.

Decision: retain Metal-only precomputed RoPE tables. Generate them in the Metal
engine so CPU model load and inference remain unchanged. This removes repeated
special-function work without adding a dispatch.

Open questions: reduce the four identical table reads per token/dimension by
sharing values inside one 128-thread QK threadgroup, but measure the required
barrier before retaining it.

Raw results: `perf/results/2026-07-19/032143-pass10-retained/` plus focused
alternate-binary/metallib A/B output.

## 2026-07-19: Optimization Loop Pass 11 - Fused CPU QK And Shared Metal RoPE

Status: CPU landed; Metal experiment rejected. Significant-gain streak reset;
consecutive no-gain passes: 0.

Current implementation: CPU jointly normalizes and rotates fixed 3Q+1K heads,
evaluating each angle once and folding Q scaling into the final write. Metal
retains one independent simdgroup per Q/K head.

References inspected:

- `QuixiCore-Metal/kernels/norms/qk_norm_rope/qk_norm_rope.metal`
- `QuixiCore-Metal/perf/optimization_status.md` barrier/staging rejections
- `llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` simdgroup RoPE layout

Correctness: the fused CPU kernel is bit-identical to separate Q and K paths in
the focused test. CPU and Metal llama.cpp parity remain 10/10 at minimum
cosines `0.999860704` and `0.999910295`, respectively.

CPU experiment: compute four RMS factors up front, evaluate each RoPE angle once
for Q and K, combine norm/rotation/Q-scale writes, and avoid a second trig loop.
Across two orderings, T=1 improved from `3.565/3.561 ms` to `3.364/3.401 ms`
(4.5-5.6%). The first pair was mixed by -4% to +1% at T=4..512; the reverse
pair favored fusion by 2.6-10%. The repeatable short-input win clears 3%, and
no long-shape regression reproduces across orderings.

Metal experiment: put all four heads in one 128-thread group, load 128 `float2`
table entries once into threadgroup memory, and synchronize once. It regressed
T=1 by 3-6% in both orders and T=32 by about 1.5%; T>=128 was neutral. The
cache-resident duplicate reads are cheaper than a barrier and larger scheduling
unit.

Decision: retain fused CPU QK norm/RoPE and its exact unit test. Restore Metal's
four independent 32-thread groups.

Open questions: CPU trig calls are still scalar; direct platform `sincosf`
availability and Metal Q4 GEMV output-row sharing remain candidates.

Raw results: focused alternate-binary/metallib A/B output and
`perf/results/2026-07-19/032638-pass11-retained/`.

## 2026-07-19: Optimization Loop Pass 12 - CPU RoPE Recurrence And Metal R4 GEMV

Status: Metal landed; CPU experiment rejected. Significant-gain streak reset;
consecutive no-gain passes: 0.

Current implementation: CPU retains exact per-dimension RoPE powers. Metal uses
one-row Q4 GEMV below T=8, four-row register-reuse GEMV at T=8..63, and 16x16
GEMM at T>=64.

References inspected:

- `llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h` (`N_R0_Q4_0=4`)
- `llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`
  `mul_vec_q_n_f32_impl`
- `QuixiCore-Metal/perf/perf.md` route-boundary and packed-weight guidance

Correctness: the R4 kernels preserve checksums and Metal llama.cpp parity is
10/10 at minimum cosine `0.999910295`. CPU recurrence passed broad golden
parity but missed the focused K tolerance by `5.66e-6`; after rebasing every
eight dimensions it still missed and changed long-shape checksums.

CPU experiment: derive each RoPE frequency from a geometric recurrence, reducing
128 `powf` calls to one, then to 17 with periodic rebasing. The bounded version
regressed T=32..512 by roughly 7-10% in its first A/B pair and did not satisfy
the existing per-element test. Exact `powf` remains the production route.

Metal experiment: cache the 16 activation values owned by each lane and update
four output-row accumulators, matching llama.cpp's Q4_0 row factor. The route
was implemented for generic, QKV, and up/gate direct projections. Across two
sequential orderings, R4 improved T=8 by about 12%, T=16 by 16%, T=32 by
14-21%, and T=63 by 21-39%. T=1 was inconsistent, so it stays on R1; T>=64
continues to use the retained GEMM route.

Decision: retain separate R1 and R4 kernels and default
`EI_METAL_GEMV_R4_MIN_TOKENS=8`. The environment route accepts 1..64 for
reproducible diagnostics. Reject CPU recurrence.

Open questions: test R2 as a lower-register route for T=1..7; do not assume R4
is optimal at the launch floor.

Raw results: focused alternate-binary/metallib A/B output and
`perf/results/2026-07-19/033410-pass12-retained/`.

## 2026-07-19: Optimization Loop Pass 13 - CPU Triple QKV And Metal R2 GEMV

Status: CPU landed; Metal experiment rejected. Significant-gain streak reset;
consecutive no-gain passes: 0.

Current implementation: one-token CPU QKV projection shares each Q8 activation
block across corresponding Q, K, and V rows, then computes remaining Q rows.
Multi-token CPU inference keeps separate Q plus dual K/V. Metal keeps R1 below
T=8 and R4 from T=8 through 63.

References inspected:

- `llama.cpp/ggml/src/ggml-cpu` multi-row Q4 GEMV declarations
- `llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` register-cached activation
  layout
- `QuixiCore-CPU/kernels/quantization/qgemv_neon.cpp`

Correctness: triple NEON/AVX2/scalar paths preserve full CPU checksums and
llama.cpp parity remains 10/10 at minimum cosine `0.999860704`. Metal R2 also
preserved checksums and 10/10 parity at `0.999910295`.

CPU experiment: compute the first 256 Q/K/V rows together, keeping Q8 values
live across three dot products, and reduce one-token scheduler work from 1280
logical rows to 768. Across two orderings, T=1 improved from `3.522/3.665 ms`
to `3.083/3.099 ms` (12-15%). Multi-token results were mixed in the first pair
and favored triple only when run first, so token-parallel inference retains the
lower-risk existing route.

Metal experiment: change the R4 kernels to two output rows and compare R1, R2,
and R4 at T=`1,4,7`. R2 was order-sensitive at T=1, improved T=4 by only 1%,
and improved T=7 by 3-4% versus R1 but was effectively tied with R4. Adding a
third route has no justified range.

Decision: retain triple QKV only for CPU T=1, including NEON and AVX2
implementations. Restore Metal R4 and its T=8 threshold.

Open questions: one-token QKV scheduler width may change now that logical work
is 40% smaller; re-sweep only that width rather than the global short width.

Raw results: focused alternate-binary/metallib A/B output and
`perf/results/2026-07-19/033914-pass13-retained/`.

## 2026-07-19: Optimization Loop Pass 14 - CPU Width And Metal T=7 Boundary

Status: Metal route update landed; CPU experiment rejected. Significant-gain
streak reset; consecutive no-gain passes: 0.

Current implementation: CPU one-token projections retain width 6. Metal uses
R1 below T=7, R4 at T=7..63, and GEMM at T>=64.

References inspected:

- `QuixiCore-Metal/perf/perf.md` route-boundary and variance rules
- Pass 3 CPU width sweep and Pass 12 R1/R4 measurements in this notebook

Correctness: every route/width produced identical checksums. No arithmetic was
changed.

CPU experiment: re-sweep `EI_CPU_SHORT_THREADS` after triple QKV reduced the
logical one-token projection from 1280 to 768 rows. An initial cool run favored
width 4 by over 20%, but a sustained alternating width 4/5/6 run measured only
1-2.5% differences and changed apparent ordering. The initial result is thermal,
not a reproducible scheduler win.

Metal experiment: compare R1 (`EI_METAL_GEMV_R4_MIN_TOKENS=8`) with R4
(`=7`) at exactly T=7 using 10 warmups and 50 iterations in four alternating
pairs. R1 medians were `2.309,2.312,2.345,2.3135 ms`; paired R4 medians were
`2.229,2.2315,2.2465,2.2205 ms`, a consistent 3.5-4.2% gain.

Decision: retain CPU width 6. Lower the default Metal R4 threshold from 8 to 7;
T<=6 remains on R1 and all other route boundaries stay unchanged.

Open questions: reduce CPU one-token pool synchronization itself; test Metal
R4 at T=6 only if a controlled run clears 3%.

Raw results: focused high-iteration same-binary route A/B output and
`perf/results/2026-07-19/034112-pass14-retained/`.

## 2026-07-19: Optimization Loop Pass 15 - CPU Adjacent Rows And Metal T=6

Status: CPU landed; Metal route experiment rejected. Significant-gain streak
reset; consecutive no-gain passes: 0.

Current implementation: one-token CPU generic Q4 projections process three
adjacent weight rows with a shared Q8 activation load. Metal R4 still starts at
T=7.

References inspected:

- `llama.cpp/ggml/src/ggml-cpu` multi-row Q4 GEMV kernels
- `QuixiCore-CPU/kernels/quantization/qgemv_neon.cpp`
- `QuixiCore-Metal/perf/perf.md` minimum route-win guidance

Correctness: the adjacent-row helper uses the exact triple dot from Pass 13;
focused output error is zero and CPU llama.cpp parity remains 10/10 at minimum
cosine `0.999860704`. Metal route checksums were identical.

CPU experiment: group adjacent rows in attention-output/down projections and
the remaining Q projection into triple NEON/AVX2 accumulators. Three alternating
T=1 comparisons measured retained/candidate medians of `3.2665/2.9395`,
`3.3135/2.8875`, and `3.2275/2.7965 ms`, consistent 10-13% gains.

Metal experiment: compare R1 (`threshold=7`) and R4 (`threshold=6`) at T=6
with four 10-warmup/50-iteration alternating pairs. R4 improved only 0.8-1.4%,
below the 3% route threshold.

Decision: retain CPU adjacent-row triple accumulation for one-token single
matrices. Keep Metal R4 minimum at 7.

Open questions: up/gate already shares two matrices but not adjacent rows;
raising accumulator count further may hit register pressure. Pool wakeup cost is
now a larger fraction of T=1 latency.

Raw results: focused alternate-binary CPU and same-binary Metal route output and
`perf/results/2026-07-19/034324-pass15-retained/`.

## 2026-07-19: Optimization Loop Pass 16 - CPU Dual Adjacent Rows And Metal T=5/6

Status: experiments rejected. Consecutive no-significant-gain passes: 1 of 3.

Current implementation: unchanged dual up/gate rows and Metal R4 threshold 7.

References inspected:

- Pass 15 adjacent-row CPU results
- `QuixiCore-Metal/perf/perf.md` low-risk 3% acceptance threshold

Correctness: all variants produced identical checksums; the CPU candidate also
passed 10/10 llama.cpp parity at `0.999860704`.

CPU experiment: replace three adjacent dual up/gate calls (three Q8 scans) with
one triple-up plus one triple-gate call (two scans). Three alternating T=1
pairs showed 2.6%, 1.0%, and 6.9% improvements. The effect is smaller than run
variance and does not clear 3% consistently.

Metal experiment: R4 versus R1 at T=6 improved only 0.8-1.4% across four pairs.
At T=5, four additional pairs improved 0.8-2.6%. Both are below threshold.

Decision: remove CPU dual-adjacent helper and retain the simpler dual dot.
Keep Metal R4 minimum at 7. This is the first pass without a retained gain
since Pass 15.

Open questions: command encoding and pool synchronization overhead; further
row-factor changes now have diminishing returns.

Raw results: focused alternating command output only; no retained result set.

## 2026-07-19: Optimization Loop Pass 17 - CPU Thread Saturation And Metal T=4

Status: experiments rejected. Consecutive no-significant-gain passes: 2 of 3.

Current implementation: CPU defaults to all 18 online cores; Metal uses R1 at
T=4.

References inspected:

- `QuixiCore-Metal/perf/perf.md` realistic-shape and route guidance
- current persistent CPU pool scheduling in `src/parallel.c`

Correctness: all CPU thread counts and Metal routes produced identical
checksums.

CPU experiment: compare 8, 12, and 18 total threads at T=`32,128,512` in
forward and reverse order. Eighteen threads measured `29.3-31.7`,
`75.1-81.6`, and `314.1-325.1 ms`; 12 threads measured `30.9-32.3`,
`100.3-102.8`, and `413.3-416.5 ms`. Eight threads was substantially slower.
Efficiency cores contribute useful projection throughput; a performance-core
cap is a regression.

Metal experiment: compare R1 and R4 at T=4 in four 10-warmup/50-iteration
pairs. R1 medians were `1.9335-1.9385 ms`; R4 was `1.9370-1.9585 ms`, neutral
to roughly 1% slower.

Decision: keep the 18-thread CPU default and R1 Metal route at T=4. This is the
second consecutive pass without a retained gain.

Open questions: T=1 Metal launch overhead and CPU pool wakeup cost remain, but
both now require structural changes rather than route tuning.

Raw results: focused alternating command output only; no retained result set.

## 2026-07-19: Optimization Loop Pass 18 - CPU Native Target And Metal Crossover

Status: Metal route update landed; CPU experiment rejected. Significant-gain
streak reset; consecutive no-gain passes: 0.

Current implementation: portable CPU target flags; Metal R4 is the default for
the entire T=7..2048 range. The 16x16 GEMM remains available through
`EI_METAL_GEMM_MIN_TOKENS` for diagnostics.

References inspected:

- Apple clang predefined architecture features with and without `-mcpu=native`
- `llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` R4 GEMV
- `QuixiCore-Metal/perf/perf.md` crossover and thermal-order guidance

Correctness: CPU compiler variants produced identical checksums. Metal R4 and
GEMM checksums differ only by expected floating-point accumulation order;
route parity was already validated at cosine 1.0 and both pass golden parity.

CPU experiment: compile the full engine with `-mcpu=native`, exposing M5 I8MM,
BF16, and SME features. Native lost 3-6% on long shapes when run second and won
2-5% when run first; T=1 was similarly order-sensitive. Current intrinsics
already use dot-product instructions, and no native-only benefit survives
thermal reversal.

Metal experiment: re-sweep R4 after Pass 12 against 16x16 staged GEMM. Across
both orders, R4 improved T=64..256 by 20-30%, T=384..768 by roughly 14-23%,
and usually improved T=1024..2048. In the adverse hot-second ordering at
T=1536/2048, R4 was only 1-2% slower than the cool-first GEMM result; the
reverse ordering favored R4 by 10-16%. GEMM has no reproducible winning shape.

Decision: retain portable CPU flags. Set the default Metal GEMM threshold to
2049, outside the supported context, so R4 serves all production prefill
shapes. Keep staged kernels and route flag for future hardware/toolchain tests.

Open questions: now that direct R4 dominates, test modest function-level
unrolling/register pressure only; staged-memory work is deprioritized.

Raw results: focused alternate-binary CPU and same-binary Metal route output and
`perf/results/2026-07-19/035114-pass18-retained/`.

## 2026-07-19: Optimization Loop Pass 19 - CPU R2 And Metal R8

Status: experiments rejected. Consecutive no-significant-gain passes: 1 of 3.

Current implementation: CPU adjacent-row factor 3 and Metal output-row factor
4 remain unchanged.

References inspected:

- `llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h` (`N_R0_Q4_0=4`)
- retained Pass 13/15 CPU triple-dot measurements

Correctness: R2/R3 and R4/R8 variants produced identical checksums; both full
backends passed 10/10 golden parity at their retained cosine levels.

CPU experiment: replace adjacent triple accumulation with pairs using the dual
dot. Across three alternating T=1 pairs, R2 measured `2.97-3.04 ms` versus
`2.57-2.83 ms` for R3, a 7-18% regression.

Metal experiment: raise all direct Q4 kernels from four to eight output rows per
simdgroup. R8 regressed T=7 by roughly 8% and T=32 by 1-8%. T=128..2048 was
order-sensitive: gains were 1-4% when R8 ran second and larger only when the R4
comparison ran last/hot. No long-shape gain reproduced above threshold.

Decision: restore CPU R3 and Metal R4. This is the first no-gain pass after
Pass 18.

Open questions: projection factors are now bounded on both sides; focus any
remaining pass on non-projection overhead.

Raw results: focused alternate-binary/metallib output only; no retained set.

## 2026-07-19: Optimization Loop Pass 20 - CPU LTO And Metal Command References

Status: experiments rejected. Consecutive no-significant-gain passes: 2 of 3.

Current implementation: portable per-file CPU compilation and normal retained
Metal command buffers remain unchanged.

References inspected:

- `QuixiCore-Metal/perf/perf.md` alternating A/B and significance rules
- Metal command-buffer descriptor ownership contract

Correctness: CPU LTO produced the same checksums at every measured shape.
Metal retained and unretained command buffers also produced identical checksums.

CPU experiment: compile the complete CPU engine using Clang LTO (`-flto`)
and alternate it with the retained `-O2` binary at T=`1,32,128,512`, using five
warmups and 20 iterations. LTO was 8-10% slower at T=32..512 in the first pair.
The reverse pair became heat/order dominated, with neither binary winning
consistently. T=1 moved in both directions and had no stable gain.

Metal experiment: create command buffers with
`MTLCommandBufferDescriptor.retainedReferences=NO`, relying on the engine-owned
resources that outlive synchronous execution. Four alternating runs at
T=`1,32,128,512`, with eight warmups and 30 iterations, were neutral within
roughly 1% through T=128. T=512 had wider variance and unretained references
were not consistently faster.

Decision: reject both experiments. Cross-file optimization does not improve the
hot CPU kernels, and Objective-C resource retaining is not material relative to
GPU execution.

Open questions: one final pass should probe submission/encoder overhead on Metal
and a local CPU scheduling or code-generation detail without changing arithmetic.

Raw results: focused alternating command output only; no retained result set.

## 2026-07-19: Optimization Loop Pass 21 - CPU QoS And Metal Enqueue

Status: experiments rejected. Consecutive no-significant-gain passes: 3 of 3.
The optimization loop stop condition is satisfied.

Current implementation: CPU workers retain their inherited scheduler class;
Metal command buffers are encoded before normal commit.

References inspected:

- `QuixiCore-Metal/perf/perf.md` variance and stable-improvement requirements
- `llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m` explicit `enqueue`
  before asynchronous multi-command-buffer encoding

Correctness: all CPU QoS and Metal enqueue variants produced identical
checksums at each measured shape.

CPU experiment: opt worker threads into `QOS_CLASS_USER_INTERACTIVE`, then run
four alternating measurements at T=`1,32,128,512`, with five warmups and 20
iterations. T=1 was neutral in the first pair and order-sensitive in the
second. Sustained shapes regressed roughly 4-8% in the first pair, with no
reverse-order recovery beyond the common thermal decline.

Metal experiment: call `enqueue` immediately after command-buffer creation,
before encoding the graph. An initial alternating set suggested a noisy T=32
benefit, so a six-run focused set used 15 warmups and 60 iterations at
T=`1,32,128`. Differences remained under about 2%, changed sign, and T=128 was
effectively identical. llama.cpp's enqueue is useful for asynchronously encoded
multiple command buffers; this engine synchronously encodes one buffer.

Decision: reject both experiments and stop after three consecutive no-gain
passes (19-21). Retained performance work remains the CPU triple/adjacent-row
projection path and fused QK processing, plus Metal R4 direct projections,
fused QK processing, and precomputed RoPE.

Open questions: future work should begin with hardware counters or a new model
shape/quant format. Further local factor, route, compiler, and submission tuning
is below the current significance threshold.

Raw results: focused alternating command output only; no retained result set.

## Final Retained Results Versus Baseline

The durable Pass 0 engine result was recorded after Pass 1's CPU SIMD
accumulation had already landed. The controlled pre-Pass-1 CPU T=1 measurement
was `6.43 ms`; the retained Pass 15 result is `2.7015 ms`, or 2.38x throughput
across the complete loop.

For a consistent durable-run comparison, use Pass 0 versus Pass 15 for CPU and
Pass 0 versus Pass 18 for Metal:

| backend | tokens | durable baseline ms | retained ms | x throughput |
|---|---:|---:|---:|---:|
| CPU | 1 | 2.8115 | 2.7015 | 1.04x after Pass 1 |
| CPU | 4 | 8.1850 | 7.3460 | 1.11x |
| CPU | 8 | 13.9265 | 12.7595 | 1.09x |
| CPU | 32 | 33.1840 | 30.4570 | 1.09x |
| CPU | 128 | 89.0110 | 79.9670 | 1.11x |
| Metal | 1 | 2.7150 | 1.7550 | 1.55x |
| Metal | 4 | 5.5565 | 1.9500 | 2.85x |
| Metal | 8 | 6.0475 | 2.2270 | 2.72x |
| Metal | 32 | 7.4550 | 3.9955 | 1.87x |
| Metal | 128 | 18.7990 | 12.4720 | 1.51x |

The retained CPU numbers are from
`perf/results/2026-07-19/034324-pass15-retained/`; retained Metal numbers are
from `perf/results/2026-07-19/035114-pass18-retained/` except T=8, which uses
Pass 15 because Pass 18 did not repeat that shape. Pass 18 did not change CPU
code, but its CPU rows were thermally degraded and are not used for comparison.

### Current 2048-Token Measurement

A fresh full-context run used three warmups and 10 timed iterations:

| backend | tokens | median ms | p20/p80 ms | tokens/s |
|---|---:|---:|---:|---:|
| CPU | 2048 | 1516.559 | 1483.693/1554.584 | 1350.4 |
| Metal | 2048 | 195.699 | 194.886/197.354 | 10465.1 |

Metal delivers 7.75x CPU throughput for this 2048-token request. Raw results:
`perf/results/2026-07-19/083016-final-2k/`.

## 2026-07-19: Concurrent 2048-Token Baseline

Status: historical serialized baseline; superseded by the dynamic batching
results below.

Current implementation: `server.c` accepts and handles one connection at a
time. CPU and Metal engines also own mutable activation workspaces, so calling
one engine concurrently is unsafe. The benchmark uses concurrent arrival
threads and a ticket-ordered engine mutex to model the existing single-engine
FIFO queue, without HTTP parsing, model loading, or unsafe overlapping writes.

Method: synthetic 2048-token requests, one warmed engine, concurrency
T=`1,2,4,8,16,32`, at least four requests per level. Forward and reverse sweeps
separate thermal/order effects. Request p50/p95 includes time waiting for the
engine. Aggregate throughput uses wall time from synchronized release through
the final completion.

Isolated reverse-order results:

| backend | concurrency | req/s | input tok/s | p50/p95 ms |
|---|---:|---:|---:|---:|
| CPU | 1 | 0.605 | 1239.0 | 1647.0/1665.5 |
| CPU | 2 | 0.612 | 1254.0 | 3258.2/3266.8 |
| CPU | 4 | 0.607 | 1242.4 | 3305.2/6593.9 |
| CPU | 8 | 0.610 | 1249.1 | 6469.0/13116.7 |
| CPU | 16 | 0.627 | 1283.9 | 12748.5/25523.0 |
| CPU | 32 | 0.616 | 1261.4 | 26128.3/50362.5 |
| Metal | 1 | 4.586 | 9392.6 | 217.6/219.6 |
| Metal | 2 | 4.607 | 9434.8 | 433.6/434.7 |
| Metal | 4 | 4.558 | 9334.2 | 447.9/877.6 |
| Metal | 8 | 4.605 | 9431.5 | 865.3/1737.1 |
| Metal | 16 | 4.699 | 9623.9 | 1700.9/3404.8 |
| Metal | 32 | 4.983 | 10204.6 | 3151.4/6214.6 |

Decision: concurrency does not scale inference throughput today. CPU remains
within 4% of `0.61 req/s`; Metal remains in a roughly 9% sustained-clock band
around `4.6-5.0 req/s`. Tail latency grows approximately with queue depth.
Connection threads alone cannot remove this ceiling; the engine needs packed
request batching or multiple workspace slots sharing one immutable model.

Raw results:

- `perf/results/2026-07-19/090738-concurrency-2k-metal-fifo/`
- `perf/results/2026-07-19/090945-concurrency-2k-cpu-fifo/`

The earlier `084206`, `084404`, and `084613` runs are superseded: their plain
pthread mutex could serve waiting clients out of order and inflate low-level
p95 latency.

## 2026-07-19: vLLM/TEI Serving And Kernel Pass

Status: retained scheduler, cache, CPU, and packed-batch changes; three Metal
and CPU fusion experiments rejected.

References inspected:

- `.reference/vllm/vllm/v1/core/sched/scheduler.py`: waiting/running queues and
  token/sequence budgets.
- `.reference/vllm/vllm/config/scheduler.py` and `config/vllm.py`: hardware
  batch budgets and asynchronous scheduling disabled by default for pooling.
- `.reference/vllm/csrc/cpu/micro_gemm/`: reuse one packed weight load across
  several activation rows.
- `.reference/vllm/vllm/model_executor/layers/quantization/utils/fp8_utils.py`:
  gated activation plus output quantization epilogues.
- `.reference/text-embeddings-inference/core/src/queue.rs` and `infer.rs`: FIFO
  token-budget batches, bounded concurrency, flattened tokens, and one backend
  owner with per-request completions.

### Retained Engine Changes

CPU fused RMS norm plus Q8 activation quantization avoids a materialized
768/1152-float normalized row. Alternating measurements retained about 1.07x
throughput at T=1 and were neutral at T=32/128, with exact output checksums.

The CPU long-input route quantizes all activation rows, then applies a four-row
Q4 x Q8 micro-GEMM so packed weights stay hot across rows. Short and medium
inputs regressed because of extra thread-pool phases, so the production
threshold is T=512. Alternating long-shape measurements retained approximately
1.08x throughput at T=512 and 1.23x at T=2048. CPU/Metal mixed-batch cosine is
1.0; llama.cpp golden acceptance remains 0.999860704 or better.

CPU packed execution serializes the two-one-token corner, which changed its
measured throughput ratio from 0.75x to 1.01x. Larger short batches use the
flattened packed graph.

### Packed Batch Results

Packed versus repeated single-request execution on one warmed engine:

| backend | tokens/request | batch | x throughput |
|---|---:|---:|---:|
| CPU | 1 | 4/8/16/32 | 1.35x/2.18x/2.23x/3.07x |
| Metal | 1 | 4/8/16/32 | 3.55x/6.43x/10.12x/14.69x |
| CPU | 32 | 2/4/8/16/32 | 1.24x/1.49x/1.62x/1.76x/1.97x |
| Metal | 32 | 2/4/8/16/32 | 1.27x/1.51x/1.66x/1.70x/1.68x |
| CPU | 2048 | 2 | 0.984x |
| Metal | 2048 | 2 | 0.974x |

Decision: pack short requests and keep sequences over 512 tokens as FIFO
singletons. This follows vLLM's shape-dependent batch-budget principle; a
larger batch is not assumed to be faster.

### Serving Architecture

This pass added a bounded socket queue, 32 HTTP workers, and one
inference-owner thread. The owner forms FIFO batches bounded by 4096 total
tokens and 32 requests, waits at most 200 us for a partial microbatch, and
dispatches immediately when a budget is full or the FIFO head cannot combine.
Array inputs are submitted atomically so one HTTP request can form one backend
batch. The TEI deep dive below later raises the production admission and backend
request defaults to 64.

An exact token-ID LRU caches final 768-float embeddings. Pending entries provide
singleflight: eight concurrent duplicates execute one backend call and share
the completion. Prefix KV caching was rejected because bidirectional attention
changes every prefix token when a suffix is added; only exact whole-result
caching is valid for this model.

The production concurrency harness submits unique cache misses through this
service. Results include queue wait and observed backend batches:

| backend | tokens | concurrency 1 req/s | concurrency 32 req/s | x throughput | avg batch at 32 |
|---|---:|---:|---:|---:|---:|
| CPU | 1 | 294.9 | 1018.0 | 3.45x | 32.0 |
| Metal | 1 | 468.7 | 6133.5 | 13.09x | 25.6 |
| CPU | 32 | 28.08 | 59.52 | 2.12x | 32.0 |
| Metal | 32 | 216.24 | 391.27 | 1.81x | 21.3 |

Full-context batches remained size 1. CPU measured 0.764 req/s at concurrency 1
and 0.719 req/s at concurrency 4; Metal measured 4.753 req/s at concurrency 1
and 4.393 req/s at concurrency 8. The small decline is queue/thermal variance,
not packed execution. Tail latency still scales with queue depth for these long
singletons.

### Rejected Experiments

- Metal matching-row up/gate projection plus GELU removed one dispatch and an
  activation memory round trip. R1/R2 variants improved T=1 by about 2%, were
  roughly neutral at T=7..128, and had no stable long-shape gain. This missed
  the 3% retention threshold and is disabled by default.
- Metal matching-row Q/K/V projection reduced short-path simdgroup count from
  1280 to 768 but regressed T=1..6 by 2.5-4.7% from accumulator pressure. It is
  disabled by default.
- CPU GELU/multiply plus Q8 quantization produced exact checksums but was neutral
  through T=512 and 2.4% slower at T=2048 in in-process alternating A/B. It is
  disabled by default.

The rejected routes remain available through diagnostic environment controls
for future GPU/CPU generations. They are not part of the measured production
configuration.

## 2026-07-19: Text Embeddings Inference Deep Dive

Status: retained service scheduling and allocation changes; model warmup,
grouped CPU scheduling, and protocol changes rejected or deferred.

References inspected:

- `.reference/text-embeddings-inference/core/src/tokenization.rs`: bounded
  persistent tokenization workers and cancellation checks.
- `.reference/text-embeddings-inference/core/src/queue.rs`: FIFO token/request
  budgets, flattened nonpadded IDs, reserved vectors, and dropped-request
  filtering.
- `.reference/text-embeddings-inference/core/src/infer.rs`: one backend owner,
  one-batch prefetch channel, and result fanout outside the backend task.
- `.reference/text-embeddings-inference/router/src/main.rs` and `lib.rs`:
  separate concurrent-request, client-array, tokenizer, and backend budgets.
- `.reference/text-embeddings-inference/backends/src/lib.rs`: production-shape
  workspace warmup.
- `.reference/text-embeddings-inference/backends/candle/src/models/gemma3.rs`:
  concatenated QKV and gate/up projection schedules, fused residual RMS norm,
  and device-side pooling.

### Retained Changes

The service now owns a persistent tokenizer pool. Array inputs tokenize in
parallel while a single input avoids worker handoff. The isolated real-tokenizer
benchmark measured:

| payload | workers 0 | workers 8 | x throughput | workers 16 |
|---|---:|---:|---:|---:|
| 32 inputs x 256 chars | 1.022 ms | 0.239 ms | 4.28x | 0.206 ms |
| 32 inputs x 4096 chars | 20.427 ms | 3.164 ms | 6.46x | 1.854 ms |

The production default remains eight workers to leave cores for the CPU
backend; Metal-heavy deployments can raise `--tokenizer-workers`. Client arrays
are capped independently at 32 inputs, matching TEI's admission boundary.

The inference owner allocates flattened IDs and output storage once at startup
instead of for every batch. CPU and Metal workspaces are also reserved at the
configured maximum before accepting connections. Allocation-only reservation
reduced a cold Metal first request from 18.540 ms to 12.704 ms in the controlled
startup probe, without running a heat-producing maximum-shape forward pass.

Server defaults now admit 64 active HTTP requests, retain a 256-connection
backlog, and permit 64 backend requests per batch while keeping the client array
limit at 32. At concurrency 64, increasing only the backend cap from 32 to 64
measured:

| backend | tokens | cap 32 req/s | cap 64 req/s | result |
|---|---:|---:|---:|---:|
| CPU | 1 | 1064-1078 | 1219-1237 | 1.14-1.16x |
| CPU | 32 | 71.1-72.5 | 72.4-73.5 | 1.01-1.03x |
| Metal | 1 | 6880-7219 | 7783-8307 | 1.08-1.21x |
| Metal | 32 | 392.7-393.0 | 392.7-393.3 | neutral |

This preserves T=32 throughput while giving dispatch-dominated requests more
room to amortize the graph. Saturated CPU T=32 p50 latency rises because one
64-request batch completes together instead of two 32-request waves; the cap
remains configurable for latency-sensitive deployments.

### Rejected And Deferred Ideas

- TEI concatenates QKV and gate/up weights into one logical projection. The
  existing short CPU and all Metal routes already combine these dispatches. A
  grouped multirow CPU schedule was order-sensitive: its first cool T=64 pair
  regressed 3.2%, the reverse pair favored it by 2.1%, and the second pair was
  within 1%. T=512 followed thermal order. The separate schedules were restored.
- A real startup forward pass, first through the engine and then through the
  complete inference service, did not lower first HTTP latency: it remained
  14-18 ms versus 4.5-7 ms afterward. Only allocation reservation is retained.
- Formatting a cached 32-embedding, 335 KiB JSON response took 2.8-3.3 ms, so
  float serialization is not the cache-miss inference bottleneck. This pass did
  not change the default Ollama-compatible contract; the later Infinity pass
  added base64 as an explicit opt-in format.
- TEI drops canceled entries before batching. The current blocking HTTP worker
  cannot observe a disconnect while awaiting inference without an asynchronous
  connection-lifecycle redesign; this is deferred rather than approximated.
- TEI prebuilds one next batch while the backend runs. Here HTTP/tokenizer
  workers already enqueue subsequent work, and backend flattening is only a
  bounded memory copy. A second scheduler thread has no measured justification.
- Padded-token accounting is inapplicable: both production engines consume
  flattened nonpadded batches. Prefix KV caching remains invalid for this
  bidirectional embedding graph; exact final-result caching is retained.

## 2026-07-19: Matryoshka Output Dimensions

Status: landed protocol and response optimization; backend pool specialization
rejected.

Current implementation: CPU and Metal produce and cache one canonical
768-dimensional L2-normalized embedding. Requests may select 768, 512, 256, or
128 dimensions. The service copies the cache entry into caller-owned memory,
then the HTTP path L2-normalizes the selected prefix and serializes only that
prefix. Dimensions are not part of the exact token-ID cache key.

Reference inspected: `.reference/text-embeddings-inference/core/src/infer.rs`
truncates Matryoshka output to the requested dimension before normalization.
The equivalent operation here is prefix normalization of the already
full-normalized vector.

Correctness: native tests cover every supported dimension, invalid values,
unit norm, canonical 768 no-op behavior, and untouched suffixes. HTTP tests on
CPU and Metal cover default compatibility, normalized-prefix equivalence,
batch behavior, and HTTP 400 errors for invalid values. Maximum observed
serialized prefix error was below `7e-8`.

The native NEON prefix operation measured 0.000028/0.000039/0.000056 ms at
128/256/512 dimensions; 768 is a deliberate no-op. The complete CPU final
RMSNorm, mean-pool, and L2 stage measured 0.000242/0.003874/0.062379 ms at
1/32/512 tokens. This is well below the 3% end-to-end retention threshold.

Production HTTP results on Apple M5 Max:

| backend | dimensions | payload bytes | cached single ms | cached batch-32 ms | cache miss ms | concurrent req/s | x vs 768 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU | 768 | 10535 | 0.294 | 2.754 | 31.597 | 6408 | 1.00x |
| CPU | 512 | 6992 | 0.273 | 1.956 | 31.524 | 7359 | 1.15x |
| CPU | 256 | 3461 | 0.251 | 1.119 | 31.480 | 7208 | 1.12x |
| CPU | 128 | 1726 | 0.226 | 0.700 | 31.591 | 7277 | 1.14x |
| Metal | 768 | 10523 | 0.270 | 2.820 | 5.588 | 6179 | 1.00x |
| Metal | 512 | 6995 | 0.263 | 1.928 | 5.526 | 6993 | 1.13x |
| Metal | 256 | 3463 | 0.228 | 1.118 | 5.548 | 7290 | 1.18x |
| Metal | 128 | 1724 | 0.207 | 0.717 | 5.523 | 7176 | 1.16x |

Decision: retain post-cache prefix normalization and shorter JSON responses.
Do not add dimension-specific CPU or Metal pool kernels. Cache-miss inference is
flat across dimensions, Metal already pools in one fused simdgroup dispatch,
and producing less than 768 channels would prevent cross-dimension cache reuse.

Raw results:

- `perf/results/2026-07-19/145805-cpu-dimensions/`
- `perf/results/2026-07-19/145832-metal-dimensions/`

## 2026-07-19: Infinity Deep Dive

Status: retained optional packed-float responses and bounded fit-aware
scheduling; uniform workloads remain neutral.

Reference inspected at Infinity commit
`1eb4396b9c7711111224b31c09eda71fbce93075`:

- `.reference/infinity/libs/infinity_emb/infinity_emb/inference/queue.py` takes
  several batch windows, sorts by estimated length, and filters completed work.
- `.reference/infinity/libs/infinity_emb/infinity_emb/inference/batch_handler.py`
  pipelines preprocessing, device inference, and postprocessing through bounded
  queues.
- `.reference/infinity/libs/infinity_emb/infinity_emb/fastapi_schemas/pymodels.py`
  offers opt-in base64 float32 embeddings.
- `.reference/infinity/libs/infinity_emb/infinity_emb/inference/caching_layer.py`
  races an experimental persistent disk cache against inference.
- `.reference/infinity/libs/infinity_emb/infinity_emb/inference/select_model.py`
  warms short and 512-token shapes and can create multiple device replicas.
- `.reference/infinity/libs/infinity_emb/infinity_emb/transformer/utils_optimum.py`
  enables TensorRT CUDA graphs for small layers.

### Retained Changes

`encoding_format="base64"` now returns each normalized embedding as little-endian
float32 bytes inside one base64 JSON string. The default `float` response is
unchanged. At 768 dimensions on Apple M5 Max:

| backend | format | bytes | cached single ms | cached batch-32 ms | c32 req/s |
|---|---|---:|---:|---:|---:|
| CPU | float | 10535 | 0.269 | 2.770 | 6049 |
| CPU | base64 | 4115 | 0.122 | 0.258 | 6767 |
| Metal | float | 10523 | 0.268 | 2.777 | 6038 |
| Metal | base64 | 4115 | 0.131 | 0.274 | 6878 |

This is a 61% smaller payload, 2.05x-2.20x cached single-response throughput,
10.1x-10.7x cached batch-32 throughput, and 1.12x-1.14x c32 throughput.
Cache-miss latency was flat, as expected. CPU and Metal HTTP tests decode every
supported Matryoshka size, verify byte counts and unit norms, and reject unknown
formats.

The scheduler adapts Infinity's backlog sorting to this runtime's flattened,
nonpadded execution. It always dispatches the oldest request, then scans at most
`8 * max_batch_requests` queued entries for sequences that fit the request and
token budgets. This avoids sorting and starvation while allowing short work to
skip over an unpackable blocker.

Paired Metal runs at concurrency 64 measured:

| token pattern | scheduler | req/s | backend batches | p50 ms |
|---|---|---:|---:|---:|
| `1,513` | FIFO | 39.32-41.68 | 48-53 | 797-844 |
| `1,513` | lookahead | 41.18-41.31 | 33-34 | 5.6-486 |
| `1,1,1,513` | FIFO | 79.60-79.62 | 29-30 | 449-502 |
| `1,1,1,513` | lookahead | 81.93-82.47 | 17 | 6.7-6.9 |

The second mix gains 1.029x-1.036x throughput, removes 41%-43% of backend
dispatches, and prevents short requests from waiting behind long singletons.
The balanced mix consistently reduced dispatches but throughput followed test
order, so it is treated as latency/efficiency evidence rather than a stable
throughput gain. Uniform T=1 executes the identical single backend batch, and
uniform T=32 remained throughput-neutral. CPU mixed throughput followed
thermal/order noise while batch count and median latency improved; uniform CPU
runs showed no stable regression. Lookahead is enabled by default and strict
FIFO remains available with `EI_BATCH_LOOKAHEAD=0`.

Raw results:

- `perf/results/2026-07-19/151552-cpu-dimensions/`
- `perf/results/2026-07-19/151602-metal-dimensions/`
- `perf/results/2026-07-19/152746-infinity-fifo-a/`
- `perf/results/2026-07-19/152747-infinity-lookahead-a/`
- `perf/results/2026-07-19/152748-infinity-lookahead-b/`
- `perf/results/2026-07-19/152749-infinity-fifo-b/`

### Rejected And Deferred Ideas

- Infinity's three-stage pipeline is already represented by persistent HTTP and
  tokenizer workers feeding the single backend owner; backend flattening and
  result fanout are small, bounded operations.
- Length sorting is useful for padded transformer batches. This runtime does not
  pad, so only bounded fit-aware selection was retained.
- The experimental disk cache adds serialization and storage latency to a hot
  path already served by an exact token-ID memory cache and duplicate
  singleflight. It may be reconsidered only for restart-heavy deployments.
- Multiple replicas improve multi-device deployments. Infinity itself treats
  CPU replication as a debugging option; one CPU or Metal device should keep a
  single workspace owner.
- Production-shape warmup was already tested and rejected because real startup
  inference did not improve first HTTP latency. Allocation reservation remains.
- Quantized/binary response formats change retrieval quality and require a
  calibration and recall evaluation, so they are not transport optimizations.
- CUDA graph capture is a strong candidate for the future CUDA and ROCm ports.
  Revisit Infinity's TensorRT setting and llama.cpp's graph capture/update path
  in `ggml/src/ggml-cuda/ggml-cuda.cu` when those backends exist. There is no
  equivalent Metal graph mechanism in the inspected references.

## 2026-07-19: Persistent HTTP And Adaptive Collection

Status: retained for CPU and Metal.

HTTP/1.1 connections now remain persistent unless the client requests close.
Worker ownership is bounded to half the HTTP pool by default, each connection
has a 100-request limit and 1000 ms idle timeout, and pipelined bytes survive
between requests. This leaves workers available to accept new connections.
`EI_HTTP_KEEPALIVE=0` restores one response per connection.

Paired cached-response measurements at concurrency 32:

| backend | format | keep-alive off req/s | on req/s | x throughput |
|---|---|---:|---:|---:|
| CPU | float | 7916-7955 | 10264-10581 | 1.29-1.33x |
| CPU | base64 | 9026-9167 | 11576-13621 | 1.26-1.51x |
| Metal | float | 7508-7784 | 10615-10621 | 1.36-1.41x |
| Metal | base64 | 8210-9108 | 15161-15206 | 1.66-1.85x |

The scheduler now skips the fixed 200 us collection wait for an isolated
synchronous request, while retaining it when the queue overlaps or recently
batched. `EI_ADAPTIVE_BATCH_WAIT=0` restores fixed waiting. T=1 latency changed
from 2.04-3.09 ms to 1.75-1.77 ms on Metal and 2.49-3.12 ms to 2.21-2.26 ms on
CPU. Saturated and mixed workloads retained the same batch behavior and showed
no stable throughput regression.

Raw results include:

- `perf/results/2026-07-19/170225-keepalive-a-metal-off/`
- `perf/results/2026-07-19/170225-keepalive-a-metal-on/`
- `perf/results/2026-07-19/170235-keepalive-a-cpu-off/`
- `perf/results/2026-07-19/170236-keepalive-a-cpu-on/`

## 2026-07-19: Exact Float Response Cache

Status: retained with a 64 MiB default capacity.

The server now has a byte-bounded, mutex-protected LRU keyed by exact request
body bytes. It stores only successful `encoding_format="float"` JSON bodies.
Base64 responses bypass it because their existing formatting path is already
cheap, and failures are never cached. Entries use exact length and byte
comparison after a 64-bit hash; reference counts prevent eviction while a
worker sends a cached body. `--response-cache-mb 0` disables the cache.

With keep-alive enabled, a repeated float response measured:

| backend | operation | cache off | cache on | result |
|---|---|---:|---:|---:|
| CPU | cached single | 0.180-0.190 ms | 0.063-0.089 ms | 2.1-3.0x |
| CPU | cached batch 32 | 2.606-2.628 ms | 0.090-0.128 ms | 20-29x |
| CPU | concurrency 32 | 9712-9906 req/s | 12942-12956 req/s | 1.31-1.33x |
| Metal | cached single | 0.185-0.205 ms | 0.066-0.068 ms | 2.7-3.1x |
| Metal | cached batch 32 | 2.619-2.698 ms | 0.091 ms | 29-30x |
| Metal | concurrency 32 | 9202-9734 req/s | 12550-12622 req/s | 1.29-1.37x |

Unique misses remained flat: roughly 29.7-30.1 ms on CPU and 4.1-4.7 ms on
Metal. The canonical token-ID embedding cache remains separate and reusable
across dimensions and formats.

Raw results include:

- `perf/results/2026-07-19/180515-float-cache-final-a-metal-on-cache0/`
- `perf/results/2026-07-19/180516-float-cache-final-a-metal-on-cache64/`
- `perf/results/2026-07-19/180517-float-cache-final-a-cpu-on-cache0/`
- `perf/results/2026-07-19/180518-float-cache-final-a-cpu-on-cache64/`

## 2026-07-19: FP16 K/V And Projection Width Follow-Up

Status: retained FP16 K/V for long Metal sequences; wider CPU and Metal
projection groups rejected.

Metal attention now stores K/V as FP16 when the longest sequence in the batch
is at least 1024 tokens. Q remains FP32, and score, online softmax, and value
accumulation remain FP32. The retained path writes V as FP16 from the R4 QKV
projection and K as FP16 from fused norm/RoPE, avoiding a conversion dispatch.
`EI_METAL_FP16_KV_MIN_TOKENS=65536` restores FP32-only diagnostics.

Two-engine alternating A/B measurements controlled temperature within each
iteration:

| tokens | FP16 K/V ms | FP32 K/V ms | x throughput | cosine |
|---:|---:|---:|---:|---:|
| 1024 | 104.97-112.79 | 107.89-116.25 | 1.028-1.031x | 0.999999945 |
| 2048 | 232.53-240.14 | 254.55-257.85 | 1.074-1.095x | 0.999999970 |

The permanent parity test measures FP32/FP16 cosine 0.999999940 at both 1024
and 2048. Packed 2048-token A/B results were 1.071x at batch 1, 1.051x at batch
2, and 1.124x at batch 4. Forced FP16 was neutral for packed 32-token batches
from B=2 through B=32 but regressed a lone short request, so production routes
by maximum sequence length rather than flattened total tokens.

Rejected experiments:

- Metal R8 was about 8% slower at T=7 and neutral at medium lengths. R16 was
  20-29% slower at short lengths. Neither produced a stable long-shape gain, so
  only R4 is compiled into the production metallib.
- AArch64 CPU batch-8 weight reuse had identical checksums, but one T=2048 pair
  was 3% slower and its reverse was 9% faster. The midpoint was only about 3%
  and followed thermal order, so the retained CPU micro-GEMM remains batch 4.

Raw whole-run Metal results include:

- `perf/results/2026-07-19/181542-fp16-kv-a-f32/`
- `perf/results/2026-07-19/181546-fp16-kv-a-f16/`
- `perf/results/2026-07-19/181940-fp16-kv-direct-a-f32/`
- `perf/results/2026-07-19/181946-fp16-kv-direct-a-f16/`

The final engine and packed-batch comparisons used the alternating native A/B
harnesses; their JSON output is summarized above.
