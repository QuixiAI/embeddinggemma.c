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

Open questions: add accelerator-backed cases once ROCm/SYCL launch
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
  CPU replication as a debugging option; one CPU, Metal, or CUDA device should keep a
  single workspace owner.
- Production-shape warmup was already tested and rejected because real startup
  inference did not improve first HTTP latency. Allocation reservation remains.
- Quantized/binary response formats change retrieval quality and require a
  calibration and recall evaluation, so they are not transport optimizations.
- CUDA graph capture is now retained in the CUDA backend below. Revisit an
  equivalent graph/update mechanism for ROCm; Metal has no direct equivalent.

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

## 2026-07-19 to 2026-07-20: CUDA SM86 Backend

Status: retained and validated on `quixi-3090-02`, one RTX 3090 (SM86), CUDA
12.9, driver 580.65.06.

References inspected:

- `llama.cpp/ggml/src/ggml-cuda/dequantize.cuh`, `vecdotq.cuh`, and
  `fattn-wmma-f16.cu`: exact Q4_0 packing, integer dot products, and FP16
  attention tiling.
- `QuixiCore-CUDA/kernels/quant/qgemv.cu`, `qgemm.cu`, and `tm_qmm.cuh`:
  warp scheduling, zero-shuffle Q4 fragment loading, and
  `mma.sync.m16n8k16` accumulator layout.
- `QuixiCore-CUDA/perf/perf.md`: correctness-first alternating A/B runs,
  realistic whole-model shapes, and a 3% retention threshold.
- Metal and CPU routes in this repository: packed block-diagonal attention,
  fused Q/K norm plus RoPE, multirow activation reuse, and final pooling.
- `.reference/vllm`: repeated-shape CUDA graph replay and token-budget batching.

### Optimization Passes

1. The initial complete backend used direct packed-Q4 FP32 warp projections and
   online FP32 attention. It established the functional baseline shown below.
2. The T=1..4 path fused RMS norm with per-32-value Q8 quantization and used
   Q4 x Q8 `dp4a` projections. At T=4 this reduced 2.882 ms to 1.997 ms,
   1.443x throughput; T=1 remained neutral. `EI_CUDA_Q8_LATENCY=0` retains the
   FP32 path for A/B.
3. The tensor route combined Q/K/V into one 1280-row GEMM and up/gate into one
   2304-row GEMM. Residual addition, next RMS norm, and FP16 conversion were
   fused. T=7 reached 1.92 ms and T=32 2.73 ms before attention tuning.
4. FP16 online attention, an eight-query shared-tile Flash-style kernel, and
   dense tensor-core QK/PV plus custom FP32 softmax were implemented. Initial
   single-sequence sweeps put their noisy crossover between 40 and 80 tokens.
   Tensor attention reduced 512 from
   10.96 to 4.96 ms, 1024 from 21.05 to 9.64 ms, and 2048 from 51.56 to
   22.99 ms, all about 2.2x throughput. Online/tensor output cosine is at least
   `0.999997914` through 2048 tokens.
5. Native packed-Q4 MMA dequantizes GGML blocks directly into
   `mma.sync.m16n8k16` fragments for every model projection. It matches the
   expanded-FP16 route at minimum cosine `0.999983191`, but direct global
   fragment dequantization is 3.2x-4.3x slower on RTX 3090. It remains available
   with `EI_CUDA_NATIVE_Q4_GEMM=1` as a low-memory diagnostic.
6. Packed attention exposed a batch-dependent crossover because dense QK/PV
   launches per sequence. Production now classifies each sequence separately:
   tensor thresholds are 80 tokens at batch 1, 128 at batches 2..4, and 192 for
   larger batches; shorter sequences use Flash in the same forward pass. This
   retained the single-sequence tensor gains while improving representative
   packed shapes to 2,030 req/s at T64/B32, 878 at T96/B4, and 819 at T128/B16.
   T192/B32 remains tensor-routed and reaches 632 req/s. Setting
   `EI_CUDA_TENSOR_ATTENTION_MIN_TOKENS` disables this heuristic for exact A/B.

Production implementation:

- One CUDA copy of the GGUF data section, plus one load-time FP16 expansion of
  the 168 projection matrices for the retained cuBLAS path.
- Model-specialized RMS norm, residual/next norm, Q/K norm plus NEOX RoPE,
  full and symmetric-SWA GQA masks, and final norm/mean/L2 pooling.
- Q8/DP4A projections below five flattened tokens; combined FP16 tensor-core
  projection GEMMs with FP32 accumulation from five tokens upward. The combined
  QKV epilogue writes normalized and RoPE-transformed Q/K plus V directly to
  FP16 attention buffers.
- FP16 Q/K/V attention with FP32 softmax and output: per-sequence shared-tile
  Flash or dense tensor-core QK/PV selected by the batch-tuned thresholds.
- A 16-shape CUDA graph cache keyed by flattened token count, batch size, and
  packed offset hash. Workspace growth invalidates pointer-bearing graphs.

End-to-end improvement on the same GPU:

| tokens | initial direct-Q4 ms | retained ms | x throughput | retained tok/s |
|---:|---:|---:|---:|---:|
| 1 | 2.027 | 1.712 | 1.184x | 584 |
| 7 | 4.595 | 1.888 | 2.434x | 3,707 |
| 32 | 15.086 | 2.303 | 6.551x | 13,897 |
| 128 | 48.007 | 2.491 | 19.274x | 51,389 |
| 512 | 189.083 | 4.788 | 39.490x | 106,932 |
| 2048 | 762.588 | 19.931 | 38.261x | 102,755 |

For unique 32-token requests, the final dynamic service scales from 417 req/s
at concurrency 1 to 3,279 req/s at concurrency 32. Direct packed execution
reaches 4,740 req/s at batch 32 with minimum packed/serialized cosine
`0.999785721`. Full-context requests remain singleton batches; the final engine
rate is 50.2 req/s. The 512-token packing cutoff is retained to avoid
multiplying latency for full-context requests.

Correctness and safety:

- All 10 llama.cpp goldens pass; retained minimum cosine is `0.999910414`.
- CPU/CUDA synthetic drift passes at 1/7/32/128 tokens; minimum is
  `0.991125166` on arbitrary token IDs.
- Packed-vs-single route drift passes at minimum cosine `0.998983860`; this is
  the one-token Q8/DP4A route crossing to FP16 GEMM in a packed batch.
- Q8/DP4A versus FP32 latency-route cosine is at least `0.996344924`.
- Online/tensor attention and expanded/native-Q4 MMA have permanent parity
  coverage. Direct-FP16 QKV split/epilogue parity is exactly `1.000000000` at
  7, 128, and 2048 tokens. HTTP Matryoshka dimensions pass on CUDA.
- Compute Sanitizer memcheck reports zero errors on Q8, Flash, tensor-attention,
  direct-FP16 QKV, banded SWA, and native packed-Q4 MMA shapes.
- The specialized server was launched without `--backend cuda` and reported the
  CUDA backend, confirming binary-specific default selection.

Raw remote results:

- `/tmp/embeddinggemma.c-cuda-dev/perf/results/2026-07-19/234250-cuda-initial/`
- `/tmp/embeddinggemma.c-cuda-dev/perf/results/2026-07-19/234821-cuda-graphs/`
- `/tmp/embeddinggemma.c-cuda-dev/perf/results/2026-07-20/004704-cuda-kernels-final/`
- `/tmp/embeddinggemma.c-cuda-dev/perf/results/2026-07-20/004705-cuda-kernels-final-t32/`
- `/tmp/embeddinggemma.c-cuda-dev/perf/results/2026-07-20/004711-cuda-kernels-final-t2048/`

### Cross-Backend Passes

7. CUDA symmetric-SWA tensor attention now submits rectangular QK/PV GEMMs
   over 1024-query bands instead of materializing the full 2048 x 2048 score
   matrix. It retained output cosine `0.999999762` and reduced T=2048 from
   about 22.93 ms to 21.08-21.15 ms (`1.084x-1.088x` throughput). Banded T=1024
   regressed about 4.3%, so production starts the route at 1536 tokens.
8. Metal residual addition and the following RMS norm were fused into one
   dispatch, carrying normalized activations directly to the next projection.
   Alternating A/B runs measured `1.09x-1.12x` at T=1, about `1.09x` at T=7,
   `1.04x-1.05x` at T=32, and `1.02x-1.045x` at T=128. Longer shapes were
   neutral, so production limits this fusion to 128 tokens.
9. CUDA combined QKV projection now normalizes Q/K, applies RoPE, and writes
   Q/K/V directly to FP16 attention buffers without intermediate split and
   conversion kernels. Checksums remained identical. Reversed-order runs
   measured `1.07x-1.14x` at T=7..128, `1.03x` at T=512, and about `1.05x` at
   T=2048, so the route is retained. A second experiment wrote attention
   context directly to the FP16 projection input; it was only `1.006x` at
   short shapes and `1.013x-1.018x` at T=512..2048. It remains available with
   `EI_CUDA_DIRECT_FP16_CONTEXT=1` but is disabled by default because it did
   not meet the 3% retention threshold.
10. A CPU 3:1-GQA attention experiment computed all three query heads together
    and shared each K/V load. It passed kernel, golden, and packed-batch parity,
    but extra live accumulators and output streams increased pressure on the M5
    Max cores. The first paired run regressed T=7..2048 by roughly 8%-15%; the
    reversed thermal order still favored the existing head-at-a-time kernel.
    The experiment was removed.
11. A Metal multi-simdgroup attention experiment adapted QuixiCore-Metal's
    four-simdgroup K/V staging geometry to this model's D=256 head. Four query
    simdgroups shared 8-token FP16 K/V tiles with FP32 online softmax. Outputs
    had identical benchmark checksums, but the extra barriers, threadgroup
    traffic, and occupancy cost reduced throughput to about `0.91x` at T=1024
    and `0.86x` at T=2048 in the first pair; reversed order remained slower.
    The experiment was removed, confirming the existing cache-fed one-simdgroup
    kernel is better on M5 Max.
12. A CUDA T=2..4 DP4A experiment assigned one warp to each output row and
    accumulated four activation rows while reusing packed Q4 blocks. It covered
    QKV, attention output, up/gate, and down projections and passed the CUDA
    suite with identical benchmark checksums. After normalizing each run to the
    unchanged T=1 control, however, it was about 1%-2% slower at T=2..4. The
    apparent first-run T=2 gain tracked GPU clock state and disappeared in
    reversed order. The kernels were removed.

The last three independent optimization passes produced no significant gain,
so this loop stops here per the handbook criterion. The retained results from
this loop are CUDA banded SWA tensor attention, Metal short-shape residual/next
norm fusion, and CUDA direct-FP16 QKV epilogues.

## 2026-07-20: XPU SYCL Functional Baseline

Status: baseline recorded; optimization passes in progress.

Environment: one Intel Arc Pro B60 selected from a two-B60 host, oneAPI 2026.1,
Level Zero 1.14.37020, immediate command lists, mixed-FP16 oneMKL GEMM, and a
resident unrelated two-GPU vLLM workload. Absolute timings therefore include
some external contention; route decisions use reversed-order A/B runs.

Current implementation: persistent device model/workspace allocations, native
Q8 embedding decode, subgroup norms and online attention, direct packed-Q4
projection diagnostics, load-time FP16 expansion of all projection matrices,
combined QKV and up/gate oneMKL GEMMs, fused QKV norm/RoPE/V epilogue, fused
residual/next norm, and dense oneMKL QK/PV attention from 128 tokens.

References inspected:

- `QuixiCore-XPU/kernels/quantization/gguf_gemv`
- `QuixiCore-XPU/kernels/norms/rms_norm`
- `QuixiCore-XPU/kernels/attention/attention`
- `vllm-xpu-kernels/csrc/xpu/sycl/decode`
- `vllm-xpu-kernels/csrc/xpu/attn/xe_2`
- `vllm-xpu-kernels/csrc/xpu/grouped_gemm`

Correctness: all 10 llama.cpp goldens pass at minimum cosine `0.999910653`.
CPU/XPU synthetic drift is at least `0.996121168`; direct packed-Q4 versus
expanded-FP16 projection is at least `0.999994934`; online versus tensor
attention is at least `0.999997437`; packed versus serialized execution is at
least `0.999954283`; HTTP Matryoshka dimensions pass.

Retained whole-engine baseline, five warmups and 20 iterations:

| tokens | median ms | tokens/s |
|---:|---:|---:|
| 1 | 4.6694 | 214.2 |
| 7 | 4.7763 | 1,465.6 |
| 32 | 7.6252 | 4,196.6 |
| 128 | 9.9928 | 12,809.3 |
| 512 | 20.3124 | 25,206.3 |
| 2048 | 67.5650 | 30,311.5 |

Dynamic unique T32 requests scale from `135.067 req/s` at concurrency 1 to
`880.631 req/s` at concurrency 32 (`6.52x`, 28,180 input tokens/s, average
batch 32). A forced collection delay was rejected: it reduced concurrency-32
throughput while increasing single-request latency.

Attention crossover experiments show that a scalar threshold is insufficient.
Tensor attention wins for one sequence from about T80, but per-sequence GEMM
submissions make it 1.48x slower at T128/B4 and 2.50x slower at T128/B16.
Tensor wins again at T256/B4 and T512/B2. This motivates grouped GEMM and a
batch-aware route before final threshold tuning.

Rejected experiment: SYCL command-graph capture including oneMKL was neutral at
T1 and 15-25% slower from T7 through T2048 than immediate in-order submission,
so it was removed.

Raw results:

- Remote `perf/results/2026-07-20/014007-xpu-retained/`
- Remote `perf/results/2026-07-20/014033-xpu-t32-retained/`
- Focused reversed-order attention and packed-route command output.

### XPU Pass 1: Event Profiling

Status: retained as an opt-in diagnostic; no measurable disabled-path cost.

`EI_XPU_PROFILE=1` now creates a profiling-enabled Level Zero queue and reports
per-label SYCL event totals for copies, kernels, and oneMKL calls. With profiling
disabled, a five-warmup/20-iteration check measured 4.645 ms at T1, 7.355 ms at
T32, 18.751 ms at T512, and 69.378 ms at T2048, within the variance of the
functional baseline.

The device-event breakdown changed the optimization order. At T1, the 47 fused
residual/next-normalization calls total about 1.31 ms. At T512, dense QK,
softmax, and PV total about 3.00 ms while pooling plus its result transfer total
about 1.02 ms. At T2048, QK/softmax/PV total about 35.9 ms, pooling plus result
transfer about 6.64 ms, and the four projection classes about 6.88 ms. Event
profiling changes queue behavior and clock state, so these values rank work but
are not used as end-to-end speedup claims.

### XPU Pass 2: Grouped oneMKL Dense Attention

Status: rejected.

The dense route was changed from six oneMKL calls per sequence and layer to one
grouped QK call, one batch-wide softmax dispatch, and one grouped PV call. Score
matrices were packed by sequence, and all grouped descriptor and pointer arrays
were kept alive until the queue completed to satisfy asynchronous lifetime
requirements. The implementation compiled, but a forced dense T128 run did not
complete a single measured iteration within 60 seconds; the retained path is
about 10 ms end to end. The same timeout occurred before and after correcting
descriptor lifetime. oneMKL 2026.1 grouped GEMM is therefore unusable for this
small heterogeneous three-head geometry on the tested Level Zero stack. The
experiment was removed; a custom fused attention kernel remains the viable way
to eliminate these submissions.

### XPU Pass 3: FP16 Q/K/V Attention Storage

Status: retained for dense-attention sequences; rejected for unconditional
online attention.

The combined QKV epilogue can now write Q/K/V directly to FP16. Online
attention converts loads to FP32 and keeps FP32 softmax/output accumulators;
dense attention uses mixed FP16 QK with FP32 scores, writes normalized
probabilities to FP16, and uses mixed FP16 PV with FP32 output. Forced-route
cosine versus the FP32 path is at least `0.999989022` over T1..T2048.

Alternating single-request A/B measured `1.0566x` at T128, `1.2771x` at T512,
and `1.4645x` at T2048. Packed dense T128 measured `1.0796x`, `1.0592x`, and
`1.1008x` at batches 4, 8, and 16. Unconditionally storing FP16 for short
online-attention batches was noisy and often slower (as low as `0.739x` in a
contended T32/B4 run), so production uses FP16 only when every sequence in the
batch takes the dense route. `EI_XPU_FP16_ATTENTION=0|1|auto` preserves exact
off, forced-on, and production A/B modes.

### XPU Pass 4: Batch-Aware Attention Routing

Status: retained.

The scalar T128 dense-attention threshold was replaced by batch-aware defaults:
T80 for one sequence, T192 for batches 2..4, T384 for batches 5..8, and T512
for larger batches. Setting `EI_XPU_TENSOR_ATTENTION_MIN_TOKENS` still forces an
exact scalar threshold for diagnostics. Paired in-process A/B against the old
T128 rule measured `1.204x`, `1.318x`, and `1.523x` at T128/B4, B8, and B16.
At T256 it measured `1.256x` at B8 and `1.461x` at B16. T192/B4 moved slightly
in favor of dense attention after FP16 Q/K/V landed, which set the final B2..4
crossover to T192. Route cosine remained at least `0.999981582` for the retained
benchmark shapes.

### XPU Pass 5: Banded Symmetric-SWA GEMMs

Status: implemented as a diagnostic and disabled by default.

Long sliding-window layers can partition queries into compact bands, compute
only the corresponding key range, apply exact edge masks, and feed FP16
probabilities to PV. A 1024-query tile preserved cosine `0.999999678` or better,
but measured `0.959x` at T1024, `0.955x` at T1536, and only `1.019x` at T2048.
Tiles of 512 and 256 were slower (`0.925x` and `0.660x` at T2048) because extra
oneMKL submissions outweighed reduced FLOPs. Production leaves the tile at zero;
`EI_XPU_SWA_TENSOR_TILE_TOKENS` and `EI_XPU_SWA_TENSOR_MIN_TOKENS` retain the
complete experiment for future oneMKL/driver retesting.

### XPU Pass 6: Single-Token V-Only Attention

Status: retained.

For a one-token sequence, softmax attention is exactly its value vector for
every query head. Batches containing only one-token sequences now project only
the 256-row V slice of the combined QKV weights, replicate V directly into the
FP16 attention-output input, and skip Q/K projection, Q/K norm, RoPE, and
attention. Alternating packed A/B measured `1.820x`, `1.805x`, `1.824x`,
`1.788x`, `1.789x`, and `1.800x` at batches 1, 2, 4, 8, 16, and 32. Minimum
cosine was `0.999988914`. `EI_XPU_SINGLE_TOKEN_V_ONLY=0|1` controls the route.

### XPU Pass 7: Cooperative RMS and Pooling

Status: retained with shape routing.

Rows up to four tokens now assign one 128-thread workgroup to each RMS row,
including the fused residual/next-normalization kernel, instead of launching
eight 16-thread subgroups with most inactive. This measured `1.612x` at T1,
`1.234x` at T2, and `1.208x` at T4, then became neutral by T7. Cosine remained
at least `0.999994620` over the routed shapes.

Final norm/mean/L2 pooling now has a cooperative 128-thread implementation with
six accumulators per lane instead of 48. It measured `1.030x` at T128, `1.048x`
at T256, `1.022x` at T512, and `1.060x` at T2048, while T32 regressed `1.4%`.
Production therefore selects it for one-token or at-least-128-token sequences.
For packed mixed-length batches, auto mode uses the cooperative kernel only
when every sequence qualifies; this keeps a short request's embedding
independent of a long neighbor's pooling route.
`EI_XPU_COOPERATIVE_RMS_MAX_ROWS=0..128` and
`EI_XPU_COOPERATIVE_POOL=0|1|auto` preserve exact experiment controls.

### XPU Pass 8: Xe2 XMX Flash Attention

Status: retained for single sequences when built with `XPU_XE2_FLASH=1`;
multi-sequence varlen dispatch remains forced-only for diagnostics.

The Xe2 chunk-prefill kernel from `vllm-xpu-kernels` is instantiated only for
EmbeddingGemma's FP16, head-256, three-query/one-KV-head geometry. The optional
build now carries the same split-barrier, 2D-block-I/O, subgroup-matrix, BMG
device-link, and 256-GRF settings as the reference build. Q/K/V are consumed
directly from the fused FP16 QKV epilogue and the FP16 context is passed
directly to the output projection, removing score-matrix storage and six
oneMKL attention calls per layer.

The initial asynchronous prototype measured `1.079x`, `1.105x`, `1.044x`, `1.233x`,
`1.190x`, `1.337x`, and `1.550x` at T32, T64, T128, T256, T512, T1024, and
T2048. Minimum single-sequence cosine was `0.999975628`. Absolute timing was
heavily affected by a resident two-GPU workload, so only paired ratios are
used. These values were superseded by the fenced production measurements in
the final-validation section below.

The imported multi-sequence varlen scheduler was faster through B16 in initial
paired runs, but repeated identical T32/B16 runs produced nondeterministic
minimum cosine from `0.998176754` to `0.999876857`. An explicit queue-completion
fence did not eliminate the drift and imposed severe latency. Production auto
mode is therefore restricted to batch size one; `EI_XPU_XE2_FLASH=1` can still
force multi-sequence experiments, while `0` disables the route exactly.
The imported launcher does not return an event that oneMKL can consume, so the
retained path completes each Flash dispatch before submitting its output
projection. Without that fence, a T129 Flash request changed later packed
results. Automatic routing begins at T256: a final mixed serialized/packed
regression
showed that selecting Flash only for serialized T32 requests reduced route
consistency to cosine `0.998735249`, while the T129 route remained above
`0.999997`. T32-T255 results remain available through forced mode but are not
used in production.

### XPU Pass 9: Fixed-Shape oneDNN XMX Projections

Status: rejected; optional experiment retained behind `XPU_ONEDNN=1` and
`EI_XPU_ONEDNN_F16=1`.

The four projection geometries now have cached oneDNN matmul primitives using
FP16 inputs/weights, FP32 outputs, and a transposed-stride view of the existing
expanded weights. This tests oneDNN's fixed-shape XMX dispatch against the
retained oneMKL row-major GEMMs without changing model precision. Alternating
A/B measured `0.960x`, `0.986x`, `0.986x`, `1.002x`, `1.013x`, `1.033x`,
`1.007x`, and `1.003x` at T1, T7, T32, T64, T128, T256, T512, and T2048.
Cosine was at least `0.999983099`. The isolated 3.3% T256 result was not broad
or large enough to justify another runtime and primitive cache in production.

### XPU Pass 10: oneDNN S4 Weight-Only Projections

Status: implemented as an optional diagnostic; disabled by default.

GGUF Q4_0 projection weights can now be repacked once into signed S4 `[K,N]`
storage with FP16 `[K/32,N]` block scales. Cached oneDNN weight-only matmuls
consume FP16 activations and produce FP32 outputs, reducing projection weight
storage by 4x versus expanded FP16 while preserving cosine `0.999984499` or
better over the measured shapes. A short seven-iteration sweep appeared to
gain `1.145x` at T32 and `1.076x` at T128, but a four-warmup/21-iteration
repeat measured only `0.928x` at T32, `0.921x` at T48, `0.942x` at T64, and
`0.958x` at T128. The initial result was GPU clock/contention variance. The
portable binary therefore does not link oneDNN or allocate repacked weights;
`XPU_ONEDNN=1 EI_XPU_ONEDNN_W4=1` preserves the complete experiment.

### XPU Pass 11: M-Tiled Native Q4 Projection

Status: implemented as a diagnostic and disabled by default.

For M2..M8, one subgroup now owns an output row, decodes each Q4_0 block once,
and accumulates all request rows in registers. This removes redundant packed
weight reads from the original token-major direct kernel. Alternating A/B
against that kernel measured `1.134x`, `1.102x`, `1.152x`, and `1.151x` at M2,
M4, M7, and M8 with cosine exactly `1.0`; M1 regressed to `0.934x`. Whole-engine
latency remained 23.8-38.5 ms versus roughly 3.7-5.0 ms for the retained FP16
oneMKL route, so `EI_XPU_Q4_M_TILED=1` is available only when direct Q4 is
forced with `EI_XPU_GEMM_MIN_TOKENS`.

### XPU Final Validation and Loop Stop

The portable build passed the XPU regression suite with CPU/XPU cosine at least
`0.996161699`, packed-Q4/FP16-GEMM cosine at least `0.999996305`, online/dense
attention cosine at least `0.999998212`. Repeated packed-versus-serialized XPU
runs varied with oneMKL's M-dependent FP16 reduction path but remained at least
`0.999484062`; the enforced XPU gate is `0.999`, stricter than CUDA's `0.998`.
The Xe2-enabled build passed the same suite at minima `0.996134818`, `0.999995351`,
and `0.999998212` respectively. The complete local CPU suite also passed,
including all ten llama.cpp goldens (minimum cosine `0.999860704`), kernel
tests, packed batches, service/cache tests, and HTTP Matryoshka dimensions.
The remote XPU host lacks the libcurl development headers. Removing the linked
libcurl dependency allowed the complete XPU HTTP executable to build there; its
HTTP Matryoshka dimensions suite passed. Model cache misses now invoke `curl`
or `wget` through POSIX `posix_spawnp`, while an existing model requires neither
command.

A final fenced three-warmup/11-iteration Xe2 A/B repeat measured `0.974x`,
`1.091x`, `1.183x`, `1.304x`, and `1.535x` at T128, T256, T512, T1024, and
T2048, with cosine at least `0.999997071`. T128 is below the production
crossover; the retained T256-and-up shapes have cosine at least `0.999997999`.

The final dynamic T32 snapshot under the resident two-GPU vLLM workload was:

| concurrency | requests/s | input tokens/s | average batch |
|---:|---:|---:|---:|
| 1 | 58.0 | 1,856 | 1.0 |
| 2 | 28.4 | 908 | 1.0 |
| 4 | 62.4 | 1,997 | 3.4 |
| 8 | 93.1 | 2,979 | 7.1 |
| 16 | 124.3 | 3,976 | 12.8 |
| 32 | 158.2 | 5,062 | 21.3 |

These absolute values are a final-state smoke measurement, not a baseline
comparison, because the unrelated workload changed GPU occupancy and clocks.
All improvement claims above use alternating in-process A/B measurements.

Passes 9, 10, and 11 produced no significant retained end-to-end gain after
the Xe2 Flash win in Pass 8. The optimization loop therefore stops after three
consecutive non-significant production passes, as required by the experiment
protocol.

## XPU Perfection Loop (2026-07-20)

### XPU Pass 12: Event-Ordered Xe2 Flash

Status: retained.

The imported CUTLASS-SYCL launcher returned no event even though its underlying
`compat::experimental::launch` call does. A model-specific launcher now returns
that event, and an explicit device-side queue barrier orders the following
oneMKL projection. This removes the per-layer `wait_and_throw` host fence and
also makes Flash device time visible to the profiler. The old fence remains
available as `EI_XPU_XE2_FLASH_HOST_FENCE=1` for A/B diagnostics.

After a contaminated first sweep was discarded, a five-warmup/21-iteration
alternating in-process repeat measured `1.034x`, `1.022x`, `1.010x`, and
`1.005x` at T256, T512, T1024, and T2048. Cosine versus the fenced path was at
least `0.999997956`. The gain is concentrated at the production Flash
crossover, but every retained shape was non-regressing. A profiled warm T512
run reported 24 Flash calls totaling about `1.48 ms` and an end-to-end forward
time of about `5.93 ms`.

### XPU Pass 13: Exact-Shape SYCL Graph Replay

Status: retained for all-single-token batches B1-B4; forced mode remains
available for diagnostics.

The B60 and its Level Zero driver expose `ext_oneapi_graph`. The engine now
captures the complete fixed-shape forward submission, including oneMKL and
CUTLASS kernels, into a bounded eight-entry LRU. IDs and sequence metadata are
copied before replay, so graph entries depend only on sequence lengths and do
not cache request data. Workspace growth clears captured graphs before device
pointers move.

Five-warmup/21-iteration alternating packed A/B at one token measured
`1.246x`, `1.165x`, `1.110x`, `1.016x`, `1.017x`, and `1.020x` for B1, B2, B4,
B8, B16, and B32. Minimum cosine was `0.999989212`. Single-request T7-T2048
and packed T32 B1-B32 improved only 1.0%-2.6%. Production auto mode therefore
captures only all-single-token batches through B4; `EI_XPU_COMMAND_GRAPH=1`
forces any exact shape and `0` disables capture. Unsupported devices disable
auto mode rather than failing initialization.

### XPU Pass 14: Xe2 W4A16 DPAS Projections

Status: retained only for the two-token up/gate projection; the general route
is rejected.

GGUF Q4_0 weights can now be repacked into the signed-int4 row-major layout
consumed by `vllm-xpu-kernels`' BMG DPAS policies. An isolated matrix suite
validates QKV, value, attention-output, fused up/gate, and down geometries at
M1, M7, and M32 with cosine at least `0.999999975`. The loader waits for each
host-to-device copy before releasing its temporary packing buffers.

Using W4A16 for all four per-layer projections was slower than FP16 oneMKL:
five-warmup/21-iteration A/B measured `0.782x` to `0.877x` from T1 through
T128. At T32, W4 projection device time was about `197 us` higher and 96
FP16-to-FP32 conversion launches added another `127 us`. Fused up/gate was the
one favorable geometry, so its FP16 output now feeds GELU directly. A
seven-warmup/31-iteration repeat measured `1.118x` for a single T2 sequence
with cosine `0.999998120`. Command-graph replay made T1 and packed one-token
B1-B2 effectively neutral (`1.006x` and `1.011x`), so production routes only
the exact single-sequence T2 shape.

### XPU Pass 15: Register-Cached Residual/RMS

Status: retained on Xe2 for M2 and larger.

The fused residual/update/next-RMS kernel now keeps each lane's updated
residual values in registers across the second reduction, eliminating the
final global residual read. The subgroup path uses 48 values per lane under
the Xe2 build's 256-GRF setting; the cooperative M2-M4 path needs six. M1 keeps
the prior cooperative kernel because register caching measured `0.989x` there.

Five-warmup/21-iteration paired A/B measured `1.167x`, `1.164x`, `1.136x`,
`1.117x`, `1.055x`, `1.086x`, `1.056x`, `1.081x`, and `1.079x` at T7, T16,
T32, T64, T128, T256, T512, T1024, and T2048. Packed T32 measured `1.140x`,
`1.139x`, `1.122x`, `1.081x`, `1.064x`, and `1.074x` at B1, B2, B4, B8,
B16, and B32. Candidate/baseline cosine was at least `0.999979124` for single
requests and `0.999384940` for packed requests. Caching projection values too
was rejected: it measured only `1.000x-1.007x` on most shapes and `0.962x` at
T128 because the additional 48 registers per lane offset the saved read.

### XPU Pass 16: Stable Packed Xe2 Flash Routing

Status: retained for packed batches whose shortest sequence is at least 128
tokens.

After the Flash launcher began returning an event, the varlen route was
retested with true forced-mode semantics. The old A/B harness had still applied
the 256-token auto threshold when `EI_XPU_XE2_FLASH=1`; forced mode now bypasses
shape thresholds, while auto mode evaluates the shortest sequence in a batch.

Actual Flash at T32 and T64 was faster but remained below the packed parity
gate at larger batches, reaching repeated minima `0.999310672` and
`0.999653101`. T128 was stable across three repeated sweeps and measured
`1.84x-2.75x` at B2-B32. T256 measured `1.22x`, `1.38x`, `3.42x`, `4.39x`, and
`4.37x` at B2, B4, B8, B16, and B32; T512 measured `1.28x-1.34x` at B2-B8.
A 20-repeat mixed `[128,129,192,255]` regression had minimum cosine
`0.999993384`, and `[256,257,384,511]` had minimum `0.999979794`. Auto mode
therefore keeps the T256 single-sequence crossover and uses a conservative
T128 minimum for packed batches.

### XPU Pass 17: Runtime Gating and Reproducible Xe2 Build

Status: retained.

The Xe2 binary now queries oneAPI's architecture ID and enables BMG-only
Flash, W4A16, and 256-GRF residual/RMS defaults only on
`intel_gpu_bmg_g21`/`intel_gpu_bmg_g31`. Forced Flash or W4 on another
architecture fails initialization instead of dispatching incompatible device
code. Portable SYCL builds keep these routes disabled.

`make xpu XPU_XE2_FLASH=1` now fetches public, pinned dependencies into
`.xpu-deps`: `vllm-xpu-kernels` commit
`bab46865358da4eda3b866c41dd71a80e878d843` and SYCL-TLA commit
`cd763790ad2f74d7294435ecf77682bac0062c3a`. Explicit checkout overrides remain
available. The model-specific Flash translation unit now includes the CUTLASS
scheduler and kernel directly, avoiding unrelated Torch dtype/paged-cache host
helpers and removing the PyTorch-header build dependency. A clean managed
checkout compiled both Flash and W4 and passed the complete Xe2 suite.

### XPU Final Validation

The local CPU suite passed all kernel, tokenizer, service/cache, packed-batch,
HTTP-dimension, and llama.cpp golden tests; golden minimum cosine was
`0.999860704`. The Metal suite passed its 29 ready pipelines, goldens at
minimum cosine `0.999910295`, packed parity `1.0`, and backend/KV/tile checks.

The portable XPU build passed goldens at minimum cosine `0.999909997`,
CPU/XPU drift at `0.996052265`, projection parity at `0.999995351`, attention
parity at `0.999996066`, packed parity at `0.999976337`, and HTTP dimensions.
The pinned Xe2 build passed goldens at `0.999910116`, CPU/XPU drift at
`0.996102154`, the default T2 W4 route at `0.998495936` versus CPU, projection
parity at `0.999995410`, attention parity at `0.999997139`, mixed Flash at
`0.999994040`, boundary Flash at `0.999993622`, packed parity at `0.999941945`,
HTTP dimensions, and every isolated W4 geometry at `0.999999975` or better.

Final default B60 medians were 1.04 ms at T1, 1.23 ms at T2, 2.17 ms at T32,
3.57 ms at T256, 5.64 ms at T512, 9.82 ms at T1024, and 20.54 ms at T2048.
Packed B32 reached about 19,850 requests/s at T1, 109,170 tokens/s at T32, and
135,860 tokens/s at T128. These are absolute final-state measurements; the
per-pass claims above use alternating in-process A/B.

## ROCm CDNA Backend and Optimization Loop (2026-07-20)

Status: retained and validated on `enc1-svr04`, one AMD Instinct MI300X
(`gfx942`), ROCm 7.2.4. Runtime testing used one MI300X, but the production
binary is not `gfx942`-specific: one `hipcc` link embeds `gfx908`, `gfx90a`,
`gfx942`, and `gfx950` code objects for CDNA1 through CDNA4. `roc-obj-ls`
confirmed all four images in `build/embeddinggemma-rocm`. `ROCM_ARCHS` and the
legacy `ROCM_ARCH` remain developer-only size/profiling overrides.

The implementation was read against llama.cpp and
`QuixiCore/QuixiCore-ROCm`. The retained design follows QuixiCore's CDNA
principles: wave64-aware layout, logical width-32 reductions where the model
shape requires them, `__builtin_amdgcn_sdot4` packed dot products, native
`__builtin_amdgcn_mfma_f32_16x16x16f16` tiles, resident weights/workspaces,
fused epilogues, correctness-gated alternating A/B runs, and measured routing
instead of architecture-name assumptions.

### Functional Baseline

The first complete backend kept the GGUF data resident, expanded projection
weights to FP16 once for hipBLAS, and provided direct packed-Q4, Q8/SDOT4,
native Q4/MFMA, online/Flash-style attention, tensor attention, pooling, batch,
and HTTP paths. The ROCm binary selects ROCm by construction; `--backend rocm`
is unnecessary outside diagnostic harnesses.

Initial warmed medians were 2.234 ms at T1, 6.195 ms at T7, 6.688 ms at T32,
10.146 ms at T128, 9.404 ms at T512, and 18.775 ms at T2048. Initial dynamic
T32 throughput at C1/C2/C4/C8/C16/C32 was 148/150/422/612/1,677/2,495 req/s.

### ROCm Pass 1: HIP Command Graphs

Status: rejected by default; retained as `EI_ROCM_COMMAND_GRAPH=1`.

The first A/B measured `0.963x` at T1 and neutral results at larger shapes.
After all later kernel changes, a 101-iteration repeat measured `1.021x` at T1,
`1.001x` at T2, `0.940x` at T4, `1.000x` at T8, and `0.978x` at T16. The gain
was neither stable nor shape-safe.

### ROCm Pass 2: Direct FP16 Context

Status: retained as `EI_ROCM_DIRECT_FP16_CONTEXT=1`.

Writing attention output directly to the next projection's FP16 input removed
one context conversion and measured `1.176x` at T128, `1.186x` at T512, and
`1.433x` at T2048.

### ROCm Pass 3: Attention Routing

Status: retained, then superseded by the batched-head retune below.

The original tensor route lost below 256 tokens, then measured `1.039x` at T256
and `1.184x` at T512. The initial threshold moved from the inherited 80 tokens
to 256. The QuixiCore BQ16 MFMA attention adaptation passed tails at
16/33/128/191 tokens but measured only `0.98x` at T16, `0.96x` at T32,
`0.94x` at T64/T128, and `0.914x` at T255, so
`EI_ROCM_MFMA_ATTENTION` remains off.

### ROCm Pass 4: Short-Shape Q8/SDOT4

Status: rejected; retained under `EI_ROCM_Q8_LATENCY` and
`EI_ROCM_Q8_TWO_ROW`.

Runtime Q8 activation quantization plus SDOT4 measured `0.821x` versus direct
FP32 packed-Q4 at T1/T2. A two-output-row QuixiCore-style kernel was exact and
up to `1.034x` faster than the one-row Q8 kernel, but did not recover the Q8
route's end-to-end loss.

### ROCm Pass 5: Native Packed-Q4 MFMA

Status: retained for 32-368 flattened tokens.

The first native Q4 MFMA projection beat expanded-FP16 hipBLAS by `1.177x` at
T128 but was neutral by T256 before launch fusion. Combining Q/K/V into one
launch and up/gate into one launch improved the native route by `1.215x` at
T128, `1.182x` at T160, `1.142x` at T192, and `1.107x` at T240. The final
lower-bound retune measured native MFMA at `1.146x` at T48, `1.261x` at T64,
`1.368x` at T80, `1.679x` at T96, and `1.867x` at T120 versus the direct packed
path; it still lost below 32. The final upper-bound comparison against hipBLAS
was `1.095x` at T320, `1.067x` at T352, `1.046x` at T368, `1.027x` at T384,
and `0.971x` at T416.

A wider-N QuixiCore MFMA variant reached `1.083x` versus the one-output-tile
native kernel at T768, but native Q4 remained slower than hipBLAS at those
large M values. `EI_ROCM_NATIVE_Q4_WIDE` is therefore diagnostic-only.

### ROCm Pass 6: Register-Cached Residual/RMS

Status: retained as `EI_ROCM_RMS_REGISTER_CACHE=1`.

Each logical lane keeps 24 residual values in registers across the next RMS
reduction. Alternating A/B measured `1.074x` at T128, `1.038x` at T192,
`1.053x` at T256, `1.048x` at T352, `1.037x` at T512, `1.034x` at T1024, and
`1.018x` at T2048, with cosine at least `0.999997146`.

### ROCm Pass 7: Native Direct QKV

Status: rejected; retained as `EI_ROCM_NATIVE_Q4_DIRECT_FP16_QKV=1`.

One kernel fused separate native-Q4 Q/K normalization, RoPE, V conversion, and
FP16 output. The first sweep ranged from `0.988x` to `1.010x`. Against the final
route it remained `0.997x-1.008x` at T96-T368, so the extra path is disabled.

### ROCm Pass 8: Dense SWA at 2K

Status: retained with `EI_ROCM_SWA_TENSOR_TILE_TOKENS=0`.

At T1536/T2048, full-query dense tensor attention measured 9.837/11.639 ms.
Query tiles of 128, 256, 512, and 1024 measured 33.356/42.686,
17.259/21.997, 12.549/15.465, and 11.298/12.733 ms respectively. On MI300X,
the extra rectangular GEMMs cost more than computing masked dense scores, so
zero tiling provides `1.149x` at T1536 and `1.094x` at T2048 versus the prior
1024-query default.

### ROCm Pass 9: Batched GQA Heads

Status: retained as `EI_ROCM_BATCHED_TENSOR_ATTENTION=1`.

Because EmbeddingGemma shares one K/V head across three query heads, tensor
attention now uses two `hipblasGemmStridedBatchedEx` calls per sequence rather
than six per-head calls. A head-aware softmax spans separate score/probability
matrices. Candidate/baseline cosine was exactly 1 in the A/B sweep; throughput
improved `1.208x` at T256, `1.146x` at T512, `1.163x` at T1024, `1.170x` at
T1536, and `1.171x` at T2048.

This changed the dense crossover. Batched tensor attention measured `0.972x`
at T64, `1.014x` at T88, `1.047x` at T96, `1.336x` at T128, `1.418x` at T160,
and `1.500x` at T192. Final automatic thresholds are 96 for one sequence, 128
for two through four, and 192 for larger batches.

### ROCm Pass 10: Fused MFMA FFN Activation

Status: retained only from 320 through the native route's 368-token ceiling.

One wave reuses each input fragment for up and gate MFMA, applies GELU, and
writes FP16 directly into an existing scratch buffer. Reduced wave count hurt
small shapes (`0.925x` at T32 and `0.942x` at T96), but sufficient high-end
occupancy measured `1.039x` at T320, `1.039x` at T336, `1.030x` at T352, and
`1.030x` at T368. Runtime routing keeps both geometries.

### ROCm Pass 11: FP16 Tensor-Attention Scores

Status: retained through 896 sequence tokens.

hipBLAS still accumulates in FP32, while its score output and the following
softmax input use FP16; softmax reductions remain FP32. This measured `1.168x`
at T96, `1.164x` at T128, `1.068x` at T256, `1.036x` at T512, `1.059x` at
T640, `1.053x` at T704, and `1.031x` at T896. It fell to `0.986x` at T1024,
so larger sequences retain FP32 scores. Minimum output cosine was `0.9999905`.

### ROCm Stop Passes

Three consecutive final passes produced no significant general gain:

1. A 128-thread tensor softmax measured `0.994x-1.003x` versus 256 threads.
2. Native direct-FP16 QKV retesting measured at most `1.008x`.
3. HIP graph retesting was shape-inconsistent and regressed T4 to `0.940x`.

The loop stopped with the measured defaults rather than accumulating
architecture-specific complexity without stable throughput.

### ROCm Final Validation and Throughput

The portable four-code-object build passed all llama.cpp goldens at minimum
cosine `0.999910414`, CPU/ROCm synthetic drift at `0.991444767`, native MFMA
projection parity at `0.999983668`, online/hipBLAS attention parity at
`0.999992847`, scalar/MFMA attention parity at `0.999997556`, direct-QKV parity
at `0.999999762`, packed parity at `0.999875247`, and HTTP Matryoshka tests.

Final warmed medians were 1.773 ms at T1, 2.307 ms at T7, 3.598 ms at T32,
3.412 ms at T96, 3.558 ms at T128, 4.788 ms at T256, 6.051 ms at T512,
7.177 ms at T1024, 8.590 ms at T1536, and 9.945 ms at T2048. Relative to the
functional baseline this is approximately `1.26x`, `2.69x`, `1.86x`, `2.85x`,
`1.55x`, and `1.89x` at T1/T7/T32/T128/T512/T2048.

At T32, packed B1/B2/B4/B8/B16/B32 reached 279/545/1,032/1,624/2,649/5,161
req/s; B32 is 165,164 input tok/s and `18.65x` serialized throughput. The real
dynamic service measured 273/279/686/990/1,284/3,507 req/s at concurrency
1/2/4/8/16/32, with an average executed batch size of 25.6 at C32.

## ROCm Cross-Backend Continuation Loop (2026-07-20)

### ROCm Pass 18: Exact Single-Token V-Only Attention

Status: retained as `EI_ROCM_SINGLE_TOKEN_V_ONLY=1`.

The XPU route demonstrated that an independent one-token sequence does not need
Q, K, RoPE, softmax, or an attention kernel: its attention output is exactly
the shared V head repeated across all three query heads. ROCm now detects an
all-singleton batch. Below the MFMA boundary, one packed-Q4 wave kernel projects
V and writes all three FP32 heads directly. In the native range, one MFMA kernel
projects V and writes repeated FP16 heads directly for the attention-output
projection. The expanded-FP16 fallback uses a V-only hipBLAS projection.

Fifteen-warmup/101-iteration single-engine A/B improved T1 from 1.768 to
1.553 ms (`1.139x`) with cosine 1. T2 was unchanged (`1.004x`), confirming the
shape guard. Seven-warmup/31-iteration packed A/B measured `1.145x`, `1.115x`,
`1.080x`, `1.127x`, `1.173x`, `1.089x`, and `1.089x` at B1/B2/B4/B8/B16/B32/B64,
all with exact parity. Candidate B64 reached 22,257 req/s.

### ROCm Pass 19: Direct-Path Residual/Next-RMS Fusion

Status: retained as `EI_ROCM_DIRECT_RMS_FUSION=1` below the 32-token native
MFMA boundary.

Metal's short-shape fusion and ROCm's own FP16 register-cached RMS showed that
the direct FP32 route was paying avoidable launch and global-read costs. A new
kernel computes post-projection RMS, updates the residual, computes the next
RMS from 24 register-held values per logical lane, and writes the next FP32
projection input. The next layer consumes that result directly, reducing the
direct route from roughly four norm launches per layer to two.

Nine-warmup/41-iteration alternating A/B measured `1.600x`, `1.523x`, `1.506x`,
`1.438x`, `1.383x`, `1.291x`, and `1.281x` at T1/T2/T4/T8/T16/T24/T31. T32,
which takes native MFMA and already had the FP16 fusion, remained neutral at
`1.008x`. Minimum candidate/baseline cosine was `0.999999318`.

### ROCm Pass 20: Direct Q4 Activation Reuse

Status: retained as `EI_ROCM_DIRECT_Q4_PAIR=1` for 16-31 flattened tokens.

CPU/Metal adjacent-row kernels suggested explicitly reusing each FP32
activation load across two Q4 output rows. The direct QKV, V-only,
attention-output, and FFN-down kernels now have paired-row variants; the fused
up/gate kernel similarly reuses each activation load across both matrices.
The row boundaries are architecture-neutral and do not depend on a specific
CDNA target.

Five-warmup/20-iteration alternating A/B measured `1.089x`, `1.077x`, and
`1.085x` at T16/T24/T31 with cosine 1. Pairing was neutral at T4/T8 and reduced
T1/T2 throughput to `0.967x`/`0.982x`, so runtime routing starts at 16 tokens
and ends naturally at the 32-token native-MFMA boundary.

### ROCm Pass 21: Fused Embedding and First RMS

Status: marginal; retained as `EI_ROCM_FUSED_EMBEDDING_RMS=1` below 32 tokens.

A register-cached kernel now dequantizes each Q8 embedding row, writes the
residual stream, computes the first attention RMS, and emits the normalized
FP32 projection input without a second global read or launch. A corresponding
FP16 output was tested for MFMA/hipBLAS shapes but was not retained there.

Seven-warmup/31-iteration alternating A/B measured `1.026x` at T1, `1.010x` at
T8, `1.029x` at T16, and `1.027x` at T31. T32/T128/T512/T2048 ranged from
`0.998x` to `1.018x`, so the route is limited to the direct path. Minimum
direct-path cosine was `0.999999349`. This is insignificant pass 1 of 3 because
no measured shape reached the 3% retention threshold.

### ROCm Pass 22: Singleton Metadata Elision

Status: retained as `EI_ROCM_SINGLETON_METADATA_ELISION=1`.

The exact V-only path never consumes positions, sequence IDs, or flash/MFMA
attention tile metadata, but the host still allocated, populated, and submitted
eight small metadata transfers for every all-singleton batch. The batch planner
now recognizes this shape before allocating those vectors and transfers only
token IDs plus pooling offsets. Mixed and multi-token batches retain the full
metadata path.

Nine-warmup/51-iteration engine A/B improved T1 by `1.034x` with cosine 1.
Packed singleton B1/B2/B4/B8/B16/B32/B64 improved by `1.028x`, `1.028x`,
`1.031x`, `1.028x`, `1.053x`, `1.014x`, and `1.008x`; B64 reached 22,383
req/s. The B16 and single-request gains exceed 3%, resetting the insignificant
pass counter.

### ROCm Pass 23: Direct Singleton Final Pool

Status: retained as `EI_ROCM_FINAL_SINGLETON_POOL=1` on the direct path only.

For independent one-token sequences, the final FFN residual update, output
RMS, mean pool, and L2 normalization can be computed by one register-cached
wave per request. The fused kernel avoids writing and rereading the final
residual. It measured `1.012x`, `1.013x`, `1.010x`, `1.017x`, and `1.037x` at
B1/B2/B4/B8/B16 with exact direct-path parity. A native-MFMA crossover test
failed parity at B32 (`0.770338356`), so the guard explicitly excludes every
GEMM/MFMA shape rather than relaxing correctness. This did not reset the stop
counter because the broad gain was below 3%.

### ROCm Pass 24: Singleton-Aware Projection Crossover

Status: retained with direct packed Q4 through 72 singleton requests.

V-only attention removes most attention work from an all-singleton batch, so
the general 32-token MFMA crossover no longer represents the complete route.
An alternating packed A/B kept the direct kernels active beyond 32 tokens.
Direct improved B32/B48/B64/B68/B72 by `1.655x`, `1.355x`, `1.130x`, `1.103x`,
and `1.068x` with minimum cosine `0.999983609`. B76 was only `1.024x`, B80 was
neutral, and MFMA won from B84 onward. The architecture-neutral runtime guard
therefore uses `EI_ROCM_SINGLETON_DIRECT_MAX_TOKENS=72`. This significant
concurrent-throughput pass resets the insignificant-pass counter.

### ROCm Pass 25: Paired Q4 at Extended Singleton Shapes

Status: retained through the singleton direct-route ceiling.

The Pass 24 crossover exposes direct shapes beyond the original T31 pairing
sweep. Exact alternating A/B at B16/B24/B32/B48/B64/B72 measured `1.111x`,
`1.091x`, `1.102x`, `1.096x`, `1.101x`, and `1.111x` for paired Q4 versus
one-row Q4. B8 remained neutral, and B76/B80 correctly remained neutral
because they route to MFMA. This interaction confirms the existing 16-token
pairing threshold and resets the insignificant-pass counter.

### ROCm Pass 26: Four-Row Direct Q4 Reuse

Status: rejected; retained only as `EI_ROCM_DIRECT_Q4_QUAD=1` for diagnostics.

Four-row V-only, QKV, attention-output, and FFN-down kernels doubled activation
reuse relative to the paired route. The extra accumulators and weight fragments
raised VGPR pressure: single-sequence T8/T16/T24/T31 ranged from `0.981x` to
`1.009x`. Packed B32/B48/B64/B72 measured `1.020x`, `1.021x`, `1.007x`, and
`1.030x`, while B16 regressed to `0.965x`. Cosine was exactly 1, but there was
no broad gain above 3%, making this insignificant pass 1 of 3.

### ROCm Pass 27: Pinned Host I/O Staging

Status: rejected; retained as `EI_ROCM_PINNED_IO_STAGING=1` for diagnostics.

Reusable HIP-pinned buffers stage token IDs, offsets, and returned embeddings,
avoiding pageable-memory handling inside asynchronous copies. Engine A/B ranged
from `1.011x` at T1 to `1.037x` at T16, but fell to `0.999x` at T512 and
`0.992x` at T2048. Packed singleton B1-B8 improved `1.008x-1.019x`; B32-B256
were only `1.002x-1.021x`. The isolated B16 result was `1.044x`, but the sweep
does not support enabling extra pinned allocations for that one point. This is
insignificant pass 2 of 3.

### ROCm Pass 28: Post-Fusion HIP Graph Replay

Status: rejected; `EI_ROCM_COMMAND_GRAPH` remains disabled.

The graph experiment was repeated after residual, embedding, metadata, and
singleton launch reductions changed the captured workload. Nine-warmup,
51-iteration A/B measured `0.991x`, `0.974x`, `0.990x`, `0.972x`, `0.992x`,
and `0.994x` at T1/T4/T8/T16/T31/T32. T128/T512/T2048 were neutral at
`1.005x`, `0.999x`, and `1.003x`, all with cosine 1. This is insignificant pass
3 of 3, so the continuation loop stops here.

### ROCm Continuation Stop

The last significant result was the paired-Q4 interaction across the new
singleton-direct range. The following three experiments produced no broad
gain above 3%: four-row Q4 reuse, pinned host staging, and post-fusion HIP graph
replay. Their guarded implementations remain diagnostic-only and disabled;
the measured two-row, fusion, metadata, and batch-aware crossover routes are
the production defaults.

### ROCm Continuation Final Validation

The expanded ROCm suite found and fixed two diagnostic-toggle guard bugs:
metadata elision now requires V-only attention, and disabling direct RMS cannot
skip intermediate FFN residuals when final singleton pooling is enabled. The
portable build then passed all 10 llama.cpp goldens at minimum cosine
`0.999911070`, CPU/ROCm synthetic drift at `0.991408229`, packed batch parity at
`0.999882340`, all new route-toggle comparisons, and HTTP Matryoshka tests. The
direct/MFMA singleton crossover comparison passed at minimum cosine
`0.999979198`. CPU and Metal regression suites also passed.

Final nine-warmup/51-iteration medians were 0.933 ms at T1, 1.576 ms at T7,
1.903 ms at T16, 2.623 ms at T31, 3.600 ms at T32, 3.409 ms at T96, 3.556 ms
at T128, 4.781 ms at T256, 6.044 ms at T512, 7.260 ms at T1024, 8.576 ms at
T1536, and 9.941 ms at T2048. That is 1,072 T1 req/s and 206,011 input tok/s at
full context.

Default explicit singleton batches measured approximately 18,522 req/s at B32
and 26,014 req/s at the B72 direct ceiling; native MFMA measured 65,358 req/s
at B256. A 256-request dynamic-service run at T32 reached 273/279/746/1,198/
1,645/3,952 req/s at concurrency 1/2/4/8/16/32, with average batch 28.4 at C32.
`roc-obj-ls` confirmed `gfx908`, `gfx90a`, `gfx942`, and `gfx950` code objects in
the final binary.

## 2026-07-20: Metal 4 Tensor Projections, Flash Attention, And Packing

Status: retained for Metal on Apple GPUs reporting `MTLGPUFamilyMetal4`.

The Metal build now embeds two libraries: the existing Metal 3.1 library in
`__DATA,__metallib` and a `-std=metal4.0` library in `__DATA,__metal4lib`.
Runtime selection requires `supportsFamily:MTLGPUFamilyMetal4`, so older
Apple Silicon keeps the prior pipelines. `scripts/stage-release.sh` verifies
both embedded sections.

`src/metal4/kernels/tensor_qgemm.metal` adapts llama.cpp's `<metal_tensor>`
MetalPerformancePrimitives Q4_0 matmul into model-specialized projections:
`ei_q4_0_f32_tensor_mm`, a fused Q/K/V variant, a fused up/gate variant, and
64-column instantiations of all three selected at exactly 64 flattened
tokens. The general tile is 128 token columns by 64 output rows with a
32-wide K tile and four simdgroups. The 64-token tile reduced the T64
in-process median from 4.7305 ms to 3.9570 ms and turned the failing 64-token
c1 HTTP cell from 202.3 versus 207.7 embeddings/s into 240.8 versus 207.8.

`src/metal4/kernels/flash_attention.metal` adds
`ei_flash_attention_f16_kv_256` for EmbeddingGemma's three query heads, one
shared KV head, and head dimension 256, with eight query rows per
threadgroup, 32-key online-softmax tiles, exact symmetric-window masking,
half Q/K/V math, and FP32 softmax/output. Single-sequence flash begins at 128
tokens and packed sequences at 192 tokens. FP16 K/V storage is zero-padded to
tile alignment. A mixed-batch routing bug that keyed packed flash eligibility
to the maximum sequence length forced short members onto FP16 flash and
produced 0.999830604 cosine for a 32-token member; eligibility now keys to
the minimum sequence length and the mixed `{1,7,32,129,192}` batch passes at
minimum cosine 0.999956071.

The request-packing cutoff rose from 512 to 1024 tokens after packed-versus-
serialized measurements of roughly 1.34-1.41x at T512 B2/B4/B8 and
1.04-1.05x at T1024 B2/B4; T2048 stayed serialized at roughly 0.997x.
Representative full-graph in-process medians: T128 5.147 ms, T512 7.749 ms,
T1024 12.673 ms, T2048 27.439 ms (about 74.6k input tokens/s, versus
175.5 ms before tensor projections and flash attention).

## 2026-07-20: Grisu2 Float Serialization For HTTP Responses

Status: retained for all backends.

The first fail-fast llama.cpp comparison stopped at 8 tokens c8 with 992.6
versus 1107.6 embeddings/s. A worker-thread `sample` profile showed
`snprintf("%.9g")` dominating through `__dtoa` and locale locks: 768 float
formats per 768-dimensional response. `src/float_format.c` now carries a
float-only C11 port of the Grisu2 shortest-round-trip formatter from
llama.cpp's bundled nlohmann JSON (MIT, Loitsch and Lohmann attribution
retained), and `src/server.c` reserves response capacity once and writes
formatted floats directly into the response buffer. `test_float_format`
verifies exact bitwise round trips for 1,000,012 finite values. The isolated
cell recovered to 1,160.1 versus 1,105.2 embeddings/s, and the full 8-token
sweep passed at 1.18/1.31/1.79/1.07/1.04/1.17x llama.cpp
(`perf/results/2026-07-20/metal4-float-format-c8/`,
`metal4-float-format-short/`).

## 2026-07-20: Demand-Aware Wave Batch Collection

Status: retained for all backends.

A restarted comparison failed at 8 tokens c16 with 1309.5 versus 1641.8
embeddings/s, conflicting with two earlier wins on the same shape. Three
quiet-host reruns showed llama.cpp stable at 1632-1766 while we swung
between 1696 and 2074 embeddings/s: the dynamic batcher was bimodal over
HTTP. The old collector armed one fixed 200 us deadline per batch, and its
readiness predicate (64 requests or 4,096 tokens) could never fire at c16,
so wave formation depended on all resubmissions landing inside one window;
when they straggled, the wave split and phase-locked (in-process, forcing an
800 us wait produced average batch 16.00 and better latency than 200 us,
proving truncation).

The scheduler now tracks the peak number of concurrently in-flight requests
since the previous dispatch. Collection starts whenever more clients are in
flight than queued, keeps extending its deadline while new requests arrive
(bounded by `EI_BATCH_WAIT_MAX_US`, default four times `--batch-wait-us`),
and dispatches the moment the queue reaches the tracked peak.
`EI_BATCH_WAVE=0` restores the fixed window. An intermediate design that
targeted recent batch sizes instead of in-flight demand locked splits in
(two of ten repeats collapsed to average batch 8.00) and was rejected.

In-process T8 (`perf_concurrency_metal`, 2,000 requests):

| concurrency | before req/s | after req/s | before batch | after batch |
|---:|---:|---:|---:|---:|
| 1 | 479.9 | 483.3 | 1.00 | 1.00 |
| 2 | 490.3 | 758.1 | 1.03 | 1.99 |
| 4 | 946.5 | 1,070.3 | 3.03 | 3.98 |
| 8 | 1,837.5 | 2,113.0 | 6.10 | 7.97 |
| 16 | 2,689.4 | 3,259.1 | 10.10 | 15.87 |
| 32 | 4,627.2 | 5,900.9 | 17.70 | 31.75 |

The c2 gain also fixes an alternating singleton pathology where the second
client's hot window expired during the first client's execution. p95 latency
fell at every level (c16: 6.98 ms to 4.97 ms). Batch parity cosine stayed
0.999956071, T512 packing still fills 4,096-token batches, and T2048 stays
serialized. Paired HTTP T8 c16 went from bimodal 1696-2074 to 2189/2186/1841
against llama.cpp's 602-1641, and isolated c32 measured 3,174.0 versus
2,774.4 (`perf/results/2026-07-20/metal4-c16-recheck-run*/`,
`metal4-wave-t8-sweep/`, `metal4-c16-wave-run*/`, `metal4-c32-wave/`).

Warm-sweep measurements also showed llama.cpp degrading under sustained
mixed-concurrency serving (8-token c32 fell from about 2,770 fresh to about
590 warm) while our server showed no hysteresis; published comparisons use
the runner's new `--fresh-servers` mode so every cell measures both engines
fresh.

## 2026-07-21: T128 c1 Serving-Path And Encode-Overlap Recovery

Status: retained for all backends (server and service changes) and Metal
(split command encoding).

The fresh-servers matrix stopped at 128 tokens c1 with 186.7 versus 189.1
embeddings/s, and three isolated reruns confirmed a stable 1.4% loss
(186.1-186.4 versus 188.6-189.2). Flash-versus-legacy and tensor-versus-
legacy A/Bs showed the engine routing was already optimal (flash 5.166 ms
versus legacy 5.306 ms; tensor 5.191 ms versus legacy 10.792 ms), and
llama-bench with flash attention measured llama.cpp's engine at 5.146 ms —
statistically equal to ours. The entire deficit was serving overhead, which a
worker-thread `sample` profile broke down as roughly 47 us of Grisu2 float
formatting, 21 us of tokenization, and 66 us of HTTP core per request.

Five changes recovered the cell, applied and measured incrementally:

1. Solo inline execution (`EI_SOLO_INLINE=0` to disable): a request arriving
   with an idle backend, empty queue, and expected wave of one executes on
   the submitting worker thread, removing two thread handoffs.
2. Ryu digit generation replacing Grisu2 in `src/float_format.c` (Ulf Adams'
   algorithm, tables generated exactly from big-integer arithmetic): 21.4 ns
   per float, 16.5 us per 768-float response versus about 47 us before. The
   1,000,012-value bitwise round-trip suite passes unchanged.
3. Single-writev responses with TCP_NODELAY and no double strlen of the
   9.7 KB body.
4. Keep-alive reads use SO_RCVTIMEO instead of poll-then-recv, removing one
   syscall and one scheduler hop per request; an idle timeout on a clean
   connection boundary closes silently as before.
5. Split Metal command encoding: the forward pass commits after layer 2 so
   GPU execution overlaps CPU encoding of the remaining 22 layers. T128
   in-process fell from 5.166 ms to 5.141 ms and T8 from about 2.04 ms to
   2.009 ms with bit-identical checksums.

Isolated T128 c1 after the fixes: 190.7/189.8/190.2/189.6 versus llama.cpp
189.3/189.1/188.5/189.3 — four consecutive wins with our floor above their
ceiling, from 186.2-186.4 before. Batch parity cosine 0.999956071, backend
parity, HTTP Matryoshka, and the T8 concurrency ladder (485/758/3,277/5,921
req/s at c1/c2/c16/c32) are unchanged
(`perf/results/2026-07-21/t128c1-*/`).

## 2026-07-21: Solo-Inline Versus Backend Dispatch Race

Status: fixed; solo inline execution retained.

The battery-power matrix reached 2048 tokens c2 and lost three consecutive
paired measurements by 0.4-3% — while our own c1 cell won 1.42x. Serialized
processing should never slow down when a second client appears, and
in-process c2 matched c1 exactly, which isolated the regression to the
service layer under HTTP timing. The cause was a race introduced with solo
inline execution: an inline run claims `backend_busy`, but the backend
dispatch loop only waited on queue emptiness — a request arriving during an
inline run signaled the backend, which dispatched it concurrently into the
non-reentrant engine. Interleaved encodes inflated per-request latency
(41.7 ms versus 31 ms forward on battery) and could corrupt concurrent
embeddings in the race window. Long forwards at 2048 tokens c2 kept
re-opening the window, which is why this shape exposed it.

The backend idle wait now also waits out `backend_busy` (inline completion
always signals `queue_ready`, covering shutdown), so engine execution is
single-flight again. Isolated 2048-token c2 on battery went from 24.0 versus
24.1 losing to 34.3 versus 26.7 embeddings/s — a 1.28x win matching the c1
margin — and batch parity (0.999956071), the T8 wave ladder, and in-process
c1/c2 equality at T2048 all held (`perf/results/2026-07-21/t2048c2-racefix-*/`).

## 2026-07-21: Complete llama.cpp Comparison Matrix

Status: accepted and published in README.

The full 54-cell fresh-servers matrix (8-2048 tokens x concurrency 1-32,
5-second targets, caches disabled, llama.cpp b8981) completed with zero
loss-retries: geometric-mean throughput 1.25x llama.cpp, range 1.01x-2.01x,
minimum cross-server cosine at or above the 0.999 gate. Measured on battery
power (recorded in the summary); ratios are paired and order-alternated, so
the uniform battery throttle affects both engines equally. Full validation
passed afterward: unit and Metal suites, HTTP Matryoshka, the
1,000,012-value float round-trip check, runner compilation, release-script
syntax, and both embedded Metal library sections
(`perf/results/2026-07-21/metal4-final-battery2/`).
