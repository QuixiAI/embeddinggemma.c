# EmbeddingGemma 2: Metal + CPU kernel and serving design

Status: design, 2026-10-09. Nothing here is implemented yet. Order of work:
Metal and CPU to the hardware ceiling first; CUDA (RTX 5090/3090) afterwards.

Sources: the original 300M Metal/CPU backends (`src/engine_metal.m`,
`src/metal*/kernels`, `src/kernels.c`, `src/quants.c`), the current gemma2 code
(`src/gemma2*.{c,m}`, `src/metal/kernels/gemma2.metal`,
`src/metal4/kernels/gemma2_*.metal`), QuixiCore-Metal, QuixiCore-CPU, and
SlimServe (`scaling_on_metal.md`, `perf/optimization_status.md` 2026-08-11..14).

## 0. North star

**The target is the hardware ceiling under many concurrent requests. It is not
llama.cpp.** llama.cpp is optimized for one request at a time; it is kept in
`perf/compare_llamacpp.py` as a sanity floor only. Every result is reported as
achieved TFLOPS / GB/s / GPU-busy under concurrent HTTP load, against the
ceilings below.

| Resource (M5 Max, 18-core CPU 6P+12E, 128 GB) | Ceiling | Source |
|---|---|---|
| GPU `mpp::tensor_ops::matmul2d` fp16 | 43–55 TFLOPS | SlimServe 2026-08-14 |
| GPU `simdgroup_matrix` MMA | 6.3–6.7 TFLOPS | same; 9x below tensor ops |
| GPU DRAM bandwidth | ~550–600 GB/s, *measure with a stream kernel* | QuixiCore triad notes |
| CPU threaded GEMV bandwidth | ~250–300 GB/s (M4 Max, saturated at 8–12 threads) | QuixiCore-CPU |
| CPU ISA | DotProd, I8MM, BF16, SME, SME2, SME2p1 (`SME_I8I32`, `SME_B16F32`) | `sysctl hw.optional.arm` |

### Workload model (text tower, from the op inventory)

- ~273 MFLOP/token of projections. FFN 55%, attention projections 31%, PLE 14%.
- 151 MB of weights per forward pass (Q8_0 layers, BF16 PLE projection).
- Attention cost per token: 1.3% of projections at T=32, 21% at T=512, 51% at
  T=2048, and 128% at T=8192. At ≥4K tokens the 4 global layers dominate.

| Regime | Bound by | Floor (estimate) | Today (engine, M5) |
|---|---|---|---|
| T=1 | weight bandwidth (151 MB) + launch latency | ~0.3 ms GPU (realistic target ≤1.5 ms) | 4.1 ms |
| T=32 packed | weights + dispatch count | target ≤3 ms | 25–27 ms |
| T=2048 single | compute: ~0.85 TFLOP | ~20 ms at 40 TFLOPS | 79 ms flags on (~10.7 TFLOPS); 2157 ms default |
| 8192-token packed wave (512-token docs) | compute: ~2.7 TFLOP | ~65 ms → ~125k tok/s | not measured |

The large-M path today reaches about 25% of the tensor-op ceiling. The
small-M path is about 15x off its floor. Both gaps are far larger than anything
llama.cpp ratios would reveal.

## 1. Serving architecture (the largest lever, per SlimServe)

The current gemma2 Metal path runs one wave at a time: it takes a global
`inference_mutex`, encodes one command buffer, calls `waitUntilCompleted`, and
returns. The host and the GPU never overlap. Media inputs hold the same lock
and are encoded one item at a time. Changes:

1. **Several waves in flight.** Use 2–3 *slots*, each with its own
   preallocated arena (activations, ids, bounds, results). One shared scratch
   buffer makes Metal's hazard tracker serialize independent work; four
   rotating split-K partial buffers fixed exactly this in SlimServe. Wave N+1 is
   collected, tokenized, packed and encoded while wave N executes. Completion
   arrives via `addCompletedHandler`, which signals the waiting requests. There
   is no `waitUntilCompleted` on the serving path.
2. **No per-call allocations; untracked hazards.** Arenas live in an
   `MTLHeap` with `MTLHazardTrackingModeUntracked`, with explicit barriers
   inside the encoder. SlimServe's memoized-allocation fix alone halved its
   step time (2.07x). Weights stay in one buffer that is wired through an
   `MTLResidencySet`. Use `newBufferWithBytesNoCopy` over the mmap instead of
   the current two full copies of the GGUFs.
3. **Replace the global mutex with a GPU submission queue.** Text waves and
   media waves become work items on one submission thread. Media CPU
   preprocessing (image decode/resize, WAV→log-mel) moves *off* the GPU path,
   onto the tokenizer workers.
4. **Wave shaping.** Keep demand-aware wave collection, but choose budgets that
   leave no tiny tail wave. SlimServe's 2176-token budget, which avoided a
   128-token tail, was a free +10%. Group sequences of similar length into a
   wave, so attention tiles stay full and the min-length flash routing rule
   (from 300M) seldom falls back.
5. **One command buffer per wave**, committed in two parts (after the first
   2–3 layers) so encoding overlaps execution. Do not split further: each
   commit costs ~50 µs on the critical path. Do not build graph/ICB capture:
   encoding is 0.2 ms vs 26 ms GPU, the same lesson as 300M.
6. **Mixed requests keep text packed.** Text items in a request containing
   media go into the text wave, not batch-of-1 service calls (today
   `src/server.c:862`).

Gate: `EI_GEMMA2_PIPELINE=N` slots, where N=1 reproduces today's behaviour
byte-for-byte.

## 2. Metal kernels

Conventions:
- Geometry and head dimensions are compile-time (template plus `[[host_name]]`
  instantiations). Runtime-selected tile width cost 44% in QuixiCore.
- The residual stream stays fp32. Activations that feed matmuls are written
  as **fp16 by the preceding fused row kernel**, halving activation traffic.
- Exception: `inp_gate` reads the *un-normalized* residual, so it keeps an
  fp32 or bf16 input until a range check proves fp16 is safe.
- Threadgroup memory ≤16 KB so 2–3 threadgroups stay resident per core.
- Every route has an env flag, default off, byte-identical when off.

### 2.1 GEMM family: three routes chosen by flattened M (crossovers measured)

**M1. `g2m_gemv_q8` — M ≤ ~8 (weight-stationary batched GEMV)**
- One simdgroup per R output rows (R1/R4 like 300M; re-sweep R).
- Each Q8_0 block is decoded once into registers and applied to all M token
  rows (SlimServe: 461–662 GB/s). Lanes take 16 bytes of a 34-byte block, and
  the fp16 scale is loaded once per block (today: once per *element*, plus a
  type switch).
- Type-specialized instantiations (Q8_0, BF16, F32) with no runtime switch.
- Fused variants:
  - `qkv` over concatenated rows (SWA: 1024+512+512, global: 2048+512+512).
  - `up_gate_geglu` writing one 2048-wide buffer (a default-on win in 300M).
  - A **normalize-on-load prologue** for K=512 inputs, so `attn_norm` /
    `ffn_norm` vanish as dispatches.

**M2. `g2m_tensor_gemm` — M ≥ ~16..32 (Metal 4 `matmul2d`)**
- Weight operand, two candidates to A/B:
  - (a) **native int8 operand** with per-32-group fp16 scales applied in a
    temporary cooperative tensor. This is how SlimServe reached 1.19x of the
    bandwidth floor with packed uint4 operands; Q8_0 maps directly.
  - (b) a one-time **load-time repack** into tile-major fp16 (Q8_0) / bf16
    (PLE BF16, not fp16: range) panels. That is about 270 MB extra, trivial on
    128 GB.
  - Pick by measurement. (a) also helps mid-M, where weight bandwidth still
    matters.
- Activations come from device tensors (fp16), not staged in threadgroup
  memory: they are L2-resident, and SlimServe measured −40% from this.
- Tiles: 64×64 (today's), plus a 32-token tile for M 16–64 (in SlimServe,
  padding 24→32 was free and 48 cost +76%), plus 128×64 for very large M.
  Double-buffer K steps; today there is one buffer with 2 barriers and no
  dequant/MMA overlap.
- **Epilogues** (on the accumulator before the store):
  - `geglu`: up and gate rows interleaved at load, so one tile holds matching
    up/gate columns.
  - `gelu_mul_ple`: for `inp_gate`.
  - `clamp`: audio clipped matmuls.
  - `bias`.
  - `fp16 store`.
- A fused gate+up *two-stream* tile lost on M1 Ultra; the interleaved-rows
  version is a different design, and both get measured on M5.
- K=512 makes a normalize-on-load prologue cheap here too: the RMS of a
  512-wide row is one simdgroup reduction.

**M3. Pool-before-project.** `output` (512→768) is linear, and mean pooling
commutes with it. So pool the normalized x per sequence first, then run one
GEMV with M = batch size. This is exact in real arithmetic. It removes n×0.79
MFLOP and an n×768 buffer. It is not byte-identical, so it gets its own flag.

### 2.2 Fused row kernels (replace 25 dispatches/layer with ~7)

| Kernel | Fuses | Replaces |
|---|---|---|
| `g2m_qkv_post` | q/k RMS-norm (weighted), v RMS-norm (unweighted), NEOX RoPE from **precomputed cos/sin tables** (bases 1e4 and 1e6, d 256/512), Q→fp16, K/V→fp16 padded to 32 for flash | 5 norm/rope + 2 `g2_half` |
| `g2m_post_add_next` | RMS(post) → `+x` (× `layer_output_scale` where present) → next RMS(pre) → fp16 normalized copy | 300M `rms_norm_residual_next` pattern; QuixiCore 1.15–1.83x |
| `g2m_ple_prologue` | token gather × √512, PLE projection (via the GEMM routes), then RMS with eps' = 512·1e-6 to absorb the 1/√512 exactly | 3 dispatches |
| `g2m_pool_l2` | final norm + per-sequence mean pool (+ M3 projection) + L2, row-parallel for long sequences | 2 + host L2 |

Target per text layer: QKV GEMM, `qkv_post`, attention, O GEMM,
`post_add_next`, up/gate+GeGLU GEMM, down GEMM, `post_add_next`, `inp_gate`
GEMM+PLE epilogue, proj GEMM, `post_add_next`. That is about 11 dispatches,
down from 25, with no fp32 2048-wide intermediates. Expected effect: large at
small M (dispatch-bound), moderate at large M (traffic).

### 2.3 Attention

**A1. `g2m_flash_varlen` — packed waves.**
- Host builds a **tile worklist** (QuixiCore `attn_varlen`): each sequence is
  padded to the Q-tile height, so tiles never straddle sequences, and each
  tile carries its sequence's start and length. This replaces the
  per-token `bounds` array.
- QKᵀ and P·V through `matmul2d` (9x the simdgroup rate), with fp32 softmax.
- Q tile 16–32 rows: QuixiCore's 8→16 rows took D=128 from 7.4 to ~11.6
  TFLOPS because of K/V re-read pressure. The output stays in cooperative
  tensors / registers, not threadgroup memory.
- **GQA in one threadgroup:** the Q heads that share a KV head are processed
  together, so K/V are loaded once. That is 2 heads per KV head on SWA layers,
  4 on global.
- SWA: tile skipping with exact window bounds. Today the start is aligned down
  to 32, so fully masked tiles are computed.
- Instantiations: d=256 (SWA ±512) and d=512 (global, threadgroup budget!).
  Vision uses d=64, full, axial RoPE fused upstream.
- The global d=512 layers are *the* cost at ≥4K tokens. Consider a split-KV
  (flash-decoding style) partition for single very long sequences, so the
  whole GPU is busy.

**A2. `g2m_attn_short` — sequences ≤ 32 tokens.**
- One threadgroup per (sequence, KV group). Whole-sequence K/V sit in
  threadgroup memory: 32×256×2×2 = 32 KB for K+V, too big, so use d-split or
  fp16 K-only staging. Measure against A1.
- Today short requests use the scalar `g2_attention` walk.

**A3. Audio relative attention.** One kernel fusing the 12-key causal window,
`q·(k+rpe)`, the 50·tanh soft-cap and the per-dim/softplus scales. QuixiCore
measured 5.3x vs composed ops. `rpe` is **precomputed at load**: its input is
a constant table, which removes 12 matmuls per chunk.

### 2.4 Media towers

- **Vision:** same GEMM/row/flash kernels at width 768.
  - Fused patch extract → fp16-round → patch GEMM → x/y position add.
  - Fused 3×3 spatial pool → RMS.
  - **Several images per GPU run** via the varlen worklist (full attention per
    image), up to a patch budget. Today each image is its own command buffer.
- **Audio:**
  - Clamps move into the GEMM prologue/epilogue, which removes 240 of 625
    dispatches.
  - GLU and SiLU move into epilogues.
  - Fused depthwise conv (kernel 5) + RMS + SiLU.
  - Several 30 s chunks per run.

## 3. CPU kernels (Apple Silicon first; x86 secondary)

Threading stays on `ei_thread_pool`: function pointer plus context, inline when
the work is small, dynamic chunk claiming. The 6P+12E layout punishes static
equal splits; 18 threads beat 12 by ~25% in the 300M work. T=1 uses a
**6-thread row-parallel width** (300M: 7–8 threads regressed). The CPU gets a
real **packed-batch graph** with per-token positions, block-diagonal attention
and persistent per-worker workspaces. Today `src/engine.c:992` loops over
sequences, with ~9 callocs per call.

### 3.1 GEMM family

**C1. `g2c_gemv_q8_sdot` — M ≤ ~4.**
- Q8_0×Q8_0 with `vdotq_s32`: two dots per 32-block, fp32 lane accumulators
  live across K (300M: +11–18%), and fp16 scales **pre-converted to an fp32
  side array at load**.
- Shapes ported from 300M: rows3, dual (K/V, up/gate) and triple (QKV at T=1).
- Activations quantized once per normalized row (today: re-quantized for each
  of q, k and v).

**C2. `g2c_gemm_q8_i8mm` — M ≥ ~4–8.**
- Load-time repack into `smmla`-native panels: 2 rows × 8 K-bytes
  interleaved, block-major, scales in a side array (QuixiCore `packed_weights`
  layout, done properly).
- Register-blocked micro-kernel around 4 tokens × 8 outputs: 8 int32 `smmla`
  accumulators reset per 32-block, then FMA'd into fp32 accumulators with
  `d_w·d_x`. Sweep 8×4 / 4×8 / 8×8 within 32 NEON registers.
- Order: activation rows outermost, contiguous output stores (QuixiCore: a
  weight-major traversal with strided stores cost up to 41%).
- No such kernel has ever been benchmarked against SDOT in QuixiCore. This is
  new ground, and the expected prize is large: smmla does 2x the MACs of sdot
  per instruction.

**C3. `g2c_gemm_sme2` — large M, experiment.**
- int8 outer products (`SME_I8I32`) into ZA tiles, BF16 (`SME_B16F32`) for the
  PLE projection.
- Streaming-mode entry/exit cost and ZA save rules decide the crossover. Use
  it only for whole-layer-sized work, where one smstart amortizes across many
  tiles. KleidiAI's SME2 `qai8dx` micro-kernels are the reference.
- Default off until it beats C2 in a same-host A/B, in both run orders.

**C4. BF16 PLE projection** (512×12288). Use `bfmmla`/SME B16 instead of
`value()` per element. Keep BF16 rather than requantizing, to stay within the
reference.

### 3.2 Fused row kernels

- `norm_quant`: RMS + Q8 quantize in one pass (300M: 1.07x; QuixiCore F6/F7:
  5–7x at 6 threads).
- `geglu_quant`: gelu·up → Q8 (QuixiCore F3 up to 2.19x).
- `post_add_next_quant`: Gemma sandwich + next pre-norm + Q8.
- `qkv_post`: q/k/v norms + RoPE from tables (QuixiCore 1.8x on M5).
- PLE `inp_gate` epilogue.
- All of them NEON and threaded. Today every norm, RoPE and elementwise pass is
  single-threaded scalar, and RoPE calls `powf/cosf/sinf` per element.

### 3.3 Attention

- **`g2c_flash`:** query blocks of 16 per task (QuixiCore's 16-score online
  tile beat 32). GQA heads share K/V loads. SWA bounds are exact. Per-worker
  scratch replaces today's calloc per (token, head).
- QKᵀ/PV on BF16 `bfmmla` or fp32 NEON. Measure which; online softmax lost
  6–8% in 300M at T ≤ 512, so keep a three-pass variant for short sequences.
- Global d=512 layers at long T: parallelize over (query block × head), not
  queries alone.

### 3.4 Media

- Vision and audio use the same GEMMs.
- Audio conv2d becomes im2col + C2.
- Depthwise conv (k=5) in NEON.
- `rpe` precomputed at load.

### 3.5 x86 (secondary)

The default x86 build has no `-mavx2`, so the 300M Q4 dot falls back to scalar.
Fix that with runtime CPUID dispatch (AVX2, AVX512-VNNI), promoted per kernel
only after a same-host A/B; AVX-512 lost 19/23 formats in QuixiCore.

## 4. Execution order and gates

Each step: measure, implement behind a flag, cosine gates (goldens ≥0.9997,
route parity ≥0.9999, packed self-parity), then measure engine-level ceiling
metrics and an HTTP concurrency matrix. llama.cpp ratios are context only.

1. **Instrumentation.** Per-wave GPU time, command buffers in flight, achieved
   TFLOPS/GB/s per kernel family (extend `EI_GEMMA2_METAL_PROFILE`), and a
   stream-bandwidth probe.
2. **Metal serving pipeline** (§1). Expect the largest concurrent-load gain.
   It is independent of the kernels.
3. **Metal M1 GEMV + fused row kernels** (§2.1 M1, §2.2). Fixes T ≤ 32.
4. **Metal M2 tensor GEMM v2**: int8 operand vs repack, epilogues, tiles.
5. **Metal A1 varlen flash** with matmul2d, GQA grouping and tile skipping.
6. **Media batching + audio/vision fusions.**
7. **CPU packed graph + C1 + fused row kernels.**
8. **CPU C2 i8mm**, then the C3 SME2 experiment, then attention.

## 5. Open questions (answer by measurement)

- Does matmul2d with an int8 operand beat a load-time fp16 repack at mid-M on
  M5?
- What are the crossovers between M1, M2 tile 32 and M2 tile 64 on M5? (M1
  Ultra crossovers were −19% on M5 in SlimServe.)
- Is fp16 safe for `inp_gate` inputs and vision activations? Check ranges on
  the golden fixtures first.
- Two command queues: does a small latency-bound wave overlap a large one?
  (In SlimServe a second queue ran concurrently, but its prefetches could not
  be aimed.)
- Where does SME2 overtake i8mm, including the streaming-mode cost?
