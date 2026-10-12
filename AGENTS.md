# Agent guide for embeddinggemma.c

A model-specialized server with two startup profiles. `embeddinggemma-2` is
the default and supports text, images, mono 16 kHz WAV audio, sampled video
frames, and mixed inputs using native CPU and Metal inference. `embeddinggemma-300m`
retains the original hand-tuned CPU, Metal, CUDA, ROCm, and XPU/SYCL kernels.
The new profile does not link llama.cpp or ggml. `README.md` documents its
content-object API; `EXTRACTION.md` records its reference graph and goldens.
`make test-gemma2` exercises the new tokenizer, graph, media, and HTTP paths.
`make test-gemma2-metal` tests the default Metal path;
`make test-gemma2-metal-routes` tests Metal 4 routes and packed sequence isolation.
Gemma 2 runs on CUDA (`src/gemma2_cuda.cu`: text, vision and audio towers,
validated on an RTX 5090); ROCm/XPU remain pending.
The performance lessons below apply to the original 300M profile. Model-specialization
is the whole edge over llama.cpp — a generic engine can't fuse or specialize the
way this one does. Build/test/env-var reference is in `CONTRIBUTING.md`; the full
retained/rejected experiment log with numbers is in
`perf/optimization_status.md`. Read both before optimizing.

This file is the durable, hard-won lessons — the things that are expensive to
re-learn. (`CLAUDE.md` is a symlink to this file.)

## Where the performance wins are — and aren't

As of v0.3.0 we beat llama.cpp in all 54 benchmark cells (~1.25–1.4x geomean).
**The big win was the dynamic batch scheduler (demand-aware wave collection), not
the kernels.** A whole investigation (see `optimization_status.md`, 2026-07-21)
established the ceiling:

- **2x vs llama.cpp is NOT reachable by same-hardware compute optimization.**
  Short sequences are weight-bandwidth-bound (you read all 300M params
  regardless, so int8 activations can't help); long sequences are a matmul race
  where our kernels and llama.cpp's are both near the achievable ceiling.
- **The big GPUs (Metal/CUDA/ROCm) are compute-bound, not launch-bound.** Metal
  CPU-encode is only 1–4% of the forward (measured with an env-gated probe), and
  the split command buffer already overlaps it — a command-buffer/graph capture
  would chase ~2%. **Do not build Metal graph capture.**
- **Fusion-to-cut-launches only helps SUBMISSION-bound backends.** That's XPU
  (SYCL immediate-command-list submission is ~97% of a small-batch forward) —
  its ctx→half fusion is a real +2–5%. On CUDA it's neutral (graph capture
  already makes launches free); on ROCm it *regresses* (−45% at T1 — coarse
  fused blocks collapse GPU occupancy). Don't blindly port a fusion across
  backends; measure per-backend.
- **W4A8 / int8 tensor cores are real but small.** The model already has int8
  activations; the lever is routing them through the int8 matrix engines
  (IMMA/MFMA/XMX). But the per-32-block Q4/Q8 scales rule out vendor INT8 GEMMs
  (cuBLASLt/CUTLASS apply one scale over the whole K). A hand-written per-block
  int8 GEMM (cp.async double-buffered) *can* beat cuBLAS fp16 1.0–1.41x on the
  projections — but projections are only ~40% of the forward and too small
  (K=768/1152) to be compute-bound (even cuBLAS runs them at ~30% of peak), so
  it nets ~3% end-to-end. int8 attention adds ~3% (Q·Kᵀ is accuracy-safe; int8
  P·V on softmax probabilities breaks accuracy). All opt-in, ~6% combined,
  high-token only. Flags: `EI_CUDA_W4A8_GEMM`, `EI_CUDA_INT8_ATTN`.
- **The real >2x is caching, not compute.** The persistent exact cache
  (`--persistent-cache-path`) turns repeat/re-index/eval workloads into cache
  hits (∞x on a hit) — that's where a production embedding server's real
  multiplier lives. Do caching before chasing more kernel percent.

## EmbeddingGemma 2 (Metal + CPU, 2026-10-09)

Design and numbers: `perf/gemma2_kernel_design.md`, `perf/optimization_status.md`.
The target is the hardware ceiling under many concurrent requests (SlimServe
`~/SlimServe` is the reference design), not llama.cpp.

- **The lever was GPU-level batching.** The original `g2_mm` had no weight
  reuse across tokens, so packing requests bought nothing. Weight-stationary
  GEMV (short waves) plus `matmul2d` GEMMs (wide waves) made packed waves
  cheap; the HTTP throughput gain at c16 was 10-27x.
- **`matmul2d` on M5:** row-major fp16 weights (dequantized at load) as a
  transposed operand with `relaxed_precision` reaches ~37 TFLOPS. The int8
  operand + per-32-group scale recipe lost ~3x once compute-bound; tile-major
  layouts and strict precision lost too. Fuse epilogues (GeGLU, PLE) in it.
- **Several waves in flight did not help** uniform loads (`--pipeline-slots`):
  one packed wave beats overlapping small ones. Keep it opt-in.
- **Every `matmul2d` flash instantiation at head width 64 crashes the M5
  driver** at first dispatch; vision uses the simdgroup flash kernel.
- **Dispatch count is not the short-wave bottleneck:** folding post-norms into
  GEMV prologues removed 72 dispatches and gained nothing.
- **CPU short waves were thread-wake bound** (condvar per call). The spin pool
  must publish generation and participant count as one atomic ticket (a split
  publish hung). `smmla` doubles `sdot`; attention needs register tiles.
- **CPU numerics:** both CPU graphs quantize activations to Q8, ~0.98-0.999
  from the FP32 Metal path on fragment inputs; random token ids exaggerate it.
  Gate CPU work on goldens and natural text, not random ids.
- **CUDA uploads must be on the engine stream.** A pageable `cudaMemcpy` can
  return before its DMA lands, and the legacy default stream does not order
  with a non-blocking stream: lazily uploaded weights raced their first kernels
  (nondeterministic vision outputs). Use `cudaMemcpyAsync` on the engine stream.
  Never call `cudaFuncSetAttribute` inside graph capture.
- **This host is shared** (user builds, a VM): compare configurations
  back-to-back in one process (`make bench-gemma2-metal`, `EI_BENCH_BASE`) or
  interleaved; separate HTTP runs are not comparable across time.

## EmbeddingGemma 2 media lessons (2026-10-10)

- **A head width the driver cannot compile can be padded.** No `matmul2d`
  flash at width 64 compiles, but 64-wide Q against 128-wide zero-padded fp16
  K/V rows is exact (pad columns contribute zeros, never stored) and runs
  1.5x the FLOPs on tensor ops: 768x768 vision 59 -> 40 ms alone.
- **Media was lock-bound, not kernel-bound, under load.** Every tower and the
  final text pass of every media request serialized on `inference_mutex` on
  one context, so c1..c64 throughput was flat. Per-request media slots (own
  queue/workspace/lock) and packing the text finishes into one wave gave
  image432 c8 51 -> 103 req/s before any kernel work counted.
- **Check what the tile route still serves.** The F32 patch embedding and the
  audio pre-encode / input projection never got fp16 copies and ran the old
  per-tile decode GEMM; packing them was 1.36x on 10 s audio for free.
- **Elementwise passes are ~20-40% of a media tower.** Vision had 17
  dispatches per layer, audio 46 (two clamp passes per GEMM): fusing into
  producers and GEMM epilogues was 1.2x each after the attention fix.
- **MPP cooperative-tensor inputs are single-simdgroup only**
  (`get_left_input_cooperative_tensor` static-asserts `execution_simdgroup`),
  so a register-resident flash (no S/P threadgroup round trip) would need
  8-row tiles per simdgroup; at head width 512 that is 128+ accumulators per
  lane. Not pursued.
- **16 tokens per GEMV decode lost** (16x1 4.3 -> 5.6 ms): the short-wave GEMV
  is bound by per-token inner work, not weight re-reads. 8 stays.
- **CPU front ends matter at c1.** Log-mel evaluated 2,304 cosf/sinf per
  frame and a dense 128x257 filter bank: 24 ms for 10 s of audio, more than
  the GPU tower. Twiddle tables and sparse filter ranges are bit-identical.
- **Packing video frames into one wave is worth ~10% at the default budget**
  (the tower is compute-bound at 729 patches; 1.25-1.43x for 64- and
  25-token frames). Every attention/RoPE route already took per-token
  bounds, so the whole change was `t % p.k` in four kernels plus a bounds
  buffer. Build on a GCC box before calling the Mac build done: GCC caught a
  use-after-free and an unprovable array bound that clang passed.
- **Single long bench runs drift 2x on this host.** The same 8-frame wave
  read 151, 106 and 62 ms in consecutive runs. `EI_BENCH_REPEAT` round-robins
  the item list; trust only interleaved, repeated numbers. Check `ps -r`
  first: a user build, a browser renderer and WindowServer compositing were
  each eating 40-100% of a core (and the GPU) during the 10-10 evening pass.

## EmbeddingGemma 2 text lessons (2026-10-10 evening)

- **The one-token forward (2.2 ms) is a latency chain, not launches or
  bandwidth.** 240 dependent trivial dispatches cost 0.2-0.8 ms here (1-3 us
  each); the real kernels cost 8-12 us each because each has a few dependent
  memory round trips and barriers. Fusion pays only when it removes round
  trips (keeping norm -> GEMV -> residual resident), which is why folding
  post-norms into GEMV prologues gained nothing on 10-09.
- **Long context is attention-bound (65% at 8192 tokens)** and the flash
  kernel ran at ~20 of ~37 TFLOPS. Doubling the simdgroups per tile (same
  32x128 tile, 64 instead of 128 accumulator floats per lane at head width
  512) was 1.1-1.27x at 2k-8k tokens with identical arithmetic; smaller
  query tiles and 256-key blocks were comparable on average but swung
  between runs. Name tile variants in the kernel host name and parse the
  geometry from it so a sweep is an env var.
- **Count a request as arriving before it is tokenized.** The wave collector
  only saw queued requests, so at 512 tokens x c16 it phase-locked into
  15 + 1 waves for the whole run. Verify scheduler changes with the wave log
  (`EI_WAVE_LOG`), not throughput: the host noise is larger than a 5% wave
  effect.
- **Linux needs `_POSIX_C_SOURCE 200809L` before any include** for
  `clock_gettime`; clang on the Mac does not complain.
- **Never launch a spinning GPU kernel without a tiny spin cap.** A
  device-atomic barrier probe with a 2x10^7 spin cap ran for minutes once
  its groups were not co-resident, and killing the process did not stop it:
  the command buffer kept the M5 at 100% device utilization long after
  (`ioreg -l -w0 | grep "Device Utilization"` shows it). Cap spins at ~10^5,
  start from 20 groups, and run such probes with a short `timeout` plus a
  check of the utilization counter afterwards.
- **Device-atomic barriers across threadgroups work on the M5** when the
  group count is modest: 160 x 256-thread groups rendezvous at 1.8-3.9 us per
  barrier (`perf/grid_barrier_probe.m`, safe caps). That is the route to the
  short-wave floor (11 round trips/layer at ~10 us each); gate it behind a
  startup probe and a bounded spin that fails the wave rather than hangs.
- **The flash kernel is not K/V-bandwidth-bound.** Doubling queries per K/V
  byte (64x64 tiles) was 0.7-0.8x, so GQA K/V sharing and fp16 Q staging
  cannot pay; the 8-simdgroup 32x128 tile stays.
- **Front-end wins that cost nothing numerically:** table base64 (18x) and
  libcompression inflate for PNG IDAT (1.65x), both byte-identical. Compare
  HTTP responses with `cmp`, and send unique bodies: the response cache
  serves repeats, and curl's `Expect: 100-continue` breaks this server's
  body read.

## EmbeddingGemma 2 text lessons (2026-10-11)

- **A device-atomic grid barrier costs what a dispatch boundary costs on the
  M5** (2.2 us at 64 groups, 3.7-4.7 at 128-256, 2.4-3.0 at 384-512; the
  device-scope fence adds ~1 us), so a persistent per-layer kernel that only
  replaces dispatches with barriers does not move the short-wave floor:
  built, bit-identical, 0.99-1.13x at 1-8 tokens and 0.5-0.8x at 20-39
  (each group runs its token groups serially instead of on more cores). It
  ships opt-in (`EI_GEMMA2_METAL_LAYER=1`). Further gains need stage fusion
  that removes round trips (norm recomputed per group, attention normalizing
  K/V on the fly), each worth one barrier of ~3 us; the one-token floor is
  the ~11 dependent memory latencies, not the sync primitive.
- **`atomic_thread_fence` needs `-std=metal3.2`**; the base library is 3.1,
  so kernels that need it live in the Metal 4 library (`gemma2_layer.metal`
  includes `gemma2.metal` for the shared bodies).
- **A probe's host timing is not the barrier cost**: 264 barriers read 16 us
  each through `finish()` and 3.7 us by GPU timestamps.
- **Attention is bidirectional (sliding window `t - w .. t + w`)**, so prefix
  K/V reuse across requests is impossible for this model; the exact token
  cache in the inference service already covers repeated text.
- **Waves pack ragged, so there is no padding to bucket away.** Ordering
  inside a wave is worth 2-4% (longest first, seven ordered pairs in-process);
  bucketing by length would buy nothing more.
- **Media repeats were the uncached case.** The HTTP response cache only hits
  identical bodies; the same image with different text re-ran decode, resize
  and the tower. Caching tower outputs by payload hash (and content objects
  by text) is 3.5x at c1 on image768 and byte-identical.

## EmbeddingGemma 2 media lessons (2026-10-12)

- **Measure packing with the bench before building a collector.** Eight
  432x432 images packed into one vision wave run 62 ms against 69 ms for
  eight single calls (768x768: 3.5%), so cross-request image packing is not
  worth a thread. Audio was the opposite: a 30 s clip costs 19 ms where three
  10 s clips cost 28, and two streams reach only 1.3x of one, so the packed
  audio tower is 1.35x at two clips and 1.48x from four (`aFxC` bench items).
- **Packing a tower needs every cross-token kernel to take bounds.** The
  audio conformer had two: the causal depthwise conv (taps now stop at the
  clip's first token) and the relative-position attention, whose `p.mode`
  key range was computed after the bounds clamp and silently overrode it.
  The subsampling convs run per clip with offset binds; their outputs are
  already contiguous rows, so the packed layout costs no copy.
- **HTTP throughput caps at the sum of the GPU passes, not the tower.** At c8
  a packed audio request costs ~6 ms of tower plus ~4 ms of its share of the
  2k-row finish wave, so the 1.48x tower is 1.1-1.4x of requests per second
  (93 req/s at the ceiling). Read the `EI_MEDIA_TIMING=1` stage means under
  load before attributing a gain to a kernel.
- **Megabyte JSON strings were parsed a byte at a time, twice.** The input
  scanner and the content-part parser both ran `parse_json_string` over the
  base64 payload (append_byte per character), and the response cache hashed
  the whole body with byte-wise FNV: 4-6 ms per request on a 1 MB image or
  video hit, 10-15% of the hit latency. A `strcspn` run copy and an 8-byte
  hash removed it; the cache hash is in-memory only, so changing it is free.
- **The image front end is not the miss cost.** PNG decode is 1.7 ms at
  432x432 and 4.8 ms at 768x768, resize 0.1-0.2 ms, against 12 and 40 ms of
  tower; hardware decode would buy under 10%.
- **The flash softmax phase is not the long-context bottleneck.** `fast::exp`
  in the online softmax was 1.00x (cosine 0.9999999) and skipping the
  running-output rescale on blocks where no row maximum moved was 0.94-1.01x,
  so the 20-of-37 TFLOPS gap is in the matmul and threadgroup round trips, not
  the exponentials. Tile names are parsed for geometry (`containsString`,
  not suffix): a variant name ending in a letter dispatched half the
  simdgroups and produced NaNs before that was fixed.

## EmbeddingGemma 2 scaling lessons (2026-10-10)

- **One CPU wave peaks at ~64 threads and collapses past it** (256-thread pool
  slower than 32). Big machines need `--cpu-workers` (independent pools, pinned
  by L3 domain): 16x16 = 3.75x the best single pool on 256 EPYC cores.
- **CPU overflow beside a fast GPU is ~worthless**: a 5090 is 15-100x all the
  cores, and CPU workers starve the HTTP threads. It stays opt-in and only
  takes real backlog. The real multipliers are more GPUs and CPU-only workers.
- **HTTP workers cap requests in flight** (one keep-alive connection per
  worker); a "slow" executor layout may just be starved. Check `busy=` in
  `EI_WAVE_LOG` before blaming kernels.
- **Scheduler predicates must agree**: a slot picker that promises work the
  taker won't take spins with the service mutex held (a full deadlock).

## How to do kernel / perf work here

- **Measure before building.** Profile to size the prize first. An env-gated
  Metal encode-time probe (1–4%) killed a multi-hour capture build before it
  started; the XPU `EI_XPU_PROFILE=1` submission timing found the 97% bottleneck.
- **Gate every new numeric path behind an env flag, default OFF, byte-identical
  when off.** That's how W4A8, int8 attention, and the fusions all shipped.
- **Validate against the cosine gates**, and know which gate exercises your code:
  - golden (`test_embed_*`, ~0.99) — but its samples are all ≤37 tokens, *below*
    the 80-token tensor threshold, so it does NOT exercise high-token / tensor /
    int8-attention paths. Use the route/synthetic tests for those.
  - route parity (`test_cuda`/`test_rocm`/`test_xpu`, ~0.99–0.999999),
    batch self-parity (`test_batch_*`, cuda 0.998 / rocm/xpu 0.999).
  - int8-activation paths have an inherent self-parity ceiling ~0.997 — that's
    the quant, not a bug.
- **Per-backend kernels build only on their hardware.** The Mac builds CPU +
  Metal only. `engine_cuda.cu` / `engine_rocm.hip` / `engine_xpu.cpp` must be
  compiled and validated on the boxes below.

## Benchmarking discipline (`perf/compare_llamacpp.py`)

- The two engines measure back-to-back; the second is penalized ~12–16% at high
  tokens by the first's thermal/battery carryover. Concurrency is the inner
  loop, so measurement order is *fixed per cell* and never averages out. Use
  `--both-orders` (averages A/B) + `--inter-engine-cooldown` for trustworthy
  high-token numbers — otherwise you get a sawtooth (e.g. T2048 1.39x/1.05x that
  is really a flat ~1.30x).
- Quiet-host guard uses a *windowed mean*, not an instantaneous sample — a
  single blip must not reset it (that once turned a 15-min run into 2 hours).
- Battery throttles the GPU ~40% but hits both engines equally, so paired ratios
  stay valid; only a mid-cell power-source *change* invalidates a cell.
- Fresh servers per cell: llama.cpp degrades ~4x when warm under sustained mixed
  load, which would flatter us.

## Operational: never strand work on a box

**tmpfs `/tmp` is wiped on reboot.** A routine XPU-box reboot ate an hour of
validated work mid-session (only recoverable because the diff was in the session
transcript). Consolidate and commit promptly; never leave the only copy of a
patch in a box's `/tmp`. Bring work *here* (the repo), don't let it live on a
machine.

Remote backend boxes (repos cloned at `~/embeddinggemma-bench` on each):

- **CUDA (5090)** — `ssh eric-5090` — RTX 5090 (SM 12.0), CUDA 13.3, repo at
  `~/embeddinggemma.c` (rsync'd working tree). Its vLLM is the systemd unit
  `slimserve-affine-king`; stop it only with the user's OK and say so
  (`sudo systemctl start slimserve-affine-king` restores it).
- **CUDA** — `ssh quixi-3090-02` — RTX 3090 (SM 8.6). GPU 0 runs a foreign vLLM:
  NEVER touch it, always `CUDA_VISIBLE_DEVICES=1`. `nvcc` at `/usr/local/cuda/bin`
  (not in PATH).
- **ROCm** — `ssh EricMI300X` — MI300X (gfx942), ROCm 7.2. All 8 GPUs usually idle.
  Its 2x EPYC 9754 (256 cores, 2 NUMA nodes) is the many-core CPU test bed;
  `~/eg2-work` holds a synced tree plus the EmbeddingGemma 2 models.
- **XPU** — `ssh -p 29732 lazarus@69.127.50.54` — Arc Pro B60. **The `~/.ssh/config`
  `QuadB60` HostName is STALE (192.168.0.115); use the direct endpoint above.**
  Build needs `source /opt/intel/oneapi/setvars.sh` in the same shell; gnome-shell
  holds the GPUs, so timing is noisy (take best-of-N). `.xpu-deps` is pinned.

Release binaries are raw executables (not tarballs), built per-platform on these
boxes; `sys/time.h`-style portability breaks only surface on Linux, not on the
Mac's Metal build — verify Linux targets on a box before releasing.
