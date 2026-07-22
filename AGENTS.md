# Agent guide for embeddinggemma.c

A tiny, model-specialized EmbeddingGemma-300M server with hand-tuned inference
kernels per accelerator (CPU, Metal, CUDA, ROCm, XPU/SYCL). Model-specialization
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

- **CUDA** — `ssh quixi-3090-02` — RTX 3090 (SM 8.6). GPU 0 runs a foreign vLLM:
  NEVER touch it, always `CUDA_VISIBLE_DEVICES=1`. `nvcc` at `/usr/local/cuda/bin`
  (not in PATH).
- **ROCm** — `ssh EricMI300X` — MI300X (gfx942), ROCm 7.2. All 8 GPUs usually idle.
- **XPU** — `ssh -p 29732 lazarus@69.127.50.54` — Arc Pro B60. **The `~/.ssh/config`
  `QuadB60` HostName is STALE (192.168.0.115); use the direct endpoint above.**
  Build needs `source /opt/intel/oneapi/setvars.sh` in the same shell; gnome-shell
  holds the GPUs, so timing is noisy (take best-of-N). `.xpu-deps` is pinned.

Release binaries are raw executables (not tarballs), built per-platform on these
boxes; `sys/time.h`-style portability breaks only surface on Linux, not on the
Mac's Metal build — verify Linux targets on a box before releasing.
