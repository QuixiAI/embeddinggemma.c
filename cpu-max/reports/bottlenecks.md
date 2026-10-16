# Ranked CPU bottlenecks — EmbeddingGemma 2 on i7-14700HX

(Paths: engine sources relative to the repository root, tools and results to `cpu-max/`.)

Method: perf/hardware counters are unavailable on this host
(`perf_event_paranoid=4`, no sudo), so bottlenecks were sized with wall-clock
instrumentation inside the engine (`EI_GEMMA2_REF_PROFILE=1` for the upstream
reference graphs, `EI_GEMMA2_CPU_PROFILE=1` for the fast graphs) and classified
by comparing achieved rates with the hardware's peak rates:

* P-core int8 VNNI peak ≈ 64 MAC/cycle (2× 256-bit `vpdpbusd`), ~4.6-5.3 GHz
* P-core fp32 FMA peak ≈ 16 MAC/cycle (2× 256-bit FMA)
* E-core: 128-bit execution, about half the P-core rates per cycle at 3.9 GHz
* DRAM bandwidth (DDR5 dual channel) tens of GB/s; L3 33 MB shared

## Upstream (unmodified QuixiAI, P8 = 8 P-core threads)

| rank | bottleneck | share | achieved | class |
|---:|---|---:|---:|---|
| 1 | vision attention (scalar fp32, per-(query, head) jobs, calloc per job) | 67.9% of i280 tower (21.5 s) | 7 GMAC/s on 8 cores | compute, instruction-bound (no SIMD, no tiling) |
| 2 | vision + audio Q8_0 GEMMs (scalar per-element dot loops) | 26-69% | ~46 GMAC/s on 8 cores | compute, instruction-bound |
| 3 | activation quantization before every GEMM | 3-8% | single-threaded | serial section (Amdahl) |
| 4 | audio subsampling convs through a per-element type switch | 11.8% of a15 | scalar | compute, instruction-bound |
| 5 | vision RoPE (`powf`/`cosf`/`sinf` per element per layer), norms, activations | ~2-5% | single-threaded | serial section |
| 6 | text attention (8-query two-pass softmax, 4×4 dot tiles + hsum) | 70% of a 2,048-token text pass | ~63 GMAC/s on 12 E-cores | compute (poor tiling) |
| 7 | BF16 per-layer-input projection (`gemm_f`, weight rows reconverted per call) | ~7% of long text | ~53 GMAC/s on 12 E-cores | compute (no reuse, hsum) |

## Fork (`5dd7dbf`), same budget (P8)

Stage profile: `scripts/profile_final.sh P8` → `reports/raw/profile_P8.txt`
(in-process, one tower or text forward at a time). Rates are MACs of the
stage divided by its wall time.

| rank | bottleneck | share | achieved | class |
|---:|---|---:|---:|---|
| 1 | vision attention (fp32 tiled online softmax, 12 heads x 64) | 84% of the 10,080-patch tower (8.7 s), 57% at 2,520 | ~288 GMAC/s on 8 P-cores (36 per core) | compute-bound: fp32 FMA at roughly half of the 2x256-bit FMA issue rate; exp/max/rescale passes and L1/L2 traffic of the S/O tiles take the rest (register-tile and block-size variants: within ±5%, log 6 and 12) |
| 2 | text backbone attention (head width 256/512, 1-2 KV heads, ±512 window) | 35% of a 512-token forward, 56% at 2,048 | ~318 GMAC/s at 2,048 tokens | compute-bound, same kernel; dominates long text and the final pass of audio/video requests (a 30 s clip is 754 tokens: 42% of its request) |
| 3 | Q8_0 GEMMs (repacked AVX-VNNI) | 47% of text at 512 tokens, 66% of the audio tower, 28% of the 2,520-patch vision tower | 0.93-1.09 TMAC/s on 8 P-cores (single P-core: 211 GMAC/s) | compute-bound; per core at P8 ~65% of the single-core rate (all-core clocks, shared L2/L3) |
| 4 | short-request overheads (16-token text: 7.1 ms) | GEMM 52%, row passes 25% | ~240 dependent parallel regions per forward | latency / synchronization-bound, not bandwidth (130 MB of weights per forward ≈ 18 GB/s) |
| 5 | row passes (norms, RoPE, activations, quantization) | 26% of the audio tower, 5-8% of vision, 11% of text | memory-bound streaming passes | memory/latency; already fused per row |
| 6 | concurrency: one engine pool and lock per server on CPU | media requests run one at a time | — | serialization (each request already uses every core; text is interleaved between tower layers since `e114de5`) |
| 7 | resident memory: repacked weights are anonymous memory, workspaces persist | +4-19% peak RSS vs upstream (t512 350 vs 322 MB, i280 539 vs 454, i1120 746 vs 720, a30 748 vs 654, m_tia 936 vs 808) | — | capacity, not speed |

What is no longer a bottleneck (was in upstream): scalar media towers
(rank 1-2 above, 25-48x), serial activation quantization, the per-element
type switch in convolutions, single-threaded log-mel (35 → 3 ms for 30 s on
12 E-cores), single-threaded weight repack at first use, short text losing
to E-core stragglers at 28 threads, and head-of-line blocking of text behind
media.

Next levers, in order of expected gain: (1) int8 Q·Kᵀ with VNNI in the
attention kernels (4x MACs per instruction for half the attention work;
changes numerics, would need the retrieval suite and must stay opt-in);
(2) a hardware-counter session (perf on a host that allows it) to resolve
the remaining ~50% FMA gap in attention; (3) per-request media workspaces
or a second pool on the E-cores so concurrent media requests overlap.
