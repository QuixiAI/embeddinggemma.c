# EmbeddingGemma-2-CPU-Max

**CPU-optimized multimodal inference for Google EmbeddingGemma 2.**

EmbeddingGemma 2 (744 M parameters: a 270 M text backbone, a 170 M vision
tower and a 300 M audio tower) maps text, images, audio, video frames and
mixed inputs into one shared 768-d embedding space. This repository is a
fork of [QuixiAI/embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c),
a dependency-free C inference server for the model, that makes the **whole
model — not only the text backbone — fast on CPUs**, plus a reproducible
benchmark and evaluation suite against the official PyTorch implementation,
the ONNX Community int8 export and the unmodified upstream engine.

It is an inference engine, not a new checkpoint: it runs the Q8_0 GGUF files
of [`ggml-org/embeddinggemma-2-GGUF`](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF)
unchanged.

## At a glance

Measured on one machine — Intel Core i7-14700HX (8 P-cores + 12 E-cores,
AVX2 + AVX-VNNI, no AVX-512), 16 GB, Ubuntu 24.04 — with engine commit
`5dd7dbf`, warm p50 latency over HTTP, identical inputs for every backend.
These are single-host results; they have not yet been reproduced elsewhere.

* **Faster than every other CPU backend on all 16 benchmark workloads**
  (text, images, audio, video, mixed): **2.46x geometric mean** over the
  fastest alternative per workload, each backend at its own best thread
  budget. Smallest margin: **1.58x** (a 1344x1920 image vs ONNX Runtime
  int8); largest: 4.4x (5 s of audio vs PyTorch FP32).
* **16-54x faster than the unmodified upstream engine** on images, audio
  and video at 8 P-core threads (text: 2.0-4.2x). Upstream runs the vision
  and audio towers in a scalar reference graph on CPU.
* **Same embeddings as the official model**: cosine ≥ 0.99914 to the
  Sentence Transformers FP32 output on every workload (ONNX int8:
  ≥ 0.99850), and the same retrieval quality on Flickr30k, MSR-VTT,
  SciFact and Clotho (98-99% identical top-1 results on the first three).

Full tables are in [Results](#results); every attempt, including the rejected
ones, is in [`cpu-max/reports/optimization_log.md`](cpu-max/reports/optimization_log.md).

## What was optimized

Upstream's SIMD CPU graph covers the text backbone only; its CPU vision and
audio towers are a scalar reference graph (32 s for one 280-token image on
8 P-cores). Each item below is one or more commits on top of upstream
`d8f9e33` (the list with commit IDs and files is [`CPU_MAX.md`](CPU_MAX.md)):

* **CPU vision and audio fast paths** — the reference graphs' math (Q8_0
  weights x Q8_0 activations in the integer GEMMs; fp32 attention, norms,
  softmax and convolutions) in fused, threaded passes. Towers 24-32x faster
  than upstream's reference graph on the same cores.
* **A repacked 256-bit AVX-VNNI Q8_0 GEMM** for CPUs without AVX-512
  (Alder Lake / Raptor Lake and other hybrid parts): 8-row interleaved
  weights, u8 activations with a precomputed offset correction, exact
  integer dot products; ~7x upstream's tile per P-core. Used by the text,
  vision and audio graphs.
* **Tiled online-softmax attention** for head widths 64-512 (GQA, sliding
  windows, packed sequences), as accurate as plain fp32; vision reads K/V as
  contiguous per-head planes (+11-14% attention throughput at 10,080
  patches).
* **A packed fp32 GEMM** for the BF16/F32-weight projections (4.7x).
* **A hybrid P/E-core aware thread pool** — P-cores first, the calling
  thread on a P-core, short text waves kept off the E-cores — so using all
  28 threads no longer slows short requests and speeds up larger media
  19-32% over 8 P-core threads.
* **Serving**: CPU media towers hand the engine to waiting text requests
  between layers (16-token texts sent during a 1120-token image: 5.5 s →
  9 ms median), the log-mel front end and the weight repacking run on the
  thread pool (cold first requests close to warm speed).
* **Memory**: GGUF pages of repacked weights are released (one resident
  copy: −28-39% peak RSS), compact repack metadata, streamed audio
  subsampling, chunked vision FFN, reused workspaces.
* **Official semantics by default.** Upstream deviates from the official
  implementation in three places (vision MLP activation, log-mel floor,
  timestamp text before every video frame); the fork follows the official
  model, measured below. `EI_GEMMA2_SEMANTICS=upstream` restores upstream's
  behaviour.

Every new path has an environment switch that restores upstream's
behaviour (see [`CPU_MAX.md`](CPU_MAX.md#flags)) and keeps the portable
fallbacks. **Not claimed here** (already in upstream): the SIMD packed-batch
text graph and its x86 kernels, the AVX-512 VNNI repacked GEMM, NEON/i8mm,
the spin thread pool, batching and caching, the HTTP API, and all GPU
backends (Metal, CUDA, ROCm, XPU), which this fork does not change.

## Build and run

Requirements: Linux x86-64, GCC or Clang, `make`. The engine has no
third-party dependencies.

```sh
git clone https://github.com/Yazeedx0/EmbeddingGemma-2-CPU-Max.git
cd EmbeddingGemma-2-CPU-Max
make                                   # upstream Makefile: -O2, runtime ISA dispatch -> build/embeddinggemma

# model files (1.0 GB, pinned revision, SHA-256 checked); needs `uv` for the hf CLI
(cd cpu-max && uv sync && scripts/download_assets.sh models)

./build/embeddinggemma --bind 127.0.0.1 --port 42666 --threads "$(nproc)" \
    --model cpu-max/models/gguf/embeddinggemma-2-Q8_0.gguf \
    --mmproj cpu-max/models/gguf/mmproj-embeddinggemma-2-Q8_0.gguf
```

```sh
curl -s http://127.0.0.1:42666/v1/embeddings -H 'Content-Type: application/json' \
    -d '{"model":"embeddinggemma-2","input":["task: search result | query: a dog on a beach"]}'
python3 cpu-max/examples/client.py --image photo.png --audio clip.wav --frames f01.png f02.png
```

The HTTP API (`/v1/embeddings`, `/api/embed`; content objects with text,
image, audio and video parts) is upstream's and documented in
[`README.upstream.md`](README.upstream.md). The server does not decode video
containers: send frames (`ffmpeg -i clip.mp4 -vf fps=1 f%02d.png`). Text is
embedded as given — add the task prefix (`task: search result | query: …`)
yourself. On a hybrid CPU use every hardware thread (`--threads $(nproc)`);
the pool places work on the P-cores first. `EI_GEMMA2_CPU_PREPACK=1` repacks
all towers at startup (+0.7 s) so even the first request runs at warm speed,
at the cost of keeping every tower resident.

## Results

Engine `5dd7dbf`; **P8** = 8 threads, one per P-core; **ALL** = 28 threads.
Native engines are measured end to end over HTTP (base64 media in the
request); Sentence Transformers (PyTorch FP32) and ONNX Runtime in-process,
which excludes HTTP and favours them. Before each workload the harness waits
until other processes use ≤ 1.5 cores and records the foreign CPU load during
the measurement; the fork has one run per budget, each baseline the better of
two runs. Raw data: [`cpu-max/reports/raw/`](cpu-max/reports/raw/); plots:
[`cpu-max/reports/plots/`](cpu-max/reports/plots/).

<!-- RESULTS:BEGIN -->
<!-- generated by cpu-max/scripts/render_results.py --fork fork_5dd7dbf; do not edit by hand -->

**Warm p50 latency, ms — 8 threads on the 8 P-cores (P8)**

| workload | this fork | Sentence Transformers FP32 | ONNX Community Q8 | QuixiAI upstream Q8 | fork vs fastest other, same budget |
|---|---:|---:|---:|---:|---:|
| text, 16 tokens | **8.6** | 20.4 | 36.8 | 17.4 | 2.03x |
| text, 128 tokens | **29.5** | 77.5 | 84.7 | 117 | 2.63x |
| text, 512 tokens | **134** | 250 | 292 | 556 | 1.87x |
| image, 70 tokens (336x480) | **152** | 341 | 1,368 | 3,909 | 2.24x |
| image, 280 tokens (672x960) | **942** | 1,829 | 3,028 | 32,082 | 1.94x |
| image, 1120 tokens (1344x1920) | **10,140** | 13,646 | 14,321 | 552,466 | 1.35x |
| audio, 5 s | **82.1** | 361 | 1,297 | 1,320 | 4.40x |
| audio, 15 s | **241** | 654 | 2,014 | 3,935 | 2.71x |
| audio, 30 s | **523** | 1,169 | 2,448 | 8,202 | 2.24x |
| video, 4 frames | **1,482** | 3,265 | 4,332 | 42,652 | 2.20x |
| video, 8 frames | **3,096** | 6,946 | 7,465 | 87,577 | 2.24x |
| video, 16 frames | **6,405** | 16,418 | 14,392 | 161,199 | 2.25x |
| text + image | **1,114** | 1,707 | 3,050 | 33,015 | 1.53x |
| text + audio | **258** | 635 | 1,997 | 3,991 | 2.46x |
| text + image + audio | **1,295** | 2,495 | 4,454 | 36,394 | 1.93x |
| text + video | **3,137** | 7,276 | 7,519 | 87,103 | 2.32x |

**Warm p50 latency, ms — all 28 hardware threads (ALL)**

| workload | this fork | Sentence Transformers FP32 | ONNX Community Q8 | QuixiAI upstream Q8 | fork vs fastest other, same budget |
|---|---:|---:|---:|---:|---:|
| text, 16 tokens | **8.9** | 28.2 | 68.0 | 21.9 | 2.47x |
| text, 128 tokens | **29.3** | 96.0 | 158 | 108 | 3.27x |
| text, 512 tokens | **128** | 344 | 408 | 398 | 2.69x |
| image, 70 tokens (336x480) | **139** | 419 | 1,824 | 2,548 | 3.03x |
| image, 280 tokens (672x960) | **783** | 3,709 | 3,294 | – | 4.21x |
| image, 1120 tokens (1344x1920) | **7,678** | 25,366 | 12,142 | – | 1.58x |
| audio, 5 s | **84.6** | 1,852 | 1,648 | 987 | 11.66x |
| audio, 15 s | **237** | 4,126 | 2,470 | 2,911 | 10.40x |
| audio, 30 s | **441** | 18,259 | 2,995 | – | 6.79x |
| video, 4 frames | **1,271** | 12,703 | 4,579 | – | 3.60x |
| video, 8 frames | **2,560** | 18,955 | 7,219 | – | 2.82x |
| video, 16 frames | **5,313** | 23,168 | 14,481 | – | 2.73x |
| text + image | **849** | 10,109 | 3,314 | – | 3.90x |
| text + audio | **254** | 6,109 | 2,384 | 2,901 | 9.37x |
| text + image + audio | **1,070** | 11,795 | 4,987 | – | 4.66x |
| text + video | **2,629** | 21,358 | 7,142 | – | 2.72x |

**Best thread budget per backend (P8 or ALL), p50 ms**

| workload | this fork | Sentence Transformers FP32 | ONNX Community Q8 | QuixiAI upstream Q8 | fork vs fastest other |
|---|---:|---:|---:|---:|---:|
| text, 16 tokens | **8.6 (P8)** | 20.4 (P8) | 36.8 (P8) | 17.4 (P8) | 2.03x vs upstream |
| text, 128 tokens | **29.3 (ALL)** | 77.5 (P8) | 84.7 (P8) | 108 (ALL) | 2.64x vs ST FP32 |
| text, 512 tokens | **128 (ALL)** | 250 (P8) | 292 (P8) | 398 (ALL) | 1.95x vs ST FP32 |
| image, 70 tokens (336x480) | **139 (ALL)** | 341 (P8) | 1,368 (P8) | 2,548 (ALL) | 2.46x vs ST FP32 |
| image, 280 tokens (672x960) | **783 (ALL)** | 1,829 (P8) | 3,028 (P8) | 32,082 (P8) | 2.34x vs ST FP32 |
| image, 1120 tokens (1344x1920) | **7,678 (ALL)** | 13,646 (P8) | 12,142 (ALL) | 552,466 (P8) | 1.58x vs ONNX Q8 |
| audio, 5 s | **82.1 (P8)** | 361 (P8) | 1,297 (P8) | 987 (ALL) | 4.40x vs ST FP32 |
| audio, 15 s | **237 (ALL)** | 654 (P8) | 2,014 (P8) | 2,911 (ALL) | 2.76x vs ST FP32 |
| audio, 30 s | **441 (ALL)** | 1,169 (P8) | 2,448 (P8) | 8,202 (P8) | 2.65x vs ST FP32 |
| video, 4 frames | **1,271 (ALL)** | 3,265 (P8) | 4,332 (P8) | 42,652 (P8) | 2.57x vs ST FP32 |
| video, 8 frames | **2,560 (ALL)** | 6,946 (P8) | 7,219 (ALL) | 87,577 (P8) | 2.71x vs ST FP32 |
| video, 16 frames | **5,313 (ALL)** | 16,418 (P8) | 14,392 (P8) | 161,199 (P8) | 2.71x vs ONNX Q8 |
| text + image | **849 (ALL)** | 1,707 (P8) | 3,050 (P8) | 33,015 (P8) | 2.01x vs ST FP32 |
| text + audio | **254 (ALL)** | 635 (P8) | 1,997 (P8) | 2,901 (ALL) | 2.50x vs ST FP32 |
| text + image + audio | **1,070 (ALL)** | 2,495 (P8) | 4,454 (P8) | 36,394 (P8) | 2.33x vs ST FP32 |
| text + video | **2,629 (ALL)** | 7,276 (P8) | 7,142 (ALL) | 87,103 (P8) | 2.72x vs ONNX Q8 |

Geometric mean over the 16 workloads: **2.46x** faster than the fastest other backend for each workload (minimum 1.58x).

**Final-embedding cosine vs the official Sentence Transformers FP32 model (same inputs)**

| workload | fork | fork_28t | onnx_q8 | upstream | upstream_no_timestamps | st_fp32_run1 |
|---|---:|---:|---:|---:|---:|---:|
| text, 16 tokens | 0.99980 | 0.99980 | 0.99993 | 0.99981 | – | 1.00000 |
| text, 128 tokens | 0.99974 | 0.99974 | 0.99989 | 0.99973 | – | 1.00000 |
| text, 512 tokens | 0.99977 | 0.99977 | 0.99993 | 0.99981 | – | 1.00000 |
| image, 70 tokens (336x480) | 0.99964 | 0.99964 | 0.99850 | 0.99867 | – | 1.00000 |
| image, 280 tokens (672x960) | 0.99978 | 0.99978 | 0.99962 | 0.99954 | – | 1.00000 |
| image, 1120 tokens (1344x1920) | 0.99986 | 0.99986 | 0.99928 | 0.99963 | – | 1.00000 |
| audio, 5 s | 0.99914 | 0.99914 | 0.99984 | 0.99856 | – | 1.00000 |
| audio, 15 s | 0.99946 | 0.99946 | 0.99985 | 0.99895 | – | 1.00000 |
| audio, 30 s | 0.99932 | 0.99932 | 0.99981 | 0.99883 | – | 1.00000 |
| video, 4 frames | 0.99989 | 0.99989 | 0.99850 | 0.87563 | 0.99979 | 1.00000 |
| video, 8 frames | 0.99989 | 0.99989 | 0.99885 | 0.94145 | 0.99983 | 1.00000 |
| video, 16 frames | 0.99987 | 0.99987 | 0.99871 | 0.96294 | – | 1.00000 |
| text + image | 0.99970 | 0.99970 | 0.99932 | 0.99956 | – | 1.00000 |
| text + audio | 0.99984 | 0.99984 | 0.99992 | 0.99980 | – | 1.00000 |
| text + image + audio | 0.99973 | 0.99973 | 0.99929 | 0.99955 | – | 1.00000 |
| text + video | 0.99982 | 0.99982 | 0.99884 | 0.95552 | 0.99977 | 1.00000 |
| **min / mean** | **0.99914** / 0.99970 | **0.99914** / 0.99970 | **0.99850** / 0.99938 | **0.87563** / 0.98300 | **0.99977** / 0.99980 | **1.00000** / 1.00000 |

**Retrieval quality (higher is better; `top1 agr.` = share of queries whose top hit equals the FP32 model's)**

| task | backend | R@1 | R@5 | R@10 | nDCG@10 | top-10 overlap vs FP32 | top1 agr. vs FP32 |
|---|---|---:|---:|---:|---:|---:|---:|
| flickr (5000 q / 1000 docs) | fork_official | 0.8630 | 0.9756 | 0.9872 | 0.9307 | 0.9782 | 0.9928 |
| flickr (5000 q / 1000 docs) | fork_upstream_semantics | 0.8634 | 0.9748 | 0.9876 | 0.9307 | 0.9706 | 0.9904 |
| flickr (5000 q / 1000 docs) | st_fp32 | 0.8636 | 0.9750 | 0.9872 | 0.9309 | 1.0000 | 1.0000 |
| clotho (521 q / 120 docs) | fork_official | 0.1305 | 0.3781 | 0.5298 | 0.3032 | 0.9056 | 0.8023 |
| clotho (521 q / 120 docs) | fork_upstream_semantics | 0.1344 | 0.3724 | 0.5605 | 0.3133 | 0.8902 | 0.7850 |
| clotho (521 q / 120 docs) | quixi_upstream | 0.1305 | 0.3551 | 0.5662 | 0.3120 | 0.8885 | 0.8100 |
| clotho (521 q / 120 docs) | st_fp32 | 0.1324 | 0.3743 | 0.5336 | 0.3025 | 1.0000 | 1.0000 |
| msrvtt (200 q / 200 docs) | fork_official | 0.6350 | 0.8450 | 0.8850 | 0.7623 | 0.9760 | 0.9800 |
| msrvtt (200 q / 200 docs) | fork_upstream_semantics | 0.6100 | 0.8400 | 0.8850 | 0.7465 | 0.6745 | 0.7500 |
| msrvtt (200 q / 200 docs) | st_fp32 | 0.6200 | 0.8550 | 0.8850 | 0.7564 | 1.0000 | 1.0000 |
| scifact (300 q / 5183 docs) | fork_official | 0.7700 | 0.9100 | 0.9467 | 0.8521 | 0.9790 | 0.9833 |
| scifact (300 q / 5183 docs) | fork_upstream_semantics | 0.7700 | 0.9100 | 0.9467 | 0.8521 | 0.9790 | 0.9833 |
| scifact (300 q / 5183 docs) | quixi_upstream | 0.7733 | 0.9100 | 0.9500 | 0.8539 | 0.9780 | 0.9833 |
| scifact (300 q / 5183 docs) | st_fp32 | 0.7733 | 0.9067 | 0.9467 | 0.8514 | 1.0000 | 1.0000 |

**Peak resident memory of a fresh server per workload (MB, P8)**

| backend | t512 | i280 | i1120 | a30 | v16 | m_tia |
|---|---:|---:|---:|---:|---:|---:|
| fork_5dd7dbf | 350 | 539 | 746 | 748 | 759 | 936 |
| fork_5dd7dbf_norelease | 494 | 838 | 1,044 | 1,203 | 1,057 | 1,546 |
| fork_5dd7dbf_prepack | 850 | 875 | 1,084 | 915 | 1,099 | 941 |
| quixi_upstream | 322 | 454 | 720 | 654 | 700 | 808 |

**Throughput with concurrent clients (requests/s, distinct inputs; range over the runs)**

| backend | budget | workload | clients | req/s |
|---|---|---|---:|---:|
| fork_5dd7dbf | ALL | a15 | 2 | 2.63-4.23 |
| fork_5dd7dbf | P8 | a15 | 2 | 4.04-4.06 |
| quixi_upstream | P8 | a15 | 2 | 0.24-0.25 |
| fork_5dd7dbf | ALL | i70 | 2 | 5.90-6.95 |
| fork_5dd7dbf | P8 | i70 | 2 | 6.11-6.26 |
| quixi_upstream | P8 | i70 | 2 | 0.24-0.25 |
| fork_5dd7dbf | ALL | t128 | 4 | 34.65-35.38 |
| fork_5dd7dbf | P8 | t128 | 4 | 33.19-36.22 |
| quixi_upstream | P8 | t128 | 4 | 7.79-8.18 |
| fork_5dd7dbf | ALL | t128 | 16 | 45.52-74.95 |
| fork_5dd7dbf | P8 | t128 | 16 | 73.18-74.83 |
| quixi_upstream | P8 | t128 | 16 | 16.85-17.39 |
| fork_5dd7dbf | ALL | t16 | 4 | 168.82-168.94 |
| fork_5dd7dbf | P8 | t16 | 4 | 151.27-157.16 |
| quixi_upstream | P8 | t16 | 4 | 61.71-63.70 |
| fork_5dd7dbf | ALL | t16 | 16 | 451.54-452.90 |
| fork_5dd7dbf | P8 | t16 | 16 | 410.25-472.45 |
| quixi_upstream | P8 | t16 | 16 | 119.41-138.46 |

**Weight precision vs quality** (FP32 model with its linear weights rounded per 32-block; cosine of the final embedding against the unrounded model)

| scheme | text | vision | audio | min cosine | worst workload |
|---|---|---|---|---:|---|
| q8_all | q8 | q8 | q8 | 0.99932 | a5 |
| q4_text | q4 | q8 | q8 | 0.96436 | t128 |
| q4_vision | q8 | q4 | q8 | 0.98767 | i70 |
| q4_audio | q8 | q8 | q4 | 0.93288 | a5 |
| q4_all | q4 | q4 | q4 | 0.91563 | a15 |

<!-- RESULTS:END -->

## Correctness and retrieval quality

* **Final embeddings** match the official Sentence Transformers FP32 model
  to cosine ≥ 0.99914 (mean 0.99970) on identical inputs; the engine keeps
  upstream's precision (Q8_0 weights, Q8_0 activations into the integer
  GEMMs, everything else fp32). The changes after the first fast paths are
  bit-identical in output: the final engine's embeddings equal those of
  `97fdcc7` byte for byte on all 16 workloads, at 8 and 28 threads.
* **Upstream's semantics** cost it up to 0.12 cosine on video (timestamp text
  before each frame) and change rankings: with them, only 75% of MSR-VTT
  top-1 results agree with the FP32 model's (fork: 98%). Indexes built with
  upstream stay comparable with `EI_GEMMA2_SEMANTICS=upstream`, which
  reproduces upstream's embeddings to cosine ≥ 0.9992.
* **High cosine alone does not prove equal retrieval**, so retrieval is
  measured directly on labelled data (Flickr30k 1k test with 5,000
  captions; a 200-video subset of MSR-VTT 1k-A; a 120-clip subset of Clotho;
  SciFact), including agreement of each backend's rankings with FP32's.
  Upstream's scalar media towers were too slow for Flickr and MSR-VTT;
  there the fork with upstream semantics stands in.
* **Weight precision**: a fake-quantization study on the FP32 model shows Q8_0
  weights cost at most 0.0007 cosine, 4-bit weights 0.036 (text) to 0.084
  (all towers) — so no lower-precision mode is offered.
* **Tests**: kernels against float64 references (`src/test_gemma2_vnni256.c`,
  `src/test_gemma2_attn.c`), fast towers against the reference towers
  (`src/test_gemma2_media.c`), the whole server with upstream's own suite
  (`make test-gemma2`), and responses under concurrent mixed load against the
  same requests sent alone (`cpu-max/bench/stress_mixed.py`).

## Limitations and trade-offs

* **One test machine.** All numbers come from a single Intel i7-14700HX
  (AVX-VNNI, no AVX-512). CPUs with AVX-512 VNNI use upstream's repacked
  GEMM for the integer GEMMs; ARM and pre-AVX2 x86 run the portable
  fallbacks. Neither was built or measured here, and the GPU backends were
  not built either.
* **Memory**: peak RSS is 4-19% above upstream's (repacked weights are
  anonymous memory where upstream reads the mapped GGUF), e.g. 936 vs
  808 MB for text + image + audio (fresh server per workload). For scale:
  over a whole benchmark run (all workloads in one process) PyTorch FP32
  peaks at ~4.8 GB and ONNX Runtime at 4.8-5.3 GB.
* **Concurrency**: on the CPU backend media requests run one at a time
  (each already uses every core); text requests are interleaved between
  tower layers, which slows a running media request by the text work done.
* **Remaining bottlenecks** (ranked, with evidence, in
  [`cpu-max/reports/bottlenecks.md`](cpu-max/reports/bottlenecks.md)):
  vision attention (84% of a 1120-token image, fp32 FMA at about half the
  issue rate), text attention for long inputs, the Q8 GEMMs.
* **Embedding compatibility**: the default (official semantics) differs from
  upstream's output for images, audio and especially video; see above.

## Reproduce the benchmarks

Everything beyond the engine lives in [`cpu-max/`](cpu-max/) (Python 3.12 via
[uv](https://docs.astral.sh/uv/), CPU-only PyTorch; `ffmpeg` and `unzip` on
the system). Downloads are pinned to the revisions used for the results.

```sh
cd cpu-max
uv sync                                    # Sentence Transformers, Transformers, ONNX Runtime (CPU)
scripts/download_assets.sh                 # models + datasets (~5 GB; MSR-VTT archive 2.2 GB)
scripts/setup_upstream.sh                  # unmodified upstream d8f9e33 as a git worktree, built
(cd .. && make build/embeddinggemma build/test_gemma2_media build/test_gemma2_vnni256 \
           build/test_gemma2_attn build/bench_gemma2_cpu)
.venv/bin/python scripts/hw_detect.py      # reports/hardware.json
.venv/bin/python scripts/make_fixtures.py  # bench/fixtures/ (identical inputs for every backend)

scripts/run_final_matrix.sh fork fork_all st onnx onnx_all_media upstream_fast memory throughput tput_all
scripts/run_upstream_P8.sh                 # upstream's scalar media towers at P8 (about an hour)
.venv/bin/python scripts/summarize_bench.py --best-of reports/raw/final_fork_<commit>_{P8_t8,ALL_t28}.json \
    reports/raw/final_st_fp32_*.json reports/raw/final_onnx_q8_*.json reports/raw/m1_upstream_P8_t8*.json \
    reports/raw/final_upstream_{P8_t8_fast,ALL_t28}.json --speedup-vs quixi_upstream --out reports/benchmarks
.venv/bin/python scripts/validate_embeddings.py --ref reports/raw/final_st_fp32_P8_t8.npz \
    --run fork=reports/raw/final_fork_<commit>_P8_t8.npz --out reports/quality_embeddings

.venv/bin/python scripts/prepare_retrieval_data.py && scripts/run_retrieval.sh   # about 3 hours
.venv/bin/python scripts/quant_study.py
scripts/profile_final.sh P8                # per-stage profiles
.venv/bin/python bench/hol_blocking.py --binary ../build/embeddinggemma --threads 8 --cpus P8 \
    --media i280 --text t16 --out reports/raw/hol.json
.venv/bin/python scripts/render_results.py --fork fork_<commit> > reports/results.md
.venv/bin/python scripts/plot_bench.py --fork fork_<commit> --budget P8
```

The benchmark host must be otherwise idle; `bench/run_bench.py --quiet-cores`
waits for that and records the foreign load per measurement. Budgets
(`P8`, `ALL`, …) are CPU lists for this machine (`bench/run_bench.py`,
`BUDGETS`); adapt them to other CPUs.

## Repository layout

| path | contents |
|---|---|
| `src/`, `Makefile`, `testdata/`, `perf/`, `scripts/` | the engine: upstream's tree plus the CPU-Max commits (`CPU_MAX.md`) |
| `README.upstream.md`, `CONTRIBUTING.md`, `EXTRACTION.md`, … | upstream's documentation, unchanged |
| `cpu-max/bench/` | benchmark harness (backends, latency/throughput/memory runs, head-of-line probe, concurrency stress) |
| `cpu-max/scripts/` | asset downloads, fixtures, run matrices, retrieval and quality evaluation, summaries, plots |
| `cpu-max/reports/` | results (JSON/CSV), optimization log, bottleneck analysis, plots, raw per-run data |
| `cpu-max/examples/` | minimal HTTP client |

## Relationship to upstream

This is an official GitHub fork of
[QuixiAI/embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c),
based on its commit `d8f9e33`, with upstream's full history preserved.
Every engine change is a separate commit on top of it
(`git log d8f9e33..HEAD`), the upstream source tree keeps its layout, and
its README is kept as [`README.upstream.md`](README.upstream.md). The unmodified
upstream engine is the main baseline of every benchmark here. This is an
independent project, not affiliated with or endorsed by QuixiAI or Google.

## License and acknowledgments

* The engine is © QuixiAI under the MIT license ([`LICENSE`](LICENSE)); the
  changes and the `cpu-max/` suite in this fork are released under the same
  license.
* **EmbeddingGemma 2** is by Google DeepMind and is subject to the
  [Gemma terms of use](https://ai.google.dev/gemma/terms); model weights are
  not included in this repository. The GGUF conversion is by
  [ggml-org](https://huggingface.co/ggml-org), the int8 ONNX export by the
  [ONNX Community](https://huggingface.co/onnx-community).
* `stb_image` (public domain / MIT) is vendored by upstream.
* Baselines and evaluation use [Sentence Transformers](https://www.sbert.net/),
  [Transformers](https://github.com/huggingface/transformers),
  [PyTorch](https://pytorch.org/) and [ONNX Runtime](https://onnxruntime.ai/).
* Evaluation data — Flickr30k (via `nlphuji/flickr_1k_test_image_text_retrieval`),
  Clotho (via `CLAPv2/Clotho`), MSR-VTT (via `friedrichor/MSR-VTT`) and SciFact
  (via `mteb/scifact`) — is downloaded by `cpu-max/scripts/download_assets.sh`,
  not redistributed, and keeps its own license.
