# EmbeddingGemma-2-CPU-Max

**CPU-optimized multimodal inference for Google EmbeddingGemma 2.**

A dependency-free C server that embeds text, images, audio, video frames and
mixed inputs into one 768-dimensional space with
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2). It is a
fork of [QuixiAI/embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c)
in which the whole model — not only the text backbone — runs as optimized
CPU code.

<table>
  <tr>
    <td align="center" width="33%"><h2>2.46×</h2>geometric-mean speedup over the fastest tested CPU alternative, per workload</td>
    <td align="center" width="33%"><h2>16 / 16</h2>workloads faster than every tested CPU backend</td>
    <td align="center" width="33%"><h2>0.99914</h2>minimum cosine similarity to the official FP32 embeddings</td>
  </tr>
</table>

<sub>Measured on one Intel Core i7-14700HX (8 P-cores + 12 E-cores, AVX2 + AVX-VNNI, no AVX-512)
against PyTorch FP32, ONNX Runtime INT8 and the unmodified upstream engine. Not yet reproduced on
other hardware.</sub>

> **Pre-release (`0.1.0-dev`).** No release binaries yet — build from source. The `v0.1.0`–`v0.3.1`
> tags in this repository are upstream releases.

## Performance

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="cpu-max/reports/plots/readme/speedup-vs-fastest-dark.svg">
  <img src="cpu-max/reports/plots/readme/speedup-vs-fastest-light.svg" width="780"
       alt="Horizontal bar chart: CPU-Max speedup over the fastest other tested CPU backend for each of 16 workloads, from 4.40x (5 s audio, vs PyTorch FP32) down to 1.58x (1120-token image, vs ONNX Runtime INT8); geometric mean 2.46x.">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="cpu-max/reports/plots/readme/speedup-vs-pytorch-onnx-dark.svg">
  <img src="cpu-max/reports/plots/readme/speedup-vs-pytorch-onnx-light.svg" width="780"
       alt="Grouped bar chart: CPU-Max speedup vs PyTorch FP32 and vs ONNX Runtime INT8 on six workloads: text 512 tokens 1.95x and 2.28x, image 280 tokens 2.34x and 3.87x, image 1120 tokens 1.78x and 1.58x, audio 30 s 2.65x and 5.55x, video 16 frames 3.09x and 2.71x, text + image + audio 2.33x and 4.16x.">
</picture>

**How it was measured.** Warm p50 latency on 16 fixed workloads (text, images, audio, video,
mixed) with identical, pre-decoded inputs for every backend. Each backend runs at its best tested
thread budget — 8 threads on the P-cores or all 28 hardware threads. CPU-Max is timed end to end
over HTTP; PyTorch and ONNX Runtime in-process, which favours them. Precision differs by design:
CPU-Max runs the Q8_0 GGUF model, PyTorch FP32, ONNX Runtime the INT8 export. A quiet-host guard
waits out background load and records it per measurement; CPU-Max gets one run, each baseline the
better of two.

At a fixed 8 P-core budget the geometric mean is 2.20× (minimum 1.35×); against the unmodified
upstream engine CPU-Max is 15.7–54× faster on images, audio and video, 2.0–4.2× on text.
**[Full benchmark report →](cpu-max/reports/BENCHMARKS.md)** — all 16 workloads, both thread
budgets, p95, memory, throughput and raw data.

## Quick start

Requires Linux x86-64, a C compiler, `make`, `curl` and `sha256sum`; the smoke test needs
`python3` (standard library only).

```sh
git clone https://github.com/Yazeedx0/EmbeddingGemma-2-CPU-Max.git
cd EmbeddingGemma-2-CPU-Max
make                                         # builds build/embeddinggemma
cpu-max/scripts/download_models.sh           # Q8_0 GGUF files (1.0 GB), SHA-256 verified

./build/embeddinggemma --bind 127.0.0.1 --port 42666 --threads "$(nproc)" &
python3 cpu-max/examples/quickstart.py --wait 120   # text, image, audio and mixed requests
```

```sh
curl -s http://127.0.0.1:42666/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-2","input":["task: search result | query: a dog on a beach"]}'
```

Images, audio (16 kHz mono WAV) and video frames are sent as base64 content parts; see
[`cpu-max/examples/client.py`](cpu-max/examples/client.py) and the HTTP API in
[`README.upstream.md`](README.upstream.md). The server has no authentication — keep
`--bind 127.0.0.1` unless it sits behind your own network boundary. To install a binary instead,
`install.sh` builds from source and installs `embeddinggemma-cpu-max`
(`curl -fsSL https://raw.githubusercontent.com/Yazeedx0/EmbeddingGemma-2-CPU-Max/main/install.sh | sh`).

## What we optimized

- **CPU vision and audio towers.** Upstream runs them in a scalar reference graph on CPU (32 s for
  one 280-token image on 8 P-cores). CPU-Max executes the same arithmetic in fused, threaded passes:
  24–32× faster towers.
- **AVX-VNNI integer GEMM.** A repacked 256-bit Q8_0 kernel for CPUs with AVX-VNNI but no AVX-512,
  with exact integer dot products: ~7× upstream's kernel per P-core, used by all three towers.
- **Attention.** Tiled online-softmax attention for head widths 64–512 (GQA, sliding windows),
  as accurate as plain fp32, with contiguous per-head K/V streams for large images.
- **Hybrid P/E-core scheduling.** P-cores first, short requests kept off the E-cores, and media
  towers yield to waiting text requests between layers (a short query during a large image: 5.5 s →
  9 ms median wait).
- **Memory and startup.** One resident copy of repacked weights (−28 to −39% peak RSS), streamed
  audio subsampling, and parallel weight repacking so cold requests run near warm speed.

Every change is a separate commit on top of upstream with an environment switch that restores
upstream's behaviour — see [`CPU_MAX.md`](CPU_MAX.md).

## Quality and limitations

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="cpu-max/reports/plots/readme/retrieval-dark.svg">
  <img src="cpu-max/reports/plots/readme/retrieval-light.svg" width="780"
       alt="Grouped bar chart of Recall@1, CPU-Max Q8_0 vs official FP32: Flickr30k 0.863 vs 0.864, MSR-VTT 0.635 vs 0.620, SciFact 0.770 vs 0.773, Clotho 0.131 vs 0.132; same top-1 result as FP32 for 99.3%, 98.0%, 98.3% and 80.2% of queries.">
</picture>

- **Near-FP32, not identical.** Embedding cosine to the official FP32 model is ≥ 0.99914 (mean
  0.99970) on all 16 workloads. Retrieval recall matches FP32 within the noise of small evaluation
  subsets, but top-1 rankings agree for only 80.2% of Clotho audio queries (near-tied clips; upstream:
  81%). These subsets do not establish model-wide quality.
- **Not interchangeable with upstream embeddings.** CPU-Max follows the official model where
  upstream deviates (vision activation, log-mel floor, per-frame timestamps); upstream's video
  embeddings are as far as 0.876 cosine from the official ones. Don't mix them in one vector index —
  re-embed, or run with `EI_GEMMA2_SEMANTICS=upstream`.
- **Memory.** Peak RSS is 4–19% higher than upstream's per workload (repacked weights are anonymous
  memory). Over a whole benchmark run in one process, CPU-Max peaks at ~1.4 GB, PyTorch FP32 at
  ~4.8 GB and ONNX Runtime at 4.8–5.3 GB.
- **Serving.** Media requests are serialized on the CPU backend; each already uses every core.
- **Hardware.** One test machine. AVX-512, non-VNNI x86, ARM64 and macOS builds fall back to
  upstream's kernels; they compile in CI but have not been benchmarked. GPU backends are unchanged
  and untested here.

## Documentation

| | |
|---|---|
| [Benchmark report](cpu-max/reports/BENCHMARKS.md) | method, all tables (both thread budgets, p95), quality, memory, throughput, reproduction |
| [Raw data and provenance](cpu-max/reports/README.md) | which raw run feeds which result, run policies, known anomalies; per-run JSON in [`raw/`](cpu-max/reports/raw/) |
| [Optimization log](cpu-max/reports/optimization_log.md) | every optimization attempt, kept or rejected, with measurements |
| [Bottleneck analysis](cpu-max/reports/bottlenecks.md) | ranked remaining bottlenecks, before and after |
| [Implementation details](CPU_MAX.md) | engine changes by commit, flags, tests |
| [Upstream README](README.upstream.md) | engine architecture and HTTP API |
| [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) | testing and benchmark rules; vulnerability reporting |

## Attribution and license

CPU-Max is an independent fork of [QuixiAI/embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c)
(based on commit `d8f9e33`), © QuixiAI, MIT license ([`LICENSE`](LICENSE)); the CPU-Max changes are
released under the same license. It is not affiliated with or endorsed by QuixiAI or Google.
**EmbeddingGemma 2** is by **Google DeepMind** and subject to the
[Gemma terms of use](https://ai.google.dev/gemma/terms); no weights are included here. The GGUF
conversion is by [ggml-org](https://huggingface.co/ggml-org), the INT8 ONNX export by the
[ONNX Community](https://huggingface.co/onnx-community). Baselines use
[Sentence Transformers](https://www.sbert.net/), [PyTorch](https://pytorch.org/) and
[ONNX Runtime](https://onnxruntime.ai/); evaluation data (Flickr30k, Clotho, MSR-VTT, SciFact) is
downloaded from its sources, not redistributed.
