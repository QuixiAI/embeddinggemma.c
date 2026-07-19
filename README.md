# embeddinggemma.c

Standalone C11 inference and HTTP serving for
`embeddinggemma-300M-qat-Q4_0.gguf`.

- GGUF v3 reader and SentencePiece tokenizer.
- Q4_0/Q8_0 inference on scalar CPU, ARM64 NEON, and x86 SIMD paths.
- Complete Apple Metal backend built as a standalone metallib.
- Model-specialized full/SWA attention, mean pooling, and L2 normalization.
- Matryoshka outputs at 768, 512, 256, or 128 dimensions.
- Token-budget dynamic batching with bounded lookahead, duplicate singleflight,
  and an exact-result LRU.
- Bounded HTTP, tokenizer, and inference queues for concurrent serving.
- `GET /api/tags` and `POST /api/embed` HTTP routes.

The CPU and Metal paths are implemented and parity-tested. CUDA, ROCm/HIP,
and XPU SYCL remain planned targets; they are not present in this tree yet.

## Build

Build the CPU server:

```sh
make
```

The output is `build/embeddinggemma`.

Build the Metal server and its colocated kernel library:

```sh
make metal
```

The outputs are `build/embeddinggemma-metal` and
`build/embeddinggemma.metallib`. The Metal build follows QuixiCore-Metal's
pattern: all `.metal` translation units are passed to one direct `xcrun metal`
invocation, producing one standalone metallib with stable function names.

The default model location is
`$XDG_CACHE_HOME/embeddinggemma.c/embeddinggemma-300M-qat-Q4_0.gguf`, or
`$HOME/.cache/embeddinggemma.c/embeddinggemma-300M-qat-Q4_0.gguf` when
`XDG_CACHE_HOME` is unset. The server creates the cache directory and downloads
the model when it is absent. Resolution order is `--model PATH`,
`EI_MODEL_PATH`, then the cache path.

## Backends

Select a backend on the Metal-capable server:

```sh
./build/embeddinggemma-metal --backend auto
./build/embeddinggemma-metal --backend metal
./build/embeddinggemma-metal --backend cpu
```

`auto` uses Metal when initialization succeeds and otherwise falls back to
CPU. Requesting `metal` fails instead of falling back. The CPU-only binary
supports `auto` and `cpu`; it reports an error for `metal`.

The retained CPU routes include NEON dot-product, AVX2, SSSE3, and scalar Q4_0
x Q8_0 kernels; SIMD quantization, norms, elementwise operations, and
attention; a persistent pthread pool; fused Q/K norm plus RoPE; triple QKV
projection; and adjacent three-row projection. `EI_THREADS` overrides the CPU
thread count. `EI_CPU_SHORT_THREADS` and `EI_CPU_DUAL_PROJECTION` are retained
diagnostic controls; their measured defaults are 6 and enabled. Fused RMS norm
plus Q8 activation quantization is enabled, and four-activation-row Q4 x Q8
projection is used from 512 total tokens upward.

CPU diagnostic controls:

- `EI_CPU_FUSED_RMS_QUANT=0`: materialize normalized activations before Q8.
- `EI_CPU_MULTIROW_MIN_TOKENS=0..65536`: disable or move the multirow boundary.
- `EI_CPU_FUSED_GELU_QUANT=1`: enable the rejected activation/Q8 experiment.

The retained Metal route uses one-row direct Q4 GEMV at T=1..6 and four-row
direct Q4 GEMV at T=7..2048. Fused QKV and up/gate projections, fused Q/K norm
plus RoPE, precomputed RoPE tables, online GQA attention, fused residual norm,
and fused final norm/pooling are enabled. At sequence lengths of 1024 and above,
K/V are stored as FP16 while score, softmax, and output accumulation remain
FP32. The direct projection and norm/RoPE epilogues write FP16 V and K without
an extra conversion dispatch. Staged 32x8 and 16x16 GEMM kernels are kept for
diagnostics but are slower than R4 on the tested Apple M5 Max.

Metal diagnostic controls:

- `EI_METALLIB_PATH`: override the metallib path.
- `EI_METAL_FUSED_QK_ROPE=0`: use separate Q/K dispatches.
- `EI_METAL_GEMV_R4_MIN_TOKENS=1..64`: change the R1/R4 boundary.
- `EI_METAL_GEMM_MIN_TOKENS=1..65536`: enable staged GEMM at a threshold.
- `EI_METAL_GEMM_TILE_TOKENS=8|16`: select a staged GEMM tile.
- `EI_METAL_FP16_KV_MIN_TOKENS=1..65536`: move the per-sequence FP16 K/V
  boundary; values above 2048 disable it.
- `EI_METAL_FUSED_UP_GATE_GELU=1`: enable the rejected fused FFN experiment.
- `EI_METAL_FUSED_UP_GATE_ROWS=2|4`: select its row grouping.
- `EI_METAL_TRIPLE_QKV_GEMV=1`: enable the rejected short Q/K/V experiment.

These route defaults were selected on the machine recorded in
`perf/optimization_status.md`; re-benchmark before changing them for another
GPU generation.

## Xcode Setup

Metal compilation requires full Xcode and the separately installed Metal
Toolchain component. Command Line Tools alone are insufficient.

Select Xcode system-wide and complete first-launch setup:

```sh
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -runFirstLaunch
```

Install the Metal Toolchain component:

```sh
xcodebuild -downloadComponent MetalToolchain
```

Verify the selected developer directory, Xcode build, component status, and
compiler:

```sh
xcode-select -p
xcodebuild -version
xcodebuild -showComponent MetalToolchain -json
xcrun --find metal
xcrun metal --version
```

The component status must be `installed`. If command-line download cannot
reach Apple's component catalog, open `Xcode -> Settings -> Components` and
install `Metal Toolchain` under Other Components. Ensure `xcode-select` points
to that same Xcode installation afterward.

The Makefile also prefers `/Applications/Xcode.app/Contents/Developer` through
`DEVELOPER_DIR` when no explicit value is supplied, so a local build does not
depend on Command Line Tools being selected globally.

## Test

Run CPU tests and llama.cpp golden parity:

```sh
make test
```

Run Metal compilation, function/pipeline validation, llama.cpp golden parity,
production route parity, FP32/FP16 K/V parity, and diagnostic GEMM tile parity:

```sh
make test-metal
```

The tests compare core GGUF metadata and all 314 tensor descriptors with the
checked-in model manifest, plus tokenizer exact matches, quantized kernels,
full/SWA attention, fused kernels, and all 10 embedding goldens. Each backend's
release acceptance is cosine similarity >= 0.999 against llama.cpp. A separate
synthetic random-token CPU/Metal drift guard uses a 0.99 threshold; it is not the
llama.cpp acceptance test.

## Performance

The performance workflow is adapted from
`QuixiCore/QuixiCore-Metal/perf/perf.md`: warm up, benchmark realistic model
shapes, alternate A/B order, require parity, and retain only stable gains of at
least 3% for low-risk changes.

```sh
make perf
make perf-engine
make perf-engine-metal
make perf-concurrency
make perf-batch
make perf-tokenization

python3 perf/bench_kernels.py --preset comprehensive --iters 100
python3 perf/bench_engine.py --backend both --tokens 1,7,32,128,512,2048
python3 perf/bench_concurrency.py --backend both --tokens 32 \
  --concurrency 1,2,4,8,16,32
python3 perf/bench_concurrency.py --backend metal --tokens 1,1,1,513 \
  --concurrency 64 --min-requests 64
python3 perf/bench_dimensions.py --backend metal --encoding-format both
python3 perf/bench_http.py --backend metal --keepalive on --response-cache-mb 64
./build/perf_engine_metal --model model/embeddinggemma-300M-qat-Q4_0.gguf \
  --backend metal --tokens 512,1024,2048 --warmup 4 --iters 14 \
  --ab-metal-fp16-kv
```

Results are written beneath `perf/results/`. The optimization log, including
the original 21-pass loop, vLLM/TEI serving work, retained changes, and rejected
experiments, is in `perf/optimization_status.md`.

The concurrency harness submits unique cache-miss requests through the actual
inference service. One backend thread owns the mutable engine workspace and
packs queued short requests into flattened, nonpadded batches. Defaults are 64
requests, 4096 total tokens, and a 200 us collection window. Sequences over 512
tokens execute as oldest-request singletons because packing two 2048-token
embeddings was 2-3% slower and delayed the first completion. For a packable
oldest request, the scheduler scans at most eight batch windows and skips
entries that do not fit. The oldest entry always runs, so this reduces mixed
length queue fragmentation without starving long requests.

At 32-token inputs, measured concurrency 1 to 32 raises CPU throughput from
28.1 to 59.5 req/s (2.12x) and Metal from 216 to 391 req/s (1.81x). At one
token, the corresponding gains are 3.45x CPU and 13.1x Metal. Full-context
requests remain singleton batches and retain approximately serialized
throughput.

## Run

```sh
./build/embeddinggemma --bind 0.0.0.0 --port 11434 --backend cpu
```

Metal:

```sh
./build/embeddinggemma-metal --bind 0.0.0.0 --port 11434 --backend metal
```

Serving controls:

- `--workers N` and `--max-queue N`: bounded HTTP concurrency and backlog;
  defaults are 64 and 256.
- `--cache-entries N`: final-embedding LRU capacity; zero disables retention.
- `--max-batch-tokens N` and `--max-batch-requests N`: packed batch budgets.
- `--max-batch-sequence-tokens N`: largest sequence eligible for packing.
- `--max-client-batch-size N`: maximum array inputs per HTTP request; default 32.
- `--tokenizer-workers N`: persistent parallel tokenizer workers; default 8.
- `--batch-wait-us N`: maximum microbatch collection delay.
- `--keepalive-connections N`: maximum workers allowed to hold persistent HTTP
  connections; defaults to half of `--workers`.
- `--keepalive-max-requests N`: request limit per persistent connection;
  default 100.
- `--keepalive-timeout-ms N`: persistent-connection idle timeout; default 1000.
- `--response-cache-mb N`: byte capacity of the exact float-JSON response LRU;
  default 64 MiB and zero disables it.
- `EI_BATCH_LOOKAHEAD=0`: restore strict FIFO batch collection for diagnostics.
- `EI_ADAPTIVE_BATCH_WAIT=0`: always apply the configured collection delay.
- `EI_HTTP_KEEPALIVE=0`: disable persistent HTTP connections for diagnostics.

The inference owner preallocates its token/output buffers, and the CPU or Metal
engine reserves the configured production workspace before the listener starts.
HTTP concurrency, client array size, and backend batch size remain separate
controls: larger admission capacity can tokenize the next work while one batch
runs without allowing an unbounded client request.

The embedding cache key is the exact token-ID sequence. Concurrent duplicates coalesce
onto one backend execution, and completed canonical 768-dimensional embeddings
are retained in the LRU. All output dimensions reuse that entry; dimensions are
not part of the cache key. A separate server-level LRU stores successful
`encoding_format="float"` response bodies by exact request bytes, avoiding JSON
parsing, tokenization, inference lookup, prefix normalization, and float
serialization on repeated HTTP requests. Base64 and error responses bypass this
second cache.
Transformer prefix KV caching is intentionally absent: EmbeddingGemma uses
bidirectional attention, so a cached prefix computed without later tokens is
not a valid prefix state for a longer request. Exact whole-result caching is
safe.

Example request:

```sh
curl -sS -X POST http://127.0.0.1:11434/api/embed \
  -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-300m","input":["search_query: what powers the cell"],"dimensions":256}'
```

`dimensions` is optional and must be one of `768`, `512`, `256`, or `128`; it
defaults to `768` and applies to every item in an input array. Reduced outputs
are the leading Matryoshka prefix of the canonical embedding, L2-normalized
again so every returned vector has unit norm. The example response is
`{"embeddings":[[256 floats]]}`.

`encoding_format` is also optional. Its default, `"float"`, preserves the JSON
array response above. `"base64"` returns each embedding as a base64 string
containing exactly `dimensions * 4` little-endian IEEE-754 float32 bytes:

```sh
curl -sS -X POST http://127.0.0.1:11434/api/embed \
  -H 'Content-Type: application/json' \
  -d '{"input":["search_query: what powers the cell"],"dimensions":256,"encoding_format":"base64"}'
```

Base64 is useful for clients that can consume packed float32 directly. At 768
dimensions it reduced payload size by 61%, made cached batch-32 responses about
10x faster to format, and improved cached concurrency-32 throughput by
1.12x-1.14x. It does not shorten cache-miss model inference.

## Design Documents

Keep both design documents. `EXTRACTION.md` records exact llama.cpp algorithm
and constant provenance needed to audit parity. `PORT_SPEC.md` defines the
fixed model graph, serving contract, backend scope, and remaining port work.
