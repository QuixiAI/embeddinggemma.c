# Benchmarking

This directory holds kernel benchmark tooling and optimization notes for
`embeddinggemma.c`.

The operating guide is [`perf.md`](perf.md). Durable optimization notes belong in
[`optimization_status.md`](optimization_status.md), with retained baseline runs
indexed in [`baseline_status.md`](baseline_status.md).

Run the native harness from the repo root:

```sh
python3 perf/bench_kernels.py --preset smoke
python3 perf/bench_kernels.py --preset quick --kernel q4gemv,rms_norm
python3 perf/bench_concurrency.py --backend both --tokens 32 \
  --concurrency 1,2,4,8,16,32
python3 perf/bench_concurrency.py --backend metal --tokens 1,1,1,513 \
  --concurrency 64 --min-requests 64
EI_BATCH_LOOKAHEAD=0 python3 perf/bench_concurrency.py --backend metal \
  --tokens 1,1,1,513 --concurrency 64 --min-requests 64
python3 perf/bench_dimensions.py --backend metal --encoding-format both
python3 perf/bench_http.py --backend metal --keepalive on --response-cache-mb 64
python3 perf/bench_servers.py --url http://127.0.0.1:42666 \
  --api embeddinggemma --model embeddinggemma-300m
./build/perf_engine_metal --model model/embeddinggemma-300M-qat-Q4_0.gguf \
  --backend metal --tokens 512,1024,2048 --warmup 4 --iters 14 \
  --ab-metal-fp16-kv
./build/perf_batch_metal --model model/embeddinggemma-300M-qat-Q4_0.gguf \
  --backend metal --tokens 2048 --batch-sizes 1,2,4 \
  --max-total-tokens 8192 --ab-metal-fp16-kv
python3 perf/bench_engine.py --backend cuda --tokens 1,7,32,128,512,2048
python3 perf/bench_concurrency.py --backend cuda --tokens 32 \
  --concurrency 1,2,4,8,16,32
python3 perf/bench_engine.py --backend rocm --tokens 1,7,32,128,512,2048
python3 perf/bench_concurrency.py --backend rocm --tokens 32 \
  --concurrency 1,2,4,8,16,32
./build/perf_batch_rocm --model model/embeddinggemma-300M-qat-Q4_0.gguf \
  --backend rocm --tokens 32 --batch-sizes 1,2,4,8,16,32
./build/perf_engine_rocm --model model/embeddinggemma-300M-qat-Q4_0.gguf \
  --backend rocm --tokens 96,128,256,512,1024,2048 --warmup 7 --iters 31 \
  --ab-rocm-batched-attention
./build/perf_engine_rocm --model model/embeddinggemma-300M-qat-Q4_0.gguf \
  --backend rocm --tokens 1,8,16,24,31,32 --warmup 7 --iters 31 \
  --ab-rocm-direct-q4-pair
./build/perf_batch_rocm --model model/embeddinggemma-300M-qat-Q4_0.gguf \
  --backend rocm --tokens 1 --batch-sizes 16,32,48,64,72,80,96 \
  --warmup 7 --iters 31 --ab-rocm-singleton-direct
./build/perf_batch_rocm --model model/embeddinggemma-300M-qat-Q4_0.gguf \
  --backend rocm --tokens 1 --batch-sizes 1,2,4,8,16,32,64 \
  --warmup 7 --iters 31 --ab-rocm-singleton-metadata
./build/perf_batch_cuda --model model/embeddinggemma-300M-qat-Q4_0.gguf \
  --backend cuda --tokens 32 --batch-sizes 1,2,4,8,16,32
EI_CUDA_TENSOR_ATTENTION_MIN_TOKENS=65536 ./build/perf_engine_cuda \
  --model model/embeddinggemma-300M-qat-Q4_0.gguf --backend cuda \
  --tokens 256,512,1024,2048 --warmup 4 --iters 20
EI_CUDA_TENSOR_ATTENTION_MIN_TOKENS=1 ./build/perf_engine_cuda \
  --model model/embeddinggemma-300M-qat-Q4_0.gguf --backend cuda \
  --tokens 256,512,1024,2048 --warmup 4 --iters 20
EI_CUDA_NATIVE_Q4_GEMM=1 ./build/perf_engine_cuda \
  --model model/embeddinggemma-300M-qat-Q4_0.gguf --backend cuda \
  --tokens 7,32,128,512
EI_CUDA_DIRECT_FP16_QKV=0 ./build/perf_engine_cuda \
  --model model/embeddinggemma-300M-qat-Q4_0.gguf --backend cuda \
  --tokens 7,32,128,512,2048 --warmup 5 --iters 20
EI_CUDA_SWA_TENSOR_TILE_TOKENS=0 ./build/perf_engine_cuda \
  --model model/embeddinggemma-300M-qat-Q4_0.gguf --backend cuda \
  --tokens 2048 --warmup 5 --iters 20
make perf-batch
```

## Server Comparisons

`bench_servers.py` applies one HTTP workload to embeddinggemma, Ollama, vLLM, and
text-embeddings-inference. Start each server separately on the same otherwise
idle device, then run the matching adapter:

```sh
python3 perf/bench_servers.py --url http://127.0.0.1:42666 \
  --api embeddinggemma --model embeddinggemma-300m --label embeddinggemma
python3 perf/bench_servers.py --url http://127.0.0.1:11434 \
  --api ollama --model embeddinggemma --label ollama
python3 perf/bench_servers.py --url http://127.0.0.1:8000 \
  --api openai --model google/embeddinggemma-300m --label vllm
python3 perf/bench_servers.py --url http://127.0.0.1:8080 \
  --api openai --model google/embeddinggemma-300m \
  --label text-embeddings-inference
```

The OpenAI adapter covers both vLLM and TEI's `/v1/embeddings` endpoint. Every
timed string has a unique suffix to bypass exact-result caches. Keep dimensions,
generated word count, explicit batch size, concurrency sweep, warmup, and timed
rounds identical. Record server versions, model artifact/dtype, command lines,
driver, device power state, and competing GPU processes beside the summaries.
Do not compare runs from different hosts in a competitive ranking.

The equivalent Make target accepts configuration variables:

```sh
make perf-servers PERF_SERVER_URL=http://127.0.0.1:8000 \
  PERF_SERVER_API=openai PERF_SERVER_MODEL=google/embeddinggemma-300m
```

For a same-model, same-host Metal comparison with llama.cpp, build llama.cpp and
run:

```sh
make perf-compare-llamacpp
```

This sweeps exact total sequence lengths `8,16,32,64,128,256,512,1024,2048`
at concurrency `1,2,4,8,16,32`. It keeps both servers loaded, alternates which
server runs first for each shape, configures both for 32 concurrent clients,
disables result and prompt caches, uses unique token sequences within each
concurrent wave, and checks 768-float output parity before timing. llama.cpp uses
a 2,048-token embedding microbatch and a 4,096-token unified scheduling context;
embeddinggemma.c uses its tuned 4,096-token scheduler default.

The runner waits for a quiet host before each shape: over a sliding window of
`--quiet-stable-seconds` one-second samples, the mean aggregate CPU of
non-benchmark processes must stay below `--quiet-total-cpu-percent` (default
150%) and at most one sample may contain a single process at 50%+ CPU or 5%+
memory. The windowed mean tolerates one-second desktop blips (WindowServer,
idle VMs) that would otherwise stall the run for hours, while still blocking
on real transients such as compiles; steady background hum below the
threshold affects both servers equally because each cell measures the pair
back-to-back in alternating order. After each paired measurement the runner
re-samples twice, one second apart; if both samples show contention the
shape is discarded and repeated.

Battery power throttles the GPU about 40%, but it throttles both servers
equally, so paired ratios remain valid. The runner records the power source
in the results and summary, and discards any pair during which the power
source changed (the two servers measure sequentially, so a mid-cell plug or
unplug would skew one side).

A cell where llama.cpp measures faster is re-measured from scratch (behind
the quiet gate) up to `--loss-retries` times, default 3; only a winning pair
is accepted, its retry count is recorded in the results, and three
consecutive paired losses abort the run. This rescues cells from transient
sub-threshold host noise — measured true margins are 1.1x-1.4x on almost
every cell, so a genuine regression still fails all attempts — without ever
averaging a loss away.

`--fresh-servers` restarts both servers before every shape. Sustained
mixed-concurrency sweeps measurably degrade llama.cpp's throughput at high
slot counts (for example 8-token c32 fell from about 2,770 to about 590
embeddings/s late in one warm sweep), and its residual activity is excluded
from the quiet-host guard, so warm-server matrices both flatter our ratios
and can contaminate adjacent cells. Fresh servers give every cell each
engine's best sustained state; use it for published comparisons. It aborts
immediately after any accepted shape where llama.cpp has higher throughput;
investigate that route before restarting the sweep.

Each run writes:

```text
perf/results/YYYY-MM-DD/<run-id>/run.json
perf/results/YYYY-MM-DD/<run-id>/results.jsonl
perf/results/YYYY-MM-DD/<run-id>/summary.md
```

`perf/results/` is ignored by git because timings are machine-specific. Copy the
summary and decision into the tracked status files when a result changes what we
keep.
