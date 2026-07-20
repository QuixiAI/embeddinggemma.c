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

Each run writes:

```text
perf/results/YYYY-MM-DD/<run-id>/run.json
perf/results/YYYY-MM-DD/<run-id>/results.jsonl
perf/results/YYYY-MM-DD/<run-id>/summary.md
```

`perf/results/` is ignored by git because timings are machine-specific. Copy the
summary and decision into the tracked status files when a result changes what we
keep.
