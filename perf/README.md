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
