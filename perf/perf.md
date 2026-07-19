# embeddinggemma.c Performance Handbook

This project has a small fixed kernel surface:

- `q4_0 x q8_0` GEMV for every projection.
- `q8_0` row dequantization for token embeddings.
- RMS norm at hidden sizes `256` and `768`.
- RoPE NEOX at head size `256`.
- GELU-gated FFN elementwise at `1152`.
- Attention softmax/value reduction over token counts up to `2048`.

Optimization should follow a controlled loop: establish correctness, measure a
baseline, change one bottleneck hypothesis at a time, and keep only measured
wins on realistic shapes.

## Commands

```sh
python3 perf/bench_kernels.py --preset smoke
python3 perf/bench_kernels.py --preset quick
python3 perf/bench_kernels.py --preset comprehensive --iters 100
python3 perf/bench_kernels.py --kernel q4dot,q4gemv --preset quick
python3 perf/bench_concurrency.py --backend both --tokens 2048 \
  --concurrency 1,2,4,8,16,32
python3 perf/bench_concurrency.py --backend metal --tokens 1,1,1,513 \
  --concurrency 64 --min-requests 64
python3 perf/bench_dimensions.py --backend metal --encoding-format both
python3 perf/bench_http.py --backend metal --keepalive on --response-cache-mb 64
./build/perf_batch --model model/embeddinggemma-300M-qat-Q4_0.gguf \
  --backend cpu --tokens 32 --batch-sizes 1,2,4,8,16,32
./build/perf_engine_metal --model model/embeddinggemma-300M-qat-Q4_0.gguf \
  --backend metal --tokens 512,1024,2048 --warmup 4 --iters 14 \
  --ab-metal-fp16-kv
make perf-tokenization
```

The Python entrypoint builds `build/perf_kernels`, runs it, and writes
schema-versioned result files under `perf/results/`.

## Measurement Rules

Every result should record:

- Git label and dirty state.
- OS, CPU, compiler, and backend.
- Kernel, variant, dtype, quant format, shape, warmups, iterations, median,
  p20/p80, coefficient of variation, correctness error, and raw result path.
- Derived throughput: packed-weight GB/s for quantized GEMV, conservative GB/s
  for memory-bound kernels, and GFLOP/s where the estimate is meaningful.

Use warmups. Small kernels must be batched before timing so clock and call
overhead do not dominate the measurement. The native C harness adapts each timed
sample to target at least 2 ms of work before dividing by the batch size.

## Kernel Metrics

For quantized GEMV, the main metric is effective packed-weight bandwidth:

```text
weight_GBps = packed_weight_bytes_read / seconds / 1e9
```

Use total conservative bytes for row kernels:

```text
rms_norm bytes ~= read x + read weight + write out
gelu_mul bytes ~= read/write gate + read up
q8_dequant bytes ~= read packed row + write f32 row
```

For GEMV FLOPs, report the dense-equivalent multiply-add count:

```text
GEMV FLOPs = 2 * N * K
```

## Shape Strategy

Prioritize this model's exact dimensions before generic sweep shapes:

- Projection GEMV: `(N,K)` = `(256,768)`, `(768,768)`, `(1152,768)`,
  `(768,1152)`.
- Dot-product inner sizes: `K=768`, `K=1152`.
- Norms: `hidden=256`, `hidden=768`.
- FFN gate: `n=1152`.
- RoPE: `heads=1`, `heads=3`, `head_dim=256`.
- Embedding dequant: `row=768`.

Add backend-specific cases under this same schema as Metal, CUDA, ROCm/HIP, and
SYCL kernels become callable.

## Concurrency Metrics

`bench_concurrency.py` drives the production inference service while excluding
HTTP parsing and model-load overhead. Every request uses distinct token IDs and
the service cache capacity is zero, so results measure dynamic batching rather
than cache hits or duplicate singleflight. Record aggregate requests/s, input
tokens/s, observed batch count/size, wall time, and p50/p95 request latency.

Measure at least three sequence regimes:

- T=1 for dispatch-dominated microrequests.
- T=32 for typical short embedding throughput.
- T=2048 to verify the long-sequence singleton policy.

Compare each concurrency level to concurrency 1 from the same run. Throughput
can increase while per-request latency also rises because requests complete at
batch granularity. The scheduler must never report a batch larger than the
configured request/token budgets.

Run concurrency levels in forward and reverse order. Long CPU runs can heat the
shared Apple SoC enough to bias a following Metal result; isolate backends when
comparing throughput. A flat requests/s curve with linearly growing tail
latency is a serialization result, not concurrent backend scaling. Full-context
requests intentionally have this shape because packed B=2 was slower on both
CPU and Metal.

Use `--ab-metal-fp16-kv` on the engine and packed-batch harnesses to create
FP32 and FP16 K/V engines in one process and alternate them every iteration.
This controls shared-SoC temperature better than comparing complete runs. Route
the production threshold by the longest sequence in a packed batch, not total
flattened tokens; attention bandwidth scales with each sequence's key length.

`bench_http.py` measures the production socket and serialization path. Pair
`--keepalive on|off` in both orders, and pair `--response-cache-mb 0|64` while
keeping keep-alive fixed. Its unique-input row is the cache-miss guard; cached
single, batch-32, and concurrency-32 rows measure response reuse.

Also measure mixed patterns such as `--tokens 1,513` and
`--tokens 1,1,1,513` at concurrency 64. Compare bounded lookahead with strict
FIFO by setting `EI_BATCH_LOOKAHEAD=0`. Record batch count and latency as well
as aggregate throughput: avoiding a blocked short request can be a meaningful
scheduler win even when long-sequence compute dominates wall time.

## Matryoshka Metrics

`bench_dimensions.py` measures 768, 512, 256, and 128-dimensional responses
through the production HTTP path. It separates cache-miss inference from cached
single responses, cached batch-32 formatting, and concurrent cached requests.
Record response bytes as well as latency and throughput: smaller dimensions do
not shorten the transformer graph, but they do reduce normalization,
serialization, copying, and network work after the canonical cache lookup.

Run both `float` and `base64` encoding formats. The base64 route measures packed
float32 serialization and transfer; decode it as little-endian IEEE-754 data.
Compare formats at the same dimension rather than treating encoding gains as a
Matryoshka gain. Cache-miss latency should remain flat because encoding occurs
after inference.

Keep the exact-result cache at 768 dimensions and normalize response prefixes in
caller-owned memory. A dimension-specific backend pool is only justified if an
end-to-end cache-miss benchmark exceeds the 3% retention threshold; isolated
pool savings do not justify cache fragmentation.

## Tokenization Metrics

`perf-tokenization` runs the real SentencePiece tokenizer through the inference
service with a zero-cost fake backend. This isolates tokenizer scheduling from
model execution, JSON formatting, and networking. Sweep worker count at both a
normal payload and a long payload; report batch latency and relative throughput
against the sequential route.

The HTTP client-array limit, active request admission, tokenizer worker count,
backend request cap, and token budget are independent controls. Benchmark them
independently. In particular, do not infer a useful backend batch size from the
32-input client limit, and do not raise tokenizer workers based only on an
isolated result when the CPU backend needs the same cores.
