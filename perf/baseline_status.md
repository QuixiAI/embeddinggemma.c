# Baseline Status

Method and measurement policy are described in [`perf.md`](perf.md). Raw
benchmark output should live under `perf/results/`; stable conclusions should be
copied into [`optimization_status.md`](optimization_status.md).

## Current Harness Index

| Area | Source | Notes |
|---|---|---|
| Native CPU kernel harness | `perf/harness/bench_kernels.c` | C benchmark cases for current kernel API |
| Shared runner | `perf/bench_kernels.py` | Builds/runs the native harness and writes `run.json`, `results.jsonl`, `summary.md` |
| Kernel notebook | `perf/optimization_status.md` | Running optimization log |

## 2026-07-19 CPU Quick Baseline

Device: Apple M5 Max, macOS 26.5.2, Apple clang 21.0.0. Route:
`cpu-neon-dotprod`. Command:

```sh
python3 perf/bench_kernels.py --preset quick --warmup 5 --iters 20
```

| Kernel | Shape | Median | Throughput |
|---|---:|---:|---:|
| Q4_0 x Q8_0 GEMV | 256 x 768 | 0.0127 ms | 30.9 GFLOP/s, 8.7 weight GB/s |
| Q4_0 x Q8_0 GEMV | 768 x 768 | 0.0381 ms | 31.0 GFLOP/s, 8.7 weight GB/s |
| Q4_0 x Q8_0 GEMV | 1152 x 768 | 0.0571 ms | 31.0 GFLOP/s, 8.7 weight GB/s |
| Q4_0 x Q8_0 GEMV | 768 x 1152 | 0.0571 ms | 31.0 GFLOP/s, 8.7 weight GB/s |
| Q8_0 dequant | 768 | 0.000043 ms | 90.6 GB/s |
| Q8_0 quantize | 768 | 0.000112 ms | 34.8 GB/s |
| RMSNorm | 768 | 0.000089 ms | 103.6 GB/s |
| GELU x up | 1152 | 0.00177 ms | 7.8 GB/s |
| full attention | T=128, H=3, D=256 | 1.432 ms | 35.2 GFLOP/s |

All Q4 dot/GEMV checks had zero error against the scalar integer reference.
Raw results: `perf/results/2026-07-19/000431-cpu-quick/` (ignored by git).
