# Benchmark and evaluation artifacts

Every number in the repository README is generated from the files in this
directory (`scripts/render_results.py`, `scripts/plot_bench.py`); nothing is
typed in by hand. This page says which raw runs feed which table, which
policies were applied, and which anomalies are known.

## Provenance

| report | built from (`raw/`) | engine |
|---|---|---|
| `benchmarks.{json,csv}` — latency tables | `final_fork_5dd7dbf_{P8_t8,ALL_t28}.json` (CPU-Max, one guarded run per budget); `final_st_fp32_*`, `final_onnx_q8_*`, `m1_upstream_P8_t8*`, `final_upstream_{P8_t8_fast,ALL_t28}.json` (baselines, better of two runs per workload: `summarize_bench.py --best-of`) | `5dd7dbf` |
| `memory.{json,csv}` — peak RSS, a fresh server per workload | `final_mem_fork_5dd7dbf_*`, `final_mem_upstream_P8_t8.json` | `5dd7dbf` |
| `throughput.{json,csv}` — concurrent clients | `final_tput_*_5dd7dbf_*` and `final_tput_*_quixi_upstream_P8_t8*` (two guarded runs each, reported as a range) | `5dd7dbf` |
| `scaling.{json,csv}`, `plots/scaling.png` — thread budgets before the hybrid pool | `final_fork_1a3e206_{P8,P16,PE20,ALL}*.json` | `1a3e206` (see anomalies) |
| `quality_embeddings.{json,csv}` — cosine vs FP32 | embeddings saved with the latency runs (`*.npz`, not committed: regenerate with the latency runs) | `5dd7dbf` |
| `retrieval.{json,csv}` | embeddings cached by `eval_retrieval.py` (not committed) | `2cd784b`; outputs bit-identical to `5dd7dbf` |
| `quant_study.json` | `quant_study.py` on the FP32 model | — |
| `profile_P8.txt`, `profile_ALL.txt`, `attn_layout_P8.txt`, `hol_*` | `profile_final.sh`, `test_gemma2_attn bench`, `bench/hol_blocking.py` | `5dd7dbf` |

Commits after `5dd7dbf` change no inference arithmetic: they add the build
identity (`--version`), detect AVX-VNNI through CPUID instead of a compiler
builtin (same dispatch on the benchmark machine), fix non-x86 builds, and
change documentation and tooling. Embeddings of the current engine were
re-checked byte for byte against the `5dd7dbf` run on five workloads.

## Policies

* **Quiet-host guard** (`run_bench.py --quiet-cores 1.5`): before each
  workload the harness waits (up to 180 s) until processes other than the
  benchmark use at most 1.5 cores, measured from `/proc/stat` while the
  backend is idle, and records `foreign_cores` — CPU used by other processes
  during the measurement — per workload in the raw JSON.
* **Run policy**: CPU-Max has one guarded run per budget (no best-of);
  baselines are the better of two runs per workload, which can only favour
  them. Throughput is reported as the range over two runs.
* **Statistic**: warm p50 and p95 of 3-10 repetitions after one warm-up
  (30 s budget per workload).
* **Excluded runs** are in [`raw/excluded/`](raw/excluded/) with the reason;
  no report reads them.

## Known anomalies

* **Scaling runs record a later repository HEAD.** The P16, PE20 and ALL runs
  of the scaling set record `fcb176f` as the git revision because the
  repository moved while the queue was running; the server binary was the
  `1a3e206` build throughout. Evidence: those runs embed the 8-frame video as
  1,193 tokens, the count produced by upstream-style per-frame timestamps,
  which `fcb176f` removed (1,138 tokens since).
* **Tail latency at 28 threads.** CPU-Max's ALL-budget video runs contain a
  few slow repetitions (v4: 1.9, 4.5, 4.6 s among 1.2-1.3 s; text + video:
  8.6 and 4.9 s among ~2.6 s) with `foreign_cores` ≈ 0. Not root-caused;
  P8 runs show no such tails (p95 ≤ 1.12x p50).
* **Throughput at 28 threads, first guarded run**: t128 with 16 clients
  measured 45.5 req/s, the second run 75.0 req/s (P8: 73-75); both are kept
  and shown as a range. The concurrency phase is not covered by the per-item
  foreign-load measurement.
* **Retrieval subsets are small** (200 MSR-VTT videos, 120 Clotho clips) and
  Clotho's top-1 agreement with FP32 is ~80% for every backend; see the
  README's Quality section.
