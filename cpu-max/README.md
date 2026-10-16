# cpu-max/: benchmark, evaluation and reports

Everything in this fork that is not the engine itself. The overview, results
and the full reproduction recipe are in the [repository README](../README.md);
the engine changes are listed in [`../CPU_MAX.md`](../CPU_MAX.md).

| path | what |
|---|---|
| `pyproject.toml`, `uv.lock` | Python 3.12 environment (`uv sync`): CPU-only PyTorch, Sentence Transformers, Transformers, ONNX Runtime |
| `scripts/download_models.sh` | the two GGUF files the engine needs (curl + checksum, no Python), into the server's default model directory or a given one |
| `scripts/download_assets.sh` | everything the benchmarks need: models, the FP32 and ONNX baselines, datasets, at the revisions used for the results |
| `examples/quickstart.py`, `examples/client.py` | smoke test of a running server (text, image, audio, mixed; standard library only) and a minimal client |
| `scripts/setup_upstream.sh` | the unmodified upstream engine (`d8f9e33`) as a git worktree in `third_party/upstream`, built |
| `scripts/make_fixtures.py` | the 16 benchmark workloads, pre-decoded at the official processor's geometry (`bench/fixtures/`) |
| `bench/run_bench.py`, `bench/backends.py` | latency / throughput / memory harness for the native engines (HTTP), Sentence Transformers and ONNX Runtime |
| `bench/hol_blocking.py`, `bench/stress_mixed.py` | text latency during media requests; concurrent mixed load vs solo responses (bit-exact) |
| `scripts/run_final_matrix.sh`, `run_upstream_P8.sh`, `run_retrieval.sh`, `profile_final.sh` | the measurement queues behind the published results |
| `scripts/validate_embeddings.py`, `eval_retrieval.py`, `prepare_retrieval_data.py`, `quant_study.py`, `compare_*_towers.py` | correctness, retrieval and precision evaluations |
| `scripts/summarize_bench.py`, `render_results.py`, `plot_bench.py` | summaries, README tables, plots |
| `reports/` | results (`*.json`, `*.csv`), `optimization_log.md`, `bottlenecks.md`, `plots/`, `raw/` per-run data (`raw/excluded/`: runs not used, with reasons); provenance and known anomalies in `reports/README.md` |

Downloaded models, datasets, fixtures, the upstream worktree, cached
embeddings and the virtual environment are git-ignored (`.gitignore`).
