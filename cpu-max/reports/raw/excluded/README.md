# Excluded runs

These raw results are **not** used by any report, table or plot. They are kept
for transparency (see `../../optimization_log.md`, sections 21-22):

| file(s) | why excluded |
|---|---|
| `*_contaminated.json` | First full run of the final engine. It overlapped restarts of other services on the shared host: the unchanged upstream binary measured 23.3 ms on t16 (17.4 ms otherwise), the fork's 28-thread t512 629 ms (128-132 ms otherwise). Re-run with the quiet-host guard. |
| `quant_study_invalid_empty_match.json` | First weight-quantization study: its module-name filter matched no layer inside the Sentence Transformers wrapper, so no weight was changed and every scheme reported cosine 1.0. Fixed and re-run (`../../quant_study.json`). |
| `baseline_*_P8_t8.json`, `prelim_summary.*` | Preliminary Sentence Transformers / ONNX runs taken while a stray server from an aborted run used ~1.6 cores; replaced by the final runs. |
| `smoke_*.json` | Harness smoke tests (one repetition, warm-up only), not measurements. |
