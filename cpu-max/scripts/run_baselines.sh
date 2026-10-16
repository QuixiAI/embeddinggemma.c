#!/usr/bin/env bash
# Baseline matrix: official FP32 (Sentence Transformers), ONNX Community int8, and the
# unmodified upstream QuixiAI build, one backend at a time on the same CPU budget.
#   scripts/run_baselines.sh [BUDGET=P8] [THREADS=8]
set -euo pipefail
cd "$(dirname "$0")/.."
BUDGET=${1:-P8}
THREADS=${2:-8}
PY=.venv/bin/python
COMMON=(--threads "$THREADS" --cpus "$BUDGET" --warmup 1 --min-repeat 3 --repeat 10 --budget-s 30)
mkdir -p logs reports/raw
stamp=$(date +%Y%m%d-%H%M%S)
$PY bench/run_bench.py --backend st "${COMMON[@]}" \
    --out "reports/raw/baseline_st_fp32_${BUDGET}_t${THREADS}.json" 2>&1 | grep --line-buffered '^\[' | tee "logs/baseline_st_${stamp}.log"
$PY bench/run_bench.py --backend onnx "${COMMON[@]}" \
    --out "reports/raw/baseline_onnx_q8_${BUDGET}_t${THREADS}.json" 2>&1 | grep --line-buffered '^\[' | tee "logs/baseline_onnx_${stamp}.log"
$PY bench/run_bench.py --backend native --binary third_party/upstream/build/embeddinggemma \
    --name quixi_upstream "${COMMON[@]}" \
    --out "reports/raw/baseline_quixi_upstream_${BUDGET}_t${THREADS}.json" 2>&1 | grep --line-buffered '^\[' | tee "logs/baseline_upstream_${stamp}.log"
echo "baselines done"
