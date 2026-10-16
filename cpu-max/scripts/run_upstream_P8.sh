#!/usr/bin/env bash
# Unmodified upstream QuixiAI build on the P8 budget (same flags as the fork runs).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BIN=third_party/upstream/build/embeddinggemma
$PY bench/run_bench.py --backend native --binary $BIN --name quixi_upstream --threads 8 --cpus P8 \
    --items t16,t128,t512,i70,i280,a5,a15,a30,v4,v8,m_ti,m_ta,m_tia,m_tv \
    --warmup 1 --min-repeat 3 --repeat 10 --budget-s 30 --out reports/raw/m1_upstream_P8_t8.json 2>&1 | grep --line-buffered '^\['
$PY bench/run_bench.py --backend native --binary $BIN --name quixi_upstream --threads 8 --cpus P8 \
    --items i1120,v16 --warmup 0 --min-repeat 2 --repeat 2 --budget-s 1 \
    --out reports/raw/m1_upstream_P8_t8_slow.json 2>&1 | grep --line-buffered '^\['
echo "upstream done"
