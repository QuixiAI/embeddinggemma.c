#!/usr/bin/env bash
# Labeled retrieval quality per backend (scripts/eval_retrieval.py), P16 budget.
# Upstream is evaluated directly on clotho + scifact only (its scalar media
# towers would need ~9 h for the 1000 Flickr images); the fork with
# EI_GEMMA2_SEMANTICS=upstream reproduces upstream's embeddings to cosine
# >= 0.9992 on every benchmark workload and stands in for it elsewhere.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FORK=../build/embeddinggemma
UP=third_party/upstream/build/embeddinggemma
run() { echo "== $*"; "$@" 2>&1 | grep --line-buffered '^\[\|Traceback\|Error'; }
run $PY scripts/eval_retrieval.py --backend native --binary $FORK --name fork_official --threads 16 --cpus P16
run $PY scripts/eval_retrieval.py --backend native --binary $FORK --name fork_upstream_semantics --threads 16 --cpus P16 \
    --env EI_GEMMA2_SEMANTICS=upstream
run $PY scripts/eval_retrieval.py --backend native --binary $UP --name quixi_upstream --threads 16 --cpus P16 \
    --tasks clotho,scifact
run $PY scripts/eval_retrieval.py --backend st --name st_fp32 --threads 16 --cpus P16
$PY scripts/eval_retrieval.py --report > /dev/null
echo "retrieval done"
