#!/usr/bin/env bash
# Final measurements of the fork at its current HEAD (run on an idle host):
# latency at P8 and ALL, memory, throughput, stage profiles, the attention
# layout A/B and head-of-line blocking on the P-cores. Baselines (ST, ONNX,
# upstream) are not re-run: their binaries and inputs did not change.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
(cd .. && make -s -j8 build/embeddinggemma build/test_gemma2_media build/bench_gemma2_cpu \
    build/test_gemma2_attn build/test_gemma2_vnni256)
uptime
scripts/run_final_matrix.sh fork fork_all memory_fork throughput tput_all
uptime
scripts/profile_final.sh P8
scripts/profile_final.sh ALL
{
  echo "# attention layout, P8 (cpus 0,2,..,14), 2 rounds"
  for r in 1 2; do for n in 2520 10080; do for hm in 0 1; do
    echo -n "n=$n headmajor=$hm: "
    EI_ATTN_BENCH_HEADMAJOR=$hm EI_THREADS=8 taskset -c 0,2,4,6,8,10,12,14 ../build/test_gemma2_attn bench $n 64 12
  done; done; done
} > reports/raw/attn_layout_P8.txt
for media in i280 i1120 a30; do for y in 0 1; do
  $PY bench/hol_blocking.py --binary ../build/embeddinggemma --threads 8 --cpus P8 --media $media --text t16 \
      --env EI_GEMMA2_CPU_YIELD=$y --out reports/raw/hol_${media}_yield${y}_P8.json
done; done
uptime
echo "final head done"
