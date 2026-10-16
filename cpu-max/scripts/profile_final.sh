#!/usr/bin/env bash
# In-process stage profiles of the fork's CPU graphs at a fixed budget
# (EI_GEMMA2_CPU_PROFILE=1 wall-clock timers; hardware counters are not
# available on the benchmark host). Output: reports/raw/profile_<budget>.txt
#   scripts/profile_final.sh P8      # 8 threads, one per P-core
#   scripts/profile_final.sh ALL     # 28 threads
set -euo pipefail
cd "$(dirname "$0")/.."
BUDGET=${1:-P8}
case $BUDGET in
  P8) CPUS=0,2,4,6,8,10,12,14; T=8 ;;
  ALL) CPUS=0-27; T=28 ;;
  E12) CPUS=16-27; T=12 ;;
  *) echo "budget: P8 | ALL | E12" >&2; exit 2 ;;
esac
B=../build
M=models/gguf
F=bench/fixtures
OUT=reports/raw/profile_${BUDGET}.txt
{
  echo "# $(date -Iseconds) engine $(git rev-parse --short HEAD) budget $BUDGET ($T threads on $CPUS)"
  echo "## vision tower (i70, i280, i1120 fixtures; 3 runs each, last lines per size)"
  EI_MEDIA_TEST_REF=0 EI_GEMMA2_CPU_PROFILE=1 EI_THREADS=$T taskset -c $CPUS $B/test_gemma2_media \
      $M/embeddinggemma-2-Q8_0.gguf $M/mmproj-embeddinggemma-2-Q8_0.gguf vision \
      $F/image/i70_0.png $F/image/i280_0.png $F/image/i1120_0.png 2>&1 | grep -E "vision-fast|^\{"
  echo "## audio tower (synthetic mel: 96, 499, 1499 frames)"
  EI_MEDIA_TEST_REF=0 EI_GEMMA2_CPU_PROFILE=1 EI_THREADS=$T taskset -c $CPUS $B/test_gemma2_media \
      $M/embeddinggemma-2-Q8_0.gguf $M/mmproj-embeddinggemma-2-Q8_0.gguf audio-synthetic 2>&1 | grep -E "audio-fast|^\{"
  echo "## text graph (tokens x batch); stage lines are per forward"
  EI_BENCH_FAST_ONLY=1 EI_GEMMA2_CPU_PROFILE=1 EI_THREADS=$T taskset -c $CPUS $B/bench_gemma2_cpu \
      $M/embeddinggemma-2-Q8_0.gguf 16x1 128x1 512x1 754x1 2048x1 2>&1
} > "$OUT"
echo "wrote $OUT"
