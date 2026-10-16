#!/usr/bin/env bash
# Final measurement queue (run on an otherwise idle host; every phase is one
# backend in its own process on the same CPU budget). Raw JSON + embeddings go
# to reports/raw/final_*. Phases can be selected: scripts/run_final_matrix.sh fork st onnx ...
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FORK=../build/embeddinggemma                    # the fork, built at the repository root
UP=third_party/upstream/build/embeddinggemma    # unmodified upstream d8f9e33 (scripts/setup_upstream.sh)
TAG=$(git rev-parse --short HEAD)
# GUARD: before each item wait (bounded) until other processes use <= 1.5 cores; each
# item records the foreign CPU load during its measurement (foreign_cores).
GUARD=(--quiet-cores 1.5 --quiet-wait-s 180)
COMMON=(--warmup 1 --min-repeat 3 --repeat 10 --budget-s 30 "${GUARD[@]}")
FIDELITY=(--env EI_GEMMA2_VISION_GELU=tanh --env EI_GEMMA2_AUDIO_LOGMEL=hf --env EI_GEMMA2_VIDEO_TIMESTAMPS=0)
run() { echo "== $*"; "$@" 2>&1 | grep --line-buffered '^\['; }
phases=("$@")
[ ${#phases[@]} -eq 0 ] && phases=(fork fidelity upstream_hflayout memory throughput st onnx scaling)
for phase in "${phases[@]}"; do
  case $phase in
    fork)
      run $PY bench/run_bench.py --backend native --binary $FORK --name "fork_$TAG" --threads 8 --cpus P8 \
          "${COMMON[@]}" --out "reports/raw/final_fork_${TAG}_P8_t8.json" ;;
    fidelity)
      run $PY bench/run_bench.py --backend native --binary $FORK --name "fork_${TAG}_fidelity" --threads 8 --cpus P8 \
          "${COMMON[@]}" "${FIDELITY[@]}" --out "reports/raw/final_fork_${TAG}_fidelity_P8_t8.json" ;;
    upstream_hflayout)  # upstream's embeddings for video in the official layout (frames as image parts)
      run $PY bench/run_bench.py --backend native --binary $UP --name quixi_upstream_hflayout --threads 8 --cpus P8 \
          --items v4,v8,m_tv --video-as-images --warmup 0 --min-repeat 1 --repeat 1 \
          --out reports/raw/final_upstream_hflayout_P8_t8.json ;;
    st)
      run $PY bench/run_bench.py --backend st --threads 8 --cpus P8 "${COMMON[@]}" \
          --out reports/raw/final_st_fp32_P8_t8.json ;;
    onnx)
      run $PY bench/run_bench.py --backend onnx --threads 8 --cpus P8 "${COMMON[@]}" \
          --out reports/raw/final_onnx_q8_P8_t8.json ;;
    scaling)
      for b in "P16 16" "PE20 20" "ALL 28"; do
        read -r budget threads <<< "$b"
        run $PY bench/run_bench.py --backend native --binary $FORK --name "fork_$TAG" --threads "$threads" --cpus "$budget" \
            --items t16,t128,t512,i70,i280,i1120,a15,v8,m_tia "${COMMON[@]}" \
            --out "reports/raw/final_fork_${TAG}_${budget}_t${threads}.json"
      done ;;
    memory)  # per-workload peak RSS: a fresh server per item
      MEMI=t512,i280,i1120,a30,v16,m_tia
      run $PY bench/run_bench.py --backend native --binary $FORK --name "fork_$TAG" --threads 8 --cpus P8 \
          --items $MEMI --fresh-server --warmup 0 --min-repeat 1 --repeat 1 \
          --out "reports/raw/final_mem_fork_${TAG}_P8_t8.json"
      run $PY bench/run_bench.py --backend native --binary $FORK --name "fork_${TAG}_norelease" --threads 8 --cpus P8 \
          --items $MEMI --fresh-server --warmup 0 --min-repeat 1 --repeat 1 --env EI_GEMMA2_CPU_RELEASE_WEIGHTS=0 \
          --out "reports/raw/final_mem_fork_${TAG}_norelease_P8_t8.json"
      run $PY bench/run_bench.py --backend native --binary $UP --name quixi_upstream --threads 8 --cpus P8 \
          --items $MEMI --fresh-server --warmup 0 --min-repeat 1 --repeat 1 \
          --out reports/raw/final_mem_upstream_P8_t8.json ;;
    memory_fork)  # the fork's rows of the memory phase only (upstream's do not change)
      MEMI=t512,i280,i1120,a30,v16,m_tia
      run $PY bench/run_bench.py --backend native --binary $FORK --name "fork_$TAG" --threads 8 --cpus P8 \
          --items $MEMI --fresh-server --warmup 0 --min-repeat 1 --repeat 1 \
          --out "reports/raw/final_mem_fork_${TAG}_P8_t8.json"
      run $PY bench/run_bench.py --backend native --binary $FORK --name "fork_${TAG}_norelease" --threads 8 --cpus P8 \
          --items $MEMI --fresh-server --warmup 0 --min-repeat 1 --repeat 1 --env EI_GEMMA2_CPU_RELEASE_WEIGHTS=0 \
          --out "reports/raw/final_mem_fork_${TAG}_norelease_P8_t8.json"
      run $PY bench/run_bench.py --backend native --binary $FORK --name "fork_${TAG}_prepack" --threads 8 --cpus P8 \
          --items $MEMI --fresh-server --warmup 0 --min-repeat 1 --repeat 1 --env EI_GEMMA2_CPU_PREPACK=1 \
          --out "reports/raw/final_mem_fork_${TAG}_prepack_P8_t8.json" ;;
    throughput)  # concurrent clients, distinct inputs (no singleflight coalescing)
      for bin in "$FORK fork_$TAG" "$UP quixi_upstream"; do
        read -r b n <<< "$bin"
        run $PY bench/run_bench.py --backend native --binary "$b" --name "$n" --threads 8 --cpus P8 \
            --items t16,t128 --warmup 1 --min-repeat 3 --repeat 3 --concurrency 4,16 --requests 64 "${GUARD[@]}" \
            --out "reports/raw/final_tput_text_${n}_P8_t8.json"
        run $PY bench/run_bench.py --backend native --binary "$b" --name "$n" --threads 8 --cpus P8 \
            --items i70,a15 --warmup 1 --min-repeat 3 --repeat 3 --concurrency 2 --requests 6 "${GUARD[@]}" \
            --out "reports/raw/final_tput_media_${n}_P8_t8.json"
      done ;;
    fork_all)  # every core (28 threads; hybrid-aware pool)
      run $PY bench/run_bench.py --backend native --binary $FORK --name "fork_$TAG" --threads 28 --cpus ALL \
          "${COMMON[@]}" --out "reports/raw/final_fork_${TAG}_ALL_t28.json" ;;
    st_all)
      run $PY bench/run_bench.py --backend st --threads 28 --cpus ALL "${COMMON[@]}" \
          --out reports/raw/final_st_fp32_ALL_t28.json ;;
    onnx_all)
      run $PY bench/run_bench.py --backend onnx --threads 28 --cpus ALL "${COMMON[@]}" \
          --out reports/raw/final_onnx_q8_ALL_t28.json ;;
    onnx_all_media)  # ONNX Runtime's best budget for the largest media is ALL
      run $PY bench/run_bench.py --backend onnx --threads 28 --cpus ALL --items i280,i1120,v8,v16,m_tv "${COMMON[@]}" \
          --out reports/raw/final_onnx_q8_ALL_t28_media.json ;;
    upstream_fast)  # upstream at P8 on the items where it is a contender (text, short audio, small image)
      run $PY bench/run_bench.py --backend native --binary $UP --name quixi_upstream --threads 8 --cpus P8 \
          --items t16,t128,t512,i70,a5,a15,m_ta "${COMMON[@]}" --out reports/raw/final_upstream_P8_t8_fast.json ;;
    upstream_all)  # items that finish in minutes on upstream's scalar media towers
      run $PY bench/run_bench.py --backend native --binary $UP --name quixi_upstream --threads 28 --cpus ALL \
          --items t16,t128,t512,i70,a5,a15,m_ta "${COMMON[@]}" --out reports/raw/final_upstream_ALL_t28.json ;;
    tput_all)
      run $PY bench/run_bench.py --backend native --binary $FORK --name "fork_$TAG" --threads 28 --cpus ALL \
          --items t16,t128 --warmup 1 --min-repeat 3 --repeat 3 --concurrency 4,16 --requests 64 "${GUARD[@]}" \
          --out "reports/raw/final_tput_text_fork_${TAG}_ALL_t28.json"
      run $PY bench/run_bench.py --backend native --binary $FORK --name "fork_$TAG" --threads 28 --cpus ALL \
          --items i70,a15 --warmup 1 --min-repeat 3 --repeat 3 --concurrency 2 --requests 6 "${GUARD[@]}" \
          --out "reports/raw/final_tput_media_fork_${TAG}_ALL_t28.json" ;;
    *) echo "unknown phase $phase"; exit 2 ;;
  esac
done
echo "final matrix done"
