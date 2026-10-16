#!/usr/bin/env bash
# Download the models and datasets used by the benchmarks and evaluations, at
# the exact revisions behind the published results. Nothing here is
# redistributed by this repository; every asset keeps its own license.
#   cpu-max/scripts/download_assets.sh [models] [datasets]     (default: both)
#
# models:   GGUF files for the native engines (cpu-max/models/gguf, checked against
#           cpu-max/models/SHA256SUMS), the official FP32 model and the ONNX
#           Community export (Hugging Face cache).
# datasets: SciFact, Flickr30k 1k test, 5 Clotho test shards (Hugging Face cache)
#           and MSR-VTT test-1k (cpu-max/data/msrvtt; the video archive is 2.2 GB).
# Needs the `hf` CLI (installed by `uv sync` in cpu-max/), unzip, sha256sum.
set -euo pipefail
cd "$(dirname "$0")/.."
HF=${HF:-.venv/bin/hf}
command -v "$HF" >/dev/null || HF=hf
what=("$@")
[ ${#what[@]} -eq 0 ] && what=(models datasets)

for w in "${what[@]}"; do
  case $w in
    models)
      "$HF" download ggml-org/embeddinggemma-2-GGUF --revision bfcd298762cc34d0357ece5ebdd31791a3a374d8 \
          embeddinggemma-2-Q8_0.gguf mmproj-embeddinggemma-2-Q8_0.gguf --local-dir models/gguf
      (cd models/gguf && sha256sum -c ../SHA256SUMS)
      "$HF" download google/embeddinggemma-2 --revision 914f7f89142e33e77833254d9c9b90c3cef7303b
      "$HF" download onnx-community/embeddinggemma-2-ONNX --revision daa72c51243991dfcaf9f9137d2c573d8f7790c0 \
          --include "*.json" --include "*.jinja" --include "onnx/model_quantized.onnx*" \
          --include "onnx/vision_encoder_quantized.onnx*" --include "onnx/audio_encoder_quantized.onnx*"
      ;;
    datasets)
      "$HF" download mteb/scifact --repo-type dataset --revision cf10ab6856b15b0e670ef8ae5dae4e266c12d035
      "$HF" download nlphuji/flickr_1k_test_image_text_retrieval --repo-type dataset \
          --revision c4a4cdc67b77fce148b45484a067957bf75ec4c3
      "$HF" download CLAPv2/Clotho --repo-type dataset --revision b491ad6569dba180ca60a0e2d17a1d6a0d5d9f4a \
          data/test/test-mnt_audio_clip_processed_datasets_Clotho_test_4907-mnt_audio_clip_processed_datasets_Clotho_test_5674-batch32.parquet \
          data/test/test-mnt_audio_clip_processed_datasets_Clotho_test_4941-mnt_audio_clip_processed_datasets_Clotho_test_5405-batch18.parquet \
          data/test/test-mnt_audio_clip_processed_datasets_Clotho_test_4962-mnt_audio_clip_processed_datasets_Clotho_test_5241-batch15.parquet \
          data/test/test-mnt_audio_clip_processed_datasets_Clotho_test_4966-mnt_audio_clip_processed_datasets_Clotho_test_5619-batch33.parquet
      mkdir -p data/msrvtt/videos
      "$HF" download friedrichor/MSR-VTT --repo-type dataset --revision c1af215a96934854f42683c19c51391aaee6f962 \
          msrvtt_test_1k.json MSRVTT_Videos.zip --local-dir data/msrvtt/download
      cp data/msrvtt/download/msrvtt_test_1k.json data/msrvtt/
      # Only the 1,000 test videos are extracted.
      python3 -c "import json; [print(m['video']) for m in json.load(open('data/msrvtt/msrvtt_test_1k.json'))]" \
          | sed 's|^|video/|' > data/msrvtt/list.txt
      unzip -q -o -j data/msrvtt/download/MSRVTT_Videos.zip $(cat data/msrvtt/list.txt) -d data/msrvtt/videos
      echo "msrvtt: $(ls data/msrvtt/videos | wc -l) videos"
      ;;
    *) echo "unknown asset group: $w (models | datasets)" >&2; exit 2 ;;
  esac
done
