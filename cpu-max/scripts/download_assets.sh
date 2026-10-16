#!/usr/bin/env bash
# Download the models and datasets used by the benchmarks and evaluations, at
# the exact revisions behind the published results. Nothing here is
# redistributed by this repository; every asset keeps its own license.
#   cpu-max/scripts/download_assets.sh [models] [datasets]     (default: both)
#
# models:   GGUF files for the native engines (cpu-max/models/gguf, via
#           download_models.sh, checked against cpu-max/models/SHA256SUMS), the
#           official FP32 model and the ONNX Community INT8 export (Hugging Face cache).
# datasets: SciFact, Flickr30k 1k test (images extracted to cpu-max/data/flickr1k),
#           4 Clotho test shards (Hugging Face cache) and MSR-VTT test-1k
#           (cpu-max/data/msrvtt; the video archive is 2.2 GB).
# Needs the environment of `uv sync` in cpu-max/ (hf CLI, huggingface_hub), unzip, sha256sum.
set -euo pipefail
cd "$(dirname "$0")/.."
HF=${HF:-.venv/bin/hf}
command -v "$HF" >/dev/null || HF=hf
PY=${PY:-.venv/bin/python}
what=("$@")
[ ${#what[@]} -eq 0 ] && what=(models datasets)

for w in "${what[@]}"; do
  case $w in
    models)
      sh scripts/download_models.sh models/gguf     # GGUF files, pinned revision, SHA-256 checked
      "$HF" download google/embeddinggemma-2 --revision 914f7f89142e33e77833254d9c9b90c3cef7303b
      "$HF" download onnx-community/embeddinggemma-2-ONNX --revision daa72c51243991dfcaf9f9137d2c573d8f7790c0 \
          --include "*.json" --include "*.jinja" --include "onnx/model_quantized.onnx*" \
          --include "onnx/vision_encoder_quantized.onnx*" --include "onnx/audio_encoder_quantized.onnx*"
      ;;
    datasets)
      "$HF" download mteb/scifact --repo-type dataset --revision cf10ab6856b15b0e670ef8ae5dae4e266c12d035
      "$HF" download nlphuji/flickr_1k_test_image_text_retrieval --repo-type dataset \
          --revision c4a4cdc67b77fce148b45484a067957bf75ec4c3
      # Flickr30k 1k test images, extracted for fixtures and retrieval (data/flickr1k/images_flickr_1k_test)
      zip=$("$PY" -c "from huggingface_hub import hf_hub_download as d; print(d('nlphuji/flickr_1k_test_image_text_retrieval', 'images_flickr_1k_test.zip', repo_type='dataset', revision='c4a4cdc67b77fce148b45484a067957bf75ec4c3'))")
      mkdir -p data/flickr1k
      unzip -q -o "$zip" -d data/flickr1k
      echo "flickr: $(ls data/flickr1k/images_flickr_1k_test | wc -l) images"
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
