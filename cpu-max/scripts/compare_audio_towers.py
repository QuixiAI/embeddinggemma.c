#!/usr/bin/env python3
"""Compare CPU audio-tower soft tokens of the native graphs with the official
HF FP32 tower (EmbeddingGemma2Model.get_audio_features), all fed the same
log-mel features from the official HF feature extractor.

  .venv/bin/python scripts/compare_audio_towers.py bench/fixtures/audio/a15_0.wav
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
GG = ROOT / "models" / "gguf"


def cos_stats(a: np.ndarray, b: np.ndarray) -> dict:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    flat = float((a64 * b64).sum() / np.sqrt((a64 * a64).sum() * (b64 * b64).sum()))
    per = (a64 * b64).sum(1) / np.sqrt((a64 * a64).sum(1) * (b64 * b64).sum(1))
    return dict(cosine=round(flat, 7), min_token=round(float(per.min()), 7), mean_token=round(float(per.mean()), 7))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--cpus", default="16-27")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    lo, hi = (int(x) for x in args.cpus.split("-"))
    os.sched_setaffinity(0, list(range(lo, hi + 1)))
    import torch
    from transformers import AutoModel, AutoProcessor

    torch.set_num_threads(args.threads)
    work = ROOT / "reports" / "raw" / "audio_parity"
    work.mkdir(parents=True, exist_ok=True)
    stem = work / Path(args.wav).stem
    audio, sr = sf.read(args.wav, dtype="float32")
    proc = AutoProcessor.from_pretrained("google/embeddinggemma-2")
    f = proc.feature_extractor([audio], return_tensors="pt")
    feats, mask = f["input_features"], f["input_features_mask"]
    valid = int(mask[0].sum())
    feats[0, :valid].numpy().astype(np.float32).tofile(str(stem) + ".mel.f32")

    env = dict(os.environ, EI_THREADS=str(args.threads), EI_MEDIA_TEST_DUMP=str(stem), EI_MEDIA_TEST_REPEAT="1",
               EI_MEDIA_TEST_MIN_COS="0", EI_MEDIA_TEST_MIN_TOKEN_COS="0")
    out = subprocess.run(["taskset", "-c", args.cpus, str(ROOT.parent / "build/test_gemma2_media"),
                          str(GG / "embeddinggemma-2-Q8_0.gguf"), str(GG / "mmproj-embeddinggemma-2-Q8_0.gguf"),
                          "audio", str(stem) + ".mel.f32"], env=env, capture_output=True, text=True, check=True).stdout
    print("   native:", out.strip())

    model = AutoModel.from_pretrained("google/embeddinggemma-2", dtype=torch.float32, vision_config=None)
    model.eval()
    with torch.inference_mode():
        o = model.get_audio_features(feats, mask, return_dict=True)
        hf = o.pooler_output[0][o.attention_mask[0]].numpy()
    ref = np.fromfile(str(stem) + ".ref.f32", dtype=np.float32).reshape(-1, 512)
    fast = np.fromfile(str(stem) + ".fast.f32", dtype=np.float32).reshape(-1, 512)
    assert ref.shape == hf.shape == fast.shape, (ref.shape, hf.shape, fast.shape)
    report = dict(wav=args.wav, mel_frames=valid, tokens=int(hf.shape[0]), ref_vs_hf=cos_stats(ref, hf),
                  fast_vs_hf=cos_stats(fast, hf), fast_vs_ref=cos_stats(fast, ref))
    print(json.dumps(report, indent=1))
    (Path(str(stem) + "_report.json")).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
