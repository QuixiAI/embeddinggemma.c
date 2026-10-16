#!/usr/bin/env python3
"""Compare CPU vision-tower soft tokens of the native graphs with the official
HF FP32 tower (EmbeddingGemma2Model.get_image_features) on identical pixels.

  .venv/bin/python scripts/compare_vision_towers.py bench/fixtures/image/i70_0.png --budget 70

Runs ../build/test_gemma2_media three ways (reference graph, fast graph,
fast graph with the HF gelu_tanh MLP activation) and the HF tower twice
(official gelu_pytorch_tanh, and QuickGELU patched in) to separate kernel
error from the activation-variant difference.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GG = ROOT / "models" / "gguf"


def cos_stats(a: np.ndarray, b: np.ndarray) -> dict:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    flat = float((a64 * b64).sum() / np.sqrt((a64 * a64).sum() * (b64 * b64).sum()))
    per = (a64 * b64).sum(1) / np.sqrt((a64 * a64).sum(1) * (b64 * b64).sum(1))
    return dict(cosine=round(flat, 7), min_token=round(float(per.min()), 7), mean_token=round(float(per.mean()), 7))


def native(image: str, prefix: Path, env: dict, threads: int, cpus: str) -> None:
    e = dict(os.environ, EI_THREADS=str(threads), EI_MEDIA_TEST_DUMP=str(prefix), EI_MEDIA_TEST_REPEAT="1",
             EI_MEDIA_TEST_MIN_COS="0", **env)
    cmd = ["taskset", "-c", cpus, str(ROOT.parent / "build/test_gemma2_media"), str(GG / "embeddinggemma-2-Q8_0.gguf"),
           str(GG / "mmproj-embeddinggemma-2-Q8_0.gguf"), "vision", image]
    out = subprocess.run(cmd, env=e, capture_output=True, text=True, check=True).stdout
    print("   native:", out.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--budget", type=int, default=280)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--cpus", default="16-27")
    args = ap.parse_args()
    work = ROOT / "reports" / "raw" / "vision_parity"
    work.mkdir(parents=True, exist_ok=True)
    stem = work / Path(args.image).stem
    native(args.image, stem.with_name(stem.name + "_quick"), {}, args.threads, args.cpus)
    native(args.image, stem.with_name(stem.name + "_tanh"), {"EI_GEMMA2_VISION_GELU": "tanh", "EI_MEDIA_TEST_REF": "0"},
           args.threads, args.cpus)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.sched_setaffinity(0, [int(c) for c in range(int(args.cpus.split("-")[0]), int(args.cpus.split("-")[1]) + 1)])
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    torch.set_num_threads(args.threads)
    proc = AutoProcessor.from_pretrained("google/embeddinggemma-2")
    model = AutoModel.from_pretrained("google/embeddinggemma-2", dtype=torch.float32, audio_config=None)
    model.eval()
    img = Image.open(args.image).convert("RGB")
    f = proc.image_processor([img], max_soft_tokens=args.budget, return_tensors="pt")
    with torch.inference_mode():
        hf = model.get_image_features(f["pixel_values"], f["image_position_ids"], return_dict=True).pooler_output[0].numpy()
        from transformers.activations import ACT2FN

        for layer in model.vision_tower.encoder.layers:
            layer.mlp.act_fn = ACT2FN["quick_gelu"]  # x * sigmoid(1.702 x)
        hf_quick = model.get_image_features(f["pixel_values"], f["image_position_ids"], return_dict=True).pooler_output[0].numpy()

    def load(suffix):
        p = Path(str(stem) + suffix)
        return np.fromfile(p, dtype=np.float32).reshape(-1, 512) if p.exists() else None

    nq_ref, nq_fast, nt_fast = load("_quick.ref.f32"), load("_quick.fast.f32"), load("_tanh.fast.f32")
    report = dict(image=args.image, budget=args.budget, tokens=int(hf.shape[0]),
                  hf_quick_vs_hf_tanh=cos_stats(hf_quick, hf),
                  ref_quick_vs_hf_quick=cos_stats(nq_ref, hf_quick), fast_quick_vs_hf_quick=cos_stats(nq_fast, hf_quick),
                  fast_quick_vs_ref_quick=cos_stats(nq_fast, nq_ref),
                  ref_quick_vs_hf_tanh=cos_stats(nq_ref, hf), fast_tanh_vs_hf_tanh=cos_stats(nt_fast, hf))
    print(json.dumps(report, indent=1))
    (Path(str(stem) + "_report.json")).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
