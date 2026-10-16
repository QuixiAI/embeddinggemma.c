#!/usr/bin/env python3
"""Weight-quantization quality study on the official FP32 model.

Fake-quantizes the linear weights of selected towers in the HF FP32 model with
block-32 symmetric schemes (q8: Q8_0-style, scale = max|w|/127; q4: Q4_0-style
symmetric 4-bit, scale = max|w|/7) and measures final-embedding cosine against
the unquantized model on the benchmark fixtures. Activations stay fp32, so the
result isolates the weight-precision cost of each mixed-precision scheme.

  .venv/bin/python scripts/quant_study.py --threads 16 --cpus 0-15
Writes reports/quant_study.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
ITEMS = ["t128", "t512", "i70", "i280", "a5", "a15", "v4", "m_tia"]
SCHEMES = {
    "q8_all": {"text": "q8", "vision": "q8", "audio": "q8"},
    "q4_text": {"text": "q4", "vision": "q8", "audio": "q8"},
    "q4_vision": {"text": "q8", "vision": "q4", "audio": "q8"},
    "q4_audio": {"text": "q8", "vision": "q8", "audio": "q4"},
    "q4_all": {"text": "q4", "vision": "q4", "audio": "q4"},
}


def tower(name: str) -> str | None:
    # Module names carry wrapper prefixes (e.g. "0.auto_model.model.audio_tower..."), so match components.
    parts = name.split(".")
    if "vision_tower" in parts or "embed_vision" in parts:
        return "vision"
    if "audio_tower" in parts or "embed_audio" in parts:
        return "audio"
    if "language_model" in parts and "embed_tokens" not in parts:
        return "text"
    return None


def fake_quant(w, scheme: str):
    import torch

    m, k = w.shape
    if k % 32:
        return w
    qmax = 127 if scheme == "q8" else 7
    b = w.reshape(m, k // 32, 32)
    scale = b.abs().amax(dim=2, keepdim=True) / qmax
    q = torch.where(scale > 0, torch.round(b / torch.where(scale > 0, scale, torch.ones_like(scale))), 0).clamp(-qmax, qmax)
    return (q * scale).reshape(m, k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--cpus", default="0-15")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.cpus.split("-"))
    os.sched_setaffinity(0, list(range(lo, hi + 1)))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from backends import STBackend, load_manifest, parts_of

    be = STBackend(args.threads)
    model = be.model
    items = {i["id"]: i for i in load_manifest()["items"]}
    prepared = {k: be.prepare(parts_of(items[k])) for k in ITEMS}

    def embed_all():
        return {k: be.run(prepared[k])[0][0] for k in ITEMS}

    linears = [(n, mod) for n, mod in model.named_modules() if isinstance(mod, torch.nn.Linear) and tower(n)]
    per_tower = {t: sum(1 for n, _ in linears if tower(n) == t) for t in ("text", "vision", "audio")}
    print("linear layers per tower:", per_tower, flush=True)
    if not all(per_tower.values()):  # an empty match would silently report cosine 1.0
        raise SystemExit(f"no linear layers matched for some tower: {per_tower}")
    original = {n: mod.weight.detach().clone() for n, mod in linears}
    ref = embed_all()
    result = {}
    for sname, scheme in SCHEMES.items():
        with torch.no_grad():
            for n, mod in linears:
                mod.weight.copy_(fake_quant(original[n], scheme[tower(n)]))
        out = embed_all()
        cos = {k: float(out[k] @ ref[k] / (np.linalg.norm(out[k]) * np.linalg.norm(ref[k]))) for k in ITEMS}
        result[sname] = dict(scheme=scheme, cosine_vs_fp32=cos, min=min(cos.values()))
        print(sname, f"min {min(cos.values()):.5f}", " ".join(f"{k}:{v:.5f}" for k, v in cos.items()), flush=True)
    with torch.no_grad():
        for n, mod in linears:
            mod.weight.copy_(original[n])
    sizes = {t: int(sum(original[n].numel() for n, _ in linears if tower(n) == t)) for t in ("text", "vision", "audio")}
    (ROOT / "reports" / "quant_study.json").write_text(json.dumps(dict(
        note="linear weights fake-quantized per block of 32 (symmetric); activations fp32; token embeddings unchanged",
        linear_layers=per_tower, linear_params=sizes, results=result), indent=1) + "\n")


if __name__ == "__main__":
    main()
