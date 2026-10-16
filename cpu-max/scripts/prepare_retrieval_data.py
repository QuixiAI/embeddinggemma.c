#!/usr/bin/env python3
"""Prepare the labeled retrieval sets once, with media already at the geometry
the official processor would produce, so every backend embeds identical pixels
and samples (no backend-side resize/resampling):

* flickr  Flickr30k 1k test: 1000 images (280-token budget, HF resize) x 5 captions
* clotho  Clotho test (4 shards): 16 kHz mono PCM16 WAV (<= 30 s, the HF extractor
          limit) x the captions packed in each row
* msrvtt  MSR-VTT test-1k, first N videos: 1 fps frames (<= F, uniform), resized
          to the HF 140-token geometry, x 1 caption
* scifact BEIR SciFact test: corpus + queries + qrels (text only)

  .venv/bin/python scripts/prepare_retrieval_data.py [--msrvtt-videos 200] [--msrvtt-frames 8]
Writes data/retrieval/<task>/manifest.json.
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "retrieval"
HF = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def snapshot(repo: str) -> Path:
    return Path(sorted(glob.glob(str(HF / repo / "snapshots" / "*")))[-1])


def hf_size(h: int, w: int, budget: int) -> tuple[int, int]:
    """transformers Gemma4 get_aspect_ratio_preserving_size (patch 16, pool 3)."""
    max_patches, side = budget * 9, 48
    factor = math.sqrt(max_patches * 256 / (h * w))
    th, tw = int(math.floor(factor * h / side)) * side, int(math.floor(factor * w / side)) * side
    if th == 0:
        th, tw = side, min(int(math.floor(w / h)) * side, (max_patches // 9) * side)
    elif tw == 0:
        tw, th = side, min(int(math.floor(h / w)) * side, (max_patches // 9) * side)
    return th, tw


def hf_resize(img: Image.Image, budget: int) -> Image.Image:
    """The official processor's resize: torchvision bicubic, antialias, on uint8."""
    import torch
    import torchvision.transforms.v2.functional as F

    img = img.convert("RGB")
    th, tw = hf_size(img.height, img.width, budget)
    if (th, tw) == (img.height, img.width):
        return img
    t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).contiguous()
    r = F.resize(t, [th, tw], interpolation=F.InterpolationMode.BICUBIC, antialias=True)
    return Image.fromarray(r.permute(1, 2, 0).numpy())


def flickr() -> None:
    d = OUT / "flickr"
    (d / "images").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(glob.glob(str(snapshot("datasets--nlphuji--flickr_1k_test_image_text_retrieval")
                                                    / "test_1k_flickr.csv"))[0])))
    docs, queries = [], []
    for i, r in enumerate(rows):
        dst = d / "images" / (Path(r["filename"]).stem + ".png")
        if not dst.exists():
            hf_resize(Image.open(ROOT / "data/flickr1k/images_flickr_1k_test" / r["filename"]), 280).save(dst)
        docs.append(dict(id=f"img{i}", image=str(dst.relative_to(ROOT))))
        for c in json.loads(r["raw"]):
            queries.append(dict(text=c, positives=[f"img{i}"]))
    (d / "manifest.json").write_text(json.dumps(dict(task="flickr_t2i", doc_modality="image", vision_tokens=280,
                                                     docs=docs, queries=queries), indent=0))
    print("flickr", len(docs), "images", len(queries), "captions")


def clotho() -> None:
    import pyarrow.parquet as pq
    from scipy.signal import resample_poly

    d = OUT / "clotho"
    (d / "audio").mkdir(parents=True, exist_ok=True)
    docs, queries = [], []
    for f in sorted(glob.glob(str(snapshot("datasets--CLAPv2--Clotho") / "data" / "test" / "*.parquet"))):
        for r in pq.read_table(f).to_pylist():
            did = r["index"].rsplit("_", 1)[-1]
            dst = d / "audio" / f"{did}.wav"
            if not dst.exists():
                a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
                a = a.mean(1)
                if sr != 16000:
                    g = np.gcd(sr, 16000)
                    a = resample_poly(a, 16000 // g, sr // g).astype(np.float32)
                sf.write(dst, np.clip(a[: 30 * 16000], -1, 1), 16000, subtype="PCM_16")
            docs.append(dict(id=f"aud{did}", audio=str(dst.relative_to(ROOT))))
            # The five captions are stored joined; each ends with a period or is followed by a capital.
            caps = [c.strip() for c in re.split(r"(?<=[.!?])\s+(?=[A-Z])", r["text"]) if c.strip()]
            for c in caps:
                queries.append(dict(text=c, positives=[f"aud{did}"]))
    (d / "manifest.json").write_text(json.dumps(dict(task="clotho_t2a", doc_modality="audio", docs=docs,
                                                     queries=queries), indent=0))
    print("clotho", len(docs), "clips", len(queries), "captions")


def msrvtt(n_videos: int, max_frames: int) -> None:
    d = OUT / "msrvtt"
    meta = json.loads((ROOT / "data/msrvtt/msrvtt_test_1k.json").read_text())[:n_videos]
    docs, queries = [], []
    for m in meta:
        fd = d / "frames" / m["video_id"]
        if not fd.exists():
            fd.mkdir(parents=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(ROOT / "data/msrvtt/videos" / m["video"]),
                            "-vf", "fps=1", str(fd / "raw%03d.png")], check=True)
            raw = sorted(fd.glob("raw*.png"))
            idx = np.linspace(0, len(raw) - 1, min(max_frames, len(raw))).round().astype(int)
            for j, k in enumerate(idx):
                hf_resize(Image.open(raw[k]), 140).save(fd / f"f{j:02d}.png")
            for r in raw:
                r.unlink()
        frames = sorted(str(p.relative_to(ROOT)) for p in fd.glob("f*.png"))
        docs.append(dict(id=m["video_id"], frames=frames))
        queries.append(dict(text=m["caption"], positives=[m["video_id"]]))
    (d / "manifest.json").write_text(json.dumps(dict(task="msrvtt_t2v", doc_modality="video", vision_tokens=140,
                                                     max_frames=max_frames, docs=docs, queries=queries), indent=0))
    print("msrvtt", len(docs), "videos", len(queries), "captions")


def scifact() -> None:
    d = OUT / "scifact"
    d.mkdir(parents=True, exist_ok=True)
    s = snapshot("datasets--mteb--scifact")
    corpus = [json.loads(l) for l in (s / "corpus.jsonl").read_text().splitlines()]
    qs = {json.loads(l)["_id"]: json.loads(l)["text"] for l in (s / "queries.jsonl").read_text().splitlines()}
    rel: dict[str, list[str]] = {}
    for r in csv.DictReader(open(s / "qrels" / "test.tsv"), delimiter="\t"):
        if int(r["score"]) > 0:
            rel.setdefault(r["query-id"], []).append(r["corpus-id"])
    docs = [dict(id=c["_id"], text=f"title: {c['title'] or 'none'} | text: {c['text']}") for c in corpus]
    queries = [dict(text=qs[q], positives=p) for q, p in rel.items()]
    (d / "manifest.json").write_text(json.dumps(dict(task="scifact", doc_modality="text", docs=docs,
                                                     queries=queries), indent=0))
    print("scifact", len(docs), "docs", len(queries), "queries")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msrvtt-videos", type=int, default=200)
    ap.add_argument("--msrvtt-frames", type=int, default=8)
    args = ap.parse_args()
    flickr()
    clotho()
    msrvtt(args.msrvtt_videos, args.msrvtt_frames)
    scifact()


if __name__ == "__main__":
    main()
