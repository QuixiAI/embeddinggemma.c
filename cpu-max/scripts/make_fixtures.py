#!/usr/bin/env python3
"""Build the reproducible benchmark workload matrix (bench/fixtures/manifest.json).

All media are pre-decoded and stored at exactly the geometry the official
processor would produce, so every backend receives identical pixels/samples and
no backend-side resize or resampling happens:

* text   t16/t128/t512: natural SciFact text with exactly N tokens incl. BOS/EOS
* image  i70/i280/i1120: Flickr30k photos at 336x480, 672x960, 1344x1920
         (H/48 * W/48 == budget, the HF resize is the identity)
* audio  a5/a15/a30: Clotho clips, 16 kHz mono PCM16 WAV (a30 joins two clips)
* video  v4/v8/v16: 1 fps frames of one MSR-VTT clip at 480x672 (140 tokens/frame)
* mixed  m_ti, m_ta, m_tia, m_tv
Pools of distinct items of identical shape are written for concurrency runs.
"""
from __future__ import annotations

import glob
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "fixtures"
HF = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
POOL = 8  # distinct items per shape for concurrency runs

IMAGE_GEOM = {70: (336, 480), 280: (672, 960), 1120: (1344, 1920)}  # (height, width)
FRAME_GEOM = (480, 672)  # 140 tokens


def snapshot(repo: str) -> Path:
    paths = sorted(glob.glob(str(HF / repo / "snapshots" / "*")))
    if not paths:
        sys.exit(f"missing {repo}; see README 'Data' section")
    return Path(paths[-1])


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("google/embeddinggemma-2")


def exact_text(tok, source: str, n: int) -> str:
    """A prefix of `source` that tokenizes to exactly n tokens with BOS/EOS."""
    ids = tok(source, add_special_tokens=False)["input_ids"]
    for cut in range(n - 2, max(n - 40, 1), -1):
        text = tok.decode(ids[:cut]).strip()
        if len(tok(text)["input_ids"]) == n:
            return text
    # pad with words until the count is reached
    text = tok.decode(ids[: n - 2])
    while len(tok(text)["input_ids"]) < n:
        text += " and"
    while len(tok(text)["input_ids"]) > n:
        text = text[:-1]
    assert len(tok(text)["input_ids"]) == n, n
    return text


def corpus_texts() -> list[str]:
    path = snapshot("datasets--mteb--scifact") / "corpus.jsonl"
    docs = [json.loads(l) for l in path.read_text().splitlines()]
    docs = [d["title"] + ". " + d["text"] for d in docs if len(d["text"]) > 4000]
    return docs


def flickr_photos() -> list[Path]:
    return sorted((ROOT / "data" / "flickr1k" / "images_flickr_1k_test").glob("*.jpg"))


def fit_crop(img: Image.Image, h: int, w: int) -> Image.Image:
    """Center-crop to the target aspect ratio, then bicubic-resize (offline, once)."""
    img = img.convert("RGB")
    sw, sh = img.size
    target = w / h
    if sw / sh > target:
        nw = int(round(sh * target))
        img = img.crop(((sw - nw) // 2, 0, (sw - nw) // 2 + nw, sh))
    else:
        nh = int(round(sw / target))
        img = img.crop((0, (sh - nh) // 2, sw, (sh - nh) // 2 + nh))
    return img.resize((w, h), Image.BICUBIC)


def clotho_clips() -> list[tuple[str, np.ndarray]]:
    import pyarrow.parquet as pq
    from scipy.signal import resample_poly

    out = []
    for f in sorted(glob.glob(str(snapshot("datasets--CLAPv2--Clotho") / "data" / "test" / "*.parquet")))[:1]:
        for r in pq.read_table(f).to_pylist():
            data, rate = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
            mono = data.mean(axis=1)
            if rate != 16000:
                g = np.gcd(rate, 16000)
                mono = resample_poly(mono, 16000 // g, rate // g).astype(np.float32)
            out.append((r["index"], np.clip(mono, -1, 1)))
    return out


def video_frames(video: Path, count: int, h: int, w: int, dst: Path) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"fps=1,scale={w}:{h}:flags=bicubic",
                    "-frames:v", str(count), str(dst / "f%02d.png")], check=True)
    frames = sorted(str(p.relative_to(OUT)) for p in dst.glob("f*.png"))
    assert len(frames) == count, (video, len(frames))
    return frames


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    items: list[dict] = []
    texts = corpus_texts()

    # ---- text --------------------------------------------------------------
    for n in (16, 128, 512):
        pool = [exact_text(tok, texts[i], n) for i in range(POOL)]
        items.append(dict(id=f"t{n}", modality="text", text=pool[0], pool=[dict(text=t) for t in pool],
                          expected_tokens=n))
    short_text = exact_text(tok, "A photo and its sounds: people walking along a busy street while a dog barks "
                                 "and cars pass by in the rain at night.", 32)

    # ---- images ------------------------------------------------------------
    photos = flickr_photos()
    for budget, (h, w) in IMAGE_GEOM.items():
        pool = []
        for i in range(POOL):
            p = OUT / "image" / f"i{budget}_{i}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            fit_crop(Image.open(photos[i * 37 % len(photos)]), h, w).save(p)
            pool.append(dict(image=str(p.relative_to(OUT))))
        items.append(dict(id=f"i{budget}", modality="image", image=pool[0]["image"], vision_tokens=budget,
                          pool=pool, expected_tokens=budget + 4))

    # ---- audio ---------------------------------------------------------------
    clips = sorted(clotho_clips(), key=lambda c: -len(c[1]))
    for sec in (5, 15, 30):
        pool = []
        for i in range(POOL):
            a = clips[i % len(clips)][1]
            if len(a) < sec * 16000:  # join a second real clip (a30 only)
                a = np.concatenate([a, clips[(i + POOL) % len(clips)][1]])
            a = a[: sec * 16000]
            p = OUT / "audio" / f"a{sec}_{i}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, a, 16000, subtype="PCM_16")
            pool.append(dict(audio=str(p.relative_to(OUT))))
        items.append(dict(id=f"a{sec}", modality="audio", audio=pool[0]["audio"], pool=pool,
                          expected_tokens=sec * 25 + 4))

    # ---- video ---------------------------------------------------------------
    meta = json.loads((ROOT / "data" / "msrvtt" / "msrvtt_test_1k.json").read_text())
    long_videos = []
    for m in meta:
        v = ROOT / "data" / "msrvtt" / "videos" / m["video"]
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(v)], capture_output=True, text=True).stdout or 0)
        if dur >= 17:
            long_videos.append(v)
        if len(long_videos) >= POOL:
            break
    for count in (4, 8, 16):
        pool = []
        for i, v in enumerate(long_videos[:POOL]):
            frames = video_frames(v, count, *FRAME_GEOM, OUT / "video" / f"v{count}_{i}")
            pool.append(dict(frames=frames, source=v.name))
        items.append(dict(id=f"v{count}", modality="video", frames=pool[0]["frames"], vision_tokens=140,
                          fps=1, pool=pool, expected_tokens=count * 142 + 2))

    # ---- mixed ---------------------------------------------------------------
    img = next(x for x in items if x["id"] == "i280")
    aud = next(x for x in items if x["id"] == "a15")
    vid = next(x for x in items if x["id"] == "v8")

    def mixed(mid, parts, pools):
        pool = []
        for i in range(POOL):
            pool.append(dict(parts=[{"type": t, **(dict(text=short_text) if t == "text" else pools[t][i])}
                                    for t in parts]))
        exp = 1 + sum(30 if t == "text" else (img["vision_tokens"] + 2 if t == "image" else
                                              (15 * 25 + 2 if t == "audio" else 8 * 142)) for t in parts) + 1
        items.append(dict(id=mid, modality="mixed", parts=pool[0]["parts"], pool=pool, expected_tokens=exp))

    pools = {"image": [dict(image=p["image"], vision_tokens=280) for p in img["pool"]],
             "audio": aud["pool"], "video": [dict(frames=p["frames"], vision_tokens=140, fps=1) for p in vid["pool"]]}
    mixed("m_ti", ["text", "image"], pools)
    mixed("m_ta", ["text", "audio"], pools)
    mixed("m_tia", ["text", "image", "audio"], pools)
    mixed("m_tv", ["text", "video"], pools)

    manifest = dict(version=1, root=str(OUT), pool_size=POOL, items=items,
                    notes="expected_tokens counts BOS/EOS and media delimiters (HF processor layout, no video "
                          "timestamps). Verified by scripts/verify_fixtures.py.")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("wrote", OUT / "manifest.json", len(items), "items")


if __name__ == "__main__":
    main()
