#!/usr/bin/env python3
"""Minimal client for the embeddinggemma server (upstream or this fork).

Embeds a text query, an image, an audio clip, a video (pre-decoded frames) and
a mixed input, then prints their cosine similarities. Only the standard library
is required; media are sent as base64 content parts.

  # from the repository root
  ./build/embeddinggemma --bind 127.0.0.1 --port 42666 --threads 8 \
      --model cpu-max/models/gguf/embeddinggemma-2-Q8_0.gguf \
      --mmproj cpu-max/models/gguf/mmproj-embeddinggemma-2-Q8_0.gguf &
  python3 cpu-max/examples/client.py --image cpu-max/bench/fixtures/image/i280_0.png \
      --audio cpu-max/bench/fixtures/audio/a5_0.wav --frames cpu-max/bench/fixtures/video/v4_0/f*.png

Video: decode frames yourself (e.g. `ffmpeg -i clip.mp4 -vf fps=1 f%02d.png`); the
server does not decode containers. Text is embedded as given, so add the task
prefix ("task: search result | query: ...") yourself. Media take no prefix.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import urllib.request


def b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def embed(url: str, inp, dimensions: int = 768) -> list[float]:
    body = json.dumps({"model": "embeddinggemma-2", "input": inp, "dimensions": dimensions}).encode()
    req = urllib.request.Request(url + "/v1/embeddings", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["data"][0]["embedding"]


def cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:42666")
    ap.add_argument("--query", default="task: search result | query: a photo of people outdoors")
    ap.add_argument("--image")
    ap.add_argument("--audio", help="mono 16 kHz WAV (PCM16 or float32)")
    ap.add_argument("--frames", nargs="*", help="video frames in temporal order (1 fps)")
    ap.add_argument("--vision-tokens", type=int, default=None, help="70..1120 (default 280 image / 140 frame)")
    ap.add_argument("--dimensions", type=int, default=768, choices=[768, 512, 256, 128])
    args = ap.parse_args()

    out = {"query": embed(args.url, [args.query], args.dimensions)}
    extra = {"vision_tokens": args.vision_tokens} if args.vision_tokens else {}
    if args.image:
        out["image"] = embed(args.url, {"content": [{"type": "image", "data": b64(args.image), **extra}]}, args.dimensions)
    if args.audio:
        out["audio"] = embed(args.url, {"content": [{"type": "audio", "data": b64(args.audio)}]}, args.dimensions)
    if args.frames:
        out["video"] = embed(args.url, {"content": [{"type": "video", "frames": [b64(f) for f in args.frames], "fps": 1,
                                                     **extra}]}, args.dimensions)
    if args.image and args.audio:
        out["mixed"] = embed(args.url, {"content": [
            {"type": "text", "text": "A scene and its sound: "},
            {"type": "image", "data": b64(args.image), **extra},
            {"type": "audio", "data": b64(args.audio)}]}, args.dimensions)
    names = list(out)
    print(f"{len(out['query'])}-d embeddings; cosine similarity:")
    print(" " * 8 + "".join(f"{n:>9s}" for n in names))
    for a in names:
        print(f"{a:8s}" + "".join(f"{cosine(out[a], out[b]):9.3f}" for b in names))


if __name__ == "__main__":
    main()
