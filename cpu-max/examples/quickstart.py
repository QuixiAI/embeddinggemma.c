#!/usr/bin/env python3
"""Quick start and smoke test for a running CPU-Max server (standard library only).

Sends a text, an image, an audio clip and a mixed text+image+audio request to
/v1/embeddings and checks every response: 768 finite values with unit L2 norm.
The image (a 480x336 gradient PNG) and the audio (2 s of a 440 Hz tone,
16 kHz mono PCM16 WAV) are generated here, so no sample files are needed.

  ./build/embeddinggemma --bind 127.0.0.1 --port 42666 &   # downloads the models on first start
  python3 cpu-max/examples/quickstart.py          # exits non-zero if any check fails

Options: --url (default http://127.0.0.1:42666), --wait SECONDS (poll /healthz
until the server is up; useful right after starting it).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import struct
import sys
import time
import urllib.error
import urllib.request
import wave
import zlib


def synthetic_png(width: int = 480, height: int = 336) -> bytes:
    rows = []
    for y in range(height):
        row = bytearray([0])  # filter type 0
        for x in range(width):
            row += bytes((x * 255 // width, y * 255 // height, ((x // 24 + y // 24) % 2) * 200 + 20))
        rows.append(bytes(row))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"".join(rows))) + \
        chunk(b"IEND", b"")


def synthetic_wav(seconds: float = 2.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / rate)))
                               for i in range(int(seconds * rate))))
    return buf.getvalue()


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url + "/v1/embeddings", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def check(name: str, response: dict) -> list[float]:
    emb = response["data"][0]["embedding"]
    norm = math.sqrt(sum(v * v for v in emb))
    ok = len(emb) == 768 and all(math.isfinite(v) for v in emb) and abs(norm - 1.0) < 1e-3
    print(f"{'ok  ' if ok else 'FAIL'} {name:6s} dims={len(emb)} norm={norm:.6f} "
          f"tokens={response.get('usage', {}).get('prompt_tokens')}")
    if not ok:
        sys.exit(f"{name}: invalid embedding")
    return emb


def wait_for(url: str, seconds: float) -> None:
    deadline = time.time() + seconds
    while True:
        try:
            with urllib.request.urlopen(url + "/healthz", timeout=5):
                return
        except (urllib.error.URLError, ConnectionError, OSError):
            if time.time() > deadline:
                sys.exit(f"server at {url} did not become ready within {seconds:.0f} s")
            time.sleep(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:42666")
    ap.add_argument("--wait", type=float, default=0, help="seconds to wait for /healthz first")
    args = ap.parse_args()
    if args.wait:
        wait_for(args.url, args.wait)
    png = base64.b64encode(synthetic_png()).decode()
    wav = base64.b64encode(synthetic_wav()).decode()
    out = {
        "text": check("text", post(args.url, {"model": "embeddinggemma-2",
                                              "input": ["task: search result | query: a colorful pattern"]})),
        "image": check("image", post(args.url, {"model": "embeddinggemma-2",
                                                "input": {"content": [{"type": "image", "data": png}]}})),
        "audio": check("audio", post(args.url, {"model": "embeddinggemma-2",
                                                "input": {"content": [{"type": "audio", "data": wav}]}})),
        "mixed": check("mixed", post(args.url, {"model": "embeddinggemma-2", "input": {"content": [
            {"type": "text", "text": "A picture and a sound: "},
            {"type": "image", "data": png}, {"type": "audio", "data": wav}]}})),
    }
    names = list(out)
    print("cosine similarity:")
    print(" " * 7 + "".join(f"{n:>8s}" for n in names))
    for a in names:
        print(f"{a:7s}" + "".join(f"{sum(x * y for x, y in zip(out[a], out[b])):8.3f}" for b in names))
    print("all checks passed")


if __name__ == "__main__":
    main()
