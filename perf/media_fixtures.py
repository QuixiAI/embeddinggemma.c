#!/usr/bin/env python3
"""Write media benchmark fixtures: request bodies for perf/loadgen_media.c
(with an @@ID@@ text prefix so the response cache never hits) plus the raw
PNG/WAV files for engine-level probes.

usage: media_fixtures.py OUT_DIR
"""
import base64, io, json, math, os, struct, sys, wave
from PIL import Image

out = sys.argv[1]
os.makedirs(out, exist_ok=True)

def photo(w, h, seed):
    # Photo-like content: smooth gradients plus texture, so PNG size and the
    # vision tower's input are realistic (not a flat colour).
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            r = int(127 + 120 * math.sin(x / 37.0 + seed) * math.cos(y / 53.0))
            g = int(127 + 120 * math.sin((x + y) / 71.0 + seed * 2))
            b = int(127 + 120 * math.cos(x / 23.0 - y / 29.0 + seed))
            t = ((x * 7 + y * 13 + seed * 31) % 17) * 3
            px[x, y] = (max(0, min(255, r + t)), max(0, min(255, g - t)), max(0, min(255, b + t // 2)))
    return img

def png_b64(img, path):
    img.save(path, format="PNG")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def wav_b64(seconds, path):
    n = 16000 * seconds
    frames = b"".join(struct.pack("<h", int(9000 * math.sin(2 * math.pi * 440 * i / 16000)
                                            + 3000 * math.sin(2 * math.pi * 1313 * i / 16000 + i / 4000.0)))
                      for i in range(n))
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(frames)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def body(parts):
    return json.dumps({"model": "embeddinggemma-2",
                       "input": {"content": [{"type": "text", "text": "@@ID@@ "}] + parts}})

img768 = png_b64(photo(768, 768, 1), f"{out}/image768.png")      # 256 soft tokens (card default budget 280)
img432 = png_b64(photo(432, 432, 2), f"{out}/image432.png")      # 81 soft tokens
frames = [png_b64(photo(432, 432, 10 + i), f"{out}/frame{i}.png") for i in range(4)]
wav10 = wav_b64(10, f"{out}/audio10s.wav")                       # 250 soft tokens
wav30 = wav_b64(30, f"{out}/audio30s.wav")                       # 750 soft tokens

with open(f"{out}/image768.json", "w") as f: f.write(body([{"type": "image", "data": img768}]))
with open(f"{out}/image432.json", "w") as f: f.write(body([{"type": "image", "data": img432}]))
with open(f"{out}/audio10s.json", "w") as f: f.write(body([{"type": "audio", "data": wav10}]))
with open(f"{out}/audio30s.json", "w") as f: f.write(body([{"type": "audio", "data": wav30}]))
with open(f"{out}/video4.json", "w") as f: f.write(body([{"type": "video", "frames": frames, "fps": 1}]))
with open(f"{out}/mixed.json", "w") as f:
    f.write(body([{"type": "text", "text": "An image and its soundtrack: "},
                  {"type": "image", "data": img432}, {"type": "audio", "data": wav10}]))
for name in os.listdir(out):
    print(name, os.path.getsize(f"{out}/{name}"))
