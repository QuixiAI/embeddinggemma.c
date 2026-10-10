#!/usr/bin/env python3
"""Native profile parity and HTTP modality regressions; no runtime Python dependencies."""
import argparse
import base64
import concurrent.futures
import io
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import time
import tempfile
import urllib.error
import urllib.request
import wave


def cosine(a, b):
    assert len(a) == len(b)
    assert all(math.isfinite(v) for v in a)
    return sum(x*y for x, y in zip(a, b)) / math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))


def request(port, payload, expected=200, route="/v1/embeddings"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read()
    assert status == expected, (status, body[:500])
    return json.loads(body)


def image(color):
    return base64.b64encode(b"P6\n48 48\n255\n" + bytes(color)*48*48).decode()


def audio(frequency):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"".join(struct.pack("<h", int(10000*math.sin(2*math.pi*frequency*i/16000)))
                                 for i in range(3200)))
    return base64.b64encode(output.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mmproj", required=True)
    args = parser.parse_args()
    goldens = json.loads(Path(__file__).with_name("gemma2-goldens.json").read_text())
    expected_backend = os.environ.get("EI_BACKEND", "cpu")
    worst = 1
    for case in goldens["texts"]:
        result = json.loads(subprocess.check_output([args.binary, args.model, args.mmproj, "text", case["text"]]))
        assert result["backend"] == expected_backend, result["backend"]
        assert result["tokens"] == case["tokens"], repr(case["text"][:80])
        score = cosine(result["embedding"], case["embedding"])
        assert score >= .999, ("text", score)
        worst = min(worst, score)
    print(f"EmbeddingGemma 2 tokenizer exact; text minimum cosine {worst:.9f}", flush=True)
    for mode in ("vision", "audio"):
        result = json.loads(subprocess.check_output([args.binary, args.model, args.mmproj, mode]))
        assert result["backend"] == expected_backend, result["backend"]
        for key in ("tower", "embedding"):
            score = cosine(result[key], goldens[mode][key])
            assert score >= .999, (mode, key, score)
            print(f"EmbeddingGemma 2 {mode} {key} cosine {score:.9f}", flush=True)

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    # Resolve both files through the real default cache path, without model/profile overrides.
    cache = tempfile.TemporaryDirectory(prefix="embeddinggemma-profile-test-")
    cache_dir = Path(cache.name) / "embeddinggemma.c"
    cache_dir.mkdir()
    (cache_dir / "embeddinggemma-2-Q8_0.gguf").symlink_to(Path(args.model).resolve())
    (cache_dir / "mmproj-embeddinggemma-2-Q8_0.gguf").symlink_to(Path(args.mmproj).resolve())
    env = os.environ.copy()
    env.pop("EI_MODEL_PATH", None)
    env["XDG_CACHE_HOME"] = cache.name
    help_text = subprocess.run([args.server, "--help"], capture_output=True, text=True).stderr
    assert "embeddinggemma-2-Q8_0.gguf" in help_text and "--profile" in help_text
    mismatch = subprocess.run([args.server, "--profile", "embeddinggemma-300m", "--model", args.model],
                              capture_output=True, text=True, timeout=20)
    assert mismatch.returncode and "disagree" in mismatch.stderr
    process = subprocess.Popen([args.server,
                                "--bind", "127.0.0.1", "--port", str(port), "--workers", "4",
                                "--max-batch-tokens", "2048", "--response-cache-mb", "0"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
    try:
        for _ in range(200):
            if process.poll() is not None:
                raise AssertionError(process.stderr.read().decode())
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1):
                    break
            except OSError:
                time.sleep(.05)
        else:
            raise AssertionError("server did not become ready")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/tags") as r:
            assert json.load(r)["models"][0]["name"] == "embeddinggemma-2"
        red, blue = image((220, 30, 50)), image((20, 40, 220))
        tone, other_tone = audio(440), audio(880)
        payloads = {
            "text": "hello world",
            "structured_text": {"content": [{"type": "text", "text": "hello world"}]},
            "image": {"content": [{"type": "image", "data": red}]},
            "other_image": {"content": [{"type": "image", "data": blue}]},
            "audio": {"content": [{"type": "audio", "data": tone}]},
            "other_audio": {"content": [{"type": "audio", "data": other_tone}]},
            "video": {"content": [{"type": "video", "frames": [red, blue], "fps": 1}]},
            "reverse_video": {"content": [{"type": "video", "frames": [blue, red], "fps": 1}]},
            "mixed": {"content": [{"type": "text", "text": "A red image and a tone: "},
                                  {"type": "image", "data": red}, {"type": "audio", "data": tone}]},
        }
        def run(item):
            key, value = item
            response = request(port, {"model": "embeddinggemma-2", "input": value})
            emb = response["data"][0]["embedding"]
            assert len(emb) == 768 and all(math.isfinite(x) for x in emb)
            assert abs(sum(x*x for x in emb)-1) < 1e-5
            assert response["usage"]["prompt_tokens"] > 1
            return key, emb
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            results = dict(pool.map(run, payloads.items()))
        assert cosine(results["text"], results["structured_text"]) > .999999
        for mode in ("image", "audio"):
            assert cosine(results[mode], goldens["http"][mode]) >= .999, mode
        for a, b in (("image", "other_image"), ("audio", "other_audio"), ("video", "reverse_video"), ("image", "mixed")):
            assert results[a] != results[b], f"{a}/{b}: modality was ignored"
        batch = request(port, {"model": "alias", "input": [payloads["image"], payloads["text"], payloads["audio"]]})
        assert batch["model"] == "alias"
        for item, name in zip(batch["data"], ("image", "text", "audio")):
            assert cosine(item["embedding"], results[name]) > .999999
        for dims in (128, 256, 512, 768):
            result = request(port, {"input": payloads["mixed"], "dimensions": dims, "encoding_format": "base64"}, route="/api/embed")
            values = struct.unpack(f"<{dims}f", base64.b64decode(result["embeddings"][0]))
            assert abs(sum(x*x for x in values)-1) < 1e-5
            assert cosine(values, results["mixed"][:dims]) > .999999
        # The collector must reserve space for one full-context sequence even with a smaller packing budget.
        long_text = "word "*2200
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(lambda _: request(port, {"model": "x", "input": long_text}), range(2)))
        assert outputs[0]["usage"]["prompt_tokens"] > 2048
        for value in ({"content": []}, {"content": [{"type": "image", "data": "!!!!"}]},
                      {"content": [{"type": "audio", "data": red}]},
                      {"content": [{"type": "video", "frames": [red], "fps": 0}]},
                      {"content": [{"type": "text", "text": "ok", "image": red}]},
                      {"content": [{"type": "image", "data": red, "vision_tokens": 69}]},
                      {"content": [{"type": "audio", "data": tone, "vision_tokens": 70}]}):
            request(port, {"model": "x", "input": value}, expected=400)
        for tokens in (70, 1120):
            request(port, {"model": "x", "input": {"content": [{"type": "image", "data": red, "vision_tokens": tokens},
                                                               {"type": "video", "frames": [red], "vision_tokens": tokens}]}})
        # Over-long input is truncated to the 8192-token context, never rejected.
        clipped = [request(port, {"model": "x", "input": "word "*words}) for words in (9000, 9100)]
        assert clipped[0]["usage"]["prompt_tokens"] == 8192
        assert cosine(clipped[0]["data"][0]["embedding"], clipped[1]["data"][0]["embedding"]) > .999999
        for tail in ([{"type": "image", "data": red}], [{"type": "audio", "data": tone}],
                     [{"type": "video", "frames": [red, red]}]):
            for words in (8100, 9000):
                request(port, {"model": "x", "input": {"content": [{"type": "text", "text": "word "*words}] + tail}})
        print("EmbeddingGemma 2 HTTP: default profile, all modalities, mixed batches, dimensions, concurrency, limits passed", flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stderr.close()
        cache.cleanup()


if __name__ == "__main__":
    main()
