#!/usr/bin/env python3
"""Benchmark Matryoshka output sizes through the production HTTP server."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import platform
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
DIMENSIONS = (768, 512, 256, 128)


def reserve_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def request_bytes(url: str, body: bytes) -> int:
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {payload!r}")
        return len(payload)


def encode_request(inputs: list[str], dimensions: int, encoding_format: str) -> bytes:
    return json.dumps(
        {
            "model": "embeddinggemma-300m",
            "input": inputs,
            "dimensions": dimensions,
            "encoding_format": encoding_format,
        },
        separators=(",", ":"),
    ).encode("utf-8")


def wait_until_ready(process: subprocess.Popen[bytes], base_url: str) -> None:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read().decode("utf-8", "replace")
            raise RuntimeError(f"server exited during startup:\n{stderr}")
        try:
            with urllib.request.urlopen(base_url + "/api/tags", timeout=0.25):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(0.05)
    raise RuntimeError("server did not become ready within 30 seconds")


def benchmark(url: str, cache_samples: int, miss_samples: int,
              concurrent_rounds: int, concurrency: int,
              encoding_format: str) -> list[dict[str, float | int | str]]:
    cached_text = "search_query: cached Matryoshka dimension benchmark"
    single_bodies = {
        d: encode_request([cached_text], d, encoding_format) for d in DIMENSIONS
    }
    batch_bodies = {
        d: encode_request([cached_text] * 32, d, encoding_format)
        for d in DIMENSIONS
    }
    for dimensions in DIMENSIONS:
        request_bytes(url, single_bodies[dimensions])

    rows = {
        d: {
            "dimensions": d,
            "payload_bytes": request_bytes(url, single_bodies[d]),
            "batch32_payload_bytes": request_bytes(url, batch_bodies[d]),
            "cached_single_ms": [],
            "cached_batch32_ms": [],
            "cache_miss_ms": [],
            "concurrent_wall_s": 0.0,
        }
        for d in DIMENSIONS
    }

    for sample in range(cache_samples):
        order = DIMENSIONS[sample % len(DIMENSIONS):] + DIMENSIONS[:sample % len(DIMENSIONS)]
        for dimensions in order:
            start = time.perf_counter()
            request_bytes(url, single_bodies[dimensions])
            rows[dimensions]["cached_single_ms"].append(
                (time.perf_counter() - start) * 1000.0
            )
            start = time.perf_counter()
            request_bytes(url, batch_bodies[dimensions])
            rows[dimensions]["cached_batch32_ms"].append(
                (time.perf_counter() - start) * 1000.0
            )

    unique = time.time_ns()
    for sample in range(miss_samples):
        order = DIMENSIONS[sample % len(DIMENSIONS):] + DIMENSIONS[:sample % len(DIMENSIONS)]
        for dimensions in order:
            text = f"search_query: uncached dimension {unique} {sample} {dimensions}"
            start = time.perf_counter()
            request_bytes(url, encode_request([text], dimensions, encoding_format))
            rows[dimensions]["cache_miss_ms"].append(
                (time.perf_counter() - start) * 1000.0
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        for dimensions in DIMENSIONS:
            for _ in range(concurrent_rounds):
                start = time.perf_counter()
                futures = [
                    executor.submit(request_bytes, url, single_bodies[dimensions])
                    for _ in range(concurrency)
                ]
                for future in futures:
                    future.result()
                rows[dimensions]["concurrent_wall_s"] += time.perf_counter() - start

    result = []
    for dimensions in DIMENSIONS:
        row = rows[dimensions]
        single = row["cached_single_ms"]
        batch = row["cached_batch32_ms"]
        misses = row["cache_miss_ms"]
        result.append({
            "encoding_format": encoding_format,
            "dimensions": dimensions,
            "payload_bytes": row["payload_bytes"],
            "batch32_payload_bytes": row["batch32_payload_bytes"],
            "cached_single_median_ms": statistics.median(single),
            "cached_single_p20_ms": percentile(single, 0.20),
            "cached_single_p80_ms": percentile(single, 0.80),
            "cached_batch32_median_ms": statistics.median(batch),
            "cached_batch32_p20_ms": percentile(batch, 0.20),
            "cached_batch32_p80_ms": percentile(batch, 0.80),
            "cache_miss_median_ms": statistics.median(misses),
            "cache_miss_p20_ms": percentile(misses, 0.20),
            "cache_miss_p80_ms": percentile(misses, 0.80),
            "concurrent_requests_per_s": (
                concurrent_rounds * concurrency / row["concurrent_wall_s"]
            ),
        })
    baseline = result[0]["concurrent_requests_per_s"]
    for row in result:
        row["concurrent_x_vs_768"] = row["concurrent_requests_per_s"] / baseline
    return result


def git_label() -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    label = revision.stdout.strip() if revision.returncode == 0 else "uncommitted"
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return label + ("-dirty" if status.stdout.strip() else "")


def summary(rows: list[dict[str, float | int | str]], metadata: dict[str, object]) -> str:
    lines = [
        "# Matryoshka HTTP dimension benchmarks",
        "",
        f"- `{metadata['git']}` - {metadata['platform']} - backend `{metadata['backend']}`",
        "",
        "| format | dimensions | payload bytes | cached single ms | cached batch-32 ms | cache miss ms | concurrent req/s | x vs 768 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['encoding_format']} | {row['dimensions']} | {row['payload_bytes']} "
            f"| {row['cached_single_median_ms']:.3f} "
            f"| {row['cached_batch32_median_ms']:.3f} "
            f"| {row['cache_miss_median_ms']:.3f} "
            f"| {row['concurrent_requests_per_s']:.1f} "
            f"| {row['concurrent_x_vs_768']:.2f}x |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("cpu", "metal", "cuda", "rocm", "xpu"), required=True
    )
    parser.add_argument("--binary", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "model" / "embeddinggemma-300M-qat-Q4_0.gguf",
    )
    parser.add_argument("--cache-samples", type=int, default=30)
    parser.add_argument("--miss-samples", type=int, default=7)
    parser.add_argument("--concurrent-rounds", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument(
        "--encoding-format",
        choices=("float", "base64", "both"),
        default="both",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    binary_name = {
        "cpu": "embeddinggemma",
        "metal": "embeddinggemma-metal",
        "cuda": "embeddinggemma-cuda",
        "rocm": "embeddinggemma-rocm",
        "xpu": "embeddinggemma-xpu",
    }[args.backend]
    binary = args.binary or REPO_ROOT / "build" / binary_name
    if not args.no_build:
        target = {
            "cpu": "all",
            "metal": "metal",
            "cuda": "cuda",
            "rocm": "rocm",
            "xpu": "xpu",
        }[args.backend]
        subprocess.run(["make", target], cwd=REPO_ROOT, check=True)

    port = reserve_port()
    process = subprocess.Popen(
        [
            str(binary), "--model", str(args.model), "--backend", args.backend,
            "--bind", "127.0.0.1", "--port", str(port), "--workers", "64",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        wait_until_ready(process, base_url)
        formats = ("float", "base64") if args.encoding_format == "both" else (
            args.encoding_format,
        )
        rows = []
        for encoding_format in formats:
            rows.extend(benchmark(
                base_url + "/api/embed",
                args.cache_samples,
                args.miss_samples,
                args.concurrent_rounds,
                args.concurrency,
                encoding_format,
            ))
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    now = dt.datetime.now()
    out_dir = args.out_dir or RESULTS_ROOT / now.strftime("%Y-%m-%d") / (
        f"{now:%H%M%S}-{args.backend}-dimensions"
    )
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": 1,
        "timestamp": now.isoformat(timespec="seconds"),
        "git": git_label(),
        "platform": platform.platform(),
        "backend": args.backend,
        "binary": str(binary),
        "model": str(args.model),
        "cache_samples": args.cache_samples,
        "miss_samples": args.miss_samples,
        "concurrent_rounds": args.concurrent_rounds,
        "concurrency": args.concurrency,
        "encoding_format": args.encoding_format,
    }
    (out_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    report = summary(rows, metadata)
    (out_dir / "summary.md").write_text(report)
    print(f"wrote {out_dir.relative_to(REPO_ROOT)}/")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
