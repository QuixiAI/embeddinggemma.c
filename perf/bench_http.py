#!/usr/bin/env python3
"""Benchmark HTTP transport and response serialization with connection reuse."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import http.client
import json
import os
import platform
import socket
import statistics
import subprocess
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = Path(__file__).resolve().parent / "results"


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


def request(connection: http.client.HTTPConnection, body: bytes) -> int:
    connection.request(
        "POST",
        "/api/embed",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    payload = response.read()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {payload!r}")
    return len(payload)


def encode_request(inputs: list[str], encoding_format: str, dimensions: int) -> bytes:
    return json.dumps(
        {
            "input": inputs,
            "dimensions": dimensions,
            "encoding_format": encoding_format,
        },
        separators=(",", ":"),
    ).encode("utf-8")


def wait_until_ready(process: subprocess.Popen[bytes], port: int) -> None:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read().decode("utf-8", "replace")
            raise RuntimeError(f"server exited during startup:\n{stderr}")
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            connection.request("GET", "/api/tags")
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status == 200:
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("server did not become ready within 30 seconds")


def benchmark_concurrent(port: int, body: bytes, concurrency: int,
                         rounds: int) -> float:
    barrier = threading.Barrier(concurrency)

    def worker() -> None:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        barrier.wait()
        try:
            for _ in range(rounds):
                request(connection, body)
        finally:
            connection.close()

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(concurrency)]
        for future in futures:
            future.result()
    elapsed = time.perf_counter() - start
    return concurrency * rounds / elapsed


def benchmark(port: int, keepalive: bool, encoding_format: str, dimensions: int,
              samples: int, miss_samples: int, concurrency: int,
              concurrent_rounds: int) -> dict[str, float | int | str | bool]:
    cached_text = "search_query: HTTP keep-alive benchmark"
    single_body = encode_request([cached_text], encoding_format, dimensions)
    batch_body = encode_request([cached_text] * 32, encoding_format, dimensions)
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    request(connection, single_body)

    single_ms = []
    batch_ms = []
    for _ in range(samples):
        start = time.perf_counter()
        payload_bytes = request(connection, single_body)
        single_ms.append((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        batch_payload_bytes = request(connection, batch_body)
        batch_ms.append((time.perf_counter() - start) * 1000.0)

    misses = []
    unique = time.time_ns()
    for sample in range(miss_samples):
        body = encode_request(
            [f"search_query: HTTP cache miss {unique} {sample}"],
            encoding_format,
            dimensions,
        )
        start = time.perf_counter()
        request(connection, body)
        misses.append((time.perf_counter() - start) * 1000.0)
    connection.close()

    concurrent_requests_per_s = benchmark_concurrent(
        port, single_body, concurrency, concurrent_rounds)
    return {
        "keepalive": keepalive,
        "encoding_format": encoding_format,
        "dimensions": dimensions,
        "payload_bytes": payload_bytes,
        "batch32_payload_bytes": batch_payload_bytes,
        "cached_single_median_ms": statistics.median(single_ms),
        "cached_single_p20_ms": percentile(single_ms, 0.20),
        "cached_single_p80_ms": percentile(single_ms, 0.80),
        "cached_batch32_median_ms": statistics.median(batch_ms),
        "cache_miss_median_ms": statistics.median(misses),
        "concurrent_requests_per_s": concurrent_requests_per_s,
    }


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


def summary(rows: list[dict[str, float | int | str | bool]],
            metadata: dict[str, object]) -> str:
    lines = [
        "# HTTP keep-alive benchmarks",
        "",
        f"- `{metadata['git']}` - {metadata['platform']} - backend `{metadata['backend']}`",
        "",
        "| keep-alive | format | D | cached single ms | cached batch-32 ms | cache miss ms | c32 req/s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {'on' if row['keepalive'] else 'off'} "
            f"| {row['encoding_format']} | {row['dimensions']} "
            f"| {row['cached_single_median_ms']:.3f} "
            f"| {row['cached_batch32_median_ms']:.3f} "
            f"| {row['cache_miss_median_ms']:.3f} "
            f"| {row['concurrent_requests_per_s']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("cpu", "metal", "cuda", "xpu"), required=True
    )
    parser.add_argument("--keepalive", choices=("on", "off"), required=True)
    parser.add_argument("--encoding-format", choices=("float", "base64", "both"),
                        default="both")
    parser.add_argument("--dimensions", type=int, choices=(128, 256, 512, 768),
                        default=768)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--miss-samples", type=int, default=7)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--concurrent-rounds", type=int, default=20)
    parser.add_argument("--response-cache-mb", type=int, default=0)
    parser.add_argument("--binary", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "model" / "embeddinggemma-300M-qat-Q4_0.gguf",
    )
    parser.add_argument("--tag", default="http")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    binary_name = {
        "cpu": "embeddinggemma",
        "metal": "embeddinggemma-metal",
        "cuda": "embeddinggemma-cuda",
        "xpu": "embeddinggemma-xpu",
    }[args.backend]
    binary = args.binary or REPO_ROOT / "build" / binary_name
    if not args.no_build:
        target = {
            "cpu": "all",
            "metal": "metal",
            "cuda": "cuda",
            "xpu": "xpu",
        }[args.backend]
        subprocess.run(
            ["make", target],
            cwd=REPO_ROOT,
            check=True,
        )

    keepalive = args.keepalive == "on"
    environment = os.environ.copy()
    environment["EI_HTTP_KEEPALIVE"] = "1" if keepalive else "0"
    port = reserve_port()
    process = subprocess.Popen(
        [
            str(binary), "--model", str(args.model), "--backend", args.backend,
            "--bind", "127.0.0.1", "--port", str(port), "--workers", "64",
            "--response-cache-mb", str(args.response_cache_mb),
        ],
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        wait_until_ready(process, port)
        formats = ("float", "base64") if args.encoding_format == "both" else (
            args.encoding_format,
        )
        rows = [
            benchmark(
                port,
                keepalive,
                encoding_format,
                args.dimensions,
                args.samples,
                args.miss_samples,
                args.concurrency,
                args.concurrent_rounds,
            )
            for encoding_format in formats
        ]
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    now = dt.datetime.now()
    out_dir = args.out_dir or RESULTS_ROOT / now.strftime("%Y-%m-%d") / (
        f"{now:%H%M%S}-{args.tag}-{args.backend}-{args.keepalive}-"
        f"cache{args.response_cache_mb}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": 1,
        "timestamp": now.isoformat(timespec="seconds"),
        "git": git_label(),
        "platform": platform.platform(),
        "backend": args.backend,
        "keepalive": keepalive,
        "dimensions": args.dimensions,
        "samples": args.samples,
        "miss_samples": args.miss_samples,
        "concurrency": args.concurrency,
        "concurrent_rounds": args.concurrent_rounds,
        "response_cache_mb": args.response_cache_mb,
        "model": str(args.model),
        "binary": str(binary),
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
