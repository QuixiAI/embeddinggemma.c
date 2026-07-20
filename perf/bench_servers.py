#!/usr/bin/env python3
"""Compare embedding HTTP servers with identical uncached text requests."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import http.client
import json
import math
import platform
import statistics
import threading
import time
import urllib.parse
from pathlib import Path


RESULTS_ROOT = Path(__file__).resolve().parent / "results"
WORDS = (
    "vector search semantic retrieval compact native inference query document "
    "ranking context language model embedding efficient portable server kernel "
    "batch token latency throughput memory compute normalized representation"
).split()


def parse_csv_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",")]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def make_input(word_count: int, request_id: int, item_id: int) -> str:
    words = [WORDS[index % len(WORDS)] for index in range(word_count)]
    return "search_query: " + " ".join(words) + f" request {request_id} item {item_id}"


def request_body(api: str, model: str, dimensions: int, word_count: int,
                 batch_size: int, request_id: int) -> bytes:
    inputs = [
        make_input(word_count, request_id, item_id)
        for item_id in range(batch_size)
    ]
    body: dict[str, object] = {"model": model, "input": inputs}
    if api != "ollama":
        body["dimensions"] = dimensions
        body["encoding_format"] = "float"
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def response_stats(api: str, payload: bytes, batch_size: int,
                   dimensions: int) -> tuple[int, int | None]:
    document = json.loads(payload)
    if api == "openai":
        embeddings = [item["embedding"] for item in document["data"]]
        usage = document.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
    else:
        embeddings = document["embeddings"]
        prompt_tokens = document.get("prompt_eval_count")
    if len(embeddings) != batch_size:
        raise RuntimeError(
            f"server returned {len(embeddings)} embeddings for batch {batch_size}"
        )
    for embedding in embeddings:
        if not isinstance(embedding, list) or len(embedding) != dimensions:
            raise RuntimeError(
                f"server returned dimension {len(embedding)}; expected {dimensions}"
            )
    return len(payload), int(prompt_tokens) if prompt_tokens is not None else None


class Endpoint:
    def __init__(self, url: str, api: str, path: str | None, api_key: str | None):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("--url must be an http:// or https:// URL")
        self.host = parsed.hostname
        self.port = parsed.port
        self.https = parsed.scheme == "https"
        default_path = {
            "embeddinggemma": "/api/embed",
            "ollama": "/api/embed",
            "openai": "/v1/embeddings",
        }[api]
        self.path = path or parsed.path or default_path
        if self.path == "/":
            self.path = default_path
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def connect(self) -> http.client.HTTPConnection:
        connection_type = (
            http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        )
        return connection_type(self.host, self.port, timeout=120)


def send(connection: http.client.HTTPConnection, endpoint: Endpoint, api: str,
         body: bytes, batch_size: int, dimensions: int) -> tuple[int, int | None]:
    connection.request("POST", endpoint.path, body=body, headers=endpoint.headers)
    response = connection.getresponse()
    payload = response.read()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {payload[:1000]!r}")
    return response_stats(api, payload, batch_size, dimensions)


def warm_up(endpoint: Endpoint, api: str, model: str, dimensions: int,
            word_count: int, batch_size: int, requests: int) -> None:
    connection = endpoint.connect()
    try:
        for request_id in range(requests):
            body = request_body(
                api, model, dimensions, word_count, batch_size, -request_id - 1
            )
            send(connection, endpoint, api, body, batch_size, dimensions)
    finally:
        connection.close()


def benchmark(endpoint: Endpoint, api: str, model: str, dimensions: int,
              word_count: int, batch_size: int, concurrency: int,
              rounds: int) -> dict[str, int | float | None]:
    barrier = threading.Barrier(concurrency)
    start_time: list[float] = []

    def mark_start() -> None:
        start_time.append(time.perf_counter())

    barrier = threading.Barrier(concurrency, action=mark_start)

    def worker(worker_id: int) -> tuple[list[float], int, int | None]:
        connection = endpoint.connect()
        latencies: list[float] = []
        response_bytes = 0
        prompt_tokens = 0
        has_token_count = True
        barrier.wait()
        try:
            for round_id in range(rounds):
                request_id = worker_id * rounds + round_id + 1_000_000
                body = request_body(
                    api, model, dimensions, word_count, batch_size, request_id
                )
                started = time.perf_counter()
                size, tokens = send(
                    connection, endpoint, api, body, batch_size, dimensions
                )
                latencies.append((time.perf_counter() - started) * 1000.0)
                response_bytes += size
                if tokens is None:
                    has_token_count = False
                else:
                    prompt_tokens += tokens
        finally:
            connection.close()
        return latencies, response_bytes, prompt_tokens if has_token_count else None

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, worker_id) for worker_id in range(concurrency)]
        worker_results = [future.result() for future in futures]
    elapsed = time.perf_counter() - start_time[0]
    latencies = [sample for result in worker_results for sample in result[0]]
    response_bytes = sum(result[1] for result in worker_results)
    token_counts = [result[2] for result in worker_results]
    prompt_tokens = (
        sum(count for count in token_counts if count is not None)
        if all(count is not None for count in token_counts)
        else None
    )
    http_requests = concurrency * rounds
    input_items = http_requests * batch_size
    return {
        "concurrency": concurrency,
        "batch_size": batch_size,
        "http_requests": http_requests,
        "input_items": input_items,
        "elapsed_s": elapsed,
        "requests_per_s": input_items / elapsed,
        "http_requests_per_s": http_requests / elapsed,
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_per_s": (
            prompt_tokens / elapsed if prompt_tokens is not None else None
        ),
        "p50_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "response_mib_per_s": response_bytes / elapsed / (1024.0 * 1024.0),
    }


def render_summary(metadata: dict[str, object], rows: list[dict[str, object]]) -> str:
    lines = [
        "# Embedding server comparison",
        "",
        f"- server: `{metadata['label']}`",
        f"- API adapter: `{metadata['api']}`",
        f"- model: `{metadata['model']}`",
        f"- host: `{metadata['platform']}`",
        f"- input: {metadata['word_count']} generated words plus a unique suffix",
        f"- dimensions: {metadata['dimensions']}",
        "- cache policy: every timed input is unique",
        "",
        "| concurrency | batch | embeddings/s | prompt tok/s | p50 ms | p95 ms |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        tokens_per_s = row["prompt_tokens_per_s"]
        token_text = f"{tokens_per_s:.1f}" if isinstance(tokens_per_s, float) else "n/a"
        lines.append(
            f"| {row['concurrency']} | {row['batch_size']} "
            f"| {row['requests_per_s']:.1f} | {token_text} "
            f"| {row['p50_ms']:.3f} | {row['p95_ms']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--api", choices=("embeddinggemma", "openai", "ollama"), required=True
    )
    parser.add_argument("--path")
    parser.add_argument("--model", default="embeddinggemma-300m")
    parser.add_argument("--label")
    parser.add_argument("--api-key")
    parser.add_argument("--dimensions", type=int, choices=(128, 256, 512, 768),
                        default=768)
    parser.add_argument("--word-count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--concurrency", type=parse_csv_ints,
                        default=parse_csv_ints("1,2,4,8,16,32"))
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.word_count <= 0 or args.batch_size <= 0 or args.rounds <= 0:
        parser.error("word count, batch size, and rounds must be positive")

    endpoint = Endpoint(args.url, args.api, args.path, args.api_key)
    warm_up(
        endpoint, args.api, args.model, args.dimensions, args.word_count,
        args.batch_size, args.warmup_requests
    )
    rows = [
        benchmark(
            endpoint, args.api, args.model, args.dimensions, args.word_count,
            args.batch_size, concurrency, args.rounds
        )
        for concurrency in args.concurrency
    ]
    metadata: dict[str, object] = {
        "schema": 1,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "label": args.label or args.api,
        "api": args.api,
        "url": args.url,
        "path": endpoint.path,
        "model": args.model,
        "platform": platform.platform(),
        "word_count": args.word_count,
        "dimensions": args.dimensions,
        "batch_size": args.batch_size,
        "rounds": args.rounds,
        "warmup_requests": args.warmup_requests,
    }
    timestamp = dt.datetime.now().strftime("%Y-%m-%d/%H%M%S")
    output_dir = args.out_dir or RESULTS_ROOT / timestamp / f"servers-{args.api}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = render_summary(metadata, rows)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
