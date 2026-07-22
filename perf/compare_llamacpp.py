#!/usr/bin/env python3
"""Benchmark embeddinggemma.c and llama.cpp with identical HTTP workloads."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import http.client
import json
import math
import os
import platform
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RESULTS_ROOT = Path(__file__).resolve().parent / "results"


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


def base36(value: int) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ""
    while value:
        value, remainder = divmod(value, len(alphabet))
        result = alphabet[remainder] + result
    return result or "0"


def run_output(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(
        command, cwd=cwd, stderr=subprocess.STDOUT, text=True
    ).strip()


def sample_host(ignore_pids: set[int], cpu_threshold: float,
                memory_threshold: float) -> tuple[list[str], float, str]:
    """One ps snapshot: (per-process violations, aggregate CPU %, top line)."""
    output = run_output(["ps", "-axo", "pid=,%cpu=,%mem=,command="])
    violations: list[str] = []
    total_cpu = 0.0
    top_cpu = 0.0
    top_line = ""
    for line in output.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) != 4:
            continue
        try:
            pid = int(fields[0])
            cpu = float(fields[1])
            memory = float(fields[2])
        except ValueError:
            continue
        if pid in ignore_pids:
            continue
        total_cpu += cpu
        if cpu > top_cpu:
            top_cpu = cpu
            top_line = f"pid {pid}: {cpu:.1f}% CPU, {fields[3]}"
        if cpu >= cpu_threshold or memory >= memory_threshold:
            violations.append(
                f"pid {pid}: {cpu:.1f}% CPU, {memory:.1f}% memory, {fields[3]}"
            )
    return violations, total_cpu, top_line


def power_source() -> str:
    """Battery throttles the GPU about 40%, but it throttles both servers
    equally; runs are labeled with the source and a pair is discarded only
    if the source changes mid-cell."""
    try:
        output = run_output(["pmset", "-g", "batt"])
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return "AC" if "AC Power" in output else "battery"


def wait_for_quiet_host(ignore_pids: set[int], cpu_threshold: float,
                        memory_threshold: float, total_cpu_threshold: float,
                        stable_seconds: float) -> None:
    """Wait until the windowed mean host load is acceptable.

    A single one-second blip must not reset the clock: the host is quiet when
    the mean aggregate non-benchmark CPU over the last `stable_seconds`
    samples is below the threshold and at most one sample in that window
    contains a per-process violation.
    """
    window_len = max(3, int(round(stable_seconds)))
    window: list[tuple[bool, float]] = []
    last_report = 0.0
    while True:
        violations, total_cpu, top_line = sample_host(
            ignore_pids, cpu_threshold, memory_threshold
        )
        window.append((bool(violations), total_cpu))
        if len(window) > window_len:
            window.pop(0)
        if len(window) == window_len:
            mean_cpu = sum(sample[1] for sample in window) / len(window)
            violation_samples = sum(1 for sample in window if sample[0])
            if mean_cpu < total_cpu_threshold and violation_samples <= 1:
                return
        now = time.monotonic()
        if now - last_report >= 30.0:
            blocker = violations[0] if violations else (
                f"aggregate non-benchmark CPU {total_cpu:.1f}% "
                f"(top: {top_line})"
            )
            print(f"waiting for quiet host: {blocker}", flush=True)
            last_report = now
        time.sleep(1.0)


def host_contended(ignore_pids: set[int], cpu_threshold: float,
                   memory_threshold: float,
                   total_cpu_threshold: float) -> str | None:
    """Post-measurement check; contention needs two consecutive bad samples."""
    first_violations, first_total, first_top = sample_host(
        ignore_pids, cpu_threshold, memory_threshold
    )
    if not first_violations and first_total < total_cpu_threshold:
        return None
    time.sleep(1.0)
    second_violations, second_total, _ = sample_host(
        ignore_pids, cpu_threshold, memory_threshold
    )
    if first_violations and second_violations:
        return first_violations[0]
    if first_total >= total_cpu_threshold and second_total >= total_cpu_threshold:
        return (
            f"aggregate non-benchmark CPU {first_total:.1f}% then "
            f"{second_total:.1f}% (top: {first_top})"
        )
    return None


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int
    path: str
    api: str

    def connect(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=180)


class ManagedServer:
    def __init__(self, command: list[str], endpoint: Endpoint,
                 health_path: str, log_path: Path):
        self.command = command
        self.endpoint = endpoint
        self.health_path = health_path
        self.log_path = log_path
        self.process: subprocess.Popen[bytes] | None = None
        self.log_file: Any = None

    def __enter__(self) -> "ManagedServer":
        self.log_file = self.log_path.open("wb")
        self.process = subprocess.Popen(
            self.command, stdout=self.log_file, stderr=subprocess.STDOUT
        )
        deadline = time.monotonic() + 180.0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                break
            connection = self.endpoint.connect()
            try:
                connection.request("GET", self.health_path)
                response = connection.getresponse()
                response.read()
                if response.status == 200:
                    return self
            except (ConnectionError, OSError, http.client.HTTPException) as exc:
                last_error = exc
            finally:
                connection.close()
            time.sleep(0.1)
        self.__exit__(None, None, None)
        tail = self.log_path.read_text(errors="replace")[-4000:]
        raise RuntimeError(
            f"server did not become ready: {last_error}\ncommand: "
            f"{' '.join(self.command)}\n{tail}"
        )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log_file is not None:
            self.log_file.close()


def post_json(connection: http.client.HTTPConnection, path: str,
              document: dict[str, object]) -> tuple[int, bytes]:
    payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
    connection.request(
        "POST", path, body=payload, headers={"Content-Type": "application/json"}
    )
    response = connection.getresponse()
    body = response.read()
    return response.status, body


def tokenize(connection: http.client.HTTPConnection, text: str) -> list[int]:
    status, payload = post_json(
        connection,
        "/tokenize",
        {"content": text, "add_special": True, "parse_special": True},
    )
    if status != 200:
        raise RuntimeError(f"tokenization failed with HTTP {status}: {payload[:1000]!r}")
    return [int(token) for token in json.loads(payload)["tokens"]]


def generate_exact_prompts(endpoint: Endpoint, token_counts: list[int],
                           count: int) -> dict[int, list[str]]:
    prompts: dict[int, list[str]] = {}
    connection = endpoint.connect()
    try:
        for token_count in token_counts:
            token_sequences: set[tuple[int, ...]] = set()
            token_prompts: list[str] = []
            for prompt_id in range(count):
                base = f"q{base36(prompt_id)}"
                base_tokens = tokenize(connection, base)
                if len(base_tokens) > token_count:
                    raise RuntimeError(
                        f"cannot fit unique prompt {base!r} in {token_count} tokens"
                    )
                candidate = base + " x" * (token_count - len(base_tokens))
                tokens = tokenize(connection, candidate)
                while len(tokens) < token_count:
                    candidate += " x"
                    tokens = tokenize(connection, candidate)
                if len(tokens) != token_count:
                    raise RuntimeError(
                        f"failed to construct an exact {token_count}-token prompt"
                    )
                sequence = tuple(tokens)
                if sequence in token_sequences:
                    raise RuntimeError(
                        f"tokenization collision while constructing prompt {prompt_id}"
                    )
                token_sequences.add(sequence)
                token_prompts.append(candidate)
            prompts[token_count] = token_prompts
    finally:
        connection.close()
    return prompts


def request_body(api: str, prompt: str, encoding_format: str) -> bytes:
    body: dict[str, object] = {
        "model": "embeddinggemma-300m",
        "input": prompt,
        "encoding_format": encoding_format,
    }
    if api == "embeddinggemma":
        body["dimensions"] = 768
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def send(connection: http.client.HTTPConnection, endpoint: Endpoint,
         body: bytes) -> int:
    connection.request(
        "POST", endpoint.path, body=body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    payload = response.read()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {payload[:1000]!r}")
    return len(payload)


def validate(endpoint: Endpoint, prompt: str,
             expected_tokens: int | None) -> list[float]:
    connection = endpoint.connect()
    try:
        status, payload = post_json(
            connection, endpoint.path,
            json.loads(request_body(endpoint.api, prompt, "float")),
        )
    finally:
        connection.close()
    if status != 200:
        raise RuntimeError(f"validation failed with HTTP {status}: {payload[:1000]!r}")
    document = json.loads(payload)
    if endpoint.api == "llamacpp":
        embeddings = [item["embedding"] for item in document["data"]]
        prompt_tokens = document.get("usage", {}).get("prompt_tokens")
        if expected_tokens is not None and prompt_tokens != expected_tokens:
            raise RuntimeError(
                f"llama.cpp reported {prompt_tokens} tokens; expected {expected_tokens}"
            )
    else:
        embeddings = document["embeddings"]
    if len(embeddings) != 1 or len(embeddings[0]) != 768:
        raise RuntimeError("server did not return one 768-dimensional embedding")
    vector = [float(value) for value in embeddings[0]]
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or abs(norm - 1.0) > 2e-3:
        raise RuntimeError(f"embedding has invalid L2 norm {norm}")
    return vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def run_requests(endpoint: Endpoint, bodies: list[bytes], rounds: int,
                 token_count: int) -> dict[str, int | float]:
    concurrency = len(bodies)
    started: list[float] = []
    barrier = threading.Barrier(
        concurrency, action=lambda: started.append(time.perf_counter())
    )

    def worker(body: bytes) -> tuple[list[float], int]:
        connection = endpoint.connect()
        latencies: list[float] = []
        response_bytes = 0
        barrier.wait()
        try:
            for _ in range(rounds):
                request_started = time.perf_counter()
                response_bytes += send(connection, endpoint, body)
                latencies.append((time.perf_counter() - request_started) * 1000.0)
        finally:
            connection.close()
        return latencies, response_bytes

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, body) for body in bodies]
        results = [future.result() for future in futures]
    elapsed = time.perf_counter() - started[0]
    latencies = [sample for result in results for sample in result[0]]
    requests = concurrency * rounds
    response_bytes = sum(result[1] for result in results)
    return {
        "concurrency": concurrency,
        "token_count": token_count,
        "rounds": rounds,
        "requests": requests,
        "elapsed_s": elapsed,
        "embeddings_per_s": requests / elapsed,
        "tokens_per_s": requests * token_count / elapsed,
        "p50_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "response_mib_per_s": response_bytes / elapsed / (1024.0 * 1024.0),
    }


def benchmark_shape(label: str, endpoint: Endpoint, prompts: list[str],
                    token_count: int, concurrency: int, target_seconds: float,
                    min_rounds: int, max_rounds: int,
                    encoding_format: str) -> dict[str, int | float | str]:
    bodies = [
        request_body(endpoint.api, prompt, encoding_format)
        for prompt in prompts[:concurrency]
    ]
    run_requests(endpoint, bodies, 1, token_count)
    pilot = run_requests(endpoint, bodies, 1, token_count)
    rounds = max(
        min_rounds,
        min(max_rounds, math.ceil(target_seconds / float(pilot["elapsed_s"]))),
    )
    row: dict[str, int | float | str] = run_requests(
        endpoint, bodies, rounds, token_count
    )
    row["server"] = label
    print(
        f"  {label:16s} {token_count:4d} tokens x c{concurrency:<2d}: "
        f"{row['embeddings_per_s']:9.1f} emb/s, "
        f"{row['tokens_per_s']:11.1f} tok/s ({rounds} rounds)",
        flush=True,
    )
    return row


def rows_by_shape(rows: list[dict[str, int | float | str]]) -> dict[tuple[int, int], dict[str, int | float | str]]:
    return {
        (int(row["token_count"]), int(row["concurrency"])): row
        for row in rows
    }


def render_matrix(title: str, token_counts: list[int], concurrencies: list[int],
                  value_fn: Any, suffix: str = "") -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| tokens \\ concurrency | " + " | ".join(str(c) for c in concurrencies) + " |",
        "|---:|" + "---:|" * len(concurrencies),
    ]
    for token_count in token_counts:
        values = [f"{value_fn(token_count, c)}{suffix}" for c in concurrencies]
        lines.append(f"| {token_count:,} | " + " | ".join(values) + " |")
    lines.append("")
    return lines


def render_summary(metadata: dict[str, object],
                   embeddinggemma_rows: list[dict[str, int | float | str]],
                   llamacpp_rows: list[dict[str, int | float | str]]) -> str:
    token_counts = [int(value) for value in metadata["token_counts"]]  # type: ignore[arg-type]
    concurrencies = [int(value) for value in metadata["concurrencies"]]  # type: ignore[arg-type]
    ours = rows_by_shape(embeddinggemma_rows)
    llama = rows_by_shape(llamacpp_rows)
    speedups = [
        float(ours[key]["embeddings_per_s"]) / float(llama[key]["embeddings_per_s"])
        for key in ours
    ]
    lines = [
        "# embeddinggemma.c vs llama.cpp",
        "",
        f"- host: {metadata['host']}",
        f"- power source: {metadata.get('power_source', 'unknown')}",
        f"- model: `{metadata['model']}`",
        f"- output: 768-dimensional `{metadata['encoding_format']}` embeddings",
        f"- timed duration target: {metadata['target_seconds']} seconds per shape",
        "- cache policy: exact-result, response, and llama.cpp prompt caches disabled",
        f"- geometric-mean throughput: {statistics.geometric_mean(speedups):.2f}x llama.cpp",
        f"- range: {min(speedups):.2f}x to {max(speedups):.2f}x llama.cpp",
        "",
    ]
    lines += render_matrix(
        "Throughput relative to llama.cpp",
        token_counts,
        concurrencies,
        lambda tokens, concurrency: (
            f"{float(ours[(tokens, concurrency)]['embeddings_per_s']) / float(llama[(tokens, concurrency)]['embeddings_per_s']):.2f}"
        ),
        "x",
    )
    lines += render_matrix(
        "embeddinggemma.c throughput (embeddings/s)",
        token_counts,
        concurrencies,
        lambda tokens, concurrency: f"{float(ours[(tokens, concurrency)]['embeddings_per_s']):,.1f}",
    )
    lines += render_matrix(
        "llama.cpp throughput (embeddings/s)",
        token_counts,
        concurrencies,
        lambda tokens, concurrency: f"{float(llama[(tokens, concurrency)]['embeddings_per_s']):,.1f}",
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--embeddinggemma-bin", type=Path,
                        default=Path("build/embeddinggemma-metal"))
    parser.add_argument("--llama-server", type=Path,
                        default=Path.home() / "llama.cpp/build/bin/llama-server")
    parser.add_argument("--token-counts", type=parse_csv_ints,
                        default=parse_csv_ints("8,16,32,64,128,256,512,1024,2048"))
    parser.add_argument("--concurrency", type=parse_csv_ints,
                        default=parse_csv_ints("1,2,4,8,16,32"))
    parser.add_argument("--target-seconds", type=float, default=3.0)
    parser.add_argument("--min-rounds", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=1000)
    parser.add_argument("--cooldown-seconds", type=float, default=10.0)
    parser.add_argument("--llama-batch-tokens", type=int, default=2048)
    parser.add_argument("--llama-context-tokens", type=int, default=4096)
    parser.add_argument("--quiet-cpu-percent", type=float, default=50.0)
    parser.add_argument("--quiet-memory-percent", type=float, default=5.0)
    parser.add_argument("--quiet-total-cpu-percent", type=float, default=150.0)
    parser.add_argument("--quiet-stable-seconds", type=float, default=5.0)
    parser.add_argument("--encoding-format", choices=("float", "base64"),
                        default="float")
    parser.add_argument("--fresh-servers", action="store_true",
                        help="restart both servers before every shape so no "
                             "cell inherits accumulated server state")
    parser.add_argument("--loss-retries", type=int, default=3,
                        help="consecutive paired losses required before a "
                             "cell aborts the run; retries are recorded in "
                             "the accepted rows")
    parser.add_argument("--both-orders", action="store_true",
                        help="measure each cell in both A/B orders and average "
                             "each engine's two results, cancelling the fixed "
                             "per-cell order bias (second-measured engine runs "
                             "on a hotter host). Doubles per-cell time; use for "
                             "publishable matrices.")
    parser.add_argument("--inter-engine-cooldown", type=float, default=2.0,
                        help="seconds to settle between the two engines within "
                             "a cell so the second is not measured on a hot "
                             "host; also inserted between A/B passes")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.target_seconds <= 0 or args.min_rounds <= 0 or args.max_rounds < args.min_rounds:
        parser.error("invalid timing or round limits")
    if (args.quiet_cpu_percent <= 0 or args.quiet_memory_percent <= 0 or
            args.quiet_total_cpu_percent <= 0 or args.quiet_stable_seconds < 0):
        parser.error("invalid quiet-host thresholds")
    if max(args.concurrency) > 32:
        parser.error("this comparison configures llama.cpp with 32 slots")
    if max(args.token_counts) > 2048:
        parser.error("EmbeddingGemma supports at most 2048 tokens")
    if args.llama_batch_tokens < max(args.token_counts):
        parser.error("llama.cpp's batch must fit the longest embedding request")
    if args.llama_context_tokens < max(args.token_counts):
        parser.error("llama.cpp's unified context must fit the longest request")

    model = args.model.expanduser().resolve()
    embeddinggemma_bin = args.embeddinggemma_bin.expanduser().resolve()
    llama_server = args.llama_server.expanduser().resolve()
    for path in (model, embeddinggemma_bin, llama_server):
        if not path.exists():
            parser.error(f"not found: {path}")

    timestamp = dt.datetime.now().strftime("%Y-%m-%d/compare-llamacpp-%H%M%S")
    output_dir = args.out_dir or RESULTS_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddinggemma_endpoint = Endpoint(
        "127.0.0.1", 42667, "/api/embed", "embeddinggemma"
    )
    llamacpp_endpoint = Endpoint(
        "127.0.0.1", 42668, "/v1/embeddings", "llamacpp"
    )
    llama_command = [
        str(llama_server), "-m", str(model), "--host", "127.0.0.1",
        "--port", str(llamacpp_endpoint.port), "--embedding", "--pooling", "mean",
        "-ngl", "all", "--parallel", "32", "--ctx-size", str(args.llama_context_tokens),
        "--kv-unified", "--batch-size", str(args.llama_batch_tokens),
        "--ubatch-size", str(args.llama_batch_tokens),
        "--flash-attn", "auto", "--no-cache-prompt", "--cache-ram", "0",
        "--no-cache-idle-slots", "--no-webui", "--log-disable",
    ]
    embeddinggemma_command = [
        str(embeddinggemma_bin), "--model", str(model), "--bind", "127.0.0.1",
        "--port", str(embeddinggemma_endpoint.port), "--workers", "64",
        "--cache-entries", "0", "--response-cache-mb", "0",
    ]

    embeddinggemma_rows: list[dict[str, int | float | str]] = []
    llamacpp_rows: list[dict[str, int | float | str]] = []

    endpoint_rows = {
        "embeddinggemma.c": (embeddinggemma_endpoint, embeddinggemma_rows),
        "llama.cpp": (llamacpp_endpoint, llamacpp_rows),
    }

    def run_ordered_pass(order: list[str], token_count: int,
                         concurrency: int) -> dict[str, dict[str, object]]:
        """Measure both engines in the given order, settling between them so
        the second-measured engine does not inherit the first's thermal or
        power-draw state."""
        rows: dict[str, dict[str, object]] = {}
        for index, label in enumerate(order):
            if index > 0 and args.inter_engine_cooldown:
                time.sleep(args.inter_engine_cooldown)
            endpoint, _ = endpoint_rows[label]
            rows[label] = benchmark_shape(
                label, endpoint, prompts[token_count], token_count,
                concurrency, args.target_seconds, args.min_rounds,
                args.max_rounds, args.encoding_format,
            )
        return rows

    def average_rows(a: dict[str, object], b: dict[str, object]) -> dict[str, object]:
        merged = dict(a)
        for key, value in a.items():
            other = b.get(key)
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and isinstance(other, (int, float))
                    and not isinstance(other, bool)):
                merged[key] = (value + other) / 2.0
        return merged

    def measure_shape(token_count: int, concurrency: int, shape_index: int,
                      ignore_pids: set[int]) -> None:
        base_order = ["embeddinggemma.c", "llama.cpp"]
        if shape_index % 2:
            base_order.reverse()
        loss_attempts = 0
        while True:
            wait_for_quiet_host(
                ignore_pids, args.quiet_cpu_percent,
                args.quiet_memory_percent, args.quiet_total_cpu_percent,
                args.quiet_stable_seconds,
            )
            power_before = power_source()
            passes = {label: [row] for label, row in
                      run_ordered_pass(base_order, token_count, concurrency).items()}
            if args.both_orders:
                if args.inter_engine_cooldown:
                    time.sleep(args.inter_engine_cooldown)
                for label, row in run_ordered_pass(
                        list(reversed(base_order)), token_count, concurrency).items():
                    passes[label].append(row)
            if power_source() != power_before:
                print(
                    f"  retrying {token_count} tokens x c{concurrency}: "
                    f"power source changed mid-cell",
                    flush=True,
                )
                continue
            busy = host_contended(
                ignore_pids, args.quiet_cpu_percent,
                args.quiet_memory_percent, args.quiet_total_cpu_percent,
            )
            if busy:
                print(
                    f"  retrying {token_count} tokens x c{concurrency}: "
                    f"contention appeared ({busy})",
                    flush=True,
                )
                continue
            final = {
                label: (average_rows(runs[0], runs[1]) if len(runs) == 2 else runs[0])
                for label, runs in passes.items()
            }
            ours = float(final["embeddinggemma.c"]["embeddings_per_s"])
            llama = float(final["llama.cpp"]["embeddings_per_s"])
            if llama > ours:
                loss_attempts += 1
                if loss_attempts >= args.loss_retries:
                    raise RuntimeError(
                        "llama.cpp regression at "
                        f"{token_count} tokens x c{concurrency} "
                        f"({loss_attempts} consecutive paired losses): "
                        f"embeddinggemma.c {ours:.1f} emb/s, "
                        f"llama.cpp {llama:.1f} emb/s"
                    )
                print(
                    f"  re-measuring {token_count} tokens x c{concurrency}: "
                    f"paired loss {loss_attempts}/{args.loss_retries} "
                    f"({ours:.1f} vs {llama:.1f} emb/s)",
                    flush=True,
                )
                continue
            for label, row in final.items():
                row["loss_retries"] = loss_attempts
                row["both_orders"] = bool(args.both_orders)
                endpoint_rows[label][1].append(row)
            return

    def check_similarities(similarities: dict[int, float]) -> None:
        if min(similarities.values()) < 0.999:
            raise RuntimeError(
                f"backend output mismatch: minimum cosine "
                f"{min(similarities.values()):.8f}"
            )

    if args.fresh_servers:
        print("generating exact-token fixtures with llama.cpp", flush=True)
        with ManagedServer(
            llama_command, llamacpp_endpoint, "/health",
            output_dir / "llamacpp-fixtures.log"
        ):
            prompts = generate_exact_prompts(
                llamacpp_endpoint, args.token_counts, max(args.concurrency)
            )
            llama_vectors = {
                token_count: validate(
                    llamacpp_endpoint, prompts[token_count][0], token_count
                )
                for token_count in args.token_counts
            }
        with ManagedServer(
            embeddinggemma_command, embeddinggemma_endpoint, "/healthz",
            output_dir / "embeddinggemma-fixtures.log"
        ):
            similarities = {
                token_count: cosine_similarity(
                    llama_vectors[token_count],
                    validate(
                        embeddinggemma_endpoint, prompts[token_count][0], None
                    ),
                )
                for token_count in args.token_counts
            }
        check_similarities(similarities)
        shape_index = 0
        for token_count in args.token_counts:
            for concurrency in args.concurrency:
                if args.cooldown_seconds:
                    time.sleep(args.cooldown_seconds)
                shape_label = f"{token_count}x{concurrency}"
                with ManagedServer(
                    llama_command, llamacpp_endpoint, "/health",
                    output_dir / f"llamacpp-{shape_label}.log"
                ) as llama_process, ManagedServer(
                    embeddinggemma_command, embeddinggemma_endpoint, "/healthz",
                    output_dir / f"embeddinggemma-{shape_label}.log"
                ) as embeddinggemma_process:
                    measure_shape(token_count, concurrency, shape_index, {
                        os.getpid(),
                        llama_process.process.pid,  # type: ignore[union-attr]
                        embeddinggemma_process.process.pid,  # type: ignore[union-attr]
                    })
                shape_index += 1
    else:
        print("starting both Metal servers", flush=True)
        with ManagedServer(
            llama_command, llamacpp_endpoint, "/health",
            output_dir / "llamacpp.log"
        ) as llama_process, ManagedServer(
            embeddinggemma_command, embeddinggemma_endpoint, "/healthz",
            output_dir / "embeddinggemma.log"
        ) as embeddinggemma_process:
            print("generating exact-token fixtures with llama.cpp", flush=True)
            prompts = generate_exact_prompts(
                llamacpp_endpoint, args.token_counts, max(args.concurrency)
            )
            llama_vectors = {
                token_count: validate(
                    llamacpp_endpoint, prompts[token_count][0], token_count
                )
                for token_count in args.token_counts
            }
            similarities = {
                token_count: cosine_similarity(
                    llama_vectors[token_count],
                    validate(
                        embeddinggemma_endpoint, prompts[token_count][0], None
                    ),
                )
                for token_count in args.token_counts
            }
            check_similarities(similarities)
            if args.cooldown_seconds:
                time.sleep(args.cooldown_seconds)
            ignore_pids = {
                os.getpid(),
                llama_process.process.pid,  # type: ignore[union-attr]
                embeddinggemma_process.process.pid,  # type: ignore[union-attr]
            }
            shape_index = 0
            for token_count in args.token_counts:
                for concurrency in args.concurrency:
                    measure_shape(
                        token_count, concurrency, shape_index, ignore_pids
                    )
                    shape_index += 1

    try:
        host = run_output(["sysctl", "-n", "machdep.cpu.brand_string"])
    except (OSError, subprocess.CalledProcessError):
        host = platform.platform()
    metadata: dict[str, object] = {
        "schema": 1,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": host,
        "platform": platform.platform(),
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "token_counts": args.token_counts,
        "concurrencies": args.concurrency,
        "target_seconds": args.target_seconds,
        "min_rounds": args.min_rounds,
        "max_rounds": args.max_rounds,
        "encoding_format": args.encoding_format,
        "llama_batch_tokens": args.llama_batch_tokens,
        "llama_context_tokens": args.llama_context_tokens,
        "quiet_cpu_percent": args.quiet_cpu_percent,
        "quiet_memory_percent": args.quiet_memory_percent,
        "quiet_total_cpu_percent": args.quiet_total_cpu_percent,
        "quiet_stable_seconds": args.quiet_stable_seconds,
        "fresh_servers": args.fresh_servers,
        "loss_retries": args.loss_retries,
        "both_orders": args.both_orders,
        "inter_engine_cooldown": args.inter_engine_cooldown,
        "power_source": power_source(),
        "embeddinggemma_command": embeddinggemma_command,
        "llamacpp_command": llama_command,
        "embeddinggemma_git": run_output(["git", "rev-parse", "HEAD"]),
        "llamacpp_git": run_output(
            ["git", "rev-parse", "HEAD"], cwd=llama_server.parents[2]
        ),
        "minimum_cosine_similarity": min(similarities.values()),
        "cosine_similarity_by_tokens": similarities,
        "environment": {
            "python": platform.python_version(),
            "pid": os.getpid(),
        },
    }
    result = {
        "metadata": metadata,
        "embeddinggemma": embeddinggemma_rows,
        "llamacpp": llamacpp_rows,
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    summary = render_summary(metadata, embeddinggemma_rows, llamacpp_rows)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
