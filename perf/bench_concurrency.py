#!/usr/bin/env python3
"""Benchmark concurrent cache-miss requests through the dynamic batcher."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = Path(__file__).resolve().parent / "results"


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, cwd=REPO_ROOT, env=env, check=True, capture_output=True, text=True
    )


def command_output(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    except OSError:
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def git_label() -> str:
    revision = command_output(["git", "rev-parse", "--short", "HEAD"])
    if not revision or revision == "unknown":
        revision = "uncommitted"
    dirty = command_output(["git", "status", "--porcelain"])
    return revision + ("-dirty" if dirty else "")


def write_summary(rows: list[dict[str, object]], meta: dict[str, object], out_dir: Path) -> None:
    lines = ["# embeddinggemma.c concurrency benchmarks", ""]
    lines.append(
        f"- `{meta['git']}` | {meta['device']} | tokens `{meta['tokens']}` | "
        f"dynamic token batcher | adaptive wait `{meta['adaptive_batch_wait']}` "
        f"| warmup `{meta['warmup']}`"
    )
    lines.extend([
        "",
        "| backend | concurrency | requests | batches | avg batch | req/s | input tok/s | p50/p95 ms | wall ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        shape = row["shape"]
        lines.append(
            f"| {row['backend']} | {shape['concurrency']} | {row['requests']} | "
            f"{row['batches']} | {row['average_batch_size']:.2f} | "
            f"{row['requests_per_second']:.3f} | {row['tokens_per_second']:.1f} | "
            f"{row['latency_p50_ms']:.1f}/{row['latency_p95_ms']:.1f} | "
            f"{row['wall_ms']:.1f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model/embeddinggemma-300M-qat-Q4_0.gguf")
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda", "both"), default="both")
    parser.add_argument(
        "--tokens",
        default="2048",
        help="token count or comma-separated repeating pattern",
    )
    parser.add_argument("--concurrency", default="1,2,4,8,16,32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--min-requests", type=int, default=4)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--tag", default="concurrency")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    backends = ("cpu", "metal") if args.backend == "both" else (args.backend,)
    if not args.no_build:
        targets = ["build/perf_concurrency"]
        if "metal" in backends:
            targets.append("build/perf_concurrency_metal")
        if "cuda" in backends:
            targets.append("build/perf_concurrency_cuda")
        run(["make", *targets])

    rows: list[dict[str, object]] = []
    commands: list[str] = []
    for backend in backends:
        binary = {
            "cpu": "./build/perf_concurrency",
            "metal": "./build/perf_concurrency_metal",
            "cuda": "./build/perf_concurrency_cuda",
        }[backend]
        cmd = [
            binary,
            "--model", args.model,
            "--backend", backend,
            "--tokens", str(args.tokens),
            "--concurrency", args.concurrency,
            "--warmup", str(args.warmup),
            "--min-requests", str(args.min_requests),
        ]
        env = os.environ.copy()
        if args.threads:
            env["EI_THREADS"] = str(args.threads)
        proc = run(cmd, env=env)
        rows.extend(json.loads(line) for line in proc.stdout.splitlines() if line.strip())
        commands.append(" ".join(cmd))

    now = dt.datetime.now()
    out_dir = args.out_dir or RESULTS_ROOT / now.strftime("%Y-%m-%d") / f"{now:%H%M%S}-{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": 1,
        "timestamp": now.isoformat(timespec="seconds"),
        "git": git_label(),
        "backends": list(backends),
        "tokens": args.tokens,
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "min_requests": args.min_requests,
        "threads": args.threads or "default",
        "batch_lookahead": os.environ.get("EI_BATCH_LOOKAHEAD", "1") != "0",
        "adaptive_batch_wait": os.environ.get(
            "EI_ADAPTIVE_BATCH_WAIT", "1"
        ) != "0",
        "commands": commands,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": command_output([
            "nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"
        ]) if "cuda" in backends else command_output(["sysctl", "-n", "hw.model"]),
        "os": command_output(["sw_vers", "-productVersion"]),
        "xcode": command_output(["xcodebuild", "-version"]).replace("\n", " "),
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    with (out_dir / "results.jsonl").open("w") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")
    write_summary(rows, meta, out_dir)
    print(f"wrote {out_dir.relative_to(REPO_ROOT)}/")
    print((out_dir / "summary.md").read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
