#!/usr/bin/env python3
"""Warmed end-to-end CPU/Metal/CUDA/XPU benchmark and result recorder."""

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
        cmd,
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def command_output(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    except OSError:
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def git_label() -> str:
    revision = command_output(["git", "rev-parse", "--short", "HEAD"]) or "uncommitted"
    dirty = command_output(["git", "status", "--porcelain"])
    return revision + ("-dirty" if dirty else "")


def write_summary(rows: list[dict[str, object]], meta: dict[str, object], out_dir: Path) -> None:
    lines = ["# embeddinggemma.c engine benchmarks", ""]
    lines.append(
        f"- `{meta['git']}` | {meta['device']} | warmup/iters "
        f"{meta['warmup']}/{meta['iters']} | tokens `{meta['tokens']}`"
    )
    lines.extend([
        "",
        "| backend | tokens | threads | median ms | p20/p80 ms | tokens/s |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['backend']} | {row['shape']['tokens']} | {row['threads']} "
            f"| {row['target_ms']:.4f} | {row['target_p20_ms']:.4f}/"
            f"{row['target_p80_ms']:.4f} | {row['tokens_per_second']:.1f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model/embeddinggemma-300M-qat-Q4_0.gguf")
    parser.add_argument(
        "--backend", choices=("cpu", "metal", "cuda", "xpu", "both"),
        default="both",
    )
    parser.add_argument("--tokens", default="1,4,8,32,128,512,2048")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--tag", default="engine")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    backends = ("cpu", "metal") if args.backend == "both" else (args.backend,)
    if not args.no_build:
        targets = ["build/perf_engine"]
        if "metal" in backends:
            targets.append("build/perf_engine_metal")
        if "cuda" in backends:
            targets.append("build/perf_engine_cuda")
        if "xpu" in backends:
            targets.append("build/perf_engine_xpu")
        run(["make", *targets])

    rows: list[dict[str, object]] = []
    commands: list[str] = []
    for backend in backends:
        binary = {
            "cpu": "./build/perf_engine",
            "metal": "./build/perf_engine_metal",
            "cuda": "./build/perf_engine_cuda",
            "xpu": "./build/perf_engine_xpu",
        }[backend]
        cmd = [
            binary,
            "--model", args.model,
            "--backend", backend,
            "--tokens", args.tokens,
            "--warmup", str(args.warmup),
            "--iters", str(args.iters),
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
        "warmup": args.warmup,
        "iters": args.iters,
        "threads": args.threads or "default",
        "commands": commands,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": command_output([
            "nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"
        ]) if "cuda" in backends else (
            command_output(["sycl-ls"]) if "xpu" in backends
            else command_output(["sysctl", "-n", "hw.model"])
        ),
        "os": platform.platform(),
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
