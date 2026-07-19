#!/usr/bin/env python3
"""Native embeddinggemma.c kernel benchmark runner (schema v1).

Run from the repo root:

    python3 perf/bench_kernels.py --preset smoke
    python3 perf/bench_kernels.py --preset quick --kernel q4gemv,rms_norm

Each run writes run metadata, JSONL rows, and a Markdown summary under
perf/results/YYYY-MM-DD/<run-id>/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = Path(__file__).resolve().parent / "results"


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def git_label() -> str:
    rev = run(["git", "rev-parse", "--short", "HEAD"], check=False)
    label = rev.stdout.strip() if rev.returncode == 0 else "uncommitted"
    status = run(["git", "status", "--porcelain"], check=False)
    if status.stdout.strip():
        label += "-dirty"
    return label


def compiler_version() -> str:
    cc = run(["make", "-s", "print-cc-version"], check=False)
    if cc.returncode == 0 and cc.stdout.strip():
        return cc.stdout.strip().splitlines()[0]
    fallback = run(["cc", "--version"], check=False)
    return fallback.stdout.strip().splitlines()[0] if fallback.stdout.strip() else "unknown"


def cpu_name() -> str:
    if sys.platform == "darwin":
        out = run(["sysctl", "-n", "machdep.cpu.brand_string"], check=False)
        if out.stdout.strip():
            return out.stdout.strip()
        out = run(["sysctl", "-n", "hw.model"], check=False)
        if out.stdout.strip():
            return out.stdout.strip()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def default_out_dir(preset: str) -> Path:
    now = dt.datetime.now()
    return RESULTS_ROOT / now.strftime("%Y-%m-%d") / f"{now:%H%M%S}-cpu-{preset}"


def shape_str(shape: dict[str, object]) -> str:
    return "x".join(str(v) for v in shape.values())


def write_summary(rows: list[dict[str, object]], meta: dict[str, object], out_dir: Path) -> None:
    lines = ["# embeddinggemma.c kernel benchmarks", ""]
    lines.append(
        f"- `{meta['git']}` · {meta['cpu']} · backend `cpu` · preset `{meta['preset']}` "
        f"· warmup/iters {meta['warmup']}/{meta['iters']}"
    )
    lines.append("")
    lines.append(
        "| kernel | variant | shape | ms | p20/p80 ms | CV | GB/s | W-GB/s | GFLOP/s | rel err | route |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        route = row.get("quant_kernel") or row.get("cpu_kernel") or ""
        lines.append(
            f"| {row['kernel']} | {row['variant']} | {shape_str(row['shape'])} "
            f"| {row['target_ms']:.6f} | {row['target_p20_ms']:.6f}/{row['target_p80_ms']:.6f} "
            f"| {row['target_cv']:.4f} | {row.get('gbps', 0.0):.1f} "
            f"| {row.get('weight_gbps', 0.0):.1f} | {row.get('gflops', 0.0):.1f} "
            f"| {row.get('max_rel_err', 0.0):.2e} | {route} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=("smoke", "quick", "comprehensive"), default="smoke")
    ap.add_argument("--kernel", default="all", help="comma list, e.g. q4dot,q4gemv,rms_norm")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--binary", default=str(REPO_ROOT / "build" / "perf_kernels"))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    binary = Path(args.binary)
    if not args.no_build:
        run(["make", str(binary.relative_to(REPO_ROOT) if binary.is_relative_to(REPO_ROOT) else binary)])

    cmd = [
        str(binary),
        "--preset",
        args.preset,
        "--kernel",
        args.kernel,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
    ]
    proc = run(cmd)
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]

    out_dir = args.out_dir or default_out_dir(args.preset)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": 1,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "git": git_label(),
        "backend": "cpu",
        "preset": args.preset,
        "kernel": args.kernel,
        "warmup": args.warmup,
        "iters": args.iters,
        "command": " ".join(cmd),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu_name(),
        "compiler": compiler_version(),
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    with (out_dir / "results.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    write_summary(rows, meta, out_dir)

    print(f"wrote {out_dir.relative_to(REPO_ROOT)}/")
    print((out_dir / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
