#!/usr/bin/env python3
"""Merge raw bench/run_bench.py JSON files into reports/benchmarks.{csv,json}.

  .venv/bin/python scripts/summarize_bench.py reports/raw/final_*.json [--speedup-vs quixi_upstream]

Rows: backend, budget (cpus), threads, item, modality, tokens, p50/p95/mean/min
end-to-end latency, preprocessing and inference p50 where the backend reports
them, single-request req/s, CPU utilization, throughput cells, run peak RSS,
load time, host load averages and versions. With --speedup-vs, a table of
p50 speedups of every backend over the named one (same item and budget).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "reports" / "benchmarks"))
    ap.add_argument("--speedup-vs", default=None)
    ap.add_argument("--best-of", action="store_true",
                    help="several runs of one backend/budget/item: keep the lowest p50 (the run least disturbed "
                         "by other load); the other runs stay listed in 'runs'")
    args = ap.parse_args()
    rows, runs = [], []
    for f in args.files:
        d = json.loads(Path(f).read_text())
        runs.append({k: d[k] for k in ("backend", "kind", "threads", "cpus", "started", "load_before", "load_after",
                                       "load_ms", "peak_rss_mb", "versions", "warmup", "repeat") if k in d} | {"file": f})
        for r in d["results"]:
            s = r["stats"]
            row = dict(backend=d["backend"], budget=d["cpus"], threads=d["threads"], item=r["item"],
                       modality=r["modality"], tokens=r["tokens"], expected_tokens=r.get("expected_tokens"),
                       n=s["total"]["n"], p50_ms=s["total"]["p50"], p95_ms=s["total"]["p95"],
                       mean_ms=s["total"]["mean"], min_ms=s["total"]["min"],
                       preprocess_p50_ms=s.get("preprocess", {}).get("p50"),
                       inference_p50_ms=s.get("inference", {}).get("p50"), rps_single=s.get("rps_single"),
                       cpu_util=r.get("cpu_util"), item_peak_rss_mb=r.get("peak_rss_mb"),
                       run_peak_rss_mb=d.get("peak_rss_mb"), load_ms=d.get("load_ms"),
                       load_before=d.get("load_before", [None])[0], foreign_cores=r.get("foreign_cores"),
                       file=f)
            for t in r.get("throughput", []):
                row[f"rps_c{t['concurrency']}"] = t["rps"]
                row[f"p50_ms_c{t['concurrency']}"] = t["p50"]
            rows.append(row)
    if args.best_of:
        best: dict = {}
        for r in rows:
            k = (r["backend"], r["budget"], r["item"])
            if k not in best or r["p50_ms"] < best[k]["p50_ms"]:
                best[k] = r
        rows = [r for r in rows if best[(r["backend"], r["budget"], r["item"])] is r]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    result = dict(runs=runs, rows=rows)
    if args.speedup_vs:
        base = {(r["budget"], r["item"]): r["p50_ms"] for r in rows if r["backend"] == args.speedup_vs}
        sp = []
        for r in rows:
            b = base.get((r["budget"], r["item"]))
            if b and r["backend"] != args.speedup_vs:
                sp.append(dict(backend=r["backend"], budget=r["budget"], item=r["item"], baseline_p50_ms=b,
                               p50_ms=r["p50_ms"], speedup=b / r["p50_ms"]))
        result["speedup_vs"] = dict(baseline=args.speedup_vs, rows=sp)
        backends = sorted({s["backend"] for s in sp})
        for budget in sorted({s["budget"] for s in sp}):
            items = []
            for s in sp:
                items += [s["item"]] if s["budget"] == budget and s["item"] not in items else []
            print(f"p50 speedup vs {args.speedup_vs} ({budget})")
            print(f"{'item':8s}" + "".join(f"{b:>22s}" for b in backends))
            tab = {(s["backend"], s["item"]): s["speedup"] for s in sp if s["budget"] == budget}
            for it in items:
                print(f"{it:8s}" + "".join(f"{tab.get((b, it), float('nan')):22.2f}" for b in backends))
    out.with_suffix(".json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    print("wrote", out.with_suffix(".csv"), out.with_suffix(".json"), len(rows), "rows")


if __name__ == "__main__":
    main()
