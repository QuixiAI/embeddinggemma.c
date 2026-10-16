#!/usr/bin/env python3
"""Final-embedding agreement of every backend with the official FP32 reference.

Reads the .npz embedding dumps that bench/run_bench.py writes next to each raw
JSON (one 768-d embedding per workload item) and reports, per item, the cosine
similarity with the Sentence Transformers FP32 embedding of the same input, plus
validity checks (finite, unit norm, 768 dims).

  .venv/bin/python scripts/validate_embeddings.py \
      --ref reports/raw/baseline_st_fp32_P8_t8.npz \
      --run onnx_q8=reports/raw/baseline_onnx_q8_P8_t8.npz \
      --run upstream=reports/raw/m1_upstream_P8_t8.npz --run fork=reports/raw/m1_fork_P8_t8.npz \
      --out reports/quality_embeddings
Writes <out>.csv and <out>.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="FP32 reference .npz")
    ap.add_argument("--run", action="append", default=[], help="label=path.npz (repeatable; several npz per label "
                                                               "may be joined with '+')")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ref = dict(np.load(args.ref))
    rows, summary = [], {}
    for spec in args.run:
        label, paths = spec.split("=", 1)
        emb: dict[str, np.ndarray] = {}
        for p in paths.split("+"):
            emb.update(dict(np.load(p)))
        cos_all = []
        for item, r in ref.items():
            if item not in emb:
                continue
            e = emb[item].astype(np.float64)
            rr = r.astype(np.float64)
            cos = float(e @ rr / (np.linalg.norm(e) * np.linalg.norm(rr)))
            rows.append(dict(backend=label, item=item, cosine_vs_fp32=round(cos, 6), dims=int(e.shape[0]),
                             norm=round(float(np.linalg.norm(e)), 6), finite=bool(np.isfinite(e).all())))
            cos_all.append(cos)
        summary[label] = dict(items=len(cos_all), min=round(min(cos_all), 6) if cos_all else None,
                              mean=round(float(np.mean(cos_all)), 6) if cos_all else None)
    items = list(ref)
    labels = [s.split("=", 1)[0] for s in args.run]
    table = {(r["backend"], r["item"]): r["cosine_vs_fp32"] for r in rows}
    print(f"{'item':8s}" + "".join(f"{l:>14s}" for l in labels))
    for it in items:
        print(f"{it:8s}" + "".join(f"{table.get((l, it), float('nan')):14.6f}" for l in labels))
    print("summary:", json.dumps(summary))
    bad = [r for r in rows if not r["finite"] or r["dims"] != 768 or abs(r["norm"] - 1) > 1e-3]
    print("invalid embeddings:", bad if bad else "none")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out.with_suffix(".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        out.with_suffix(".json").write_text(json.dumps(dict(reference=args.ref, runs=args.run, summary=summary,
                                                            rows=rows), indent=1) + "\n")


if __name__ == "__main__":
    main()
