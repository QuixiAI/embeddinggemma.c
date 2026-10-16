#!/usr/bin/env python3
"""Markdown result tables for the README, generated from the report files.

  .venv/bin/python scripts/render_results.py --fork fork_final > reports/results.md

Reads reports/benchmarks.json (latency), reports/memory.json (peak RSS),
reports/throughput.json, reports/quality_embeddings.json and
reports/retrieval.json (when present). Every number printed comes from those
files; nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
ITEMS = [("t16", "text, 16 tokens"), ("t128", "text, 128 tokens"), ("t512", "text, 512 tokens"),
         ("i70", "image, 70 tokens (336x480)"), ("i280", "image, 280 tokens (672x960)"),
         ("i1120", "image, 1120 tokens (1344x1920)"), ("a5", "audio, 5 s"), ("a15", "audio, 15 s"),
         ("a30", "audio, 30 s"), ("v4", "video, 4 frames"), ("v8", "video, 8 frames"), ("v16", "video, 16 frames"),
         ("m_ti", "text + image"), ("m_ta", "text + audio"), ("m_tia", "text + image + audio"),
         ("m_tv", "text + video")]
BASES = [("st_fp32", "Sentence Transformers FP32"), ("onnx_q8", "ONNX Community Q8"),
         ("quixi_upstream", "QuixiAI upstream Q8")]
SHORT = {"Sentence Transformers FP32": "ST FP32", "ONNX Community Q8": "ONNX Q8", "QuixiAI upstream Q8": "upstream"}


def load(name: str):
    p = REP / name
    return json.loads(p.read_text()) if p.exists() else None


def ms(v: float | None) -> str:
    if v is None:
        return "–"
    return f"{v:,.0f}" if v >= 100 else f"{v:.1f}"


def latency(fork: str) -> list[str]:
    rows = load("benchmarks.json")["rows"]
    by = {(r["backend"], r["budget"], r["item"]): r["p50_ms"] for r in rows}
    out = []
    for budget, title in (("P8", "8 threads on the 8 P-cores (P8)"), ("ALL", "all 28 hardware threads (ALL)")):
        out += [f"**Warm p50 latency, ms — {title}**", "",
                "| workload | this fork | " + " | ".join(n for _, n in BASES) + " | fork vs fastest other, same budget |",
                "|---|---:|" + "---:|" * len(BASES) + "---:|"]
        for item, label in ITEMS:
            f = by.get((fork, budget, item))
            others = [(by.get((b, budget, item)), n) for b, n in BASES]
            if f is None:
                continue
            avail = [(v, n) for v, n in others if v is not None]
            best = min(avail) if avail else None
            sp = f"{best[0] / f:.2f}x" if best else "–"
            out.append(f"| {label} | **{ms(f)}** | " + " | ".join(ms(v) for v, _ in others) + f" | {sp} |")
        out.append("")
    # Best configuration per backend (P8 or ALL, whichever is faster for that backend and workload).
    out += ["**Best thread budget per backend (P8 or ALL), p50 ms**", "",
            "| workload | this fork | " + " | ".join(n for _, n in BASES) + " | fork vs fastest other |",
            "|---|---:|" + "---:|" * len(BASES) + "---:|"]
    logs = []
    for item, label in ITEMS:
        def best_of(b):
            vals = [(by[(b, bud, item)], bud) for bud in ("P8", "ALL") if (b, bud, item) in by]
            return min(vals) if vals else None
        f = best_of(fork)
        if f is None:
            continue
        others = [(best_of(b), n) for b, n in BASES]
        avail = [(v[0], n) for v, n in others if v]
        best = min(avail)
        logs.append(math.log(best[0] / f[0]))
        cell = lambda v: f"{ms(v[0])} ({v[1]})" if v else "–"
        out.append(f"| {label} | **{cell(f)}** | " + " | ".join(cell(v) for v, _ in others)
                   + f" | {best[0] / f[0]:.2f}x vs {SHORT[best[1]]} |")
    out += ["", f"Geometric mean over the {len(logs)} workloads: **{math.exp(sum(logs) / len(logs)):.2f}x** faster than the "
            f"fastest other backend for each workload (minimum {math.exp(min(logs)):.2f}x).", ""]
    return out


def memory() -> list[str]:
    m = load("memory.json")
    if not m:
        return []
    by = {(r["backend"], r["item"]): r.get("item_peak_rss_mb") for r in m["rows"]}
    backends = sorted({b for b, _ in by})
    items = [i for i, _ in ITEMS if any((b, i) in by for b in backends)]
    out = ["**Peak resident memory of a fresh server per workload (MB, P8)**", "",
           "| backend | " + " | ".join(items) + " |", "|---|" + "---:|" * len(items)]
    for b in backends:
        out.append(f"| {b} | " + " | ".join(f"{by[(b, i)]:,.0f}" if by.get((b, i)) else "–" for i in items) + " |")
    return out + [""]


def quality() -> list[str]:
    q = load("quality_embeddings.json")
    if not q:
        return []
    labels = list(q["summary"])
    by = {(r["backend"], r["item"]): r["cosine_vs_fp32"] for r in q["rows"]}
    out = ["**Final-embedding cosine vs the official Sentence Transformers FP32 model (same inputs)**", "",
           "| workload | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for item, label in ITEMS:
        out.append(f"| {label} | " + " | ".join(f"{by[(l, item)]:.5f}" if (l, item) in by else "–" for l in labels) + " |")
    out.append("| **min / mean** | " + " | ".join(
        f"**{q['summary'][l]['min']:.5f}** / {q['summary'][l]['mean']:.5f}" for l in labels) + " |")
    return out + [""]


def retrieval() -> list[str]:
    r = load("retrieval.json")
    if not r:
        return []
    names = sorted({b for t in r.values() for b in t})
    out = ["**Retrieval quality (higher is better; `top1 agr.` = share of queries whose top hit equals the FP32 "
           "model's)**", "",
           "| task | backend | R@1 | R@5 | R@10 | nDCG@10 | top-10 overlap vs FP32 | top1 agr. vs FP32 |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for task, bs in r.items():
        for b in names:
            if b not in bs:
                continue
            m = bs[b]
            f = lambda k: f"{m[k]:.4f}" if isinstance(m.get(k), float) else "–"
            out.append(f"| {task} ({m['queries']} q / {m['docs']} docs) | {b} | {f('R1')} | {f('R5')} | {f('R10')} | "
                       f"{f('nDCG10')} | {f('top10_overlap_vs_fp32')} | {f('top1_agreement_vs_fp32')} |")
    return out + [""]


def throughput() -> list[str]:
    t = load("throughput.json")
    if not t:
        return []
    cells: dict = {}
    for r in t["rows"]:
        for c in (2, 4, 16):
            v = r.get(f"rps_c{c}")
            if v:
                cells.setdefault((r["backend"], r["budget"], r["item"], c), []).append(v)
    if not cells:
        return []
    out = ["**Throughput with concurrent clients (requests/s, distinct inputs; range over the runs)**", "",
           "| backend | budget | workload | clients | req/s |", "|---|---|---|---:|---:|"]
    for (b, bud, it, c), vs in sorted(cells.items(), key=lambda kv: (kv[0][2], kv[0][3], kv[0][1], kv[0][0])):
        rng = f"{min(vs):.2f}" if len(vs) == 1 or abs(max(vs) - min(vs)) < 0.005 else f"{min(vs):.2f}-{max(vs):.2f}"
        out.append(f"| {b} | {bud} | {it} | {c} | {rng} |")
    return out + [""]


def quant() -> list[str]:
    q = load("quant_study.json")
    if not q:
        return []
    out = ["**Weight precision vs quality** (FP32 model with its linear weights rounded per 32-block; cosine of the "
           "final embedding against the unrounded model)", "",
           "| scheme | text | vision | audio | min cosine | worst workload |", "|---|---|---|---|---:|---|"]
    for name, r in q["results"].items():
        worst = min(r["cosine_vs_fp32"], key=r["cosine_vs_fp32"].get)
        sc = r["scheme"]
        out.append(f"| {name} | {sc['text']} | {sc['vision']} | {sc['audio']} | {r['min']:.5f} | {worst} |")
    return out + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", required=True, help="backend label of the fork in reports/benchmarks.json")
    args = ap.parse_args()
    print("\n".join(latency(args.fork) + quality() + retrieval() + memory() + throughput() + quant()))


if __name__ == "__main__":
    main()
