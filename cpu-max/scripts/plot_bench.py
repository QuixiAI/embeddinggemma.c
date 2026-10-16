#!/usr/bin/env python3
"""Performance plots from reports/benchmarks.json (scripts/summarize_bench.py)
and reports/quality_embeddings.json (scripts/validate_embeddings.py).

  .venv/bin/python scripts/plot_bench.py --budget P8
Writes PNGs to reports/plots/. Colors follow the entity (fixed slot per backend,
validated categorical order); one value axis per chart; log axis where the
range spans orders of magnitude.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "plots"
# Validated categorical slots (dataviz reference palette, light mode), fixed per entity.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# One slot per backend in every chart (color follows the entity).
ENTITY_SLOT = {"st_fp32": 1, "onnx_q8": 2, "quixi_upstream": 3}


def slot_of(backend: str) -> str:
    if backend.endswith("_norelease"):
        return SLOT[4]
    return SLOT[ENTITY_SLOT.get(backend, 0)]  # the fork (any commit label) is slot 1 (blue)
GROUPS = [("Text", ["t16", "t128", "t512"]), ("Image", ["i70", "i280", "i1120"]), ("Audio", ["a5", "a15", "a30"]),
          ("Video", ["v4", "v8", "v16"]), ("Mixed", ["m_ti", "m_ta", "m_tia", "m_tv"])]
LABEL = {"t16": "16 tok", "t128": "128 tok", "t512": "512 tok", "i70": "70 tok", "i280": "280 tok",
         "i1120": "1120 tok", "a5": "5 s", "a15": "15 s", "a30": "30 s", "v4": "4 fr", "v8": "8 fr", "v16": "16 fr",
         "m_ti": "txt+img", "m_ta": "txt+aud", "m_tia": "txt+img+aud", "m_tv": "txt+vid"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def latency(rows, backends, names, budget):
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(15, 3.6), facecolor=SURFACE)
    for ax, (title, items) in zip(axes, GROUPS):
        style(ax)
        h = 0.8 / len(backends)
        for bi, b in enumerate(backends):
            for ii, it in enumerate(items):
                r = rows.get((b, budget, it))
                if not r:
                    continue
                y = ii + (bi - (len(backends) - 1) / 2) * h
                ax.barh(y, r["p50_ms"] / 1000, height=h * 0.85, color=slot_of(b), edgecolor=SURFACE, linewidth=1)
        ax.set_yticks(range(len(items)), [LABEL[i] for i in items])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("p50 latency (s, log)", color=INK2, fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=slot_of(b)) for b in backends]
    fig.legend(handles, [names[b] for b in backends], loc="upper center", ncol=len(backends), frameon=False,
               fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / f"latency_{budget}.png", dpi=150)
    plt.close(fig)


def speedup(rows, fork, bases, names, budget):
    items = [i for _, its in GROUPS for i in its]
    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    w = 0.8 / len(bases)
    for bi, (b, slot) in enumerate(bases):
        xs, ys = [], []
        for ii, it in enumerate(items):
            f, r = rows.get((fork, budget, it)), rows.get((b, budget, it))
            if f and r:
                xs.append(ii + (bi - (len(bases) - 1) / 2) * w)
                ys.append(r["p50_ms"] / f["p50_ms"])
        ax.bar(xs, ys, width=w * 0.85, color=slot_of(b), edgecolor=SURFACE, linewidth=1, label=f"vs {names[b]}")
        for x, y in zip(xs, ys):
            if y >= 10 or y < 1:
                ax.text(x, y * 1.08, f"{y:.0f}x" if y >= 10 else f"{y:.2f}x", ha="center", fontsize=6.5, color=INK2)
    ax.axhline(1, color=INK2, linewidth=1)
    ax.set_yscale("log")
    ax.set_xticks(range(len(items)), [LABEL[i] for i in items], rotation=45, ha="right")
    ax.set_ylabel("fork speedup (p50, log)", color=INK2, fontsize=8)
    ax.set_title(f"End-to-end speedup of the fork ({budget})", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"speedup_{budget}.png", dpi=150)
    plt.close(fig)


def scaling(srows, hybrid_rows, fork, hybrid, budgets, items):
    """Speed relative to P8 of the pre-hybrid fork per budget, plus the hybrid-pool fork on ALL."""
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE)
    style(ax)
    # Each series relative to the P8 run of the same commit.
    series = [(bud, srows, fork, bud if bud == "P8" else f"{bud}, plain pool") for bud in budgets] + \
        [("ALL", hybrid_rows, hybrid, "ALL, hybrid pool (final)")]
    w = 0.8 / len(series)
    for si, (bud, rows, name, label) in enumerate(series):
        xs, ys = [], []
        for ii, it in enumerate(items):
            r, base = rows.get((name, bud, it)), rows.get((name, "P8", it))
            if r and base:
                xs.append(ii + (si - (len(series) - 1) / 2) * w)
                ys.append(base["p50_ms"] / r["p50_ms"])
        ax.bar(xs, ys, width=w * 0.85, color=SLOT[si], edgecolor=SURFACE, linewidth=1, label=label)
    ax.axhline(1, color=INK2, linewidth=1)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(range(len(items)), [LABEL[i] for i in items], rotation=30, ha="right")
    ax.set_ylabel("speed relative to P8 (8 P-core threads)", color=INK2, fontsize=8)
    ax.set_title(f"Thread-budget scaling on the hybrid i7-14700HX (higher is faster)\n"
                 f"each series relative to its own P8 run; plain pool = {fork.split('_')[-1]}, before the hybrid P/E pool",
                 color=INK, fontsize=9, loc="left")
    ax.set_ylim(0, 1.6)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "scaling.png", dpi=150)
    plt.close(fig)


def _unused_scaling(rows, fork, budgets, items):
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    style(ax)
    w = 0.8 / len(budgets)
    for bi, bud in enumerate(budgets):
        xs, ys = [], []
        for ii, it in enumerate(items):
            r, base = rows.get((fork, bud, it)), rows.get((fork, "P8", it))
            if r and base:
                xs.append(ii + (bi - (len(budgets) - 1) / 2) * w)
                ys.append(base["p50_ms"] / r["p50_ms"])
        ax.bar(xs, ys, width=w * 0.85, color=SLOT[bi], edgecolor=SURFACE, linewidth=1, label=bud)
    ax.axhline(1, color=INK2, linewidth=1)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(range(len(items)), [LABEL[i] for i in items], rotation=30, ha="right")
    ax.set_ylabel("speed relative to P8 (8 P-core threads)", color=INK2, fontsize=8)
    ax.set_title("Fork thread-budget scaling (higher is faster)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=len(budgets))
    fig.tight_layout()
    fig.savefig(OUT / "scaling.png", dpi=150)
    plt.close(fig)


def memory(mem, names):
    items = [it for it in ["t512", "i280", "i1120", "a30", "v16", "m_tia"]]
    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor=SURFACE)
    style(ax)
    keys = [k for k in mem if any((k, it) in mem[k] for it in items) or mem[k]]
    w = 0.8 / max(len(keys), 1)
    for ki, k in enumerate(keys):
        xs = [ii + (ki - (len(keys) - 1) / 2) * w for ii, it in enumerate(items) if it in mem[k]]
        ys = [mem[k][it] for it in items if it in mem[k]]
        ax.bar(xs, ys, width=w * 0.85, color=slot_of(k), edgecolor=SURFACE, linewidth=1, label=names.get(k, k))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(range(len(items)), [LABEL[i] for i in items])
    ax.set_ylabel("peak RSS of a fresh server (MB)", color=INK2, fontsize=8)
    ax.set_title("Peak resident memory per workload (P8)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "memory.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="P8")
    ap.add_argument("--fork", required=True, help="backend label of the fork run")
    ap.add_argument("--scaling-fork", default="fork_1a3e206", help="backend label in reports/scaling.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "reports" / "benchmarks.json").read_text())
    rows = {(r["backend"], r["budget"], r["item"]): r for r in data["rows"]}
    names = {args.fork: "fork (this work)", "st_fp32": "Sentence Transformers FP32", "onnx_q8": "ONNX Community Q8",
             "quixi_upstream": "QuixiAI upstream Q8"}
    backends = [b for b in (args.fork, "st_fp32", "onnx_q8", "quixi_upstream") if any(k[0] == b for k in rows)]
    latency(rows, backends, names, args.budget)
    speedup(rows, args.fork, [(b, i) for i, b in enumerate(backends) if b != args.fork], names, args.budget)
    sc = json.loads((ROOT / "reports" / "scaling.json").read_text())
    srows = {(r["backend"], r["budget"], r["item"]): r for r in sc["rows"]}
    scaling(srows, rows, args.scaling_fork, args.fork, ["P8", "P16", "PE20", "ALL"],
            ["t16", "t128", "t512", "i70", "i280", "i1120", "a15", "v8", "m_tia"])
    mem: dict = {}
    for r in json.loads((ROOT / "reports" / "memory.json").read_text())["rows"]:
        if r.get("item_peak_rss_mb") and not r["backend"].endswith("_prepack"):  # prepack: table only
            mem.setdefault(r["backend"], {})[r["item"]] = r["item_peak_rss_mb"]
    if mem:
        names.update({k: "fork, weight-page release off" if k.endswith("_norelease") else names.get(k, k) for k in mem})
        memory(mem, names)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
