#!/usr/bin/env python3
"""Performance plots from reports/benchmarks.json (scripts/summarize_bench.py)
and reports/quality_embeddings.json (scripts/validate_embeddings.py).

  .venv/bin/python scripts/plot_bench.py --fork fork_<commit> --budget P8   # report plots (PNG)
  .venv/bin/python scripts/plot_bench.py --fork fork_<commit> --readme      # README charts (SVG)
Writes PNGs to reports/plots/ and the README charts to reports/plots/readme/. Colors follow the entity (fixed slot per backend,
validated categorical order); one value axis per chart; log axis where the
range spans orders of magnitude.
"""
from __future__ import annotations

import argparse
import json
import math
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
    ax.set_ylabel("CPU-Max speedup (p50, log)", color=INK2, fontsize=8)
    ax.set_title(f"End-to-end speedup of CPU-Max ({budget}, p50)", color=INK, fontsize=10, loc="left")
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



# ---- README charts: SVG in a light and a dark variant ------------------------
# Colors follow the entity (CPU-Max blue, PyTorch orange, ONNX Runtime aqua);
# chrome and series steps from the dataviz reference palette, validated against
# GitHub's page backgrounds (#ffffff, #0d1117). Text is converted to outlines so
# the SVGs render identically everywhere; each variant has its theme's page color
# as background, and value labels sit on a patch of it so reference lines pass
# behind them.
README_OUT = OUT / "readme"
THEMES = {
    "light": dict(page="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#6b6a65", grid="#e1e0d9",
                  axis="#c3c2b7", cpumax="#2a78d6", pytorch="#eb6834", onnx="#1baf7a"),
    "dark": dict(page="#0d1117", ink="#f0efec", ink2="#c3c2b7", muted="#a3a29b", grid="#2c2c2a",
                 axis="#4a4a46", cpumax="#3987e5", pytorch="#d95926", onnx="#199e70"),
}
WORKLOAD = {"t16": "Text · 16 tokens", "t128": "Text · 128 tokens", "t512": "Text · 512 tokens",
            "i70": "Image · 70 tokens", "i280": "Image · 280 tokens", "i1120": "Image · 1120 tokens",
            "a5": "Audio · 5 s", "a15": "Audio · 15 s", "a30": "Audio · 30 s",
            "v4": "Video · 4 frames", "v8": "Video · 8 frames", "v16": "Video · 16 frames",
            "m_ti": "Text + image", "m_ta": "Text + audio", "m_tia": "Text + image + audio",
            "m_tv": "Text + video"}
COMPETITOR = {"st_fp32": "PyTorch FP32", "onnx_q8": "ONNX Runtime INT8", "quixi_upstream": "upstream (Q8_0)"}
SHORT_COMPETITOR = {"st_fp32": "PyTorch", "onnx_q8": "ONNX RT", "quixi_upstream": "upstream"}
FOOTER = ["Intel Core i7-14700HX, one machine · data: cpu-max/reports/benchmarks.json",
          "CPU-Max timed over HTTP; PyTorch and ONNX Runtime in-process"]
FIG_W = 7.2  # inches; SVG width 691 px (large type relative to width: legible on phones)


def _readme_style():
    plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "svg.fonttype": "path",
                         "svg.hashsalt": "cpu-max", "font.size": 14})


def _best(rows: list, backend: str, item: str):
    vals = [(r["p50_ms"], r["budget"]) for r in rows if r["backend"] == backend and r["item"] == item]
    return min(vals) if vals else None


def _figure(height: float, title: str, subtitle: list[str], footer: list[str], t: dict,
            left: float, right: float, legend_row: bool, xlabel: bool = False):
    """Figure with a header (title, subtitle lines, optional legend row) and a footer; returns (fig, ax)."""
    fig = plt.figure(figsize=(FIG_W, height))
    y = height - 0.30
    fig.text(0.16 / FIG_W, y / height, title, color=t["ink"], fontsize=19, fontweight="bold", va="top")
    y -= 0.44
    for line in subtitle:
        fig.text(0.16 / FIG_W, y / height, line, color=t["ink2"], fontsize=13, va="top")
        y -= 0.26
    legend_y = (y - 0.20) / height
    top = y - (0.58 if legend_row else 0.16)
    bottom = 0.30 + 0.24 * len(footer) + (0.66 if xlabel else 0.44)
    for k, line in enumerate(footer):
        fig.text(0.16 / FIG_W, (0.18 + 0.24 * (len(footer) - 1 - k)) / height, line, color=t["muted"],
                 fontsize=11.5, va="bottom")
    ax = fig.add_axes((left, bottom / height, right - left, (top - bottom) / height))
    ax.set_facecolor("none")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(axis="x", colors=t["muted"], labelsize=13, length=0, pad=6)
    ax.tick_params(axis="y", colors=t["ink"], labelsize=14.5, length=0, pad=10)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax, legend_y


def _legend(fig, t, entries, y):
    import matplotlib.patches as mpatches

    handles = [mpatches.Patch(color=c, label=l) for l, c in entries]
    leg = fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.12 / FIG_W, y), ncol=len(entries),
                     frameon=False, fontsize=13.5, handlelength=1.0, handleheight=0.9, columnspacing=1.4)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])


def _value(ax, x, y, text, t, bold=True, size=14):
    return ax.text(x, y, text, va="center", ha="left", color=t["ink"], fontsize=size, zorder=5,
            fontweight="bold" if bold else "normal",
            bbox=dict(facecolor=t["page"], edgecolor="none", pad=1.2))


def _save(fig, name, theme):
    README_OUT.mkdir(parents=True, exist_ok=True)
    # Opaque page color: seamless on the matching GitHub theme, and still a readable card
    # if a viewer's GitHub theme and browser color scheme disagree.
    fig.patch.set_facecolor(THEMES[theme]["page"])
    fig.savefig(README_OUT / f"{name}-{theme}.svg", format="svg", metadata={"Date": None},
                facecolor=THEMES[theme]["page"])
    plt.close(fig)


def readme_headline(rows, fork, theme, record):
    t = THEMES[theme]
    data = []
    for item in WORKLOAD:
        f = _best(rows, fork, item)
        best = min((_best(rows, b, item)[0], b) for b in COMPETITOR if _best(rows, b, item))
        data.append((item, best[0] / f[0], best[1], f[0], best[0]))
    data.sort(key=lambda d: d[1])  # barh draws bottom-up: largest on top
    speedups = [d[1] for d in data]
    geo = math.exp(sum(math.log(v) for v in speedups) / len(speedups))
    if theme == "light":
        for item, sp, comp, f_ms, o_ms in reversed(data):
            record.append(dict(chart="speedup-vs-fastest", item=item, series=f"vs {COMPETITOR[comp]}",
                               value=round(sp, 4), cpumax_p50_ms=round(f_ms, 2), other_p50_ms=round(o_ms, 2)))
    fig, ax, _ = _figure(9.0, "CPU-Max vs Fastest Tested CPU Alternative",
                         [f"{sum(v > 1 for v in speedups)} of {len(speedups)} faster · geometric mean "
                          f"{geo:.2f}× · minimum {min(speedups):.2f}×",
                          "p50 latency speedup; each backend at its best thread budget"],
                         FOOTER, t, left=0.33, right=0.97, legend_row=False)
    y = list(range(len(data)))
    ax.barh(y, speedups, height=0.64, color=t["cpumax"], zorder=2)
    ax.set_yticks(y, [WORKLOAD[d[0]] for d in data])
    xmax = max(speedups) * 1.62  # room for the value and the competitor's name
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(data) + 0.15)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.axvline(1, color=t["axis"], linewidth=1.2, zorder=1)
    ax.axvline(geo, color=t["ink2"], linewidth=1.3, linestyle=(0, (4, 3)), zorder=3, ymax=0.975)
    for yi, (item, sp, comp, *_rest) in zip(y, data):
        label = _value(ax, sp + xmax * 0.01, yi, f"{sp:.2f}×", t)
        ax.annotate(f" vs {SHORT_COMPETITOR[comp]}", xy=(1, 0.5), xycoords=label, va="center", ha="left",
                    color=t["muted"], fontsize=13, zorder=5,
                    bbox=dict(facecolor=t["page"], edgecolor="none", pad=1.2))
    ax.text(geo, len(data) - 0.25, f"geometric mean {geo:.2f}×", ha="center", va="bottom", color=t["ink"],
            fontsize=13.5, fontweight="bold", zorder=5, bbox=dict(facecolor=t["page"], edgecolor="none", pad=1.5))
    _save(fig, "speedup-vs-fastest", theme)
    return geo, min(speedups)


README_B = ["t512", "i280", "i1120", "a30", "v16", "m_tia"]


def readme_baselines(rows, fork, theme, record):
    t = THEMES[theme]
    series = [("st_fp32", "vs PyTorch FP32", t["pytorch"]), ("onnx_q8", "vs ONNX Runtime INT8", t["onnx"])]
    other = {(b, it): _best(rows, b, it)[0] for b, _, _ in series for it in README_B}
    sp = {(b, it): other[(b, it)] / _best(rows, fork, it)[0] for b, _, _ in series for it in README_B}
    if theme == "light":
        for it in README_B:
            for b, label, _ in series:
                record.append(dict(chart="speedup-vs-pytorch-onnx", item=it, series=label, value=round(sp[(b, it)], 4),
                                   cpumax_p50_ms=round(_best(rows, fork, it)[0], 2), other_p50_ms=round(other[(b, it)], 2)))
    fig, ax, legend_y = _figure(6.8, "CPU-Max Speedup vs PyTorch and ONNX Runtime",
                                ["Representative workloads (all 16 in the benchmark report)",
                                 "p50 latency speedup; each backend at its best thread budget"],
                                FOOTER, t, left=0.33, right=0.97, legend_row=True)
    h = 0.38
    items = list(reversed(README_B))
    for si, (b, label, color) in enumerate(series):
        ys = [i + (0.5 - si) * h * 1.12 for i in range(len(items))]
        xs = [sp[(b, it)] for it in items]
        ax.barh(ys, xs, height=h, color=color, zorder=2)
        for yv, xv in zip(ys, xs):
            _value(ax, xv + 0.07, yv, f"{xv:.2f}×", t, size=13.5)
    _legend(fig, t, [(label, color) for _, label, color in series], legend_y)
    ax.set_yticks(range(len(items)), [WORKLOAD[i] for i in items])
    ax.axvline(1, color=t["axis"], linewidth=1.2, zorder=1)
    top = max(sp.values())
    ax.set_xlim(0, top * 1.18)
    ax.set_xticks(range(0, int(top) + 1))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_ylim(-0.6, len(items) - 0.4)
    _save(fig, "speedup-vs-pytorch-onnx", theme)


RETRIEVAL = [("flickr", "Flickr30k · text→image", "5,000 queries · 1,000 images"),
             ("msrvtt", "MSR-VTT · text→video", "200 queries · 200 videos"),
             ("scifact", "SciFact · text→text", "300 queries · 5,183 docs"),
             ("clotho", "Clotho · text→audio", "521 queries · 120 clips")]


def readme_retrieval(retr, theme, record):
    t = THEMES[theme]
    series = [("fork_official", "CPU-Max Q8_0", t["cpumax"]), ("st_fp32", "FP32 (official)", t["pytorch"])]
    if theme == "light":
        for task, *_ in RETRIEVAL:
            for b, label, _ in series:
                record.append(dict(chart="retrieval", item=task, series=f"{label} R@1", value=retr[task][b]["R1"]))
            record.append(dict(chart="retrieval", item=task, series="CPU-Max top-1 = FP32 top-1",
                               value=retr[task]["fork_official"]["top1_agreement_vs_fp32"]))
    fig, ax, legend_y = _figure(7.0, "Retrieval Quality vs Official FP32",
                                ["Recall@1, a ranking metric (not embedding cosine)",
                                 "Small subsets; Clotho's clips are near-ties for all"],
                                ["Right: queries whose top result equals FP32's",
                                 "Engine 2cd784b (= 5dd7dbf output) · cpu-max/reports/retrieval.json"],
                                t, left=0.40, right=0.78, legend_row=True, xlabel=True)
    h = 0.38
    items = list(reversed(RETRIEVAL))
    for si, (b, label, color) in enumerate(series):
        ys = [i + (0.5 - si) * h * 1.12 for i in range(len(items))]
        xs = [retr[task][b]["R1"] for task, *_ in items]
        ax.barh(ys, xs, height=h, color=color, zorder=2)
        for yv, xv in zip(ys, xs):
            _value(ax, xv + 0.015, yv, f"{xv:.3f}", t, bold=False, size=13)
    _legend(fig, t, [(label, color) for _, label, color in series], legend_y)
    ax.set_yticks(range(len(items)), ["" for _ in items])
    for i, (task, name, size) in enumerate(items):
        ax.text(-0.04, i + 0.15, name, transform=ax.get_yaxis_transform(), ha="right", va="center",
                color=t["ink"], fontsize=14)
        ax.text(-0.04, i - 0.25, size, transform=ax.get_yaxis_transform(), ha="right", va="center",
                color=t["muted"], fontsize=12)
        agree = retr[task]["fork_official"]["top1_agreement_vs_fp32"]
        ax.text(1.10, i, f"{agree * 100:.1f}%", transform=ax.get_yaxis_transform(), ha="left", va="center",
                color=t["ink"], fontsize=14.5, fontweight="bold")
    ax.text(1.10, len(items) - 0.42, "Same\ntop-1", transform=ax.get_yaxis_transform(), ha="left",
            va="bottom", color=t["muted"], fontsize=12, linespacing=1.1)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.5, 1.0])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Recall@1 (higher is better)", color=t["muted"], fontsize=12.5, labelpad=6)
    ax.set_ylim(-0.6, len(items) - 0.4)
    _save(fig, "retrieval", theme)


def readme_charts(fork: str) -> None:
    import csv

    _readme_style()
    rows = json.loads((ROOT / "reports" / "benchmarks.json").read_text())["rows"]
    retr = json.loads((ROOT / "reports" / "retrieval.json").read_text())
    record: list[dict] = []
    for theme in THEMES:
        geo, low = readme_headline(rows, fork, theme, record)
        readme_baselines(rows, fork, theme, record)
        readme_retrieval(retr, theme, record)
    with open(README_OUT / "chart_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chart", "item", "series", "value", "cpumax_p50_ms", "other_p50_ms"])
        w.writeheader()
        w.writerows(record)
    print(f"README charts in {README_OUT}: geometric mean {geo:.4f}x, minimum {low:.4f}x, {len(record)} values")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="P8")
    ap.add_argument("--fork", required=True, help="backend label of the fork run")
    ap.add_argument("--scaling-fork", default="fork_1a3e206", help="backend label in reports/scaling.json")
    ap.add_argument("--readme", action="store_true", help="only the README charts (SVG, light + dark)")
    args = ap.parse_args()
    if args.readme:
        readme_charts(args.fork)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "reports" / "benchmarks.json").read_text())
    rows = {(r["backend"], r["budget"], r["item"]): r for r in data["rows"]}
    names = {args.fork: "CPU-Max (Q8_0)", "st_fp32": "PyTorch FP32 (Sentence Transformers)", "onnx_q8": "ONNX Runtime INT8",
             "quixi_upstream": "upstream embeddinggemma.c (Q8_0)"}
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
        names.update({k: "CPU-Max, weight-page release off" if k.endswith("_norelease") else names.get(k, k) for k in mem})
        memory(mem, names)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
