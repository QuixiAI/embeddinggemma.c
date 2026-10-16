#!/usr/bin/env python3
"""Run one backend over the workload matrix and write raw JSON + embeddings.

Examples (each backend in its own process so peak RSS / load time are clean):
  .venv/bin/python bench/run_bench.py --backend st --threads 8 --cpus P8
  .venv/bin/python bench/run_bench.py --backend onnx --threads 8 --cpus P8
  .venv/bin/python bench/run_bench.py --backend native --binary third_party/upstream/build/embeddinggemma \
        --name quixi_upstream --threads 8 --cpus P8
Budgets: P8 = one thread per physical P-core (cpus 0,2,..,14); P16 = all P-core threads
(0-15); PE20 = one thread per physical core (8 P + 12 E); ALL = every logical CPU (0-27).
Any taskset list is accepted too.
Single-request latency uses distinct warm runs of item variant 0; throughput (--concurrency)
uses distinct pool variants: concurrent HTTP clients for native servers, one batch of c
inputs for the Python frameworks (their native serving mode).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import resource
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BUDGETS = {"P8": "0,2,4,6,8,10,12,14", "P16": "0-15", "PE20": "0,2,4,6,8,10,12,14,16-27", "ALL": "0-27", "E12": "16-27"}


def parse_cpus(spec: str) -> list[int]:
    spec = BUDGETS.get(spec, spec)
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize(trials: list[dict]) -> dict:
    out = {}
    for key in ("total", "preprocess", "inference", "towers", "text", "client_encode"):
        vals = [t[key] for t in trials if key in t and t[key] == t[key]]
        if vals:
            out[key] = dict(p50=pct(vals, .5), p95=pct(vals, .95), mean=statistics.fmean(vals), min=min(vals),
                            max=max(vals), n=len(vals))
    if "total" in out:
        out["rps_single"] = 1000.0 / out["total"]["mean"]
    return out


def sys_busy_seconds() -> float:
    """CPU-seconds spent busy by the whole host so far (all CPUs, /proc/stat)."""
    f = [int(x) for x in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
    user, nice, system, idle, iowait, irq, softirq, steal = (f + [0] * 8)[:8]
    return (user + nice + system + irq + softirq + steal) / os.sysconf("SC_CLK_TCK")


def wait_quiet(max_cores: float, timeout_s: float, tag: str) -> float:
    """Wait (bounded) until other processes use at most max_cores; returns the last 1 s sample."""
    t_end = time.perf_counter() + timeout_s
    while True:
        b0, t0 = sys_busy_seconds(), time.perf_counter()
        time.sleep(1.0)
        busy = (sys_busy_seconds() - b0) / (time.perf_counter() - t0)
        if busy <= max_cores or time.perf_counter() > t_end:
            if busy > max_cores:
                print(f"{tag} host still busy ({busy:.1f} cores) after {timeout_s:.0f} s; measuring anyway", flush=True)
            return busy
        time.sleep(4.0)


def proc_cpu_seconds() -> float:
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["st", "onnx", "native"], required=True)
    ap.add_argument("--binary", help="native server binary")
    ap.add_argument("--name", help="label for native backends")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cpus", default="P8")
    ap.add_argument("--items", default="all", help="comma list of item ids or 'all'")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=10, help="max measured repeats per item")
    ap.add_argument("--min-repeat", type=int, default=3, help="min measured repeats per item")
    ap.add_argument("--budget-s", type=float, default=60.0, help="stop repeating an item after this many seconds")
    ap.add_argument("--concurrency", default="", help="e.g. 4,8 (throughput runs)")
    ap.add_argument("--requests", type=int, default=16, help="requests per throughput cell (native)")
    ap.add_argument("--video-as-images", action="store_true", help="native: send frames as image parts (HF layout)")
    ap.add_argument("--server-arg", action="append", default=[], help="extra native server argument (repeatable)")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the native server (repeatable)")
    ap.add_argument("--fresh-server", action="store_true",
                    help="native: start a new server per item and record its peak RSS (per-workload memory)")
    ap.add_argument("--quiet-cores", type=float, default=0.0,
                    help="before each item wait until other processes use at most this many cores (0: off)")
    ap.add_argument("--quiet-wait-s", type=float, default=300.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cpus = parse_cpus(args.cpus)
    cpu_spec = ",".join(map(str, cpus))
    if args.backend != "native":
        os.sched_setaffinity(0, cpus)
    from backends import NativeBackend, ONNXBackend, STBackend, load_manifest, parts_of  # noqa: E402
    import numpy as np  # noqa: E402

    manifest = load_manifest()
    items = manifest["items"] if args.items == "all" else [i for i in manifest["items"] if i["id"] in args.items.split(",")]
    load_before = os.getloadavg()
    t_load = time.perf_counter()
    if args.backend == "st":
        be = STBackend(args.threads)
    elif args.backend == "onnx":
        be = ONNXBackend(args.threads)
    else:
        env = dict(kv.split("=", 1) for kv in args.env)
        be = NativeBackend(args.binary, args.threads, cpu_spec, args.name or "native", args.video_as_images,
                           args.server_arg, env)
    name = be.name if args.backend != "native" else (args.name or "native")
    print(f"[{name}] loaded in {be.load_ms:.0f} ms; threads={args.threads} cpus={args.cpus}", flush=True)

    results = []
    embeddings = {}
    concurrency = [int(c) for c in args.concurrency.split(",") if c]
    def new_native():
        env = dict(kv.split("=", 1) for kv in args.env)
        return NativeBackend(args.binary, args.threads, cpu_spec, args.name or "native", args.video_as_images,
                             args.server_arg, env)

    for item in items:
        if args.fresh_server and args.backend == "native":
            be.close()
            be = new_native()
        parts = parts_of(item)
        prepared = be.prepare(parts)
        background = wait_quiet(args.quiet_cores, args.quiet_wait_s, f"[{name}] {item['id']}") \
            if args.quiet_cores > 0 else float("nan")
        for _ in range(args.warmup):
            be.run(prepared)
        trials, embs = [], []
        cpu0 = be.cpu_seconds() if args.backend == "native" else proc_cpu_seconds()
        own0 = proc_cpu_seconds() if args.backend == "native" else 0.0
        sys0 = sys_busy_seconds()
        w0 = time.perf_counter()
        for r in range(args.repeat):
            e, t = be.run(prepared)
            trials.append(t)
            embs.append(e[0])
            if r + 1 >= args.min_repeat and time.perf_counter() - w0 > args.budget_s:
                break
        wall = time.perf_counter() - w0
        cpu1 = be.cpu_seconds() if args.backend == "native" else proc_cpu_seconds()
        own1 = proc_cpu_seconds() if args.backend == "native" else 0.0
        # CPU used by everything except the measured backend (and this client) while it ran.
        foreign = max(0.0, (sys_busy_seconds() - sys0) - (cpu1 - cpu0) - (own1 - own0)) / wall
        e0 = embs[0]
        spread = float(max(np.dot(e0, e) for e in embs) - min(np.dot(e0, e) for e in embs))
        rec = dict(item=item["id"], modality=item["modality"], tokens=trials[0].get("tokens"),
                   expected_tokens=item.get("expected_tokens"), stats=summarize(trials),
                   cpu_util=(cpu1 - cpu0) / wall / len(cpus), norm=float(np.linalg.norm(e0)),
                   finite=bool(np.isfinite(e0).all()), repeat_cos_spread=spread,
                   background_cores_before=background, foreign_cores=foreign,
                   trials=[{k: v for k, v in t.items()} for t in trials])
        embeddings[item["id"]] = e0
        line = f"[{name}] {item['id']:6s} tok={rec['tokens']}/{rec['expected_tokens']} p50={rec['stats']['total']['p50']:9.1f} ms " \
               f"p95={rec['stats']['total']['p95']:9.1f} pre={rec['stats'].get('preprocess', {}).get('p50', float('nan')):7.1f} " \
               f"inf={rec['stats'].get('inference', {}).get('p50', float('nan')):9.1f} cpu={rec['cpu_util']:.2f} " \
               f"foreign={foreign:.2f}"
        print(line, flush=True)

        thr = []
        for c in concurrency:
            if args.backend == "native":
                bodies = [be.prepare(parts_of(item, v % len(item["pool"])))[0] for v in range(len(item["pool"]))]
                lat: list[float] = []
                lock = threading.Lock()
                counter = [0]
                import http.client

                barrier = threading.Barrier(c + 1)

                def client(idx):
                    conn = http.client.HTTPConnection("127.0.0.1", be.port, timeout=600)
                    be.post(bodies[idx % len(bodies)], conn)  # warm connection, not timed
                    barrier.wait()
                    while True:
                        with lock:
                            if counter[0] >= args.requests:
                                break
                            k = counter[0]
                            counter[0] += 1
                        t = time.perf_counter()
                        be.post(bodies[k % len(bodies)], conn)
                        with lock:
                            lat.append((time.perf_counter() - t) * 1000)
                    conn.close()

                ths = [threading.Thread(target=client, args=(i,)) for i in range(c)]
                for th in ths:
                    th.start()
                barrier.wait()  # every client is connected and warm; time from here
                cpu0 = be.cpu_seconds()
                w0 = time.perf_counter()
                for th in ths:
                    th.join()
                wall = time.perf_counter() - w0
                cpu1 = be.cpu_seconds()
                n = len(lat)
                thr.append(dict(concurrency=c, mode="concurrent_http", requests=n, wall_s=wall, rps=n / wall,
                                p50=pct(lat, .5), p95=pct(lat, .95), cpu_util=(cpu1 - cpu0) / wall / len(cpus)))
            else:
                be.run(prepared, batch=c)  # warm the batch shape
                reps = max(2, min(5, args.repeat // 2))
                cpu0 = proc_cpu_seconds()
                w0 = time.perf_counter()
                lat = []
                for _ in range(reps):
                    e, t = be.run(prepared, batch=c)
                    lat.append(t["total"])
                wall = time.perf_counter() - w0
                cpu1 = proc_cpu_seconds()
                thr.append(dict(concurrency=c, mode="batch", batches=reps, wall_s=wall, rps=c * reps / wall,
                                p50=pct(lat, .5), p95=pct(lat, .95), cpu_util=(cpu1 - cpu0) / wall / len(cpus)))
            print(f"[{name}] {item['id']:6s} c={c} rps={thr[-1]['rps']:.2f} p50={thr[-1]['p50']:.1f} ms", flush=True)
        rec["throughput"] = thr
        if args.fresh_server and args.backend == "native":
            rec["peak_rss_mb"] = be.rss_peak_mb()
            print(f"[{name}] {item['id']:6s} peak RSS {rec['peak_rss_mb']:.0f} MB (fresh server)", flush=True)
        results.append(rec)

    peak = be.rss_peak_mb() if args.backend == "native" else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    out = dict(backend=name, kind=args.backend, threads=args.threads, cpus=args.cpus, cpu_list=cpus,
               started=dt.datetime.now().isoformat(timespec="seconds"),
               load_before=load_before, load_after=os.getloadavg(), load_ms=be.load_ms, peak_rss_mb=peak,
               versions=be.versions, warmup=args.warmup, repeat=args.repeat, argv=sys.argv, results=results)
    be.close()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(args.out) if args.out else ROOT / "reports" / "raw" / f"{name}_{args.cpus}_t{args.threads}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    np.savez(path.with_suffix(".npz"), **embeddings)
    print(f"[{name}] peak RSS {peak:.0f} MB; wrote {path}", flush=True)


if __name__ == "__main__":
    main()
