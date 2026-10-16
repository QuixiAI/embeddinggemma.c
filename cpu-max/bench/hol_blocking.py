#!/usr/bin/env python3
"""Head-of-line blocking on the CPU backend: latency of short text requests
alone and while one large media request is in flight on the same server.

  .venv/bin/python bench/hol_blocking.py --binary ../build/embeddinggemma --threads 12 --cpus E12 \
      --media i1120 --text t16 --out reports/raw/hol_E12.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import http.client  # noqa: E402

from backends import NativeBackend, load_manifest, parts_of  # noqa: E402
from run_bench import BUDGETS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cpus", default="P8")
    ap.add_argument("--media", default="i1120")
    ap.add_argument("--text", default="t16")
    ap.add_argument("--n", type=int, default=20, help="text requests per phase")
    ap.add_argument("--text-clients", type=int, default=1, help="closed-loop text clients during the media request")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    items = {i["id"]: i for i in load_manifest()["items"]}
    env = dict(e.split("=", 1) for e in args.env)
    be = NativeBackend(args.binary, args.threads, BUDGETS.get(args.cpus, args.cpus), "hol", timing=False, env=env)
    texts = [be.prepare(parts_of(items[args.text], v)) for v in range(8)]
    media = be.prepare(parts_of(items[args.media]))
    for t in texts:
        be.run(t)
    be.run(media)  # warm (repack)

    def text_latencies(n):
        out = []
        for i in range(n):
            t0 = time.perf_counter()
            be.run(texts[i % len(texts)])
            out.append((time.perf_counter() - t0) * 1e3)
        return out

    alone = text_latencies(args.n)
    media_ms, done = [], threading.Event()

    media_conn = http.client.HTTPConnection("127.0.0.1", be.port, timeout=3600)

    def media_once():
        t0 = time.perf_counter()
        be.post(media[0], conn=media_conn)
        media_ms.append((time.perf_counter() - t0) * 1e3)
        done.set()

    th = threading.Thread(target=media_once)
    th.start()
    time.sleep(0.05)  # the media request is in flight
    loaded, lk = [], threading.Lock()

    def text_client(c):
        conn = http.client.HTTPConnection("127.0.0.1", be.port, timeout=3600)
        i = c
        while not done.is_set():
            t0 = time.perf_counter()
            be.post(texts[i % len(texts)][0], conn=conn)
            with lk:
                loaded.append((time.perf_counter() - t0) * 1e3)
            i += args.text_clients

    tcs = [threading.Thread(target=text_client, args=(c,)) for c in range(args.text_clients)]
    for t in tcs:
        t.start()
    th.join()
    for t in tcs:
        t.join()
    be.close()
    res = dict(binary=args.binary, threads=args.threads, cpus=args.cpus, media=args.media, text=args.text,
               text_clients=args.text_clients,
               media_ms=media_ms[0], text_alone_ms=alone, text_during_media_ms=loaded,
               text_alone_p50=statistics.median(alone), text_during_media_max=max(loaded),
               text_during_media_p50=statistics.median(loaded), text_during_media_n=len(loaded))
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in res.items() if not isinstance(v, list)}))


if __name__ == "__main__":
    main()
