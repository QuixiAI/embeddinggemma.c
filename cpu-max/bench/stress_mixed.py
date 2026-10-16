#!/usr/bin/env python3
"""Concurrency stress for the CPU backend: several clients send a random mix
of text and media requests; every response must equal (bit for bit) the
embedding of the same request sent alone. Catches deadlocks (timeouts) and
cross-request interference in the engine lock / tower yield paths.

  .venv/bin/python bench/stress_mixed.py --binary ../build/embeddinggemma --threads 12 --cpus E12
"""
from __future__ import annotations

import argparse
import http.client
import json
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends import NativeBackend, load_manifest, parts_of  # noqa: E402
from run_bench import BUDGETS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--cpus", default="E12")
    ap.add_argument("--items", default="t16,t128,i70,a5,m_ta,m_tia,v4")
    ap.add_argument("--clients", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--env", action="append", default=[])
    args = ap.parse_args()
    items = {i["id"]: i for i in load_manifest()["items"]}
    env = dict(e.split("=", 1) for e in args.env)
    be = NativeBackend(args.binary, args.threads, BUDGETS.get(args.cpus, args.cpus), "stress", timing=False, env=env)
    reqs = []
    for it in args.items.split(","):
        for v in range(2):
            body = be.prepare(parts_of(items[it], v))[0]
            reqs.append((f"{it}_{v}", body, np.array(be.post(body)["data"][0]["embedding"], dtype=np.float32)))
    errors, counts, lock = [], {}, threading.Lock()
    stop = time.time() + args.seconds

    def client(seed):
        rng = random.Random(seed)
        conn = http.client.HTTPConnection("127.0.0.1", be.port, timeout=300)
        while time.time() < stop:
            name, body, ref = rng.choice(reqs)
            try:
                got = np.array(be.post(body, conn=conn)["data"][0]["embedding"], dtype=np.float32)
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(f"{name}: {e!r}")
                return
            with lock:
                counts[name] = counts.get(name, 0) + 1
                if not np.array_equal(got, ref):
                    errors.append(f"{name}: differs from the solo embedding (max abs {np.abs(got - ref).max():.3g})")

    ths = [threading.Thread(target=client, args=(s,)) for s in range(args.clients)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    alive = be.proc.poll() is None
    be.close()
    print(json.dumps(dict(requests=sum(counts.values()), per_item=counts, errors=errors[:10], n_errors=len(errors),
                          server_alive=alive)))
    sys.exit(1 if errors or not alive else 0)


if __name__ == "__main__":
    main()
