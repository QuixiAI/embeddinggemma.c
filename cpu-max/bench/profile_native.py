#!/usr/bin/env python3
"""Post manifest items to a native server with profiling env vars and print
its per-stage (EI_MEDIA_TIMING) and per-op (EI_GEMMA2_REF_PROFILE) lines.

  .venv/bin/python bench/profile_native.py --binary ../build/embeddinggemma \
      --threads 8 --cpus P8 --items i70,i280,a15 [--repeat 1] [--env KEY=VAL]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from backends import NativeBackend, load_manifest, parts_of  # noqa: E402
from run_bench import parse_cpus  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--binary", required=True)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--cpus", default="P8")
ap.add_argument("--items", required=True)
ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--env", action="append", default=[])
args = ap.parse_args()
env = {"EI_GEMMA2_REF_PROFILE": "1", "EI_GEMMA2_CPU_PROFILE": "1"}
env.update(dict(kv.split("=", 1) for kv in args.env))
be = NativeBackend(args.binary, args.threads, ",".join(map(str, parse_cpus(args.cpus))), "profile", env=env)
items = {i["id"]: i for i in load_manifest()["items"]}
for iid in args.items.split(","):
    prepared = be.prepare(parts_of(items[iid]))
    for r in range(args.repeat):
        n0 = len(be.stderr_lines)
        emb, t = be.run(prepared)
        time.sleep(0.05)
        print(f"== {iid} run {r}: total {t['total']:.1f} ms, tokens {t['tokens']}")
        for line in be.stderr_lines[n0:]:
            if line.startswith(("ref-profile", "media", "gemm_q8")):
                print("   ", line.rstrip())
be.close()
