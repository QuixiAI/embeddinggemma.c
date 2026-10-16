#!/usr/bin/env python3
"""Labeled retrieval quality and ranking agreement per backend.

Tasks (scripts/prepare_retrieval_data.py): flickr_t2i, clotho_t2a, msrvtt_t2v,
scifact. Queries get the model card's search prefix ("task: search result |
query: "), media documents none, SciFact documents "title: ... | text: ...";
every backend embeds exactly the same strings, pixels and samples.

  .venv/bin/python scripts/eval_retrieval.py --backend st --threads 16 --cpus P16
  .venv/bin/python scripts/eval_retrieval.py --backend native --binary ../build/embeddinggemma \
      --name fork --threads 16 --cpus P16 [--env EI_GEMMA2_VISION_GELU=tanh ...]
  .venv/bin/python scripts/eval_retrieval.py --report       # metrics + agreement vs st_fp32

Embeddings are cached in reports/raw/retrieval/<name>/<task>_{docs,queries}.npy.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
DATA = ROOT / "data" / "retrieval"
CACHE = ROOT / "reports" / "raw" / "retrieval"
TASKS = ["flickr", "clotho", "msrvtt", "scifact"]
QUERY_PREFIX = "task: search result | query: "


def manifest(task: str) -> dict:
    return json.loads((DATA / task / "manifest.json").read_text())


# ----------------------------------------------------------------------------- embedding
def embed_st(task: dict, threads: int) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from PIL import Image
    import soundfile as sf
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(threads)
    model = SentenceTransformer("google/embeddinggemma-2", device="cpu", model_kwargs={"dtype": torch.float32})
    q = model.encode([QUERY_PREFIX + x["text"] for x in task["queries"]], batch_size=32, normalize_embeddings=True)
    kind = task["doc_modality"]
    docs = []
    t0 = time.time()
    for i, d in enumerate(task["docs"]):
        if kind == "text":
            x, kw = d["text"], {}
        elif kind == "image":
            x, kw = Image.open(ROOT / d["image"]).convert("RGB"), {"processing_kwargs": {"image": {"max_soft_tokens": 280}}}
        elif kind == "audio":
            a, _ = sf.read(ROOT / d["audio"], dtype="float32")
            x, kw = {"audio": {"array": a, "sampling_rate": 16000}}, {}
        else:
            x = {"video": [Image.open(ROOT / f).convert("RGB") for f in d["frames"]]}
            kw = {"processing_kwargs": {"video": {"do_sample_frames": False, "max_soft_tokens": 140}}}
        docs.append(model.encode([x], normalize_embeddings=True, **kw)[0])
        if i % 100 == 0:
            print(f"  st {task['task']} doc {i}/{len(task['docs'])} ({time.time()-t0:.0f}s)", flush=True)
    return q, np.stack(docs)


def embed_native(task: dict, be, concurrency: int) -> tuple[np.ndarray, np.ndarray]:
    import http.client

    def b64(path: str) -> str:
        return base64.b64encode((ROOT / path).read_bytes()).decode()

    def post(body: dict) -> list[list[float]]:
        conn = http.client.HTTPConnection("127.0.0.1", be.port, timeout=3600)
        conn.request("POST", "/v1/embeddings", body=json.dumps(body), headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        data = json.loads(r.read())
        conn.close()
        if r.status != 200:
            raise RuntimeError(data)
        return [e["embedding"] for e in data["data"]]

    texts = [QUERY_PREFIX + x["text"] for x in task["queries"]]
    q = []
    for i in range(0, len(texts), 32):
        q += post({"model": "embeddinggemma-2", "input": texts[i:i + 32]})
    kind = task["doc_modality"]

    def doc_body(d):
        if kind == "text":
            return {"model": "embeddinggemma-2", "input": [d["text"]]}
        if kind == "image":
            part = [{"type": "image", "data": b64(d["image"]), "vision_tokens": 280}]
        elif kind == "audio":
            part = [{"type": "audio", "data": b64(d["audio"])}]
        else:
            part = [{"type": "video", "frames": [b64(f) for f in d["frames"]], "fps": 1, "vision_tokens": 140}]
        return {"model": "embeddinggemma-2", "input": {"content": part}}

    t0 = time.time()
    docs = [None] * len(task["docs"])
    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        futs = {pool.submit(post, doc_body(d)): i for i, d in enumerate(task["docs"])}
        for n, f in enumerate(concurrent.futures.as_completed(futs)):
            docs[futs[f]] = f.result()[0]
            if n % 100 == 0:
                print(f"  native {task['task']} doc {n}/{len(docs)} ({time.time()-t0:.0f}s)", flush=True)
    norm = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)  # noqa: E731
    return norm(np.array(q, dtype=np.float32)), norm(np.array(docs, dtype=np.float32))


# ----------------------------------------------------------------------------- metrics
def metrics(task: dict, q: np.ndarray, d: np.ndarray) -> tuple[dict, np.ndarray]:
    ids = [x["id"] for x in task["docs"]]
    index = {k: i for i, k in enumerate(ids)}
    scores = q @ d.T
    top = np.argsort(-scores, axis=1)[:, :100]
    r1 = r5 = r10 = mrr = ndcg = 0.0
    for qi, qq in enumerate(task["queries"]):
        pos = {index[p] for p in qq["positives"] if p in index}
        ranks = [r for r, di in enumerate(top[qi]) if di in pos]
        first = ranks[0] if ranks else 10**9
        r1 += first < 1
        r5 += first < 5
        r10 += first < 10
        mrr += 1 / (first + 1) if first < 10 else 0
        dcg = sum(1 / np.log2(r + 2) for r in ranks if r < 10)
        idcg = sum(1 / np.log2(r + 2) for r in range(min(len(pos), 10)))
        ndcg += dcg / idcg if idcg else 0
    n = len(task["queries"])
    return dict(queries=n, docs=len(ids), R1=r1 / n, R5=r5 / n, R10=r10 / n, MRR10=mrr / n, nDCG10=ndcg / n), top


def report() -> None:
    names = sorted(p.name for p in CACHE.iterdir() if p.is_dir())
    out: dict = {}
    ref = "st_fp32"
    for task_name in TASKS:
        task = manifest(task_name)
        tops = {}
        for name in names:
            qp, dp = CACHE / name / f"{task_name}_queries.npy", CACHE / name / f"{task_name}_docs.npy"
            if not (qp.exists() and dp.exists()):
                continue
            m, top = metrics(task, np.load(qp), np.load(dp))
            tops[name] = top
            out.setdefault(task_name, {})[name] = m
        if ref in tops:
            for name, top in tops.items():
                overlap = np.mean([len(set(a[:10]) & set(b[:10])) / 10 for a, b in zip(top, tops[ref])])
                same_top1 = float(np.mean(top[:, 0] == tops[ref][:, 0]))
                out[task_name][name].update(top10_overlap_vs_fp32=float(overlap), top1_agreement_vs_fp32=same_top1)
    print(json.dumps(out, indent=1))
    (ROOT / "reports" / "retrieval.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = ["task,backend,queries,docs,R1,R5,R10,MRR10,nDCG10,top10_overlap_vs_fp32,top1_agreement_vs_fp32"]
    for t, bs in out.items():
        for b, m in bs.items():
            lines.append(",".join([t, b] + [f"{m.get(k, float('nan')):.4f}" if isinstance(m.get(k), float) else str(m.get(k, ""))
                                            for k in ["queries", "docs", "R1", "R5", "R10", "MRR10", "nDCG10",
                                                      "top10_overlap_vs_fp32", "top1_agreement_vs_fp32"]]))
    (ROOT / "reports" / "retrieval.csv").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["st", "native"])
    ap.add_argument("--binary")
    ap.add_argument("--name")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--cpus", default="P16")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from run_bench import parse_cpus

    cpus = parse_cpus(args.cpus)
    name = args.name or ("st_fp32" if args.backend == "st" else "native")
    be = None
    if args.backend == "st":
        os.sched_setaffinity(0, cpus)
    else:
        from backends import NativeBackend

        be = NativeBackend(args.binary, args.threads, ",".join(map(str, cpus)), name,
                           env=dict(kv.split("=", 1) for kv in args.env), timing=False)
    for task_name in args.tasks.split(","):
        out = CACHE / name
        out.mkdir(parents=True, exist_ok=True)
        if (out / f"{task_name}_docs.npy").exists():
            print(f"[{name}] {task_name}: cached")
            continue
        task = manifest(task_name)
        t0 = time.time()
        q, d = embed_st(task, args.threads) if be is None else embed_native(task, be, args.concurrency)
        np.save(out / f"{task_name}_queries.npy", q)
        np.save(out / f"{task_name}_docs.npy", d)
        m, _ = metrics(task, q, d)
        print(f"[{name}] {task_name}: {json.dumps(m)} ({time.time()-t0:.0f} s)", flush=True)
        (out / f"{task_name}_meta.json").write_text(json.dumps(dict(seconds=time.time() - t0, argv=sys.argv,
                                                                    metrics=m), indent=1))
    if be:
        be.close()


if __name__ == "__main__":
    main()
