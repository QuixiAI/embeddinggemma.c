#!/usr/bin/env python3
"""Minimal GGUF v3 reader: list metadata keys and tensor (name, type, shape, bytes).

Usage: python3 scripts/gguf_inspect.py model.gguf [--summary]
"""
import struct
import sys
from collections import Counter, defaultdict

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 30: "BF16"}
BLOCK = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q8_0": (32, 34), "Q4_0": (32, 18)}


def read_gguf(path):
    f = open(path, "rb")

    def u32():
        return struct.unpack("<I", f.read(4))[0]

    def u64():
        return struct.unpack("<Q", f.read(8))[0]

    def string():
        return f.read(u64()).decode("utf-8", "replace")

    scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def value(t):
        if t == 8:
            return string()
        if t == 9:
            et, n = u32(), u64()
            if et == 8:
                items = [string() for _ in range(n)]
                return items if n < 64 else f"<{n} strings>"
            fmt = scalar[et]
            size = struct.calcsize(fmt)
            data = f.read(size * n)
            return list(struct.unpack("<" + fmt[1] * n, data)) if n < 64 else f"<{n} values>"
        fmt = scalar[t]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    assert f.read(4) == b"GGUF"
    version = u32()
    n_tensors, n_kv = u64(), u64()
    kv = {}
    for _ in range(n_kv):
        k = string()
        kv[k] = value(u32())
    tensors = []
    for _ in range(n_tensors):
        name = string()
        nd = u32()
        ne = [u64() for _ in range(nd)]
        t = TYPES.get(u32(), "?")
        off = u64()
        tensors.append((name, t, ne, off))
    return version, kv, tensors


def main():
    path = sys.argv[1]
    version, kv, tensors = read_gguf(path)
    print(f"GGUF v{version}: {len(kv)} keys, {len(tensors)} tensors")
    for k, v in kv.items():
        if "tokenizer.ggml" in k:
            continue
        print(f"  {k} = {v}")
    by_prefix = defaultdict(Counter)
    total = Counter()
    for name, t, ne, _ in tensors:
        n = 1
        for d in ne:
            n *= d
        blk, nbytes = BLOCK.get(t, (1, 0))
        prefix = name.split(".")[0] if not name.startswith(("v.blk", "a.blk", "blk")) else ".".join(name.split(".")[:1])
        key = name.split(".")[0]
        by_prefix[key][t] += n
        total[t] += n
        if "--summary" not in sys.argv:
            print(f"  {name:48s} {t:5s} {ne}")
    print("parameters by prefix and type:")
    for p, c in sorted(by_prefix.items()):
        print(f"  {p:28s} " + ", ".join(f"{t}={n/1e6:.2f}M" for t, n in c.items()))
    print("total:", ", ".join(f"{t}={n/1e6:.2f}M" for t, n in total.items()), f"sum={sum(total.values())/1e6:.2f}M")


if __name__ == "__main__":
    main()
