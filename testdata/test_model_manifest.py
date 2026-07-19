#!/usr/bin/env python3
"""Compare the C GGUF reader's tensor inventory with the checked-in manifest."""

import argparse
import json
import subprocess


def fail(message):
    raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    result = subprocess.run(
        [args.binary, args.model],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = json.loads(result.stdout)
    with open(args.manifest, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    kv = manifest["kv"]
    expected_core = {
        "version": kv["GGUF.version"],
        "n_kv": kv["GGUF.kv_count"],
        "n_tensors": kv["GGUF.tensor_count"],
        "alignment": kv.get("general.alignment", 32),
        "data_off": min(tensor["offset"] for tensor in manifest["tensors"]),
    }
    for key, expected in expected_core.items():
        if actual.get(key) != expected:
            fail(f"GGUF {key}: got {actual.get(key)!r}, expected {expected!r}")

    actual_tensors = actual.get("tensors", [])
    expected_tensors = manifest["tensors"]
    if len(actual_tensors) != len(expected_tensors):
        fail(
            f"GGUF tensor count: got {len(actual_tensors)}, "
            f"expected {len(expected_tensors)}"
        )

    for index, (got, source) in enumerate(zip(actual_tensors, expected_tensors)):
        expected = {
            "name": source["name"],
            "type": int(source["type"]),
            "shape": source["shape"],
            "abs_offset": source["offset"],
            "n_bytes": source["n_bytes"],
        }
        if got != expected:
            fail(
                f"GGUF tensor {index} mismatch:\n"
                f"  got:      {json.dumps(got, sort_keys=True)}\n"
                f"  expected: {json.dumps(expected, sort_keys=True)}"
            )

    print(f"GGUF manifest: {len(actual_tensors)} tensor descriptors matched")


if __name__ == "__main__":
    main()
