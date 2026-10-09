#!/usr/bin/env python3
"""Verify that a portable x86-64 executable keeps AVX code inside the
runtime-dispatched ei_avx2_* kernels.

Every other function must run on an SSE2-only CPU, so it must not contain a
VEX-encoded instruction (mnemonics starting with "v") or touch a ymm/zmm
register. Fails if the dispatched kernels are missing or AVX leaks elsewhere.
"""
import argparse
import re
import subprocess
import sys

FUNCTION = re.compile(r"^[0-9a-f]+ <(?P<name>[^>]+)>:$")
INSTRUCTION = re.compile(r"^\s*[0-9a-f]+:\s+(?P<mnemonic>[a-z][a-z0-9.]*)\s*(?P<operands>.*)$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary")
    parser.add_argument("--objdump", default="objdump")
    args = parser.parse_args()

    listing = subprocess.run(
        [args.objdump, "-d", "--no-show-raw-insn", args.binary],
        check=True, capture_output=True, text=True,
    ).stdout

    function = None
    dispatched = set()
    leaks = {}
    for line in listing.splitlines():
        match = FUNCTION.match(line)
        if match:
            function = match.group("name")
            continue
        match = INSTRUCTION.match(line)
        if not match or function is None:
            continue
        mnemonic = match.group("mnemonic")
        operands = match.group("operands")
        if not (mnemonic.startswith("v") or "%ymm" in operands or "%zmm" in operands):
            continue
        if function.startswith("ei_avx2_"):
            dispatched.add(function)
        else:
            leaks.setdefault(function, f"{mnemonic} {operands}".strip())

    if leaks:
        for name, example in sorted(leaks.items()):
            print(f"AVX instruction outside dispatched kernels: {name}: {example}",
                  file=sys.stderr)
        return 1
    if not dispatched:
        print("no ei_avx2_* kernels found; runtime dispatch is missing", file=sys.stderr)
        return 1
    print(f"AVX code isolated to {len(dispatched)} dispatched ei_avx2_* kernels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
