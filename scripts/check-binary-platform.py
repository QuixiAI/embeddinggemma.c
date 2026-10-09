#!/usr/bin/env python3
"""Reject a binary whose native format or architecture disagrees with its label."""
from pathlib import Path
import struct
import sys

platform, filename = sys.argv[1:]
data = Path(filename).read_bytes()
if platform.startswith("linux-"):
    assert data[:6] == b"\x7fELF\x02\x01", "expected a little-endian ELF64 executable"
    machine = struct.unpack_from("<H", data, 18)[0]
    expected = {"linux-x86_64": 62, "linux-arm64": 183}[platform]
    assert machine == expected, f"wrong ELF architecture: {machine}"
elif platform == "darwin-arm64":
    assert data[:4] == b"\xcf\xfa\xed\xfe", "expected a little-endian Mach-O 64 executable"
    assert struct.unpack_from("<I", data, 4)[0] == 0x0100000C, "expected ARM64 Mach-O"
    assert struct.unpack_from("<I", data, 12)[0] == 2, "expected MH_EXECUTE"
elif platform == "windows-x86_64":
    assert data[:2] == b"MZ", "expected Windows PE executable"
    offset = struct.unpack_from("<I", data, 0x3C)[0]
    assert data[offset:offset + 4] == b"PE\0\0", "missing PE signature"
    assert struct.unpack_from("<H", data, offset + 4)[0] == 0x8664, "expected AMD64 PE"
    assert struct.unpack_from("<H", data, offset + 24)[0] == 0x20B, "expected PE32+"
else:
    raise ValueError(f"unsupported platform: {platform}")
print(f"{filename}: verified native {platform} executable")
