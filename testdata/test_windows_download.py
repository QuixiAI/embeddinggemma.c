#!/usr/bin/env python3
"""Test native Windows download spawning without network or household data."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--downloader", required=True)
    args = parser.parse_args()
    binary = str(Path(args.binary).resolve())
    downloader = Path(args.downloader).resolve()
    with tempfile.TemporaryDirectory(prefix="embeddinggemma fictional ") as temp:
        root = Path(temp)
        for mode in ("curl", "fallback", "wget", "missing", "profile", "explicit"):
            directory = root / mode
            directory.mkdir()
            tools = directory / "tools with spaces"
            tools.mkdir()
            if mode not in ("wget", "missing"):
                shutil.copyfile(downloader, tools / "curl.exe")
            if mode in ("fallback", "wget"):
                shutil.copyfile(downloader, tools / "wget.exe")
            env = os.environ.copy()
            for key in ("XDG_CACHE_HOME", "EI_MODEL_PATH", "EI_TEST_CURL_FAIL", "HOME"):
                env.pop(key, None)
            env.update(PATH=str(tools), LOCALAPPDATA=str(directory), USERPROFILE=str(directory))
            command = [binary]
            expected = directory / "embeddinggemma.c" / "embeddinggemma-300M-qat-Q4_0.gguf"
            if mode == "fallback":
                env["EI_TEST_CURL_FAIL"] = "1"
            if mode == "profile":
                env.pop("LOCALAPPDATA")
                expected = directory / ".cache" / "embeddinggemma.c" / expected.name
            if mode == "explicit":
                expected = directory / "explicit path" / "model.gguf"
                command += ["--model", str(expected)]
            result = subprocess.run(command, env=env, capture_output=True, timeout=20)
            assert result.returncode != 0, result.stderr
            if mode == "missing":
                assert b"neither curl nor wget is available" in result.stderr, result.stderr
                assert not expected.exists()
            else:
                assert expected.read_bytes() == b"fictional", result.stderr
                assert b"not a GGUF file" in result.stderr, result.stderr
            assert not list(directory.rglob("*.download.*"))
    print("Windows download paths, quoting, fallback and cleanup: passed")


if __name__ == "__main__":
    main()
