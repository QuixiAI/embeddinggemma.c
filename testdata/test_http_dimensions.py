#!/usr/bin/env python3
"""End-to-end checks for the /api/embed Matryoshka dimensions contract."""

import argparse
import base64
import json
import math
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request


def reserve_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def post(url, payload, expected_status=200):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
            body = response.read()
    except urllib.error.HTTPError as error:
        status = error.code
        body = error.read()
    if status != expected_status:
        raise AssertionError(
            f"expected HTTP {expected_status}, got {status}: {body!r}"
        )
    return json.loads(body)


def wait_until_ready(process, url):
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read().decode("utf-8", "replace")
            raise RuntimeError(f"server exited during startup:\n{stderr}")
        try:
            with urllib.request.urlopen(url + "/api/tags", timeout=0.25):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(0.05)
    raise RuntimeError("server did not become ready within 30 seconds")


def read_raw_response(stream):
    status_line = stream.readline().decode("ascii")
    if not status_line.startswith("HTTP/1.1 200 "):
        raise AssertionError(f"unexpected HTTP status line: {status_line!r}")
    headers = {}
    while True:
        line = stream.readline()
        if line == b"\r\n":
            break
        if not line:
            raise AssertionError("connection closed in response headers")
        name, value = line.decode("ascii").split(":", 1)
        headers[name.lower()] = value.strip().lower()
    length = int(headers["content-length"])
    body = stream.read(length)
    if len(body) != length:
        raise AssertionError("connection closed in response body")
    return headers, json.loads(body)


def check_keep_alive(base_url):
    port = int(base_url.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
        connection.sendall(
            b"GET /api/tags HTTP/1.1\r\nHost: localhost\r\n\r\n"
            b"GET /api/tags HTTP/1.1\r\nHost: localhost\r\n"
            b"Connection: close\r\n\r\n"
        )
        stream = connection.makefile("rb")
        first_headers, first_body = read_raw_response(stream)
        second_headers, second_body = read_raw_response(stream)
        if first_headers.get("connection") != "keep-alive":
            raise AssertionError("first pipelined response did not retain the connection")
        if second_headers.get("connection") != "close":
            raise AssertionError("Connection: close was not honored")
        if first_body != second_body or "models" not in first_body:
            raise AssertionError("pipelined responses were not order-preserving")
        if connection.recv(1) != b"":
            raise AssertionError("server did not close after Connection: close")


def check_dimensions(base_url):
    url = base_url + "/api/embed"
    text = "search_query: Matryoshka dimensions regression test"
    default = post(url, {"model": "embeddinggemma-300m", "input": [text]})
    full = default["embeddings"][0]
    explicit = post(url, {"input": [text], "dimensions": 768})
    if full != explicit["embeddings"][0]:
        raise AssertionError("omitted dimensions must be identical to dimensions=768")
    explicit_float = post(url, {"input": [text], "encoding_format": "float"})
    if full != explicit_float["embeddings"][0]:
        raise AssertionError("omitted encoding must be identical to encoding_format=float")

    for dimensions in (768, 512, 256, 128):
        result = post(url, {"dimensions": dimensions, "input": [text]})
        embedding = result["embeddings"][0]
        if len(embedding) != dimensions:
            raise AssertionError(
                f"dimensions={dimensions} returned {len(embedding)} values"
            )
        norm = math.sqrt(sum(value * value for value in embedding))
        if abs(norm - 1.0) > 2e-5:
            raise AssertionError(f"dimensions={dimensions} has L2 norm {norm}")
        prefix = full[:dimensions]
        prefix_norm = math.sqrt(sum(value * value for value in prefix))
        max_error = max(
            abs(got - want / prefix_norm)
            for got, want in zip(embedding, prefix)
        )
        if max_error > 2e-6:
            raise AssertionError(
                f"dimensions={dimensions} normalized-prefix error {max_error}"
            )

    batch = post(
        url,
        {"input": [text, text + " batch item"], "dimensions": 128},
    )["embeddings"]
    if [len(embedding) for embedding in batch] != [128, 128]:
        raise AssertionError("dimensions must apply to every batch item")

    for dimensions in (768, 512, 256, 128):
        encoded = post(
            url,
            {
                "input": [text],
                "dimensions": dimensions,
                "encoding_format": "base64",
            },
        )["embeddings"][0]
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) != dimensions * 4:
            raise AssertionError(
                f"base64 dimensions={dimensions} decoded to {len(raw)} bytes"
            )
        embedding = struct.unpack(f"<{dimensions}f", raw)
        norm = math.sqrt(sum(value * value for value in embedding))
        if abs(norm - 1.0) > 2e-5:
            raise AssertionError(
                f"base64 dimensions={dimensions} has L2 norm {norm}"
            )
        prefix = full[:dimensions]
        prefix_norm = math.sqrt(sum(value * value for value in prefix))
        max_error = max(
            abs(got - want / prefix_norm)
            for got, want in zip(embedding, prefix)
        )
        if max_error > 2e-6:
            raise AssertionError(
                f"base64 dimensions={dimensions} prefix error {max_error}"
            )

    encoded_batch = post(
        url,
        {
            "input": [text, text + " batch item"],
            "dimensions": 128,
            "encoding_format": "base64",
        },
    )["embeddings"]
    if len(encoded_batch) != 2 or any(
        len(base64.b64decode(item, validate=True)) != 128 * 4
        for item in encoded_batch
    ):
        raise AssertionError("base64 must encode every batch item as float32 bytes")

    invalid = (0, 127, 129, 767, 769, 1024, "128", None, 128.5, -128)
    for dimensions in invalid:
        error = post(
            url,
            {"input": [text], "dimensions": dimensions},
            expected_status=400,
        )
        if "dimensions" not in error.get("error", ""):
            raise AssertionError(f"unexpected error for dimensions={dimensions!r}")

    for encoding in ("binary", "BASE64", "", None, 1):
        error = post(
            url,
            {"input": [text], "encoding_format": encoding},
            expected_status=400,
        )
        if "encoding_format" not in error.get("error", ""):
            raise AssertionError(f"unexpected error for encoding_format={encoding!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--backend", choices=("cpu", "metal", "cuda", "xpu"), required=True
    )
    args = parser.parse_args()

    port = reserve_port()
    command = [
        args.binary,
        "--model",
        args.model,
        "--backend",
        args.backend,
        "--bind",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "4",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        wait_until_ready(process, base_url)
        check_keep_alive(base_url)
        check_dimensions(base_url)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    print(f"HTTP Matryoshka dimensions ({args.backend}): passed")


if __name__ == "__main__":
    main()
