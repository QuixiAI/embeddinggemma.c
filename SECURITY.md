# Security policy

EmbeddingGemma-2-CPU-Max is a fork of
[QuixiAI/embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c). It
runs an HTTP server that parses JSON, base64, PNG/JPEG (vendored
`stb_image`) and WAV input from clients.

## Supported versions

CPU-Max has not published a release yet; security fixes go to the `main`
branch. The tags `v0.1.0` … `v0.3.1` are upstream releases and are not
maintained here.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/Yazeedx0/EmbeddingGemma-2-CPU-Max/security/advisories/new)
for this repository, not in a public issue. Include the commit
(`embeddinggemma --version`), the request or input that triggers the problem,
and its impact. If the issue is in code this fork has not changed (see
[`CPU_MAX.md`](CPU_MAX.md) for what changed), please also report it to
upstream.

## Deployment notes

* The server has no authentication and, like upstream, listens on
  `0.0.0.0:42666` by default. Run it with `--bind 127.0.0.1` unless it sits
  behind your own network boundary or an authenticating reverse proxy, and
  size the queue, batch and keep-alive limits for your load (`--help`).
* Started without `--model`, the server downloads the model files over HTTPS
  from a pinned Hugging Face revision, without a checksum check;
  `cpu-max/scripts/download_models.sh` verifies them against
  `cpu-max/models/SHA256SUMS`.
* `install.sh` builds from source; it never downloads executables.
