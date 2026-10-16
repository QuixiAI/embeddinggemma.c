#!/bin/sh
# Download the EmbeddingGemma 2 Q8_0 GGUF files that CPU-Max runs (text model +
# multimodal projector, ggml-org/embeddinggemma-2-GGUF at the revision used for
# every published result) and verify them against cpu-max/models/SHA256SUMS.
# Needs only curl (or wget) and sha256sum (or shasum).
#
#   cpu-max/scripts/download_models.sh [DIRECTORY]
#
# DIRECTORY defaults to the server's own default model directory,
# $XDG_CACHE_HOME/embeddinggemma.c or ~/.cache/embeddinggemma.c, so that
# `embeddinggemma` started without --model finds the files. Files already
# present with the right checksum are not downloaded again.
set -eu

here=$(cd "$(dirname "$0")/.." && pwd)
sums=$here/models/SHA256SUMS
revision=bfcd298762cc34d0357ece5ebdd31791a3a374d8
base=https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/$revision

die() {
    printf 'download_models: %s\n' "$*" >&2
    exit 1
}

if [ "$#" -ge 1 ]; then
    dir=$1
elif [ -n "${XDG_CACHE_HOME:-}" ]; then
    dir=$XDG_CACHE_HOME/embeddinggemma.c
else
    [ -n "${HOME:-}" ] || die 'HOME is not set; pass a directory'
    dir=$HOME/.cache/embeddinggemma.c
fi

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        die 'sha256sum or shasum is required'
    fi
}

fetch() {
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --retry-delay 2 --connect-timeout 30 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --timeout=30 -O "$2" "$1"
    else
        die 'curl or wget is required'
    fi
}

[ -f "$sums" ] || die "missing $sums"
mkdir -p "$dir"
for file in embeddinggemma-2-Q8_0.gguf mmproj-embeddinggemma-2-Q8_0.gguf; do
    expected=$(awk -v name="$file" '$2 == name { print $1; exit }' "$sums")
    [ -n "$expected" ] || die "no checksum for $file in $sums"
    if [ -f "$dir/$file" ] && [ "$(sha256 "$dir/$file")" = "$expected" ]; then
        printf 'ok        %s (already present)\n' "$dir/$file"
        continue
    fi
    printf 'download  %s\n' "$base/$file"
    fetch "$base/$file" "$dir/$file.part"
    actual=$(sha256 "$dir/$file.part")
    if [ "$actual" != "$expected" ]; then
        rm -f "$dir/$file.part"
        die "checksum mismatch for $file (expected $expected, got $actual)"
    fi
    mv -f "$dir/$file.part" "$dir/$file"
    printf 'ok        %s\n' "$dir/$file"
done
printf 'models in %s\n' "$dir"
