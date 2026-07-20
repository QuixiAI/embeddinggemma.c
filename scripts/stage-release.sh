#!/bin/sh

set -eu

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

if [ "$#" -ne 3 ]; then
    die "usage: $0 VARIANT BINARY DIST_DIR"
fi

variant=$1
binary=$2
dist_dir=$3

[ -f "$binary" ] || die "binary not found: $binary"
[ -x "$binary" ] || die "binary is not executable: $binary"

case "$(uname -s)" in
    Darwin) platform=darwin ;;
    Linux) platform=linux ;;
    *) die "unsupported release operating system: $(uname -s)" ;;
esac

case "$(uname -m)" in
    arm64|aarch64) architecture=arm64 ;;
    x86_64|amd64) architecture=x86_64 ;;
    *) die "unsupported release architecture: $(uname -m)" ;;
esac

case "$platform-$architecture-$variant" in
    darwin-arm64-cpu|darwin-arm64-metal|linux-x86_64-cpu|\
    linux-x86_64-cuda|linux-x86_64-rocm|linux-x86_64-xpu) ;;
    *) die "unsupported release asset: $platform-$architecture-$variant" ;;
esac

mkdir -p "$dist_dir"
asset="embeddinggemma-$platform-$architecture-$variant"
temporary="$dist_dir/.$asset.tmp"
trap 'rm -f "$temporary"' EXIT HUP INT TERM

cp "$binary" "$temporary"
chmod 0755 "$temporary"

if [ "$platform" = darwin ]; then
    strip -x "$temporary"
    codesign --force --sign - "$temporary"
    codesign --verify --verbose=2 "$temporary"
    if [ "$variant" = metal ]; then
        otool -l "$temporary" | grep -q '__metallib' ||
            die 'Metal release binary has no embedded __metallib section'
    fi
else
    strip --strip-unneeded "$temporary"
fi

mv -f "$temporary" "$dist_dir/$asset"
trap - EXIT HUP INT TERM
file "$dist_dir/$asset"
printf 'Staged %s\n' "$dist_dir/$asset"
