#!/bin/sh

set -eu

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: scripts/release-assets.sh checksums [DIST_DIR]
       scripts/release-assets.sh verify [DIST_DIR]

Writes or verifies the complete set of raw release executables.
EOF
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 "$1" | awk '{print $NF}'
    else
        die 'sha256sum, shasum, or openssl is required'
    fi
}

[ "$#" -ge 1 ] && [ "$#" -le 2 ] || {
    usage >&2
    exit 1
}

command=$1
dist_dir=${2:-dist}
assets='embeddinggemma-darwin-arm64-cpu
embeddinggemma-darwin-arm64-metal
embeddinggemma-linux-x86_64-cpu
embeddinggemma-linux-x86_64-cuda
embeddinggemma-linux-x86_64-rocm
embeddinggemma-linux-x86_64-xpu'

require_assets() {
    printf '%s\n' "$assets" | while IFS= read -r asset; do
        path=$dist_dir/$asset
        [ -f "$path" ] || die "missing release asset: $path"
        [ ! -L "$path" ] || die "release asset must not be a symlink: $path"
        [ -s "$path" ] || die "release asset is empty: $path"
        [ -x "$path" ] || die "release asset is not executable: $path"

        description=$(file -b "$path")
        case "$asset" in
            embeddinggemma-darwin-arm64-*)
                printf '%s\n' "$description" | grep -Eq 'Mach-O.*arm64' ||
                    die "$asset is not an arm64 Mach-O executable: $description"
                ;;
            embeddinggemma-linux-x86_64-*)
                printf '%s\n' "$description" | grep -Eq 'ELF 64-bit.*x86-64' ||
                    die "$asset is not an x86-64 ELF executable: $description"
                ;;
        esac
    done
}

case "$command" in
    checksums)
        require_assets
        temporary=$dist_dir/.SHA256SUMS.tmp
        trap 'rm -f "$temporary"' EXIT HUP INT TERM
        : > "$temporary"
        printf '%s\n' "$assets" | while IFS= read -r asset; do
            printf '%s  %s\n' "$(sha256_file "$dist_dir/$asset")" "$asset"
        done >> "$temporary"
        mv -f "$temporary" "$dist_dir/SHA256SUMS"
        trap - EXIT HUP INT TERM
        printf 'Wrote %s/SHA256SUMS\n' "$dist_dir"
        ;;
    verify)
        require_assets
        checksums=$dist_dir/SHA256SUMS
        [ -f "$checksums" ] || die "missing release asset: $checksums"
        [ "$(awk 'NF { count++ } END { print count + 0 }' "$checksums")" -eq 6 ] ||
            die 'SHA256SUMS must contain exactly six entries'
        printf '%s\n' "$assets" | while IFS= read -r asset; do
            expected=$(awk -v name="$asset" '$2 == name { print $1; count++ } END { if (count != 1) exit 1 }' "$checksums") ||
                die "SHA256SUMS must contain exactly one entry for $asset"
            actual=$(sha256_file "$dist_dir/$asset")
            [ "$actual" = "$expected" ] || die "checksum mismatch: $asset"
        done
        printf 'Verified six raw executables and %s/SHA256SUMS\n' "$dist_dir"
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
