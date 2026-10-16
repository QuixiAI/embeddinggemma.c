#!/bin/sh
# EmbeddingGemma-2-CPU-Max installer: builds the CPU server from source.
#
# CPU-Max publishes no prebuilt binaries yet, so this script never downloads
# executables: it builds this repository's CPU engine with the system C
# compiler, checks that the result identifies itself as CPU-Max, and installs it
# as `embeddinggemma-cpu-max` (a different name from upstream's
# `embeddinggemma`, so an existing upstream install is never replaced).
#
#   curl -fsSL https://raw.githubusercontent.com/Yazeedx0/EmbeddingGemma-2-CPU-Max/main/install.sh | sh
#   ./install.sh                      # from a checkout: builds that checkout
#
# GPU variants (Metal, CUDA, ROCm, XPU) are unchanged upstream code and are not
# built here; for those use upstream's installer (see --variant below).

set -eu

repo=${CPU_MAX_REPO:-https://github.com/Yazeedx0/EmbeddingGemma-2-CPU-Max.git}
ref=${CPU_MAX_REF:-main}
install_dir=${CPU_MAX_INSTALL_DIR:-}
name=${CPU_MAX_NAME:-embeddinggemma-cpu-max}
source_dir=
with_models=0
jobs=

usage() {
    cat <<'EOF'
Usage: install.sh [--ref REF] [--source DIR] [--install-dir DIR] [--name NAME]
                  [--with-models] [--jobs N] [--variant cpu]

Builds the EmbeddingGemma-2-CPU-Max CPU server from source and installs it as
~/.local/bin/embeddinggemma-cpu-max (by default).

  --ref REF          branch, tag or commit to build when cloning (default: main)
  --source DIR       build an existing checkout instead of cloning
                     (default when the script runs from a checkout)
  --install-dir DIR  installation directory (default: ~/.local/bin)
  --name NAME        installed executable name (default: embeddinggemma-cpu-max)
  --with-models      also download the EmbeddingGemma 2 Q8_0 model files (1.0 GB,
                     SHA-256 verified) into the server's default model directory
  --jobs N           parallel build jobs (default: number of CPUs)
  --variant cpu      only the CPU variant exists in CPU-Max; GPU variants are
                     upstream's and are rejected with instructions

Requirements: a C11 compiler (gcc or clang), make, git (when cloning),
curl or wget and sha256sum or shasum (for --with-models).

Environment overrides: CPU_MAX_REF, CPU_MAX_REPO, CPU_MAX_INSTALL_DIR, CPU_MAX_NAME.
EOF
}

die() {
    printf 'cpu-max installer: %s\n' "$*" >&2
    exit 1
}

need_value() {
    [ "$#" -ge 2 ] || die "$1 requires a value"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --ref) need_value "$@"; ref=$2; shift 2 ;;
        --source) need_value "$@"; source_dir=$2; shift 2 ;;
        --install-dir) need_value "$@"; install_dir=$2; shift 2 ;;
        --name) need_value "$@"; name=$2; shift 2 ;;
        --jobs) need_value "$@"; jobs=$2; shift 2 ;;
        --with-models) with_models=1; shift ;;
        --variant)
            need_value "$@"
            case "$2" in
                cpu|auto) ;;
                metal|cuda|rocm|xpu)
                    die "the $2 variant is not built by CPU-Max (its GPU backends are unchanged upstream code).
Install upstream's binary knowingly instead:
  curl -fsSL https://raw.githubusercontent.com/QuixiAI/embeddinggemma.c/main/install.sh | sh -s -- --variant $2" ;;
                *) die "invalid variant: $2" ;;
            esac
            shift 2
            ;;
        --version)
            die '--version selected an upstream release binary in the original installer; CPU-Max builds from source (use --ref)' ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

case "$name" in
    ''|*/*) die "invalid executable name: $name" ;;
esac
case "$ref" in
    ''|-*|*[!A-Za-z0-9._/-]*) die "invalid ref: $ref" ;;
esac
case "${jobs:-1}" in
    *[!0-9]*|0) die "invalid --jobs: $jobs" ;;
esac

if [ -z "$install_dir" ]; then
    [ -n "${HOME:-}" ] || die 'HOME is not set; use --install-dir'
    install_dir=$HOME/.local/bin
fi

platform=$(uname -s)
architecture=$(uname -m)
case "$platform-$architecture" in
    Linux-x86_64|Linux-amd64)
        ;;
    Linux-aarch64|Linux-arm64|Darwin-arm64|Darwin-x86_64)
        printf '%s\n' \
            "cpu-max installer: warning: $platform-$architecture is not tested by CPU-Max." \
            "  The CPU-Max kernels are x86-64 (AVX2 / AVX-VNNI); here the build uses upstream's" \
            "  portable or NEON paths, and CPU-Max's published results do not apply." >&2
        ;;
    *)
        die "unsupported platform: $platform-$architecture (CPU-Max builds on Linux x86-64; see README)"
        ;;
esac

command -v make >/dev/null 2>&1 || die 'make is required'
if [ -z "${CC:-}" ]; then
    for compiler in cc gcc clang; do
        if command -v "$compiler" >/dev/null 2>&1; then
            CC=$compiler
            break
        fi
    done
fi
[ -n "${CC:-}" ] || die 'a C compiler (cc, gcc or clang) is required'
export CC

if [ -z "$jobs" ]; then
    jobs=$( (getconf _NPROCESSORS_ONLN || nproc || echo 2) 2>/dev/null | head -n 1)
fi

tmpdir=
staged=
cleanup() {
    [ -n "$tmpdir" ] && rm -rf "$tmpdir"
    [ -n "$staged" ] && rm -f "$staged"
    return 0
}
trap cleanup EXIT HUP INT TERM

# Default to the checkout this script belongs to, if it is one.
if [ -z "$source_dir" ] && [ -f "$0" ]; then
    script_dir=$(cd "$(dirname "$0")" && pwd)
    if [ -f "$script_dir/CPU_MAX.md" ] && [ -f "$script_dir/Makefile" ] && [ -f "$script_dir/src/cpu_max_version.h" ]; then
        source_dir=$script_dir
    fi
fi

if [ -n "$source_dir" ]; then
    [ -f "$source_dir/src/cpu_max_version.h" ] || die "$source_dir is not an EmbeddingGemma-2-CPU-Max checkout"
    printf 'Building from %s\n' "$source_dir"
else
    command -v git >/dev/null 2>&1 || die 'git is required to fetch the source (or use --source)'
    tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/cpu-max-install.XXXXXX") || die 'could not create a temporary directory'
    source_dir=$tmpdir/src
    printf 'Fetching %s (%s)\n' "$repo" "$ref"
    if ! git clone --quiet --depth 1 --branch "$ref" "$repo" "$source_dir" 2>/dev/null; then
        rm -rf "$source_dir"
        git clone --quiet "$repo" "$source_dir" || die "could not clone $repo"
        git -C "$source_dir" checkout --quiet "$ref" || die "ref not found: $ref"
    fi
fi

printf 'Building the CPU server with %s (%s jobs)\n' "$CC" "$jobs"
make -C "$source_dir" -j "$jobs" build/embeddinggemma CC="$CC" >/dev/null ||
    die "build failed; run 'make' in $source_dir to see the errors"

binary=$source_dir/build/embeddinggemma
identity=$("$binary" --version 2>/dev/null) || die 'the built binary does not run'
case "$identity" in
    EmbeddingGemma-2-CPU-Max*) ;;
    *) die "the built binary does not identify as CPU-Max: $identity" ;;
esac

mkdir -p "$install_dir"
staged=$(mktemp "$install_dir/.$name.XXXXXX") || die "could not create a temporary file in $install_dir"
cp "$binary" "$staged"
chmod 0755 "$staged"
mv -f "$staged" "$install_dir/$name"
staged=

printf 'Installed %s\n  as %s/%s\n' "$identity" "$install_dir" "$name"

if [ "$with_models" = 1 ]; then
    sh "$source_dir/cpu-max/scripts/download_models.sh"
fi

case ":${PATH:-}:" in
    *":$install_dir:"*) ;;
    *) printf 'Add %s to PATH to run %s directly.\n' "$install_dir" "$name" ;;
esac
cat <<EOF

Next:
  $name --bind 127.0.0.1 --port 42666 --threads \$(nproc)   # downloads the models on first start if needed
  curl -s http://127.0.0.1:42666/v1/embeddings -H 'Content-Type: application/json' \\
      -d '{"model":"embeddinggemma-2","input":["hello"]}'
EOF
