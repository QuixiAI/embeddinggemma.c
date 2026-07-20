#!/bin/sh
set -eu

deps_dir=${1:-.xpu-deps}
vllm_commit=bab46865358da4eda3b866c41dd71a80e878d843
cutlass_commit=cd763790ad2f74d7294435ecf77682bac0062c3a

fetch_checkout() {
    url=$1
    commit=$2
    destination=$3

    if [ -e "$destination" ] && [ ! -d "$destination/.git" ]; then
        echo "$destination exists but is not a Git checkout" >&2
        exit 1
    fi
    if [ ! -d "$destination/.git" ]; then
        git clone --filter=blob:none "$url" "$destination"
    fi
    if [ "$(git -C "$destination" remote get-url origin)" != "$url" ]; then
        echo "$destination has an unexpected origin URL" >&2
        exit 1
    fi
    if [ -n "$(git -C "$destination" status --porcelain)" ]; then
        echo "$destination has local changes; refusing to replace them" >&2
        exit 1
    fi
    if ! git -C "$destination" cat-file -e "$commit^{commit}" 2>/dev/null; then
        git -C "$destination" fetch --depth=1 origin "$commit"
    fi
    git -C "$destination" checkout --quiet --detach "$commit"
}

mkdir -p "$deps_dir"
fetch_checkout \
    https://github.com/Lazarus-AI-Research/vllm-xpu-kernels \
    "$vllm_commit" "$deps_dir/vllm-xpu-kernels"
fetch_checkout \
    https://github.com/intel/sycl-tla.git \
    "$cutlass_commit" "$deps_dir/sycl-tla"
