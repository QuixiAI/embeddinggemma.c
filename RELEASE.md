# Release Runbook

This runbook publishes one GitHub release containing raw executable files for
every supported platform. GitHub generates source archives automatically; do
not wrap binary assets in tarballs or zip files.

## Release Contract

- Obtain explicit approval for the exact version before editing `VERSION`,
  creating a tag, or publishing a release.
- Build every binary from the same clean release commit.
- Publish all six executables on every release, even when only one backend
  changed.
- Do not publish model weights, GGUF files, standalone metallibs, object files,
  debug bundles, or dependency source trees.
- Keep the stable asset names below. The installer depends on them.
- Strip each executable. Ad-hoc sign and verify both Darwin executables.
- Publish `SHA256SUMS` covering exactly the six executable assets.

Expected assets:

```text
embeddinggemma-darwin-arm64-cpu
embeddinggemma-darwin-arm64-metal
embeddinggemma-linux-x86_64-cpu
embeddinggemma-linux-x86_64-cuda
embeddinggemma-linux-x86_64-rocm
embeddinggemma-linux-x86_64-xpu
SHA256SUMS
```

## Prepare The Release Commit

1. Confirm the exact approved version, including the leading `v`.
2. Update `VERSION` and every version-pinned README example.
3. Update benchmark tables, compatibility notes, and the optimization log for
   user-visible backend changes.
4. Run CPU tests and script validation.
5. Commit and push the release commit before distributing it to build hosts.

```sh
version=$(cat VERSION)
git status --short
make check-scripts
make test
git grep -n "$version" README.md
git push origin main
```

Create a detached worktree from that exact commit. Use its contents for every
local and remote build:

```sh
commit=$(git rev-parse HEAD)
worktree="/tmp/embeddinggemma-$version-$commit"
git worktree add --detach "$worktree" "$commit"
export MODEL="$HOME/.cache/embeddinggemma.c/embeddinggemma-300M-qat-Q4_0.gguf"
```

Do not build a release from a development directory containing stale objects.
The `release-*` targets clean the selected backend before compiling it.

## Darwin ARM64 CPU And Metal

Requirements: Apple Silicon, macOS 14 or newer deployment target, full Xcode,
and the installed Metal Toolchain component.

```sh
cd "$worktree"
xcode-select -p
xcodebuild -version
xcrun --find metal
make test
make test-metal
make release-darwin DIST=/tmp/quixiembed-dist
```

`release-darwin` rebuilds CPU and Metal, strips and ad-hoc signs both files, and
checks that the Metal executable contains the embedded `__DATA,__metallib`
section. Validate the staged executables, not only the unstripped build output:

```sh
codesign --verify --verbose=2 /tmp/quixiembed-dist/embeddinggemma-darwin-arm64-cpu
codesign --verify --verbose=2 /tmp/quixiembed-dist/embeddinggemma-darwin-arm64-metal
python3 testdata/test_http_dimensions.py \
  --binary /tmp/quixiembed-dist/embeddinggemma-darwin-arm64-cpu \
  --model "$MODEL" --backend cpu
python3 testdata/test_http_dimensions.py \
  --binary /tmp/quixiembed-dist/embeddinggemma-darwin-arm64-metal \
  --model "$MODEL" --backend metal
```

The metallib is embedded in the Metal executable. It is an intermediate build
file, not a release asset.

## Linux X86_64 CPU

Use a broadly compatible Linux build host and avoid enabling host-specific ISA
flags such as `-march=native`.

```sh
cd "$worktree"
make test
make release-linux-cpu DIST=/tmp/quixiembed-dist
python3 testdata/test_http_dimensions.py \
  --binary /tmp/quixiembed-dist/embeddinggemma-linux-x86_64-cpu \
  --model "$MODEL" --backend cpu
ldd /tmp/quixiembed-dist/embeddinggemma-linux-x86_64-cpu
```

Review `ldd` output for unexpected non-system dependencies.

## Linux X86_64 CUDA

Build with the production CUDA toolkit. Do not set `CUDA_ARCH`, `CUDA_ARCHS`,
or `CUDA_PTX_ARCH` for a release. The default emits native cubins for every
architecture supported by the installed compiler and PTX for the newest one.

```sh
cd "$worktree"
make test-cuda NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCHS=86
make release-linux-cuda DIST=/tmp/quixiembed-dist \
  NVCC=/usr/local/cuda/bin/nvcc CUDA_HOME=/usr/local/cuda
python3 testdata/test_http_dimensions.py \
  --binary /tmp/quixiembed-dist/embeddinggemma-linux-x86_64-cuda \
  --model "$MODEL" --backend cuda
cuobjdump --list-elf /tmp/quixiembed-dist/embeddinggemma-linux-x86_64-cuda
cuobjdump --list-ptx /tmp/quixiembed-dist/embeddinggemma-linux-x86_64-cuda
ldd /tmp/quixiembed-dist/embeddinggemma-linux-x86_64-cuda
```

The focused test may use the host architecture for speed. The release command
must run without an architecture override and must show both cubins and PTX.

## Linux X86_64 ROCm

Build one portable CDNA executable. Do not set `ROCM_ARCH` or `ROCM_ARCHS` for
the release; the default includes `gfx908`, `gfx90a`, `gfx942`, and `gfx950`.

```sh
cd "$worktree"
make test-rocm HIPCC=/opt/rocm/bin/hipcc ROCM_ARCHS=gfx942
make release-linux-rocm DIST=/tmp/quixiembed-dist \
  HIPCC=/opt/rocm/bin/hipcc ROCM_HOME=/opt/rocm
python3 testdata/test_http_dimensions.py \
  --binary /tmp/quixiembed-dist/embeddinggemma-linux-x86_64-rocm \
  --model "$MODEL" --backend rocm
/opt/rocm/bin/roc-obj-ls \
  /tmp/quixiembed-dist/embeddinggemma-linux-x86_64-rocm
ldd /tmp/quixiembed-dist/embeddinggemma-linux-x86_64-rocm
```

The staged file must list all four CDNA code objects and no missing runtime
libraries.

## Linux X86_64 XPU SYCL

Use the pinned dependencies fetched by `make xpu-deps`. The production Xe2
build enables the specialized FlashAttention and packed-W4 routes and embeds
device images for PVC and BMG targets.

```sh
cd "$worktree"
source /opt/intel/oneapi/setvars.sh
make xpu-deps
make test-xpu XPU_XE2_FLASH=1 SYCL_CXX=icpx
make release-linux-xpu DIST=/tmp/quixiembed-dist SYCL_CXX=icpx
python3 testdata/test_http_dimensions.py \
  --binary /tmp/quixiembed-dist/embeddinggemma-linux-x86_64-xpu \
  --model "$MODEL" --backend xpu
ldd /tmp/quixiembed-dist/embeddinggemma-linux-x86_64-xpu
```

Run on a Level Zero GPU and confirm the specialized route is active in the
startup diagnostics. Review `ldd` for oneAPI, oneMKL, and Level Zero runtime
availability.

## Assemble And Verify

Copy the six staged executables into one empty directory on the release
workstation. Preserve the names exactly, then generate and verify checksums:

```sh
cd "$worktree"
make release-checksums DIST=/tmp/quixiembed-dist
make release-verify DIST=/tmp/quixiembed-dist
make release-ready DIST=/tmp/quixiembed-dist
```

`release-ready` rejects a dirty worktree, a malformed `VERSION`, a README that
does not mention the version, an incomplete asset matrix, wrong executable
formats, duplicate checksum entries, and checksum mismatches.

Create an annotated tag only after all staged artifacts pass:

```sh
version=$(cat VERSION)
git tag -a "$version" -m "embeddinggemma.c $version"
git push origin "$version"
```

Prepare release notes that summarize user-visible changes, tested hardware,
runtime requirements, correctness validation, and architecture coverage. Then
publish the raw files:

```sh
gh release create "$version" /tmp/quixiembed-dist/* \
  --title "embeddinggemma.c $version" \
  --notes-file /tmp/release-notes.md
```

Verify the published asset list before declaring the release complete:

```sh
gh release view "$version" --json tagName,name,url,assets \
  --jq '{tag:.tagName,name,url,assets:[.assets[].name]}'
```

## Installer Acceptance

Test a pinned install from GitHub after publication. Use a temporary directory
so the developer's active binary is not replaced:

```sh
./install.sh --version "$(cat VERSION)" --variant cpu \
  --install-dir /tmp/quixiembed-install
/tmp/quixiembed-install/quixiembed --help
```

On each accelerator platform, repeat with its explicit variant. Also test
automatic selection and Linux CPU fallback when accelerator runtime libraries
are unavailable.

If publication is incomplete or an asset is wrong, fix the release before
announcing it. Do not silently move an existing tag to a different commit.
