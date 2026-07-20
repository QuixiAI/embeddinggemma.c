# Contributing

Contributions should preserve the project's narrow scope: standalone inference
and serving for `embeddinggemma-300M-qat-Q4_0.gguf` on CPU, Metal, CUDA, ROCm,
and Intel XPU SYCL.

## Before Starting

- Open an issue before changing the HTTP contract, model graph, quantization
  format, dependency policy, supported platform matrix, or release artifacts.
- Keep changes scoped. Do not combine kernel work with unrelated server or
  formatting changes.
- Do not commit GGUF files, model weights, generated benchmark runs, build
  output, or managed XPU dependencies.
- Read `EXTRACTION.md` for llama.cpp algorithm provenance and `PORT_SPEC.md` for
  the fixed graph and backend contract.
- For performance work, read `perf/perf.md` and the current entries in
  `perf/optimization_status.md` before changing a kernel.

The project is MIT licensed. Model acquisition and use are separate from this
repository and remain subject to the model provider's terms.

## Development Setup

The default model path is:

```text
${XDG_CACHE_HOME:-$HOME/.cache}/embeddinggemma.c/embeddinggemma-300M-qat-Q4_0.gguf
```

The server can populate that cache with `curl` or `wget`. Tests use `MODEL`, so
an existing file can be selected explicitly:

```sh
make test MODEL=/path/to/embeddinggemma-300M-qat-Q4_0.gguf
```

Common requirements are a C11 compiler, `make`, POSIX threads, Python 3, and a
local model. Backend toolchains are platform-specific:

| backend | required toolchain/runtime |
|---|---|
| CPU | Clang or GCC |
| Metal | full Xcode with the Metal Toolchain component |
| CUDA | CUDA toolkit, cuBLAS, and a supported NVIDIA driver |
| ROCm | ROCm HIP, hipBLAS, and an AMDGPU runtime |
| XPU SYCL | Intel oneAPI DPC++, oneMKL, and Level Zero |

Run `make help` for the current build, test, performance, and release targets.

## Code Map

- `src/engine.c` and `src/kernels.c`: portable CPU graph and kernels.
- `src/engine_metal.m` and `src/metal/`: Metal host code and kernel families.
- `src/engine_cuda.cu`: CUDA engine and kernels.
- `src/engine_rocm.hip`: ROCm engine and kernels.
- `src/engine_xpu*.cpp`: portable SYCL and specialized Xe2 routes.
- `src/inference_service.c`: batching, tokenization, singleflight, and result
  caching.
- `src/server.c`: HTTP transport, request parsing, and response caching.
- `perf/`: native harnesses, Python runners, retained results, and experiment
  decisions.

Prefer the existing backend boundary and model-specialized shapes over generic
abstractions. Shared behavior belongs in the portable engine only when all
backends can preserve the same contract.

## Build And Test

CPU changes must pass:

```sh
make clean-cpu
make test
```

Backend changes must pass the complete suite on matching hardware:

```sh
make test-metal
make test-cuda NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCHS=86
make test-rocm HIPCC=/opt/rocm/bin/hipcc ROCM_ARCHS=gfx942
source /opt/intel/oneapi/setvars.sh
make test-xpu XPU_XE2_FLASH=1 SYCL_CXX=icpx
```

Architecture overrides above are for fast local validation. Release builds
must use the portable architecture sets in `RELEASE.md`.

Every inference change must retain:

- Exact tokenizer output against checked-in vectors.
- Cosine similarity of at least `0.999` against llama.cpp embedding goldens.
- Backend drift and packed-batch thresholds enforced by the backend suite.
- Correct HTTP output at 768, 512, 256, and 128 Matryoshka dimensions.
- Bounded queue, workspace, and client-batch behavior.

Run the shell checks after changing scripts or release tooling:

```sh
make check
```

## Performance Changes

Treat optimization as an experiment, not an intuition-only change:

1. Record the machine, compiler, driver, commit, shapes, warmups, and iteration
   count.
2. Establish a same-process or alternating-order baseline.
3. Change one bottleneck hypothesis at a time.
4. Run correctness before trusting timing data.
5. Retain low-risk changes only when the representative gain is stable and at
   least 3%; document useful neutral and rejected experiments too.
6. Measure single requests, packed batches, and service concurrency. Include
   token lengths 1, 32, 512, and 2048 where the route applies.
7. Append each pass and its decision to `perf/optimization_status.md`.

Raw machine-specific runs belong under ignored `perf/results/`. Summaries that
justify retained behavior belong in the tracked optimization log.

## Style

- Keep portable code at C11. Backend translation units may use Objective-C,
  CUDA C++, HIP C++, or SYCL C++ as required.
- Preserve `-Wall -Wextra -Werror` cleanliness for host C code.
- Use ASCII unless an existing file requires otherwise.
- Add comments only for non-obvious invariants, synchronization, numerical
  choices, or architecture constraints.
- Avoid new runtime dependencies unless they are available on every supported
  deployment or isolated to an optional backend.

## Pull Requests

Describe the behavioral change, hardware and toolchain used, exact test
commands, correctness result, and before/after performance for kernel work.
Call out untested backends explicitly. Do not update `VERSION`, create a tag,
or change pinned install examples unless the release owner has approved the
exact next version.

Release procedure and artifact acceptance are defined in `RELEASE.md`.
