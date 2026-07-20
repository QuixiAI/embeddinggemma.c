# embedding-inference — Port Specification

A minimal inference engine for exactly one model:
`ggml-org/embeddinggemma-300M-qat-q4_0-GGUF`. Hand-ported from the llama.cpp
reference with no ggml dependency. CPU ARM64/x64, Metal, CUDA, ROCm/HIP, and
XPU SYCL are implemented. The server exposes the
`/api/embed` HTTP contract that `db/03` `get_embedding()` already speaks.

**Parity target**: llama.cpp CPU inference of the file below is ground truth.
`testdata/goldens-llamacpp.json` holds raw and L2-normalized mean-pooled
embeddings for the 10 strings in `testdata/test-strings.json` (generated with
`llama-embedding --pooling mean -ngl 0`, llama.cpp commit `d77599234`).
Acceptance for every backend: cosine ≥ 0.999 against the L2 goldens.

## Model file

- URL: https://huggingface.co/ggml-org/embeddinggemma-300M-qat-q4_0-GGUF/resolve/main/embeddinggemma-300M-qat-Q4_0.gguf
- sha256 `50d28e22432a148f6f8a86eab3700f92add5d1f54baf7790675a2a4dadbccf26`,
  277,852,192 bytes. Local dev copy: `model/embeddinggemma-300M-qat-Q4_0.gguf`
  (gitignored). `hexis init` downloads + verifies it when absent.
- GGUF v3, alignment 32, 35 KVs, 314 tensors, data offset 6,530,176.
- Full KV + tensor inventory: `testdata/model-manifest.json`.

## Hyperparameters (from the GGUF KVs)

| param | value |
|---|---|
| arch | `gemma-embedding` |
| n_ctx (train) | 2048 |
| n_embd | 768 |
| n_layer | 24 |
| n_ff | 1152 |
| n_head / n_head_kv | 3 / 1 (GQA, KV head shared) |
| head_dim (key & value) | 256 |
| rms_eps | 1e-6 |
| rope freq_base (full-attn layers) | 1,000,000 |
| rope freq_base (SWA layers) | 10,000 (default; `rope.freq_base_swa` absent) |
| sliding window | 512, **symmetric** (bidirectional), pattern period 6 |
| causal | no (encoder; full bidirectional within mask) |
| pooling | mean (`pooling_type = 1`) |
| dense head | **absent** in this file (`dense_2`/`dense_3` optional in arch; not present) |
| vocab | SentencePiece (`tokenizer.ggml.model = llama`), 262,144 tokens |
| special ids | bos 2, eos 1, unk 3, pad 0; add_bos **and** add_eos; no space-prefix |

SWA pattern period 6 (llama.cpp `set_swa_pattern(6)` default — exact
layer→SWA/full mapping per the extraction report).

## Tensor inventory (quant → kernel demands)

- **Q4_0 ×168** — every matmul weight: per layer `attn_q [768,768]`,
  `attn_k [768,256]`, `attn_v [768,256]`, `attn_output [768,768]`,
  `ffn_up [768,1152]`, `ffn_gate [768,1152]`, `ffn_down [1152,768]`.
- **Q8_0 ×1** — `token_embd [768, 262144]` (row-gather + dequant only).
- **F32 ×145** — all norms: per layer `attn_norm`, `attn_q_norm [256]`,
  `attn_k_norm [256]`, `post_attention_norm`, `ffn_norm`, `post_ffw_norm`,
  plus `output_norm`.

So the kernel surface is exactly: q4_0×f32 matmul (with q8_0 activation
quantization on CPU, per llama.cpp reference), q8_0 row dequant, RMS-norm,
RoPE, softmax(+mask), GELU-gated FFN elementwise, mean-pool.

## Forward graph (from `src/models/gemma-embedding.cpp`)

```
x = token_embd[ids] * sqrt(768)
for il in 0..23:
    h  = rmsnorm(x, attn_norm[il])
    q,k,v = h @ Wq, h @ Wk, h @ Wv          # heads: q 3×256, kv 1×256
    q  = rmsnorm(q, attn_q_norm[il]); q = rope(q, pos, base(il))
    k  = rmsnorm(k, attn_k_norm[il]); k = rope(k, pos, base(il))
    q *= f_attention_scale                   # kq_scale stays 1.0
    a  = softmax(q k^T + mask(il)) v         # bidirectional; SWA mask on 5/6 layers
    a  = a @ Wo
    a  = rmsnorm(a, post_attention_norm[il])
    x  = x + a
    h  = rmsnorm(x, ffn_norm[il])
    h  = (gelu(h @ Wgate) ⊙ (h @ Wup)) @ Wdown
    h  = rmsnorm(h, post_ffw_norm[il])
    x  = x + h
x = rmsnorm(x, output_norm)
emb = mean over tokens; serve L2-normalized
```

Exact values/algorithms for: `f_attention_scale`, RMS-norm weight offset
(Gemma `1+w` quirk), GELU variant, RoPE rotation pairing (NEOX vs NORM),
symmetric-SWA mask boundary condition, SPM tokenizer algorithm, and q4_0/q8_0
block layouts + dot-product reference — see `EXTRACTION.md` (produced from the
llama.cpp deep-read) before implementing.

## Serving contract (db/03 `get_embedding()`)

- `GET /` -> `200` JSON service discovery with documentation, OpenAPI, health,
  and inference-route links.
- `GET /docs` -> `200` self-contained HTML API documentation.
- `GET /openapi.json` -> `200` OpenAPI 3.1 JSON.
- `GET /healthz` -> `200` JSON after the model and workspace are ready.
- `GET /api/tags` → `200` JSON (health probe).
- `POST /api/embed` accepts `{"model": "...", "input": ["...", ...],
  "dimensions": 256, "encoding_format": "float"}` and returns L2-normalized,
  order-preserving vectors.
  `dimensions` is optional, defaults to 768, applies request-wide, and accepts
  exactly 768, 512, 256, or 128. Unsupported values return HTTP 400.
- `encoding_format` is optional and accepts exactly `float` or `base64`.
  `float` is the default array-of-numbers contract. `base64` returns one string
  per embedding containing `dimensions * 4` little-endian IEEE-754 float32
  bytes. Unsupported values return HTTP 400.
- The engine and exact-result cache always produce one canonical 768-vector.
  Lower-dimensional responses take its leading Matryoshka prefix and L2
  normalize that prefix. Dimensions therefore do not fragment the cache.
- The DB sends texts already prefixed by `ensure_embedding_prefix`:
  `search_document: ` / `search_query: ` / `clustering: ` /
  `classification: `. The engine embeds bytes as given — no templating, ever.
- Default bind `0.0.0.0:42666` (containers reach the host); configurable
  `--port/--bind`; `EMBEDDING_SERVICE_URL` remains the generic DB-side override.
- Dynamic batching uses a bounded fit-aware queue scan while always dispatching
  the oldest request. `EI_BATCH_LOOKAHEAD=0` restores strict FIFO diagnostics.
- A lone request skips the collection delay unless overlapping work was
  recently observed. `EI_ADAPTIVE_BATCH_WAIT=0` restores fixed-delay behavior.
- HTTP/1.1 keep-alive is supported with bounded worker ownership, request count,
  and idle timeout. At least half the default worker pool remains available for
  newly accepted connections.
- Successful float-JSON responses may be cached by exact request-body bytes in
  a byte-bounded server LRU. Base64 and error responses are not stored. This is
  separate from canonical token-ID embedding caching and duplicate singleflight.

## Backend Status

| target | status | implementation/build |
|---|---|---|
| macOS ARM64 CPU | implemented, parity tested | C11 + NEON dot-product path |
| x86 CPU | implemented, parity tested | scalar, SSSE3, and AVX2 paths |
| macOS Metal | implemented, parity tested | Objective-C host + standalone metallib; short-shape residual/next-norm fusion; FP16 K/V at long sequence lengths with FP32 accumulation |
| NVIDIA CUDA | implemented, parity tested | Q4 x Q8 DP4A latency path; direct-FP16 combined-QKV epilogue; Flash, online, banded-SWA, and tensor-core attention; native packed-Q4 MMA diagnostics; CUDA graph replay |
| ROCm/HIP | implemented, parity tested | portable CDNA fat binary; paired packed-Q4 wave and MFMA projection kernels; exact singleton V-only attention; batched hipBLAS GQA; FP16/FP32 score routing; fused residual/RMS and pooling epilogues |
| XPU SYCL | implemented, parity tested | resident FP16 weights; oneMKL XMX GEMM; fused norm/RoPE, online attention, and pooling; optional Xe2 Flash Attention |

Core inference is C11 and links the OS C library, `libm`, and pthreads. The
server has no linked HTTP client dependency: a model cache miss invokes `curl`
or `wget` with POSIX `posix_spawnp`, and an existing model requires neither.
CPU SIMD lives in `quants.c` and `kernels.c`; the Metal host is
`engine_metal.m`; Metal source is split by family under `src/metal/kernels/` and
compiled into one metallib. The CUDA host and kernels live in `engine_cuda.cu`
and link CUDA Runtime plus cuBLAS. Production CUDA attention uses shared-tile
FP16 GQA for short sequences and tensor-core QK/PV for long sequences, with
rectangular symmetric-SWA bands, FP32 softmax, and FP32 accumulation. The XPU
SYCL backend lives in `engine_xpu.cpp`, with optional Xe2 kernels adapted from
`vllm-xpu-kernels` in `engine_xpu_flash.cpp`.
The ROCm host and kernels live together in `engine_rocm.hip` and link HIP
Runtime plus hipBLAS. Its default build embeds native code objects for
`gfx908`, `gfx90a`, `gfx942`, and `gfx950`; `ROCM_ARCHS` is a developer override,
not a release default. Native packed-Q4 MFMA serves the measured midrange and
batched hipBLAS serves large projection and attention shapes. Independent
one-token sequences elide Q/K/RoPE/softmax and metadata, remain on paired direct
Q4 through 72 flattened tokens, and use a fused final residual/pool epilogue.

## Phase order

0. Ground truth, extraction report, manifest, and goldens - **done**
1. CPU scalar engine and llama.cpp parity - **done**
2. HTTP server and model download - **done**
3. NEON, AVX2/SSSE3, and persistent CPU scheduling - **done**
4. Metal kernels, host dispatch, and parity - **done**
5. CUDA, ROCm/HIP, and XPU SYCL - **done**
6. External service integration, installer, and release packaging - **done**
