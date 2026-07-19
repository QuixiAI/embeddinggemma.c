# EXTRACTION — gemma-embedding inference algorithms from llama.cpp

Deep-read of `/Users/eric/llama.cpp` at commit `d77599234`. Every claim carries a
file:line ref into that checkout. This is the algorithm reference for the
clean-room port; `PORT_SPEC.md` holds the model/product spec.

## 0. Hyperparameters — provenance

Loaded in `src/llama-model.cpp:1657-1685` (`case LLM_ARCH_GEMMA_EMBEDDING`):
- `hparams.swa_type = LLAMA_SWA_TYPE_SYMMETRIC` (1659)
- `swa_period = 6` → `set_swa_pattern(6)` (1660-1662)
- `causal_attn = false` (1664) — bidirectional
- `n_swa` from `attention.sliding_window` (1667) → 512
- `f_norm_rms_eps` (1668) → 1e-6
- `rope_freq_base_train_swa` from `rope.freq_base_swa`, **default 10000.0f**
  (`src/llama-hparams.h:116`; optional read at `llama-model.cpp:1666`)
- **`f_attention_scale = 1.0f / sqrt(head_dim) = 1/sqrt(256) = 0.0625`** (1683)
  — NOT the Gemma2/3 `query_pre_attn_scalar` variant (those branches are
  1572/1605 for GEMMA2/3 only).
- Head dims: `key_length`/`value_length` KVs override defaults
  (`llama-model.cpp:832-833`); `n_rot` = 256 via `rope.dimension_count`
  (839-841). Pooling MEAN set at conversion (`convert_hf_to_gguf.py:7260`).

## 1. Tokenizer — SentencePiece (`tokenizer.ggml.model = "llama"` → SPM)

### Defaults + GGUF overrides
SPM defaults (`src/llama-vocab.cpp:2166-2171`): `add_space_prefix=true`,
`clean_spaces=false`, `add_bos=true`, `add_eos=false`. Overridden by KVs:
`add_space_prefix` ← **false** for this model (2193;
`convert_hf_to_gguf.py:7136`), `bos=2`, `eos=1` (2287-2321), `add_bos` ←
**true**, `add_eos` ← **true** (2328-2333). Scores/types read per token
(2204-2252); types: UNDEFINED=0, NORMAL=1, UNKNOWN=2, CONTROL=3,
USER_DEFINED=4, UNUSED=5, BYTE=6.

### Top-level flow (`src/llama-vocab.cpp:3066-3131`)
1. Seed fragment list with the raw text; run `tokenizer_st_partition`
   (special-token split, below).
2. `is_prev_special = true`.
3. `add_special && add_bos` → push bos (3090-3094).
4. Per RAW fragment: if `add_space_prefix && is_prev_special` prepend one `' '`
   (3101-3103 — **never happens here**, add_space_prefix=false); apply
   `llama_escape_whitespace`; run the SPM core; `is_prev_special=false`.
   Per TOKEN fragment: push id, `is_prev_special=true` (3114-3117).
5. `add_special && add_eos` → push eos (3127-3130).

Empty string with add_special → `[bos, eos]`.

### Whitespace normalization (3038-3040)
`replace_all(text, " ", "\xe2\x96\x81")` — every 0x20 → U+2581 `▁` (3-byte
UTF-8). No other normalization (no NFKC) on the SPM path.

### Core SPM — bigram merge via max-priority-queue (96-238)
Structures: `llm_symbol {int prev, next; const char *text; size_t n;}` —
doubly-linked list over the byte buffer (80-86). `llm_bigram_spm {left, right,
score, size}` (96-108). Priority queue comparator: higher score first; ties →
**smaller left index first** (97-101). `rev_merge: map<string, pair<int,int>>`.

`tokenize` (117-173):
1. Split into UTF-8 chars: `len = unicode_len_utf8(lead byte)` (1..4),
   `sym.n = min(len, remaining)`; link prev/next (118-131).
2. Seed: `try_add_bigram(i-1, i)` for every adjacent pair (134-136).
3. Loop (139-167): pop best bigram; skip stale entries
   (`left.n==0 || right.n==0 || left.n+right.n != bigram.size`); merge right
   into left (`left.n += right.n; right.n = 0`), unlink right; propose
   `(left.prev, left)` and `(left, left.next)`.
4. Emit: walk chain, `resegment` each symbol (169-172).

`try_add_bigram` (202-229): `text = concat(left,right)`;
`token = text_to_token(text)` (plain hash lookup, 3625-3632); if found push
`{left, right, score=token.score, size=len(text)}` and set `rev_merge[text]`.

`resegment` (176-200): if the symbol's text is a vocab token → push id; else if
in `rev_merge` → recurse into the two halves; else **byte-fallback**: per byte
push `byte_to_token(b)`.

### Byte fallback (3595-3609)
`byte_to_token(ch)`: format `"<0xNN>"` with **uppercase** hex; fall back to the
raw 1-byte string if that entry is missing. Newline = `<0x0A>` (2259-2261).

### Special-token partition (`tokenizer_st_partition`, 2903-3019)
For each cached special token (sorted): if `!parse_special` and attr is
CONTROL or UNKNOWN → skip (2909-2915). Otherwise exact-substring scan of each
RAW fragment (2935), splitting into left RAW | TOKEN | right RAW. LSTRIP/RSTRIP
attrs trim adjacent whitespace (2952-2956, 2977-2982). Runs before the SPM
core, so specials never fragment. Goldens were generated with
`--no-parse-special` (CONTROL/UNKNOWN not matched; USER_DEFINED still is).

## 2. SWA pattern + mask

### Pattern (`src/llama-hparams.cpp:8-18`)
`swa_layers[il] = (il % 6 < 5)` — SWA unless `il % 6 == 5`. For 24 layers:
**full attention only at il ∈ {5, 11, 17, 23}**; all other 20 layers are
sliding-window.

### Masks (`src/llama-graph.cpp:2067-2087, 381-445`)
Two F32 masks `[n_tokens, n_tokens]`: full and SWA; `build_attn` picks SWA
when `is_swa(il)` (2110-2112). Fill: init all to `-INFINITY` (422, 437), set
`0.0f` where attended. Non-causal ⇒ no future masking; full mask attends
everywhere within the sequence (424).

### Symmetric window (`src/llama-hparams.h:338-347`)
```
half = n_swa / 2 = 256;
masked ⇔ (p1 - p0 < -half) || (p1 - p0 > half)
```
**Attend iff |Δpos| ≤ 256, inclusive** (513-position span). Mask added
pre-softmax.

## 3. RoPE

- Type: **NEOX** (`src/llama-model.cpp:9292` → `:9322`).
- Applied to Q and K **after** their RMS norms, **before** the
  `f_attention_scale` multiply on Q. `n_rot = 256` (full head).
- Per-layer base (`llama-model.cpp:8401-8407`): SWA layer → 10000 (train_swa);
  full layer → 1e6. freq_scale = 1 everywhere.
- YaRN off: `ext_factor` resolves to **0.0** (`llama-context.cpp:96-97`),
  `attn_factor=1`, so (`ggml/src/ggml-cpu/ops.cpp:5610-5641`):
  `theta_k = pos * freq_base^(-2k/256)` for k = 0..127; `cos/sin(theta_k)`.
- NEOX pairing (`ops.cpp:5715-5731, 5857`): dim `k` pairs with `k+128`
  (first-half/second-half, NOT interleaved):
```
x0 = src[k]; x1 = src[k+128];
dst[k]     = x0*cos - x1*sin;
dst[k+128] = x0*sin + x1*cos;
```
  No pass-through channels (n_rot == head_dim).

## 4. Attention

- `build_qkv` (`llama-graph.cpp:1064-1138`): separate wq/wk/wv, **no biases**,
  no clamp. Q → `[256, 3, T]`, K/V → `[256, 1, T]`. ggml `mul_mat(W, x)` =
  `W^T·x` (weights stored transposed).
- Order per layer: q = rms(q, attn_q_norm); q = rope(q); k = rms(k,
  attn_k_norm); k = rope(k); **q *= 0.0625**; V untouched.
- `wo_s` (attn_output.scale) is optional and **absent** in this GGUF → no-op.
- `build_attn_mha` (1932-2065): `kq = softmax(QK^T * 1.0 + mask)`;
  `kqv = V·kq`; heads concatenated → `[768, T]`; then output proj `wo`.
  GQA: the single KV head broadcasts across all 3 Q heads.
  Softmax semantics (`ops.cpp:5299-5336`): `wp = kq*scale + 1.0*mask`, then
  row softmax; masked = exp(−INF) = 0.

## 5. Norms + FFN

- RMS norm (`llama-graph.cpp:1028-1061`; `ops.cpp:3716-3765`):
  `y = x / sqrt(mean(x²) + eps) * w`, eps **inside** the sqrt, eps = 1e-6.
  No bias.
- **The Gemma "+1.0" quirk is baked into the GGUF at conversion**
  (`convert_hf_to_gguf.py:7182-7191`): every `*norm.weight` already includes
  +1.0 — apply as a plain multiply. (Only matters if re-deriving from raw HF
  safetensors.)
- FFN (`llama-graph.cpp:1141-1303`): `up = Wup·x; gate = Wgate·x;
  h = gelu_tanh(gate) ⊙ up; y = Wdown·h`. GELU is the **tanh approximation**
  (`ggml/src/ggml-cpu/vec.h:981-987`):
```
GELU_COEF_A    = 0.044715
SQRT_2_OVER_PI = 0.79788456080286535587989211986876
gelu(x) = 0.5*x*(1 + tanh(SQRT_2_OVER_PI * x * (1 + GELU_COEF_A*x*x)))
```
  Applied to the **gate** tensor (`ggml_geglu_split(a=gate, b=up)`,
  `ops.cpp:2990-3036`, `vec.h:1448-1452`). No FFN biases.
- Residual sandwich per layer (`src/models/gemma-embedding.cpp`):
```
attn_in   = rms(x, attn_norm)
attn_out  = rms(attn(attn_in), post_attention_norm)
sa_out    = attn_out + x
ffn_in    = rms(sa_out, ffn_norm)
ffn_out   = rms(ffn(ffn_in), post_ffw_norm)
x         = ffn_out + sa_out
```
- Final: `x = rms(x, output_norm)`.

## 6. Embedding scale + pooling + dense

- Input: `x = token_embd[ids] * sqrt(768)` (≈ 27.712813); raw (non-token)
  embeddings would get scale 1.0.
- Mean pooling (`llama-graph.cpp:2711-2809, 207-251`): unweighted arithmetic
  mean over ALL tokens of the sequence **including bos and eos**.
- Dense head (`build_dense_out`, 2686-2708; loaded `llama-model.cpp:4473-4475`
  as `dense_2.weight` `{768, 3072}` / `dense_3.weight` `{3072, 768}`,
  TENSOR_NOT_REQUIRED): applied after pooling, `dense_2` then `dense_3`, no
  biases. **Absent from the ggml-org GGUF → skip entirely.**

## 7. Q4_0 / Q8_0

Structs (`ggml/src/ggml-common.h:184-188, 241-245`):
```
block_q4_0 { fp16 d; uint8_t qs[16]; }  // 18 B / 32 elements
block_q8_0 { fp16 d; int8_t  qs[32]; }  // 34 B / 32 elements
```
- q4_0 dequant (`ggml-quants.c:397-415`): element j = `(qs[j] & 0xF) - 8`,
  element j+16 = `(qs[j] >> 4) - 8`, × d. **Low nibbles = elements 0..15,
  high = 16..31, zero-point −8.**
- q8_0 dequant (491-505): `y[j] = qs[j] * d`.
- Activation quantization (`quantize_row_q8_0_ref`, 234-257), per 32 block:
  `amax = max|x|; d = amax/127; qs[j] = roundf(x[j]/d)` (0 if amax==0);
  `block.d = fp32_to_fp16(d)`.
- q4_0×q8_0 dot (`ggml/src/ggml-cpu/quants.c:174-208`), per block:
```
sumi = Σ_{j<16} ((x.qs[j]&0xF)-8)*y.qs[j] + ((x.qs[j]>>4)-8)*y.qs[j+16]
sumf += sumi * fp16_to_fp32(x.d) * fp16_to_fp32(y.d)
```
  mul_mat pipeline: F32 activation row → q8_0 (`vec_dot_type`), then integer
  dot per weight row (`ggml-cpu.c:226-234`).

## 8. GGUF v3 (`ggml/include/gguf.h`, `ggml/src/gguf.cpp:402-733`)

Magic "GGUF", version u32 (=3), n_tensors i64, n_kv i64. KV: key string
(u64 len + bytes), type u32; ARRAY: elem-type u32 + count u64 + payload;
strings = u64 len + bytes, bool = int8. Types: U8..F64 = 0..12 (STRING=8,
ARRAY=9). Alignment: `general.alignment` (default 32, power of 2). Tensor
info: name string, n_dims u32 (≤4), ne i64×n_dims, type i32, offset u64
(relative, must be aligned). Data section starts at
`GGML_PAD(end_of_header, alignment)`; each tensor padded to alignment;
`offset` must equal the running padded sum (validated 715-732).
`GGML_PAD(x,n) = (x + n - 1) & ~(n - 1)`.

## Implementation checklist (all constants)

- rms eps 1e-6; embd scale sqrt(768); q-scale 0.0625; softmax kq_scale 1.0.
- head_dim 256, n_head 3, n_kv 1, n_embd 768, n_ff 1152, n_layer 24,
  n_swa 512 (±256 inclusive), vocab 262144, n_ctx 2048.
- RoPE NEOX half-split, n_rot 256, base 10000 (SWA) / 1e6 (full),
  freq_scale 1, YaRN off.
- Full-attention layers: {5, 11, 17, 23}. All others SWA.
- bos 2 / eos 1 / unk 3 / pad 0; add_bos AND add_eos; add_space_prefix false;
  byte tokens `<0xNN>` uppercase; space → U+2581 before SPM.
- GELU tanh approx on the gate; gelu(gate)·up → down.
- Norm weights pre-shifted (+1 baked in) — plain multiply.
- Dense head absent — skip.
- Mean pool includes bos/eos; serve L2-normalized.
