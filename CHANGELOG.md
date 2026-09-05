# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `reng-synapse`: a Llama-style decoder (RMSNorm, RoPE, grouped-query
  attention, SwiGLU) as one fused SynapseAI recipe; attention batched over
  heads in 4-D tensors; a KV cache updated inside the recipe (placement
  gemm plus add, ping-pong buffers) with a narrow decode recipe sharing the
  prefill recipe's weights and cache; a two-sentinel readback protocol that
  is exact on a stack whose stream sync returns before DMA copies land;
  `run_node` probes and contract tests for `batch_gemm`, broadcast add,
  4-D softmax, batched RoPE and `transpose`.
- `reng-model`: HuggingFace `config.json` + `model.safetensors` loading,
  prefill (`reng-prefill`), KV-cached greedy generation (`reng-generate`,
  teacher-forced against an HF reference with a bf16 near-tie tolerance),
  and `reng-bench` (prefill and decode tok/s next to the roofline ceiling).
- `tools/oracle/`: HF transformers CPU reference scripts.
- `bench` workflow: runs `reng-bench` on a self-hosted Gaudi2 runner after
  every merge to main and publishes the history to GitHub Pages, plus the
  decode-versus-batch and prefill-versus-context sweeps (`tools/sweep.py`).
- Batched decode (`BatchedModel`, `reng-bench --batch`): B sequences advance
  one token per launch over a B-slot KV cache; the cache is updated in place
  by a ScatterND node and the greedy token is an argmax on the device.
- Qwen2-style attention biases; sharded safetensors checkpoints; verified
  models: SmolLM2-135M/360M/1.7B, Qwen2.5-0.5B/1.5B/3B/7B,
  TinyLlama-1.1B, Falcon3-1B-Base, DeepSeek-R1-Distill-Qwen-1.5B,
  DeepSeek-R1-Distill-Llama-8B and Phi-3-mini-4k-instruct (its fused
  qkv_proj and gate_up_proj weights are split into row blocks by the
  loader).
- Sliding-window attention (Phi-3, Mistral, Qwen2 with
  `use_sliding_window`): the host-built masks admit only the last
  `sliding_window` positions. Phi-3-mini at a 2100-token prompt agrees
  with the reference (last-logits cosine 0.9999) where full attention
  did not (0.41).
- SmolLM3: NoPE layers (`no_rope_layers` in the config; those layers skip
  the rotary nodes); SmolLM3-3B verified.
- Qwen3: per-head q/k RMSNorm (`q_norm`/`k_norm` gains, applied after the
  projection and before RoPE; the attention scale folds into the q gain)
  and an explicit `head_dim` whose `num_attention_heads * head_dim` may
  differ from `hidden_size`; verified Qwen3-0.6B and Qwen3-1.7B.
- Granite 3.x dense (`model_type: granite`): the four config scalars
  need no new graph node. `embedding_multiplier` scales the host
  embedding gather, `attention_multiplier` replaces `1/sqrt(head_dim)`
  as the attention scale (folded into `wq` as before),
  `residual_multiplier` is folded into `o_proj` and `down_proj` and
  `1/logits_scaling` into the LM head at load (scaled bf16 copies; the
  tied embedding stays as stored). `scale_bf16` is now public in
  `reng-synapse`. The loader refuses `mlp_bias: true`. Verified
  granite-3.1-2b-instruct: 8/8 exact over a 345-token prompt, 7/8 plus
  one reference near-tie over a 5-token prompt.
- Verified on the roster: Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B
  (llama3 rope scaling), Qwen3-4B, Qwen3-8B, phi-4 (14.7B: 72 tok/s at
  batch 1, 84% of the HBM ceiling).
- OLMo-2 (`model_type: olmo2`): the two layer norms sit on the branch
  outputs (`post_attention_layernorm` and `post_feedforward_layernorm`
  normalise the attention and MLP outputs before the residual adds; no
  input norm) and the q/k norms span the whole projection (one RMSNorm
  over `n_heads * head_dim` before the head reshape, distinct from the
  Qwen3 per-head form, chosen from the gain length). Verified
  OLMo-2-0425-1B: 8/8 exact, prefill at 257 tokens 249/257 argmax
  agreement with last-logits cosine 1.0000; b1 512 tok/s, b8 4205.
  `reng-layer-test` uses the dense input generator of the other tests
  and `reng-norm-test` takes `[scale] [eps]`.
- Mistral-7B verified without engine changes: v0.3 (vocab 32768,
  rope_theta 1e6, no window) 7/8 exact plus a reference near-tie, 296/304
  argmax agreement with last-logits cosine 1.0000 at 304 tokens, b1 136
  tok/s (79.0%), b8 1079 (79.1%); v0.1 (`sliding_window 4096`) 8/8 exact,
  and at a 4500-token prompt with a password stated in the first tokens
  and asked for at the end, the windowed prefill reproduces the
  reference's forgetting (top-1 match, cosine 0.9991) while
  `RENG_NO_WINDOW=1` does not (top-1 differs, cosine 0.9656).
- `reng-sdpa-test` and `reng-msoftmax-test` probe the fused attention
  (`sdpa_recomp_fwd_bf16`) and masked-softmax kernels against host
  references; `tools/profile/` holds the trace timeline and decode-step
  gap scripts.
- `reng-argmax-test`: probes the argmax kernels with the row maximum
  planted at chosen positions and values, single and multi row.
- `RENG_HOST_ARGMAX=1` makes `reng-generate` read the logits and take
  the argmax on the host; `RENG_ARGMAX_CHECK=1` reads both the device id
  and the logits after every cached step and reports a disagreement.
- `RENG_MODULE_ID` picks the card by its SynapseAI module id
  (`synDeviceAcquireByModuleId`); without it the runtime takes any free
  card. `HABANA_VISIBLE_DEVICES` never steered the acquire, so the bench
  workflow now relies on the runner exporting `RENG_MODULE_ID`.
- `reng-mme-bench` takes the gemm shape (m k n iters transpose_b) and runs
  on the graph runtime: the prefill gemm `[1024 x 2048] x [2048 x 8192]`
  reaches 82% of the MME peak standalone, so the 44% seen inside the
  prefill recipe is the recipe's doing.
- Prefill writes its block into the KV cache through a ScatterND whose
  output is a second buffer, alternating buffers per block: the in-place
  form runs the block's rows serially (0.16 ms per layer and cache at 1024
  rows, a third of the layer). SmolLM2-1.7B prefill at 1024 tokens goes
  from 31.2k to 35.2k tok/s. Attention projections are plain gemms over
  the natural weights (shared with the batched decode recipe) plus a
  transpose into the head layout instead of per-head batch_gemms, which
  the MME runs at N = head_dim.
- Llama 3.1 style `rope_scaling` (`rope_type: llama3`): the low-frequency
  rotary dims are rescaled as in transformers. Over a 1000-token prompt
  the 8B distill's per-position argmax agreement with the reference goes
  from 953 to 993 of 1000 and the last-logits cosine from 0.9936 to
  0.9998.
- Weights stay bf16 in the checkpoint's own `[out, in]` layout from the
  safetensors file to the device: the loader no longer converts to f32
  or transposes, the graph borrows the slices (the gemms take them as
  transposed B operands), and the upload is a memcpy into the pinned
  staging buffer. Half the host memory and most of the launch time of a
  model go away.
- Compiled recipes are cached on disk (`$HOME/.cache/reng/recipes`, or
  `RENG_RECIPE_CACHE`; `0` disables) keyed by a digest of the graph
  structure, the SynapseAI version and the compiler's environment knobs,
  so a graph of a known shape loads instead of compiling.
- The `bench` workflow runs the cache and batch tests and teacher-forced
  generation against the runner's HF references before recording numbers.
- `reng-attn-bench` and `reng-scatter-bench` time attention gemm
  orientations and cache-write kernels standalone.
- Capacity buckets for batched decode: the recipes are compiled for the
  smallest bucket of cache positions holding the longest live sequence and
  regrown on demand (`RENG_MIN_CAP` sets the floor), since attention reads
  the whole cache every step. SmolLM2-1.7B batch 64 goes from 22% to 46% of
  the HBM ceiling with 160-token sequences.

- Workspace scaffold with five crates: `reng-core`, `reng-hal`, `reng-ceiling`,
  `reng-cli`, `reng-synapse`.
- `reng-core`: dtype, tensor shape, and device-identifier types.
- `reng-hal`: Gaudi2 device discovery through the kernel `accel` subsystem,
  with silicon-stepping detection and no vendor-userspace dependency.
- `reng-ceiling`: first-principles roofline calculator for prefill and decode
  ceilings on Gaudi2, MoE-aware, driven from a HuggingFace `config.json`.
- `reng-synapse`: Rust FFI to the SynapseAI graph API and a bf16 MME matmul
  (`reng-hello-mme`), verified against a CPU reference on real hardware.
- `reng devices`, `reng ceiling`, and `reng grid` CLI commands.
- Vendored (by reference) the habanalabs driver, with patch 0001 guarding the
  Gaudi2 MIN/MAX macros so the 1.19.0 driver builds on kernel 6.8+.
- CI pipeline (build, format, clippy, tests, coverage) and self-hosted status
  badges.

### Fixed

- RMSNorm uses the epsilon from the config. The `rms_norm_fwd_bf16` node
  was given the backward kernel's `ns_RmsNorm` parameter layout (epsilon
  first), which the forward kernel ignores in favour of a fixed 1e-5; it
  now gets `ns_LayerNormKernel::ParamsRmsNorm` (`epsValid`, `eps`, axis
  bitmaps, `normalizedShapeDims`, `fastMath`). Invisible for real
  activations (mean squares far above the epsilon) but exact now for
  every model that says 1e-6; `reng-norm-test 256 256 0.001 <eps>` shows
  the difference (rel_L2 0.54 before, 0.0027 after at eps 1e-6).
- The device argmax of the LM head casts the logits to f32 first:
  `argmax_fwd_bf16` is wrong for a single-row input (the decode shape)
  whenever the row's maximum is small or negative, returning 0 or an
  index past the vocabulary (Phi-3-mini returned 32384 of 32064 at
  200-token prompts, whose maximum logits are negative); `argmax_fwd_f32`
  is right in every probed case. Phi-3-mini now matches its reference
  over a 300-token prompt.

