# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `reng-synapse`: HCCL bindings (`ffi.rs`: the `hccl.h` collectives, the
  1032-byte `hcclUniqueId` passed by value, `synEventElapsedTime`) and a
  `hccl` module with one card per process (`Card`), a communicator
  (`Comm`, unique id carried by file) and the `reng-hccl-test` probe: the
  coordinator spawns one worker per module id, verifies a summed
  all-reduce, times 16 KB / 1 MB / 16 MB all-reduces and checks that a
  recipe launch, a collective and another launch on one stream stay
  ordered. `Runtime::new_on` compiles a recipe onto a caller-owned device
  and stream; `Gb::scratch_alias_typed` aliases an f32 scratch tensor.
  On the Gaudi2 box the first collective fails inside HCL
  (`credit_manager.cpp: No available intermediate buffer`) because the
  stack's completion-queue counters never advance; see the multi-card
  report.

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
- Gemma-3 text (`gemma3_text`) and Gemma-2 (`gemma2`), from a sub-agent
  patch: `(1 + w)` norm gains folded at load, post-attention and
  post-MLP norms on top of the input norms (four per layer), the
  `sqrt(hidden)` embedding scale on the host in f32,
  `query_pre_attn_scalar` folded into the q-norm gain, per-layer RoPE
  tables (local theta on sliding layers, global on full layers) and
  per-layer sliding masks on the prefill, cached and batched paths (the
  model-wide window of the previous entry became a per-layer field),
  `tie_word_embeddings` defaulting to true for Gemma, GELU-tanh composed
  as `x * sigmoid(c1 x + c3 x^3)` (`gelu_fwd_bf16` only offers the erf
  form; `reng-gelu-test`), and the Gemma-2 attention and final logit
  softcaps. Verified Gemma-3-270m (8/8; 798-token prompt 789/798 with
  cosine 0.9998, 643/798 without the window), Gemma-3-1B (7/8 plus a
  near-tie; 795/798, cosine 1.0000) and Gemma-2-2B (7/8 plus a
  near-tie); b1 939 / 474 / 282 tok/s.
- Mistral-7B verified without engine changes: v0.3 (vocab 32768,
  rope_theta 1e6, no window) 7/8 exact plus a reference near-tie, 296/304
  argmax agreement with last-logits cosine 1.0000 at 304 tokens, b1 136
  tok/s (79.0%), b8 1079 (79.1%); v0.1 (`sliding_window 4096`) 8/8 exact,
  and at a 4500-token prompt with a password stated in the first tokens
  and asked for at the end, the windowed prefill reproduces the
  reference's forgetting (top-1 match, cosine 0.9991) while
  `RENG_NO_WINDOW=1` does not (top-1 differs, cosine 0.9656).
- Qwen2.5-32B verified without engine changes (64 layers, hidden 5120,
  40 heads over 8 KV heads, head_dim 128, intermediate 27648, vocab
  152064, untied head, rope_theta 1e6, `use_sliding_window: false`):
  7/8 exact plus a reference near-tie over the 5-token prompt, 8/8
  exact over a 315-token prompt, and at 315 tokens 301/315 argmax
  agreement with last-logits cosine 0.9984; b1 33 tok/s (87.1% of the
  HBM ceiling), b8 262 (86.1%), 30 ms per decode step. The 65.5 GB
  checkpoint puts 64.7 GB on the card (the embedding table stays on the
  host), memory-mapped in 2-3 s, first token 15-20 s after start with a
  cached recipe. `RENG_RECIPE_TRACE` now reports the device bytes each
  runtime allocates (inputs, scratch, output, workspace).
- `reng-sdpa-test` and `reng-msoftmax-test` probe the fused attention
  (`sdpa_recomp_fwd_bf16`) and masked-softmax kernels against host
  references; `tools/profile/` holds the trace timeline and decode-step
  gap scripts.
- Fused attention: one `sdpa_recomp_fwd_bf16` node per layer replaces
  the qk `batch_gemm`, mask add, softmax and av `batch_gemm`. The kernel
  takes the engine's own tensors: it broadcasts the size-1 K/V heads dim
  of `[hd, keys, 1, groups]` over the query heads of a group, reads the
  additive `[keys, queries, 1, 1]` mask (and the batched recipe's 5-D
  tensors with one mask row per sequence) as they are, so no tiling or
  reshape is needed; the scale stays folded into q. Softcapped layers
  (Gemma-2) keep the four nodes. It is the default in the single-sequence
  decode recipe, where a step is never slower and Llama-3.2-3B's is 2%
  faster; prefill blocks and batched decode keep the chain (a wash on
  SmolLM2-1.7B, Llama-3.2-3B and Qwen2.5-7B, 5% and 2.5% slower on
  Qwen2.5-1.5B). `RENG_SDPA=1` fuses every recipe, `RENG_SDPA=0` none
  (read when a graph is built). Every verified model agrees with its
  reference as before (two near-ties become exact matches);
  `reng-sdpa-shapes` probes the kernel's mask, batch and rank contracts
  at the target models' shapes.
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
- Zero-copy loading: the safetensors files are memory-mapped and every
  bf16 matrix is a `Bf16Slice` view of its file (a sub-view for the
  Phi-3 row-block splits, a shared view for a tied LM head); only
  converted (f32/f16 checkpoints), scaled (Granite) or unaligned tensors
  are copied. A background thread prefaults each map. The runtime
  uploads through a ring of at most four 256 MiB pinned buffers, each
  reuse fenced, instead of one pinned buffer per tensor kept for the
  model's lifetime (per-step inputs get theirs on first re-upload), with
  the copies into the ring split over threads. `reng-bench` and
  `reng-generate` print the load time and the time to the first token;
  `RENG_RECIPE_TRACE` shows the device acquire time separately.
  DeepSeek-R1-Distill-Llama-8B (16 GB): weights loaded in 13.6 s before
  and 0.8 s after, first token 24-26 s after start before and 7.2 s
  after, peak resident set 32-33 GB before and 18.8 GB after (of which
  15.4 GB are the mapped file pages, reclaimable by the kernel); results
  bit-identical.
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
- Device-resident decode loop for single-sequence generation
  (`RENG_DEVICE_LOOP`, on by default; `0` keeps the per-step path). The
  one-row recipe's only per-launch inputs are an int32 token id and an
  int32 position: the embedding row is a `gather_fwd_bf16` over the bf16
  table (the LM head's device copy when tied; Gemma's `sqrt(hidden)` is
  applied in f32 on the device and rounds like the host), the RoPE rows
  are gathers over the full tables, each mask row is a gather over a
  static pattern at an index the position shifts (the windowed masks and
  Gemma's per-layer masks included), and the ScatterND triples are two
  int32 nodes. The id, position and `IDS` tensors are rebound per launch
  into a position table and an id ring (one cache line per position), so
  `Generator::generate(seed, n)` enqueues `n` launches back to back and
  reads `n` ids once; `feed_id` for one token is the same with `n = 1`,
  and the seed is uploaded only when the last launch did not leave that
  id in place. `reng-gather-test` pins the kernels down; `reng-cache-test`
  feeds its one-row tail as ids through the loop and checks a run of loop
  steps against the CPU reference and against one launch at a time.
  Teacher-forced verdicts are unchanged on SmolLM2-135M, Qwen3-0.6B,
  Phi-3-mini (300-token prompt), Gemma-3-270m, OLMo-2-1B and Llama-3.2-3B,
  and free-running ids over 32 tokens are identical with and without the
  switch on Qwen2.5-1.5B and Phi-3-mini. Decode at batch 1 (64 new
  tokens, 128-token prompt, then 1024-token prompt): SmolLM2-135M 754 to
  855 tok/s (772 to 847), Qwen2.5-1.5B 377 to 418 (383 to 414),
  Llama-3.2-3B 259 to 273 (258 to 272), Qwen2.5-7B 133 to 139 (134 to
  139); the decode step's device window is unchanged (2.37 ms on
  Qwen2.5-1.5B either way) and the saving is the 0.15 to 0.2 ms per step
  of launch and readback round trip.
- Device-resident decode loop for the batched path (`BatchedModel`,
  `BatchedGenerator::generate`, `reng-bench --batch`; the same
  `RENG_DEVICE_LOOP` switch, on by default when the caller gives an
  embedding table). The batched decode recipe's only per-launch inputs
  are `B` int32 token ids and `B` int32 positions, one per slot: the `B`
  embedding rows are one gather, the RoPE rows are gathers at the `B`
  positions, each slot's mask row is a gather over the static pattern at
  `[keys, B]` indices that a `sub_fwd_i32` shifts by that slot's own
  position (the windowed and per-layer masks kept), and the ScatterND
  quadruples `(b, g, 0, position_b)` are two int32 nodes. The ids and
  positions are rebound per launch into an id ring and a position table
  of one row of `B` int32s per launch, so `run_ids(ids, n)` uploads the
  run's `n` position rows (and the seeds, unless the previous run left
  them in place), enqueues `n` launches back to back and reads the
  `n * B` ids once; a bucket growth recompiles the loop recipe like the
  per-step one. The loop runs a fixed `n` for every slot; a sequence that
  finishes early keeps advancing on its own output until the caller
  resets its slot. `reng-batch-test` feeds its steps as ids through the
  loop and checks a multi-step run against the CPU reference and against
  one launch at a time (bucket growth inside a run included);
  `reng-gather-test` pins down the batched kernel forms; `reng-bench`
  prints a hash of the decode ids. BENCH_NUMBERS_TBD

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

