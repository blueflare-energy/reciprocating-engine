# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- N-card parallelism strategies and the ceiling each one admits
  (`reng_ceiling::strategy`, and a `reng-ceiling <model_dir> --cards N
  [--batch b]` binary that prints them). Two objectives, because they do
  not have the same winner: single-stream tokens per second at batch 1,
  and aggregate tokens per second over every card at a batch per replica.
  Five strategies for N in {1, 2, 4, 8}. Data parallel: N replicas, a
  stream unchanged, the aggregate N times the single-card ceiling, and
  the model has to fit one card. Tensor parallel: each card streams 1/N
  of the layer weights and 1/N of the KV cache per token at 2.45 TB/s
  with the LM head replicated and the embedding still a lookup, plus the
  collective floor of two all-reduces per layer at `hidden x batch x 4`
  bytes (the 70B at world 2 reproduces the 28.8 ms per token the README
  carries). Pipeline
  parallel: consecutive layers in N stages, a single stream no faster
  than one card because the token still reads every layer, one activation
  hand-off per boundary, and up to N times the aggregate with N
  micro-batches in flight; it is the split that runs a model larger than
  one card for the least communication. The hybrids `dp2 x tp4` and
  `dp4 x tp2` compose the two. Expert parallel, for a mixture of experts:
  the E routed experts over N cards, a card streaming the shared weights
  plus k/N of the expert bytes per token, projected for OLMoE-1B-7B and
  Mixtral-8x7B from their configs and marked projected because the engine
  has no MoE layer yet and no all-to-all is measured on this box. The
  chooser returns the best strategy per model, card count and objective
  with its reasons: does it fit one card, do the heads divide the world,
  and what the split saves in weight time against what the collective and
  the host launch floor cost. Where no split is admissible it says so and
  lists every rejection instead of naming a pick.
  Three terms, kept apart: the physical ceiling (bytes per card over
  2.45 TB/s), the measured collective floor, and the measured host launch
  floor. The practical ceiling is
  `max(physical + collective, launch)`, so it is never above the rate the
  host can enqueue. The all-reduce latency table is the 8 KB to 64 MB
  sweep measured on the box on 2026-09-06, interpolated in the log of the
  message size and overridable with `--collectives <file.json>`; the
  launch table is the cheapest per-step host enqueue measured per world
  (60 us plus 75.6 us a layer at world 8, which is Llama-3.2-1B's 1.27 ms
  step and its 787 tok/s enqueue floor), marked engine cost rather than
  physics and overridable with `--launch <file.json>`.
- `tools/sweep_tp.py`: the multi-card sweep. `reng-tp` over a roster at
  every admissible world (the Megatron gate is checked from config.json,
  so an inadmissible world is skipped with its reason recorded rather
  than run), and N data-parallel replicas of `reng-bench` started
  together, one per module, summed. Three repeats a cell and the median
  reported, which is how the README's table is built, since one world-8
  run in five stalls on a collective. It writes the same JSON shape as
  the single-card benches plus a world and a strategy per entry, and a
  Markdown table. It holds every card it is given, so the `bench`
  workflow only runs it from a manual dispatch with `multi_card` ticked.
  `tools/test_sweep_tp.py` covers the parts that need no card (the
  argument parser, the Megatron gate, the median over repeats and the
  entry shape) and `ci` runs it beside `py_compile` over `tools/`.
- README: an "Across cards" table beside the single-card one. Per model,
  the measured tensor-parallel batch-1 throughput at worlds 2, 4 and 8
  against that world's practical ceiling, the measured eight-replica
  data-parallel aggregate at batch 8 against eight times the single-card
  ceiling, and `reng-ceiling`'s best eight-card strategy for each
  objective. Rows the split does not admit carry the reason (the head
  counts, or the layer form `TpModel::new` rejects); the ceilings block
  beside it gives all three terms per world, so a reader can see which
  floor binds.
- RoPE scaling types `linear`, `yarn` and `longrope` next to `llama3`
  (`rope_spec`: transformers' inverse-frequency vector and attention
  factor, the factor multiplied into the sin/cos tables). longrope picks
  its short or long factor list from the length of the sequence the
  tables serve (the prompt length of a prefill,
  `LlamaConfig::rope_caches_for`; the cache capacity of a generator), the
  config's top-level `max_position_embeddings` and
  `original_max_position_embeddings` feed the derived factors, and
  Phi-3's legacy `su` / `yarn` type names mean longrope. The `llama3` and
  unscaled tables are unchanged bit for bit, which a test pins against a
  copy of the original recipe. Measured on Gaudi2 with these tables,
  Phi-3.5-mini-instruct against its f32 reference at 300, 2000, 4096
  (short factors) and 4500 (long factors) tokens: argmax agreement 94 to
  97.5 percent with last-logits cosine 1.0000, where the unscaled tables
  gave 82, 86, 55 and 50 percent.
- `tools/oracle/rope_reference.py` writes the inverse frequencies,
  attention factors and `cos` / `sin` rows transformers 5.16 computes for
  Phi-3.5-mini-instruct, Phi-4-mini-instruct, google/gemma-3-4b-pt and
  three yarn configurations into
  `crates/reng-model/testdata/rope_reference.json`; a unit test parses
  each checkpoint's own `config.json` and compares the engine's tables
  against that reference at positions on both sides of the pretraining
  length.
- Multimodal Gemma-3 checkpoints (`model_type: gemma3`, the 4B and up):
  `LlamaConfig::from_json` flattens the `text_config`, fills
  `Gemma3TextConfig`'s defaults for the keys the files leave out, and
  the loader reads the weights under `language_model.model.` (the vision
  tower is skipped). Gemma-3-4B: greedy 8/8 and prefill agreement 97 to
  98 percent at 300 to 4500 tokens, cosine 1.0000.
- Partial rotations (`partial_rotary_factor`: Phi-4-mini rotates 96 of
  its 128 head dims, pairing `i` with `i + 48` and passing the rest
  through) need no graph change. The loader permutes each head's q and k
  rows so that HF's rotary pairs sit on the kernel's `j, j + head_dim / 2`
  pairs and the tables give the pass-through pairs cos 1 / sin 0; `q . k`
  does not depend on the order of the head dims, and `v` and `o_proj` are
  untouched. A `partial_rotary_factor` that does not give a whole number
  of rotary pairs, and a longrope factor list of the wrong length, are
  refused at config load.
- Tensor-parallel decoding over the cards of one HCCL communicator
  (`reng_synapse::tp`, `reng_model::TpGenerator`, the `reng-tp` binary):
  a coordinator spawns one worker process per module id, each rank holds
  its Megatron shard of every layer plus the replicated norms, embedding
  and LM head, and a layer runs as two recipes with an in-place f32
  all-reduce after `o_proj` and after `down_proj` on the rank's stream,
  every launch and collective of a decode run enqueued back to back
  without host synchronisation (the embedding and head recipes bound per
  launch to an id ring, as the device decode loop). The recipes are
  compiled once per kind and shape and bound to each layer's buffers per
  launch (`Runtime::new_bound` over `Bindings`, `Store` for the layers'
  device buffers, `Runtime::rebind_at`; a recipe's read-back buffer now
  joins its `Bindings`, so a child recipe of `Runtime::new_with` can share
  the parent's, which `cached.rs` and `batched.rs` could not do before).
  Prefill runs the same two recipes at the block width; `--batch` decodes
  several sequences in lockstep in the 5-D batched form. A sequence is
  prefilled once, from position 0: the wide recipe's ScatterND is out of
  place, so its blocks alternate between the sequence's cache slot and a
  shared scratch buffer, and a second prefill onto a non-empty sequence is
  rejected unless its block count is even. `--prompt-file <json>` takes
  the prompt from the `"prompt"` array of a `generate.py` reference file.
  DeepSeek-R1-Distill-Llama-70B on two cards reproduces its f32 reference
  8/8 exact and decodes at 27 tok/s at batch 1 and 207 tok/s at batch 8;
  the 8B distill reproduces the single card's ids over a 1000-token
  prompt (four prefill blocks) as well as a five-token one; world 1
  reproduces `reng-generate`'s ids exactly.
- Multi-card hand-shake and interfaces: the coordinator sets
  `HCCL_SOCKET_IFNAME` for every worker rather than leaving the choice to
  the library, whose default is the first interface not named `lo` or
  `docker` (on this box the BMC's virtual NIC). `reng-tp --ifname <nic>`
  names it; otherwise an inherited setting stands, and failing that the
  interface carrying the default route is used
  (`reng_synapse::hccl::pick_ifname`, which skips interfaces that are
  down, the loopback and `docker`/`veth`/`br-`/`virbr`/`tun`/`tap`/`vnet`
  devices - a default route through one of those is skipped too, since
  the sideband has to reach a peer host - and, unless nothing else is
  left, MAC-named USB ones). `--ifname ""` asks for the library's own
  enumeration back, since an empty variable is not an unset one.
  The hand-shake directory (mode 0700: its `id.bin` carries rank 0's
  address) is removed when the run ends or is interrupted, and kept when a
  rank failed, since the workers' SynapseAI logs under it say why
  (`RENG_TP_KEEP_DIR` keeps it always). One left behind by an earlier run
  whose pid the kernel reused is detected and removed rather than joined,
  where before its stale `go` would skip this run's acquire barrier and
  its stale `id.bin` would name a dead HCCL coordinator. A worker's wait
  for `go` is bounded by `--timeout` (forwarded to the workers) instead of
  a fixed 180 s, and its wait for rank 0's unique id by the same value
  capped at 180 s; both end at once on `abort` or on a coordinator that
  went away, so a rank never holds a card waiting for one that is not
  there. A rank that gives up waiting for `go` now exits with
  `EXIT_ACQUIRE`, which is what makes the coordinator relaunch the group.
  At `--batch B` every sequence's ids are written to `rank<r>.ids` and
  compared across the ranks, not sequence 0's alone, and a rank whose own
  sequences disagree with each other fails the run.
- Multi-card lifecycle: a rank never outlives its coordinator holding a
  card. SIGINT and SIGTERM are caught: the coordinator asks the ranks to
  abort their communicators, reaps them, removes the hand-shake directory
  and leaves with 130, and a worker signalled with its coordinator (the
  terminal signals the whole process group) aborts its communicator
  instead of dying inside a collective (a second signal leaves at once,
  without that, which is what a card wedged inside a collective costs).
  Each worker runs a watchdog thread that polls the hand-shake
  directory's `abort` file and its own parent id, and on either aborts the
  communicator (`hcclCommAbort`) and leaves through `_exit`;
  `Group::wait_all` writes `abort` and waits out a 10 s grace before it
  kills anything, and `Group`'s new `Drop` does the same for a coordinator
  that returns or panics early. A worker's error path leaves the same way
  rather than through the destructor chain, which after an HCL failure can
  hang in libSynapse for minutes while still holding the card. `Comm` is
  finalized (`hcclCommFinalize`) before it is destroyed and is declared
  before the `Card`, so the communicator goes away while the device is
  still acquired; that removes the `hccl_device.cpp:45 ... device not
  initialized` line every world-2 process used to print at teardown.
- Strided weight uploads: `Gb::input_bf16_strided` and `Stride` describe
  a column window of a mapped row-major matrix that the staging ring
  gathers row by row into the pinned slot, so `LlamaWeights::shard` keeps
  the o and down column blocks as views (`LayerTensors::wo_pitch`,
  `wd_pitch`, `LayerWeights::wo_pitch`, `wd_pitch`) instead of gathering
  the 24 GB they come to per rank of the 70B. `RENG_SHARD_GATHER` restores
  the copies for comparison: 32 GB owned and 31 s to load against 14 GB
  and 23 s.
- A safetensors tensor whose data offset is not 2-aligned (94 of the 70B
  distill's 723, 19.2 GB) cannot be read in place as `[u16]`. It is now a
  view that copies when it is first read (`Bf16Slice::Unaligned`) rather
  than a copy of the whole tensor made while the checkpoint is loaded, and
  a shard narrows it first: `Bf16Slice::sub` keeps an unaligned view
  unaligned, and `Bf16Slice::column_block` gathers a column window
  straight out of the map, which `LlamaWeights::shard` uses for the o and
  down projections of such a tensor (`wo_pitch` / `wd_pitch` 0) instead of
  a strided view whose read would copy the whole row range. A rank now
  copies its own shard of an unaligned tensor once - `1 / world` of a
  split one, all of a replicated one (`lm_head`, 2.1 GB of the 70B's
  19.2 GB of odd bytes, is copied whole on every rank, which is why the
  figure below is 10.7 GB and not 9.6 GB): 10.7 GB owned per rank of the
  70B at world 2 against 13.7 GB, and 10.7 GB copied over the run against
  32.9 GB (19.2 at load, 13.7 at the shard).
  `LlamaWeights::footprint` counts an unaligned view without reading it.
- The attention scale is applied to `wq` while the rows are staged for
  the upload (`Gb::input_bf16_scaled`, an f32 product rounded to bf16
  exactly as the old host copy was) instead of by a scaled copy of every
  q matrix on the host: the 8B distill now loads with 0 GB of owned
  weights (was 1 GB) and the logits are bit-identical.
- Tensor-parallel shards of a model: `LlamaConfig::shard(rank, world)`
  and `LlamaWeights::shard(cfg, rank, world)` give one card's slice of a
  Megatron split (this rank's query and KV heads and MLP columns as views
  into the mapped checkpoint for the q/k/v and gate/up projections,
  strided views of the o and down projections' column blocks, sliced biases
  and full-width q/k gains, shared norms, embedding and LM head), which
  the `reng-tp` path above runs on.
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

