# Reciprocating Engine

[![ci](https://github.com/blueflare-energy/reciprocating-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/blueflare-energy/reciprocating-engine/actions/workflows/ci.yml)
[![bench](https://github.com/blueflare-energy/reciprocating-engine/actions/workflows/bench.yml/badge.svg)](https://blueflare-energy.github.io/reciprocating-engine/dev/bench/)
![lines of code](.github/badges/loc.svg)
![crates](.github/badges/crates.svg)
![dependencies](.github/badges/deps.svg)
![tests](.github/badges/tests.svg)
![coverage](.github/badges/coverage.svg)
![code quality](.github/badges/quality.svg)

An open inference engine for Intel Gaudi2 (HL-225) accelerators, written in
Rust.

> Status: early. The hardware abstraction layer and tooling are taking shape
> first; the compute and serving paths are under active development. Interfaces
> will change without notice until a tagged release.

## Why

Gaudi2 has a comparatively thin open software surface. Most inference stacks
target it through large vendor frameworks. Reciprocating Engine is a small,
auditable Rust codebase that talks to the accelerator directly, so every layer
from device discovery to the compute graph is readable, measurable and tunable.

The project is benchmark-driven. Every model and batch shape is measured
against a hardware ceiling derived from first principles (memory bandwidth,
compute throughput, interconnect), so performance work is relative to that
ceiling rather than to an arbitrary baseline.

## What works today

- `reng-core`: shared data types (dtypes, tensor shapes, device identifiers).
- `reng-hal`: Gaudi2 device discovery through the kernel `accel` subsystem,
  including silicon-stepping detection, with no dependency on the vendor
  userspace.
- `reng`: a CLI that enumerates the accelerators on a host.
- `reng-synapse`: a Llama-style decoder (RMSNorm, RoPE, grouped-query
  attention with a causal mask, SwiGLU) compiled as one fused SynapseAI
  recipe and launched once, with a readback protocol that survives the
  driver's late-completing writes.
- `reng-model`: loads a Hugging Face `config.json` plus `model.safetensors`
  and runs prefill (`reng-prefill`) and greedy generation (`reng-generate`)
  on the device. Generation compiles the model once with a KV cache and
  launches the recipe once per token; keys and values stay in HBM and only
  the new token's logits cross the bus.

```console
$ reng devices
INDEX  PCI              STEPPING
0      0000:cc:00.0     A0
1      0000:cd:00.0     A0
...
```

## Benchmarks

Generation compiles two recipes over the same weights and cache: a wide one
for prompt blocks and a one-row one for decode steps. `--batch` decodes
several sequences in lockstep over a multi-slot cache. Measured on one card
(bf16, 128-token prompt, `tools/oracle` references match), percentages
against the `reng-ceiling` roofline:

| Model | Decode b1 tok/s | Decode b8 tok/s | Decode b64 tok/s | Prefill 1024 tok/s |
|---|---|---|---|---|
| SmolLM2-135M | 858 (9.6%) | 7339 (11.1%) | 39965 (12.3%) | 137.0k (10.8%) |
| Gemma-3-270m | 1048 (23.1%) | 8111 (23.1%) |  |  |
| SmolLM2-360M | 643 (19.2%) | 5745 (22.6%) | 29918 (21.0%) | 71.9k (14.1%) |
| Qwen2.5-0.5B | 802 (32.4%) | 6219 (31.8%) | 37118 (26.1%) | 103.9k (25.9%) |
| Qwen3-0.6B | 638 (31.5%) | 4698 (31.7%) |  |  |
| Llama-3.2-1B | 608 (61.4%) | 4709 (60.3%) |  |  |
| Gemma-3-1B | 503 (41.1%) | 3963 (41.0%) |  |  |
| OLMo-2-0425-1B | 528 (55.6%) | 4358 (60.2%) |  |  |
| Qwen2.5-1.5B | 428 (54.0%) | 3304 (52.6%) | 20195 (43.2%) | 60.2k (45.5%) |
| Qwen3-1.7B | 393 (55.5%) | 2987 (54.5%) |  |  |
| Falcon3-1B-Base | 544 (62.4%) | 4172 (61.5%) | 22492 (49.9%) |  |
| TinyLlama-1.1B | 586 (49.5%) | 4785 (51.1%) | 27058 (39.3%) | 50.0k (26.1%) |
| DeepSeek-R1-Distill-Qwen-1.5B | 428 (54.0%) | 3297 (52.5%) | 19792 (42.3%) |  |
| SmolLM2-1.7B | 399 (56.2%) | 3177 (59.1%) | 14016 (46.8%) | 34.2k (28.7%) |
| granite-3.1-2b-instruct | 281 (58.2%) | 2180 (57.4%) |  |  |
| Gemma-2-2B | 291 (62.2%) | 2273 (62.1%) |  |  |
| Qwen2.5-3B | 257 (64.8%) | 2004 (63.5%) | 12932 (53.7%) | 31.6k (47.4%) |
| Llama-3.2-3B | 273 (71.7%) | 2102 (70.3%) |  |  |
| SmolLM3-3B | 253 (63.7%) | 2074 (66.0%) |  |  |
| Phi-3-mini-4k-instruct | 210 (64.2%) | 1620 (65.3%) | 5426 (38.3%) |  |
| Qwen3-4B | 202 (66.4%) | 1555 (65.2%) |  |  |
| Qwen2.5-7B | 140 (80.6%) | 1083 (78.5%) | 7516 (70.3%) | 18.5k (62.5%) |
| Mistral-7B-v0.3 | 138 (80.6%) | 1093 (80.2%) |  |  |
| Llama-3.1-8B | 132 (81.2%) | 1041 (80.5%) |  |  |
| Qwen3-8B | 127 (78.5%) | 988 (77.2%) |  |  |
| DeepSeek-R1-Distill-Llama-8B | 132 (81.2%) | 1040 (80.4%) | 6894 (71.3%) |  |
| phi-4 | 73 (84.3%) | 583 (84.8%) |  |  |
| Qwen2.5-32B | 34 (88.1%) | 264 (86.6%) |  |  |

Decode percentages are against the HBM roofline, prefill against the MME
roofline. Every model is verified against a Hugging Face f32 reference
before it is listed (8 greedy tokens, exact up to bf16 near-ties).
SmolLM2-135M matches the f32 transformers reference on per-position argmax
(cosine 1.000 on the last logits at 128 tokens), and greedy decoding matches
token for token except at f32 near-ties, which bf16 cannot resolve.
`tools/oracle/` holds the reference scripts.

The roofline is a physical limit, not a target: at batch 1 every token
streams the whole model from HBM, so the ceiling is the weight bytes per
token over the 2.45 TB/s of the card, one formula for every row. For
Qwen2.5-32B that is 65 GB per token, 26.5 ms, 37.7 tok/s; the compute
for the same token takes under a millisecond. Real HBM sustains 85 to
92% of its spec on a streaming pattern, so 88% is the end of the road on
one card in bf16. Higher single-stream numbers come only from fewer
bytes per token (FP8 halves them, INT4 quarters them), more cards
(tensor parallelism adds a card's bandwidth per card, see
[Multi-card](#multi-card)), or more accepted tokens per pass
(speculative decoding). Batching raises throughput, not latency: batch 8
rides eight tokens on one pass over the weights.

After every merge, the `bench` workflow regenerates two tables:

- decode versus batch (the plan's Chart 2):
  [dev/sweep/latest.md](https://blueflare-energy.github.io/reciprocating-engine/dev/sweep/latest.md)
- prefill versus context (Chart 1):
  [dev/sweep/prefill.md](https://blueflare-energy.github.io/reciprocating-engine/dev/sweep/prefill.md)

## Layout

| Path       | Purpose                                                              |
|------------|----------------------------------------------------------------------|
| `crates/`  | The Rust workspace (one crate per layer).                            |
| `tools/`   | Repository tooling (badge generation, helpers).                      |
| `vendor/`  | Vendored upstream driver sources, unmodified. See its README.        |
| `patches/` | The project's driver patches as a standalone series. See its README. |
| `.github/` | CI, and the rendered status badges it commits.                       |

## Build

Requires the Rust toolchain pinned in `rust-toolchain.toml`.

```console
cargo build --workspace
cargo test --workspace
```

Correctness and clippy/format checks run in CI on every push and pull request.
Hardware tests run on a Gaudi2 host, not on the hosted runners. The `bench`
workflow runs `reng-bench` on a self-hosted Gaudi2 runner after every merge
to main and appends the result to the
[benchmark history](https://blueflare-energy.github.io/reciprocating-engine/dev/bench/)
(prefill and decode tok/s, and each as a percentage of the roofline ceiling
from `reng-ceiling`).

## How it works

The safetensors files are memory-mapped and the bf16 weights are uploaded
from the maps in the checkpoint's own layout, through a bounded ring of
pinned buffers, so a model is never copied on the host. An 8B model
reaches its first token about 7 s after launch, with the mapped file
pages (reclaimable by the kernel) as most of its resident set. Compiled
recipes are cached under `~/.cache/reng/recipes`, keyed by the graph's
structure and the SynapseAI version, so a model at a known shape skips
the graph compiler on later runs. Set `RENG_RECIPE_CACHE` to another
directory, or to `0` to disable the cache.

The decode path, one mechanism per item:

- The KV cache is updated in place by a ScatterND node and the greedy
  token is an argmax on the device, so a decode step moves four bytes per
  sequence over the bus.
- Attention is four nodes per layer (two `batch_gemm`s, the mask add and
  the softmax) in the prefill and batched decode recipes. The
  single-sequence decode recipe uses the fused `sdpa_recomp_fwd_bf16`
  kernel over the same tensors, where it is never slower and up to 2%
  faster. `RENG_SDPA=1` fuses every recipe and `RENG_SDPA=0` none.
- The one-row recipe takes only a token id and a position. It gathers the
  embedding row, the RoPE rows and the mask row on the device, builds its
  own cache-write indices, and lands its argmax in an id ring that the
  next launch reads from. `Generator::generate` therefore enqueues every
  step back to back and reads all the ids once.
- The batched recipe does the same for `B` sequences at once from `B` ids
  and `B` positions per launch (`BatchedGenerator::generate`).
  `RENG_DEVICE_LOOP=0` restores the per-step uploads and readback on both
  paths.
- Attention reads the whole cache every step, so the batched decoder
  compiles its recipes for the smallest bucket of positions (256,
  doubling) that holds the longest live sequence and grows on demand,
  recompiling and copying the used rows across.

Single-sequence decode is bound by per-node dispatch across the recipe,
which is where the performance work continues.

## Multi-card

Every model in the benchmark table fits one card (Qwen2.5-32B occupies
65 GB of the 96 GB). Larger models run across cards with tensor
parallelism over HCCL: `reng-tp` spawns one worker process per card
(`--modules 4,1`), each holding its Megatron shard of every layer
(`LlamaConfig::shard` / `LlamaWeights::shard`; the o/down column blocks
stay mapped views uploaded through a strided path) and a replica of the
norms, embedding and LM head. A layer is two recipes with an f32
all-reduce after `o_proj` and after `down_proj`, all enqueued on the
rank's one stream without host synchronisation; the recipes are compiled
once per kind and bound to each layer's buffers per launch. Every rank
computes the same argmax after the last all-reduce and the coordinator
checks that they agree. The prompt is the trailing ids or, for a long
one, `--prompt-file <json>` (the `"prompt"` array of a `generate.py`
reference file). A sequence is prefilled once, from position 0: the wide
prefill recipe's ScatterND is out of place, so its blocks alternate
between the sequence's cache slot and a shared scratch buffer, and a
second prefill onto a non-empty sequence is rejected rather than silently
overwriting the keys already there.

No rank outlives its coordinator holding a card. A watchdog thread in
each worker polls the hand-shake directory's `abort` file and its own
parent id; on either it calls `hcclCommAbort` and leaves through `_exit`,
and the coordinator writes `abort` and waits out a grace period before it
kills anything. Ctrl-C on a two-card 8B decode has both workers gone and
both cards free within 15 s.

DeepSeek-R1-Distill-Llama-70B (141 GB bf16) on two cards reproduces its
f32 reference 8/8 exact (free-running and teacher-forced) and decodes at
27.4 tok/s at batch 1 (36.5 ms per token, 79% of the two-card HBM
ceiling: 70.6 GB of weights per card per token, so 28.8 ms at 2.45 TB/s,
counting the embedding table as the lookup it is, as `reng-ceiling` does)
and 207 tok/s at batch 8; the 8B distill on two cards matches the single
card's ids exactly and runs 1.3x faster (166 against 130 tok/s at batch
1, 1224 against 960 at batch 8). A 1000-token prompt through the 8B
distill (four 256-row prefill blocks) gives the same eight ids on two
cards, on one card through the same path, through single-card
`reng-generate`, and in the f32 oracle. Per layer of the 70B at batch 1:
recipe A 90 us, the two all-reduces 38 us, recipe B 315 us. The host
enqueue (about 100 us per launch or collective, 322 of them per token) is
close to the device time, so fewer or cheaper launches are the next step.
With one module id the same path runs on one card without a communicator
and reproduces `reng-generate`'s ids. The box needs the kernel option
`iommu=pt`; without it the scheduler's completion counters never reach
the host.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option. Unless you explicitly state
otherwise, any contribution intentionally submitted for inclusion in this work,
as defined in the Apache-2.0 license, shall be dual licensed as above, without
any additional terms or conditions.
