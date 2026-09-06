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
- `reng-fp8`: host-side FP8 quantization of a `[out, in]` weight matrix to
  E4M3 or E5M2 codes with per-output-channel absmax scales, round to
  nearest even and saturation. Bit-exact against PyTorch's own
  `float8_e4m3fn` and `float8_e5m2` casts; `reng-fp8-quantize` reports what
  a checkpoint costs in fp8. The device gemm form is not wired yet, so the
  `--fp8` switch is off by default and refuses to run rather than fall back
  to bf16 silently.
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

### Across cards

Two objectives decide which split of N cards is the right one, and they do not
have the same answer. Single-stream tokens per second is one sequence at batch
1: the latency one user waits on. Aggregate tokens per second is every card
summed at a batch per replica: what a server serves. Five strategies compete
for them. Data parallelism runs N independent replicas; nothing crosses the
interconnect, so a stream is exactly as fast as on one card and the machine
does N times the work, as long as the model fits one card. Tensor
parallelism splits every layer over N cards, so each streams `1/N` of the
weight bytes and `1/N` of the KV cache per token, with the LM head replicated,
and pays two all-reduces per layer for it. Pipeline parallelism cuts the
layers into N consecutive stages: a token still reads every layer, one stage at
a time, so a single stream is no faster than on one card, but N micro-batches
in flight give up to N times the throughput, and the model no longer has to fit
one card. Of the splits that buy capacity it is the one that communicates
least: one activation per stage boundary against two all-reduces per layer.
The hybrids (`dp2 x tp4`, `dp4 x tp2`) compose the arithmetic of the first
two. Expert parallelism spreads a mixture of experts' routed experts over
the cards so a card streams the shared weights plus `k/N` of the expert bytes
per token; the engine has no MoE layer yet, so those rows are projections from
the config and `reng-ceiling` marks them so.

Two floors decide what a split delivers, and neither shrinks with the model.
The collective floor is two all-reduces of `hidden x batch x 4` bytes per
layer: 36 us per layer at world 2, 39 at world 4 and 49 at world 8, measured.
The host launch floor is what the step's `2 + 4L` enqueues cost the host
before the device does anything -- one recipe launch for the embedding and one
for the head, plus recipe A, recipe B and two collectives per layer -- and the
count is the same at every world, so more cards do not move it. The cheapest
step measured on this roster is 60 us plus 75.6 us a layer at world 8, which is
Llama-3.2-1B's 1.27 ms per step. That launch floor is engine cost, not physics;
it is what fewer and larger launches would move.

Both are charged, so a ceiling is the larger of the two step times:
`max(bandwidth + collective, launch)`. Llama-3.2-1B is the clearest case.
Its whole token is 1.01 ms of weight time on one card at context 192.
Splitting it eight ways takes 697 us of that off each card, and costs 790 us of
collective for the privilege, so the device side is already 1.10 ms -- and the
host cannot issue the step in under 1.27 ms whatever the device does. Eight
cards give 787 tok/s against 989 on one. The measured rows agree: every model
below 3B is slower on two cards than on one at batch 1 (Qwen2.5-0.5B 0.68x,
Qwen3-0.6B 0.72x, Llama-3.2-1B 0.89x), and the line where the eightfold
bandwidth starts to outrun both floors falls between Llama-3.2-3B and the 8B
distill. Pipeline parallelism is the mirror image: it never makes a token
faster, because the token reads the same bytes either way. What it buys is
capacity and throughput -- a model larger than one card runs, and N stages with
N micro-batches in flight retire N times the tokens without a collective on the
token path. That is why it is the eight-card pick for the 70B's aggregate row
below and for nothing else.

Measured on the eight-card box on 2026-09-06: bf16, a 128-token prompt, 64
timed decode steps, three repeats, medians. Every `reng-tp` run generated the
same 64 ids as the single card, at every world and batch; the 70B, which does
not fit one card, is identical across worlds instead. Tensor-parallel columns
are batch 1, one figure per world, with the percentage of that world's
practical ceiling; the data-parallel column is eight replicas at batch 8,
aggregate, against eight times the single-card ceiling. The last two columns
are `reng-ceiling`'s pick for eight cards, per objective, with the ceiling it
picked.

| Model | TP b1 w2 | TP b1 w4 | TP b1 w8 | DP x8 b8 | Best on 8 cards, 1 stream | Best on 8 cards, aggregate b8 |
|---|---|---|---|---|---|---|
| SmolLM2-135M | 9 heads | 9 heads | 9 heads | 59271 (12%) | data 8962 | data 515198 |
| Gemma-3-270m | layer form | layer form | layer form | not measured | data 4540 | data 277819 |
| SmolLM2-360M | 15 heads | 15 heads | 15 heads | not measured | data 3350 | data 199382 |
| Qwen2.5-0.5B | 483 (82%) | 14 heads | 14 heads, 2 kv | not measured | data 2474 | data 155741 |
| Qwen3-0.6B | 406 (80%) | 417 (84%) | 13 (3%)* | not measured | data 2018 | data 114609 |
| Llama-3.2-1B | 513 (61%) | 585 (69%) | 575 (73%)* | 37976 (61%) | data 989 | data 62177 |
| Gemma-3-1B | layer form | layer form | layer form | not measured | data 1222 | data 76848 |
| OLMo-2-0425-1B | layer form | layer form | layer form | not measured | data 948 | data 56815 |
| Qwen2.5-1.5B | 339 (67%) | 2 kv heads | 2 kv heads | not measured | data 792 | data 50077 |
| Qwen3-1.7B | 329 (65%) | 356 (72%) | 159 (35%)* | not measured | data 707 | data 43350 |
| Falcon3-1B-Base | 474 (63%) | 535 (71%) | 4 kv heads | not measured | data 870 | data 53790 |
| TinyLlama-1.1B | 459 (72%) | 505 (81%) | 4 kv heads | not measured | data 1182 | data 74544 |
| DeepSeek-R1-Distill-Qwen-1.5B | 339 (67%) | 2 kv heads | 2 kv heads | not measured | data 792 | data 50077 |
| SmolLM2-1.7B | 360 (61%) | 435 (76%) | 295 (55%)* | not measured | data 708 | data 42099 |
| granite-3.1-2b-instruct | 233 (65%) | 275 (79%) | 141 (44%)* | not measured | data 482 | data 30197 |
| Gemma-2-2B | layer form | layer form | layer form | not measured | dp2 x tp4 521 (shape only) | data 29081 |
| Qwen2.5-3B | 235 (63%) | 2 kv heads | 2 kv heads | not measured | data 397 | data 25177 |
| Llama-3.2-3B | 269 (67%) | 323 (65%) | 316 (69%)* | 16889 (71%) | dp2 x tp4 495 | data 23753 |
| SmolLM3-3B | layer form | layer form | layer form | not measured | data 397 | data 25035 |
| Phi-3-mini-4k-instruct | layer form | layer form | layer form | not measured | dp2 x tp4 435 (shape only) | data 19481 |
| Qwen3-4B | 204 (64%) | 247 (64%) | 216 (60%)* | not measured | dp2 x tp4 388 | data 18958 |
| Qwen2.5-7B | 177 (74%) | 241 (69%) | 28 heads, 4 kv | 8675 (79%) | dp2 x tp4 347 | data 11020 |
| Mistral-7B-v0.3 | 176 (73%) | 242 (68%) | 178 (44%)* | not measured | tensor 403 | data 10867 |
| Llama-3.1-8B | 166 (74%) | 223 (70%) | 270 (74%) | not measured | tensor 366 | data 10309 |
| Qwen3-8B | 154 (72%) | 202 (68%) | 220 (66%)* | not measured | tensor 333 | data 10207 |
| DeepSeek-R1-Distill-Llama-8B | 166 (74%) | 223 (70%) | 216 (59%)* | 8337 (81%) | tensor 366 | data 10309 |
| phi-4 | 104 (78%) | 10 kv heads | 10 kv heads | not measured | dp4 x tp2 133 | data 5482 |
| Qwen2.5-32B | 52 (81%) | 79 (75%) | 106 (74%)* | 2114 (87%) | tensor 143 | data 2436 |
| DeepSeek-R1-Distill-Llama-70B | 27 (88%) | 45 (82%) | 67 (79%) | not measured | tensor 84 | pipeline 1013 |

`*` the world-8 median is below the best of the three repeats, so at least one
repeat lost time the other did not. That is every world-8 cell except
Llama-3.1-8B and the 70B, whose three repeats agree. The cause is world 8's
own: 16 of 78 world-8 runs measured a per-layer all-reduce of hundreds of
microseconds to milliseconds instead of tens, and none of the 342 runs at
worlds 1, 2 and 4 did. Only Qwen3-0.6B has a median that is itself one of those
stalled runs (two of its three repeats stalled); the others lost less than a
whole repeat. The best of three is a clean run in every case: Qwen3-0.6B
425.5 tok/s against the 13.3 median, Llama-3.2-1B 628.9 against 574.7,
Mistral-7B-v0.3 296.7 against 177.9. World 8 is not a slower configuration so
much as an unreliable one.

`layer form` is `TpModel::new` rejecting the layer outright, not a divisibility
failure: GeLU-tanh activations, post-attention or post-MLP norms, a sliding
window, an attention softcap or per-layer RoPE (the three Gemmas,
Phi-3-mini-4k-instruct, SmolLM3-3B, OLMo-2-0425-1B). The head counts are
`LlamaConfig::shard`'s gate: `num_attention_heads`, `num_key_value_heads` and
`intermediate_size` must all divide the world, and KV heads split as whole GQA
groups, so 4 KV heads stop at world 4 and 2 at world 2. `reng-ceiling` models
the divisibility gate but not the layer form, so a pick marked `(shape only)`
is a shape projection the engine's tensor-parallel path would not run today.

The ceilings the percentages are against, tokens per second at batch 1 and
context 192: the practical ceiling, then in brackets the physical ceiling and
the host launch floor. The physical ceiling is the per-card bytes over
2.45 TB/s alone. The practical ceiling is the larger of the two step times the
other terms give, `max(bandwidth + collective, launch)`, so it is never above
the rate the host can enqueue: for every model below about 8B at world 8 the
launch floor is the binding one, and the bracket says so (Llama-3.2-1B 787
against the 3184 its bandwidth would allow). A `-` is a split the shape or the
memory does not admit, in the tensor columns and in the data-parallel one
alike: eight replicas of the 70B need 141.6 GB on a 103.1 GB card.

```
Qwen2.5-0.5B                    w2 588 (3881, 588)    w4 -                  w8 -                  dp8 b8 155741
Qwen3-0.6B                      w2 507 (3213, 507)    w4 495 (4563, 495)    w8 459 (5778, 459)    dp8 b8 114609
Llama-3.2-1B                    w2 841 (1632, 867)    w4 847 (2417, 847)    w8 787 (3184, 787)    dp8 b8 62177
Qwen2.5-1.5B                    w2 507 (1377, 507)    w4 -                  w8 -                  dp8 b8 50077
Qwen3-1.7B                      w2 507 (1199, 507)    w4 495 (1839, 495)    w8 459 (2507, 459)    dp8 b8 43350
Falcon3-1B-Base                 w2 751 (1461, 775)    w4 758 (2214, 758)    w8 -                  dp8 b8 53790
TinyLlama-1.1B                  w2 640 (2223, 640)    w4 625 (3973, 625)    w8 -                  dp8 b8 74544
DeepSeek-R1-Distill-Qwen-1.5B   w2 507 (1377, 507)    w4 -                  w8 -                  dp8 b8 50077
SmolLM2-1.7B                    w2 588 (1338, 588)    w4 575 (2411, 575)    w8 533 (4025, 533)    dp8 b8 42099
granite-3.1-2b-instruct         w2 358 (927, 358)     w4 350 (1723, 350)    w8 324 (3019, 324)    dp8 b8 30197
Qwen2.5-3B                      w2 373 (721, 397)     w4 -                  w8 -                  dp8 b8 25177
Llama-3.2-3B                    w2 398 (677, 507)     w4 495 (1112, 495)    w8 459 (1638, 459)    dp8 b8 23753
Qwen3-4B                        w2 320 (554, 397)     w4 388 (942, 388)     w8 359 (1450, 359)    dp8 b8 18958
Qwen2.5-7B                      w2 241 (321, 507)     w4 347 (563, 495)     w8 -                  dp8 b8 11020
Mistral-7B-v0.3                 w2 240 (337, 445)     w4 357 (651, 435)     w8 403 (1215, 403)    dp8 b8 10867
Llama-3.1-8B                    w2 223 (305, 445)     w4 321 (539, 435)     w8 366 (875, 403)     dp8 b8 10309
Qwen3-8B                        w2 213 (299, 397)     w4 299 (519, 388)     w8 333 (821, 359)     dp8 b8 10207
DeepSeek-R1-Distill-Llama-8B    w2 223 (305, 445)     w4 321 (539, 435)     w8 366 (875, 403)     dp8 b8 10309
phi-4                           w2 133 (167, 358)     w4 -                  w8 -                  dp8 b8 5482
Qwen2.5-32B                     w2 63 (75, 226)       w4 105 (143, 220)     w8 143 (262, 204)     dp8 b8 2436
DeepSeek-R1-Distill-Llama-70B   w2 31 (35, 181)       w4 56 (67, 177)       w8 84 (127, 164)      dp8 b8 -
```

`reng-ceiling <model_dir> --cards N [--batch b]` prints all of this for one
model: every strategy, its physical ceiling, its launch floor and its practical
ceiling for both objectives, whether the split is admissible and why not, and
the pick per objective with the reasons (does it fit one card, do the heads
divide, what does the split save against what the collective and the launch
floor cost). Where nothing is admissible it prints the rejection of every
strategy instead of a pick, and the JSON says `"admissible": false`.
`--collectives <file.json>` replaces the measured latency table with another
machine's, and `--launch <file.json>` the launch floor -- the second is the one
to replace after a change to the launch path, since it is engine cost rather
than physics. `tools/sweep_tp.py` runs the measured half -- `reng-tp` at every
admissible world and N data-parallel replicas of `reng-bench`, three repeats a
cell, median -- and writes the same JSON shape as the single-card benches plus
a world and a strategy per entry. It holds every card it is given, so the
`bench` workflow never runs it by default: it is a `workflow_dispatch` job with
`multi_card` ticked, or the documented command on the box.

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
stay mapped views uploaded through a strided path, and a tensor whose
data offset is not 2-aligned, which cannot be viewed at all, is copied
only where the rank's own rows and columns fall) and a replica of the
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
overwriting the keys already there. With `--batch B` every sequence's ids
are compared across the ranks, not sequence 0's alone.

The coordinator pins the interface HCCL uses for its sideband TCP
connections (`HCCL_SOCKET_IFNAME`): `--ifname <nic>`, or an inherited
setting, or the interface carrying the default route (`--ifname ""` asks
for the library's own enumeration back). The library's own default is the
first interface whose name is not `lo` or `docker`, which on this box is
the BMC's virtual NIC. `--timeout <s>` bounds the run and the workers'
wait for the coordinator's `go`; their wait for rank 0's unique id is
bounded by the same value capped at 180 s, and both end at once if the
coordinator aborts or goes away, so no rank waits on a card for a
coordinator that is not there. The hand-shake directory
(`$TMPDIR/reng-tp-<pid>-<attempt>`, mode 0700) is removed when the run
ends or is interrupted and kept when a rank failed, since the workers'
SynapseAI logs are under it (`RENG_TP_KEEP_DIR` keeps it always); a stale
one from a reused pid is removed rather than joined.

No rank outlives its coordinator holding a card. A watchdog thread in
each worker polls the hand-shake directory's `abort` file and its own
parent id; on either it calls `hcclCommAbort` and leaves through `_exit`,
and the coordinator writes `abort` and waits out a grace period before it
kills anything. Ctrl-C is caught rather than fatal: the coordinator asks
the ranks to abort, reaps them, removes the hand-shake directory and
leaves with 130. Ctrl-C on a two-card 8B decode has both workers gone and
both cards free within 15 s; a second Ctrl-C before that is through
leaves at once instead, which is the old behaviour and its old cost - no
`hcclCommAbort`, a rank possibly killed inside a collective, and the
hand-shake directory left behind.

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
That host cost is the launch floor `reng-ceiling` charges against every
multi-card ceiling (see [Across cards](#across-cards)): 60 us for the step
plus 75.6 us a layer at world 8, measured.
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
