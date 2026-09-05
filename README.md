# Reciprocating Engine

[![ci](https://github.com/blueflare-energy/reciprocating-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/blueflare-energy/reciprocating-engine/actions/workflows/ci.yml)
[![bench](https://github.com/blueflare-energy/reciprocating-engine/actions/workflows/bench.yml/badge.svg)](https://blueflare-energy.github.io/reciprocating-engine/dev/bench/)
![lines of code](.github/badges/loc.svg)
![crates](.github/badges/crates.svg)
![dependencies](.github/badges/deps.svg)
![tests](.github/badges/tests.svg)
![coverage](.github/badges/coverage.svg)
![code quality](.github/badges/quality.svg)

An open inference engine built from the ground up for Intel Gaudi2 (HL-225)
accelerators, written in Rust.

> Status: early. The hardware abstraction layer and tooling are taking shape
> first; the compute and serving paths are under active development. Interfaces
> will change without notice until a tagged release.

## Why

Gaudi2 is fast silicon with a comparatively thin open software surface. Most
inference stacks target it through large vendor frameworks. Reciprocating
Engine takes the opposite approach: a small, auditable Rust codebase that talks
to the accelerator directly, so every layer from device discovery to the
compute graph is readable, measurable, and tunable.

The project is benchmark-driven. Every model and batch shape we run is measured
against a hardware ceiling derived from first principles (memory bandwidth,
compute throughput, interconnect), so performance work is always relative to
what the machine can actually do rather than to an arbitrary baseline.

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
  the new token's logits cross the bus. SmolLM2-135M matches the f32
  transformers reference on per-position argmax (cosine 1.000 on the last
  logits at 128 tokens) and greedy decoding matches token for token except
  at f32 near-ties, which bf16 cannot resolve. `tools/oracle/` holds the
  reference scripts.

```console
$ reng devices
INDEX  PCI              STEPPING
0      0000:cc:00.0     A0
1      0000:cd:00.0     A0
...
```

Generation compiles two recipes over the same weights and cache: a wide one
for prompt blocks and a one-row one for decode steps; `--batch` decodes
several sequences in lockstep over a multi-slot cache. Measured on one card
(bf16, 128-token prompt, `tools/oracle` references match), percentages
against the `reng-ceiling` roofline:

| Model | Decode b1 tok/s | Decode b8 tok/s | Decode b64 tok/s | Prefill 1024 tok/s |
|---|---|---|---|---|
| SmolLM2-135M | 770 (8.6%) | 6764 (10.2%) | 39965 (12.3%) | 137.0k (10.8%) |
| Gemma-3-270m | 939 (20.7%) | 7413 (21.1%) | | |
| SmolLM2-360M | 592 (17.6%) | 5382 (21.2%) | 29918 (21.0%) | 71.9k (14.1%) |
| Qwen2.5-0.5B | 727 (29.4%) | 5744 (29.4%) | 37118 (26.1%) | 103.9k (25.9%) |
| Qwen3-0.6B | 567 (28.0%) | 4518 (30.5%) | | |
| Llama-3.2-1B | 573 (57.9%) | 4486 (57.4%) |  |  |
| Gemma-3-1B | 474 (38.7%) | 3698 (38.3%) | | |
| OLMo-2-0425-1B | 505 (53.2%) | 4191 (57.9%) |  |  |
| Qwen2.5-1.5B | 387 (48.9%) | 3156 (50.2%) | 20195 (43.2%) | 60.2k (45.5%) |
| Qwen3-1.7B | 366 (51.6%) | 2903 (52.9%) | | |
| Falcon3-1B-Base | 503 (57.8%) | 3930 (57.9%) | 22492 (49.9%) |  |
| TinyLlama-1.1B | 487 (41.2%) | 4592 (49.1%) | 27058 (39.3%) | 50.0k (26.1%) |
| DeepSeek-R1-Distill-Qwen-1.5B | 399 (50.4%) | 3161 (50.3%) | 19792 (42.3%) |  |
| SmolLM2-1.7B | 380 (53.5%) | 3091 (57.5%) | 14016 (46.8%) | 34.2k (28.7%) |
| granite-3.1-2b-instruct | 270 (55.9%) | 2128 (56.0%) |  |  |
| Gemma-2-2B | 282 (60.4%) | 2188 (59.7%) | | |
| Qwen2.5-3B | 244 (61.5%) | 1940 (61.5%) | 12932 (53.7%) | 31.6k (47.4%) |
| Llama-3.2-3B | 257 (67.6%) | 2057 (68.8%) |  |  |
| SmolLM3-3B | 247 (62.2%) | 2012 (64.0%) |  |  |
| Phi-3-mini-4k-instruct | 204 (62.4%) | 1590 (64.0%) | 5426 (38.3%) |  |
| Qwen3-4B | 196 (64.6%) | 1525 (63.9%) |  |  |
| Qwen2.5-7B | 137 (78.9%) | 1063 (77.1%) | 7516 (70.3%) | 18.5k (62.5%) |
| Mistral-7B-v0.3 | 136 (79.2%) | 1073 (78.7%) |  |  |
| Llama-3.1-8B | 130 (79.8%) | 1030 (79.7%) |  |  |
| Qwen3-8B | 125 (77.1%) | 980 (76.5%) |  |  |
| DeepSeek-R1-Distill-Llama-8B | 130 (79.8%) | 1030 (79.7%) | 6894 (71.3%) |  |
| phi-4 | 72 (83.5%) | 578 (84.1%) |  |  |

Decode percentages are against the HBM roofline, prefill against the MME
roofline. Every model is verified against a Hugging Face f32 reference
before it is listed (8 greedy tokens, exact up to bf16 near-ties). The `bench` workflow regenerates the decode-versus-batch table
(the plan's Chart 2) at
[dev/sweep/latest.md](https://blueflare-energy.github.io/reciprocating-engine/dev/sweep/latest.md)
and the prefill-versus-context table (Chart 1) at
[dev/sweep/prefill.md](https://blueflare-energy.github.io/reciprocating-engine/dev/sweep/prefill.md)
after every merge.

The safetensors files are memory-mapped and the bf16 weights are uploaded
from the maps in the checkpoint's own layout, through a bounded ring of
pinned buffers, so a model is never copied on the host: an 8B model
reaches its first token about 7 s after launch, with the mapped file
pages (reclaimable by the kernel) as most of its resident set. Compiled
recipes are cached under
`~/.cache/reng/recipes` (set
`RENG_RECIPE_CACHE` to another directory or to `0` to disable), keyed by
the graph's structure and the SynapseAI version, so a model at a known
shape skips the graph compiler on later runs.

The KV cache is updated in place by a ScatterND node and the greedy token
is an argmax on the device, so a decode step moves four bytes per sequence
over the bus. Attention is four nodes per layer (two `batch_gemm`s, the
mask add and the softmax) in the prefill and batched decode recipes and
the fused `sdpa_recomp_fwd_bf16` kernel over the same tensors in the
single-sequence decode recipe, where it is never slower and up to 2%
faster; `RENG_SDPA=1` fuses every recipe and `RENG_SDPA=0` none.
Single-sequence decode goes further: the one-row recipe takes only a
token id and a position, gathers the embedding row, the RoPE rows and
the mask row on the device and builds its own cache-write indices, and
its argmax lands in an id ring that the next launch reads from, so
`Generator::generate` enqueues every step back to back and reads all the
ids once (`RENG_DEVICE_LOOP=0` restores the per-step uploads and
readback). Attention reads the whole cache every step, so the batched
decoder compiles its recipes for the smallest bucket of positions (256,
doubling) that holds the longest live sequence and grows on demand,
recompiling and copying the used rows across. Single-sequence decode is
bound by per-node dispatch across the recipe, which is where the
performance work continues.

## Layout

| Path        | Purpose                                                        |
|-------------|----------------------------------------------------------------|
| `crates/`   | The Rust workspace (one crate per layer).                      |
| `tools/`    | Repository tooling (badge generation, helpers).                |
| `vendor/`   | Vendored upstream driver sources, unmodified. See its README.  |
| `patches/`  | Our driver patches as a standalone series. See its README.     |
| `.github/`  | CI, and the rendered status badges it commits.                 |

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

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option. Unless you explicitly state
otherwise, any contribution intentionally submitted for inclusion in this work,
as defined in the Apache-2.0 license, shall be dual licensed as above, without
any additional terms or conditions.
