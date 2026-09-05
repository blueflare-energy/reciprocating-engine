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
| SmolLM2-135M | 781 (8.7%) | 6782 (10.2%) | 39171 (12.0%) | 137.0k (10.8%) |
| SmolLM2-360M | 599 (17.8%) | 5432 (21.4%) | 30206 (21.2%) | 71.9k (14.1%) |
| Qwen2.5-0.5B | 750 (30.3%) | 5722 (29.3%) | 36982 (26.0%) | 103.9k (25.9%) |
| Qwen3-0.6B | 567 (28.0%) | 4518 (30.5%) | | |
| Llama-3.2-1B | 581 (58.8%) | 4545 (58.2%) | | |
| Qwen2.5-1.5B | 398 (50.3%) | 3175 (50.5%) | 20205 (43.2%) | 60.2k (45.5%) |
| Qwen3-1.7B | 366 (51.6%) | 2903 (52.9%) | | |
| Falcon3-1B-Base | 499 (57.3%) | 3867 (57.0%) | | |
| TinyLlama-1.1B | 553 (46.8%) | 4584 (49.0%) | 27828 (40.4%) | 50.0k (26.1%) |
| DeepSeek-R1-Distill-Qwen-1.5B | 396 (50.0%) | 3138 (50.0%) | | |
| SmolLM2-1.7B | 385 (54.3%) | 3049 (56.8%) | 13971 (46.6%) | 34.2k (28.7%) |
| granite-3.1-2b-instruct | 272 (56.5%) | 2014 (53.0%) | | |
| Qwen2.5-3B | 245 (61.9%) | 1932 (61.2%) | 12903 (53.6%) | 31.6k (47.4%) |
| Llama-3.2-3B | 260 (68.4%) | 2059 (68.9%) | | |
| SmolLM3-3B | 249 (62.7%) | 2021 (64.3%) | | |
| Phi-3-mini-4k-instruct | 205 (62.8%) | 1589 (64.0%) | | |
| Qwen3-4B | 197 (65.0%) | 1524 (63.9%) | | |
| Qwen2.5-7B | 133 (76.6%) | 1051 (76.2%) | 7372 (69.0%) | 18.5k (62.5%) |
| Llama-3.1-8B | 130 (79.7%) | 1032 (79.8%) | | |
| Qwen3-8B | 125 (77.3%) | 980 (76.5%) | | |
| DeepSeek-R1-Distill-Llama-8B | 129 (79.3%) | 1023 (79.2%) | | |
| phi-4 | 72 (83.7%) | 578 (84.1%) | | |

Decode percentages are against the HBM roofline, prefill against the MME
roofline. Every model is verified against a Hugging Face f32 reference
before it is listed (8 greedy tokens, exact up to bf16 near-ties). The `bench` workflow regenerates the decode-versus-batch table
(the plan's Chart 2) at
[dev/sweep/latest.md](https://blueflare-energy.github.io/reciprocating-engine/dev/sweep/latest.md)
and the prefill-versus-context table (Chart 1) at
[dev/sweep/prefill.md](https://blueflare-energy.github.io/reciprocating-engine/dev/sweep/prefill.md)
after every merge.

Weights are read from the safetensors file as bf16 in the checkpoint's own
layout and uploaded as they are; loading a 3B model takes a few seconds
and about 14 GB of host memory. Compiled recipes are cached under
`~/.cache/reng/recipes` (set
`RENG_RECIPE_CACHE` to another directory or to `0` to disable), keyed by
the graph's structure and the SynapseAI version, so a model at a known
shape skips the graph compiler on later runs.

The KV cache is updated in place by a ScatterND node and the greedy token
is an argmax on the device, so a decode step moves four bytes per sequence
over the bus. Attention reads the whole cache every step, so the batched
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
