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
for prompt blocks and a one-row one for decode steps. SmolLM2-135M at batch 1
decodes at about 730 tok/s (1.3 ms per step, 8% of the HBM roofline); the
step is dominated by per-node dispatch across the 30-layer recipe, which is
where the performance work against the ceiling continues.

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
