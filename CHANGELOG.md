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
  every merge to main and publishes the history to GitHub Pages.

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
