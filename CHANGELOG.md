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
  models: SmolLM2-135M/360M/1.7B, Qwen2.5-0.5B/1.5B/3B/7B.
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
