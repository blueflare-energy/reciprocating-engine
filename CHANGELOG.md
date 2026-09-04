# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Workspace scaffold with four crates: `reng-core`, `reng-hal`, `reng-ceiling`,
  `reng-cli`.
- `reng-core`: dtype, tensor shape, and device-identifier types.
- `reng-hal`: Gaudi2 device discovery through the kernel `accel` subsystem,
  with silicon-stepping detection and no vendor-userspace dependency.
- `reng-ceiling`: first-principles roofline calculator for prefill and decode
  ceilings on Gaudi2, MoE-aware, driven from a HuggingFace `config.json`.
- `reng devices` and `reng ceiling` CLI commands.
- Vendored (by reference) the habanalabs driver, with patch 0001 guarding the
  Gaudi2 MIN/MAX macros so the 1.19.0 driver builds on kernel 6.8+.
- CI pipeline (build, format, clippy, tests, coverage) and self-hosted status
  badges.
