# Contributing

Thanks for your interest in Reciprocating Engine.

## Ground rules

- The workspace must build, and `cargo test --workspace` must pass.
- Code must be free of `clippy` warnings and formatted with `rustfmt`. CI
  enforces both with `-D warnings`; run them before pushing:

  ```console
  cargo fmt --all
  cargo clippy --workspace --all-targets --all-features
  cargo test --workspace
  ```

- Keep changes focused. New behavior should come with tests. Hardware-specific
  tests that require a Gaudi2 accelerator must degrade gracefully (skip, not
  fail) on hosts without one, so the hosted CI stays green.

## Drivers

Upstream driver sources are vendored unmodified under `vendor/`. Do not edit
them in place. Our changes live as a patch series under `patches/`; see
`patches/README.md`.

## Licensing of contributions

Unless you state otherwise, any contribution you submit for inclusion is dual
licensed under Apache-2.0 and MIT, matching the project license.
