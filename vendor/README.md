# Vendored upstream sources

This directory holds third-party sources that we build against or patch,
vendored **unmodified** so that our changes are always visible as a separate,
reviewable patch series.

## Contents

Nothing is vendored yet. The primary planned entry is the open-source Intel
`habanalabs` accelerator driver, pinned to a specific upstream revision.

## Rules

- Files here are a faithful copy of a specific upstream revision. Record the
  source URL and the exact commit or tag in a `SOURCE` file alongside each
  vendored tree.
- Do not edit vendored files in place. All modifications live under
  `../patches/` and are applied on top of the pinned revision.
- Upstream licenses are preserved verbatim within each vendored tree and govern
  that tree regardless of the repository's own license.
