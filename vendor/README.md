# Vendored upstream sources

Third-party sources we build against or patch, pinned by reference rather than
committed in full, so our changes stay visible as a separate patch series.

## habanalabs

The open-source Intel `habanalabs` accelerator driver. The upstream source is
47 MB, so it is not committed. Instead:

- `habanalabs/SOURCE` pins the exact upstream package, version, and sha256.
- `habanalabs/fetch.sh` downloads that package, verifies the checksum, extracts
  the source into `habanalabs/src/` (git-ignored), and applies the patch series
  from `../patches/habanalabs-1.19.0-561/`.

```console
vendor/habanalabs/fetch.sh          # needs the Intel Gaudi vault apt repo
DEB_PATH=/path/to.deb vendor/habanalabs/fetch.sh   # or a pre-downloaded .deb
```

## Rules

- `SOURCE` records the exact upstream revision and checksum. Do not edit the
  fetched tree in place; all changes live under `../patches/` and are applied by
  `fetch.sh`.
- Upstream licenses (the habanalabs driver is GPL-2.0) govern the fetched tree
  regardless of this repository's own license.
