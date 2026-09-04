# Driver patches

Our changes to the vendored upstream sources in `../vendor/`, kept as an
ordered patch series and applied by `../vendor/<tree>/fetch.sh`.

## habanalabs-1.19.0-561

Patches onto the `habanalabs` DKMS driver (see `../vendor/habanalabs/SOURCE`).

| Patch | What |
|---|---|
| `0001-guard-min-max-macros.patch` | Guard the Gaudi2 HBM-bringup `MIN`/`MAX` macros with `#ifndef` so the driver builds on kernel 6.8+, which now defines them in `<linux/minmax.h>`. |

`series` lists the apply order.

## Rules

- Patches apply cleanly, in order, onto the exact upstream revision recorded in
  the matching `vendor/<tree>/SOURCE`.
- One logical change per patch, with a message explaining what and why.
- When bumping the pinned revision, rebase the series and update `SOURCE`.
