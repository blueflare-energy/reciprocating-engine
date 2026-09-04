# Driver patches

Our modifications to the vendored upstream sources under `../vendor/` live here
as a standalone, ordered patch series, so they can be reviewed on their own and
rebased onto new upstream revisions.

## Layout

Each patched upstream tree gets its own subdirectory whose name matches the
vendored tree it targets. Within it, patches are numbered to define apply order:

```
patches/
  <vendored-tree>/
    0001-short-description.patch
    0002-short-description.patch
    series            # optional: explicit ordered list, one patch per line
```

## Rules

- Patches apply cleanly, in order, onto the exact upstream revision recorded in
  the corresponding `vendor/<tree>/SOURCE`.
- One logical change per patch, with a message explaining the what and the why.
- When bumping the vendored revision, rebase the series and update both trees in
  the same change.
