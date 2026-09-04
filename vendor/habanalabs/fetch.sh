#!/usr/bin/env bash
# Reconstruct the vendored habanalabs driver source and apply our patch series.
#
# The upstream source (47 MB) is not committed. This script pins and fetches
# the exact package (see SOURCE), verifies its checksum, extracts the source
# into ./src, then applies the patches in
# ../../patches/habanalabs-1.19.0-561/. Re-runnable and idempotent.
#
# Provide a pre-downloaded .deb with DEB_PATH=/path/to.deb, otherwise the
# Intel Gaudi vault apt repo must be configured so `apt-get download` resolves.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
pkg="habanalabs-dkms"
ver="1.19.0-561"
deb="${pkg}_${ver}_all.deb"
sha="9545f71e93c6d137558189faa4d52b613e5b36e8ef44468279cda5a6bc6e8abb"
srcdir="$here/src"
patchdir="$root/patches/habanalabs-${ver}"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work"

if [ -n "${DEB_PATH:-}" ]; then
  cp "$DEB_PATH" "$deb"
else
  apt-get download "${pkg}=${ver}"
fi

echo "${sha}  ${deb}" | sha256sum -c -

rm -rf "$srcdir"
mkdir -p "$srcdir"
dpkg-deb -x "$deb" ex
cp -a "ex/usr/src/habanalabs-${ver}/." "$srcdir/"

if [ -f "$patchdir/series" ]; then
  while IFS= read -r p; do
    [ -z "$p" ] && continue
    case "$p" in \#*) continue ;; esac
    echo "applying $p"
    patch -p1 -d "$srcdir" <"$patchdir/$p"
  done <"$patchdir/series"
fi

echo "vendored source ready at: $srcdir"
