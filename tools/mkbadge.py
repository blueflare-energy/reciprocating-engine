#!/usr/bin/env python3
"""Minimal flat-square SVG badge generator with no external dependencies.

Usage: mkbadge.py LABEL VALUE COLOR OUTPUT.svg

Used both locally and in CI so committed badges are fully reproducible and do
not depend on any third-party badge service.
"""

import sys
from xml.sax.saxutils import escape


def _width(text: str) -> int:
    # Rough width estimate for an 11px sans-serif face, plus horizontal padding.
    return int(len(text) * 6.5) + 12


def badge(label: str, value: str, color: str) -> str:
    lw = _width(label)
    vw = _width(value)
    total = lw + vw
    el, ev = escape(label), escape(value)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total}" height="20" '
        f'role="img" aria-label="{el}: {ev}">'
        f"<title>{el}: {ev}</title>"
        f'<rect width="{total}" height="20" rx="3" fill="#555"/>'
        f'<rect x="{lw}" width="{vw}" height="20" rx="3" fill="{escape(color)}"/>'
        f'<rect x="{lw}" width="4" height="20" fill="{escape(color)}"/>'
        f'<g fill="#fff" text-anchor="middle" '
        f'font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">'
        f'<text x="{lw / 2:.0f}" y="14">{el}</text>'
        f'<text x="{lw + vw / 2:.0f}" y="14">{ev}</text>'
        f"</g></svg>"
    )


def main() -> int:
    if len(sys.argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    _, label, value, color, out = sys.argv
    with open(out, "w", encoding="utf-8") as f:
        f.write(badge(label, value, color))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
