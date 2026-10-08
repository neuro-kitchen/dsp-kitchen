"""Checks the local links of the assembled documentation site (target/site, from docs/build.sh).

Every relative `href` / `src` of the landing page, the guide and the Python API must point at a
file of the site: this catches broken links between the guide, the Python API and the Rust API,
which each tool's own build cannot see. External links (http, mailto) are not followed. Pages of
the Rust API are targets only (rustdoc checks its own links).

Run: python3 docs/check_links.py [site directory]   (default: target/site)
Exits 1 and lists the broken links when there are any.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"""(?:href|src)\s*=\s*["']([^"']+)["']""")
CHECKED = ["index.html", "guide", "api/python"]


def target(page: Path, link: str, site: Path) -> Path | None:
    """The file a relative link resolves to (None for external links and same-page anchors)."""
    parts = urlsplit(link)
    if parts.scheme or parts.netloc or link.startswith(("#", "javascript:", "data:")):
        return None
    path = unquote(parts.path)
    if not path:
        return None
    resolved = (site / path.lstrip("/")) if path.startswith("/") else (page.parent / path)
    if path.endswith("/") or resolved.is_dir():
        resolved = resolved / "index.html"
    return resolved


def main() -> int:
    site = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "target/site"
    if not (site / "index.html").is_file():
        print(f"{site}: no site here; build it with docs/build.sh", file=sys.stderr)
        return 2
    pages = []
    for entry in CHECKED:
        path = site / entry
        pages += [path] if path.is_file() else sorted(path.rglob("*.html"))
    # 404 pages link from the host's root (they are served for any missing path), so their links
    # only resolve once deployed
    pages = [p for p in pages if p.name != "404.html"]
    broken = []
    for page in pages:
        for link in LINK.findall(page.read_text(errors="replace")):
            resolved = target(page, link, site)
            if resolved is None:
                continue
            try:
                resolved.resolve().relative_to(site.resolve())
            except ValueError:
                broken.append(f"{page.relative_to(site)}: {link} (outside the site)")
                continue
            if not resolved.exists():
                broken.append(f"{page.relative_to(site)}: {link}")
    for b in sorted(set(broken)):
        print(b)
    print(f"{len(pages)} pages checked, {len(set(broken))} broken links")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
