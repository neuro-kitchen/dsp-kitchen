#!/usr/bin/env bash
# Builds the documentation site into target/site/:
#
#   target/site/              landing page (docs/site/index.html)
#   target/site/guide/        the guide (mdBook, docs/book)
#   target/site/api/python/   the Python API (Zensical + mkdocstrings, docs/api/python, from the stubs)
#   target/site/api/rust/     the Rust API (rustdoc of the library crates)
#
# Then checks every local link of the landing page, the guide and the Python API (docs/check_links.py).
# Completeness checks (stub, docstrings, warnings, doctests) are in docs/check.sh.
#
# Usage: docs/build.sh [--stubs]
#   --stubs   regenerate the Python stub first (cargo run -p dsp_kitchen_py --bin stub_gen)
#
# Needs: mdbook, uv (fetches zensical and mkdocstrings for the build; nothing is added to the
# project), cargo. Serve locally with: python3 -m http.server -d target/site
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
site="$root/target/site"
cd "$root"

# Library crates documented in the Rust API (the app and the CLI are programs, not libraries)
crates=(dsp-core dsp-io dsp-base dsp-synapse dsp-synapse-ml dsp-synapse-hub dsp-stream dsp-view)

if [[ "${1:-}" == "--stubs" ]]; then
  echo "==> Python stub"
  cargo run -q -p dsp_kitchen_py --bin stub_gen
fi

rm -rf "$site"
mkdir -p "$site/api"

echo "==> Landing page"
cp docs/site/index.html "$site/index.html"

echo "==> Guide (mdBook)"
mdbook build docs/book --dest-dir "$site/guide"

echo "==> Python API (Zensical + mkdocstrings)"
rm -rf docs/api/python/site
uv run --no-project --with zensical --with 'mkdocstrings[python]' zensical build --clean --strict -f docs/api/python/mkdocs.yml
cp -r docs/api/python/site "$site/api/python"

echo "==> Rust API (rustdoc)"
cargo doc --no-deps $(printf -- '-p %s ' "${crates[@]}")
cp -r target/doc "$site/api/rust"

echo "==> Links between the parts"
python3 docs/check_links.py "$site"

echo "Site in $site"
