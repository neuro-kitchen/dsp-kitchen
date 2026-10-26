#!/usr/bin/env bash
# Checks that the documentation is complete and true to the code:
#
#   1. Python stub up to date: regenerated from the bindings and compared with the committed one
#   2. Python docstrings: numpydoc validation, constructor arguments, examples parse
#      (docs/check_docstrings.py)
#   3. Python API site builds with no warnings (Zensical --strict: broken references, unknown
#      parameters)
#   4. Rust API builds with no warnings (broken or ambiguous intra-doc links)
#   5. Rust doctests pass (they run on the default compute runtime)
#
# Usage: docs/check.sh [--no-doctests]
#   --no-doctests   skip step 5 (machines without a GPU runtime, e.g. CI)
#
# Needs: cargo, uv. Exits non-zero at the first failing step.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

crates=(dsp-core dsp-io dsp-base dsp-synapse dsp-synapse-ml dsp-synapse-hub dsp-stream dsp-view)
stub=dsp_kitchen_py/dsp_kitchen_bindings/dsp_kitchen_bindings.pyi

echo "==> 1. Python stub up to date"
before="$(mktemp)"
trap 'rm -f "$before"' EXIT
cp "$stub" "$before"
cargo run -q -p dsp_kitchen_py --bin stub_gen
if ! diff -q "$before" "$stub" >/dev/null; then
  diff -u "$before" "$stub" | head -40 || true
  echo "The stub was out of date with the bindings; it has been regenerated: commit $stub." >&2
  exit 1
fi

echo "==> 2. Python docstrings"
uv run --no-project --with numpydoc python docs/check_docstrings.py

echo "==> 3. Python API site (strict)"
rm -rf docs/api/python/site
uv run --no-project --with zensical --with 'mkdocstrings[python]' \
  zensical build --clean --strict -f docs/api/python/mkdocs.yml

echo "==> 4. Rust API (warnings are errors)"
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps $(printf -- '-p %s ' "${crates[@]}")

if [[ "${1:-}" != "--no-doctests" ]]; then
  echo "==> 5. Rust doctests"
  cargo test --doc $(printf -- '-p %s ' "${crates[@]}")
fi

echo "Documentation checks passed."
