# Documentation and Python typing plan (2026-10-08)

Goal: dsp-kitchen easy to learn from Python. Editors show every class, argument, default and
docstring (completion, hover, type checking); a scikit-learn-style API reference (Parameters,
Returns, Examples per item); one entry page to every part of the documentation.

Decided with the user:
- **mdBook stays the guide** (concepts, pipelines, sorters, GPU engineering, benchmarks).
- **Python API reference: Zensical + mkdocstrings**, generated from the stubs and their numpydoc
  docstrings (griffe reads `.pyi` statically: the docs build does not need the compiled extension).
  Not MkDocs: MkDocs 2.0 removes the plugin system (mkdocstrings would stop working) and Material for
  MkDocs is in maintenance; Zensical is its authors' successor, reads the same `mkdocs.yml`, and the
  mkdocstrings author is on its team. Checked 2026-10-08: our config builds unchanged ("No issues
  found"). Not pinned.
- **Rust API reference: rustdoc** (`cargo doc`).
- **One landing page** linking the three, served from one site:
  `/` (landing), `/guide/` (mdBook), `/api/python/` (MkDocs), `/api/rust/` (rustdoc).

Why the editor showed nothing: the bindings are a compiled module with no `.pyi` stubs (the old
hand-written stub was parked: PY14), while `py.typed` tells editors the package is typed; the config
constructors took `**kwargs`, so no argument was visible; the Python wrappers had no type hints and
stale docstrings.

| # | Phase | State |
|---|---|---|
| D1 | Stub generation: `pyo3-stub-gen` 0.23.1 (PyO3 0.27–0.29) in `dsp_kitchen_py`, a `stub_gen` binary writing `dsp_kitchen_bindings/dsp_kitchen_bindings.pyi` | **done**: all 87 items annotated, 1 388-line stub |
| D2 | Sorters surface (`synapse/ml.rs`): stub annotations, explicit config signatures (defaults from Rust `Default`; unknown names still an error), numpydoc docstrings (shapes, dtypes, **units**, defaults and their source, returns, example); typed Python wrappers | **done** (EMUsort nested config was already shared: no fix needed) |
| D3 | Python API site (`docs/api/python/`): `mkdocs.yml`, mkdocstrings pages per module, built with `uv run --with zensical --with mkdocstrings[python] zensical build -f docs/api/python/mkdocs.yml` (no new project dependencies) | **done** (content of the non-sorter pages waits for D5) |
| D4 | Landing page + `docs/build.sh` assembling the one site (`target/site/`) | **done**: `docs/site/index.html`, `docs/build.sh [--stubs]`, book `site-url` → `/dsp-kitchen/guide/`, `.github/workflows/docs.yml` builds and publishes the whole site; all landing links resolve |
| D5 | The rest of the bindings: io / recordings, probes, pipeline and filters, math, linalg, spatial, synapse | **done**: all 329 stub items documented; every public function and method with arguments has a numpydoc `Parameters` section (constructors on their class; settings classes per property), with units, shapes, defaults and `Returns` / `Raises`; array returns typed `NDArray[float32]`. Corrected on the way: `cluster_kde_merge` (`dip_threshold` is the `3·(1 − valley/peak)` dip score; `min_cluster_size` bounds the initial k and each side of a re-cut), the grid-convolution localizer, `ModelHub.verify` statuses (`"[INSTALLED]"` …), `Pipeline` parameters moved to its constructor, a docstring that had landed on a private task (`SpatialWhitening.run`) |
| D6 | Rust reference | **done**: rustdoc warnings 52 → 0 (`fn@` / `mod@` on 39 ambiguous links; math written as code spans instead of `$x[n]$`; private or feature-gated targets as plain code); front pages for dsp-base and dsp-synapse (what is where, how to get a device client); module docs for `filter`, `spatial`, `math`, `linalg`; `Pipeline` fully documented; `# Errors` on all 34 public fallible functions of the two crates; doctests (run on the default runtime): `Pipeline` (CAR + band-pass on the device), `Pipeline::validate`, dsp-synapse detection + spatial deduplication. Not done: per-function `# Examples` on the other ~480 public functions (front pages and module docs point to the entry points instead) |
| D7 | Keep it true | **done**: `docs/check.sh [--no-doctests]`: (1) stub regenerated and compared, (2) `docs/check_docstrings.py`: numpydoc validation (checks not followed listed with reasons), constructor arguments vs documented parameters, `>>>` examples parse, (3) strict Python API build, (4) rustdoc with `-D warnings`, (5) Rust doctests. `docs/build.sh` ends with `docs/check_links.py`: every local link of the landing page, guide and Python API resolves in the assembled site. CI (`docs.yml`) runs `check.sh --no-doctests` before building (no GPU there) and now also triggers on `dsp_kitchen_py/src/**`. Guide ↔ API links: introduction and each sorter's parameters page → Python API; Python sorter pages → guide pages; Python index → Rust API. Python examples are checked for syntax only: running them needs recordings |

Notes:
- `numpydoc lint` reports on **stderr** (the first version of the checker read stdout and saw nothing).
- Zensical caches pages: builds use `--clean`, otherwise edited docstrings can be served stale.
- Links from Python API pages to the other parts are written relative to the source `.md` file;
  Zensical adds one level for directory URLs.

Docstring convention (numpydoc):

```text
One-line summary.

Longer description: what it does, where it comes from (paper, upstream).

Parameters
----------
name : type, default value
    Meaning, **unit** (µV, whitened σ, samples, ms, Hz), shape for arrays
    (`[channels, samples]`, float32).

Returns
-------
name : type
    Meaning.

Examples
--------
>>> ...
```
