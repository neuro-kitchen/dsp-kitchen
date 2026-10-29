# Changelog

All notable changes, newest first. Versions follow [Semantic Versioning](https://semver.org); in
0.0.x any release may change the API. The crates, the Python package and `dspk-sorters` share one
version number.

## Unreleased

- CI runs only the checks a change can affect; documentation alone runs none.
- `CITATION.cff`, issue forms, pull request template, `CONTRIBUTING.md`, `SECURITY.md`, this
  changelog.

## 0.0.4 — 2026-10-29

### Added

- `dspk-sorters`: every spike sorter (Kilosort4, EMUsort, MountainSort 5, SpyKING CIRCUS 2,
  Tridesclous 2) in one program, GPL-3.0 as a whole, attached to each GitHub release.
  `dspk sort …` runs it as a separate process; `dspk sorters install` downloads and verifies it
  (SHA-256 manifest); `dspk-sorters licenses` prints every licence.
- `dspk-pipeline` is published on crates.io.
- dspk-ops: `math::execute_channel_mad` (NumPy median and median absolute deviation per channel,
  on the device); `reduce::row_kth` (percentiles) and `row_abs_dev_kth`.
- `benchmarks/signal-operations/scipy_timing.py`: time and agreement against scipy / NumPy.
- An example pipeline spec, `playground/base/pipelines/neuropixels_preprocess.json`.

### Changed

- Vectorized kernels (`Vector<F, N>`), chosen per device by the autotuner and bit-identical to the
  scalar ones: FIR, `resample_poly`, running median, common median and common average reference,
  IIR pass. On CubeCL's CPU runtime 2–10× faster (e.g. FIR 175 → 33 ms, common median reference
  1561 → 239 ms on 384 × 60 000); several GPU kernels faster too.
- Common median reference by bitonic sort on GPUs (autotuned against rank counting).
- The `dspk` command line no longer links any sorter: it is MIT OR Apache-2.0.
- Synapse split into one crate per sorter, each with its licence and notice: `dspk-ks4`
  (GPL-3.0), `dspk-ms5` (Apache-2.0), `dspk-sc2`, `dspk-tdc2`; shared parts in `dspk-synapse`.
- SpyKING CIRCUS 2 and Tridesclous 2 compute their noise levels on the device.

### Fixed

- A process could abort at exit after its work ("thread local panicked on drop", upstream
  tracel-ai/cubecl#1981): only wgpu devices are shut down at exit now.
- The Python sdist carries its licence files (PyPI rejected the 0.0.3 sdist).
- Release workflow: a re-run finishes a partial crates.io publish; crates new to crates.io are
  skipped with a warning; PyPI receives only wheels and the sdist.

## 0.0.3 — 2026-10-28

Pipeline engine v2 and `dspk run`; the full set of Python operations; recordings uploaded in their
stored integer type and converted on the device; a faster CPU runtime (contiguous work per thread).

## 0.0.2 — 2026-10-28

Python bindings (`pip install dspk`), a fix to the dspk-ops CPU runtime, CI and tag-driven releases
with trusted publishing.

## 0.0.1 — 2026-10-28

First release of the crates on crates.io.
