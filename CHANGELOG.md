# Changelog

All notable changes, newest first. Versions follow [Semantic Versioning](https://semver.org); in
0.0.x any release may change the API. The crates, the Python package and `dspk-sorters` share one
version number.

## 0.0.5 — 2026-11-05

### Added

- Python: `dspk.ops.math.channel_mad` (median absolute deviation per channel, on the device).
- dspk-pipeline: `RunReport::timings` (`StageTimings`: read wait, device, write wait, encode,
  disk), in the CLI output, `dspk run --json` and the Python reports.
- dspk-io: `ZarrWriter`, a Zarr recording written from blocks of columns in order.
- dspk-ops: `core::buffer::{fill, copy}`, `core::layout::transpose_strided`.

### Changed

- **Synapse ready for publication, one crate per upstream lineage** (none of these names was on
  crates.io before):
  - `dspk-synapse-io` merged into `dspk-synapse` (`dspk_synapse::io`).
  - `dspk-synapse-hub` → `dspk-hub` (`hub/`): generic verified downloads with data catalogs per
    domain (feature `synapse`: Kilosort4's `wTEMP.npz`); `dspk_hub::pull(id)`. The
    `dspk-sorters` installer moved into the CLI. `dspk hub` lists the hub catalog.
  - `dspk-synapse-ml`: manifests only say how to run a model; downloads come from the hub
    catalog (no `ModelHub`; `DSPK_MODEL_CATALOG` overrides model manifests, `DSPK_HUB_CATALOG`
    downloads). The CLI no longer depends on it.
  - Sorters: `dspk-ks4` → `dspk-sorter-kilosort` (features `ks4`, `emusort`; templates through
    `dspk-hub`, no ML dependency; provenance text says upstream's code was read);
    `dspk-ms5` → `dspk-sorter-ms5` (with its own isosplit6); `dspk-sc2` + `dspk-tdc2` →
    `dspk-sorter-si` (features `sc2`, `tdc2`; SpikeInterface's isosplit and waveform cutting, no
    MS5 dependency). isosplit is duplicated into its two owners, so `dspk-synapse` is MIT OR
    Apache-2.0 with no Apache-only part.
  - `dspk` (umbrella): features `synapse`, `hub`.
- **Breaking (Rust):** dspk-ops has one error type, `OpsError` / `OpsResult` (was `FilterError`
  for filters and `dspk_core::DspResult` for stored-sample unpacking); `From<OpsError> for
  DspError` keeps `?` working in code returning `DspResult`.
- **Breaking (Rust):** every operation on device buffers is `execute_<op>`, data first:
  `execute_filter(client, input, output, channels, samples, sample_rate, spec)` (was the filter
  first), `resample_poly[_with]` → `execute_resample_poly[_with]`, `decimate` →
  `execute_decimate`. Parameter names of `execute_scaling` / `execute_clamp`: `len`, `scale`,
  `offset` / `min`, `max`.
- Vectorized kernels run on every row length: rows the vector width does not divide (385
  channels, 66 234-sample zero-phase windows) used to fall back to scalar code; FIR, running
  median, `resample_poly`, CAR and CMR now store their last vector lane by lane, and the IIR's
  time-major lanes are padded to the width. Still bit-identical to the scalar kernels. Autotune
  keys renamed, so cached choices are measured again once.
- `dspk run` / `dspk.pipeline.run`: a writer thread takes downloaded windows through a bounded
  queue, so writing overlaps reading and the device; Zarr output is written chunk by chunk as the
  run goes instead of converted from a temporary raw file at the end.
- **Parallel, compressed writing (default on):** Zarr chunks are compressed and stored on several
  threads, raw output frames built on several threads. New `write` node settings `compression`
  (`{"codec": "zstd", "level": 3}` default, `blosc_lz4`, `none`), `chunk_samples` (Zarr chunk
  length, default 5000, independent of the processing windows) and `parallel` (default `true`);
  the same in `dspk_io::ZarrOptions` and Python `dspk.io.write_zarr(..., compression="zstd",
  level=None, parallel=True)` (its `chunk_samples` default is now 5000, `ZARR_CHUNK_SAMPLES`).
  Output is float32 whatever the input's stored type. **Breaking (Rust):** `dspk_io::write_zarr`
  and `ZarrWriter::create` take `&ZarrOptions` instead of a chunk length.
- Dependencies: pyo3 / numpy 0.30, gpui-kit 0.9 (dspk-app), tonic 0.14.9, toml 1.2, maturin 1.16.

### Sorters (speed)

- MountainSort 5 on 2 s × 384 channels (CUDA): **117 s → 51 s**.
  - The per-channel classifiers are fitted on several threads (each with its own device stream;
    same arithmetic per channel; byte-identical output) and their PCA is exact through the
    snippets' Gram matrix (`TopComponentsOptions::dual_max_rows`, MS5
    `classifier_dual_max_rows = 1024`): classifiers 84 s → 14 s. Exact like upstream's, so
    results can differ from 0.0.4's randomized fit (one spike of 115 on the test excerpt).
  - The isosplit6 subdivision runs the two halves of each split on two threads (same labels).
  - `Mountainsort5Config::parallel` (default `true`) switches both off.
- dspk-ops: `TopComponentsOptions::dual_max_rows` (default 0, off): exact PCA of few rows
  through their Gram matrix.
- CLI (`dspk sort`, `dspk-sorters`): a recording with more channels than its probe has sites
  (SpikeGLX's sync channel) is read on the probe's channels, with a note, instead of refused.

### Project

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
