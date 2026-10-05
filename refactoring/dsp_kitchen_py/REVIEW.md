# dsp_kitchen_py — review (2026-10-05)

Read-only review; no code changed. About 5.1 k lines of Rust (PyO3 0.29, numpy 0.29), a Python
SDK (`dsp_kitchen`), a stub (`dsp_kitchen_bindings.pyi`, 819 lines) and 6 test files. Built
with maturin from the root `pyproject.toml`.

Each finding below has an id, what is wrong, and a proposal. **D** = a decision needed from you.

## Build state

It does not build: the bindings depend on every crate below them, and the cleanup renamed or
parked those APIs. The breakage lists are already in `refactoring/{dsp-base,dsp-io,dsp-synapse,
dsp-synapse-ml}/README.md`; in summary:

| Area | Binding file(s) | Upstream change |
|---|---|---|
| B1 probes | `synapse/probe.rs` (17 sites), `dsp_core::SensorLayout` | probes moved to `dsp_io::neuro::probe`; `SensorLayout` is in dsp-io |
| B2 streaming | `synapse/streaming.rs` | `StreamingSpikeRunner` / `StreamingSortConfig` / `StreamingSortResult` → `StreamingDetector` / `StreamingDetectionConfig` / `StreamingDetectionResult` |
| B3 sorting | `synapse/sorting.rs` | `cluster_isosplit` → `cluster_kde_merge`; `match_spikes_omp` → `match_spikes_matching_pursuit`; GMM / matcher take a client |
| B4 pipeline | `pipeline/engine.rs`, `math/arithmetic.rs`, `filter/bindings.rs` | `execute::<R, F>`; `SubtractBaseline { baseline }`; `Median { width, edge }`; `TeagerKaiser { edge }` |
| B5 linalg / whitening | `linalg/{pca,ppca,ica}.rs`, `spatial/whitening.rs` | fits take a client |
| B6 synapse-ml | `synapse_ml/*` (1 050 lines) | every bound type is parked (`Kilosort4BasisEmbedder`, `Kilosort4Detector`, `Emusort*`) |
| B7 dsp-io | `buffer/*` | needs `features = ["neuro"]` for NWB |
| B8 paths | `synapse/spatial.rs`, `dsp_synapse::traits` | old alias paths (`traits` now `core::traits`) |

## Findings

### Structure

- **PY1 — The API is flat.** `lib.rs` registers about 80 names in one module; the SDK then rebuilds
  the subpackages (`filter`, `spatial`, `synapse`, …) in Python by re-exporting them. *Proposal:*
  register PyO3 submodules that mirror the Rust crates (`dsp_kitchen.base.filter`,
  `dsp_kitchen.synapse.detection`, `dsp_kitchen.synapse_ml.sorters`, …), so one layer defines the
  layout. **D**
- **PY2 — Aliases.** `pub use buffer as mmap`, the `Myomatrix*` aliases (they name the array
  hardware, not a sorter), `dsp_kitchen.filters` (with a `sys.modules` patch), `DspSession` (marked
  legacy), the `*_layout` probe aliases. The project rule is no aliases. *Proposal:* remove them all.
- **PY3 — Only one way to pick a device.** Native calls use `ComputeTarget::from_env()` (only the
  `DSP_KITCHEN_RUNTIME` variable); the ML classes take a separate `backend: str`. There is no way to
  list runtimes or choose one from Python. *Proposal:* a `dsp_kitchen.runtime` with `available()`,
  `current()`, `set(name)`, plus an optional `runtime=` argument; one rule for every call. **D**
- **PY4 — Everything goes through the host per call.** Each function uploads a NumPy array,
  runs, and downloads (correct, and the GIL is released). This throws away the main design point
  (data stays on the device). *Proposal:* keep the one-shot functions, and add bindings for the
  streaming objects (`PipelineWorkspace`, `StreamingDetector.run_on(source)`), so long recordings
  are processed in Rust from a file path, not from NumPy chunks. **D**

### Semantics

- **PY5 — Sample-rate defaults.** 19 signatures default `fs` / `sample_rate_hz` to `30000.0`
  (and `DspSession` to 384 channels). A wrong default rate gives silently wrong filters and times.
  scipy makes `fs` explicit when it matters. *Proposal:* make the rate required (or take it from a
  recording object).
- **PY6 — µV in names.** `noise_std_uv`, `peak_amplitude_uv`, `channel_sigmas_uv`, `gain_uv`,
  `baseline_uv`, `min_explained_energy` (µV²), and others. The Rust side is now unit-neutral (units are
  data). *Proposal:* drop the unit suffixes, and report units from the recording.
- **PY7 — Defaults that differ from the Rust side and scipy.** `LowpassFilter(cutoff_hz=300.0)` (a
  300 Hz low-pass default makes no sense); `detect_spikes(refractory_samples=30)` has no
  `distance_rule`; filters have no `edge` / `start` (`FilterStart`) / Chebyshev / FIR; and
  `DspSession` hard-codes `Scale(0.195)` + a 60 Hz notch. *Proposal:* mirror the Rust defaults
  (which follow scipy), with no defaults where Rust has none.
- **PY8 — Names that overstate.** `sort_recording` is streaming detection with per-channel
  templates, not sorting; `match_spikes_omp` and `cluster_isosplit` (see B3). *Proposal:* use the
  Rust names (`detect_recording` / `StreamingDetector`, `match_spikes_matching_pursuit`,
  `cluster_kde_merge`).
- **PY9 — The sorter bindings model the parked design.** `EmusortSortConfig(template_samples=150,
  threshold_sigma=6.5, …, min/max_conduction_velocity)`, `preset_32ch_grid`, `from_canonical`,
  `Kilosort4Detector.from_hub(threshold_sigma=…)`, and `z_score_per_channel` (which is ignored:
  `let _ = z_score_per_channel`). *Proposal:* rewrite against `sorters::{kilosort4, emusort}`
  (`Kilosort4Config`, `EmusortConfig`, `learn_universal_templates`, `detect_universal`,
  `ChannelDelayEstimator`) and expose `provenance()` / `citation()`.
- **PY10 — `ModelHub.list` / `info`** return dicts built from the old manifest
  (`ModelStatus`, which no longer exists). *Proposal:* bind behind the `hub` feature, and return the
  manifest plus its provenance.

### Quality

- **PY11 — Unused Rust dependencies:** `memmap2`, `serde`, `thiserror`, `tracing` (no use in `src/`).
- **PY12 — Python dependencies:** `h5py`, `matplotlib` and `polars` are required but unused;
  `python-dotenv` is loaded **at import** (`load_dotenv` changes the environment of the
  calling program). *Proposal:* no runtime dependency other than `numpy`; drop the import-time
  dotenv.
- **PY13 — Workspace-path helpers in the package.** `get_local_path()` walks up from `__file__` to
  the repository, and `load_recording()` defaults to `playground/data/mock_signal_384ch.bin`. Both
  break in an installed wheel. *Proposal:* remove them, or move them to the playground.
- **PY14 — Stub drift.** The `.pyi` lacks the `Myomatrix*` aliases and `__version__`; it is
  hand-written, so it will drift again. *Proposal:* generate it (`pyo3-stub-gen`) or check it in a
  test. **D**
- **PY15 — Tests.**
  - The four sorter tests use the parked APIs, and need NWB files that are not in the repository
    (they skip).
  - `test_emusort_pipeline.py` imports `EMUsortDetector` (wrong case).
  - Results are not checked against scipy or numpy, and Python tests do not run in CI (only
    `docs.yml` exists).

  *Proposal:* scipy / numpy equivalence tests for the filters, `find_peaks`, `correlate` and
  `var`; parked sorter tests; and a CI job (maturin develop + pytest, CPU runtime).
- **PY16 — Fine as is.**
  - `array.rs` borrows contiguous float32 without copying and hands results back without copying.
  - `MmapRecording` gives read-only, zero-copy views with alignment checks and a documented
    `SAFETY` note; the array keeps the mapping alive.
  - `py.detach` releases the GIL in native calls.
  - Errors map to `ValueError` / `RuntimeError`.
  - Keep all of this.

## Proposed order (for the fix-up pass, after dsp-cli / dsp-stream)

1. Decide on **D** items: PY1, PY3, PY4, PY14.
2. Park `synapse_ml/*`, the four sorter tests and the SDK's `synapse/ml.py` (PY9).
3. Fix B1–B8 against the cleaned crates; apply PY2, PY5–PY8, PY11–PY13.
4. Rebind the sorters and the hub (PY9, PY10) with provenance.
5. Stub, tests, CI (PY14, PY15); a "Python" page in the book.

## Recommendation for the open decisions (2026-10-05, not yet approved)

**PY1 — one package whose modules mirror the crates**, built from one private extension:

```text
dsp_kitchen/
├── _native.*.so      the only compiled extension
├── __init__.py       __version__, runtime, subpackages — nothing else
├── runtime.py        available(), current(), set(name)                         ← dsp-core
├── io.py             open(path), Recording, probes, sorting files              ← dsp-io
├── base/             filter, spatial, linalg, peaks, math, pipeline            ← dsp-base
├── synapse/          detection, extraction, features, spatial, sorting, metrics, storage, streaming
├── synapse_ml/       sorters/{kilosort4, emusort} with provenance() / citation()
└── hub.py            Hub, Artifact — optional extra `dsp-kitchen[hub]`        ← dsp-synapse-hub
```

- The names match the Rust crates and the mdBook pages.
- There is one package instead of two (`dsp_kitchen_bindings` plus `dsp_kitchen`).
- Rust registers the PyO3 submodules. Each `.py` file is a one-line re-export, because a PyO3
  submodule cannot be imported as `import a.b`; there is no `sys.modules` patching.
- The binding sources mirror the layout: `src/{core, io, base/, synapse/, synapse_ml/, hub}.rs`.
- There are no aliases and no flat top level, and `fs` is a required argument.

**PY3:** the `runtime` module, plus an optional `runtime=` argument on calls; it replaces the `backend` strings.

**PY4:** keep the one-shot NumPy functions for exploring data, and add `PipelineWorkspace` and
`StreamingDetector.run(recording)` so long recordings run in Rust from a file.

**PY14:** generate the stubs with `pyo3-stub-gen`, and add a test that fails when they differ from
the committed files.

The plan once approved:
1. Park the old SDK and the sorter bindings.
2. Rebuild against dsp-core, dsp-io and dsp-base.
3. Then synapse and synapse-ml.
