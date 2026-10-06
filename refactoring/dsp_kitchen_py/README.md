# dsp_kitchen_py — changes

Review: `REVIEW.md`. User decisions (2026-10-06): **keep the SDK style** (PY1: `dsp_kitchen`
with `filter` / `spatial` / `math` / `linalg` / `pipeline` / `io` / `synapse` subpackages, stage
objects + functional forms); PY3 / PY4 / PY14 as recommended (runtime module + `runtime=`,
streaming objects bound, generated stubs).

## Step 1 — parked (2026-10-06)

| Parked | Why | Replacement |
|---|---|---|
| `src/synapse_ml/{kilosort4, emusort, hub, mod}.rs` → `synapse_ml/` | Bind the parked Kilosort4 / EMUsort design (`Kilosort4BasisEmbedder`, `Kilosort4Detector`, `Emusort*`, `Myomatrix*` aliases, `EmusortSortConfig` with invented defaults, ignored `z_score_per_channel`) and the old hub (`ModelStatus`). | Step 3: `sorters.kilosort4` / `sorters.emusort` on `dsp_synapse_ml::sorters`, hub on `ModelHub`, with provenance. |
| `tests/test_{emusort_bindings, emusort_phy, emusort_pipeline, kilosort4_phy}.py` → `tests/` | Test the parked API; need NWB files not in the repository (always skipped); one imports a misspelled name. | Step 4. |
| `dsp_kitchen/synapse/ml.py` → `sdk/synapse/ml.py` | Re-exports the parked classes and uppercase `EMUsort*` aliases. | Step 3. |

## Steps 2–4 — rebuilt (2026-10-06)

`cargo check -p dsp_kitchen_py` (default `wgpu` + `hub`, and `--no-default-features --features wgpu`)
and `clippy` pass with no warnings in the bindings. **Not built as an extension nor run** (needs
`maturin`; end pass).

### Structure (PY1: SDK style kept)
- Native module flat; each Rust module has `register(m)`; Rust files mirror the SDK areas
  (`filter/{iir, fir, non_linear, template}.rs`, `linalg/mod.rs`, `synapse/{…, ml}.rs`).
- SDK: subpackages import straight from `dsp_kitchen_bindings` (`_bindings.py`, a third copy of
  every name, removed); top level = subpackages + `runtime` + `__version__` (no flat duplicates).

### By finding
| Finding | Change |
|---|---|
| B1–B8 | All upstream breakage fixed (probes from `dsp_io::neuro::probe`, streaming, sorting renames, pipeline stages with edges, fits with a client, paths). |
| PY2 | Removed: `pub use buffer as mmap`, `Myomatrix*` / `EMUsort*` aliases, `filters` + `sys.modules` patch, `DspSession`, `*_layout` probe aliases, `num_channels` (= `total_channels`), direction / order / format / method string aliases, `export_to_phy` / `read_kilosort` / `save_nwb_units` (= `save_sorting` / `load_sorting`), `mean_accuracy`. |
| PY3 | `runtime.rs`: `dk.runtime.available()`, `current()`, `set(name / None)`; every device call takes `runtime=`; one rule (argument, `set`, `DSP_KITCHEN_RUNTIME`, first compiled-in). |
| PY4 | One-shot NumPy functions kept; `detect_recording(Recording, Pipeline, ProbeLayout, …)` runs a whole recording in Rust, out of core, on the device. `PipelineWorkspace` for chunk-by-chunk Python streaming is **not** bound (generic over the runtime; needs a runtime-erased wrapper) — open. |
| PY5 | `fs` required (keyword) wherever time matters; filters need it only when a filter stage is present (`RATE_NOT_USED_HZ` otherwise). |
| PY6 | No `_uv` in Python names (`amplitude`, `noise_sigmas`, `peak_to_peak`, `gain` / `offset` for raw files documented as the format's µV); `Recording.units` reports each channel's unit. |
| PY7 | Defaults only where Rust has them, resolved from the Rust items: `DEFAULT_BUTTERWORTH_ORDER`, `FilterMode` / `FilterStart` defaults, `*_DEFAULT_EDGE`, `MEDIAN9_RADIUS`, `DEFAULT_MAX_LAG`, `StreamingDetectionConfig::default()`, `GmmClusterer::default()`, k-means and matching-pursuit `DEFAULT_*`, `Kilosort4Config` / `EmusortConfig` defaults, scikit-learn's FastICA defaults; new `dsp_synapse::metrics::DEFAULT_*` (SpikeInterface: ISI 1.5 ms, presence 60 s, correlograms 1 / 50 ms, comparison 0.4 ms / 0.5) also used by `QualityCriteria::default()`. Required now: `Scale.alpha`, `Clamp` bounds, notch `q`, cutoffs, whitening `epsilon`, Laplacian `k_neighbors`, HD-EMG pitch, extraction windows, dedup radius / window, drift binning, kriging scales, density-peaks / KDE-merge / HDBSCAN / CBSS parameters, rate / PSTH / STA / MEP windows, `total_samples` / `duration_sec` where rates depend on it. New: Chebyshev I, `start=`, `edge=`, median `width`, Gaussian smoothing (`dk.filter.fir`, was empty). |
| PY8 | `NwbZarrRecording` → `Recording` (any format; `source=`, `list_sources`); `sort_recording` → `detect_recording` / `StreamingDetectionResult`; `cluster_isosplit` → `cluster_kde_merge`; `match_spikes_omp` → `match_spikes_matching_pursuit`; `median_filter_9p` → `median_filter(width)`; `quantify_mep` (built a fake STA, ignored its baseline) → `compute_mep(data, triggers, …)`. |
| PY9 | `synapse/ml.rs`: `Kilosort4Config` / `EmusortConfig` (upstream names, Rust defaults, unknown keywords refused), `UniversalTemplates` (`from_npz`, `from_hub`), `extract_clips`, `learn_universal_templates` (device), `TemplateCentres`, `detect_universal` (device), `ChannelDelayEstimator`, `apply_channel_delays`, `Provenance` (`citation`, `to_dict`), `kilosort4_provenance` / `emusort_provenance`. SDK: `dk.synapse.ml.kilosort4` / `.emusort` (`Config`, `provenance`). |
| PY10 | `ModelHub` (feature `hub`): `list`, `info` (with citation), `provenance(id)`, `pull`, `verify`, `remove`, `clean`, `cache_dir`. |
| PY11 | Rust deps: `memmap2`, `serde`, `thiserror`, `tracing` dropped; `dsp-io` `neuro`; `dsp-synapse-ml` without Burn backends; features `wgpu` (default), `cpu`, `cuda`, `hip`, `hub` (default). |
| PY12 | Python deps: only `numpy` (h5py, matplotlib, polars, python-dotenv dropped); no import-time dotenv. |
| PY13 | `get_local_path`, `resolve_data_path`, `open_nwb_zarr`, `load_recording` (repository paths) removed. |
| PY14 | Hand-written stub parked (`stub/`, it described the removed API). **Open:** generate with `pyo3-stub-gen` (not in the cargo cache; needs a fetch and per-item annotations). |
| PY15 | `tests/test_sdk.py` rewritten against NumPy (exact references for scale, CAR, TKEO, median, PCA variances, channel delays) and scipy (`butter` + `sosfiltfilt` / `sosfilt`, skipped without scipy); sorting round trips, sorter provenance and defaults, runtime selection; `test_bindings_soundness.py` updated (keyword `fs`, missing-`fs` errors). Not run. **Open:** CI job (maturin develop + pytest on the CPU runtime). |

Fabrications removed on the way: 100 µV default drift weights, 50 µV default amplitudes, channel
0 for spikes without a channel, last-spike-based `total_samples`, `0..1000` depth fallbacks,
`std = 1` templates (now NaN, unused by matching), zero-SD MEP input, silently empty `ipt`,
0-valued means over no matches (now NaN).

## 2026-10-06 — sorter bindings follow the merged runner

`run_emusort` returns `Kilosort4Result` (new getters `sorter`, `channel_delays`,
`sample_rate_hz`, `total_samples`); `EmusortResult` and the `run_front_end` shim are gone (also
from `kilosort4.py`, with the `FrontEndResult` alias). `to_sorting_output(probe=None)`: no
`sample_rate_hz` / `total_samples` defaults. `ChannelDelayEstimator.add_batch(batch, pad, *,
runtime=None)` and `apply_channel_delays(batch, delays, *, runtime=None)` run on the device.
`tests/test_sdk.py::test_channel_delays_are_recovered` updated (pad 25, checks the shift).
Also: `Recording.from_array(data, fs, *, name="array")` (in-memory `MemoryRecording`, for tests
and derived signals); `kilosort4.run(..., templates=None)` passes predefined universal
templates; `Kilosort4Runner` is gone on the Rust side (`Kilosort4::run`). Playground:
`sorters/00_sorters_synthetic.py` (no data) checks delays, Kilosort4, EMUsort, every runtime and
export; the two data scripts export `to_sorting_output(probe)` and the Kilosort4 one also runs
with the predefined templates.

**Later 2026-10-06:** `ChannelDelayEstimator` (one cross-correlation download per batch) replaced
by `estimate_channel_delays(batches, *, pad, max_lag, runtime=None)` (all batches, one download).
`kilosort4.run` / `emusort.run` take `preprocessing_from=` (reuse a result's fitted
preprocessing). Playground: the Kilosort4 script's predefined-templates run reuses the first
run's preprocessing; the synthetic script no longer re-runs Kilosort4 on the default runtime.
`synapse.hdbscan(features, min_cluster_size, *, runtime=None)` and `synapse.kmeans(..., runtime=None)`
run on the device; `kmeans` checks `1 ≤ k ≤ n` (was a panic).
