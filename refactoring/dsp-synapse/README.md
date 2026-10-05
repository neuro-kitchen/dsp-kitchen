# refactoring/dsp-synapse

Changes to `crates/dsp-synapse` (review findings in `REVIEW.md`).

- Branch / commit at start: `versions/v0.14` @ `0f7e600`
- Build state: **intentionally broken** until the final fix-up pass (dsp-synapse is not compiled
  during the cleanup; only core / io / base are checked).

## To do when the synapse cleanup ends (user, 2026-10-05)

1. Root `README.md`: a warning **at the top** — the code is under development; fixes to
   dsp-synapse-ml, dsp-stream and dsp-app are coming soon.
2. mdBook: highlight (a) CubeCL — one code base runs on any GPU (CUDA / ROCm / Metal / Vulkan via
   wgpu) or the CPU, chosen at run time (`dsp_core::compute`); (b) how the framework is built to
   cut data transport — data stays on the device between stages (pipelines, device candidate
   compaction, only results downloaded), out-of-core reading with prefetch, bounded host RAM.

## Moved to other crates (not parked)

| Original path | New path | When | Reason |
|---|---|---|---|
| `src/storage/npy.rs` | `crates/dsp-io/src/container/npy/mod.rs` | 2026-10-05 (dsp-io phase 2) | `.npy` is a storage container; one copy for the workspace. |
| `src/storage/zarr_store.rs` | `crates/dsp-io/src/container/zarr/mod.rs` | 2026-10-05 (dsp-io phase 2) | Zarr helpers are a storage container. |
| `zarr_store::infer_nwb_sample_rate` | `crates/dsp-io/src/neuro/nwb/units.rs` | same | NWB-specific. |
| `src/probe/neighbors.rs` (`find_k_nearest_neighbors`, `precompute_knn_table`) | `crates/dsp-io/src/neuro/probe/neighbors.rs` | 2026-10-05 (synapse step 1a) | Geometry queries live with `SensorLayout`. |
| `src/probe/{neuropixels, hdemg_grid, tetrode, utah}.rs` | `crates/dsp-io/src/neuro/probe/presets.rs` (rewritten, same geometry; spacings named) | same | All presets in one place; NP 1.0 moved there from `SensorLayout::neuropixels_1_0_standard` (method removed). |
| dependency `dsp-stream` (for `PrefetchReader`) | dependency `dsp-io` (`zarr`, `neuro`); `dsp_io::PrefetchReader` | 2026-10-05 | `PrefetchReader` is local I/O. |

## Removed (step 1a)

| File | Removed |
|---|---|
| `src/lib.rs` | `pub mod probe;` and `pub use probe::{…}`; aliases `pub use core::{bands, traits}`, `metrics::correlogram`, `sorting as clustering`, `sorting as matching`, `spatial as localization`, `spatial as motion`; the merged `pub mod kernels { … }`. |
| `src/probe/` | whole module (moved, see above). |

## Imports repointed (step 1a)

`dsp_core::{SensorLayout, layout::*, Position3D, SensorSite}`, `ProbeLayout` (alias, gone) and
`crate::probe::*` → `dsp_io::neuro::probe::*` in: `core/{traits, sorting_output}.rs`,
`detection/dedup.rs`, `extraction/extractor.rs`, `spatial/{center_of_mass, dipole, grid_convolution,
kriging, monopolar}.rs`, `storage/{nwb_units, phy, phy_sorting, zarr_analyzer}.rs`,
`streaming/{runner, kernels/mod}.rs`, `tests/{extraction_alignment, streaming_invariance}.rs`
(`dsp_synapse::kernels::…` → `dsp_synapse::extraction::…`).

## Step 1b — storage split: Phy (2026-10-05)

Files → dsp-io (`neuro/phy/`, `neuro/templates.rs`); conversion to / from `SortingOutput` stays here.

| Before | After |
|---|---|
| `storage/phy_sorting.rs` `PhySorting` (read, `save_curation`, conversions, similarity on load) | dsp-io `PhyFolder` (`read`, `write`, `write_curation`, `is_phy_folder`; stores values as found, computes nothing) + synapse `storage/phy.rs` (`from_sorting_output`, `to_sorting_output`, `fill_similarity`) — **file deleted** |
| `storage/phy.rs` `save_phy_folder` (second, separate Phy writer) | `from_sorting_output(so).write(dir)`: one writer. `spike_positions.npy` is now `[N, 2]` (as Kilosort 4), was `[N, 3]`; `cluster_KSLabel.tsv` now written |
| `PhyParams`, `ClusterTables`, `resolve_dat_path`, `ClusterId` | dsp-io `neuro::phy` (`PhyParams::write`, `ClusterTables::write` added) |
| `core/template.rs` `DenseTemplates`, `TemplateAxisOrder` (+ `from_sparse`, `from_channels_samples`, `trace`, `peak_to_peak`, `best_channel`, `top_channels`, `amplitude`) | dsp-io `neuro::templates` |
| `DenseTemplates::pack_units` / `unpack_unit` / `waveform` | synapse free fns `pack_templates` / `unpack_template` / `dense_waveform` (`core/template.rs`) |
| alias `PhyTemplates` | removed |
| `storage/mod.rs` | `zarr_store` now `dsp_io::container::zarr` (until format detection moves); Phy detection via `PhyFolder::is_phy_folder` |
| `lib.rs` exports `ClusterTables, PhyParams, PhySorting, PhyTemplates, resolve_dat_path, DenseTemplates, TemplateAxisOrder` | removed; added `fill_similarity, dense_waveform, pack_templates, unpack_template` |

Known, left for later steps: `core/template.rs` imports `dsp_base::resampler::minmax::peak_to_peak`
(parked in dsp-base) — step 3; `to_sorting_output` `total_samples = last spike` (PS1) — step 2.

## Step 1b — storage split: NWB units, `.sorting.zarr`, detection (2026-10-05)

| Before | After |
|---|---|
| `storage/nwb_units.rs` (zarr reads / writes + conversion) | dsp-io `neuro::nwb::NwbUnitsTable` (`read(dir, Option<rate>)`, `write`, `is_units_table`, `spikes_of`) + synapse `storage/nwb_units.rs` (`from_sorting_output`, `to_sorting_output`, `save_nwb_units`, `load_nwb_units`). `firing_rate` column now also read (unused by the conversion). |
| `storage/zarr_analyzer.rs` | renamed **`storage/sorting_zarr.rs`** (conversion only) + dsp-io `neuro::sorting_zarr::{SortingZarr, SortingZarrManifest, SortingZarrUnit, SORTING_ZARR_FORMAT}`. Documented as dsp-kitchen's **own** format (not SpikeInterface). Manifest keeps `recording_meta` / `drift` as JSON and `quality_label` as text (same bytes on disk as before). |
| `SortingFormat { Phy, ZarrAnalyzer, NwbUnits }` (synapse) and detection inside `load_sorting` / `save_sorting` | dsp-io `neuro::{SortingFormat { Phy, SortingZarr, NwbUnits }, detect_sorting}`, `SortingFormat::from_path_name`. Synapse `load_sorting` / `save_sorting` only dispatch. NWB detection now needs `spike_times_index` (was: `/units/spike_times` alone). |
| `lib.rs` export `SortingFormat` | removed (import from `dsp_io::neuro`). |

## Step 2 — bug fixes (2026-10-05; none run yet except the dsp-base Cholesky tests)

| # | Review | File(s) | Fix |
|---|---|---|---|
| 1 | G1 | `sorting/gmm.rs` | Covariance inverse / log-det by **Cholesky** (new dsp-base `linalg::{cholesky, spd_inverse_logdet}`), was 80-rotation `SymmetricEig` (parked). Not positive definite → diagonal jitter `reg·10^i` (≤ 6 tries, `CHOLESKY_JITTER_ATTEMPTS`), then diagonal-only fallback. |
| 2 | SP1 | `core/snippets.rs`, `extraction/extractor.rs`, `spatial/{dipole, kriging}.rs` | `SnippetBatch` gains **`peak_index`** (= `pre_samples`; `new(ch, samples, peak_index)`, `from_raw_parts(…, num_samples, peak_index, …)`); `DipoleLocalizer` reads the potential there (was `num_samples / 2`). |
| 3 | DD2 | `detection/dedup.rs`, `detection/kernels/dedup.rs` | Stronger crossing = larger **`|amplitude|`** (host `stronger()`, was `deeper()`; device kernel gets magnitudes, compares `>`). Same result for negative troughs. Test added. |
| 4 | D2 | `detection/{neo, matched_filter}.rs` | Refractory state per channel (`Option<usize>`), was `events.is_empty()` over all channels. |
| 5 | DR1 | `spatial/drift.rs` | Reference = **first time bin with spikes**; bins without spikes hold the previous drift (were registered against nothing → `−max_drift`); no spikes → zero drift. Applies to non-rigid blocks too. Test added. Single-reference design (no iterative template) left for later. |
| 6 | F4 | `features/morphology.rs`, `core/snippets.rs` | Measures the primary channel only (new `WaveformSnippet::primary_trace`). Test added. |
| 7 | SG2 | `storage/phy.rs`; dsp-io `PhyFolder::recording_samples` | Recording length from the `params.py` binary (`(size − offset) / (n_channels_dat · dtype)`), else last spike **+ 1** (was last spike). |
| 8a | I2, I3 | `sorting/isosplit.rs` | Dead `ca[0.min(0)]` projection removed; after a boundary re-cut that moves points the pass restarts (fresh centroids); passes stop when nothing changed. |
| 8b | K1 | `spatial/kriging.rs` | Hand-written Gauss–Jordan (skipped tiny pivots) removed; **`cholesky_solve`** (new, dsp-base). `compute_kriging_weight_matrix` now returns `DspResult<Vec<f32>>` (error when not positive definite). |

New in dsp-base `linalg/cholesky.rs`: `cholesky`, `spd_inverse_logdet`, `cholesky_solve` (host, f64,
3 tests passing).

Still open from the review (later steps): NEO / matched filter remain negative-only and greedy
(D1); drift correlation unnormalized (DR2); morphology assumes a negative trough and calls it
`peak_amplitude_uv` (F4, naming step); IsoSplit name (I1).

## Step 3a — peak finding moved to dsp-base (2026-10-05)

New dsp-base module `peaks/` (scipy `find_peaks` semantics):
- `find_peaks(x, polarity, &PeakOptions)` (host, generic `num_traits::Float`): `height`,
  `threshold`, `distance` (+ `distance_rule`), `prominence` (+ `wlen`), `width` (+ `rel_height`);
  flat peaks → middle sample; `Polarity { Positive, Negative, Both }`.
- `local_extrema`, `select_by_distance(peaks, priority, distance, rule)`.
- `DistanceRule { Scipy (default in dsp-base), LocallyExclusive }` — user asked to keep both.
- `find_peak_candidates::<R, F>` (device, generic `F`, autotuned block length): synapse's
  count → scan → write compaction, now both polarities and a height per channel; flat peaks →
  first sample.

dsp-synapse detection rewired (the 5 copies of the peak loop are gone):

| Before | After |
|---|---|
| `detection/kernels/threshold.rs` (kernels + dispatcher + `DetectionCarry`) | **deleted**; `detection/device.rs` `execute_detect_spikes_in_vram(client, trace, heights, channels, samples, emit, global_offset, polarity, spacing)` over dsp-base candidates. Scans `emit ± distance`; no carry. |
| `refractory_samples: usize` arguments; greedy "first crossing wins" | `SpikeSpacing { refractory_samples, rule }` (`detection/spacing.rs`; min distance = refractory + 1); default rule **`LocallyExclusive`** (largest wins; streaming = whole recording). Every detector struct gains `distance_rule`. |
| `detect_spikes_multichannel{,_polarity}`, `detect_spikes_with_sigma{,_polarity}` | `detect_spikes_multichannel(data, ch, samples, factor, polarity, spacing)`, `detect_spikes_with_sigma(…, sigmas, factor, polarity, spacing)`; new `detection_heights(sigmas, factor)` (σ ≤ 0 → `+∞`). |
| NEO / matched filter / adaptive loops | candidates from `local_extrema`, detector score as priority (NEO energy, filter score, |x|), spaced by `SpikeSpacing`. Threshold now inclusive (`≥`, as scipy `height`). |
| `StreamingSortConfig` (negative only, greedy + carry) | adds `polarity` (default `Negative`) and `distance_rule` (default `LocallyExclusive`); runner uploads `detection_heights`. |
| `crate::traits::*` (removed alias) in `detection/{neo, matched_filter}.rs`, `spatial/{center_of_mass, monopolar, grid_convolution}.rs` | `crate::core::*` |

Behaviour change (intended, review D1): within the refractory period the **larger** crossing wins
(was: the first). Tests updated: `streaming/kernels` dense-crossings test, `tests/streaming_invariance`
burst check (only the deepest trough of a burst stays); new tests in `threshold.rs`, dsp-base
`peaks` (scipy reference values, both distance rules, device = host).

## Step 3b — remaining primitives moved to dsp-base (2026-10-05)

| Synapse before | dsp-base now | Synapse after |
|---|---|---|
| 3 lag cross-correlation loops (`sorting/similarity.rs`, `features/conduction.rs`, `spatial/drift.rs`) + `extraction::parabolic_subsample_offset` | `math::xcorr::{lagged_dot, cross_correlation, peak_lag, LagPeak, parabolic_vertex_offset}` | loops replaced; `parabolic_subsample_offset` **removed** (callers: `parabolic_vertex_offset`; flat tolerance named, 1e-12 instead of 1e-6). |
| 3 windowed-sinc interpolators: `alignment::{interpolate_window, sinc_tap}`, `resample_sinc_1d` / `resample_sinc_multichannel` (edge-renormalized, unused), device kernel with inline Blackman-Harris literals | `resampler::fractional::{fractional_delay, fractional_delay_taps, fractional_delay_taps_device, windowed_sinc_weight}`; Blackman-Harris coefficients now named `pub` consts in `math::windows` | `alignment.rs` keeps only `SINC_KERNEL_RADIUS`; the four functions **removed**. Device extraction = `trough_shift_kernel` (per spike) → dsp-base taps (once per spike, was per output sample) → `extract_snippets_kernel`; generic `F` (`execute_extract_sinc_in_vram::<R, F>`), dsp-base `buffer` helpers. The `|shift| ≤ 1e-4 → copy` shortcut (host and device) removed: taps at shift 0 are the identity. |
| 4 window averages (`compute_mean_template` two-pass f32 /n, STA Welford /(n−1), `TemplateAccumulator` Welford + Chan) | `math::moments::RunningMoments` (f64 Welford push, Chan merge, `variance(ddof)` / `std(ddof)`) | all three host versions use it; conventions named: `TEMPLATE_STD_DDOF = 0` (templates, as Phy / SpikeInterface), `STA_STD_DDOF = 1` (STA, feeds the SE). The device reducer (`template_reduce`) is unchanged and merges into it. |
| `spatial::center_of_mass::waveform_peak_to_peak` + broken import of parked `resampler::minmax::peak_to_peak` | `math::stats::peak_to_peak` | copy **removed**, all localizers and `core/template.rs` use dsp-base. |

dsp-base template subtraction (`filter/template/subtraction.rs`) keeps its own lag search: it
searches full-overlap positions only, a different operation.

## Downstream breakage (to fix at the end)

- Removed extraction / spatial helpers: dsp-synapse-ml `examples/emusort_nwb_zarr.rs`
  (`waveform_peak_to_peak` → `dsp_base::math::peak_to_peak`); any user of
  `resample_sinc_1d`, `interpolate_window`, `parabolic_subsample_offset`,
  `extract_sinc_snippets_kernel` (none other found); `execute_extract_sinc_in_vram::<R>` →
  `::<R, f32>`.

- Detection API (`SpikeSpacing`, new signatures, `DetectionCarry` gone, `execute_detect_spikes_in_vram`
  arguments): dsp-cli `commands/benchmark.rs`, dsp_kitchen_py `synapse/detection.rs`,
  dsp-app `engine/data/events.rs`.

- `SnippetBatch` literal / `new` / `from_raw_parts` need `peak_index`: dsp-synapse-ml
  `models/{kilosort4/basis, emusort/basis}.rs`, `models/dartsort/denoiser.rs`.
- `compute_kriging_weight_matrix` returns `DspResult`: any external caller (none found).

- `SortingFormat` now `dsp_io::neuro::SortingFormat`; `ZarrAnalyzer` → `SortingZarr`:
  dsp_kitchen_py `synapse/storage.rs`.
- `dsp_synapse::storage::zarr_store::has_array`: dsp-app `viewmodels/curation.rs` (use
  `dsp_io::neuro::detect_sorting` / `PhyFolder::is_phy_folder` / `NwbUnitsTable::is_units_table`).

- dsp-app `engine/curation/{mod, derived}.rs`: `PhySorting` → `dsp_io::neuro::phy::PhyFolder`,
  `PhySorting::save_curation` → `PhyFolder::write_curation`, `ClusterId` from dsp-io;
  `load_spikes` still from dsp-synapse. dsp_kitchen_py `synapse/storage.rs`.

- Probe functions now from `dsp_io::neuro::probe`: dsp-cli (`commands/benchmark.rs`,
  `commands/probe.rs`), dsp-synapse-ml (`models/emusort/mod.rs`, `models/kilosort4/mod.rs`,
  `examples/emusort_nwb_zarr.rs`), dsp_kitchen_py (`synapse/probe.rs`, 17 sites).
- Alias paths: dsp_kitchen_py `synapse/spatial.rs` (`dsp_synapse::{localization|motion|…}`).
