# refactoring/dsp-synapse-ml

Items removed from `crates/dsp-synapse-ml`.

- Branch / commit at removal: `versions/v0.14` @ `0f7e600`
- Build state: **intentionally broken** until the final fix-up pass.

## Parked files

| Parked path | Original path | Removed on | Reason | Replacement |
|---|---|---|---|---|
| `hub/npy.rs` (`NpyTensorF32`) | `crates/dsp-synapse-ml/src/hub/npy.rs` | 2026-10-05 (dsp-io phase 2) | Second `.npy` reader (f32 only, `anyhow` errors). Duplicates the complete reader now in `dsp-io`. | `dsp_io::container::npy::read_npy::<f32>(path)` → `NpyArray { data, shape }` (C order, any dtype converted, Fortran order handled). |

## Lines removed from files that stayed

| File | Removed |
|---|---|
| `src/hub/mod.rs` | `pub mod npy;`, `pub use npy::NpyTensorF32;` |
| `src/lib.rs` | `NpyTensorF32` from `pub use hub::{…}` |

## Step 1 — catalog, provenance, synthetic models (2026-10-05)

Scope today: Kilosort4 and EMUsort only (the ones that can be validated).

| Change | Detail |
|---|---|
| `catalog/models.json` | **Parked** the old catalog in `refactoring/dsp-synapse-ml/catalog/models.json` (placeholder hashes, nonexistent URLs, wrong DOI; see `REVIEW.md`). New catalog (schema 2) holds one verified entry, `kilosort4/wtemp-v1`: Kilosort4's `wTEMP.npz` from OSF (3432 B, SHA-256 `cae1c96f…abd8`, arrays `wPCA` / `wTEMP` 6 × 61). |
| `src/provenance.rs` (new) | `Provenance { name, kind, paper: Option<Paper>, code: UpstreamCode, artifacts, notes }`, `Paper { title, authors, venue, year, doi, license }`, `UpstreamCode { url, license, version }`, `ArtifactSource { name, url, sha256, size_bytes, license }`, `ProvenanceKind { ReimplementedFromPaper, UpstreamWeights }`, trait `Attributed`, `Provenance::citation()`. Exported at the crate root. |
| `hub/manifest.rs` `ModelManifest` | `version`, `upstream_repo`, `paper_url`, `license` removed → `provenance: Provenance`; `io_spec` now `Option`; new `arrays: Vec<ArraySpec>`, `sample_rate_hz: Option<f64>`; `ModelFormat::Npz`. |
| `hub/downloader.rs` `download_and_verify` | takes `expected_size`; **refuses an empty SHA-256** (was: skipped verification) and checks the size (HUB1, HUB2). |
| `hub/mod.rs` tests | catalog test now requires a 64-hex hash, a size and a paper DOI per entry, and checks the Kilosort4 entry; local pull test uses schema 2. |
| `models/emusort/` | **Removed** `EmusortBasisEmbedder::from_canonical`, `EmusortTemplateMatcher::from_canonical` (hand-made shapes presented as pretrained, MOD1), the test built on them, and the made-up ids `EMUSORT_{BASIS,TEMPLATES}_MODEL_ID` / `MYOMATRIX_*_MODEL_ID`. `from_npy` stays. |
| `models/{dartsort, spikenet2, unitrefine}` | only `io_spec: manifest.io_spec` (now optional); their catalog entries are parked, so `from_hub` reports an unknown id until they are validated. |

## Step 2 — `dsp-synapse-hub` (2026-10-05)

New crate `crates/dsp-synapse-hub` (workspace member): `providers.rs` and `download.rs` (moved
from `src/hub/{providers/mod.rs, downloader.rs}`), `cache.rs` (rewritten: generic `Artifact`,
status from `index.json` + file size instead of re-hashing, `verify` re-hashes), `lib.rs`
(`Artifact`, `Hub { pull, status, verify, remove, clean }`, `ArtifactReport`, `ArtifactStatus`).
Network dependencies (`reqwest`, `sha2`, `hex`, `dirs`) only there. `cargo check` passes.

dsp-synapse-ml: `src/hub/` keeps `catalog.rs`, `manifest.rs` (`ModelManifest::{file_name,
artifact}`), `safetensors.rs`; new `model_hub.rs` (`ModelHub`, `ModelHubEntry`) behind feature
**`hub`**; `runtime::pull_model` and every model's `from_hub` behind it. Catalog and safetensors
now return `DspResult` (was `anyhow`). Parked: `hub/registry.rs` (`ProbePreset`,
`ModelPresetConfig`, unused), `hub/pytorch_remap.rs` (unused). Removed: `ModelStatus`,
`ModelVerifyReport`, `HubCache` / downloader / providers re-exports.

## Step 3 — sorters (2026-10-05)

New `src/sorters/`:
- `kilosort4/` — `Kilosort4Config` (upstream defaults), `kilosort4_provenance`, `Kilosort4`
  (`Attributed`); `templates.rs` (`UniversalTemplates::from_npz` for `wTEMP.npz`,
  `extract_clips`, `learn_universal_templates`: SVD via device eigensolver, k-means, optional
  HDBSCAN); `detect.rs` + `kernels.rs` (`TemplateCentres`, `detect_universal` on the device:
  correlation, centre responses, neighbour max, local peak score, device compaction, features).
- `emusort/` — `EmusortConfig` (paper defaults over `Kilosort4Config`), `emusort_provenance`,
  `Emusort`; `ChannelDelayEstimator`, `apply_channel_delays`.
New in dsp-synapse: `sorting/kmeans.rs` (sklearn semantics), `sorting/hdbscan.rs` (sklearn
defaults, O(n²·d)). New in dsp-io: `container/npy/npz.rs` (`read_npz`, `read_npz_entries`),
`read_npy_bytes`.
Verified in a scratch crate (the workspace crates do not compile yet): kernels and detection
compile against real cubecl / dsp-core / dsp-base; 6 unit tests pass (k-means, HDBSCAN, clip
isolation, channel delays, EMUsort defaults, citation).
Parked: `models/kilosort4/`, `models/emusort/` (incl. `latency.rs`, `config.rs`),
`runtime/kernels/` (basis projection, universal-template energy filter), `examples/
emusort_nwb_zarr.rs`, `build.rs`.
Not implemented (listed in the sorter docs): Kilosort4 preprocessing driver, drift correction,
clustering, learned-template deconvolution, merging.

## Step 4 — runtime (deferred)

`burn-onnx` replacement deferred: it serves only the ONNX models (DARTsort VAE, SpikeNet2,
UnitRefine), none of which has a validated artifact (catalog entries parked). The runtime doc
states the plan.

## Step 5 — cleanup (2026-10-05)

Removed: module alias `myomatrix`, all `Myomatrix*` / `Kilosort4Detector` / `EmusortDetector`
aliases (with their modules), `runtime::default_compute_target` (models now take a
`ComputeTarget`, no `Option` fallback), re-exports of dsp-core compute types, `DspError::Model`
(→ `InvalidConfig` / `ComputeError`), dependencies `memmap2`, `thiserror`, `tracing`, `sha2`,
`hex`, `dirs`, `reqwest`; `anyhow` only with feature `hub`. dsp-io is now a normal dependency
(feature `neuro`).

## Downstream breakage (to fix at the end)

- Steps 2–5: dsp-cli `commands/hub.rs` (use `dsp_synapse_hub::Hub` + `dsp_synapse_ml::ModelHub`
  with feature `hub`; `ModelStatus` → `ArtifactStatus`), `commands/benchmark.rs`;
  dsp_kitchen_py `synapse_ml/{kilosort4, emusort, hub, mod}.rs` (old KS4 / EMUsort types →
  `sorters::{kilosort4, emusort}`), `lib.rs`; model constructors take `ComputeTarget` (not
  `Option`).

- Step 1: `from_canonical` callers — `examples/emusort_nwb_zarr.rs`, dsp_kitchen_py
  `synapse_ml/emusort.rs`; removed `EMUSORT_* / MYOMATRIX_*_MODEL_ID` exports; `ModelManifest`
  fields (`version`, `upstream_repo`, `paper_url`, `license`) in dsp-cli `commands/hub.rs` and
  dsp_kitchen_py `synapse_ml/hub.rs`; `download_and_verify` gains `expected_size`.

`NpyTensorF32::from_file` callers (all only use `from_file`; `from_bytes` had no external users):
- `src/models/kilosort4/basis.rs`, `src/models/kilosort4/matcher.rs`
- `src/models/emusort/basis.rs`, `src/models/emusort/matcher.rs`

`dsp-synapse-ml` already depends on `dsp-io`.

## 2026-10-06

- `models/dartsort/denoiser.rs`: `SnippetBatch` keeps the input's `peak_index`.
- Built without default features (as dsp-cli does, `hub` only), `runtime/burn_engine.rs` warns
  about unused code (`linear_on_backend`, `conv1d_on_backend`, unused arguments): gate them on
  the backend features when the runtime is reworked (burn-onnx, step 4).

## 2026-10-06 — front end driver

`sorters/kilosort4/frontend.rs`: `run_front_end(client, source, probe, &FrontEndOptions)` over a
whole recording on dsp-core `ChunkSchedule` (batches = windows of `batch_size`, halos = filter
settling + `nt` + max channel delay), dsp-io `PrefetchReader` and dsp-base `PipelineWorkspace`:
whitening from the covariance averaged over every `nskip`-th window, EMUsort delays, clips and
templates, detection on every window (spikes kept in each window's valid range, recording
samples). Without delays the windows stay on the device for detection; with delays they make a
host round trip (a device shift kernel is open). `FrontEndOptions::{kilosort4, emusort}`.
Named: `HIGHPASS_ORDER = 3`, `WHITENING_EPSILON = 1e-6` (both to verify against upstream); the
common reference is a mean (upstream may use a median). dsp-base gained
`SpatialWhitening::local_knn_from_covariance` (`fit_local_knn` now uses it). Python:
`kilosort4.run_front_end`, `FrontEndResult`. The playground's Python batching helper is gone.


## 2026-10-06 — one Kilosort4-family runner, delays on the device

Replaces the duplicated `kilosort4/runner.rs` / `emusort/runner.rs` (from `97eacad`).

- `kilosort4/runner.rs`: `RunPlan` (`kilosort4(&cfg)`, `emusort(&cfg, fs)` in `emusort/mod.rs`),
  `fit_preprocessing` → `FittedPreprocessing`, `run_plan` → `Kilosort4Result` (one result type,
  `channel_delays: Option<ChannelDelays>`, `sorter`, `sample_rate_hz`, `total_samples`,
  `centre_positions`). `Kilosort4::run` and `Emusort::run` call `run_plan` (`Kilosort4Runner` removed);
  `fit_kilosort4_preprocessing` kept. `EmusortRunner` / `EmusortResult` removed.
- Matches upstream (`snel-repo/EMUsort` `a06bb60`, `ks4mods`; behaviour compared, code not
  ported): whitening = uncentred `X Xᵀ / n` per window, equal weight, on every `nskip`-th window
  **except the last**; channel delays estimated in the same pass on the high-passed data
  (**before whitening**; before this they were estimated on whitened data); template clips on
  every `nskip`-th window **including the last**. Fit pass: one read of the learning windows for
  both statistics (was three).
- `emusort/kernels.rs`: `ChannelDelayEstimator<R>` (envelope, lagged cross-correlation summed on
  the device, one download; edge reads repeat the edge sample, like upstream's batch padding),
  `ChannelAligner<R>` (persistent output, shifts uploaded per window length),
  `delays_from_cross_correlation`. Host `ChannelDelayEstimator` / `apply_channel_delays` removed
  (kept only as test oracles in `kernels.rs` tests). One GPU path for clips and detection.
- Clips: still on the host (one download per learning window); the fallback scan streams the
  remaining windows with read-ahead and stops once there are enough (`stream_while`).
- No silent defaults: `to_sorting_output(probe)` uses the run's rate and length; spike `x` from
  the centre position. `templates_from_data = false` takes `RunPlan::templates` or the hub; no
  more relative-path search for `wTEMP.npz`.
- `learn_universal_templates`: warns (`tracing`) when HDBSCAN keeps fewer than `n_templates`
  clips and falls back to all clips. An uncommitted change from another tool capped HDBSCAN to a
  2 000-clip subsample; removed (user: match upstream, HDBSCAN on all clips).
- `to_sorting_output`: primary channel = channel nearest the unit's most frequent centre
  (`Kilosort4Result::centre_channels`), no noise floor (amplitudes are whitened σ, SNR left
  undefined) instead of channel 0 and σ = 1. Predefined templates must match `nt`. Python
  `kilosort4.run(..., templates=None)` passes predefined templates.
- Named: `MIN_LEARNING_WINDOWS = 5`, `KILOSORT4_SORTER`, `EMUSORT_SORTER`.

## 2026-10-06 — fewer CPU↔GPU transfers

- `kilosort4/detect.rs`: `UniversalDetector<R>` uploads `wTEMP`, `wPCA`, `iC`, `iC2`, weights and
  thresholds once and keeps its scratch buffers (`B` is `[channels, n_templates, max_samples]`).
  Per window it reads back twice: the candidate counts (dsp-base
  `find_peak_candidates_on_device`), then spikes, arg-max, features and responses after every
  kernel ran. Before: six constant uploads, five allocations and four reads per window (candidates,
  arg-max, features), with the candidates re-uploaded between them. `spike_features_kernel` now
  reads the arg-max and decodes the template on the device; `gather_args_kernel` removed.
  `detect_universal` is a one-off `UniversalDetector`. Spikes are ordered by centre, then sample.
- `emusort/kernels.rs`: `delay_cc_kernel` (one unit per pair × lag, adds to the running sum; a
  branch-free loop unless the interior is within `max_lag` of an edge) replaces the split partial
  buffer (`[C², lags, splits]`: 476 MB per window at 256 channels) and its merge kernel; the
  estimator keeps mean / std / envelope buffers. `apply_channel_delays_kernel` wraps with one
  subtraction instead of `%`.
- Clip pass: `buffer::download_prefix` reads only the window's part of the max-sized buffer.
- `RunPlan::fitted` + `FitSettings`: a run reuses an earlier run's `FittedPreprocessing` when the
  fit settings match (error otherwise). `Kilosort4Result` now holds `fitted: FittedPreprocessing`
  (its `whitening`, `channel_delays`, `halos`, `windows`, `preprocessing` fields moved there).

## 2026-10-06 — universal templates on the device

`learn_universal_templates`: the scaled clips are uploaded once (`DevicePoints`); the Gram matrix
is `SecondMomentAccumulator` on them (`into_sum`, read by `symmetric_eigen` on the device; was a
host `f64` loop plus an upload), HDBSCAN (`hdbscan_points`) and k-means (`kmeans_points`) read the
same copy, inliers are gathered on the device. The host keeps the clip scale and the final row
normalisation.

## 2026-10-06 — progress

`run_plan(…, progress: &dyn ProgressSink)` and `fit_preprocessing(…, progress)` report the stages
`STAGE_FIT`, `STAGE_CLIPS`, `STAGE_TEMPLATES`, `STAGE_DETECTION` (those the run has; per window,
learning as one step); `Kilosort4::run_with_progress`, `Emusort::run_with_progress` (`run` reports
nothing). **Breaking:** `run_plan` / `fit_preprocessing` take the sink.
- `learn_universal_templates_with_progress`: "Learning templates" counts real steps (1 for `wPCA`,
  one per HDBSCAN launch, one per k-means restart) instead of 0/1. Fixed: a second
  "Finding clips" line when the pass had already reached its total.

## Deleted (2026-10-08)

Removed from `refactoring/` (superseded; history in git): `models/kilosort4/`, `models/emusort/`
(the fixed-template "Kilosort4" detector and the invented EMUsort design; replaced by
`sorters::kilosort4` / `sorters::emusort`), `examples/emusort_nwb_zarr.rs` (used them), `hub/{npy,
registry, pytorch_remap}.rs` (replaced by `dsp_io::container::npy` and `dsp-synapse-hub`). Kept:
`catalog/` and `runtime/kernels/` (the deferred ONNX runtime), `build.rs`. Also deleted:
`refactoring/docs/` (the former `docs/sorters/`, replaced by the book's *Sorters* pages; see DOC1–DOC2
in `REVIEW.md`: invented designs, parameters Kilosort4 does not have, APIs that do not exist).
