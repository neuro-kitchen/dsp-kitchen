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
