# dsp-synapse-ml

## Intent

Distributes neuro sorters and models from the literature, each traceable to its authors:

- **Sorters** (`sorters/`): algorithms reimplemented here from their papers and published
  defaults — [Kilosort4](../sorters/kilosort4/intro.md) and its fork
  [EMUsort](../sorters/emusort/intro.md).
- **Models** (`models/`): pretrained networks run as their authors released them.

Every sorter and model carries a [`Provenance`](#provenance); results can always be cited.

### Owns
- Sorter stages specific to a published sorter, their settings and provenance.
- The catalog of published artifacts (`catalog/models.json`) and the model wrappers.

### Must not contain
- Generic DSP or clustering (dsp-base, dsp-synapse), file formats (dsp-io), network code
  (dsp-synapse-hub).
- Invented weights: an entry without a verified artifact is not shipped.
- Device selection: GPU stages take a cubecl `Client`; models take an explicit `ComputeTarget`.

## Features

| Feature | Default | Enables |
|---|---|---|
| `wgpu` / `cpu` / `cuda` | `wgpu` | the CubeCL / Burn runtime |
| `flex` | yes | Burn's CPU backend for the models |
| `hub` | no | `ModelHub`, `from_hub` constructors: catalog ids resolved to verified files through [dsp-synapse-hub](dsp-synapse-hub.md) |

Without `hub` the crate has no network access; artifacts are loaded from paths.

## Module map

```text
dsp-synapse-ml/
├── catalog/models.json        verified artifacts (today: Kilosort4 wTEMP.npz)
└── src/
    ├── provenance.rs          Provenance, Paper, UpstreamCode, ArtifactSource, Attributed
    ├── sorters/
    │   ├── kilosort4/         config; runner.rs (RunPlan, run_plan: fit, templates, detection over a
    │   │                      recording); templates.rs (clips, learned or wTEMP.npz); detect.rs
    │   │                      (TemplateCentres, UniversalDetector); kernels.rs
    │   └── emusort/           config, RunPlan::emusort, Emusort::run; kernels.rs (ChannelDelayEstimator,
    │                          ChannelAligner on the device); reuses kilosort4 + HDBSCAN outliers
    ├── models/                dartsort (denoiser, VAE), spikenet2, unitrefine — no verified artifacts yet
    ├── hub/                   catalog, manifests, safetensors reader, ModelHub (feature hub)
    └── runtime/               Burn operations and an ONNX evaluator for the models
```

## Running a sorter

`Kilosort4::new(config).run(client, source, probe)` and `Emusort::new(config).run(…)` stream a
whole recording through one runner (`run_plan` with a `RunPlan`): preprocessing fitted on the
recording, universal templates, detection, clustering, learned templates, learned-template matching
and the clustering of the matches; see the [Kilosort4 pipeline](../sorters/kilosort4/pipeline.md).
They return a `Kilosort4Result` (`fitted`, `templates`, `detected`, `first_clusters`, `learned`,
`spikes`, `clusters`, `reproducible`, `device`, `to_sorting_output(probe)` with one unit per cluster
and its waveform template). Modules: `detect` (universal templates), `clustering` (sections, merging
tree, splits), `learned` (template alignment and merging), `matching` (`TemplateMatcher`, matching
pursuit), each with its kernels under `kernels/`. Results and times: [Benchmarks](../sorters/benchmarks.md).
Python: `dsp_kitchen.synapse.ml.kilosort4.run` / `emusort.run` (with `templates=`,
`preprocessing_from=`, `progress=`, `runtime=`).

**Progress.** `run_plan(…, progress)` / `run_with_progress` report each stage to a
`dsp_core::ProgressSink`: `Fitting preprocessing`, `Finding clips`, `Learning templates`,
`Detecting spikes` (the ones the run has; windows, or one step for learning). In Python a bar is
shown by default (`progress=False` to silence it, or a callable
`(stage, step, steps, done, total, unit)`):

```text
[1/4] Fitting preprocessing  ████████████████████████  12/12 windows  0:04
[2/4] Finding clips          ████████████░░░░░░░░░░░░  6/12 windows  0:03 < 0:03
```

Data movement follows the workspace rule ([Architecture](../architecture.md#data-movement)): each
window goes up once; the fit statistics come back once; constants (templates, centre tables) are
uploaded once per run; per window only candidate counts and spikes come back (clip extraction
reads learning windows back).

## Provenance

```rust,ignore
use dsp_synapse_ml::{Attributed, sorters::Emusort};

let p = Emusort::default().provenance();
println!("{}", p.citation());
// O’Connell … Pandarinath (2026). High performance sorting of motor unit action potentials
// with EMUsort. eLife 15:RP110417 (2026). https://doi.org/10.7554/eLife.110417.1.
// Code: https://github.com/snel-repo/EMUsort (a06bb60 …, GPL-3.0)
```

| Field | Meaning |
|---|---|
| `name`, `kind` | As published; `ReimplementedFromPaper` or `UpstreamWeights`. |
| `paper` | Title, authors, venue, year, DOI (resolved through doi.org when added), paper license. |
| `code` | Upstream repository, its license (SPDX, or none stated), the version followed. |
| `artifacts` | Each downloaded file: URL, SHA-256, size, license. |
| `notes` | What differs from upstream, what is not implemented. |

The catalog stores the same record per artifact; [Citations](../reference/citations.md) lists all.

## Catalog

`ModelCatalog::load_default()` reads the embedded `catalog/models.json` (schema 2) and merges an
override from `DSP_KITCHEN_HUB_CATALOG`. Each `ModelManifest` has the download URI, SHA-256, size,
the arrays or tensor ports it holds, and its `provenance`. An entry is listed only with a
downloaded-and-hashed artifact, a resolved DOI and the upstream license.

| Id | Artifact |
|---|---|
| `kilosort4/wtemp-v1` | Kilosort4's predefined `wTEMP.npz` (`wPCA`, `wTEMP`, 6 × 61), OSF, 3432 B, SHA-256 `cae1c96f…abd8` |

## Models

`DartsortWaveformDenoiser`, `DartsortVaeEmbedder`, `SpikeNet2Detector`, `UnitRefineClassifier`
load user-supplied files (`from_safetensors_file`, `from_onnx_file`) on an explicit
`ComputeTarget`. None has a verified published artifact yet, so none is in the catalog.

## Limitations

- The model runtime moves data between host and device for every operation; it is to be replaced
  by `burn-onnx` with device-resident weights on the shared CubeCL client when a model artifact
  is validated.
- Sorters: everything up to the clustering of the learned-template matches runs; the refractory
  (CCG) criteria, global merges, duplicate-spike removal, drift correction and EMUsort's unit score
  are not implemented yet (see each sorter's page).
- Clip extraction for template learning runs on the host (one download per learning window).
