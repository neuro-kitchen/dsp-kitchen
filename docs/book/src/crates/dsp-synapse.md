# dsp-synapse

## Intent

Spikes and units: detecting action potentials in multi-channel recordings, cutting and aligning
their waveforms, describing them (features, positions), following the probe's drift, grouping
them into units, matching templates, measuring quality, and converting sortings to and from
files. Built on [dsp-base](dsp-base.md) primitives and [dsp-io](dsp-io.md) files.

### Owns
- Spike and unit types (`SpikeEvent`, `DeduplicatedSpike`, `WaveformSnippet`, `SnippetBatch`,
  `WaveformTemplate`, `SortedUnit`, `SortingOutput`) and the traits algorithms implement
  (`SpikeDetector`, `FeatureEmbedder`, `PeakLocalizer`, `SpikeMatcher`, `WaveformDenoiser`).
- Detection, deduplication, extraction, features, localization, drift, clustering, matching,
  metrics, streaming detection.
- Conversion between `SortingOutput` and the sorting files of dsp-io.

### Must not contain
- File layouts (dsp-io), generic DSP (dsp-base), probe geometry (dsp-io `neuro::probe`).
- Sorters reimplemented from papers or pretrained networks (dsp-synapse-ml).
- Device selection: GPU entry points take a cubecl `Client`.

## Module map

```text
dsp-synapse/src/
├── core/        spike / snippet / template / unit / sorting types, traits, neural bands
├── detection/   threshold, adaptive, NEO, matched-filter detectors; SpikeSpacing; device detection;
│                spatial deduplication (host, streaming, device)
├── extraction/  snippet cutting with sub-sample realignment (host and device)
├── features/    PCA / PPCA / wavelet embedders, morphology, HD-EMG conduction velocity
├── spatial/     localizers (centre of mass, monopolar, dipole, grid convolution), rigid and
│                non-rigid drift, kriging drift correction (host and device)
├── sorting/     GMM (EM on the device), KDE valley merge, density peaks, k-means and HDBSCAN (device,
│                points.rs: DevicePoints), CBSS, matching pursuit (device), template similarity
├── metrics/     firing (ISI, presence, amplitude cutoff, contamination), isolation, correlograms,
│                rates and bursts, evoked responses, sorting comparison
├── storage/     Phy / NWB units / .sorting.zarr ↔ SortingOutput
└── streaming/   StreamingDetector: out-of-core detection with per-channel templates
```

## Detection

| Item | What |
|---|---|
| `detect_spikes_multichannel`, `detect_spikes_with_sigma` | Local extrema (`dsp_base::peaks`) of a polarity above `threshold_factor · σ` (σ = `median(|x|) / 0.6745`, or given), spaced by `SpikeSpacing`. |
| `SpikeSpacing { refractory_samples, rule }` | Spikes on one channel more than `refractory_samples` apart; `rule`: `DistanceRule::LocallyExclusive` (default: a spike stays unless a larger one is nearer, exact in chunks) or `DistanceRule::Scipy` (`find_peaks` `distance`). |
| `ThresholdSpikeDetector`, `AdaptiveThresholdDetector`, `NeoSpikeDetector`, `MatchedFilterSpikeDetector` | `SpikeDetector`s: fixed σ; σ per block smoothed across blocks; Teager-Kaiser energy (trough within ±2 samples); correlation with a biphasic prototype. Each has a `distance_rule`. |
| `execute_detect_spikes_in_vram` | Device detection of a trace already on the device (candidates compacted on the device; scans the emit range ± the spacing distance). |
| `deduplicate_spikes_spatial`, `StreamingDedup`, `DedupNeighbours` | Locally exclusive across space and time: a crossing survives unless a stronger one (larger `|amplitude|`) is within the radius and window; streaming form equal to the whole-recording result; device form with a once-per-probe neighbour table. |

## Extraction and features

- `extract_snippet_batch_multichannel` / `execute_extract_sinc_in_vram`: snippets on the `k`
  nearest channels, realigned so the trough (parabolic sub-sample position) lands on sample
  `pre`, with the Blackman-Harris windowed-sinc fractional delay of dsp-base
  (`SINC_KERNEL_RADIUS = 5`). `SnippetBatch::peak_index` records that sample.
- `compute_mean_template` / `TemplateAccumulator`: means and SDs with `TEMPLATE_STD_DDOF = 0`
  (as Phy / SpikeInterface), through `dsp_base::math::RunningMoments`.
- `PcaFeatureEmbedder`, `PpcaFeatureEmbedder`, `WaveletFeatureEmbedder` (Haar + IQR ranking),
  `compute_morphology` (primary channel), `estimate_hdemg_conduction_velocity`.

## Localization and drift

- Localizers (`PeakLocalizer`): centre of mass, monopolar triangulation (Levenberg-Marquardt),
  dipole (variable projection + compass search), grid convolution (monopole footprints).
- `estimate_rigid_drift` / `estimate_nonrigid_drift`: activity profiles per time bin registered
  to the first bin with spikes by cross-correlation (`dsp_base::math`).
- `TraceKriging`: Gaussian-process interpolation onto drift-shifted positions, weights cached per
  0.1 µm drift step; host and device (`correct_in_vram`) forms.

## Sorting and matching

| Item | What |
|---|---|
| `GmmClusterer`, `cluster_gmm_bic` | Gaussian mixture by EM on the device (diagonal, full, masked covariances), BIC over `k`, Cholesky inverses; convergence on the mean log-likelihood (sklearn `tol`). |
| `cluster_kde_merge` | KDE valley merge of an over-clustering (a heuristic in the spirit of IsoSplit; no significance test). |
| `kmeans`, `hdbscan` (`kmeans_points`, `hdbscan_points` on `DevicePoints`) | sklearn semantics (k-means++ seeding, restarts; HDBSCAN with excess-of-mass selection), on the device. Points are uploaded once, feature-major (`DevicePoints`, subsets gathered on the device). k-means: assignments, sums, trial potentials and inertia on the device; the host keeps the random draws and the `[k, d]` centres (reads back `k · d` sums per iteration, one block of weights per draw). HDBSCAN: core distances and a Borůvka minimum spanning tree on the device (`O(n²·d)` per round, ~`log₂ n` rounds, `O(n)` read back per round), points shared in tiles through shared memory and each pass split into launches of at most `PAIR_TERMS_PER_LAUNCH` terms (progress after each, `hdbscan_points_with_progress`); the condensed tree and cluster selection on the host. |
| `cluster_density_peaks`, `ConvolutiveBssDecomposer` | Rodriguez–Laio density peaks; convolutive blind source separation for HD-EMG motor units. |
| `match_spikes_matching_pursuit`, `MatchingPursuitMatcher` | Greedy matching pursuit with bounded amplitudes on the device; picks found and compacted on the device. |

## Metrics

Firing (`compute_isi_violations`, `compute_presence_ratio`, `compute_amplitude_cutoff`,
`compute_llobet_contamination`), isolation (`compute_snr`, `compute_d_prime` and
`compute_isolation_distance` with diagonal covariance, `compute_silhouette_score`), correlograms,
firing rates and bursts, evoked responses (`compute_psth`, `compute_stimulus_triggered_average`
with `STA_STD_DDOF = 1`, `quantify_mep`), and `compare_sortings` (greedy unit matching). Undefined
results are NaN, never a made-up value.

## Storage

`load_sorting` / `save_sorting` convert between `SortingOutput` and the files read and written by
dsp-io (`detect_sorting` picks the format): Phy / Kilosort folders (`load_phy_folder`,
`save_phy_folder`, `load_spikes` for curation), NWB `/units`, and `.sorting.zarr`.

## Streaming detection

`StreamingDetector::run_on(client, source, pipeline, probe)` (or `run_with(target, …)`):
calibrates noise on chunks spread over the recording (on the device), then streams halo windows
through a persistent `PipelineWorkspace` with `dsp_core::WindowLoader`, detects, deduplicates, extracts
snippets and accumulates per-channel templates — all on the device. It is detection with
per-channel templates, **not** clustering: `StreamingDetectionResult::to_sorting_output` reports
one unit per primary channel. Any batch size gives the whole-recording result.

## Design rules

1. Algorithms take a cubecl `Client`; nothing chooses a device.
2. Generic DSP lives in dsp-base, file layouts in dsp-io; this crate composes them.
3. Undefined results are NaN; documented names say what an algorithm really is (for example
   matching pursuit, not OMP; KDE valley merge, not IsoSplit).

## Limitations

- Localizers run on the host, one spike at a time.
- `cluster_kde_merge` runs on the host.
- `hdbscan` is exact: `O(n²·d)` work per Borůvka round on the device (no spatial index), so very
  large point sets (hundreds of thousands) still take long.
- Drift registration uses a single reference bin (no iterative template).
