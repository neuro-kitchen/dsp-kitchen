---
title: "Kilosort4: pipeline"
description: "The stages of Kilosort4."
sidebar:
  label: "Pipeline"
  order: 2
---

The stages below follow the paper (Pachitariu et al. 2024, Methods) and the published defaults.
**Implemented** stages show the dsp-kitchen API; the others are described so the remaining work is
clear. Where the paper is silent and we had to choose, the choice is named. Measured results are in
[Benchmarks](/dsp-kitchen/plugins/synapse/validation/).

Input to every stage is **preprocessed** data in batches of `batch_size = 60000` samples with halos
on each side, in whitened units (thresholds are multiples of the noise σ).

## Running over a recording — *implemented*

```rust
use dspk_sorter_kilosort::ks4::{Kilosort4, Kilosort4Config};

let result = Kilosort4::new(Kilosort4Config::default()).run(&client, &recording, &probe)?;
let sorting = result.to_sorting_output(Some(probe));   // one unit per cluster, with templates
```


One runner serves Kilosort4 and EMUsort (`run_plan` with `RunPlan::kilosort4` /
`RunPlan::emusort`). Batches are the windows of a dspk-core `ChunkSchedule` (`batch_size`, with
halos covering the filter's settling, `nt` and, for EMUsort, the largest channel delay), streamed
by `dspk_core::WindowLoader` (the next window is read while the device works) through a
`PipelineWorkspace`, so memory is bounded whatever the recording length. The passes, each a stage
of the progress report:

1. **Fit**: every `nskip`-th window except the last (upstream `range(0, n_batches − 1, nskip)`),
   high-passed (and CAR): the whitening second moment and, for EMUsort, the channel-delay
   cross-correlation accumulate on the device and come back once.
2. **Templates**: every `nskip`-th window (upstream `range(0, n_batches, nskip)`: a recording of
   `nskip` windows or fewer is learned from window 0, as upstream). Each window comes down once for
   clip extraction on the host; only if too few clips are found are the other windows scanned.
3. **Drift** (`STAGE_DRIFT`, when `config.drift.nblocks >= 1`): amplitude-only universal detection
   (`UniversalDetector::detect_for_drift`, skipping PCA feature projection) across every window;
   `estimate_drift` estimates per-batch depth shifts on the host, and `DriftCorrector` precomputes
   per-batch `[channels, channels]` kriging matrices `M_b` (within a 256 MiB budget) applied on the
   device immediately after whitening in subsequent passes.
4. **Detection**: every window, preprocessed, drift-corrected, and detected on the device; only
   spikes and their `wPCA` features come back (`HaloWindow::remap_event`).
5. **Clustering** of the detected spikes, section by section (stage 4 below), and the **learned
   templates** (stage 5).
6. **Matching**: every window again, drift-corrected on the device and matched against the learned
   templates (stage 6).
7. **Clustering of the matched spikes** into the final units (stage 7), then the **global merges**
   (stage 8); duplicate spikes are removed and units labelled (stage 9).

| Result (`Kilosort4Result`) | |
|---|---|
| `fitted` (`FittedPreprocessing`) | the preprocessing `Pipeline`, `SpatialWhitening`, `channel_delays` (EMUsort), schedule and halos, and the `FitSettings` they were fitted with |
| `templates` | the universal templates (`wPCA`, `wTEMP`) |
| `drift` | `Option<DriftModel>` with per-batch, per-block depth shifts (µm) and block centres (`None` when `nblocks == 0`) |
| `detected`, `first_clusters` | the universal-template spikes and their first clustering |
| `learned` | the learned templates (`LearnedTemplates`) |
| `spikes`, `clusters` | the matched spikes (sample, template, amplitude, position, features) and their units |
| `merges`, `duplicates` | global merges applied, spikes removed as duplicates |
| `reproducible`, `device` | whether the run pinned its result-changing tuned choices, and the device it ran on |
| `to_sorting_output(probe)` | one unit per cluster: spike times, amplitudes (whitened σ), positions, and a waveform template (mean PC features × `wPCA`); primary channel where that template is largest; label *good* / *mua* from the unit's auto-correlogram (stage 9); EMUsort's composite score; no SNR in µV (amplitudes are whitened σ) |

A run reuses an earlier run's preprocessing (`RunPlan::fitted`)
when the fit settings match, e.g. to compare learned and predefined templates without a second fit.

**Reproducibility.** With `reproducible = true` (the default) the autotuned choices that change the
numbers (IIR time blocks, matrix-product routine) are pinned, k-means++ draws come from a seeded
generator, and every reduction is ordered: the same input on the same device gives the same spikes
and units (checked run to run on both test recordings).

## 1. Preprocessing — *implemented (approximation)*

Three switches build the filter chain, in this order: common average reference (`do_car`; a mean,
upstream may use a median), one Butterworth band-pass (`do_bandpass`, `bandpass_low_hz` …
`bandpass_high_hz`, `BANDPASS_ORDER` = 3 per edge, forward-backward), and a line-noise notch
(`do_notch`, `notch_hz`, `notch_q`). Kilosort4's defaults: CAR on, the action-potential band
300–6000 Hz (`NeuralBand::Ap`), notch off. **Upstream Kilosort4 high-passes at 300 Hz with no upper
edge**; the 6 kHz upper edge is ours (decided 2026-10-09). Then local whitening over
the `whitening_range = 32` nearest channels from the uncentred second moment `X Xᵀ / n` of each
fit window's interior, averaged with equal weight per window, as upstream `get_whitening_matrix`
(`SecondMomentAccumulator`, `SpatialWhitening::local_knn_from_covariance`, `WHITENING_EPSILON`).

## 2. Universal templates — *implemented*

**Learned from the data** (default):

1. **Clips.** On every `nskip`-th batch, a sample is a *peak* when `|x|` is the maximum over
   ±4 channel indices × ±5 samples (upstream `loc_range = [4, 5]` over `[channels, samples]`) and
   above `Th_single_ch`; peaks with another peak within ±6 channel indices × ±`nt / 2` samples are
   dropped (only isolated ones are kept), as are the first and last `nt` samples. Each clip is
   `x[ch, t − nt0min .. t − nt0min + nt]`; at most 500 000.
2. **Scale** (`ClipScaling`): Kilosort4 scales **each clip to unit norm**, so k-means groups shapes
   alone; EMUsort's fork scales all clips by one factor (`sqrt(std of ‖clip‖²)`), keeping relative
   amplitudes.
3. **`wPCA`**: the top `n_pcs` right singular vectors of the clip matrix (not centred).
4. **`wTEMP`**: k-means centres (`n_templates`, 10 initialisations), rows normalized to unit length.

Clips are found on the host (`extract_clips`); learning runs on the device from one upload of the
scaled clips: the `nt × nt` Gram matrix, its eigendecomposition, and k-means. Against Kilosort4's
saved `wPCA` / `wTEMP` on the Neuropixels test recording, every learned row has a best |cosine| of
0.94–1.00.

**Predefined** (`templates_from_data = false`): `UniversalTemplates::from_npz(path)` reads
Kilosort4's `wTEMP.npz` (fetched and verified through dspk-hub
with id `kilosort4/wtemp-v1`).

## 3. Universal-template detection — *implemented (device)*

1. **Template centres** (once per probe): on each shank, a grid every `dmin / 2` vertically
   (`dmin` = median vertical contact spacing unless set) and with
   `round((xmax − xmin) / (dminx / 2)) + 1` columns (`dminx = 32` µm). Each centre uses its
   `nearest_chans = 10` contacts; centres farther than `max_channel_distance = 32` µm from every
   contact are dropped. Spatial weights `exp(−d² / σ²)` for `template_sizes = 5` widths
   `σ = min_template_size · (s + 1)`, normalized per centre. On the Neuropixels test probe the
   1532 centres equal Kilosort4's (`xcup`, `ycup`); their channel sets differ only where channels
   are equidistant.
2. **Responses.** Every channel is correlated with every `wTEMP` row; for each centre, the weighted
   sum over its contacts is taken for every size and template, and the largest magnitude kept.
3. **Local maxima** over the `nearest_templates = 100` neighbouring centres and ±`nt0min` samples,
   above `Th_universal = 9`. **Exact ties keep one spike** (`drop_tied_peaks`): where contacts are far
   apart compared with the template sizes (an HD-EMG grid: 100 µm pitch, centres every 16 µm),
   several centres' weights collapse onto one contact and report bit-identical responses; the paper
   asks that "no spike is detected twice", so the lowest centre of a tie is kept. (On the HD-EMG test
   recording, 56 000 of 94 000 detections were such copies.)
4. **Per spike**: amplitude, template, centre, features (the centre's `nearest_chans` snippets
   projected onto `wPCA`), and a **position**: the centre of mass of the centre's contacts weighted
   by the rectified template response, in x and y (the paper's spike position). **Time** is the
   waveform's trough, `t − nt/2 + nt0min` for a correlation peak at `t`, as Kilosort4 reports spike
   times.

## 4. Clustering — *implemented (device)*

`kilosort4::clustering::cluster_spikes`, written from the paper's *Graph-based clustering*:

1. **Sections.** Spikes are binned by vertical position (`section_um = 40`), per shank, and embedded
   on the union of their centres' channels (`[slots · n_pcs]` features, 0 elsewhere), built on the
   device from the compact features.
2. **Bipartite k-NN graph** ([`dspk_synapse::sorting::bipartite`](/dsp-kitchen/gpu/clustering/)): each spike
   links to its 10 nearest neighbours among a subset (every `cluster_downsampling = 20`-th spike, at
   most 25 000), so neighbourhood scales do not shrink as recordings grow.
3. **Assignment.** From 200 k-means++ seeds, right nodes then left nodes take the cluster maximizing
   the paper's bipartite modularity gain `n_tc − γ·k_t·K_c / 2m` (γ = 1), until no spike moves.
4. **Merging tree.** The pair of clusters with the highest `γ̂ = 2m·K_ij / (k_i·k_j)` is merged,
   repeatedly (host; at most 200 leaves).
5. **Decisions** from the top: a node splits when `γ̂ < 0.2` or when its halves are bimodal along their
   weighted regression axis (score ≥ 0.6: 400 bins over `[−2, 2]`, Gaussian smoothing of 4 bins,
   trough in bins 175–225, score `1 − max(x_min/x₁, x_min/x₂)`). Sections with fewer than 1000 spikes
   are one unit.

Choices of ours: the regression has a bias term (the paper's axis passes through the origin; spike
features are not centred, and without it a half sitting at the origin projects into the trough); a
node only joins a cluster among its neighbours'; ties go to the smallest label.

## 5. Learned templates — *implemented (device)*

`kilosort4::learned::learned_templates`, from the paper's *Template learning*: each unit's template
(its mean PC features) is **aligned** in time by its best correlation with the `wTEMP` prototypes,
then templates whose correlation, maximized over lags, is ≥ 0.9 and whose norms differ by less than
20% are **merged** (largest units first, in passes until nothing merges). All pairs at all lags are
one `matmul` in PC space plus one kernel (see [the GPU case study](/dsp-kitchen/gpu/clustering/)).

## 6. Learned-template matching — *implemented (device)*

`kilosort4::matching::TemplateMatcher`, from the paper's *Spike detection with learned templates and
matching pursuit* and *Extracting PC features with background subtraction*:

1. **Scores**: every channel correlated with the `wPCA` rows, then one `matmul` with the unit-norm
   templates in PC form gives every template's projection `c_j(t)` at every sample.
2. **Matching pursuit**, `max_peels = 50` rounds: per sample, the template explaining the most variance
   at its average norm, `V = 2·μ·c − μ²`; spikes where `V` is the maximum over ±`nt` and
   `c ≥ Th_learned = 8`; their contributions are subtracted from the data and, through the template
   products at every lag, from the scores.
3. **Features**: the residual's PC projections plus the spike's own template contribution.

Choices of ours: a template's average norm is its mean waveform's norm; a matched spike is placed at
its template's position (centre of mass of its channel energies).

## 7. Clustering of the matched spikes — *implemented (device)*

The matched spikes' features are clustered as in stage 4, giving the final units, with the paper's
second criterion: before a node's bimodality is tested, its halves' **cross-correlogram** is
(`dspk_synapse::metrics::compute_ccg_refractory`). The paper's *Refractory auto- and cross-correlograms*: 1 ms
bins over ±0.5 s; with `n_k` the coincidences in the central `−k..k` bins and `R` the baseline per
bin (the larger shoulder), `R12 = min_k n_k / ((2k+1)·R)` and `Q12 = min_k ½(1 + erf((n_k − λ_k) /
√(ε + 2λ_k)))`, `λ_k = (2k+1)·R`; a CCG is refractory when `R12 < 0.25` and `Q12 < 0.05`.

**A refractory pair is not split.** The paper's text says the opposite ("If the pair of units has a
refractory cross-correlogram, then the split is always performed"). Two halves that never fire
within a few ms of each other behave as one neuron, which is how the paper's own global merges use
the same test ("A merge is performed if the CCG is refractory"), so we keep them together.
Choices of ours where the paper is silent: `k` up to 10 bins (10 ms) and shoulders beyond 250 ms
(`RefractoryOptions`).

## 8. Global merges — *implemented*

`kilosort4::merges::global_merges`, from the paper's *Global merges*: units are taken in decreasing
spike count; every unit still in the pool whose waveform similarity (correlation maximized over
lags, computed in PC space on the device as for the learned templates) exceeds
`global_merge.min_similarity = 0.5` is tested, most similar first, and merged when their CCG is
refractory; a merged unit is tested again, and a unit that merges with nothing is complete and
leaves the pool. Choice of ours: a merged template is the spike-count-weighted mean (no shift).

## 9. Duplicates and labels — *implemented*

- **Duplicates**: a unit's spike within `duplicate_spike_ms = 0.25` ms of its previous kept spike is
  removed (matching pursuit re-matching a template's own residual). EMUsort's
  `duplicate_spike_bins = 7` is the same setting in samples (0.23 ms at 30 kHz).
- **Labels**: a unit is *good* when its auto-correlogram is refractory (`R12 < 0.2`, `Q12 < 0.2`;
  the paper's Methods say 0.1, Kilosort4 runs with `acg_threshold = 0.2`), else *mua*; written to Phy's `cluster_KSLabel.tsv` and `cluster_group.tsv`.
- **Composite score**: EMUsort's per-unit score (see the [EMUsort pipeline](/dsp-kitchen/plugins/synapse/reference/sorters/emusort-pipeline/)),
  written to `cluster_info.tsv` (`composite_score`).

## 10. Drift estimation and correction — *implemented (device)*

`kilosort4::drift::{estimate_drift, DriftOptions, DriftModel, DriftCorrector}` follows the
paper's *Drift correction* (`datashift.py` and `get_drift_matrix`):

1. **Fingerprints** (host, from `STAGE_DRIFT` amplitude-only spikes): each batch builds a
   `[depth_bins, 20]` histogram over `binning_depth_um` depth bins (starting 1 µm below the
   lowest channel) and 20 logarithmic amplitude bins from the universal threshold to 99, scaled
   as `log2(1 + count)`.
2. **Rigid alignment:** mean-subtracted fingerprints are shifted over `−15..=15` depth bins to
   maximize correlation against a reference target (initial middle batch, or batch 300), updating
   the target as the aligned mean over 10 rounds.
3. **Fine block alignment:** across `2 · nblocks − 1` half-overlapping depth blocks, each block
   searches `−5..=5` bins, smooths the `(shift, batch, block)` score volume with a 3D reflect
   Gaussian filter, upsamples ×10 with a Gaussian interpolation kernel, and adds the sub-bin peak
   to produce `dshift[batch, block]` in µm.
4. **Device correction (`DriftCorrector`):** per batch, each channel's vertical shift is
   linearly interpolated between block centres (extrapolated past the ends), and the window `X`
   becomes `M_b · X`, where `M_b = K(y − shift_b, x) · (K(x, x) + 0.01·I)⁻¹` using a Gaussian
   kernel with width `sig_interp_um`. Matrices `M_b` are precomputed once at construction (up to
   `MAX_PRECOMPUTED_KRIGING_BYTES = 256 MiB`, falling back to on-demand computation for larger
   recordings) so `STAGE_DETECTION` and `STAGE_MATCHING` pay only `M_b · X` on the device.
