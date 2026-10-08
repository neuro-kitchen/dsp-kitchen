# Kilosort4: pipeline

The stages below follow the paper (Pachitariu et al. 2024, Methods) and the published defaults.
**Implemented** stages show the dsp-kitchen API; the others are described so the remaining work is
clear. Where the paper is silent and we had to choose, the choice is named. Measured results are in
[Benchmarks](../benchmarks.md).

Input to every stage is **preprocessed** data in batches of `batch_size = 60000` samples with halos
on each side, in whitened units (thresholds are multiples of the noise σ).

## Running over a recording — *implemented*

```rust,ignore
use dsp_synapse_ml::sorters::{Kilosort4, Kilosort4Config};

let result = Kilosort4::new(Kilosort4Config::default()).run(&client, &recording, &probe)?;
let sorting = result.to_sorting_output(Some(probe));   // one unit per cluster, with templates
```

```python
from dsp_kitchen.synapse.ml import kilosort4

result = kilosort4.run(recording, probe, kilosort4.Config())
print(result.n_units, result.n_learned_templates)
sorting = result.to_sorting_output(probe)
```

One runner serves Kilosort4 and EMUsort (`run_plan` with `RunPlan::kilosort4` /
`RunPlan::emusort`). Batches are the windows of a dsp-core `ChunkSchedule` (`batch_size`, with
halos covering the filter's settling, `nt` and, for EMUsort, the largest channel delay), streamed
by `dsp_core::WindowLoader` (the next window is read while the device works) through a
`PipelineWorkspace`, so memory is bounded whatever the recording length. The passes, each a stage
of the progress report:

1. **Fit**: every `nskip`-th window except the last (upstream `range(0, n_batches − 1, nskip)`),
   high-passed (and CAR): the whitening second moment and, for EMUsort, the channel-delay
   cross-correlation accumulate on the device and come back once.
2. **Templates**: every `nskip`-th window (upstream `range(0, n_batches, nskip)`: a recording of
   `nskip` windows or fewer is learned from window 0, as upstream). Each window comes down once for
   clip extraction on the host; only if too few clips are found are the other windows scanned.
3. **Detection**: every window, on the device; only spikes come back. Each window keeps its own
   spikes (`HaloWindow::remap_event`).
4. **Clustering** of the detected spikes, section by section (stage 4 below), and the **learned
   templates** (stage 5).
5. **Matching**: every window again, matched against the learned templates (stage 6).
6. **Clustering of the matched spikes** into the final units (stage 7).

| Result (`Kilosort4Result`) | |
|---|---|
| `fitted` (`FittedPreprocessing`) | the preprocessing `Pipeline`, `SpatialWhitening`, `channel_delays` (EMUsort), schedule and halos, and the `FitSettings` they were fitted with |
| `templates` | the universal templates (`wPCA`, `wTEMP`) |
| `detected`, `first_clusters` | the universal-template spikes and their first clustering |
| `learned` | the learned templates (`LearnedTemplates`) |
| `spikes`, `clusters` | the matched spikes (sample, template, amplitude, position, features) and their units |
| `reproducible`, `device` | whether the run pinned its result-changing tuned choices, and the device it ran on |
| `to_sorting_output(probe)` | one unit per cluster: spike times, amplitudes (whitened σ), positions, and a waveform template (mean PC features × `wPCA`); primary channel where that template is largest; no SNR (amplitudes are whitened σ) |

A run reuses an earlier run's preprocessing (`RunPlan::fitted`, Python `preprocessing_from=result`)
when the fit settings match, e.g. to compare learned and predefined templates without a second fit.

**Reproducibility.** With `reproducible = true` (the default) the autotuned choices that change the
numbers (IIR time blocks, matrix-product routine) are pinned, k-means++ draws come from a seeded
generator, and every reduction is ordered: the same input on the same device gives the same spikes
and units (checked run to run on both test recordings).

## 1. Preprocessing — *implemented (approximation)*

High-pass at `highpass_cutoff_hz = 300` Hz (`HIGHPASS_ORDER` = 3, forward-backward), common
average reference when `do_car` (a mean; upstream may use a median), and local whitening over
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
Kilosort4's `wTEMP.npz` (fetched and verified through [dsp-synapse-hub](../../crates/dsp-synapse-hub.md)
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
2. **Bipartite k-NN graph** ([`dsp_synapse::sorting::bipartite`](../../gpu/clustering.md)): each spike
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
one `matmul` in PC space plus one kernel (see [the GPU case study](../../gpu/clustering.md)).

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

## 7. Clustering of the matched spikes — *implemented (device), partly*

The matched spikes' features are clustered as in stage 4, giving the final units. Not yet: the
paper's refractory cross-correlogram criterion (a pair whose CCG is refractory is never split).

## 8. Not yet

- **Global merges** (paper: units sorted by size; pairs with waveform similarity above 0.5 merged
  when their CCG is refractory) and **duplicate-spike removal** (`duplicate_spike_ms`).
- **Drift correction.** On the Neuropixels test recording Kilosort4's own correction is ±0.5 µm
  (`ops['dshift']`), so it does not explain any difference there.
