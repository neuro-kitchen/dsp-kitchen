# Kilosort4: pipeline

The stages below follow the paper and the published defaults. **Implemented** stages show the
dsp-kitchen API; the others are described so the remaining work is clear.

Input to every stage is **preprocessed** data in batches of `batch_size = 60000` samples with
`nt` samples of padding on each side, in whitened units (thresholds are multiples of the noise σ).

## Running over a recording — *implemented*

```rust,ignore
use dsp_synapse_ml::sorters::{Kilosort4, Kilosort4Config};

let result = Kilosort4::new(Kilosort4Config::default()).run(&client, &recording, &probe)?;
let sorting = result.to_sorting_output(Some(probe));   // one unit per universal template
```

```python
from dsp_kitchen.synapse.ml import kilosort4

result = kilosort4.run(recording, probe, kilosort4.Config())
sorting = result.to_sorting_output(probe)
```

One runner serves Kilosort4 and EMUsort (`run_plan` with `RunPlan::kilosort4` /
`RunPlan::emusort`). Batches are the windows of a dsp-core `ChunkSchedule` (`batch_size`, with
halos covering the filter's settling, `nt` and, for EMUsort, the largest channel delay), streamed
by `dsp_core::WindowLoader` (the next window is read while the device works) through a
`PipelineWorkspace`, so memory is bounded whatever the recording length. Three passes:

1. **Fit**: every `nskip`-th window except the last (upstream `range(0, n_batches − 1, nskip)`),
   high-passed (and CAR): the whitening second moment and, for EMUsort, the channel-delay
   cross-correlation accumulate on the device and come back once.
2. **Templates**: every `nskip`-th window including the last (upstream
   `range(0, n_batches, nskip)`), preprocessed (and delay-aligned) on the device; each window
   comes down once for clip extraction on the host. If too few clips are found, the remaining
   windows are scanned until there are enough. Learning itself runs on the device.
3. **Detection**: every window, on the device; only spikes come back. Each window keeps its own
   spikes (`HaloWindow::remap_event`), reported as recording samples.

For recordings with `nskip` windows or fewer, the learning stride samples ~5 windows
(`MIN_LEARNING_WINDOWS`) instead of only window 0.

| Result (`Kilosort4Result`) | |
|---|---|
| `fitted` (`FittedPreprocessing`) | the preprocessing `Pipeline`, `SpatialWhitening`, `channel_delays` (EMUsort), schedule and halos, and the `FitSettings` they were fitted with |
| `templates`, `spikes` | universal templates; detected spikes (sample, centre, amplitude in whitened σ, template, size, position, features) |
| `to_sorting_output(probe)` | one unit per universal template at the recording's rate and length; spike locations are the detecting centre's `x` and the response-weighted `y`; primary channel nearest the unit's most frequent centre; no SNR (amplitudes are whitened σ) |

A run reuses an earlier run's preprocessing (`RunPlan::fitted`, Python
`preprocessing_from=result`) when the fit settings match (`FitSettings`, an error otherwise),
e.g. to compare learned and predefined templates without a second fit pass. Predefined templates
are passed with `RunPlan::templates` (Python `templates=`), else pulled from the hub.

## 1. Preprocessing — *implemented (approximation)*

High-pass at `highpass_cutoff_hz = 300` Hz (`HIGHPASS_ORDER` = 3, forward-backward), common
average reference when `do_car` (a mean; upstream may use a median), and local whitening over
the `whitening_range = 32` nearest channels from the uncentred second moment `X Xᵀ / n` of each
fit window's interior, averaged with equal weight per window, as upstream `get_whitening_matrix`
(`SecondMomentAccumulator`, `SpatialWhitening::local_knn_from_covariance`, `WHITENING_EPSILON`).
The reference and ε are to be verified against upstream.

## 2. Universal templates — *implemented*

**Learned from the data** (default):

1. **Clips.** On every `nskip`-th batch, a sample is a *peak* when `|x|` is the maximum over ±4
   samples × ±5 channel indices and above `Th_single_ch`; peaks with another peak within ±6
   channel indices × ±`nt / 2` samples are dropped (only isolated ones are kept), as are the first
   and last `nt` samples. Each clip is `x[ch, t − nt0min .. t − nt0min + nt]`; at most 500 000.
2. **Scale.** All clips are divided by one factor, `sqrt(std over clips of ‖clip‖²)`, so their
   relative amplitudes are kept.
3. **`wPCA`**: the top `n_pcs` right singular vectors of the clip matrix (not centred).
4. **`wTEMP`**: k-means centres (`n_templates`, 10 initialisations) of the clips, rows normalized
   to unit length.

```rust,ignore
use dsp_synapse_ml::sorters::kilosort4::{extract_clips, learn_universal_templates, Kilosort4Config};

let cfg = Kilosort4Config::default();
let mut clips = Vec::new();
for batch in whitened_batches.iter().step_by(cfg.nskip) {
    extract_clips(batch, channels, padded_samples, &cfg.clip_options(), &mut clips);
}
let templates = learn_universal_templates(&client, &clips, cfg.nt, &cfg.learn_options())?;
```

Clips are found on the host (`extract_clips`); learning runs on the device from one upload of
the scaled clips (feature-major `DevicePoints`): the `nt × nt` Gram matrix
(`SecondMomentAccumulator`), its eigendecomposition (dsp-base `linalg`, read on the device), and
k-means (`dsp_synapse::sorting::kmeans_points`).

**Predefined** (`templates_from_data = false`): `UniversalTemplates::from_npz(path)` reads
Kilosort4's `wTEMP.npz` (fetched and verified through [dsp-synapse-hub](../../crates/dsp-synapse-hub.md)
with id `kilosort4/wtemp-v1`).

## 3. Universal-template detection — *implemented (device)*

1. **Template centres** (once per probe): on each shank, a grid of virtual positions every
   `dmin / 2` vertically (`dmin` = median vertical contact spacing unless set) and with
   `round((xmax − xmin) / (dminx / 2)) + 1` columns (`dminx = 32` µm). Each centre uses its
   `nearest_chans = 10` contacts; centres farther than `max_channel_distance = 32` µm from every
   contact are dropped. Spatial weights `exp(−d² / σ²)` for `template_sizes = 5` widths
   `σ = min_template_size · (s + 1)` (`min_template_size = 10` µm), normalized per centre.
2. **Responses.** Every channel is correlated with every `wTEMP` row; for each centre, the
   weighted sum over its contacts is taken for every size and template, and the largest magnitude
   kept (with its template, size and sign).
3. **Local maxima.** The response is maximized over the `nearest_templates = 100` neighbouring
   centres and over ±`nt0min` samples; a spike is a centre and sample whose response equals that
   maximum and exceeds `Th_universal = 9`. Candidates are compacted and stay on the device.
4. **Per spike**: amplitude, template, centre, a vertical position (contact positions weighted by
   the rectified template response), and features — the centre's `nearest_chans` snippets projected
   onto `wPCA` (`nearest_chans × n_pcs`), computed on the device for every candidate; a batch
   reads back only the candidate counts and then the spikes.

`UniversalDetector` uploads the templates and centre tables once and keeps its scratch buffers;
`detect_universal` is a one-off detector for a single batch.

```rust,ignore
use dsp_synapse_ml::sorters::kilosort4::{TemplateCentres, UniversalDetector};

let centres = TemplateCentres::new(&probe, &cfg.centres)?;
let mut detector = UniversalDetector::new(&client, channels, max_padded_samples, &centres,
                                          &templates, cfg.th_universal, cfg.nt0min())?;
for (batch_handle, padded_samples) in batches {
    let spikes = detector.detect(&batch_handle, padded_samples)?;
}
```

## 4. Drift correction — *not yet*

Kilosort4 estimates vertical drift from the detected spikes' positions and corrects the data
before the second detection. dsp-synapse has rigid / non-rigid drift estimation and kriging
correction that a driver can use; the Kilosort4-specific procedure is not written.

## 5. Clustering, learned templates, deconvolution, merging — *not yet*

Clustering of the features in local probe regions (graph-based), multi-channel templates from the
clusters, a second detection by template matching with `Th_learned = 8`, and merging of similar
units. dsp-synapse has matching pursuit and template similarity that these stages can build on.
