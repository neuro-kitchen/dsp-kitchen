# Kilosort4: pipeline

The stages below follow the paper and the published defaults. **Implemented** stages show the
dsp-kitchen API; the others are described so the remaining work is clear.

Input to every stage is **preprocessed** data in batches of `batch_size = 60000` samples with
`nt` samples of padding on each side, in whitened units (thresholds are multiples of the noise σ).

## 1. Preprocessing — *stages available, driver not yet*

Common average reference (`do_car`), high-pass at `highpass_cutoff_hz = 300` Hz, and local
whitening over the `whitening_range = 32` nearest channels. The building blocks are dsp-base
pipeline stages (`PipelineStage::CommonAverageReference`, a high-pass `FilterSpec`,
`SpatialWhitening::fit_local_knn`); a Kilosort4 driver that fits and chains them is not written yet.

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

The SVD runs as an eigendecomposition of the `nt × nt` Gram matrix on the device
(dsp-base `linalg`); k-means is `dsp_synapse::sorting::kmeans`.

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
   maximum and exceeds `Th_universal = 9`. Candidates are compacted on the device; only spikes are
   downloaded.
4. **Per spike**: amplitude, template, centre, a vertical position (contact positions weighted by
   the rectified template response), and features — the centre's `nearest_chans` snippets projected
   onto `wPCA` (`nearest_chans × n_pcs`).

```rust,ignore
use dsp_synapse_ml::sorters::kilosort4::{detect_universal, TemplateCentres};

let centres = TemplateCentres::new(&probe, &cfg.centres)?;
let spikes = detect_universal(&client, &batch_handle, channels, padded_samples,
                              &centres, &templates, cfg.th_universal, cfg.nt0min())?;
```

## 4. Drift correction — *not yet*

Kilosort4 estimates vertical drift from the detected spikes' positions and corrects the data
before the second detection. dsp-synapse has rigid / non-rigid drift estimation and kriging
correction that a driver can use; the Kilosort4-specific procedure is not written.

## 5. Clustering, learned templates, deconvolution, merging — *not yet*

Clustering of the features in local probe regions (graph-based), multi-channel templates from the
clusters, a second detection by template matching with `Th_learned = 8`, and merging of similar
units. dsp-synapse has matching pursuit and template similarity that these stages can build on.
