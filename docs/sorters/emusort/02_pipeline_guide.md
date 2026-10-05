# Myomatrix & EMUsort: Step-by-Step Pipeline & Implementation Guide

This guide walks through each stage of the Myomatrix / EMUsort motor unit spike sorting pipeline as implemented in the `dsp-kitchen` ecosystem, providing side-by-side Python (`dsp_kitchen`) and Rust (`dsp-synapse`, `dsp-synapse-ml`) code examples.

---

## Pipeline Overview

1. **Array Configuration**: Setup probe geometry (4×8 32-channel grid, 8-channel intramuscular thread, or 8×8 array).
2. **EMG Preprocessing**: Bandpass filter (100–2000 Hz) + CAR (or Longitudinal Differential Bipolar).
3. **150-Sample Universal MUAP Detection**: Convolve against canonical MUAP templates (`wTEMP_EMG`).
4. **Spatial Deduplication**: Enforce muscle-specific refractory constraints ($\ge 2.5\,\text{ms}$) across contact pitch ($1000\text{--}4000\,\mu\text{m}$).
5. **Multi-Contact Conduction Latency Alignment**: Correct for muscle fiber propagation delays ($v \approx 2\text{--}6\,\text{m/s}$).
6. **Extended Snippet Extraction ($T=150$)**: Extract 150-sample waveform windows with sinc realignment.
7. **12-Component Muscle Basis Projection**: Project snippets onto `wPCA_EMG` to retain polyphasic dynamics.
8. **Motor Unit Clustering (cBSS / Density Peaks / GMM)**: Deconvolve distinct motor units.
9. **Pulse-to-Noise Ratio (PNR) & Phy2 Export**: Compute PNR metrics and output Phy2-compliant sorting files.

---

## Step 1: Array Configuration & Channel Mapping

Define the Myomatrix electrode grid with proper inter-electrode distance (IED).

```py
import dsp_kitchen.synapse as syn

# Configure standard 32-channel 4x8 planar Myomatrix array with 4.0 mm pitch
probe = syn.hdemg_4x8_layout(ied_mm=4.0)
print(f"Loaded layout: {probe.name} ({probe.total_channels()} electrodes)")
```

```rs
use dsp_core::layout::SensorLayout;
use dsp_synapse::probe::hdemg_4x8;

let probe: SensorLayout = hdemg_4x8(4000.0); // 4000 µm pitch
assert_eq!(probe.total_channels(), 32);
println!("Configured Myomatrix 4x8 array: {} contacts", probe.total_channels());
```

---

## Step 2: EMG Temporal Filtering & Referencing

Filter signals between 100 Hz and 2000 Hz and apply Common Average Reference (CAR).

```py
from dsp_kitchen.pipeline import Pipeline
from dsp_kitchen.filter.iir import BandpassFilter
from dsp_kitchen.spatial import CommonAverageReference

pipe = Pipeline([
    BandpassFilter(low_hz=100.0, high_hz=2000.0, order=4, direction="forward-backward"),
    CommonAverageReference(),
])
filtered_emg = pipe.run(raw_emg, fs=30000.0)
```

```rs
use dsp_base::filter::iir::{SosFilter, SosFilterDesign};
use dsp_base::spatial::CommonAverageReference;

let sos = SosFilterDesign::butterworth_bandpass(100.0, 2000.0, 30_000.0, 4)?;
let mut filter = SosFilter::new(sos);
filter.filter_forward_backward_multichannel(&mut emg_data, channels, samples)?;

let mut car = CommonAverageReference::new();
car.apply_in_place(&mut emg_data, channels, samples);
```

---

## Step 3: 150-Sample Universal MUAP Matched Filtering

Convolve multi-channel EMG traces with 150-sample universal MUAP templates (`wTEMP_EMG`).

```py
from dsp_kitchen.synapse.ml import EmusortDetector

# Detect MUAP crossings with 6.5 sigma threshold and 2.5 ms refractory window
detector = EmusortDetector.from_hub(threshold_sigma=6.5, refractory_samples=75)
muap_events = detector.detect(filtered_emg, sample_rate_hz=30000.0)
print(f"Detected {len(muap_events):,} candidate MUAP crossings")
```

```rs
use dsp_synapse::SpikeDetector;
use dsp_synapse_ml::models::emusort::EmusortTemplateMatcher;

let matcher = EmusortTemplateMatcher::from_canonical(6.5, 75, None)?;
let events = matcher.detect(&filtered_emg, channels, samples, 30_000.0)?;
println!("Extracted {} candidate MUAP events", events.len());
```

---

## Step 4: Spatial Deduplication with Motor Unit Refractory Constraints

Deduplicate events using muscle physiological constraints (refractory period $\ge 2.5\,\text{ms}$, spatial radius $\approx 4000\,\mu\text{m}$).

```py
# Deduplicate across 4 mm radius and 2.5 ms window (75 samples at 30 kHz)
dedup_muaps = syn.deduplicate_spikes(
    muap_events,
    probe,
    radius_um=4000.0,
    window_samples=75,
)
print(f"Retained {len(dedup_muaps):,} deduplicated MUAPs")
```

```rs
use dsp_synapse::detection::deduplicate_spikes_spatial;

let dedup_muaps = deduplicate_spikes_spatial(
    &events,
    &probe,
    4000.0, // radius in µm
    75,     // window in samples (2.5 ms at 30 kHz)
);
println!("Retained {} spatially deduplicated MUAPs", dedup_muaps.len());
```

---

## Step 5: Multi-Contact Conduction Latency Alignment

Measure propagation velocity along the muscle fiber direction and align waveforms.

```py
from dsp_kitchen.synapse.ml import EmusortLatencyAligner

aligner = EmusortLatencyAligner(max_lag_samples=25)
for i in range(waveforms.shape[0]):
    lags = aligner.estimate_channel_lags(waveforms[i], ref_ch=0)
    waveforms[i] = aligner.align_snippet(waveforms[i], lags)
```

```rs
use dsp_synapse_ml::models::emusort::EmusortLatencyAligner;

let aligner = EmusortLatencyAligner::new(25);
let lags = aligner.estimate_channel_lags(&snippet, channels, window_len, 0);
let aligned = aligner.align_snippet(&snippet, channels, window_len, &lags);
```

---

## Step 6: Extended Snippet Extraction ($T=150$) & 12-PC Basis Embedding

Extract 150-sample snippets (pre=70, post=80) and project onto the 12-component muscle PCA basis (`wPCA_EMG`).

```py
from dsp_kitchen.synapse.ml import EmusortBasisEmbedder

# Extract 150-sample snippets across nearest channels
snippets = syn.extract_snippets(
    filtered_emg,
    dedup_muaps,
    probe,
    k_neighbors=8,
    pre_samples=70,
    post_samples=80,
    apply_sinc_shift=True,
)

# Project onto 12-component muscle basis
embedder = EmusortBasisEmbedder.from_hub()
waveforms = np.stack([s.waveform() for s in snippets], axis=0).astype(np.float32)
features = embedder.embed(waveforms)  # [N_muaps, K * 12]
```

```rs
use dsp_synapse::FeatureEmbedder;
use dsp_synapse::extraction::extract_snippet_batch_multichannel;
use dsp_synapse_ml::models::emusort::EmusortBasisEmbedder;

let snippet_batch = extract_snippet_batch_multichannel(
    &filtered_emg,
    channels,
    samples,
    &dedup_muaps,
    &probe,
    8,   // k_neighbors
    70,  // pre_samples
    80,  // post_samples
    true, // apply_sinc_shift
);

let embedder = EmusortBasisEmbedder::from_canonical(None)?;
let (features, embed_dim) = embedder.embed(&snippet_batch)?;
```

---

## Step 7: Motor Unit Pulse Trains & PNR Curation

Cluster feature vectors, compute the Pulse-to-Noise Ratio (PNR), and export to Phy2.

```py
import dsp_kitchen as dk

# Cluster features into motor units via GMM / contact-localized density peaks
cluster_res = dk.cluster_gmm(features, min_clusters=3, max_clusters=12, covariance_type="diagonal")
labels = cluster_res["labels"]

# Build SortingOutput container
sorting = dk.SortingOutput.from_clusters(
    sorter_name="emusort",
    spike_samples=[int(s.center_sample) for s in snippets],
    labels=labels,
    sample_rate_hz=30000.0,
    total_samples=samples,
    probe=probe,
    snippets=snippets,
)

# Export to Phy2 folder
sorting.export_to_phy("data/sorters/phy_emusort_output")
print("Exported EMUsort sorting to Phy2!")
```

```rs
use dsp_synapse::core::SortingOutput;
use dsp_synapse::sorting::GmmClusterer;
use dsp_synapse::storage::save_phy_dataset;
use std::path::Path;

let mut clusterer = GmmClusterer::new(3, 12, dsp_synapse::sorting::GmmCovarianceKind::Diagonal);
let res = clusterer.fit(&features, num_muaps, num_features, None);

let sorting = SortingOutput::new(
    "emusort",
    30_000.0,
    samples as u64,
    Some(probe),
    units,
    None,
);
save_phy_dataset(&sorting, Path::new("data/sorters/phy_emusort_output"))?;
```
