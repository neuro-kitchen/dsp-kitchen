# Kilosort4: Step-by-Step Pipeline & Implementation Guide

This guide walks through each stage of the Kilosort4 spike-sorting pipeline as implemented across the `dsp-kitchen` ecosystem, providing side-by-side Python (`dsp_kitchen`) and Rust (`dsp-synapse`, `dsp-synapse-ml`) code examples.

---

## Pipeline Overview

1. **Probe Ingestion & Channel Mapping**: Filter out synchronization/auxiliary lines and establish physical electrode coordinates.
2. **Median Common Average Referencing (CAR)**: Subtract channel medians across time to remove widespread non-biological noise.
3. **Temporal Bandpass Filtering**: Zero-phase IIR forward-backward filtering (300–6000 Hz).
4. **Local 32-NN ZCA Spatial Whitening**: Eliminate inter-channel spatial correlations across nearest neighbors.
5. **Universal Matched-Filter Detection**: Convolve against 6 universal templates (`wTEMP.npy`) to compute energy envelopes.
6. **Localized Spatial Deduplication**: Prune redundant crossings across adjacent contacts within a $35\,\mu\text{m}$ radius.
7. **Sinc-Realigned Snippet Extraction**: Extract fractional sub-sample realigned waveforms.
8. **Temporal Basis Projection**: Project snippets onto Kilosort4's 6-PC temporal basis (`wPCA.npy`).
9. **Probe-Tiled Unit Clustering**: Cluster spikes in local probe depth/channel spaces into single units.
10. **Phy2 & NWB Units Export**: Persist templates, spike assignments, and quality metrics.

---

## Step 1: Probe Ingestion & Channel Mapping

Isolate active recording sites from digital sync/auxiliary channels and map sensor coordinates.

```py
import numpy as np
import dsp_kitchen.synapse as syn

# Load Neuropixels 1.0 probe layout (384 channels, one shank, staggered 20 µm pitch)
probe = syn.neuropixels_1_0_layout()
print(f"Configured probe: {probe.name} ({probe.total_channels()} recording sites)")
```

```rs
use dsp_core::layout::SensorLayout;
use dsp_synapse::probe::neuropixels_1_0;

let probe: SensorLayout = neuropixels_1_0();
assert_eq!(probe.total_channels(), 384);
println!("Initialized probe: {} with {} sites", probe.name, probe.contacts.len());
```

---

## Step 2: Median Common Average Referencing (CAR)

Remove common-mode electrical noise and motion artifacts prior to temporal filtering.

```py
import dsp_kitchen as dk
from dsp_kitchen.pipeline import Pipeline
from dsp_kitchen.spatial import CommonAverageReference

# raw_voltage shape: [channels, samples] in microvolts
car_pipe = Pipeline([CommonAverageReference()])
car_data = car_pipe.run(raw_voltage, fs=30000.0)
```

```rs
use dsp_base::pipeline::Pipeline;
use dsp_base::spatial::CommonAverageReference;

let mut car = CommonAverageReference::new();
// data: &mut [f32] of shape [channels, samples]
car.apply_in_place(&mut data, channels, samples);
```

---

## Step 3: Zero-Phase Temporal Bandpass Filtering

Filter signals between 300 Hz and 6000 Hz using a 4th-order forward-backward Butterworth filter.

```py
from dsp_kitchen.filter.iir import BandpassFilter

bp_pipe = Pipeline([
    BandpassFilter(low_hz=300.0, high_hz=6000.0, order=4, direction="forward-backward")
])
filtered_data = bp_pipe.run(car_data, fs=30000.0)
```

```rs
use dsp_base::filter::iir::{SosFilter, SosFilterDesign};

let sos = SosFilterDesign::butterworth_bandpass(300.0, 6000.0, 30_000.0, 4)?;
let mut filter = SosFilter::new(sos);
// Apply zero-phase forward-backward filtering across channels
filter.filter_forward_backward_multichannel(&mut data, channels, samples)?;
```

---

## Step 4: Local 32-NN ZCA Spatial Whitening

Decorrelate background activity across each electrode's 32 nearest spatial neighbors.

```py
from dsp_kitchen.spatial import SpatialWhitening

# Fit local ZCA whitening matrix on first 2 seconds of noise/data
whitener = SpatialWhitening.fit_local_knn(
    filtered_data[:, :60000],
    positions=probe_positions,  # list of [x, y] in µm
    k_neighbors=32,
    epsilon=1e-5,
)
whitened_data = Pipeline([whitener]).run(filtered_data, fs=30000.0)
```

```rs
use dsp_base::spatial::SpatialWhitening;

let whitener = SpatialWhitening::fit_local_knn(
    &filtered_data[..channels * 60_000],
    channels,
    60_000,
    &channel_positions_xy,
    32,
    1e-5,
);
let mut whitened = filtered_data.clone();
whitener.apply_in_place(&mut whitened, channels, samples);
```

---

## Step 5: Universal Matched-Filter Spike Detection

Convolve traces with Kilosort4's pretrained universal templates (`wTEMP.npy`) to identify candidate spikes.

```py
from dsp_kitchen.synapse.ml import Kilosort4Detector

# Threshold set in units of MAD noise floor (Quiroga sigma)
detector = Kilosort4Detector.from_hub(threshold_sigma=5.5, refractory_samples=30)
raw_events = detector.detect(whitened_data, sample_rate_hz=30000.0)
print(f"Detected {len(raw_events):,} candidate crossings")
```

```rs
use dsp_synapse::SpikeDetector;
use dsp_synapse_ml::models::kilosort4::Kilosort4TemplateMatcher;

let matcher = Kilosort4TemplateMatcher::from_hub(5.5, 30, None)?;
let events = matcher.detect(&whitened_data, channels, samples, 30_000.0)?;
println!("Extracted {} candidate crossings via wTEMP", events.len());
```

---

## Step 6: Localized Spatial Deduplication

Eliminate duplicate detections of the same physical spike across nearby contacts ($35\,\mu\text{m}$ radius).

```py
# Prune crossings within 35 µm and 0.8 ms (24 samples at 30 kHz)
dedup_spikes = syn.deduplicate_spikes(
    raw_events,
    probe,
    radius_um=35.0,
    window_samples=24,
)
print(f"Retained {len(dedup_spikes):,} unique events")
```

```rs
use dsp_synapse::detection::deduplicate_spikes_spatial;

let dedup_spikes = deduplicate_spikes_spatial(
    &events,
    &probe,
    35.0, // radius in µm
    24,   // window in samples
);
println!("Retained {} spatially deduplicated spikes", dedup_spikes.len());
```

---

## Step 7: Sinc-Realigned Snippet Extraction & wPCA Projection

Extract 61-sample snippets with continuous sub-sample alignment and project onto temporal principal components.

```py
from dsp_kitchen.synapse.ml import Kilosort4BasisEmbedder

# Extract multi-channel snippets centered at sample 20 (pre=20, post=41)
snippets = syn.extract_snippets(
    whitened_data,
    dedup_spikes,
    probe,
    k_neighbors=8,
    pre_samples=20,
    post_samples=41,
    apply_sinc_shift=True,
)

# Project onto 6-component Kilosort4 temporal basis (wPCA.npy)
embedder = Kilosort4BasisEmbedder.from_hub()
waveforms = np.stack([s.waveform() for s in snippets], axis=0).astype(np.float32)
features = embedder.embed(waveforms)  # [N_spikes, K * 6]
```

```rs
use dsp_synapse::extraction::extract_snippet_batch_multichannel;
use dsp_synapse_ml::models::kilosort4::Kilosort4BasisEmbedder;

let snippet_batch = extract_snippet_batch_multichannel(
    &whitened_data,
    channels,
    samples,
    &dedup_spikes,
    &probe,
    8,  // k_neighbors
    20, // pre_samples
    41, // post_samples
    true, // apply_sinc_shift
);

let embedder = Kilosort4BasisEmbedder::from_hub(None)?;
let (features, n_components) = embedder.project_waveforms(
    snippet_batch.waveforms(),
    snippet_batch.num_snippets(),
    61,
)?;
```

---

## Step 8: Probe-Tiled Unit Clustering

Form units locally across probe channels rather than imposing a single global cluster cap.

```py
# Group spikes by primary recording contact
spikes_by_ch = {}
for s in dedup_spikes:
    spikes_by_ch.setdefault(s.primary_channel, []).append(s)

units = []
for ch in range(probe.total_channels()):
    spks = spikes_by_ch.get(ch, [])
    if len(spks) < 30:
        continue
    amps = np.array([abs(s.peak_amplitude_uv) for s in spks])
    med_amp = float(np.median(amps))
    # Detect multi-unit bimodality
    high_mask = amps > med_amp * 1.5
    if np.sum(high_mask) >= 35 and np.sum(~high_mask) >= 35:
        units.append({'primary_ch': ch, 'spikes': [s for i, s in enumerate(spks) if not high_mask[i]]})
        units.append({'primary_ch': ch, 'spikes': [s for i, s in enumerate(spks) if high_mask[i]]})
    else:
        units.append({'primary_ch': ch, 'spikes': spks})

print(f"Resolved {len(units)} units across probe channels")
```

```rs
use std::collections::BTreeMap;
use dsp_synapse::core::{DeduplicatedSpike, SortedUnit};

let mut spikes_by_ch: BTreeMap<usize, Vec<DeduplicatedSpike>> = BTreeMap::new();
for spk in dedup_spikes {
    spikes_by_ch.entry(spk.primary_channel).or_default().push(spk);
}

let mut resolved_units: Vec<SortedUnit> = Vec::new();
for (ch, spks) in spikes_by_ch {
    if spks.len() < 30 { continue; }
    let times: Vec<u64> = spks.iter().map(|s| s.sample_index).collect();
    let amps: Vec<f32> = spks.iter().map(|s| s.peak_amplitude_uv).collect();
    let unit = SortedUnit::from_spikes(
        resolved_units.len(),
        ch,
        times,
        amps,
        Vec::new(),
        None,
        30_000.0,
        samples as u64,
        10.0,
    );
    resolved_units.push(unit);
}
```

---

## Step 9: Multi-Channel Template Accumulation & Export

Accumulate full-probe templates $\mathbf{T} \in \mathbb{R}^{N_{\text{units}} \times 61 \times C}$ and export to Phy2.

```py
# Build SortingOutput container
sorting = dk.SortingOutput.from_clusters(
    sorter_name="kilosort4",
    spike_samples=all_spike_times,
    labels=all_spike_clusters,
    sample_rate_hz=30000.0,
    total_samples=samples,
    probe=probe,
    snippets=snippets,
)

# Export all Phy2 / Kilosort files
sorting.export_to_phy("data/kilosort4/saved_results_dsp_synapse")
print("Phy2 export complete!")
```

```rs
use dsp_synapse::core::SortingOutput;
use dsp_synapse::storage::save_phy_dataset;
use std::path::Path;

let sorting = SortingOutput::new(
    "kilosort4",
    30_000.0,
    samples as u64,
    Some(probe),
    resolved_units,
    None,
);

save_phy_dataset(&sorting, Path::new("data/kilosort4/saved_results_dsp_synapse"))?;
println!("Successfully saved Phy2 dataset");
```
