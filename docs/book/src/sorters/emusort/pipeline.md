# EMUsort: pipeline

EMUsort runs the [Kilosort4 pipeline](../kilosort4/pipeline.md) with the changes below. Only the
differences are described here.

## 1. Preprocessing

As Kilosort4, **without** the common average reference (`do_car = false`). The whole front end
(preprocessing, delays, templates, detection) runs over a recording with
`EmusortRunner::new(config).run(client, source, probe)` (or in Python with `emusort.run(recording, probe, config)`).

## 2. Channel-delay removal — *implemented*

1. **Estimate** (over every `nskip`-th preprocessed batch): each channel is divided by its standard
   deviation and rectified (`|x|`). For every pair of channels `a, b` and every lag within
   ±`fs / 500` samples (2 ms), `mean_t x_a[t − lag] · x_b[t]` is accumulated over the batch's
   unpadded samples.
2. **Reference**: the channel whose best-lag correlations with all channels sum highest.
3. **Delays**: for each channel, the lag of its best correlation with the reference.
4. **Remove** from every batch, after whitening: `x[i, t] ← x[i, t + delay_i]` (a circular shift
   within the padded batch, so only padding wraps).

```rust,ignore
use dsp_synapse_ml::sorters::emusort::{apply_channel_delays, ChannelDelayEstimator, EmusortConfig};

let cfg = EmusortConfig::default();
let mut est = ChannelDelayEstimator::new(channels, cfg.max_delay_samples(fs));
for batch in batches.iter().step_by(cfg.kilosort4.nskip) {
    est.add_batch(batch, padded_samples, cfg.kilosort4.nt);
}
let (delays, reference) = est.delays();
for batch in &mut batches {
    apply_channel_delays(batch, padded_samples, &delays);
}
```

## 3. Universal templates — *implemented*

As Kilosort4, with:

- clips pooled over **every threshold** of `Th_single_ch = [6, 9, 12, 15]` (a peak found at several
  thresholds counts once);
- `n_pcs = 9`, `n_templates = 9`, `nskip = 2`;
- **HDBSCAN outlier removal** before k-means: clips labelled noise by HDBSCAN
  (`min_cluster_size = 20`, Euclidean, excess-of-mass selection) are dropped; skipped when there
  are fewer than `max(20, min_cluster_size)` clips. `wPCA` is learned from all clips.

```rust,ignore
let templates = learn_universal_templates(&client, &clips, cfg.kilosort4.nt, &cfg.learn_options())?;
```

`dsp_synapse::sorting::hdbscan` is exact but `O(n²·d)` on the host: fine for tens of thousands of
clips, slow near the 500 000-clip cap.

## 4. Detection and later stages

Universal-template detection is Kilosort4's, with EMUsort's templates and settings
(`detect_universal`). Drift, clustering, deconvolution and merging are not implemented yet (as in
Kilosort4).
