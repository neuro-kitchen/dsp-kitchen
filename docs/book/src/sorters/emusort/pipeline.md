# EMUsort: pipeline

EMUsort runs the [Kilosort4 pipeline](../kilosort4/pipeline.md) with the changes below. Only the
differences are described here.

## 1. Preprocessing

As Kilosort4, **without** the common average reference (`do_car = false`). The whole run
(preprocessing, delays, templates, detection) is Kilosort4's runner with EMUsort's plan:
`Emusort::new(config).run(client, source, probe)` (`run_plan` with `RunPlan::emusort`), or in
Python `emusort.run(recording, probe, config)`; it returns a `Kilosort4Result` with
`channel_delays` set. The halos also cover the largest delay.

## 2. Channel-delay removal — *implemented, on the device*

Semantics checked against upstream (`snel-repo/EMUsort` `a06bb60`, `ks4mods/preprocessing.py`
`get_channel_delays`, `ks4mods/io.py`); the code is not ported.

1. **Estimate** in the fit pass, on the same high-passed data as the whitening (before whitening,
   as upstream), over every `nskip`-th window except the last: each channel is divided by its
   standard deviation over the padded window and rectified (`|x|`). For every pair of channels
   `a, b` and every lag within ±`fs / 500` samples (2 ms), `mean_t x_a[t − lag] · x_b[t]` over the
   window's interior is accumulated on the device (`kernels::ChannelDelayEstimator`, one unit per
   channel pair and lag adding to a running sum; scratch buffers kept between windows); lagged
   reads past the window repeat its edge sample, as upstream pads its first and last batches. One
   download at the end. In Python, `emusort.estimate_channel_delays(batches, pad=, max_lag=)`
   takes all batches in one call for the same reason.
2. **Reference**: the channel whose best-lag correlations with all channels sum highest.
3. **Delays**: for each channel, the lag of its best correlation with the reference
   (`kernels::delays_from_cross_correlation`).
4. **Remove** from every window, after whitening, on the device (`kernels::ChannelAligner`):
   `x[i, t] ← x[i, t + delay_i]` (a circular shift within the padded window, so only padding
   wraps). Template learning and detection both use it. Spike times stay in this aligned frame
   (the reference channel's time), as upstream.

```rust,ignore
use dsp_synapse_ml::sorters::emusort::kernels::{ChannelAligner, ChannelDelayEstimator};

let mut est = ChannelDelayEstimator::new(&client, channels, cfg.max_delay_samples(fs));
est.add(&filtered_window, window.read_len(), window.valid_local.clone()); // per fit window
let (delays, reference) = est.delays();
let mut aligner = ChannelAligner::new(&client, delays, schedule.max_read_samples());
let aligned = aligner.align(&whitened_window, window.read_len()); // stays on the device
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

`dsp_synapse::sorting::hdbscan_points` is exact and runs on the device on the clips already
uploaded for `wPCA`: `O(n²·d)` per Borůvka round (~`log₂ n` rounds), so tens of thousands of clips
take seconds while the 500 000-clip cap is still slow. Inliers are gathered on the device for
k-means.

## 4. Detection and later stages

Universal-template detection is Kilosort4's (`UniversalDetector`), on the delay-aligned windows
with EMUsort's templates and settings. Drift, clustering, deconvolution and merging are not
implemented yet (as in Kilosort4).
