//! EMUsort (O'Connell et al., openRxiv 2026): Kilosort4 adapted to motor-unit action potentials
//! from high-density intramuscular arrays (Myomatrix). Written from the paper and the published
//! defaults — the upstream GPL-3.0 code is not ported.
//!
//! EMUsort is a Kilosort4 fork, so it reuses every stage of [`crate::sorters::kilosort4`] with
//! its own settings ([`EmusortConfig`]) and adds:
//! - **channel-delay removal** ([`ChannelDelayEstimator`], [`apply_channel_delays`]): the lag
//!   (within ±2 ms) aligning each channel with the reference channel that correlates best with
//!   all others, removed before detection;
//! - **HDBSCAN outlier removal** before the universal templates are clustered
//!   ([`EmusortConfig::learn_options`]).

use crate::provenance::{Attributed, Paper, Provenance, ProvenanceKind, UpstreamCode};
use crate::sorters::kilosort4::{ClipOptions, Kilosort4Config, LearnOptions};

/// Largest channel delay searched: `fs / MAX_DELAY_DIVISOR` samples (2 ms).
pub const MAX_DELAY_DIVISOR: f64 = 500.0;
/// HDBSCAN `min_cluster_size` of outlier removal (`hdbscan_min_cluster_size`).
pub const HDBSCAN_MIN_CLUSTER_SIZE: usize = 20;

/// EMUsort settings: Kilosort4's, with the paper's changes.
#[derive(Debug, Clone, PartialEq)]
pub struct EmusortConfig {
    pub kilosort4: Kilosort4Config,
    /// Estimate and remove per-channel delays (`remove_chan_delays`).
    pub remove_channel_delays: bool,
    /// HDBSCAN outlier removal before k-means of the universal templates (`remove_spike_outliers`).
    pub remove_spike_outliers: bool,
    pub hdbscan_min_cluster_size: usize,
}

impl Default for EmusortConfig {
    fn default() -> Self {
        Self {
            kilosort4: Kilosort4Config {
                th_single_ch: vec![6.0, 9.0, 12.0, 15.0],
                n_pcs: 9,
                n_templates: 9,
                nskip: 2,
                do_car: false,
                ..Kilosort4Config::default()
            },
            remove_channel_delays: true,
            remove_spike_outliers: true,
            hdbscan_min_cluster_size: HDBSCAN_MIN_CLUSTER_SIZE,
        }
    }
}

impl EmusortConfig {
    pub fn clip_options(&self) -> ClipOptions {
        self.kilosort4.clip_options()
    }

    pub fn learn_options(&self) -> LearnOptions {
        LearnOptions {
            outlier_min_cluster_size: self.remove_spike_outliers.then_some(self.hdbscan_min_cluster_size),
            ..self.kilosort4.learn_options()
        }
    }

    /// Largest delay (samples) at `sample_rate_hz`.
    pub fn max_delay_samples(&self, sample_rate_hz: f64) -> usize {
        (sample_rate_hz / MAX_DELAY_DIVISOR).floor() as usize
    }
}

/// Accumulates the lagged cross-correlations of channel envelopes over batches: each channel of
/// a batch is divided by its standard deviation and rectified; `CC[a, b, lag] += mean_t
/// x_a[t − lag] · x_b[t]` over the batch's unpadded samples.
#[derive(Debug, Clone)]
pub struct ChannelDelayEstimator {
    channels: usize,
    max_lag: usize,
    /// `[channels, channels, 2·max_lag + 1]`.
    cc: Vec<f64>,
    batches: usize,
}

impl ChannelDelayEstimator {
    pub fn new(channels: usize, max_lag: usize) -> Self {
        Self { channels, max_lag, cc: vec![0.0; channels * channels * (2 * max_lag + 1)], batches: 0 }
    }

    /// Adds a `[channels, samples]` batch with `pad ≥ max_lag` samples of context on each side.
    pub fn add_batch(&mut self, x: &[f32], samples: usize, pad: usize) {
        let (c, l) = (self.channels, self.max_lag);
        assert_eq!(x.len(), c * samples, "batch size mismatch");
        assert!(pad >= l && samples > 2 * pad, "padding must cover the largest delay");
        let env: Vec<f64> = (0..c)
            .flat_map(|ch| {
                let row = &x[ch * samples..(ch + 1) * samples];
                let mean = row.iter().map(|&v| v as f64).sum::<f64>() / samples as f64;
                let sd = (row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / samples as f64).sqrt();
                let inv = if sd > 0.0 { 1.0 / sd } else { 0.0 };
                row.iter().map(move |&v| (v as f64 * inv).abs()).collect::<Vec<_>>()
            })
            .collect();
        let inner = samples - 2 * pad;
        let n_lags = 2 * l + 1;
        for a in 0..c {
            for b in 0..c {
                for (li, lag) in (-(l as isize)..=l as isize).enumerate() {
                    let mut s = 0.0;
                    for t in pad..samples - pad {
                        s += env[a * samples + (t as isize - lag) as usize] * env[b * samples + t];
                    }
                    self.cc[(a * c + b) * n_lags + li] += s / inner as f64;
                }
            }
        }
        self.batches += 1;
    }

    /// `(delays, reference)`: the reference channel maximizes `Σ_a max_lag CC[a, b, ·]`; each
    /// channel's delay is the lag of its best correlation with the reference. Zeros before any
    /// batch.
    pub fn delays(&self) -> (Vec<isize>, usize) {
        let (c, l) = (self.channels, self.max_lag);
        let n_lags = 2 * l + 1;
        if self.batches == 0 || c == 0 {
            return (vec![0; c], 0);
        }
        let peak = |a: usize, b: usize| {
            let row = &self.cc[(a * c + b) * n_lags..(a * c + b + 1) * n_lags];
            row.iter().enumerate().fold((0usize, f64::NEG_INFINITY), |best, (i, &v)| if v > best.1 { (i, v) } else { best })
        };
        let reference = (0..c)
            .map(|b| (b, (0..c).map(|a| peak(a, b).1).sum::<f64>()))
            .fold((0usize, f64::NEG_INFINITY), |best, x| if x.1 > best.1 { x } else { best })
            .0;
        let delays = (0..c).map(|b| peak(reference, b).0 as isize - l as isize).collect();
        (delays, reference)
    }
}

/// Removes `delays` from a `[channels, samples]` batch: `x[i, t] ← x[i, (t + delay_i) mod
/// samples]` (a circular shift within the batch, as upstream; with `max_lag` samples of padding
/// only the padding wraps).
pub fn apply_channel_delays(x: &mut [f32], samples: usize, delays: &[isize]) {
    assert_eq!(x.len(), delays.len() * samples, "batch size mismatch");
    for (row, &d) in x.chunks_exact_mut(samples).zip(delays) {
        let shift = d.rem_euclid(samples as isize) as usize;
        row.rotate_left(shift);
    }
}

/// Paper and code of EMUsort.
pub fn emusort_provenance() -> Provenance {
    Provenance {
        name: "EMUsort".into(),
        kind: ProvenanceKind::ReimplementedFromPaper,
        paper: Some(Paper {
            title: "High performance sorting of motor unit action potentials with EMUsort".into(),
            authors: ["O’Connell", "Michaels", "Wang", "Mamidipaka", "Venkatesh", "Aresh", "Pachitariu", "Pruszynski", "Sober", "Pandarinath"]
                .map(String::from)
                .to_vec(),
            venue: "openRxiv (bioRxiv)".into(),
            year: 2026,
            doi: "10.64898/2026.01.06.697952".into(),
            license: Some("CC-BY-4.0".into()),
        }),
        code: UpstreamCode { url: "https://github.com/snel-repo/EMUsort".into(), license: Some("GPL-3.0".into()), version: "a06bb60 (Kilosort4 fork snel-repo/Kilosort4, base v4.0.18)".into() },
        artifacts: Vec::new(),
        notes: "Written from the paper and the published defaults; GPL code not ported. Universal templates are always learned from the recording (no EMUsort artifacts exist). Implemented: channel-delay removal, HDBSCAN outlier removal, and the Kilosort4 stages of crate::sorters::kilosort4.".into(),
    }
}

/// EMUsort stages configured by an [`EmusortConfig`].
#[derive(Debug, Clone, Default)]
pub struct Emusort {
    pub config: EmusortConfig,
}

impl Attributed for Emusort {
    fn provenance(&self) -> Provenance {
        emusort_provenance()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_and_removes_channel_delays() {
        let (channels, samples, pad, max_lag) = (3usize, 2_000usize, 61usize, 20usize);
        // Bursts on channel 0; channel 1 delayed by +5, channel 2 by −3
        let burst = |t: isize| -> f32 { if (t % 200).abs() < 4 { 10.0 } else { 0.1 * ((t * 7919) % 13) as f32 } };
        let mut x = vec![0.0f32; channels * samples];
        for t in 0..samples as isize {
            x[t as usize] = burst(t);
            x[samples + t as usize] = burst(t - 5);
            x[2 * samples + t as usize] = burst(t + 3);
        }
        let mut est = ChannelDelayEstimator::new(channels, max_lag);
        est.add_batch(&x, samples, pad);
        let (delays, _) = est.delays();
        assert_eq!(delays[1] - delays[0], 5);
        assert_eq!(delays[2] - delays[0], -3);
        apply_channel_delays(&mut x, samples, &delays);
        for t in (pad..samples - pad).step_by(200) {
            assert!((x[t] - x[samples + t]).abs() < 1e-6 && (x[t] - x[2 * samples + t]).abs() < 1e-6, "aligned at {t}");
        }
    }

    #[test]
    fn defaults_follow_the_paper() {
        let c = EmusortConfig::default();
        assert_eq!((c.kilosort4.n_pcs, c.kilosort4.n_templates, c.kilosort4.nskip), (9, 9, 2));
        assert_eq!(c.kilosort4.th_single_ch, vec![6.0, 9.0, 12.0, 15.0]);
        assert!(!c.kilosort4.do_car && c.remove_channel_delays && c.remove_spike_outliers);
        assert_eq!(c.learn_options().outlier_min_cluster_size, Some(20));
    }
}
