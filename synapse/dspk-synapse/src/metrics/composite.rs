//! EMUsort's composite score of sort quality (O'Connell et al. 2026, eLife 15:RP110417, Methods,
//! *Producing composite scores for agnostic estimation of sort quality*).
//!
//! Each cluster's score is the product of four components in `[0, 1]`, so a single kind of error
//! is enough to reject it:
//!
//! - type I (false positives): `S_T1 = 1 − R_C`, `R_C` the Llobet contamination (1 ms refractory
//!   period);
//! - type II (false negatives): `S_T2 = R_P · (1 − A_C)`, presence ratio (20 s bins, mean-rate
//!   ratio 0.5) and amplitude cutoff (32 bins);
//! - firing-rate validity: `S_FR = 1/(1 + e^(R_F1 − 200)) · 1/(1 + e^(R_F2 − 200))`, mean rate and
//!   firing range (95th − 5th percentile of the rate in 0.5 s bins), Hz;
//! - SNR: `S_SNR = 1 − 1/(1 + e^(SNR − 4))`.
//!
//! The sort's score is the mean over clusters. Choices of ours where the paper is silent: a
//! component that cannot be computed (NaN: too few spikes for the contamination or the amplitude
//! cutoff, a recording shorter than one presence bin) scores 0, the conservative reading of "the
//! presence of a single type of error results in rejection"; components are clamped to `[0, 1]`;
//! percentiles interpolate linearly (as `numpy.percentile`).

use super::firing::{compute_amplitude_cutoff_with, compute_llobet_contamination, compute_presence_ratio, AMPLITUDE_CUTOFF_MIN_RATIO, AMPLITUDE_CUTOFF_SMOOTHING};

/// Settings of the composite score (the paper's values).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompositeScoreOptions {
    /// Refractory and censored periods of the Llobet contamination (ms).
    pub refractory_ms: f64,
    pub censored_ms: f64,
    /// Presence-ratio bin (s) and mean-rate ratio.
    pub presence_bin_s: f64,
    pub presence_mean_fr_ratio: f64,
    /// Bins of the amplitude-cutoff histogram.
    pub amplitude_bins: usize,
    /// Rate (Hz) at which the firing-rate sigmoids reach 0.5.
    pub max_rate_hz: f64,
    /// Firing-range bin (s) and percentiles.
    pub firing_range_bin_s: f64,
    pub firing_range_percentiles: (f64, f64),
    /// SNR at which the SNR sigmoid reaches 0.5.
    pub snr_midpoint: f64,
}

impl Default for CompositeScoreOptions {
    fn default() -> Self {
        Self {
            refractory_ms: 1.0,
            censored_ms: 0.0,
            presence_bin_s: 20.0,
            presence_mean_fr_ratio: 0.5,
            amplitude_bins: 32,
            max_rate_hz: 200.0,
            firing_range_bin_s: 0.5,
            firing_range_percentiles: (5.0, 95.0),
            snr_midpoint: 4.0,
        }
    }
}

/// One cluster's components and their product.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompositeScore {
    pub type_1: f64,
    pub type_2: f64,
    pub firing_rate: f64,
    pub snr: f64,
    /// `type_1 · type_2 · firing_rate · snr`.
    pub score: f64,
}

/// Composite score of a cluster from its spike times (samples), amplitudes and SNR.
pub fn composite_score(
    spike_samples: &[u64],
    amplitudes: &[f32],
    snr: f64,
    total_samples: u64,
    sample_rate_hz: f64,
    opts: &CompositeScoreOptions,
) -> CompositeScore {
    let unit = |v: f64| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
    let logistic_below = |x: f64, mid: f64| 1.0 / (1.0 + (x - mid).exp());

    let contamination = compute_llobet_contamination(spike_samples, total_samples, sample_rate_hz, opts.refractory_ms, opts.censored_ms);
    let type_1 = unit(1.0 - contamination);

    let presence = compute_presence_ratio(spike_samples, total_samples, sample_rate_hz, opts.presence_bin_s, opts.presence_mean_fr_ratio);
    let cutoff = compute_amplitude_cutoff_with(amplitudes, opts.amplitude_bins, AMPLITUDE_CUTOFF_SMOOTHING, AMPLITUDE_CUTOFF_MIN_RATIO);
    let type_2 = unit(unit(presence) * unit(1.0 - cutoff));

    let duration_s = total_samples as f64 / sample_rate_hz;
    let rate = if duration_s > 0.0 { spike_samples.len() as f64 / duration_s } else { f64::NAN };
    let range = firing_range(spike_samples, total_samples, sample_rate_hz, opts.firing_range_bin_s, opts.firing_range_percentiles);
    let firing_rate = if rate.is_finite() && range.is_finite() {
        unit(logistic_below(rate, opts.max_rate_hz) * logistic_below(range, opts.max_rate_hz))
    } else {
        0.0
    };

    let snr = if snr.is_finite() { unit(1.0 - logistic_below(snr, opts.snr_midpoint)) } else { 0.0 };
    CompositeScore { type_1, type_2, firing_rate, snr, score: type_1 * type_2 * firing_rate * snr }
}

/// Mean of the clusters' scores: the sort's composite score (NaN without clusters).
pub fn sort_composite_score(scores: &[CompositeScore]) -> f64 {
    scores.iter().map(|s| s.score).sum::<f64>() / scores.len() as f64
}

/// Spread of a unit's firing rate (Hz): `high − low` percentile of its rate in `bin_s` bins over
/// the recording (SpikeInterface `firing_range`). NaN when the recording is shorter than one bin.
pub fn firing_range(spike_samples: &[u64], total_samples: u64, sample_rate_hz: f64, bin_s: f64, (low, high): (f64, f64)) -> f64 {
    let bin = (bin_s * sample_rate_hz).round() as u64;
    if bin == 0 || total_samples < bin {
        return f64::NAN;
    }
    let bins = (total_samples / bin) as usize;
    let mut counts = vec![0u64; bins];
    for &s in spike_samples {
        if let Some(c) = counts.get_mut((s / bin) as usize) {
            *c += 1;
        }
    }
    let mut rates: Vec<f64> = counts.iter().map(|&c| c as f64 / bin_s).collect();
    rates.sort_by(f64::total_cmp);
    percentile_linear(&rates, high) - percentile_linear(&rates, low)
}

/// `numpy.percentile` (linear interpolation) of sorted `values`.
fn percentile_linear(sorted: &[f64], q: f64) -> f64 {
    let pos = (q / 100.0).clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let (i, frac) = (pos.floor() as usize, pos.fract());
    match sorted.get(i + 1) {
        Some(&next) => sorted[i] + frac * (next - sorted[i]),
        None => sorted[i],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 30_000.0;

    #[test]
    fn percentiles_interpolate_like_numpy() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile_linear(&v, 50.0), 2.5);
        assert!((percentile_linear(&v, 5.0) - 1.15).abs() < 1e-12);
        assert_eq!(percentile_linear(&v, 100.0), 4.0);
    }

    /// A regular 10 Hz unit with Gaussian-like amplitudes well above threshold and SNR 10 scores
    /// near 1; the same unit at SNR 2 scores low; a 300 Hz unit scores ~0.
    #[test]
    fn components_follow_the_paper() {
        let total = 120 * 30_000u64;
        let spikes: Vec<u64> = (0..1200).map(|i| i * 3_000 + 100).collect();
        let amps: Vec<f32> = (0..1200).map(|i| 50.0 + ((i * 37 % 41) as f32 - 20.0) * 0.5).collect();
        let opts = CompositeScoreOptions::default();
        let good = composite_score(&spikes, &amps, 10.0, total, FS, &opts);
        assert_eq!(good.type_1, 1.0, "no refractory violations");
        assert!(good.score > 0.9, "{good:?}");
        let noisy = composite_score(&spikes, &amps, 2.0, total, FS, &opts);
        assert!((noisy.snr - 1.0 / (1.0 + 2f64.exp())).abs() < 1e-12);
        let fast: Vec<u64> = (0..36_000).map(|i| i * 100).collect();
        let fast_amps = vec![50.0f32; fast.len()];
        let fast = composite_score(&fast, &fast_amps, 10.0, total, FS, &opts);
        assert!(fast.firing_rate < 1e-6, "{fast:?}");
    }

    #[test]
    fn too_few_spikes_score_zero() {
        let s = composite_score(&[100], &[50.0], 10.0, 30 * 30_000, FS, &CompositeScoreOptions::default());
        assert_eq!(s.score, 0.0);
    }
}
