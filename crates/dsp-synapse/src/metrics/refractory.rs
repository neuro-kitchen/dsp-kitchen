//! Refractory auto- and cross-correlograms (Kilosort4 paper, Methods, *Refractory auto- and
//! cross-correlograms*).
//!
//! The correlogram is binned in 1 ms bins over ±0.5 s. With `n_k` the coincidences in the central
//! `−k..=k` bins and `R` the baseline coincidences per bin (the larger of the left and right
//! shoulders, as correlograms may be asymmetric):
//!
//! - `R12 = min_k n_k / ((2k + 1) · R)`: the central count against what the baseline predicts;
//! - `Q12 = min_k P_k`, `P_k = ½ (1 + erf((n_k − λ_k) / √(ε + 2 λ_k)))`, `λ_k = (2k + 1) · R`,
//!   `ε = 10⁻¹⁰`: the (Gaussian-approximated) Poisson probability of seeing `n_k` or fewer.
//!
//! A cross-correlogram is refractory when `R12 < 0.25` and `Q12 < 0.05` (the two trains never fire
//! within a few ms of each other: one neuron); an auto-correlogram when `R12 < 0.2` and `Q12 < 0.2`.
//! The paper's Methods give 0.1 for auto-correlograms, but Kilosort4 runs with `acg_threshold =
//! 0.2` (its saved `ops` of the test recording; its docs: *good* is < 20% contamination), as does
//! the paper's own benchmark ("refractory violations had a rate < 0.2").
//!
//! The central `ignored_central_bins` bins (the zero-lag bin, ±0.5 ms) are left out of `n_k` and
//! `λ_k`, as Kilosort4 does ("normally ignores the central 1 ms of the correlogram", its docs on
//! `duplicate_spike_ms`): matching pursuit and duplicate removal empty that bin for every unit, so
//! counting it would make every unit look refractory.
//!
//! The paper does not give the range of `k` or where the shoulders start: here `k` runs up to
//! [`RefractoryOptions::max_central_bins`] (10 ms) and the shoulders are the bins beyond
//! [`RefractoryOptions::shoulder_start_ms`] (250 ms, the outer half of the window). Both are settings.
//!
//! Computed on the host: the spike times are on the host when these tests run (clustering
//! decisions, merges, labels), and a correlogram costs one pass over the spikes of the two trains
//! with the ±0.5 s neighbours of each (milliseconds for typical units).

use super::correlogram::{compute_autocorrelogram, compute_crosscorrelogram, Correlogram};

/// `ε` of the paper's Gaussian approximation (keeps the denominator positive when `λ_k = 0`).
const POISSON_EPSILON: f64 = 1e-10;

/// Settings of the refractory tests; defaults are the paper's where it gives them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefractoryOptions {
    /// Correlogram bin (ms).
    pub bin_ms: f32,
    /// Half window of the correlogram (ms).
    pub window_ms: f32,
    /// Largest `k` (bins on each side of zero) tested (not given in the paper).
    pub max_central_bins: usize,
    /// Bins at the centre left out of `n_k` (1: the zero-lag bin), as Kilosort4.
    pub ignored_central_bins: usize,
    /// Bins at least this far from zero (ms) form the shoulders (not given in the paper).
    pub shoulder_start_ms: f32,
    /// Cross-correlogram thresholds: refractory when `R12 <` ratio and `Q12 <` probability.
    pub ccg_ratio: f64,
    pub ccg_probability: f64,
    /// Auto-correlogram thresholds.
    pub acg_ratio: f64,
    pub acg_probability: f64,
}

impl Default for RefractoryOptions {
    fn default() -> Self {
        Self {
            bin_ms: 1.0,
            window_ms: 500.0,
            max_central_bins: 10,
            ignored_central_bins: 1,
            shoulder_start_ms: 250.0,
            ccg_ratio: 0.25,
            ccg_probability: 0.05,
            acg_ratio: 0.2,
            acg_probability: 0.2,
        }
    }
}

/// The paper's two statistics of a correlogram.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefractoryStats {
    /// `R12`: smallest ratio of central coincidences to their baseline expectation (∞ without a
    /// baseline).
    pub ratio: f64,
    /// `Q12`: smallest probability of so few central coincidences under the baseline rate (1
    /// without a baseline).
    pub probability: f64,
    /// `R`: baseline coincidences per bin.
    pub baseline: f64,
}

impl RefractoryStats {
    /// Whether both statistics are below `ratio` and `probability`.
    pub fn is_refractory(&self, ratio: f64, probability: f64) -> bool {
        self.ratio < ratio && self.probability < probability
    }
}

/// `R12`, `Q12` and `R` of a correlogram with `counts[half + i]` at lag `i` bins.
pub fn refractory_stats(correlogram: &Correlogram, options: &RefractoryOptions) -> RefractoryStats {
    let counts = &correlogram.counts;
    let half = counts.len() / 2;
    let shoulder = ((options.shoulder_start_ms / correlogram.bin_size_ms).round() as usize).clamp(1, half);
    let mean = |bins: &[u64]| if bins.is_empty() { 0.0 } else { bins.iter().sum::<u64>() as f64 / bins.len() as f64 };
    let left = mean(&counts[..=half - shoulder]);
    let right = mean(&counts[half + shoulder..]);
    let baseline = left.max(right);
    if baseline <= 0.0 {
        return RefractoryStats { ratio: f64::INFINITY, probability: 1.0, baseline };
    }
    let mut ratio = f64::INFINITY;
    let mut probability = 1.0f64;
    // Bins |lag| < `skip` are left out; `n_k` sums the bins skip..=k on both sides (and the centre
    // when nothing is skipped)
    let skip = options.ignored_central_bins;
    let mut n_k = 0.0f64;
    let mut bins = 0usize;
    for k in skip..=options.max_central_bins.min(half.saturating_sub(1)) {
        if k == 0 {
            n_k += counts[half] as f64;
            bins += 1;
        } else {
            n_k += (counts[half - k] + counts[half + k]) as f64;
            bins += 2;
        }
        let lambda = bins as f64 * baseline;
        ratio = ratio.min(n_k / lambda);
        let p = 0.5 * (1.0 + erf((n_k - lambda) / (POISSON_EPSILON + 2.0 * lambda).sqrt()));
        probability = probability.min(p);
    }
    RefractoryStats { ratio, probability, baseline }
}

/// Refractory statistics of the cross-correlogram of two sorted spike trains (samples), and whether
/// it is refractory (the trains look like one neuron).
pub fn ccg_refractory(sorted_a: &[u64], sorted_b: &[u64], sample_rate_hz: f64, options: &RefractoryOptions) -> (RefractoryStats, bool) {
    let ccg = compute_crosscorrelogram(sorted_a, sorted_b, sample_rate_hz, options.bin_ms, options.window_ms);
    let stats = refractory_stats(&ccg, options);
    (stats, stats.is_refractory(options.ccg_ratio, options.ccg_probability))
}

/// Refractory statistics of the auto-correlogram of a sorted spike train (samples), and whether it
/// is refractory (a well-isolated unit).
pub fn acg_refractory(sorted: &[u64], sample_rate_hz: f64, options: &RefractoryOptions) -> (RefractoryStats, bool) {
    let acg = compute_autocorrelogram(sorted, sample_rate_hz, options.bin_ms, options.window_ms);
    let stats = refractory_stats(&acg, options);
    (stats, stats.is_refractory(options.acg_ratio, options.acg_probability))
}

/// Error function (Abramowitz & Stegun 7.1.26, |error| < 1.5 · 10⁻⁷).
fn erf(x: f64) -> f64 {
    const P: f64 = 0.327_591_1;
    const A: [f64; 5] = [0.254_829_592, -0.284_496_736, 1.421_413_741, -1.453_152_027, 1.061_405_429];
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let poly = A.iter().rev().fold(0.0, |acc, &a| acc * t + a) * t;
    sign * (1.0 - poly * (-x * x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 30_000.0;
    /// 10 minutes.
    const DURATION: u64 = 600 * 30_000;

    /// Deterministic pseudo-random generator (splitmix64).
    struct Rng(u64);
    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// A Poisson train at `rate_hz`, with no spike within `dead_ms` of the previous one.
    fn train(seed: u64, rate_hz: f64, dead_ms: f64) -> Vec<u64> {
        let mut rng = Rng(seed);
        let dead = dead_ms * FS / 1000.0;
        let mut t = 0.0f64;
        let mut out = Vec::new();
        loop {
            t += dead - (1.0 - rng.uniform()).ln() * FS / rate_hz;
            if t >= DURATION as f64 {
                return out;
            }
            out.push(t as u64);
        }
    }

    fn merge(a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut m: Vec<u64> = a.iter().chain(b).copied().collect();
        m.sort_unstable();
        m
    }

    #[test]
    fn erf_matches_known_values() {
        for (x, e) in [(0.0, 0.0), (0.5, 0.520_499_877_8), (1.0, 0.842_700_792_9), (-2.0, -0.995_322_265)] {
            assert!((erf(x) - e).abs() < 2e-7, "erf({x})");
        }
    }

    #[test]
    fn poisson_train_is_not_refractory() {
        let t = train(1, 10.0, 0.0);
        let (stats, refractory) = acg_refractory(&t, FS, &RefractoryOptions::default());
        assert!(!refractory, "{stats:?}");
        assert!(stats.ratio > 0.5, "{stats:?}");
    }

    /// Duplicate removal empties the zero-lag bin of any unit: that alone must not make a Poisson
    /// train refractory.
    #[test]
    fn an_empty_zero_lag_bin_alone_is_not_refractory() {
        let t = train(6, 30.0, 0.5);
        let (stats, refractory) = acg_refractory(&t, FS, &RefractoryOptions::default());
        assert!(!refractory, "{stats:?}");
        let counting_centre = RefractoryOptions { ignored_central_bins: 0, ..Default::default() };
        assert!(acg_refractory(&t, FS, &counting_centre).1, "with the centre counted it looks refractory");
    }

    #[test]
    fn train_with_dead_time_is_refractory() {
        let t = train(2, 10.0, 3.0);
        let (stats, refractory) = acg_refractory(&t, FS, &RefractoryOptions::default());
        assert!(refractory, "{stats:?}");
    }

    #[test]
    fn halves_of_one_neuron_have_a_refractory_ccg() {
        let t = train(3, 20.0, 3.0);
        let (a, b): (Vec<u64>, Vec<u64>) = t.iter().enumerate().fold((Vec::new(), Vec::new()), |(mut a, mut b), (i, &s)| {
            if i % 2 == 0 { a.push(s) } else { b.push(s) }
            (a, b)
        });
        let (stats, refractory) = ccg_refractory(&a, &b, FS, &RefractoryOptions::default());
        assert!(refractory, "{stats:?}");
    }

    #[test]
    fn independent_neurons_have_a_flat_ccg() {
        let (a, b) = (train(4, 10.0, 3.0), train(5, 10.0, 3.0));
        let (stats, refractory) = ccg_refractory(&a, &b, FS, &RefractoryOptions::default());
        assert!(!refractory, "{stats:?}");
        // And their union (two neurons in one cluster) is not a refractory unit
        let (stats, refractory) = acg_refractory(&merge(&a, &b), FS, &RefractoryOptions::default());
        assert!(!refractory, "{stats:?}");
    }

    #[test]
    fn empty_or_sparse_trains_are_not_refractory() {
        let options = RefractoryOptions::default();
        assert!(!acg_refractory(&[], FS, &options).1);
        assert!(!ccg_refractory(&[100], &[], FS, &options).1);
    }
}
