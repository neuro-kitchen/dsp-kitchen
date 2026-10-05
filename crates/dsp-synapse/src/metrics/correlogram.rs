//! Fast $O(N)$ Auto-Correlogram (ACG) & Cross-Correlogram (CCG) (`acg_ccg.rs`).
//!
//! Uses a two-pointer sliding window over sorted spike timestamps to build symmetric
//! lag histograms over $[-W, +W]$ without $O(N^2)$ pairwise comparisons.

use serde::{Deserialize, Serialize};

/// Symmetric spike train lag histogram over `[-window_ms, +window_ms]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Correlogram {
    /// Bin center lags in milliseconds (`len == 2 * half_bins + 1`).
    pub bin_centers_ms: Vec<f32>,
    /// Spike pair counts per lag bin.
    pub counts: Vec<u64>,
    pub bin_size_ms: f32,
    pub window_ms: f32,
}

/// Computes the symmetric Auto-Correlogram (ACG) of a sorted spike train (in sample indices),
/// excluding the trivial zero-lag self-coincidence ($i = j$).
///
/// Counts are over **ordered** pairs $(i, j)$, $i \ne j$, at lag $t_j - t_i$ (as SpikeInterface's
/// correlograms): every unordered pair adds one count at $+\Delta t$ and one at $-\Delta t$, so two
/// distinct spikes in the same sample add 2 to the centre bin, and the ACG equals the CCG of the
/// train with itself minus the $i = j$ terms.
pub fn compute_autocorrelogram(
    sorted_samples: &[u64],
    sample_rate_hz: f64,
    bin_size_ms: f32,
    window_ms: f32,
) -> Correlogram {
    let bin_ms = bin_size_ms.max(0.05);
    let half_bins = ((window_ms.max(bin_ms) / bin_ms).round() as usize).max(1);
    let num_bins = 2 * half_bins + 1;
    let max_lag_ms = (half_bins as f32 + 0.5) * bin_ms;

    let mut bin_centers_ms = Vec::with_capacity(num_bins);
    for b in 0..num_bins {
        bin_centers_ms.push(((b as isize - half_bins as isize) as f32) * bin_ms);
    }

    let mut counts = vec![0u64; num_bins];
    let n = sorted_samples.len();
    if n < 2 || sample_rate_hz <= 0.0 {
        return Correlogram {
            bin_centers_ms,
            counts,
            bin_size_ms: bin_ms,
            window_ms: (half_bins as f32) * bin_ms,
        };
    }

    let ms_per_sample = 1000.0 / sample_rate_hz;

    for i in 0..n {
        let t_i = sorted_samples[i];
        for j in (i + 1)..n {
            let dt_ms = ((sorted_samples[j].saturating_sub(t_i)) as f64 * ms_per_sample) as f32;
            if dt_ms >= max_lag_ms {
                break;
            }
            let offset_bins = ((dt_ms / bin_ms).round() as usize).min(half_bins);
            counts[half_bins + offset_bins] += 1;
            counts[half_bins - offset_bins] += 1;
        }
    }

    Correlogram {
        bin_centers_ms,
        counts,
        bin_size_ms: bin_ms,
        window_ms: (half_bins as f32) * bin_ms,
    }
}

/// Computes the Cross-Correlogram (CCG) between reference train `samples_a` and target train `samples_b`
/// (lags $t_b - t_a \in [-W, +W]$).
pub fn compute_crosscorrelogram(
    sorted_samples_a: &[u64],
    sorted_samples_b: &[u64],
    sample_rate_hz: f64,
    bin_size_ms: f32,
    window_ms: f32,
) -> Correlogram {
    let bin_ms = bin_size_ms.max(0.05);
    let half_bins = ((window_ms.max(bin_ms) / bin_ms).round() as usize).max(1);
    let num_bins = 2 * half_bins + 1;
    let max_lag_ms = (half_bins as f32 + 0.5) * bin_ms;

    let mut bin_centers_ms = Vec::with_capacity(num_bins);
    for b in 0..num_bins {
        bin_centers_ms.push(((b as isize - half_bins as isize) as f32) * bin_ms);
    }

    let mut counts = vec![0u64; num_bins];
    if sorted_samples_a.is_empty() || sorted_samples_b.is_empty() || sample_rate_hz <= 0.0 {
        return Correlogram {
            bin_centers_ms,
            counts,
            bin_size_ms: bin_ms,
            window_ms: (half_bins as f32) * bin_ms,
        };
    }

    let ms_per_sample = 1000.0 / sample_rate_hz;
    let mut left_b = 0usize;
    let n_b = sorted_samples_b.len();

    for &t_a in sorted_samples_a {
        while left_b < n_b {
            let dt_ms = ((sorted_samples_b[left_b] as f64 - t_a as f64) * ms_per_sample) as f32;
            if dt_ms < -max_lag_ms {
                left_b += 1;
            } else {
                break;
            }
        }

        for j in left_b..n_b {
            let dt_ms = ((sorted_samples_b[j] as f64 - t_a as f64) * ms_per_sample) as f32;
            if dt_ms >= max_lag_ms {
                break;
            }
            let signed_bin = (dt_ms / bin_ms).round() as isize;
            let bin_idx = (half_bins as isize + signed_bin).clamp(0, (num_bins - 1) as isize) as usize;
            counts[bin_idx] += 1;
        }
    }

    Correlogram {
        bin_centers_ms,
        counts,
        bin_size_ms: bin_ms,
        window_ms: (half_bins as f32) * bin_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acg_and_ccg_refractory_dip_and_lag_peak() {
        // 1000 Hz sample rate -> 1 sample = 1 ms
        // Train A fires every 10 ms (refractory dip for |lag| < 10 ms)
        let train_a: Vec<u64> = (0..20).map(|i| i * 10).collect();
        let acg = compute_autocorrelogram(&train_a, 1000.0, 1.0, 25.0);
        // Zero counts in (-9 ms .. +9 ms)
        let center = acg.counts.len() / 2;
        for lag in 0..9 {
            assert_eq!(acg.counts[center + lag], 0);
            assert_eq!(acg.counts[center - lag], 0);
        }
        assert!(acg.counts[center + 10] > 0);

        // Train B fires exactly 3 ms after Train A (monosynaptic excitation lag = +3 ms)
        let train_b: Vec<u64> = train_a.iter().map(|&t| t + 3).collect();
        let ccg = compute_crosscorrelogram(&train_a, &train_b, 1000.0, 1.0, 25.0);
        assert_eq!(ccg.counts[center + 3], 20);
    }

    #[test]
    fn acg_counts_ordered_pairs_and_matches_self_ccg() {
        let train = [100u64, 100, 130, 400, 415, 1_000];
        let acg = compute_autocorrelogram(&train, 30_000.0, 0.5, 5.0);
        let ccg = compute_crosscorrelogram(&train, &train, 30_000.0, 0.5, 5.0);
        let centre = acg.counts.len() / 2;
        // Self-CCG has the 6 i = j terms in the centre bin; the coincident pair (100, 100) adds 2.
        assert_eq!(ccg.counts[centre] - acg.counts[centre], 6);
        assert_eq!(acg.counts[centre], 2);
        let mut rest = ccg.counts.clone();
        rest[centre] -= 6;
        assert_eq!(rest, acg.counts);
        // Symmetric
        assert!(acg.counts.iter().eq(acg.counts.iter().rev()));
    }
}
