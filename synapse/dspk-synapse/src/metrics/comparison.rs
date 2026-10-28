//! Sorter-to-Sorter & Ground-Truth Spike Train Comparison (`comparison.rs`).
//!
//! Implements the SpikeInterface / StudyAnalysis benchmark comparison methodology:
//! 1. **Pairwise Spike Train Matching**: Two-pointer $O(N_A + N_B)$ 1-to-1 spike matching within
//!    tolerance $\pm \Delta t$ samples (`delta_time_ms`), preventing double-counting.
//! 2. **Agreement Score (IoU)**:
//!    $$\text{Agreement}(A, B) = \frac{N_{\text{match}}}{N_A + N_B - N_{\text{match}}}$$
//! 3. **Confusion & Agreement Matrix**: Full $[K_A \times K_B]$ matrix across all unit pairs in
//!    two [`SortingOutput`]s, plus greedy best-match assignment (not Hungarian: a unit can lose a better global pairing) and per-unit
//!    Accuracy, Precision, Recall, False Positive Rate, and False Negative Rate.

use serde::{Deserialize, Serialize};
use crate::core::SortingOutput;

/// Pairwise comparison metrics between a reference spike train $A$ and a tested spike train $B$.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PairwiseTrainMatch {
    pub num_spikes_a: usize,
    pub num_spikes_b: usize,
    /// Number of 1-to-1 matched spikes within `+-tolerance_samples` ($TP$).
    pub num_matches: usize,
    /// Unmatched spikes in $A$ ($FN = N_A - N_{\text{match}}$).
    pub false_negatives: usize,
    /// Unmatched spikes in $B$ ($FP = N_B - N_{\text{match}}$).
    pub false_positives: usize,
    /// Intersection-over-Union agreement score: $N_{\text{match}} / (N_A + N_B - N_{\text{match}})$.
    pub agreement_score: f32,
    /// Precision ($TP / (TP + FP) = N_{\text{match}} / N_B$).
    pub precision: f32,
    /// Recall / Sensitivity ($TP / (TP + FN) = N_{\text{match}} / N_A$).
    pub recall: f32,
    /// Accuracy ($TP / (TP + FN + FP)$ — identical to `agreement_score`).
    pub accuracy: f32,
    /// Miss rate ($FN / N_A = 1 - \text{recall}$).
    pub false_negative_rate: f32,
    /// False discovery rate ($FP / N_B = 1 - \text{precision}$).
    pub false_positive_rate: f32,
}

/// Summary of a matched unit pair `(unit_a, unit_b)` between two sortings.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UnitMatchSummary {
    pub unit_id_a: usize,
    pub unit_id_b: usize,
    pub metrics: PairwiseTrainMatch,
}

/// Full comparison between two [`SortingOutput`] runs (`sorting_a` vs. `sorting_b`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SortingComparison {
    pub sorter_a: String,
    pub sorter_b: String,
    pub unit_ids_a: Vec<usize>,
    pub unit_ids_b: Vec<usize>,
    pub delta_time_ms: f64,
    pub tolerance_samples: u64,
    /// Row-major `[num_units_a, num_units_b]` agreement score matrix in `[0.0, 1.0]`.
    pub agreement_matrix: Vec<f32>,
    /// 1-to-1 matched unit pairs sorted by decreasing agreement score.
    pub matched_units: Vec<UnitMatchSummary>,
    /// Units in `sorting_a` whose best agreement score >= `agreement_threshold`.
    pub well_detected_units_a: Vec<usize>,
    /// Units in `sorting_a` with no match above `agreement_threshold`.
    pub missed_units_a: Vec<usize>,
    /// Units in `sorting_b` with no match above `agreement_threshold`.
    pub false_positive_units_b: Vec<usize>,
    /// Mean agreement score across all `matched_units`.
    pub mean_matched_agreement: f32,
}

/// Matches two spike trains (`train_a` and `train_b`, in sample indices) within `+-tolerance_samples`
/// using a 1-to-1 two-pointer algorithm and computes Agreement, Precision, Recall, FPR, and FNR.
pub fn compare_spike_trains(
    train_a: &[u64],
    train_b: &[u64],
    tolerance_samples: u64,
) -> PairwiseTrainMatch {
    let mut a = train_a.to_vec();
    let mut b = train_b.to_vec();
    if !a.windows(2).all(|w| w[0] <= w[1]) {
        a.sort_unstable();
    }
    if !b.windows(2).all(|w| w[0] <= w[1]) {
        b.sort_unstable();
    }

    let n_a = a.len();
    let n_b = b.len();
    if n_a == 0 && n_b == 0 {
        return PairwiseTrainMatch {
            num_spikes_a: 0,
            num_spikes_b: 0,
            num_matches: 0,
            false_negatives: 0,
            false_positives: 0,
            // Two empty trains: every score is undefined
            agreement_score: f32::NAN,
            precision: f32::NAN,
            recall: f32::NAN,
            accuracy: f32::NAN,
            false_negative_rate: f32::NAN,
            false_positive_rate: f32::NAN,
        };
    }

    let mut i = 0usize;
    let mut j = 0usize;
    let mut matches = 0usize;

    while i < n_a && j < n_b {
        let ta = a[i];
        let tb = b[j];
        let diff = ta.abs_diff(tb);
        if diff <= tolerance_samples {
            matches += 1;
            i += 1;
            j += 1;
        } else if ta < tb {
            i += 1;
        } else {
            j += 1;
        }
    }

    let fn_count = n_a.saturating_sub(matches);
    let fp_count = n_b.saturating_sub(matches);
    let union = (n_a + n_b).saturating_sub(matches);
    let agreement = if union > 0 {
        (matches as f32) / (union as f32)
    } else {
        0.0
    };
    let precision = if n_b > 0 {
        (matches as f32) / (n_b as f32)
    } else {
        0.0
    };
    let recall = if n_a > 0 {
        (matches as f32) / (n_a as f32)
    } else {
        0.0
    };

    PairwiseTrainMatch {
        num_spikes_a: n_a,
        num_spikes_b: n_b,
        num_matches: matches,
        false_negatives: fn_count,
        false_positives: fp_count,
        agreement_score: agreement,
        precision,
        recall,
        accuracy: agreement,
        false_negative_rate: if n_a > 0 { 1.0 - recall } else { 0.0 },
        false_positive_rate: if n_b > 0 { 1.0 - precision } else { 0.0 },
    }
}

/// Compares two [`SortingOutput`] runs (`sorting_a` vs. `sorting_b`) with spike-timing tolerance
/// `delta_time_ms` (typically `0.4` ms) and unit match threshold `agreement_threshold` (typically `0.5`).
pub fn compare_sortings(
    sorting_a: &SortingOutput,
    sorting_b: &SortingOutput,
    delta_time_ms: f64,
    agreement_threshold: f32,
) -> SortingComparison {
    let fs = sorting_a.sample_rate_hz.max(sorting_b.sample_rate_hz).max(1.0);
    let tolerance_samples = ((delta_time_ms.max(0.0) * 1e-3 * fs).round() as u64).max(1);

    let k_a = sorting_a.units.len();
    let k_b = sorting_b.units.len();
    let unit_ids_a: Vec<usize> = sorting_a.units.iter().map(|u| u.unit_id).collect();
    let unit_ids_b: Vec<usize> = sorting_b.units.iter().map(|u| u.unit_id).collect();

    let mut agreement_matrix = vec![0.0f32; k_a * k_b];
    let mut pair_metrics: Vec<(usize, usize, PairwiseTrainMatch)> = Vec::with_capacity(k_a * k_b);

    for (ia, ua) in sorting_a.units.iter().enumerate() {
        for (ib, ub) in sorting_b.units.iter().enumerate() {
            let m = compare_spike_trains(&ua.spike_samples, &ub.spike_samples, tolerance_samples);
            agreement_matrix[ia * k_b + ib] = m.agreement_score;
            pair_metrics.push((ia, ib, m));
        }
    }

    // Greedy maximum-weight bipartite matching in descending order of agreement_score
    pair_metrics.sort_by(|x, y| {
        y.2.agreement_score
            .partial_cmp(&x.2.agreement_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut used_a = vec![false; k_a];
    let mut used_b = vec![false; k_b];
    let mut matched_units = Vec::new();

    for (ia, ib, m) in pair_metrics {
        if m.agreement_score <= 0.0 {
            break;
        }
        if !used_a[ia] && !used_b[ib] {
            used_a[ia] = true;
            used_b[ib] = true;
            matched_units.push(UnitMatchSummary {
                unit_id_a: unit_ids_a[ia],
                unit_id_b: unit_ids_b[ib],
                metrics: m,
            });
        }
    }

    let mut well_detected_units_a = Vec::new();
    let mut missed_units_a = Vec::new();
    for (ia, &uid_a) in unit_ids_a.iter().enumerate() {
        let best_score = (0..k_b)
            .map(|ib| agreement_matrix[ia * k_b + ib])
            .fold(0.0f32, f32::max);
        if best_score >= agreement_threshold {
            well_detected_units_a.push(uid_a);
        } else {
            missed_units_a.push(uid_a);
        }
    }

    let mut false_positive_units_b = Vec::new();
    for (ib, &uid_b) in unit_ids_b.iter().enumerate() {
        let best_score = (0..k_a)
            .map(|ia| agreement_matrix[ia * k_b + ib])
            .fold(0.0f32, f32::max);
        if best_score < agreement_threshold {
            false_positive_units_b.push(uid_b);
        }
    }

    let mean_matched_agreement = if matched_units.is_empty() {
        0.0
    } else {
        matched_units
            .iter()
            .map(|u| u.metrics.agreement_score)
            .sum::<f32>()
            / (matched_units.len() as f32)
    };

    SortingComparison {
        sorter_a: sorting_a.sorter_name.clone(),
        sorter_b: sorting_b.sorter_name.clone(),
        unit_ids_a,
        unit_ids_b,
        delta_time_ms,
        tolerance_samples,
        agreement_matrix,
        matched_units,
        well_detected_units_a,
        missed_units_a,
        false_positive_units_b,
        mean_matched_agreement,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SortedUnit;

    #[test]
    fn test_compare_spike_trains_and_sortings() {
        let fs = 30_000.0;
        let total_samples = 300_000;
        // Unit 0 ground truth: 100 spikes every 2000 samples
        let gt_u0: Vec<u64> = (1..=100).map(|i| i * 2000).collect();
        // Unit 1 ground truth: 80 spikes every 2500 samples
        let gt_u1: Vec<u64> = (1..=80).map(|i| i * 2500 + 300).collect();

        // Sorter B recovers Unit 0 with +-3 sample jitter (within 0.4 ms = 12 samples) as unit_id=10,
        // and recovers 76/80 spikes of Unit 1 as unit_id=20, plus a spurious noise unit_id=99.
        let b_u10: Vec<u64> = gt_u0.iter().map(|&s| s + 3).collect();
        let b_u20: Vec<u64> = gt_u1[..76].iter().map(|&s| s - 2).collect();
        let b_u99: Vec<u64> = vec![500, 15_000, 45_000, 95_000];

        let sorting_gt = SortingOutput::new(
            "ground_truth",
            fs,
            total_samples,
            None,
            vec![
                SortedUnit::from_spikes(0, 0, gt_u0, Vec::new(), Vec::new(), None, fs, total_samples, 5.0),
                SortedUnit::from_spikes(1, 1, gt_u1, Vec::new(), Vec::new(), None, fs, total_samples, 5.0),
            ],
            None,
        );

        let sorting_test = SortingOutput::new(
            "test_sorter",
            fs,
            total_samples,
            None,
            vec![
                SortedUnit::from_spikes(10, 0, b_u10, Vec::new(), Vec::new(), None, fs, total_samples, 5.0),
                SortedUnit::from_spikes(20, 1, b_u20, Vec::new(), Vec::new(), None, fs, total_samples, 5.0),
                SortedUnit::from_spikes(99, 2, b_u99, Vec::new(), Vec::new(), None, fs, total_samples, 5.0),
            ],
            None,
        );

        let cmp = compare_sortings(&sorting_gt, &sorting_test, 0.4, 0.8);
        assert_eq!(cmp.matched_units.len(), 2);
        assert_eq!(cmp.matched_units[0].unit_id_a, 0);
        assert_eq!(cmp.matched_units[0].unit_id_b, 10);
        assert!((cmp.matched_units[0].metrics.agreement_score - 1.0).abs() < 1e-6);

        assert_eq!(cmp.matched_units[1].unit_id_a, 1);
        assert_eq!(cmp.matched_units[1].unit_id_b, 20);
        assert!((cmp.matched_units[1].metrics.recall - 76.0 / 80.0).abs() < 1e-5);
        assert!((cmp.matched_units[1].metrics.precision - 1.0).abs() < 1e-5);

        assert_eq!(cmp.well_detected_units_a, vec![0, 1]);
        assert!(cmp.missed_units_a.is_empty());
        assert_eq!(cmp.false_positive_units_b, vec![99]);
    }
}
