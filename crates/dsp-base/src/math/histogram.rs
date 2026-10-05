//! Order statistics and histograms of plain slices: percentiles (selection, no full sort) and
//! fixed-bin counts. Shared by renderers (auto-scale, DC offset) and spike-train metrics.

/// The `q`-th percentile (`0..=100`, nearest rank) of `values`, reordering them in place.
/// `None` when empty. NaN values must be removed by the caller.
pub fn percentile(values: &mut [f32], q: f64) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let at = (((values.len() - 1) as f64) * (q.clamp(0.0, 100.0) / 100.0)).round() as usize;
    Some(*values.select_nth_unstable_by(at, f32::total_cmp).1)
}

/// Counts of `values` in `bins` equal bins over `[lo, hi)` (the value `hi` itself goes in the
/// last bin); values outside are not counted.
pub fn histogram(values: impl IntoIterator<Item = f64>, lo: f64, hi: f64, bins: usize) -> Vec<u64> {
    let mut counts = vec![0u64; bins];
    if bins == 0 || hi.is_nan() || lo.is_nan() || hi <= lo {
        return counts;
    }
    let scale = bins as f64 / (hi - lo);
    for v in values {
        if v < lo || v > hi || v.is_nan() {
            continue;
        }
        let b = (((v - lo) * scale) as usize).min(bins - 1);
        counts[b] += 1;
    }
    counts
}

/// Centres of `bins` equal bins over `[lo, hi)`.
pub fn bin_centers(lo: f64, hi: f64, bins: usize) -> Vec<f64> {
    let w = (hi - lo) / bins.max(1) as f64;
    (0..bins).map(|b| lo + (b as f64 + 0.5) * w).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile() {
        let mut v = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(percentile(&mut v, 0.0), Some(1.0));
        assert_eq!(percentile(&mut v, 50.0), Some(3.0));
        assert_eq!(percentile(&mut v, 100.0), Some(5.0));
        assert_eq!(percentile(&mut [], 50.0), None);
    }

    #[test]
    fn test_histogram_edges() {
        let h = histogram([0.0, 0.5, 0.99, 1.0, 2.0, 3.0, -1.0], 0.0, 3.0, 3);
        assert_eq!(h, vec![3, 1, 2], "3.0 lands in the last bin, -1 is dropped");
        assert_eq!(bin_centers(0.0, 3.0, 3), vec![0.5, 1.5, 2.5]);
        assert_eq!(histogram([1.0], 1.0, 1.0, 4), vec![0; 4]);
    }
}
