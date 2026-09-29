//! Screen-space Min-Max LOD decimation algorithm.
//!
//! Provides $O(W)$ memory and compute decimation: for any time window containing
//! potentially millions of raw samples, it aggregates data into $W$ screen-column
//! buckets storing only `(min_val, max_val)` per column.

/// A single screen-space column bucket storing the extreme values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LodBucket {
    pub min: f32,
    pub max: f32,
}

impl Default for LodBucket {
    fn default() -> Self {
        Self { min: 0.0, max: 0.0 }
    }
}

/// Decimates a 1D signal slice into `num_buckets` screen-space columns.
///
/// # Arguments
/// * `channel_data` - Full raw sample buffer for a single channel
/// * `start_sample` - First sample index in the visible window
/// * `window_samples` - Total number of samples in the visible window
/// * `num_buckets` - Screen pixel width (number of column buckets)
pub fn decimate_min_max(
    channel_data: &[f32],
    start_sample: usize,
    window_samples: usize,
    num_buckets: usize,
) -> Vec<LodBucket> {
    if num_buckets == 0 || window_samples == 0 || channel_data.is_empty() {
        return vec![LodBucket::default(); num_buckets];
    }

    let total_len = channel_data.len();
    let clamped_start = start_sample.min(total_len);
    let clamped_end = (start_sample + window_samples).min(total_len);
    let actual_samples = clamped_end.saturating_sub(clamped_start);

    if actual_samples == 0 {
        return vec![LodBucket::default(); num_buckets];
    }

    let mut buckets = Vec::with_capacity(num_buckets);

    for col in 0..num_buckets {
        let s0 = clamped_start + ((col as f64 / num_buckets as f64) * actual_samples as f64) as usize;
        let s1 = (clamped_start + (((col + 1) as f64 / num_buckets as f64) * actual_samples as f64) as usize)
            .min(clamped_end)
            .max(s0 + 1);

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        let slice = &channel_data[s0..s1];
        for &v in slice {
            if v < min_val { min_val = v; }
            if v > max_val { max_val = v; }
        }

        if min_val.is_infinite() {
            min_val = 0.0;
            max_val = 0.0;
        }

        buckets.push(LodBucket { min: min_val, max: max_val });
    }

    buckets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decimate_constant_signal() {
        let data = vec![42.0f32; 10_000];
        let buckets = decimate_min_max(&data, 0, 10_000, 100);
        assert_eq!(buckets.len(), 100);
        for b in buckets {
            assert_eq!(b.min, 42.0);
            assert_eq!(b.max, 42.0);
        }
    }

    #[test]
    fn test_decimate_impulse_preservation() {
        // Even if 100,000 samples are decimated to 10 buckets, a single large spike must be captured!
        let mut data = vec![0.0f32; 100_000];
        data[45_000] = -150.0; // Negative action potential spike in bucket 4
        data[45_001] = 80.0;

        let buckets = decimate_min_max(&data, 0, 100_000, 10);
        assert_eq!(buckets.len(), 10);
        assert_eq!(buckets[4].min, -150.0);
        assert_eq!(buckets[4].max, 80.0);
    }

    #[test]
    fn test_decimate_empty_and_out_of_bounds() {
        let data = vec![1.0, 2.0, 3.0];
        let empty = decimate_min_max(&data, 100, 50, 10);
        assert_eq!(empty.len(), 10);
        assert_eq!(empty[0], LodBucket::default());

        let zero_buckets = decimate_min_max(&data, 0, 3, 0);
        assert_eq!(zero_buckets.len(), 0);
    }
}
