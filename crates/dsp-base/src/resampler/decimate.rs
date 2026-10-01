use super::minmax::{fold, EMPTY};

/// Fast min-max envelope decimation for real-time visualization.
/// For each bucket of samples, emits `[min, max]` to preserve high-frequency spike peaks.
pub fn min_max_decimate(
    input_samples: &[f32],
    target_buckets: usize,
) -> Vec<f32> {
    if input_samples.is_empty() || target_buckets == 0 {
        return Vec::new();
    }

    if input_samples.len() <= target_buckets * 2 {
        return input_samples.to_vec();
    }

    let mut output = Vec::with_capacity(target_buckets * 2);
    let bucket_size = input_samples.len() as f64 / target_buckets as f64;

    for i in 0..target_buckets {
        let start = (i as f64 * bucket_size).floor() as usize;
        let end = (((i + 1) as f64 * bucket_size).ceil() as usize).min(input_samples.len());

        let [min_val, max_val] = fold(&input_samples[start..end], EMPTY);
        output.push(min_val);
        output.push(max_val);
    }

    output
}

/// Allocation-free min-max decimation into a caller-owned buffer of `[min, max]` pairs.
///
/// Unlike [`min_max_decimate`], every output bucket is always a `[min, max]` pair, even when
/// there are fewer input samples than buckets (neighbouring buckets then repeat a sample), so
/// renderers can map bucket `i` to screen column `i` without special cases.
/// An empty input fills `output` with `[0.0, 0.0]`.
pub fn min_max_decimate_into(input_samples: &[f32], output: &mut [[f32; 2]]) {
    let n = input_samples.len();
    let buckets = output.len();
    if n == 0 {
        output.fill([0.0, 0.0]);
        return;
    }

    for (i, bucket) in output.iter_mut().enumerate() {
        let start = ((i as u64 * n as u64) / buckets as u64) as usize;
        let end = ((((i + 1) as u64 * n as u64) / buckets as u64) as usize)
            .max(start + 1)
            .min(n);
        let start = start.min(n - 1);
        *bucket = fold(&input_samples[start..end], EMPTY);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_max_preserves_peaks() {
        let mut data = vec![0.0f32; 1000];
        data[250] = -150.0; // Spike trough
        data[255] = 80.0;   // Spike peak

        let decimated = min_max_decimate(&data, 10);
        assert_eq!(decimated.len(), 20); // 10 buckets * 2 (min and max)
        
        let min_overall = decimated.iter().copied().fold(f32::INFINITY, f32::min);
        let max_overall = decimated.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        
        assert_eq!(min_overall, -150.0);
        assert_eq!(max_overall, 80.0);
    }

    #[test]
    fn test_min_max_into_preserves_peaks_and_pairs() {
        let mut data = vec![0.0f32; 1000];
        data[250] = -150.0;
        data[255] = 80.0;

        let mut out = vec![[0.0f32; 2]; 10];
        min_max_decimate_into(&data, &mut out);
        assert_eq!(out[2], [-150.0, 80.0]);
        assert_eq!(out[0], [0.0, 0.0]);
    }

    #[test]
    fn test_min_max_into_fewer_samples_than_buckets() {
        let data = [1.0f32, 2.0, 3.0];
        let mut out = vec![[0.0f32; 2]; 6];
        min_max_decimate_into(&data, &mut out);
        // Each bucket maps to exactly one sample, in order
        assert_eq!(out, vec![[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [3.0, 3.0], [3.0, 3.0]]);

        let mut empty = vec![[9.0f32; 2]; 3];
        min_max_decimate_into(&[], &mut empty);
        assert_eq!(empty, vec![[0.0, 0.0]; 3]);
    }
}
