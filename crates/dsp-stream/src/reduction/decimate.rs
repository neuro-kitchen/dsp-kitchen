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

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for &val in &input_samples[start..end] {
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }

        output.push(min_val);
        output.push(max_val);
    }

    output
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
}
