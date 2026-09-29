/// Computes the robust estimate of the background noise standard deviation (Quiroga et al., 2004):
/// `sigma_n = median(|x|) / 0.6745`
pub fn estimate_noise_std(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }

    let mut abs_vals: Vec<f32> = signal.iter().map(|&x| x.abs()).collect();
    let mid = abs_vals.len() / 2;
    // Partial sort to find median efficiently in O(N) time
    abs_vals.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = abs_vals[mid];

    median / 0.6745f32
}
