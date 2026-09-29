/// Computes the signal-to-noise ratio (SNR) of an action potential against background noise:
/// `SNR = |peak_amplitude| / noise_std`
pub fn compute_snr(peak_amplitude_uv: f32, noise_std_uv: f32) -> f32 {
    if noise_std_uv <= 1e-6 {
        0.0
    } else {
        peak_amplitude_uv.abs() / noise_std_uv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snr_calc() {
        assert_eq!(compute_snr(-100.0, 20.0), 5.0);
    }
}
