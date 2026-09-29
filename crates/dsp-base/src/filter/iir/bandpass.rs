use cubecl::prelude::*;
use super::biquad::{BiquadCoeffs, execute_cascaded_biquad_4th};

/// Cascaded 4th-order Bandpass Filter Coefficients (High-pass Section + Low-pass Section).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandpassCoeffs {
    pub hp_sec: BiquadCoeffs,
    pub lp_sec: BiquadCoeffs,
    pub low_cutoff_hz: f64,
    pub high_cutoff_hz: f64,
    pub sample_rate_hz: f64,
}

/// Designs a 4th-order Butterworth bandpass filter (2nd-order High-Pass cascaded with 2nd-order Low-Pass).
///
/// Standard for electrophysiology action potential band (e.g. 300 Hz – 6,000 Hz at 30 kHz).
pub fn design_butterworth_bandpass_4th(
    low_cutoff_hz: f64,
    high_cutoff_hz: f64,
    sample_rate_hz: f64,
) -> BandpassCoeffs {
    let sqrt2_over_2 = std::f64::consts::SQRT_2 / 2.0;

    // 1. High-Pass Butterworth section at low_cutoff_hz
    let wl = 2.0 * std::f64::consts::PI * (low_cutoff_hz / sample_rate_hz);
    let cos_wl = wl.cos();
    let sin_wl = wl.sin();
    let alpha_l = sin_wl * sqrt2_over_2;

    let hp_a0 = 1.0 + alpha_l;
    let hp_b0 = (1.0 + cos_wl) / (2.0 * hp_a0);
    let hp_b1 = -(1.0 + cos_wl) / hp_a0;
    let hp_b2 = hp_b0;
    let hp_a1 = (-2.0 * cos_wl) / hp_a0;
    let hp_a2 = (1.0 - alpha_l) / hp_a0;

    let hp_sec = BiquadCoeffs::new(
        hp_b0 as f32,
        hp_b1 as f32,
        hp_b2 as f32,
        hp_a1 as f32,
        hp_a2 as f32,
    );

    // 2. Low-Pass Butterworth section at high_cutoff_hz
    let wh = 2.0 * std::f64::consts::PI * (high_cutoff_hz / sample_rate_hz);
    let cos_wh = wh.cos();
    let sin_wh = wh.sin();
    let alpha_h = sin_wh * sqrt2_over_2;

    let lp_a0 = 1.0 + alpha_h;
    let lp_b0 = (1.0 - cos_wh) / (2.0 * lp_a0);
    let lp_b1 = (1.0 - cos_wh) / lp_a0;
    let lp_b2 = lp_b0;
    let lp_a1 = (-2.0 * cos_wh) / lp_a0;
    let lp_a2 = (1.0 - alpha_h) / lp_a0;

    let lp_sec = BiquadCoeffs::new(
        lp_b0 as f32,
        lp_b1 as f32,
        lp_b2 as f32,
        lp_a1 as f32,
        lp_a2 as f32,
    );

    BandpassCoeffs {
        hp_sec,
        lp_sec,
        low_cutoff_hz,
        high_cutoff_hz,
        sample_rate_hz,
    }
}

/// Executes a 4th-order cascaded Butterworth bandpass filter across all channels in a single kernel pass.
pub fn execute_bandpass<R: Runtime>(
    client: &ComputeClient<R>,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    coeffs: BandpassCoeffs,
    channels: usize,
    samples: usize,
    is_cpu: bool,
) {
    execute_cascaded_biquad_4th::<R>(
        client,
        input,
        output,
        coeffs.hp_sec,
        coeffs.lp_sec,
        channels,
        samples,
        is_cpu,
    );
}
