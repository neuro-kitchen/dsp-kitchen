use cubecl::prelude::*;
use crate::geometry::LaunchGeometry;
use super::biquad::BiquadCoeffs;
use super::kernels::notch_filter_kernel;

/// Designs a 2nd-order digital notch filter for electrical interference rejection (e.g. 50 Hz or 60 Hz).
pub fn design_notch_coeffs(notch_freq_hz: f64, sample_rate_hz: f64, quality_factor: f64) -> BiquadCoeffs {
    let w0 = 2.0 * std::f64::consts::PI * (notch_freq_hz / sample_rate_hz);
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * quality_factor);

    let b0 = 1.0;
    let b1 = -2.0 * cos_w0;
    let b2 = 1.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    BiquadCoeffs {
        b0: (b0 / a0) as f32,
        b1: (b1 / a0) as f32,
        b2: (b2 / a0) as f32,
        a1: (a1 / a0) as f32,
        a2: (a2 / a0) as f32,
    }
}

/// High-level host dispatcher for 50/60 Hz notch filtering.
pub fn execute_notch<R: Runtime>(
    client: &ComputeClient<R>,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    coeffs: BiquadCoeffs,
    channels: usize,
    samples: usize,
    is_cpu: bool,
) {
    let geom = LaunchGeometry::for_channel_sequence(channels, is_cpu);
    let total_elements = channels * samples;

    unsafe {
        notch_filter_kernel::launch::<R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(input.clone(), total_elements),
            ArrayArg::from_raw_parts(output.clone(), total_elements),
            coeffs.b0,
            coeffs.b1,
            coeffs.b2,
            coeffs.a1,
            coeffs.a2,
            channels as u32,
            samples as u32,
        );
    }
}
