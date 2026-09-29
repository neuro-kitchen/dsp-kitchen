use cubecl::prelude::*;
use crate::geometry::LaunchGeometry;
use super::kernels::{biquad_filter_kernel, cascaded_biquad_4th_kernel};

/// Coefficients for a single Second-Order Section (Biquad) IIR filter.
/// Assumes $a_0 = 1.0$ (pre-normalized).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiquadCoeffs {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

impl BiquadCoeffs {
    pub fn new(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Self { b0, b1, b2, a1, a2 }
    }
}

/// Dispatches a single-stage Biquad IIR filter across all channels.
pub fn execute_biquad<R: Runtime>(
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
        biquad_filter_kernel::launch::<R>(
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

/// Dispatches a 4th-order cascaded Biquad (2 SOS sections) in a single kernel pass.
pub fn execute_cascaded_biquad_4th<R: Runtime>(
    client: &ComputeClient<R>,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    sec1: BiquadCoeffs,
    sec2: BiquadCoeffs,
    channels: usize,
    samples: usize,
    is_cpu: bool,
) {
    let geom = LaunchGeometry::for_channel_sequence(channels, is_cpu);
    let total_elements = channels * samples;

    unsafe {
        cascaded_biquad_4th_kernel::launch::<R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(input.clone(), total_elements),
            ArrayArg::from_raw_parts(output.clone(), total_elements),
            sec1.b0,
            sec1.b1,
            sec1.b2,
            sec1.a1,
            sec1.a2,
            sec2.b0,
            sec2.b1,
            sec2.b2,
            sec2.a1,
            sec2.a2,
            channels as u32,
            samples as u32,
        );
    }
}
