//! Device kernels for EMUsort channel delay alignment in VRAM.

use cubecl::prelude::*;
use dsp_base::core::buffer;
use dsp_core::compute::{channel_position, sample_position, LaunchGeometry};

/// Circularly shifts each channel's row in VRAM by its specified non-negative shift:
/// `output[ch, t] = input[ch, (t + shift[ch]) mod samples]`.
#[cube(launch)]
pub fn apply_channel_delays_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    shifts: &Array<u32>,
    channels: u32,
    samples: u32,
) {
    let t = sample_position();
    let ch = channel_position();
    if ch < channels && t < samples {
        let shift = shifts[ch as usize];
        let src_t = (t + shift) % samples;
        output[(ch * samples + t) as usize] = input[(ch * samples + src_t) as usize];
    }
}

/// Executes circular channel delay alignment entirely within VRAM on the device.
/// Takes precomputed non-negative circular shifts: `delays.rem_euclid(samples)`.
pub fn execute_apply_channel_delays<R: Runtime>(
    client: &ComputeClient<R>,
    input: &cubecl::server::Handle,
    shifts: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
) -> cubecl::server::Handle {
    let output = buffer::empty::<R, f32>(client, channels * samples);
    let geom = LaunchGeometry::channels_samples(client, channels, samples);
    let total = channels * samples;

    unsafe {
        apply_channel_delays_kernel::launch::<f32, R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(input.clone(), total),
            ArrayArg::from_raw_parts(output.clone(), total),
            ArrayArg::from_raw_parts(shifts.clone(), channels),
            channels as u32,
            samples as u32,
        );
    }
    output
}
