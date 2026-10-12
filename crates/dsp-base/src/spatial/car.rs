use cubecl::prelude::*;
use dsp_core::compute::LaunchGeometry;

use crate::core::DspFloat;
pub use super::kernels::{common_median_kernel, direct_car_kernel};

/// High-level host dispatcher for direct, single-pass Common Average Referencing (CAR).
/// Computes and subtracts the common average directly in VRAM without auxiliary buffers.
pub fn execute_direct_car<F: DspFloat>(
    client: &Client,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
) {
    let geom = LaunchGeometry::per_sample(client, samples);
    let total_elements = channels * samples;

    unsafe {
        direct_car_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(input.clone(), total_elements),
            BufferArg::from_raw_parts(output.clone(), total_elements),
            channels as u32,
            samples as u32,
        );
    }
}

/// Common median reference: subtracts the median over channels at every sample (SpikeInterface's
/// `common_reference(operator="median")`), one cube per sample on the device.
pub fn execute_common_median<F: DspFloat>(
    client: &Client,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
) {
    if channels == 0 || samples == 0 {
        return;
    }
    let geom = LaunchGeometry::per_row(client, samples, channels);
    let total = channels * samples;
    // SAFETY: both buffers hold `channels · samples` values
    unsafe {
        common_median_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim.clone(),
            BufferArg::from_raw_parts(input.clone(), total),
            BufferArg::from_raw_parts(output.clone(), total),
            channels as u32,
            samples as u32,
            channels as u32,
            geom.cube_dim.x,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::buffer;

    fn medians(client: &Client) {
        for channels in [1usize, 4, 5, 37] {
            let samples = 300;
            let x: Vec<f32> = (0..channels * samples).map(|i| ((i * 7919) % 211) as f32 - 100.0).collect();
            let out = buffer::empty::<f32>(client, x.len());
            execute_common_median::<f32>(client, &buffer::upload(client, &x), &out, channels, samples);
            let got = buffer::download::<f32>(client, out);
            for s in 0..samples {
                let mut col: Vec<f32> = (0..channels).map(|c| x[c * samples + s]).collect();
                col.sort_by(f32::total_cmp);
                let m = if channels % 2 == 1 { col[channels / 2] } else { 0.5 * (col[channels / 2 - 1] + col[channels / 2]) };
                for c in 0..channels {
                    assert_eq!(got[c * samples + s], x[c * samples + s] - m, "{channels} channels, sample {s}");
                }
            }
        }
    }
    runtime_test!(test_common_median_reference, medians);
}
