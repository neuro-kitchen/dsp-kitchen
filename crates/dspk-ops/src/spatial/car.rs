use cubecl::prelude::*;
use dspk_core::compute::LaunchGeometry;

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

    /// Exact against a sorted reference, in `F` (signed fractional values, so the key's sign
    /// handling and both middle values of an even count are exercised).
    fn medians_of<F: DspFloat + std::fmt::Debug>(client: &Client) {
        for channels in [1usize, 4, 5, 37, 384] {
            let samples = 300;
            let x: Vec<F> = (0..channels * samples).map(|i| crate::core::cast::<F>(((i * 7919) % 211) as f64 * 0.37 - 40.0)).collect();
            let out = buffer::empty::<F>(client, x.len());
            execute_common_median::<F>(client, &buffer::upload(client, &x), &out, channels, samples);
            let got = buffer::download::<F>(client, out);
            let half = crate::core::cast::<F>(0.5);
            for s in 0..samples {
                let mut col: Vec<F> = (0..channels).map(|c| x[c * samples + s]).collect();
                col.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
                let m = if channels % 2 == 1 { col[channels / 2] } else { (col[channels / 2 - 1] + col[channels / 2]) * half };
                for c in 0..channels {
                    assert_eq!(got[c * samples + s], x[c * samples + s] - m, "{channels} channels, sample {s}");
                }
            }
        }
    }

    fn medians(client: &Client) {
        medians_of::<f32>(client);
        // float64 where the runtime has it (not WebGPU)
        if client.properties().supports_type(f64::elem_type_native()) {
            medians_of::<f64>(client);
        }
    }
    runtime_test!(test_common_median_reference, medians);
}
