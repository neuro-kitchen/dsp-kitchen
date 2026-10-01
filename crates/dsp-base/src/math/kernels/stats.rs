use cubecl::prelude::*;

/// CubeCL per-channel Welford online mean and sample variance reduction kernel.
/// Dispatched with [`dsp_core::compute::LaunchGeometry::per_channel`].
#[cube(launch)]
pub fn channel_mean_variance_kernel(
    input: &Array<f32>,
    out_mean: &mut Array<f32>,
    out_std: &mut Array<f32>,
    num_channels: u32,
    num_samples: u32,
) {
    let ch = ABSOLUTE_POS_X;
    if ch < num_channels {
        if num_samples == 0u32 {
            out_mean[ch as usize] = 0.0f32;
            out_std[ch as usize] = 0.0f32;
        } else {
            let offset = ch * num_samples;
            let mut mean = 0.0f32;
            let mut m2 = 0.0f32;
            let mut t = 0u32;

            while t < num_samples {
                let x = input[(offset + t) as usize];
                let count = (t + 1u32) as f32;
                let delta = x - mean;
                mean = mean + delta / count;
                let delta2 = x - mean;
                m2 = m2 + delta * delta2;
                t = t + 1u32;
            }

            out_mean[ch as usize] = mean;
            let var = m2 / (num_samples as f32);
            out_std[ch as usize] = f32::sqrt(var);
        }
    }
}
