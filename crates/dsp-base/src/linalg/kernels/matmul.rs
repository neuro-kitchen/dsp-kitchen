use cubecl::prelude::*;
use dsp_core::compute::{channel_position, sample_position};

/// Projection `Y = Wᵀ (X − mean)`:
/// - `input_x`: `[channels, samples]`
/// - `weights_w`: `[channels, components]`
/// - `mean`: `[channels]`
/// - `output_y`: `[components, samples]`
#[cube(launch)]
pub fn pca_project_kernel<F: Float>(
    input_x: &Array<F>,
    weights_w: &Array<F>,
    mean: &Array<F>,
    output_y: &mut Array<F>,
    num_channels: u32,
    num_samples: u32,
    num_components: u32,
) {
    let sample_idx = sample_position();
    let comp_idx = channel_position();

    if sample_idx < num_samples && comp_idx < num_components {
        let mut sum = F::new(0.0f32);
        let mut c = 0u32;
        while c < num_channels {
            let centered = input_x[(c * num_samples + sample_idx) as usize] - mean[c as usize];
            sum += centered * weights_w[(c * num_components + comp_idx) as usize];
            c += 1u32;
        }
        output_y[(comp_idx * num_samples + sample_idx) as usize] = sum;
    }
}
