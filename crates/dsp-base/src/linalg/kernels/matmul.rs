use cubecl::prelude::*;

/// CubeCL kernel for matrix projection: `Y = W^T * (X - mean)`.
/// - `input_x`: Flat 2D matrix [channels, samples]
/// - `weights_w`: Flat matrix of top-k principal components [channels, k]
/// - `mean`: Channel mean vector [channels]
/// - `output_y`: Flat projection matrix [k, samples]
#[cube(launch)]
pub fn pca_project_kernel(
    input_x: &Array<f32>,
    weights_w: &Array<f32>,
    mean: &Array<f32>,
    output_y: &mut Array<f32>,
    num_channels: u32,
    num_samples: u32,
    num_components: u32,
) {
    let sample_idx = ABSOLUTE_POS_X;
    let comp_idx = ABSOLUTE_POS_Y;

    if sample_idx < num_samples && comp_idx < num_components {
        let mut sum = 0.0f32;

        for c in 0u32..num_channels {
            let x_val = input_x[(c * num_samples + sample_idx) as usize];
            let mean_val = mean[c as usize];
            let centered = x_val - mean_val;

            let w_val = weights_w[(c * num_components + comp_idx) as usize];
            sum += centered * w_val;
        }

        output_y[(comp_idx * num_samples + sample_idx) as usize] = sum;
    }
}
