use cubecl::prelude::*;
use dsp_core::compute::sample_position;

/// Single-pass Common Average Referencing: one unit per sample ([`sample_position`]) averages that
/// sample over all channels and subtracts the average, with no intermediate buffer. Units of a plane
/// read consecutive samples of one channel at a time (coalesced).
#[cube(launch)]
pub fn direct_car_kernel<F: Float>(input: &[F], output: &mut [F], num_channels: u32, num_samples: u32) {
    let sample_idx = sample_position();

    if sample_idx < num_samples {
        let mut sum = F::new(0.0f32);
        let mut c = 0u32;
        while c < num_channels {
            sum += input[(c * num_samples + sample_idx) as usize];
            c += 1u32;
        }

        let avg = sum / F::cast_from(num_channels);

        c = 0u32;
        while c < num_channels {
            let idx = (c * num_samples + sample_idx) as usize;
            output[idx] = input[idx] - avg;
            c += 1u32;
        }
    }
}
