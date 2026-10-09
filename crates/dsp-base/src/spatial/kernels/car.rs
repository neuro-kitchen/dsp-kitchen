use cubecl::prelude::*;
use dsp_core::compute::sample_position;

/// Single-pass Common Average Referencing: one unit per sample ([`fn@sample_position`]) averages that
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

/// Common **median** reference: one cube per sample ([`fn@dsp_core::compute::row_position`]); its
/// units load the sample's channels into shared memory, find the median by rank counting (each
/// value's count of smaller and equal values; NumPy's median: the mean of the two middle values
/// for an even count), and subtract it. `max_channels` (comptime) sizes the shared buffer.
#[cube(launch)]
pub fn common_median_kernel<F: Float>(
    input: &[F],
    output: &mut [F],
    num_channels: u32,
    num_samples: u32,
    #[comptime] max_channels: u32,
    #[comptime] units: u32,
) {
    let s = dsp_core::compute::row_position();
    if s < num_samples {
        let unit = UNIT_POS_X;
        let mut col = Shared::<[F]>::new_slice(comptime!(max_channels as usize));
        let mut mid = Shared::<[F]>::new_slice(2usize);
        let mut c = unit;
        while c < num_channels {
            col[c as usize] = input[(c * num_samples + s) as usize];
            c += units;
        }
        sync_cube();
        let k1 = (num_channels - 1u32) / 2u32;
        let k2 = num_channels / 2u32;
        c = unit;
        while c < num_channels {
            let x = col[c as usize];
            let mut less = 0u32;
            let mut equal = 0u32;
            let mut j = 0u32;
            while j < num_channels {
                let y = col[j as usize];
                if y < x {
                    less += 1u32;
                }
                if y == x {
                    equal += 1u32;
                }
                j += 1u32;
            }
            if k1 >= less && k1 < less + equal {
                mid[0] = x;
            }
            if k2 >= less && k2 < less + equal {
                mid[1] = x;
            }
            c += units;
        }
        sync_cube();
        let median = (mid[0] + mid[1]) * F::new(0.5f32);
        c = unit;
        while c < num_channels {
            output[(c * num_samples + s) as usize] = col[c as usize] - median;
            c += units;
        }
    }
}
