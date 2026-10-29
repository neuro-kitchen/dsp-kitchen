use cubecl::prelude::*;
use dspk_core::compute::{element_position, sample_position};

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

/// Common **median** reference: one cube per sample ([`fn@dspk_core::compute::row_position`]); its
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
    let s = dspk_core::compute::row_position();
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

/// Common median reference by **bitonic sort**: one cube per sample ([`fn@dspk_core::compute::row_position`]);
/// its units load the sample's channels into a shared column padded to `padded` (a power of two)
/// with the largest finite value (WGSL has no infinity literal; padding only ever sorts after the
/// channels), sort it with Batcher's bitonic network (`log₂ P · (log₂ P + 1) / 2` steps, each unit
/// doing `P / (2 · units)` compare-exchanges per step, one barrier per step), take the two middle
/// values (NumPy's median) and subtract their mean from the original samples.
///
/// The alternative to [`fn@common_median_kernel`]'s rank counting (`C` comparisons per value, no
/// barriers): the autotuner picks per device; barriers are cheap on GPUs and costly on the CPU
/// runtime.
#[cube(launch)]
pub fn common_median_bitonic_kernel<F: Float>(
    input: &[F],
    output: &mut [F],
    num_channels: u32,
    num_samples: u32,
    #[comptime] padded: u32,
    #[comptime] units: u32,
) {
    let s = dspk_core::compute::row_position();
    // Shared memory declared outside any branch (CubeCL lowering)
    let mut col = Shared::<[F]>::new_slice(comptime!(padded as usize));
    // `s` is uniform across the cube, so every unit takes the same branch
    if s < num_samples {
        let unit = UNIT_POS_X;
        let mut i = unit;
        while i < padded {
            // A statement, not an `if` expression: both arms of a select would be evaluated, and
            // the read is out of bounds for padding rows
            let mut v = F::max_value();
            if i < num_channels {
                v = input[(i * num_samples + s) as usize];
            }
            col[i as usize] = v;
            i += units;
        }
        sync_cube();
        // Runtime counters: a `let mut` from a constant would be comptime and could not halve
        let k = RuntimeCell::<u32>::new(2u32);
        while k.read() <= padded {
            let kk = k.read();
            let j = RuntimeCell::<u32>::new(kk / 2u32);
            while j.read() > 0u32 {
                let jj = j.read();
                let mut a_i = unit;
                while a_i < padded {
                    let b_i = a_i ^ jj;
                    if b_i > a_i {
                        let (a, b) = (col[a_i as usize], col[b_i as usize]);
                        let ascending = (a_i & kk) == 0u32;
                        let swap = if ascending { a > b } else { a < b };
                        if swap {
                            col[a_i as usize] = b;
                            col[b_i as usize] = a;
                        }
                    }
                    a_i += units;
                }
                sync_cube();
                j.store(jj / 2u32);
            }
            k.store(kk * 2u32);
        }
        let median = (col[((num_channels - 1u32) / 2u32) as usize] + col[(num_channels / 2u32) as usize]) * F::new(0.5f32);
        let mut c = unit;
        while c < num_channels {
            let idx = (c * num_samples + s) as usize;
            output[idx] = input[idx] - median;
            c += units;
        }
    }
}

/// Common median reference by rank counting on `N` consecutive samples per unit (`num_samples` a
/// multiple of `N`): channel `c` of the unit's samples is one aligned vector load, and each lane
/// counts, for every channel value, the channels below and equal to it, as
/// [`fn@common_median_kernel`] does for one sample. Same comparisons, same median `(m₁ + m₂) / 2`,
/// so the output is bit-identical; no shared memory or barriers (CubeCL's CPU runtime maps the
/// vectors to SIMD).
#[cube(launch)]
pub fn common_median_vector_kernel<F: Float, N: Size>(input: &[Vector<F, N>], output: &mut [Vector<F, N>], num_channels: u32, vectors_per_row: u32) {
    let v = element_position();
    if v < vectors_per_row as usize {
        let row = vectors_per_row as usize;
        let one = Vector::<u32, N>::new(1u32);
        let zero = Vector::<u32, N>::new(0u32);
        let k1 = Vector::<u32, N>::new((num_channels - 1u32) / 2u32);
        let k2 = Vector::<u32, N>::new(num_channels / 2u32);
        let mut mid1 = Vector::<F, N>::new(F::new(0.0f32));
        let mut mid2 = Vector::<F, N>::new(F::new(0.0f32));
        let mut c = 0u32;
        while c < num_channels {
            let x = input[c as usize * row + v];
            let mut less = zero;
            let mut equal = zero;
            let mut j = 0u32;
            while j < num_channels {
                let y = input[j as usize * row + v];
                less += select_many(y.less_than(&x), one, zero);
                equal += select_many(y.equal(&x), one, zero);
                j += 1u32;
            }
            let upto = less + equal;
            mid1 = select_many(k1.greater_equal(&less).vec_and(k1.less_than(&upto)), x, mid1);
            mid2 = select_many(k2.greater_equal(&less).vec_and(k2.less_than(&upto)), x, mid2);
            c += 1u32;
        }
        let median = (mid1 + mid2) * Vector::new(F::new(0.5f32));
        c = 0u32;
        while c < num_channels {
            let idx = c as usize * row + v;
            output[idx] = input[idx] - median;
            c += 1u32;
        }
    }
}

/// [`fn@direct_car_kernel`] on `N` consecutive samples per unit (`num_samples` a multiple of `N`):
/// channel `c` of the unit's samples is one aligned vector load; channels are summed in the same
/// order, so the output is bit-identical.
#[cube(launch)]
pub fn direct_car_vector_kernel<F: Float, N: Size>(input: &[Vector<F, N>], output: &mut [Vector<F, N>], num_channels: u32, vectors_per_row: u32) {
    let v = element_position();
    if v < vectors_per_row as usize {
        let row = vectors_per_row as usize;
        let mut sum = Vector::<F, N>::new(F::new(0.0f32));
        let mut c = 0u32;
        while c < num_channels {
            sum += input[c as usize * row + v];
            c += 1u32;
        }
        let avg = sum / Vector::new(F::cast_from(num_channels));
        c = 0u32;
        while c < num_channels {
            let idx = c as usize * row + v;
            output[idx] = input[idx] - avg;
            c += 1u32;
        }
    }
}
