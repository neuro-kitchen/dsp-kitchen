use cubecl::prelude::*;

/// Copies columns `0..cols` of `rows` strided rows into a split-major buffer `[splits, rows,
/// split]`: `output[(s · rows + r) · split + j] = input[offset + r · row_stride + t] − mean[r]` with
/// `t = s · split + j`, zero where `t ≥ cols` (the padded end of the last split; zero columns add
/// nothing to a product with itself). `centred = false` skips the mean (`mean` is not read). The
/// offset is applied here, so the input binds at the buffer's start. One unit per output element
/// (`ABSOLUTE_POS`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn split_rows_kernel<F: Float>(
    input: &[F],
    mean: &[F],
    output: &mut [F],
    offset: u32,
    row_stride: u32,
    cols: u32,
    rows: u32,
    split: u32,
    total: u32,
    #[comptime] centred: bool,
) {
    let e = ABSOLUTE_POS as u32;
    if e < total {
        let j = e % split;
        let sr = e / split;
        let r = sr % rows;
        let s = sr / rows;
        let t = s * split + j;
        let mut value = F::new(0.0f32);
        if t < cols {
            value = input[(offset + r * row_stride + t) as usize];
            if comptime!(centred) {
                value -= mean[r as usize];
            }
        }
        output[e as usize] = value;
    }
}

/// `out[e] = scale · Σ_s partial[s · len + e]` over the `slices` partial products; `accumulate`
/// adds to `out` instead of overwriting it. One unit per element (`ABSOLUTE_POS`).
#[cube(launch)]
pub fn sum_slices_kernel<F: Float + CubeElement + LaunchArg>(
    partial: &[F],
    out: &mut [F],
    len: u32,
    slices: u32,
    scale: F,
    #[comptime] accumulate: bool,
) {
    let e = ABSOLUTE_POS as u32;
    if e < len {
        let mut acc = F::new(0.0f32);
        let mut s = 0u32;
        while s < slices {
            acc += partial[(s * len + e) as usize];
            s += 1u32;
        }
        let value = acc * scale;
        if comptime!(accumulate) {
            out[e as usize] += value;
        } else {
            out[e as usize] = value;
        }
    }
}
