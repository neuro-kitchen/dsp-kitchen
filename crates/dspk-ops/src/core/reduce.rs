//! Per-row reductions of a `[rows, cols]` buffer, one cube per row
//! ([`LaunchGeometry::per_row`]).
//!
//! Each unit folds a strided share of the row (`col = unit, unit + units, …`, so neighbouring units
//! read neighbouring values), then the cube merges the partial results pairwise in shared memory.
//! Pairwise merging keeps the error of long rows close to that of short sums.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dspk_core::compute::LaunchGeometry;

use super::kernels::{row_kth_radix_kernel, row_mean_std_kernel, KEY_ABS, KEY_ABS_DEV, KEY_SIGNED};
use super::DspFloat;

/// Per-row mean and population standard deviation of a `[rows, cols]` buffer of `F` into `out_mean`
/// and `out_std` (`rows` values each).
pub fn row_mean_std<F: DspFloat>(
    client: &Client,
    input: &Handle,
    out_mean: &Handle,
    out_std: &Handle,
    rows: usize,
    cols: usize,
) {
    let geom = LaunchGeometry::per_row(client, rows, cols);
    unsafe {
        row_mean_std_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim.clone(),
            BufferArg::from_raw_parts(input.clone(), rows * cols),
            BufferArg::from_raw_parts(out_mean.clone(), rows),
            BufferArg::from_raw_parts(out_std.clone(), rows),
            rows as u32,
            cols as u32,
            geom.cube_dim.x,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::buffer;

    fn reductions(client: &Client) {
        // Rows longer and shorter than a cube, one empty-ish row of a single value
        for cols in [1usize, 5, 1_000, 70_001] {
            let rows = 3;
            let data: Vec<f32> = (0..rows * cols).map(|i| ((i * 7919) % 1013) as f32 * 0.1 - 40.0 + (i / cols) as f32 * 100.0).collect();
            let input = buffer::upload(client, &data);
            let (mean, std) = (buffer::empty::<f32>(client, rows), buffer::empty::<f32>(client, rows));
            row_mean_std::<f32>(client, &input, &mean, &std, rows, cols);
            let (mean, std) = (buffer::download::<f32>(client, mean), buffer::download::<f32>(client, std));
            for r in 0..rows {
                let row: Vec<f64> = data[r * cols..(r + 1) * cols].iter().map(|v| *v as f64).collect();
                let m = row.iter().sum::<f64>() / cols as f64;
                let s = (row.iter().map(|v| (v - m).powi(2)).sum::<f64>() / cols as f64).sqrt();
                assert!((mean[r] as f64 - m).abs() < 1e-3 * m.abs().max(1.0), "{} cols {cols} row {r}: mean {} vs {m}", client.name(), mean[r]);
                assert!((std[r] as f64 - s).abs() < 1e-3 * s.max(1.0), "{} cols {cols} row {r}: std {} vs {s}", client.name(), std[r]);
            }
        }
    }
    runtime_test!(test_row_reductions_match_host, reductions);

    fn abs_kth(client: &Client) {
        // Odd / even lengths, repeated values, a column sub-range of a longer row
        let (rows, stride) = (3usize, 2_003usize);
        let data: Vec<f32> = (0..rows * stride).map(|i| ((i * 7919) % 211) as f32 * 0.25 - 26.0).collect();
        let input = buffer::upload(client, &data);
        for cols in [0..1usize, 0..2, 5..1_006, 0..stride] {
            for k in [0, cols.len() / 2, cols.len() - 1] {
                let out = buffer::empty::<f32>(client, rows);
                row_abs_kth::<f32>(client, &input, &out, rows, stride, cols.clone(), k);
                let got = buffer::download::<f32>(client, out);
                for r in 0..rows {
                    let mut abs: Vec<f32> = data[r * stride + cols.start..r * stride + cols.end].iter().map(|v| v.abs()).collect();
                    abs.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
                    assert_eq!(got[r], abs[k], "{} cols {cols:?} k {k} row {r}", client.name());
                }
            }
        }
    }
    runtime_test!(test_row_abs_kth_matches_host, abs_kth);

    /// The `f64` path (64-bit keys) is exact too, on runtimes with `f64`.
    fn abs_kth_f64(client: &Client) {
        if !client.properties().supports_type(f64::elem_type_native()) {
            return;
        }
        let (rows, cols) = (2usize, 1_001usize);
        let data: Vec<f64> = (0..rows * cols).map(|i| ((i * 7919) % 211) as f64 * 0.125 - 13.0 + 1e-9 * i as f64).collect();
        let input = buffer::upload(client, &data);
        for k in [0, cols / 2, cols - 1] {
            let out = buffer::empty::<f64>(client, rows);
            row_abs_kth::<f64>(client, &input, &out, rows, cols, 0..cols, k);
            let got = buffer::download::<f64>(client, out);
            for r in 0..rows {
                let mut abs: Vec<f64> = data[r * cols..(r + 1) * cols].iter().map(|v| v.abs()).collect();
                abs.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
                assert_eq!(got[r], abs[k], "{} f64 k {k} row {r}", client.name());
            }
        }
    }
    runtime_test!(test_row_abs_kth_f64, abs_kth_f64);

    /// Signed and deviation selection, exact against a host sort, for `F` (negatives, repeats,
    /// odd / even lengths, a column sub-range).
    fn signed_and_dev_kth_of<F: DspFloat + std::fmt::Debug>(client: &Client) {
        let (rows, stride) = (3usize, 1_003usize);
        let data: Vec<F> = (0..rows * stride).map(|i| crate::core::cast::<F>(((i * 7919) % 211) as f64 * 0.25 - 26.0)).collect();
        let centers: Vec<F> = (0..rows).map(|r| crate::core::cast::<F>(r as f64 * 3.5 - 2.0)).collect();
        let (input, center) = (buffer::upload(client, &data), buffer::upload(client, &centers));
        let sorted_k = |mut v: Vec<F>, k: usize| {
            v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            v[k]
        };
        for cols in [0..1usize, 0..2, 5..1_000, 0..stride] {
            for k in [0, cols.len() / 2, cols.len() - 1] {
                let (signed, dev) = (buffer::empty::<F>(client, rows), buffer::empty::<F>(client, rows));
                row_kth::<F>(client, &input, &signed, rows, stride, cols.clone(), k);
                row_abs_dev_kth::<F>(client, &input, &center, &dev, rows, stride, cols.clone(), k);
                let (signed, dev) = (buffer::download::<F>(client, signed), buffer::download::<F>(client, dev));
                for r in 0..rows {
                    let row = &data[r * stride + cols.start..r * stride + cols.end];
                    assert_eq!(signed[r], sorted_k(row.to_vec(), k), "{} signed cols {cols:?} k {k} row {r}", client.name());
                    // Host |d| (`.abs()` on `F` is CubeCL's kernel-only `abs`)
                    let zero = crate::core::cast::<F>(0.0);
                    let devs: Vec<F> = row.iter().map(|&v| v - centers[r]).map(|d| if d < zero { zero - d } else { d }).collect();
                    assert_eq!(dev[r], sorted_k(devs, k), "{} deviation cols {cols:?} k {k} row {r}", client.name());
                }
            }
        }
    }

    fn signed_and_dev_kth(client: &Client) {
        signed_and_dev_kth_of::<f32>(client);
        if client.properties().supports_type(f64::elem_type_native()) {
            signed_and_dev_kth_of::<f64>(client);
        }
    }
    runtime_test!(test_row_signed_and_deviation_kth_match_host, signed_and_dev_kth);
}

/// Per-row `k`-th smallest `|x|` (0-based, a sample value, exact) over columns `cols` of a buffer
/// of `rows` rows `row_stride` apart, into `out` (`rows` values of `F`). `k < cols.len()`. A radix
/// select over the bits of `|x|` ([`fn@row_kth_radix_kernel`]): 4 passes over each row for
/// `f32`, 8 for `f64`.
///
/// # Panics
/// If `F` is neither 32 nor 64 bits wide (no unsigned key of its width on every runtime).
pub fn row_abs_kth<F: DspFloat>(
    client: &Client,
    input: &Handle,
    out: &Handle,
    rows: usize,
    row_stride: usize,
    cols: std::ops::Range<usize>,
    k: usize,
) {
    row_kth_select::<F>(client, input, None, out, rows, row_stride, cols, k, KEY_ABS);
}

/// [`row_abs_kth`] of the signed values: the per-row `k`-th smallest `x` (a percentile, exact).
pub fn row_kth<F: DspFloat>(
    client: &Client,
    input: &Handle,
    out: &Handle,
    rows: usize,
    row_stride: usize,
    cols: std::ops::Range<usize>,
    k: usize,
) {
    row_kth_select::<F>(client, input, None, out, rows, row_stride, cols, k, KEY_SIGNED);
}

/// [`row_abs_kth`] of `|x − center[row]|` (`center`: `rows` values of `F`): the per-row `k`-th
/// smallest deviation from a centre, written as the deviation.
#[allow(clippy::too_many_arguments)]
pub fn row_abs_dev_kth<F: DspFloat>(
    client: &Client,
    input: &Handle,
    center: &Handle,
    out: &Handle,
    rows: usize,
    row_stride: usize,
    cols: std::ops::Range<usize>,
    k: usize,
) {
    row_kth_select::<F>(client, input, Some(center), out, rows, row_stride, cols, k, KEY_ABS_DEV);
}

#[allow(clippy::too_many_arguments)]
fn row_kth_select<F: DspFloat>(
    client: &Client,
    input: &Handle,
    center: Option<&Handle>,
    out: &Handle,
    rows: usize,
    row_stride: usize,
    cols: std::ops::Range<usize>,
    k: usize,
    mode: u32,
) {
    assert!(k < cols.len() && cols.end <= row_stride, "selection outside the rows");
    let geom = LaunchGeometry::per_row(client, rows, cols.len());
    let args = (rows as u32, row_stride as u32, cols.start as u32, cols.len() as u32, k as u32, (8 * size_of::<F>()) as u32, geom.cube_dim.x);
    // Modes without a centre get a one-value placeholder the kernel never reads (never the output
    // itself: one buffer bound twice, once written, is an aliased binding)
    let (center, center_len) = match center {
        Some(c) => (c.clone(), rows),
        None => (crate::core::buffer::empty::<F>(client, 1), 1),
    };
    // SAFETY: `input` holds `rows · row_stride`, `center` and `out` `rows` values of `F`
    unsafe {
        let (input, center, out) = (
            BufferArg::from_raw_parts(input.clone(), rows * row_stride),
            BufferArg::from_raw_parts(center, center_len),
            BufferArg::from_raw_parts(out.clone(), rows),
        );
        match size_of::<F>() {
            4 => row_kth_radix_kernel::launch::<F, u32>(client, geom.cube_count, geom.cube_dim, input, center, out, args.0, args.1, args.2, args.3, args.4, args.5, args.6, mode),
            8 => row_kth_radix_kernel::launch::<F, u64>(client, geom.cube_count, geom.cube_dim, input, center, out, args.0, args.1, args.2, args.3, args.4, args.5, args.6, mode),
            other => panic!("row_kth: no radix key for {other}-byte floats"),
        }
    }
}
