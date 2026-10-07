//! Per-row reductions of a `[rows, cols]` buffer, one cube per row
//! ([`LaunchGeometry::per_row`]).
//!
//! Each unit folds a strided share of the row (`col = unit, unit + units, …`, so neighbouring units
//! read neighbouring values), then the cube merges the partial results pairwise in shared memory.
//! Pairwise merging keeps the error of long rows close to that of short sums.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_core::compute::{row_position, LaunchGeometry};

use super::kernels::row_abs_kth_radix_kernel;
use super::DspFloat;

/// Merges unit `b`'s `(count, mean, m2)` into unit `a`'s (Chan et al. parallel variance).
#[cube]
fn merge_moments<F: Float>(count: &mut Shared<[F]>, mean: &mut Shared<[F]>, m2: &mut Shared<[F]>, a: usize, b: usize) {
    let (na, nb) = (count[a], count[b]);
    let n = na + nb;
    if nb > F::new(0.0f32) {
        let delta = mean[b] - mean[a];
        let m2_b = m2[b];
        mean[a] += delta * nb / n;
        m2[a] += m2_b + delta * delta * na * nb / n;
        count[a] = n;
    }
}

/// Mean and population standard deviation (`/ cols`) of every row. `units` is the cube's x size
/// (a power of two).
#[cube(launch)]
pub fn row_mean_std_kernel<F: Float>(
    input: &[F],
    out_mean: &mut [F],
    out_std: &mut [F],
    rows: u32,
    cols: u32,
    #[comptime] units: u32,
) {
    let row = row_position();
    // `row` is uniform across the cube, so every unit of a cube takes the same branch
    if row < rows {
        let unit = UNIT_POS_X;
        let base = (row * cols) as usize;

        // Welford over this unit's strided share
        let mut n = F::new(0.0f32);
        let mut mean = F::new(0.0f32);
        let mut m2 = F::new(0.0f32);
        let mut col = unit;
        while col < cols {
            let x = input[base + col as usize];
            n += F::new(1.0f32);
            let delta = x - mean;
            mean += delta / n;
            m2 += delta * (x - mean);
            col += units;
        }

        let mut count_s = Shared::<[F]>::new_slice(comptime!(units as usize));
        let mut mean_s = Shared::<[F]>::new_slice(comptime!(units as usize));
        let mut m2_s = Shared::<[F]>::new_slice(comptime!(units as usize));
        count_s[unit as usize] = n;
        mean_s[unit as usize] = mean;
        m2_s[unit as usize] = m2;
        sync_cube();

        let stride = RuntimeCell::<u32>::new(units / 2u32);
        while stride.read() > 0u32 {
            let s = stride.read();
            if unit < s {
                merge_moments::<F>(&mut count_s, &mut mean_s, &mut m2_s, unit as usize, (unit + s) as usize);
            }
            sync_cube();
            stride.store(s / 2u32);
        }

        if unit == 0u32 {
            out_mean[row as usize] = mean_s[0];
            let total = F::max(count_s[0], F::new(1.0f32));
            out_std[row as usize] = F::sqrt(m2_s[0] / total);
        }
    }
}

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
}

/// Per-row `k`-th smallest `|x|` (0-based, a sample value, exact) over columns `cols` of a buffer
/// of `rows` rows `row_stride` apart, into `out` (`rows` values of `F`). `k < cols.len()`. A radix
/// select over the bits of `|x|` ([`row_abs_kth_radix_kernel`]): 4 passes over each row for
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
    assert!(k < cols.len() && cols.end <= row_stride, "selection outside the rows");
    let geom = LaunchGeometry::per_row(client, rows, cols.len());
    let args = (rows as u32, row_stride as u32, cols.start as u32, cols.len() as u32, k as u32, (8 * size_of::<F>()) as u32, geom.cube_dim.x);
    // SAFETY: `input` holds `rows · row_stride` and `out` `rows` values of `F`
    unsafe {
        let (input, out) = (BufferArg::from_raw_parts(input.clone(), rows * row_stride), BufferArg::from_raw_parts(out.clone(), rows));
        match size_of::<F>() {
            4 => row_abs_kth_radix_kernel::launch::<F, u32>(client, geom.cube_count, geom.cube_dim, input, out, args.0, args.1, args.2, args.3, args.4, args.5, args.6),
            8 => row_abs_kth_radix_kernel::launch::<F, u64>(client, geom.cube_count, geom.cube_dim, input, out, args.0, args.1, args.2, args.3, args.4, args.5, args.6),
            other => panic!("row_abs_kth: no radix key for {other}-byte floats"),
        }
    }
}
