//! Per-row reductions of a `[rows, cols]` buffer, one cube per row
//! ([`LaunchGeometry::per_row`]).
//!
//! Each unit folds a strided share of the row (`col = unit, unit + units, …`, so neighbouring units
//! read neighbouring values), then the cube merges the partial results pairwise in shared memory.
//! Pairwise merging keeps the error of long rows close to that of short sums.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_core::compute::{row_position, LaunchGeometry};

use super::DspFloat;

/// Merges unit `b`'s `(count, mean, m2)` into unit `a`'s (Chan et al. parallel variance).
#[cube]
fn merge_moments<F: Float>(count: &mut SharedMemory<F>, mean: &mut SharedMemory<F>, m2: &mut SharedMemory<F>, a: usize, b: usize) {
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
    input: &Array<F>,
    out_mean: &mut Array<F>,
    out_std: &mut Array<F>,
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

        let mut count_s = SharedMemory::<F>::new(comptime!(units as usize));
        let mut mean_s = SharedMemory::<F>::new(comptime!(units as usize));
        let mut m2_s = SharedMemory::<F>::new(comptime!(units as usize));
        count_s[unit as usize] = n;
        mean_s[unit as usize] = mean;
        m2_s[unit as usize] = m2;
        sync_cube();

        let mut stride = comptime!(units / 2);
        while stride > 0u32 {
            if unit < stride {
                merge_moments::<F>(&mut count_s, &mut mean_s, &mut m2_s, unit as usize, (unit + stride) as usize);
            }
            sync_cube();
            stride /= 2u32;
        }

        if unit == 0u32 {
            out_mean[row as usize] = mean_s[0];
            let total = F::max(count_s[0], F::new(1.0f32));
            out_std[row as usize] = F::sqrt(m2_s[0] / total);
        }
    }
}

/// Halvings of the value range `[−1, max |x|]` in [`row_abs_kth_kernel`]. The final bracket is
/// `(max |x| + 1) · 2⁻⁶⁴` wide: one distinct f32 value except for values within ~1e-9 · max of zero;
/// otherwise (f64, near-zero values) the result is off by at most the bracket width.
pub const ROW_SELECT_ITERATIONS: u32 = 64;

/// Sum of `values[0..units]` into `values[0]` (pairwise, in shared memory).
#[cube]
fn shared_sum_u32(values: &mut SharedMemory<u32>, unit: u32, #[comptime] units: u32) {
    let mut stride = comptime!(units / 2);
    while stride > 0u32 {
        if unit < stride {
            let other = values[(unit + stride) as usize];
            values[unit as usize] += other;
        }
        sync_cube();
        stride /= 2u32;
    }
}

/// One cube per row: the `k`-th smallest (0-based) `|x|` among columns `col_start..col_start + cols`
/// of row `row` (rows `row_stride` apart). The value range is halved [`ROW_SELECT_ITERATIONS`]
/// times keeping `#(|x| ≤ lo) ≤ k < #(|x| ≤ hi)`, then the smallest `|x|` above `lo` is the answer
/// (a sample value, not an interpolation).
#[cube(launch)]
pub fn row_abs_kth_kernel<F: Float>(
    input: &Array<F>,
    out: &mut Array<F>,
    rows: u32,
    row_stride: u32,
    col_start: u32,
    cols: u32,
    k: u32,
    #[comptime] units: u32,
    #[comptime] iterations: u32,
) {
    let row = row_position();
    if row < rows {
        let unit = UNIT_POS_X;
        let base = (row * row_stride + col_start) as usize;
        let mut vals = SharedMemory::<F>::new(comptime!(units as usize));
        let mut counts = SharedMemory::<u32>::new(comptime!(units as usize));

        // Bracket: lo below every |x|, hi = max |x|
        let mut hi_u = F::new(0.0f32);
        let mut col = unit;
        while col < cols {
            hi_u = F::max(hi_u, F::abs(input[base + col as usize]));
            col += units;
        }
        vals[unit as usize] = hi_u;
        sync_cube();
        let mut stride = comptime!(units / 2);
        while stride > 0u32 {
            if unit < stride {
                let other = vals[(unit + stride) as usize];
                vals[unit as usize] = F::max(vals[unit as usize], other);
            }
            sync_cube();
            stride /= 2u32;
        }
        let mut lo = F::new(-1.0f32);
        let mut hi = vals[0];
        sync_cube();

        // Runtime loop (a comptime range would unroll every halving into the kernel)
        let mut it: u32 = 0u32;
        while it < iterations {
            let mid = (lo + hi) / F::new(2.0f32);
            let mut n = 0u32;
            let mut col = unit;
            while col < cols {
                if F::abs(input[base + col as usize]) <= mid {
                    n += 1u32;
                }
                col += units;
            }
            counts[unit as usize] = n;
            sync_cube();
            shared_sum_u32(&mut counts, unit, units);
            if counts[0] > k {
                hi = mid;
            } else {
                lo = mid;
            }
            sync_cube();
            it += 1u32;
        }

        // Smallest |x| above lo
        let mut best = F::new(f32::INFINITY);
        let mut col = unit;
        while col < cols {
            let a = F::abs(input[base + col as usize]);
            if a > lo {
                best = F::min(best, a);
            }
            col += units;
        }
        vals[unit as usize] = best;
        sync_cube();
        let mut stride = comptime!(units / 2);
        while stride > 0u32 {
            if unit < stride {
                let other = vals[(unit + stride) as usize];
                vals[unit as usize] = F::min(vals[unit as usize], other);
            }
            sync_cube();
            stride /= 2u32;
        }
        if unit == 0u32 {
            out[row as usize] = vals[0];
        }
    }
}

/// Per-row mean and population standard deviation of a `[rows, cols]` buffer of `F` into `out_mean`
/// and `out_std` (`rows` values each).
pub fn row_mean_std<R: Runtime, F: DspFloat>(
    client: &ComputeClient<R>,
    input: &Handle,
    out_mean: &Handle,
    out_std: &Handle,
    rows: usize,
    cols: usize,
) {
    let geom = LaunchGeometry::per_row(client, rows, cols);
    unsafe {
        row_mean_std_kernel::launch::<F, R>(
            client,
            geom.cube_count,
            geom.cube_dim.clone(),
            ArrayArg::from_raw_parts(input.clone(), rows * cols),
            ArrayArg::from_raw_parts(out_mean.clone(), rows),
            ArrayArg::from_raw_parts(out_std.clone(), rows),
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

    fn reductions<R: Runtime>(client: &ComputeClient<R>) {
        // Rows longer and shorter than a cube, one empty-ish row of a single value
        for cols in [1usize, 5, 1_000, 70_001] {
            let rows = 3;
            let data: Vec<f32> = (0..rows * cols).map(|i| ((i * 7919) % 1013) as f32 * 0.1 - 40.0 + (i / cols) as f32 * 100.0).collect();
            let input = buffer::upload(client, &data);
            let (mean, std) = (buffer::empty::<R, f32>(client, rows), buffer::empty::<R, f32>(client, rows));
            row_mean_std::<R, f32>(client, &input, &mean, &std, rows, cols);
            let (mean, std) = (buffer::download::<R, f32>(client, mean), buffer::download::<R, f32>(client, std));
            for r in 0..rows {
                let row: Vec<f64> = data[r * cols..(r + 1) * cols].iter().map(|v| *v as f64).collect();
                let m = row.iter().sum::<f64>() / cols as f64;
                let s = (row.iter().map(|v| (v - m).powi(2)).sum::<f64>() / cols as f64).sqrt();
                assert!((mean[r] as f64 - m).abs() < 1e-3 * m.abs().max(1.0), "{} cols {cols} row {r}: mean {} vs {m}", R::name(client), mean[r]);
                assert!((std[r] as f64 - s).abs() < 1e-3 * s.max(1.0), "{} cols {cols} row {r}: std {} vs {s}", R::name(client), std[r]);
            }
        }
    }
    runtime_test!(test_row_reductions_match_host, reductions);

    fn abs_kth<R: Runtime>(client: &ComputeClient<R>) {
        // Odd / even lengths, repeated values, a column sub-range of a longer row
        let (rows, stride) = (3usize, 2_003usize);
        let data: Vec<f32> = (0..rows * stride).map(|i| ((i * 7919) % 211) as f32 * 0.25 - 26.0).collect();
        let input = buffer::upload(client, &data);
        for cols in [0..1usize, 0..2, 5..1_006, 0..stride] {
            for k in [0, cols.len() / 2, cols.len() - 1] {
                let out = buffer::empty::<R, f32>(client, rows);
                row_abs_kth::<R, f32>(client, &input, &out, rows, stride, cols.clone(), k);
                let got = buffer::download::<R, f32>(client, out);
                for r in 0..rows {
                    let mut abs: Vec<f32> = data[r * stride + cols.start..r * stride + cols.end].iter().map(|v| v.abs()).collect();
                    abs.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
                    assert_eq!(got[r], abs[k], "{} cols {cols:?} k {k} row {r}", R::name(client));
                }
            }
        }
    }
    runtime_test!(test_row_abs_kth_matches_host, abs_kth);
}

/// Per-row `k`-th smallest `|x|` (0-based, a sample value) over columns `cols` of a buffer of
/// `rows` rows `row_stride` apart, into `out` (`rows` values of `F`). `k < cols.len()`.
pub fn row_abs_kth<R: Runtime, F: DspFloat>(
    client: &ComputeClient<R>,
    input: &Handle,
    out: &Handle,
    rows: usize,
    row_stride: usize,
    cols: std::ops::Range<usize>,
    k: usize,
) {
    assert!(k < cols.len() && cols.end <= row_stride, "selection outside the rows");
    let geom = LaunchGeometry::per_row(client, rows, cols.len());
    // SAFETY: `input` holds `rows · row_stride` and `out` `rows` values of `F`
    unsafe {
        row_abs_kth_kernel::launch::<F, R>(
            client,
            geom.cube_count,
            geom.cube_dim.clone(),
            ArrayArg::from_raw_parts(input.clone(), rows * row_stride),
            ArrayArg::from_raw_parts(out.clone(), rows),
            rows as u32,
            row_stride as u32,
            cols.start as u32,
            cols.len() as u32,
            k as u32,
            geom.cube_dim.x,
            ROW_SELECT_ITERATIONS,
        );
    }
}
