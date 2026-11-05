//! FIR kernels that compute `N` consecutive outputs per unit (`Vector<F, N>`): CubeCL's CPU runtime
//! maps the vectors to SIMD, GPUs get wider loads.
//!
//! Output `y[i … i+N−1]` needs input samples at every offset, which aligned vector loads cannot
//! give. Instead each input sample `x[j]` is broadcast and multiplied by a vector of taps,
//! `table[m]`, holding in lane `l` the tap that `x[j]` meets at output `i + l` (zero where it meets
//! none). The table is built once per call by [`fn@fir_vector_table_kernel`]; every load is
//! aligned, so the same source runs on every runtime. Each lane adds its taps in the order of the
//! scalar kernels (`k = 0 … K−1`, plus exact zeros), so the outputs are bit-identical to theirs.
//!
//! Rows whose length `N` does not divide run the `_unaligned` kernels: the same vectors, stored
//! lane by lane ([`fn@store_lanes`]), the last one of each row partly.

use cubecl::prelude::*;
use dspk_core::compute::element_position;

use crate::core::kernels::store_lanes;
use crate::core::read_extended;

/// The tap table (`num_taps + width − 1` vectors of `width`, flattened) of [`fn@fir_vector_kernel`]
/// (`table[m]` lane `l` = `taps[m + l − (width − 1)]`) or, if `centered`, of
/// [`fn@fir_centered_vector_kernel`] (`taps[m − l]`); zero outside `0 … num_taps − 1`.
#[cube(launch)]
pub fn fir_vector_table_kernel<F: Float>(taps: &[F], table: &mut [F], num_taps: u32, width: u32, #[comptime] centered: bool) {
    let s = element_position();
    if s < table.len() {
        let m = s as u32 / width;
        let l = s as u32 % width;
        let mut value = F::new(0.0f32);
        if comptime!(centered) {
            if m >= l && m - l < num_taps {
                value = taps[(m - l) as usize];
            }
        } else if m + l >= width - 1u32 && m + l - (width - 1u32) < num_taps {
            value = taps[(m + l - (width - 1u32)) as usize];
        }
        table[s] = value;
    }
}

/// Outputs `i … i + N − 1` of the causal filter on the row at `input[base]`, input samples visited
/// newest first (`x[i + N − 1 − m]`, `m = 0 … K + N − 2`), so lane `l` meets taps
/// `k = m + l − (N − 1)` in increasing order. Lanes past the row's end read its edge extension.
#[cube]
fn fir_vector_at<F: Float, N: Size>(input: &[F], table: &[Vector<F, N>], base: usize, i: u32, num_samples: u32, num_taps: u32, #[comptime] edge: u32) -> Vector<F, N> {
    let n = N::value() as u32;
    let history = num_taps - 1u32;
    let span = num_taps + n - 1u32;
    let mut acc = Vector::<F, N>::new(F::new(0.0f32));
    let mut m = 0u32;
    if i >= history && i + n <= num_samples {
        let newest = base + (i + n - 1u32) as usize;
        while m < span {
            acc += table[m as usize] * Vector::new(input[newest - m as usize]);
            m += 1u32;
        }
    } else {
        // Logical sample i + N − 1 − m is extended position i + N − 1 − m + history
        while m < span {
            acc += table[m as usize] * Vector::new(read_extended::<F>(input, base, num_samples, history, i + n - 1u32 + history - m, edge));
            m += 1u32;
        }
    }
    acc
}

/// Outputs `i … i + N − 1` of the centred filter on the row at `input[base]`: input samples
/// `x[i − radius + m]`, `m = 0 … K + N − 2`, oldest first, so lane `l` meets taps `k = m − l` in
/// increasing order. Lanes past the row's end read its edge extension.
#[cube]
fn fir_centered_vector_at<F: Float, N: Size>(input: &[F], table: &[Vector<F, N>], base: usize, i: u32, num_samples: u32, radius: u32, #[comptime] edge: u32) -> Vector<F, N> {
    let n = N::value() as u32;
    let span = 2u32 * radius + n;
    let mut acc = Vector::<F, N>::new(F::new(0.0f32));
    let mut m = 0u32;
    if i >= radius && i + n - 1u32 + radius < num_samples {
        let oldest = base + (i - radius) as usize;
        while m < span {
            acc += table[m as usize] * Vector::new(input[oldest + m as usize]);
            m += 1u32;
        }
    } else {
        // Logical sample i − radius + m is extended position i + m
        while m < span {
            acc += table[m as usize] * Vector::new(read_extended::<F>(input, base, num_samples, radius, i + m, edge));
            m += 1u32;
        }
    }
    acc
}

/// [`fn@super::fir_filter_kernel`] with `N` outputs per unit (`num_samples` a multiple of `N`).
#[cube(launch)]
pub fn fir_vector_kernel<F: Float, N: Size>(
    input: &[F],
    output: &mut [Vector<F, N>],
    table: &[Vector<F, N>],
    num_samples: u32,
    num_taps: u32,
    #[comptime] edge: u32,
) {
    let v = element_position();
    if v < output.len() {
        let n = N::value() as u32;
        let per_row = num_samples / n;
        let i = (v as u32 % per_row) * n;
        let base = ((v as u32 / per_row) * num_samples) as usize;
        output[v] = fir_vector_at::<F, N>(input, table, base, i, num_samples, num_taps, edge);
    }
}

/// [`fn@fir_vector_kernel`] for any `num_samples`: `ceil(num_samples / N)` units per row
/// (`units_per_row`), outputs stored lane by lane.
#[cube(launch)]
pub fn fir_vector_unaligned_kernel<F: Float, N: Size>(
    input: &[F],
    output: &mut [F],
    table: &[Vector<F, N>],
    num_samples: u32,
    num_taps: u32,
    units_per_row: u32,
    num_units: u32,
    #[comptime] edge: u32,
) {
    let v = element_position() as u32;
    if v < num_units {
        let n = N::value() as u32;
        let i = (v % units_per_row) * n;
        let base = ((v / units_per_row) * num_samples) as usize;
        let acc = fir_vector_at::<F, N>(input, table, base, i, num_samples, num_taps, edge);
        store_lanes::<F, N>(output, base + i as usize, acc, u32::min(n, num_samples - i));
    }
}

/// [`fn@super::fir_centered_filter_kernel`] with `N` outputs per unit (`num_samples` a multiple of
/// `N`).
#[cube(launch)]
pub fn fir_centered_vector_kernel<F: Float, N: Size>(
    input: &[F],
    output: &mut [Vector<F, N>],
    table: &[Vector<F, N>],
    num_samples: u32,
    radius: u32,
    #[comptime] edge: u32,
) {
    let v = element_position();
    if v < output.len() {
        let n = N::value() as u32;
        let per_row = num_samples / n;
        let i = (v as u32 % per_row) * n;
        let base = ((v as u32 / per_row) * num_samples) as usize;
        output[v] = fir_centered_vector_at::<F, N>(input, table, base, i, num_samples, radius, edge);
    }
}

/// [`fn@fir_centered_vector_kernel`] for any `num_samples` (as [`fn@fir_vector_unaligned_kernel`]).
#[cube(launch)]
pub fn fir_centered_vector_unaligned_kernel<F: Float, N: Size>(
    input: &[F],
    output: &mut [F],
    table: &[Vector<F, N>],
    num_samples: u32,
    radius: u32,
    units_per_row: u32,
    num_units: u32,
    #[comptime] edge: u32,
) {
    let v = element_position() as u32;
    if v < num_units {
        let n = N::value() as u32;
        let i = (v % units_per_row) * n;
        let base = ((v / units_per_row) * num_samples) as usize;
        let acc = fir_centered_vector_at::<F, N>(input, table, base, i, num_samples, radius, edge);
        store_lanes::<F, N>(output, base + i as usize, acc, u32::min(n, num_samples - i));
    }
}
