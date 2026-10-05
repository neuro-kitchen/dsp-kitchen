//! Spatial `[C, C]` operators on the device, stored dense or as sparse rows depending on how many
//! weights are nonzero.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_core::compute::LaunchGeometry;

use super::kernels::sparse_rows_multiply_kernel;
use super::whitening::execute_spatial_matrix_multiply;
use crate::core::{buffer, cast_f32, DspFloat};

/// A matrix is stored as sparse rows when its widest row holds at most this fraction of the
/// columns; denser matrices run the dense kernel (fewer index reads).
pub const SPARSE_MAX_ROW_FILL: f64 = 0.5;

/// Rows of a square matrix keeping only nonzero weights, padded to the widest row (ELLPACK).
#[derive(Debug, Clone, PartialEq)]
pub struct SparseRows {
    pub channels: usize,
    /// Entries per row (the widest row's nonzero count).
    pub width: usize,
    /// `[channels, width]` weights; padding entries are 0.
    pub values: Vec<f32>,
    /// `[channels, width]` input channel of each weight; padding entries point at channel 0.
    pub indices: Vec<u32>,
}

impl SparseRows {
    /// The nonzero entries of row-major `matrix` (`[channels, channels]`).
    pub fn from_dense(matrix: &[f32], channels: usize) -> Self {
        assert_eq!(matrix.len(), channels * channels);
        let rows: Vec<Vec<(u32, f32)>> = matrix
            .chunks(channels.max(1))
            .map(|row| row.iter().enumerate().filter(|(_, w)| **w != 0.0).map(|(i, w)| (i as u32, *w)).collect())
            .collect();
        let width = rows.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let mut values = vec![0.0f32; channels * width];
        let mut indices = vec![0u32; channels * width];
        for (o, row) in rows.iter().enumerate() {
            for (k, &(i, w)) in row.iter().enumerate() {
                values[o * width + k] = w;
                indices[o * width + k] = i;
            }
        }
        Self { channels, width, values, indices }
    }

    /// Fraction of the columns the widest row holds.
    pub fn fill(&self) -> f64 {
        self.width as f64 / self.channels.max(1) as f64
    }
}

/// `output = W · input` for a sparse-rows `W` (`values` / `indices` uploaded from [`SparseRows`]).
#[allow(clippy::too_many_arguments)]
pub fn execute_sparse_rows_multiply<R: Runtime, F: DspFloat>(
    client: &ComputeClient<R>,
    input: &Handle,
    values: &Handle,
    indices: &Handle,
    output: &Handle,
    channels: usize,
    samples: usize,
    width: usize,
) {
    let geom = LaunchGeometry::channels_samples(client, channels, samples);
    let total = channels * samples;
    unsafe {
        sparse_rows_multiply_kernel::launch::<F, R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(input.clone(), total),
            ArrayArg::from_raw_parts(values.clone(), channels * width),
            ArrayArg::from_raw_parts(indices.clone(), channels * width),
            ArrayArg::from_raw_parts(output.clone(), total),
            channels as u32,
            samples as u32,
            width as u32,
        );
    }
}

/// A `[C, C]` spatial operator uploaded once, dense or as sparse rows (chosen by
/// [`SPARSE_MAX_ROW_FILL`] from the matrix itself).
#[derive(Debug, Clone)]
pub enum DeviceSpatialMatrix {
    Dense { weights: Handle },
    Sparse { values: Handle, indices: Handle, width: usize },
}

impl DeviceSpatialMatrix {
    /// Uploads row-major `matrix` (`[channels, channels]`) as `F`.
    pub fn upload<R: Runtime, F: DspFloat>(client: &ComputeClient<R>, matrix: &[f32], channels: usize) -> Self {
        let sparse = SparseRows::from_dense(matrix, channels);
        if sparse.fill() <= SPARSE_MAX_ROW_FILL {
            Self::Sparse {
                values: buffer::upload(client, &cast_f32::<F>(&sparse.values)),
                indices: buffer::upload(client, &sparse.indices),
                width: sparse.width,
            }
        } else {
            Self::Dense { weights: buffer::upload(client, &cast_f32::<F>(matrix)) }
        }
    }

    /// `output = W · input` on a `[channels, samples]` buffer of `F`.
    pub fn apply<R: Runtime, F: DspFloat>(&self, client: &ComputeClient<R>, input: &Handle, output: &Handle, channels: usize, samples: usize) {
        match self {
            Self::Dense { weights } => execute_spatial_matrix_multiply::<R, F>(client, input, weights, output, channels, samples),
            Self::Sparse { values, indices, width } => {
                execute_sparse_rows_multiply::<R, F>(client, input, values, indices, output, channels, samples, *width)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::SurfaceLaplacian;

    fn sparse_matches_dense<R: Runtime>(client: &ComputeClient<R>) {
        let lap = SurfaceLaplacian::from_grid_2d(4, 8);
        let (channels, samples) = (32usize, 257usize);
        let x: Vec<f32> = (0..channels * samples).map(|i| ((i * 7919) % 211) as f32 - 100.0).collect();
        let input = buffer::upload(client, &x);

        let op = DeviceSpatialMatrix::upload::<R, f32>(client, &lap.matrix, channels);
        assert!(matches!(op, DeviceSpatialMatrix::Sparse { width: 5, .. }), "a 5-point Laplacian is stored as 5-wide rows");
        let out = buffer::empty::<R, f32>(client, x.len());
        op.apply::<R, f32>(client, &input, &out, channels, samples);
        let got = buffer::download::<R, f32>(client, out);

        let want = lap.apply_cpu(&x, channels, samples);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-3, "{}: {g} vs {w}", R::name(client));
        }
    }
    runtime_test!(test_sparse_rows_match_dense, sparse_matches_dense);
}
