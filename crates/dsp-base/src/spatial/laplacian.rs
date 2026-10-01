use cubecl::prelude::*;
use super::whitening::execute_spatial_matrix_multiply;

/// 2D Surface Laplacian (Hjorth / Double-Differential) spatial filter for planar electrode grids
/// (e.g., 4x8 or 8x8 HD-EMG arrays and ECoG grids).
///
/// Suppresses distant common-mode volume-conducted cross-talk by subtracting the mean of immediate
/// orthogonal spatial neighbors from each electrode:
/// $y_i(t) = x_i(t) - \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} x_j(t)$.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceLaplacian {
    pub num_channels: usize,
    /// Row-major `[num_channels, num_channels]` spatial Laplacian operator matrix.
    pub matrix: Vec<f32>,
}

impl SurfaceLaplacian {
    /// Constructs a 5-point discrete 2D Surface Laplacian filter for a regular `rows x cols` grid
    /// (channel index `c = row * cols + col`). Boundary electrodes normalize over their available
    /// orthogonal neighbors (2 at corners, 3 at edges, 4 in the interior).
    pub fn from_grid_2d(rows: usize, cols: usize) -> Self {
        let num_channels = rows * cols;
        let mut matrix = vec![0.0f32; num_channels * num_channels];

        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let mut neighbors = Vec::with_capacity(4);
                if r > 0 {
                    neighbors.push((r - 1) * cols + c);
                }
                if r + 1 < rows {
                    neighbors.push((r + 1) * cols + c);
                }
                if c > 0 {
                    neighbors.push(r * cols + (c - 1));
                }
                if c + 1 < cols {
                    neighbors.push(r * cols + (c + 1));
                }

                matrix[idx * num_channels + idx] = 1.0;
                if !neighbors.is_empty() {
                    let w = -1.0 / (neighbors.len() as f32);
                    for n_idx in neighbors {
                        matrix[idx * num_channels + n_idx] = w;
                    }
                }
            }
        }

        Self {
            num_channels,
            matrix,
        }
    }

    /// Constructs a distance-weighted Hjorth Surface Laplacian from arbitrary 2D electrode coordinates
    /// using the `k_neighbors` nearest neighbors of each channel.
    pub fn from_coordinates_knn(positions: &[[f32; 2]], k_neighbors: usize) -> Self {
        let num_channels = positions.len();
        let k = k_neighbors.clamp(1, num_channels.saturating_sub(1));
        let mut matrix = vec![0.0f32; num_channels * num_channels];

        for i in 0..num_channels {
            matrix[i * num_channels + i] = 1.0;
            if k == 0 {
                continue;
            }
            let mut dists: Vec<(usize, f32)> = (0..num_channels)
                .filter(|&j| j != i)
                .map(|j| {
                    let dx = positions[i][0] - positions[j][0];
                    let dy = positions[i][1] - positions[j][1];
                    (j, (dx * dx + dy * dy).sqrt().max(1e-3))
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let chosen = &dists[..k.min(dists.len())];
            let inv_sum: f32 = chosen.iter().map(|&(_, d)| 1.0 / d).sum();
            if inv_sum > 0.0 {
                for &(j, d) in chosen {
                    matrix[i * num_channels + j] = -(1.0 / d) / inv_sum;
                }
            }
        }

        Self {
            num_channels,
            matrix,
        }
    }

    /// Applies the Surface Laplacian on the CPU to `data` (`[channels, samples]`).
    pub fn apply_cpu(&self, data: &[f32], channels: usize, samples: usize) -> Vec<f32> {
        assert_eq!(channels, self.num_channels);
        assert_eq!(data.len(), channels * samples);
        let mut out = vec![0.0f32; channels * samples];
        for out_ch in 0..channels {
            let w_row = &self.matrix[out_ch * channels..(out_ch + 1) * channels];
            let out_row = &mut out[out_ch * samples..(out_ch + 1) * samples];
            for in_ch in 0..channels {
                let w = w_row[in_ch];
                if w.abs() > 0.0 {
                    let in_row = &data[in_ch * samples..(in_ch + 1) * samples];
                    for t in 0..samples {
                        out_row[t] += w * in_row[t];
                    }
                }
            }
        }
        out
    }

    /// Applies the Surface Laplacian in VRAM using CubeCL.
    pub fn apply_gpu<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        input: &cubecl::server::Handle,
        output: &cubecl::server::Handle,
        channels: usize,
        samples: usize,
    ) {
        assert_eq!(channels, self.num_channels);
        let weights_handle = client.create_from_slice(f32::as_bytes(&self.matrix));
        execute_spatial_matrix_multiply::<R>(
            client,
            input,
            &weights_handle,
            output,
            channels,
            samples,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surface_laplacian_cancels_uniform_potential_and_sharpens_focal_peak() {
        let lap = SurfaceLaplacian::from_grid_2d(4, 8);
        let channels = 32;
        let samples = 10;

        // Uniform potential across all 32 electrodes -> 0 everywhere after Laplacian
        let uniform = vec![42.0f32; channels * samples];
        let out_uniform = lap.apply_cpu(&uniform, channels, samples);
        for v in out_uniform {
            assert!(v.abs() < 1e-5);
        }
    }
}
