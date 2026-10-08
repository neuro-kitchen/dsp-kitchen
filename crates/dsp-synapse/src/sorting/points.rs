//! Point sets on the device for clustering ([`mod@super::hdbscan`], [`mod@super::kmeans`]): uploaded once,
//! **feature-major** (`[d, n]`, see [`super::kernels::points`]), subsets gathered on the device.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_base::core::buffer;
use dsp_core::compute::LaunchGeometry;

use super::kernels::points::{block_sums_kernel, gather_points_kernel};

/// Values each unit of [`block_sums`] adds before the host adds the blocks in `f64`.
pub const SUM_BLOCK: usize = 1024;

/// `n` points of `d` features on the device, feature-major (`f32`).
#[derive(Clone)]
pub struct DevicePoints {
    pub handle: Handle,
    pub n: usize,
    pub d: usize,
}

impl DevicePoints {
    /// Uploads `x` (`[n, d]` row-major, as callers hold points), transposed to feature-major.
    pub fn upload(client: &Client, x: &[f32], n: usize, d: usize) -> Self {
        assert_eq!(x.len(), n * d, "points: data size mismatch");
        let mut t = vec![0.0f32; n * d];
        for i in 0..n {
            for f in 0..d {
                t[f * n + i] = x[i * d + f];
            }
        }
        Self { handle: buffer::upload(client, &t), n, d }
    }

    /// The points `index` (in that order), gathered on the device.
    pub fn gather(&self, client: &Client, index: &[u32]) -> Self {
        let (n, d, m) = (self.n, self.d, index.len());
        let out = buffer::empty::<f32>(client, m * d);
        if m > 0 {
            let geom = LaunchGeometry::elementwise(client, m * d);
            // SAFETY: `handle` holds `d · n`, `index` `m` (all < n), `out` `d · m` values
            unsafe {
                gather_points_kernel::launch::<f32>(
                    client,
                    geom.cube_count,
                    geom.cube_dim,
                    BufferArg::from_raw_parts(self.handle.clone(), d * n),
                    BufferArg::from_raw_parts(buffer::upload(client, index), m),
                    BufferArg::from_raw_parts(out.clone(), m * d),
                    n as u32,
                    m as u32,
                    d as u32,
                );
            }
        }
        Self { handle: out, n: m, d }
    }

    /// Features of point `i` (`d` values; reads only those).
    pub fn point(&self, client: &Client, i: usize) -> Vec<f32> {
        let one = self.gather(client, &[i as u32]);
        buffer::download_prefix::<f32>(client, one.handle, self.d)
    }
}

/// Block sums of the first `n` values of a device buffer (`f32`), [`SUM_BLOCK`] values per block,
/// read back for the host to add in `f64`.
pub fn block_sums(client: &Client, values: &Handle, n: usize) -> Vec<f32> {
    let blocks = n.div_ceil(SUM_BLOCK).max(1);
    let sums = buffer::empty::<f32>(client, blocks);
    let geom = LaunchGeometry::elementwise(client, blocks);
    // SAFETY: `values` holds at least `n` values, `sums` `blocks`
    unsafe {
        block_sums_kernel::launch::<f32>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(values.clone(), n.max(1)),
            BufferArg::from_raw_parts(sums.clone(), blocks),
            n as u32,
            SUM_BLOCK as u32,
            blocks as u32,
        );
    }
    buffer::download_prefix::<f32>(client, sums, blocks)
}

/// `Σ` of the first `n` values of a device buffer, in `f64` over [`block_sums`].
pub fn device_sum(client: &Client, values: &Handle, n: usize) -> f64 {
    block_sums(client, values, n).iter().map(|&v| v as f64).sum()
}
