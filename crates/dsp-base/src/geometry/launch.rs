use cubecl::prelude::*;

/// Kernel launch geometry for the three access patterns used by the DSP kernels.
///
/// There is one geometry per pattern and no device-specific layout: CubeCL maps cubes and units to
/// whatever runtime executes the kernel (CPU threads, GPU workgroups).
#[derive(Debug, Clone)]
pub struct LaunchGeometry {
    pub cube_dim: CubeDim,
    pub cube_count: CubeCount,
}

/// Units per cube for elementwise kernels.
const ELEMENTWISE_UNITS: u32 = 256;
/// Tile of the channels × samples pattern: 64 samples (x) × 4 channels (y).
const TILE_SAMPLES: u32 = 64;
const TILE_CHANNELS: u32 = 4;
/// Units per cube for one-unit-per-channel kernels.
const CHANNEL_UNITS: u32 = 64;

impl LaunchGeometry {
    /// One unit per element, indexed with the linear `ABSOLUTE_POS`. The cubes are spread over the
    /// runtime's cube-count dimensions, so kernels must bound-check `ABSOLUTE_POS`.
    pub fn elementwise<R: Runtime>(client: &ComputeClient<R>, num_elements: usize) -> Self {
        let cube_dim = CubeDim::new_1d(ELEMENTWISE_UNITS);
        let cube_count = cubecl::calculate_cube_count_elemwise(client, num_elements.max(1), cube_dim);
        Self { cube_dim, cube_count }
    }

    /// One unit per `(channel, sample)` of a `[channels, samples]` buffer: `ABSOLUTE_POS_X` = sample,
    /// `ABSOLUTE_POS_Y` = channel. Kernels bound-check both.
    ///
    /// # Panics
    /// If `samples` needs more cubes along x than the runtime allows (≥ 4 M samples on WGPU).
    pub fn channels_samples<R: Runtime>(client: &ComputeClient<R>, channels: usize, samples: usize) -> Self {
        let count_x = (samples.max(1) as u32).div_ceil(TILE_SAMPLES);
        let count_y = (channels.max(1) as u32).div_ceil(TILE_CHANNELS);
        let max = client.properties().hardware.max_cube_count;
        assert!(
            count_x <= max.0 && count_y <= max.1,
            "{channels} channels × {samples} samples exceed the runtime's cube grid; process shorter chunks"
        );
        Self {
            cube_dim: CubeDim::new_2d(TILE_SAMPLES, TILE_CHANNELS),
            cube_count: CubeCount::Static(count_x, count_y, 1),
        }
    }

    /// One unit per sample for kernels that walk all channels of a sample (`ABSOLUTE_POS_X` =
    /// sample, bound-checked by the kernel).
    ///
    /// # Panics
    /// If `samples` needs more cubes along x than the runtime allows (≥ 16 M samples on WGPU).
    pub fn per_sample<R: Runtime>(client: &ComputeClient<R>, samples: usize) -> Self {
        let count_x = (samples.max(1) as u32).div_ceil(ELEMENTWISE_UNITS);
        assert!(
            count_x <= client.properties().hardware.max_cube_count.0,
            "{samples} samples exceed the runtime's cube grid; process shorter chunks"
        );
        Self { cube_dim: CubeDim::new_1d(ELEMENTWISE_UNITS), cube_count: CubeCount::Static(count_x, 1, 1) }
    }

    /// One unit per channel for kernels that walk each channel sequentially in time:
    /// `ABSOLUTE_POS_X` = channel, bound-checked by the kernel.
    pub fn per_channel(channels: usize) -> Self {
        Self {
            cube_dim: CubeDim::new_1d(CHANNEL_UNITS),
            cube_count: CubeCount::Static((channels.max(1) as u32).div_ceil(CHANNEL_UNITS), 1, 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    fn cubes(count: &CubeCount) -> u64 {
        match count {
            CubeCount::Static(x, y, z) => *x as u64 * *y as u64 * *z as u64,
            _ => unreachable!(),
        }
    }

    #[test]
    fn elementwise_covers_large_buffers_within_limits() {
        let client = WgpuRuntime::client(&WgpuDevice::default());
        let max = client.properties().hardware.max_cube_count;
        // 384 channels × 10 s at 30 kHz: more cubes than one dimension allows.
        let n = 384 * 330_000;
        let geom = LaunchGeometry::elementwise(&client, n);
        let CubeCount::Static(x, y, z) = geom.cube_count else { unreachable!() };
        assert!(x <= max.0 && y <= max.1 && z <= max.2);
        assert!(cubes(&geom.cube_count) * ELEMENTWISE_UNITS as u64 >= n as u64);
    }

    #[test]
    fn tiles_and_channels_cover_their_ranges() {
        let client = WgpuRuntime::client(&WgpuDevice::default());
        let geom = LaunchGeometry::channels_samples(&client, 385, 30_001);
        let CubeCount::Static(x, y, _) = geom.cube_count else { unreachable!() };
        assert!(x * TILE_SAMPLES >= 30_001 && y * TILE_CHANNELS >= 385);
        let geom = LaunchGeometry::per_channel(385);
        assert!(cubes(&geom.cube_count) * CHANNEL_UNITS as u64 >= 385);
    }
}
