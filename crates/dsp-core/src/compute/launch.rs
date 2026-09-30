use cubecl::prelude::*;

/// Kernel launch geometry for the access patterns used by the DSP kernels.
///
/// Cube sizes come from the runtime ([`CubeDim::new`]: its plane size along x, a plane count set
/// by the runtime's cores or limits along y), never from constants, and cube counts from the
/// problem size within the runtime's grid limits. CubeCL maps cubes and units to whatever executes
/// the kernel (CPU threads, GPU workgroups).
#[derive(Debug, Clone)]
pub struct LaunchGeometry {
    pub cube_dim: CubeDim,
    pub cube_count: CubeCount,
}

/// Sample index of the unit in a [`LaunchGeometry::channels_samples`] or
/// [`LaunchGeometry::per_sample`] launch. Samples run along x and continue along z when they need
/// more cubes than one grid dimension holds; kernels bound-check the result.
#[cube]
pub fn sample_position() -> u32 {
    (CUBE_POS_Z * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X
}

/// Channel (row) index of the unit in a [`LaunchGeometry::channels_samples`] launch.
#[cube]
pub fn channel_position() -> u32 {
    ABSOLUTE_POS_Y
}

impl LaunchGeometry {
    /// One unit per element, indexed with the linear `ABSOLUTE_POS`. The cubes are spread over the
    /// runtime's cube-count dimensions, so kernels must bound-check `ABSOLUTE_POS`.
    pub fn elementwise<R: Runtime>(client: &ComputeClient<R>, num_elements: usize) -> Self {
        let n = num_elements.max(1);
        let cube_dim = CubeDim::new(client, n);
        let cube_count = cubecl::calculate_cube_count_elemwise(client, n, cube_dim);
        Self { cube_dim, cube_count }
    }

    /// One unit per `(channel, sample)` of a `[channels, samples]` buffer: [`sample_position`]
    /// and [`channel_position`]. The cube's x size is the runtime's plane, so a plane walks
    /// neighbouring samples of one channel.
    ///
    /// # Panics
    /// If the channels or samples need more cubes than the runtime's whole grid holds.
    pub fn channels_samples<R: Runtime>(client: &ComputeClient<R>, channels: usize, samples: usize) -> Self {
        let cube_dim = CubeDim::new(client, channels.max(1) * samples.max(1));
        let (x, z) = Self::spill(client, samples, cube_dim.x);
        let y = (channels.max(1) as u32).div_ceil(cube_dim.y);
        assert!(
            y <= client.properties().hardware.max_cube_count.1,
            "{channels} channels exceed the runtime's cube grid"
        );
        Self { cube_dim, cube_count: CubeCount::Static(x, y, z) }
    }

    /// One unit per sample for kernels that walk all channels of a sample ([`sample_position`]).
    pub fn per_sample<R: Runtime>(client: &ComputeClient<R>, samples: usize) -> Self {
        let cube_dim = Self::flat(client, samples);
        let (x, z) = Self::spill(client, samples, cube_dim.x);
        Self { cube_dim, cube_count: CubeCount::Static(x, 1, z) }
    }

    /// One unit per channel for kernels that walk each channel sequentially in time:
    /// `ABSOLUTE_POS_X` = channel, bound-checked by the kernel.
    pub fn per_channel<R: Runtime>(client: &ComputeClient<R>, channels: usize) -> Self {
        let cube_dim = Self::flat(client, channels);
        Self { cube_dim, cube_count: CubeCount::Static((channels.max(1) as u32).div_ceil(cube_dim.x), 1, 1) }
    }

    /// Units that execute in lock-step (the runtime's plane / warp / subgroup size; 1 where every
    /// unit runs on its own, as on the CPU runtime). It is the x size of
    /// [`Self::channels_samples`] cubes, so consecutive `sample_position`s of a plane share a cube.
    pub fn plane_lanes<R: Runtime>(client: &ComputeClient<R>) -> u32 {
        client.properties().hardware.plane_size_max.max(1)
    }

    /// The runtime's cube for `work` units, flattened onto x (for kernels indexed along x only).
    fn flat<R: Runtime>(client: &ComputeClient<R>, work: usize) -> CubeDim {
        CubeDim::new_1d(CubeDim::new(client, work.max(1)).num_elems())
    }

    /// Cube counts `(x, z)` covering `units` along x with `per_cube` units per cube, continuing
    /// along z past the grid's x limit.
    fn spill<R: Runtime>(client: &ComputeClient<R>, units: usize, per_cube: u32) -> (u32, u32) {
        let max = client.properties().hardware.max_cube_count;
        let cubes = (units.max(1) as u64).div_ceil(per_cube as u64);
        let x = cubes.min(max.0 as u64);
        let z = cubes.div_ceil(x);
        assert!(z <= max.2 as u64, "{units} samples exceed the runtime's cube grid");
        (x as u32, z as u32)
    }
}

#[cfg(all(test, feature = "wgpu"))]
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
    fn geometries_cover_their_ranges_within_limits() {
        let client = WgpuRuntime::client(&WgpuDevice::default());
        let max = client.properties().hardware.max_cube_count;
        let within = |c: &CubeCount| matches!(c, CubeCount::Static(x, y, z) if *x <= max.0 && *y <= max.1 && *z <= max.2);

        // 384 channels × 10 s at 30 kHz: more cubes than one dimension allows.
        let n = 384 * 330_000;
        let geom = LaunchGeometry::elementwise(&client, n);
        assert!(within(&geom.cube_count) && cubes(&geom.cube_count) * geom.cube_dim.num_elems() as u64 >= n as u64);

        // Samples beyond one grid dimension spill into z.
        for samples in [30_001usize, 5_000_000] {
            let geom = LaunchGeometry::channels_samples(&client, 385, samples);
            let CubeCount::Static(x, y, z) = geom.cube_count else { unreachable!() };
            assert!(within(&geom.cube_count));
            assert!(x as u64 * z as u64 * geom.cube_dim.x as u64 >= samples as u64 && y * geom.cube_dim.y >= 385);
            assert_eq!(geom.cube_dim.x, LaunchGeometry::plane_lanes(&client));
        }
        let geom = LaunchGeometry::per_channel(&client, 385);
        assert!(cubes(&geom.cube_count) * geom.cube_dim.num_elems() as u64 >= 385);
    }
}
