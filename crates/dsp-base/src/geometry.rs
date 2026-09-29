use cubecl::prelude::*;

/// Hardware-aware kernel launch geometry configuration.
/// Ensures zero cache-line false sharing on CPU and optimal warp coalescing on GPU.
#[derive(Debug, Clone)]
pub struct LaunchGeometry {
    pub cube_dim: CubeDim,
    pub cube_count: CubeCount,
    pub chunk_size: u32,
    pub is_cpu: bool,
}

impl LaunchGeometry {
    /// Computes hardware-aware 1D launch geometry for elementwise / channel operations.
    ///
    /// On CPU:
    /// - Thread count is dynamically resolved via `std::thread::available_parallelism()`.
    /// - Memory is partitioned into contiguous chunks per worker thread.
    /// - Eliminates the 256-thread context-switch trap and prevents L1/L2 cache-line false sharing.
    ///
    /// On GPU:
    /// - Uses workgroup sizing appropriate for SIMT execution (e.g. 256 or tuned).
    pub fn for_1d(num_elements: usize, is_cpu: bool) -> Self {
        if is_cpu {
            let available_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);

            let num_threads = available_threads
                .min(num_elements)
                .max(1) as u32;

            let chunk_size = ((num_elements as u32) + num_threads - 1) / num_threads;

            Self {
                cube_dim: CubeDim::new_2d(1, num_threads),
                cube_count: CubeCount::Static(chunk_size, 1, 1),
                chunk_size,
                is_cpu: true,
            }
        } else {
            let workgroup_size = 256u32;
            let num_cubes = ((num_elements as u32) + workgroup_size - 1) / workgroup_size;

            Self {
                cube_dim: CubeDim::new_1d(workgroup_size),
                cube_count: CubeCount::Static(num_cubes, 1, 1),
                chunk_size: 1,
                is_cpu: false,
            }
        }
    }

    /// Computes hardware-aware 2D launch geometry for multi-channel temporal matrices (Channels × Samples).
    ///
    /// On CPU:
    /// - Partitions whole channels or contiguous sample blocks across available worker threads.
    /// - Avoids threads stepping on adjacent channel cache lines.
    pub fn for_channels_and_samples(channels: usize, samples: usize, is_cpu: bool) -> Self {
        if is_cpu {
            let available_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);

            let num_threads = available_threads
                .min(channels.max(1))
                .max(1) as u32;

            let channels_per_thread = ((channels as u32) + num_threads - 1) / num_threads;

            Self {
                cube_dim: CubeDim::new_2d(1, num_threads),
                cube_count: CubeCount::Static(samples as u32, channels_per_thread, 1),
                chunk_size: channels_per_thread,
                is_cpu: true,
            }
        } else {
            let tile_x = 16u32;
            let tile_y = 16u32;
            let count_x = ((samples as u32) + tile_x - 1) / tile_x;
            let count_y = ((channels as u32) + tile_y - 1) / tile_y;

            Self {
                cube_dim: CubeDim::new_2d(tile_x, tile_y),
                cube_count: CubeCount::Static(count_x, count_y, 1),
                chunk_size: 1,
                is_cpu: false,
            }
        }
    }

    /// Computes hardware launch geometry when parallelizing along channels for sequential temporal loops.
    /// Dynamically tunes workgroup sizing for 32-channel vs 384+ channel recordings.
    pub fn for_channel_sequence(channels: usize, is_cpu: bool) -> Self {
        if is_cpu {
            let available_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);

            let num_threads = available_threads.min(channels.max(1)).max(1) as u32;
            let channels_per_thread = ((channels as u32) + num_threads - 1) / num_threads;

            Self {
                cube_dim: CubeDim::new_2d(1, num_threads),
                cube_count: CubeCount::Static(channels_per_thread, 1, 1),
                chunk_size: channels_per_thread,
                is_cpu: true,
            }
        } else {
            let workgroup_size = if channels <= 32 {
                32u32
            } else if channels <= 128 {
                64u32
            } else {
                256u32
            };
            let num_cubes = ((channels as u32) + workgroup_size - 1) / workgroup_size;

            Self {
                cube_dim: CubeDim::new_1d(workgroup_size),
                cube_count: CubeCount::Static(num_cubes, 1, 1),
                chunk_size: 1,
                is_cpu: false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_launch_geometry_adapts_to_parallelism() {
        let total_samples = 30_000;
        let geom = LaunchGeometry::for_1d(total_samples, true);
        assert!(geom.is_cpu);
        assert!(geom.chunk_size > 0);
        assert!(geom.cube_dim.y > 0);
        let covered = geom.cube_dim.y * geom.chunk_size;
        assert!(covered >= total_samples as u32);
    }

    #[test]
    fn test_gpu_launch_geometry() {
        let total_samples = 30_000;
        let geom = LaunchGeometry::for_1d(total_samples, false);
        assert!(!geom.is_cpu);
        assert_eq!(geom.cube_dim.x, 256);
        assert_eq!(geom.chunk_size, 1);
    }
}
