use dsp_base::LaunchGeometry;

pub fn run_inspect(channels: usize, samples: usize) {
    println!("=== Hardware-Aware Launch Geometry Inspector ===");
    let available_parallelism = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!("Detected CPU Physical/Logical Threads: {available_parallelism}");
    println!("Simulated Matrix Size:                {channels} channels × {samples} samples");

    // 1D Geometry
    let geom_1d_cpu = LaunchGeometry::for_1d(channels * samples, true);
    let geom_1d_gpu = LaunchGeometry::for_1d(channels * samples, false);

    println!("\n[1D Kernel Geometry (Scaling / Baseline / Elementwise)]");
    println!(
        "  CPU: CubeDim (x={}, y={}, z={}), Chunk Size: {}",
        geom_1d_cpu.cube_dim.x, geom_1d_cpu.cube_dim.y, geom_1d_cpu.cube_dim.z, geom_1d_cpu.chunk_size
    );
    println!(
        "  GPU: CubeDim (x={}, y={}, z={}), Chunk Size: {}",
        geom_1d_gpu.cube_dim.x, geom_1d_gpu.cube_dim.y, geom_1d_gpu.cube_dim.z, geom_1d_gpu.chunk_size
    );

    // 2D Multi-Channel Geometry (Smooth 1..N scaling)
    let geom_2d_gpu = LaunchGeometry::for_channels_and_samples(channels, samples, false);
    println!("\n[2D Multi-Channel Geometry (IIR / Notch / CAR / TKEO)]");
    println!(
        "  GPU Workgroup (CubeDim): ({}, {}, {})",
        geom_2d_gpu.cube_dim.x, geom_2d_gpu.cube_dim.y, geom_2d_gpu.cube_dim.z
    );
    println!(
        "  GPU Grid (CubeCount):    {:?}",
        geom_2d_gpu.cube_count
    );
    println!("  Dynamic Warp Alignment:  Optimal for {} channels", channels);
}
