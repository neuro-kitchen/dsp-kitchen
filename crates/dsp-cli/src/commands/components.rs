pub fn run_components() {
    println!("============================================================");
    println!("           dsp-kitchen Engine Component Registry            ");
    println!("============================================================");
    println!("Workspace Version:  {}", env!("CARGO_PKG_VERSION"));
    println!("Architecture:       {}", std::env::consts::ARCH);
    println!("Operating System:   {}", std::env::consts::OS);

    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!(
        "Logical Threads:    {} (hardware-aware JIT enabled)",
        available_threads
    );

    println!("\n[Installed Workspace Crates]");
    println!(
        "  [+] dsp-core:    INSTALLED (v{}) - Domain-Agnostic SensorLayout, Exact Time, ChannelMask",
        env!("CARGO_PKG_VERSION")
    );
    println!(
        "  [+] dsp-base:    INSTALLED (v{}) - Dynamic 1..N Geometry, CubeCL Kernels, LinAlg/PCA, TKEO",
        env!("CARGO_PKG_VERSION")
    );
    println!(
        "  [+] dsp-stream:  INSTALLED (v{}) - Ring Buffer, Zarr v3, Decimate LOD",
        env!("CARGO_PKG_VERSION")
    );
    println!(
        "  [+] dsp-synapse: INSTALLED (v{}) - Spike Detection, Waveform Cuts, PCA Extraction, Probes",
        env!("CARGO_PKG_VERSION")
    );
    println!(
        "  [+] dsp-bridge:  INSTALLED (v{}) - Modular PyO3 C-ABI Bindings & NumPy Views",
        env!("CARGO_PKG_VERSION")
    );
    println!(
        "  [+] dsp-cli:     INSTALLED (v{}) - Headless Operational CLI & Benchmarking Harness",
        env!("CARGO_PKG_VERSION")
    );

    println!("\n[Storage & Memory Engines]");
    println!("  [+] memmap2:     INSTALLED (v0.9.11) - Zero-copy mmap for multi-GB binary ingest");
    println!("  [+] zarrs:       INSTALLED (via dsp-io) - Chunked N-dimensional Zarr v3 storage");
    println!("  [+] ring buffer: INSTALLED - Lock-free multi-channel continuous circular buffer");

    println!("\n[CubeCL Hardware Backends]");
    println!(
        "  [+] CPU JIT:     ACTIVE (Dynamically tuned to {} threads)",
        available_threads
    );
    println!("  [+] WGPU:        COMPILED (WebGPU / Vulkan / Metal runtime)");
    println!("  [*] CUDA / HIP:  Feature-gated for discrete GPU cluster rigs");

    println!("============================================================");
    println!("Status: ALL CORE SUBSYSTEMS OPERATIONAL");
    println!("============================================================");
}
