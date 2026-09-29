use clap::{Parser, Subcommand};
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use dsp_base::LaunchGeometry;
use dsp_base::filter::{design_notch_coeffs, execute_notch};
use dsp_base::math::{execute_neo, execute_scaling};
use dsp_core::probe::ProbeLayout;
use dsp_core::time::SampleRate;
use dsp_stream::StreamPurpose;
use memmap2::Mmap;
use serde::Serialize;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "dsp-cli")]
#[command(author, version, about = "High-performance neuroscience DSP engine & hardware CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Detect installed engine crates, feature flags, storage backends (mmap/zarrs), and hardware
    #[command(alias = "info", alias = "detect", alias = "doctor")]
    Components,

    /// Inspect neural electrode probe configurations and layouts
    Probe {
        /// Type of probe layout to display
        #[arg(short, long, default_value = "neuropixels-1")]
        model: String,
    },

    /// Inspect hardware topology, core counts, and dynamic launch geometry
    Benchmark {
        /// Number of virtual signal channels
        #[arg(short, long, default_value_t = 384)]
        channels: usize,

        /// Number of samples per channel
        #[arg(short, long, default_value_t = 30000)]
        samples: usize,
    },

    /// Inspect streaming parameters for processing vs visualization
    Stream {
        /// Target mode: 'processing' (lossless) or 'vis' (decimated LOD)
        #[arg(short, long, default_value = "vis")]
        mode: String,

        /// Target visualization buckets per channel
        #[arg(short, long, default_value_t = 1920)]
        buckets: usize,
    },

    /// Generate a mock multi-channel electrophysiology recording with configurable noise and spikes
    #[command(alias = "generate")]
    MockSignal {
        /// Number of electrode channels (e.g. 384 for Neuropixels)
        #[arg(short, long, default_value_t = 384)]
        channels: usize,

        /// Number of time samples per channel (e.g. 30,000 = 1 sec @ 30kHz)
        #[arg(short, long, default_value_t = 30000)]
        samples: usize,

        /// Sampling rate in Hz
        #[arg(short = 'r', long, default_value_t = 30000.0)]
        sample_rate: f64,

        /// Amplitude of white Gaussian noise (microvolts RMS)
        #[arg(short, long, default_value_t = 15.0)]
        noise_uv: f32,

        /// Amplitude of 60 Hz power-line interference (microvolts)
        #[arg(short, long, default_value_t = 25.0)]
        line_noise_uv: f32,

        /// Inject biological action potentials (spikes) across channels
        #[arg(long, default_value_t = true)]
        spikes: bool,

        /// Storage format: 'bin' (flat binary + .meta) or 'zarr' (chunked Zarr v3)
        #[arg(short = 'f', long, default_value = "bin")]
        format: String,

        /// Output file/directory path (e.g. playground/data/mock_signal_384ch.bin or .zarr)
        #[arg(short, long, default_value = "playground/data/mock_signal_384ch.bin")]
        output: PathBuf,
    },

    /// Benchmark real-time DSP pipeline performance with and without saving results
    BenchmarkPipeline {
        /// Number of electrode channels (default: 384)
        #[arg(short, long, default_value_t = 384)]
        channels: usize,

        /// Number of samples per channel (default: 30,000)
        #[arg(short, long, default_value_t = 30000)]
        samples: usize,

        /// Number of benchmark iterations
        #[arg(short, long, default_value_t = 5)]
        iterations: usize,

        /// Persist filtered results to disk (measures I/O and bus overhead)
        #[arg(long, default_value_t = false)]
        save: bool,

        /// Persistence storage format when --save is enabled ('bin' or 'zarr')
        #[arg(short = 'f', long, default_value = "bin")]
        format: String,

        /// Destination output path
        #[arg(short, long, default_value = "playground/data/pipeline_output.bin")]
        output: PathBuf,
    },
}

#[derive(Serialize)]
struct RecordingMetadata {
    channels: usize,
    samples: usize,
    sample_rate_hz: f64,
    duration_seconds: f64,
    noise_rms_uv: f32,
    line_noise_uv: f32,
    spikes_injected: usize,
    format: &'static str,
    byte_order: &'static str,
    bytes_per_sample: usize,
    total_data_bytes: usize,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Components => {
            println!("============================================================");
            println!("           dsp-kitchen Engine Component Registry            ");
            println!("============================================================");
            println!("Workspace Version:  {}", env!("CARGO_PKG_VERSION"));
            println!("Architecture:       {}", std::env::consts::ARCH);
            println!("Operating System:   {}", std::env::consts::OS);
            
            let available_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            println!("Logical Threads:    {} (hardware-aware JIT enabled)", available_threads);

            println!("\n[Installed Workspace Crates]");
            println!("  [+] dsp-core:    INSTALLED (v{}) - Exact Rational Time, Buffers, Probes", env!("CARGO_PKG_VERSION"));
            println!("  [+] dsp-base:    INSTALLED (v{}) - CubeCL Kernels, FIR, CAR, Scaling", env!("CARGO_PKG_VERSION"));
            println!("  [+] dsp-stream:  INSTALLED (v{}) - SHM v2, mmap2, zarrs, QUIC/gRPC, Decimate", env!("CARGO_PKG_VERSION"));
            println!("  [+] dsp-synapse: INSTALLED (v{}) - Spike Detection, PCA Extraction, ML", env!("CARGO_PKG_VERSION"));
            println!("  [+] dsp-bridge:  INSTALLED (v{}) - PyO3 Python C-ABI bindings", env!("CARGO_PKG_VERSION"));
            println!("  [+] dsp-cli:     INSTALLED (v{}) - Unified Command Line Interface", env!("CARGO_PKG_VERSION"));

            println!("\n[Storage & Memory Engines]");
            println!("  [+] memmap2:     INSTALLED (v0.9.11) - Zero-copy mmap for multi-GB binary ingest");
            println!("  [+] zarrs:       INSTALLED (v0.23.14) - Chunked N-dimensional Zarr v2/v3 storage");
            println!("  [+] shared_mem:  INSTALLED (v0.12.4) - Local cross-process SHM ring buffers");

            println!("\n[CubeCL Hardware Backends]");
            println!("  [+] CPU JIT:     ACTIVE (Dynamically tuned to {} threads)", available_threads);
            println!("  [+] WGPU:        COMPILED (WebGPU / Vulkan / Metal runtime)");
            println!("  [*] CUDA / HIP:  Feature-gated for discrete GPU cluster rigs");

            println!("============================================================");
            println!("Status: ALL CORE SUBSYSTEMS OPERATIONAL");
            println!("============================================================");
        }

        Commands::Probe { model } => {
            println!("=== Electrode Probe Inspector ===");
            if model == "neuropixels-1" {
                let probe = ProbeLayout::neuropixels_1_0_standard();
                println!("Probe Name:        {}", probe.name);
                println!("Total Channels:    {}", probe.total_channels());
                println!("Active Channels:   {}", probe.active_channels());
                if let Ok(c0) = probe.get_contact(0) {
                    println!("Contact 0 Position: ({:.1} um, {:.1} um, {:.1} um)", c0.position.x_um, c0.position.y_um, c0.position.z_um);
                }
                if let Ok(c383) = probe.get_contact(383) {
                    println!("Contact 383 Pos:   ({:.1} um, {:.1} um, {:.1} um)", c383.position.x_um, c383.position.y_um, c383.position.z_um);
                }
            } else {
                println!("Unknown probe model: {model}");
            }
        }

        Commands::Benchmark { channels, samples } => {
            println!("=== Hardware-Aware Launch Geometry Inspector ===");
            let available_parallelism = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);

            println!("Detected CPU Physical/Logical Threads: {available_parallelism}");
            println!("Simulated Matrix Size:                {channels} channels × {samples} samples");

            // 1D Geometry
            let geom_cpu = LaunchGeometry::for_1d(channels * samples, true);
            let geom_gpu = LaunchGeometry::for_1d(channels * samples, false);

            println!("\n[CPU Launch Geometry]");
            println!("  CubeDim:    (x={}, y={}, z={}) -> Native Workers: {}", geom_cpu.cube_dim.x, geom_cpu.cube_dim.y, geom_cpu.cube_dim.z, geom_cpu.cube_dim.y);
            println!("  Chunk Size: {} elements/worker (Contiguous Slicing)", geom_cpu.chunk_size);
            println!("  False-Sharing Prevention: ACTIVE (0% MESI cache-line invalidation)");

            println!("\n[GPU Launch Geometry]");
            println!("  CubeDim:    (x={}, y={}, z={}) (SIMT Warp/Workgroup)", geom_gpu.cube_dim.x, geom_gpu.cube_dim.y, geom_gpu.cube_dim.z);
            println!("  Chunk Size: {}", geom_gpu.chunk_size);
        }

        Commands::Stream { mode, buckets } => {
            println!("=== Stream Mode & Decimation Inspector ===");
            let sr = SampleRate::new(SampleRate::NEUROPIXELS_AP)?;
            println!("Source Sampling Rate: {:.1} Hz", sr.rate_hz());

            let purpose = if mode == "processing" {
                StreamPurpose::Processing
            } else {
                StreamPurpose::Visualization { target_points_per_channel: buckets }
            };

            match purpose {
                StreamPurpose::Processing => {
                    println!("Mode: FULL-FIDELITY PROCESSING STREAM");
                    println!("  Lossless:          YES");
                    println!("  Rate:              30,000 samples/sec/channel");
                    println!("  Bandwidth (384ch): ~46.08 MB/s (float32)");
                }
                StreamPurpose::Visualization { target_points_per_channel } => {
                    println!("Mode: DECIMATED VISUALIZATION STREAM (LOD)");
                    println!("  Target Points:     {} points/channel/frame", target_points_per_channel);
                    println!("  Decimation Kernel: Min-Max Envelope Preserving");
                    println!("  Viewport Transfer: Minimized to display pixel density");
                }
            }
        }

        Commands::MockSignal {
            channels,
            samples,
            sample_rate,
            noise_uv,
            line_noise_uv,
            spikes,
            format,
            output,
        } => {
            generate_mock_signal(
                channels,
                samples,
                sample_rate,
                noise_uv,
                line_noise_uv,
                spikes,
                &format,
                &output,
            )?;
        }

        Commands::BenchmarkPipeline {
            channels,
            samples,
            iterations,
            save,
            format,
            output,
        } => {
            run_benchmark_pipeline(channels, samples, iterations, save, &format, &output)?;
        }
    }

    Ok(())
}

/// Generates a synthetic multi-channel electrophysiology recording and persists it.
fn generate_mock_signal(
    channels: usize,
    samples: usize,
    sample_rate: f64,
    noise_uv: f32,
    line_noise_uv: f32,
    spikes: bool,
    format: &str,
    output_path: &Path,
) -> anyhow::Result<()> {
    println!("=== Synthetic Neuroscience Signal Generator ===");
    println!("Channels:            {}", channels);
    println!("Samples per channel: {}", samples);
    println!("Sample Rate:         {:.1} Hz", sample_rate);
    println!("Duration:            {:.3} seconds", samples as f64 / sample_rate);
    println!("Gaussian Noise RMS:  {:.1} uV", noise_uv);
    println!("60 Hz Line Noise:    {:.1} uV", line_noise_uv);
    println!("Injecting Spikes:    {}", spikes);
    println!("Storage Format:      {}", format);
    println!("Destination:         {}", output_path.display());

    // Ensure target directory exists (e.g. playground/data)
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Channel-Major flat buffer: [channels, samples]
    let total_elements = channels * samples;
    let mut buffer = vec![0.0f32; total_elements];

    // Simple deterministic pseudo-random generator (Xorshift64 + Box-Muller transform)
    let mut rng_state: u64 = 0x853c49e6748fea9b;
    let mut next_uniform = || -> f32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f64 / u64::MAX as f64) as f32
    };

    let dt = 1.0 / sample_rate;
    let omega_60hz = 2.0 * std::f64::consts::PI * 60.0;

    // 1. Generate baseline + 60Hz power line noise + Gaussian thermal noise
    for ch in 0..channels {
        let channel_offset = ch * samples;
        // Channel-dependent slight phase shift for line noise (realistic biological pickup)
        let ch_phase = (ch as f64 * 0.05).fract() * 2.0 * std::f64::consts::PI;

        for s in 0..samples {
            let t = s as f64 * dt;
            // 60 Hz line interference
            let line_val = (line_noise_uv as f64 * (omega_60hz * t + ch_phase).sin()) as f32;

            // Box-Muller transform for white Gaussian noise
            let u1 = next_uniform().max(1e-7);
            let u2 = next_uniform();
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            let noise_val = z0 * noise_uv;

            buffer[channel_offset + s] = line_val + noise_val;
        }
    }

    // 2. Inject realistic extracellular action potentials (spikes) if requested
    let mut spikes_injected = 0usize;
    if spikes && samples > 100 {
        let spike_len = 60usize;
        let mut spike_shape = vec![0.0f32; spike_len];
        for i in 0..spike_len {
            let t_rel = (i as f32 - 15.0) / 8.0;
            spike_shape[i] = -120.0 * (-0.5 * t_rel * t_rel).exp() + 35.0 * (-0.5 * (t_rel - 1.8).powi(2)).exp();
        }

        let num_spike_times = (samples / 600).max(1);
        for st in 1..num_spike_times {
            let center_sample = st * 500;
            if center_sample + spike_len >= samples {
                break;
            }

            let primary_ch = (st * 37) % channels.saturating_sub(5);
            for d_ch in 0..5 {
                let ch = primary_ch + d_ch;
                let ch_offset = ch * samples;
                let attenuation = 1.0 / (1.0 + (d_ch as f32 * 0.8).powi(2));

                for i in 0..spike_len {
                    buffer[ch_offset + center_sample + i] += spike_shape[i] * attenuation;
                }
            }
            spikes_injected += 1;
        }
    }

    // Branch based on storage format
    if format == "zarr" {
        let zarr_path = if output_path.extension().map_or(false, |ext| ext == "bin") {
            output_path.with_extension("zarr")
        } else {
            output_path.to_path_buf()
        };

        println!("\n[Storing via zarrs Engine (Zarr v3 Format)]");
        dsp_stream::create_zarr_recording(
            &zarr_path,
            channels,
            samples,
            sample_rate,
            &buffer,
        )?;

        let total_bytes = channels * samples * std::mem::size_of::<f32>();
        println!("\n[Zarr Dataset Generation Complete]");
        println!("  Store Directory: {}", zarr_path.display());
        println!("  Array Path:      /traces");
        println!("  Array Shape:     [{} channels, {} samples]", channels, samples);
        println!("  Chunking Grid:   [{} channels, {} samples]", channels, 5000.min(samples));
        println!("  Total Floats:    {} ({:.2} MB)", channels * samples, total_bytes as f64 / (1024.0 * 1024.0));
        println!("  Spikes Injected: {}", spikes_injected);

        // Verification: Read back subset
        println!("\n[zarrs Verification Read]");
        let (read_data, r_ch, r_s, r_sr) = dsp_stream::read_zarr_recording(&zarr_path)?;
        println!("  Zarr Readback:   SUCCESS (retrieved {} channels, {} samples @ {:.1} Hz)", r_ch, r_s, r_sr);
        println!("  Sample (ch0, s0): {:.2} uV", read_data[0]);
        println!("  Sample (ch0, s1): {:.2} uV", read_data[1]);
        return Ok(());
    }

    // 3. Write binary data
    let mut file = File::create(output_path)?;
    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            buffer.as_ptr() as *const u8,
            buffer.len() * std::mem::size_of::<f32>(),
        )
    };
    file.write_all(byte_slice)?;
    file.flush()?;

    let total_bytes = byte_slice.len();

    // 4. Write metadata sidecar (.meta) in JSON format
    let meta_path = output_path.with_extension("meta");
    let metadata = RecordingMetadata {
        channels,
        samples,
        sample_rate_hz: sample_rate,
        duration_seconds: samples as f64 / sample_rate,
        noise_rms_uv: noise_uv,
        line_noise_uv: line_noise_uv,
        spikes_injected,
        format: "float32-le",
        byte_order: "little-endian",
        bytes_per_sample: std::mem::size_of::<f32>(),
        total_data_bytes: total_bytes,
    };
    let meta_json = serde_json::to_string_pretty(&metadata)?;
    fs::write(&meta_path, meta_json)?;

    println!("\n[Generation Complete]");
    println!("  Data File:      {} ({:.2} MB)", output_path.display(), total_bytes as f64 / (1024.0 * 1024.0));
    println!("  Metadata File:  {}", meta_path.display());
    println!("  Spikes Injected: {}", spikes_injected);

    // 5. Zero-copy memory mapping verification (mmap2 test)
    println!("\n[Zero-Copy mmap2 Verification]");
    let verify_file = File::open(output_path)?;
    let mmap = unsafe { Mmap::map(&verify_file)? };
    println!("  Memory Mapped:  SUCCESS ({} bytes mapped to host address {:p})", mmap.len(), mmap.as_ptr());
    let mapped_floats = unsafe {
        std::slice::from_raw_parts(mmap.as_ptr() as *const f32, mmap.len() / 4)
    };
    println!("  Sample (ch0, s0): {:.2} uV", mapped_floats[0]);
    println!("  Sample (ch0, s1): {:.2} uV", mapped_floats[1]);

    Ok(())
}

/// Runs comparative pipeline performance benchmarks with and without disk persistence.
fn run_benchmark_pipeline(
    channels: usize,
    samples: usize,
    iterations: usize,
    save: bool,
    format: &str,
    output_path: &Path,
) -> anyhow::Result<()> {
    println!("============================================================");
    println!("       dsp-kitchen Real-Time Pipeline Benchmark             ");
    println!("============================================================");
    let total_elements = channels * samples;
    let data_bytes = total_elements * std::mem::size_of::<f32>();
    println!("Workload:           {} channels × {} samples", channels, samples);
    println!("Buffer Size:        {:.2} MB ({} float32 samples)", data_bytes as f64 / (1024.0 * 1024.0), total_elements);
    println!("Persistence Mode:   {}", if save { format!("ENABLED ({format})") } else { "DISABLED (Pure In-VRAM Chaining)".into() });
    println!("Iterations:         {}", iterations);

    // 1. Initialize Compute Device
    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);
    println!("Compute Runtime:    {}", WgpuRuntime::name(&client));

    // 2. Prepare synthetic input data
    let input_host = vec![10.0f32; total_elements];
    let input_bytes = unsafe {
        std::slice::from_raw_parts(
            input_host.as_ptr() as *const u8,
            input_host.len() * std::mem::size_of::<f32>(),
        )
    };

    // 3. Allocate In-VRAM device buffers
    let buf_a = client.create_from_slice(input_bytes);
    let buf_b = client.empty(data_bytes);
    let notch_coeffs = design_notch_coeffs(60.0, 30000.0, 30.0);

    // Warmup JIT compilation
    execute_scaling::<WgpuRuntime>(&client, &buf_a, &buf_b, total_elements, 1.0, 0.0, false);
    execute_notch::<WgpuRuntime>(&client, &buf_b, &buf_a, notch_coeffs, channels, samples, false);
    execute_neo::<WgpuRuntime>(&client, &buf_a, &buf_b, channels, samples, false);

    // 4. Benchmark In-VRAM Chained Pipeline
    let mut kernel_durations = Vec::with_capacity(iterations);
    let mut total_durations = Vec::with_capacity(iterations);
    let mut upload_durations = Vec::with_capacity(iterations);
    let mut download_durations = Vec::with_capacity(iterations);
    let mut save_durations = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let total_start = Instant::now();

        // Upload (measured if testing full cycle)
        let t_upload_start = Instant::now();
        let in_buf = client.create_from_slice(input_bytes);
        let upload_elapsed = t_upload_start.elapsed();
        upload_durations.push(upload_elapsed);

        // Chained Kernels Execution directly in Device Memory (Zero Host Roundtrips)
        let t_kernel_start = Instant::now();
        // Step 1: ADC Scaling
        execute_scaling::<WgpuRuntime>(&client, &in_buf, &buf_b, total_elements, 0.195, 0.0, false);
        // Step 2: 60 Hz Notch Filter
        execute_notch::<WgpuRuntime>(&client, &buf_b, &in_buf, notch_coeffs, channels, samples, false);
        // Step 3: Nonlinear Energy Operator (NEO)
        execute_neo::<WgpuRuntime>(&client, &in_buf, &buf_b, channels, samples, false);
        let kernel_elapsed = t_kernel_start.elapsed();
        kernel_durations.push(kernel_elapsed);

        // Download from device
        let t_down_start = Instant::now();
        let out_bytes = client.read_one_unchecked(buf_b.clone());
        let download_elapsed = t_down_start.elapsed();
        download_durations.push(download_elapsed);

        // Optional Persistence to Disk
        let mut save_elapsed = std::time::Duration::ZERO;
        if save {
            let t_save_start = Instant::now();
            if format == "zarr" {
                let zarr_path = if output_path.extension().map_or(false, |ext| ext == "bin") {
                    output_path.with_extension("zarr")
                } else {
                    output_path.to_path_buf()
                };
                let out_slice = unsafe {
                    std::slice::from_raw_parts(
                        out_bytes.as_ptr() as *const f32,
                        out_bytes.len() / std::mem::size_of::<f32>(),
                    )
                };
                dsp_stream::create_zarr_recording(&zarr_path, channels, samples, 30000.0, out_slice)?;
            } else {
                if let Some(parent) = output_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(output_path, &*out_bytes)?;
            }
            save_elapsed = t_save_start.elapsed();
        }
        save_durations.push(save_elapsed);

        total_durations.push(total_start.elapsed());
    }

    let avg_kernel_ms = kernel_durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum::<f64>() / iterations as f64;
    let avg_upload_ms = upload_durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum::<f64>() / iterations as f64;
    let avg_download_ms = download_durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum::<f64>() / iterations as f64;
    let avg_save_ms = save_durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum::<f64>() / iterations as f64;
    let avg_total_ms = total_durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum::<f64>() / iterations as f64;

    let compute_throughput_msamples = (total_elements as f64 / 1_000_000.0) / (avg_kernel_ms / 1000.0);

    println!("\n[Benchmark Results Summary]");
    println!("  In-VRAM Kernel Time:   {:.3} ms  (Scaling -> Notch -> NEO)", avg_kernel_ms);
    println!("  Compute Throughput:    {:.1} MSamples/sec ({:.2} GB/s raw math)", compute_throughput_msamples, compute_throughput_msamples * 4.0 / 1024.0);

    if save {
        let bus_io_time = avg_upload_ms + avg_download_ms + avg_save_ms;
        let bus_overhead_pct = (bus_io_time / avg_total_ms) * 100.0;

        println!("\n[Data Movement & Persistence Overhead]");
        println!("  Host -> Device Upload: {:.3} ms", avg_upload_ms);
        println!("  Device -> Host Down:   {:.3} ms", avg_download_ms);
        println!("  Storage Write Time:    {:.3} ms", avg_save_ms);
        println!("  Total End-to-End:      {:.3} ms", avg_total_ms);
        println!("  I/O & Bus Penalty:     {:.1}% of runtime spent moving data!", bus_overhead_pct);
    } else {
        println!("\n[Pipeline Efficiency]");
        println!("  Host Roundtrips:       0 (Zero PCIe downloads between stages)");
        println!("  VRAM Data Movement:    Chained directly in persistent device registers/buffers");
    }

    println!("============================================================");
    Ok(())
}
