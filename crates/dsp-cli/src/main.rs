mod commands;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "dsp-cli")]
#[command(
    author,
    version,
    about = "High-performance signal processing engine & hardware CLI",
    long_about = None
)]
struct Cli {
    /// Compute runtime: wgpu, cpu, cuda or hip (default: first compiled-in, GPU first)
    #[arg(long, global = true, env = "DSP_KITCHEN_RUNTIME")]
    runtime: Option<String>,

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
        /// Type of probe layout to display: 'neuropixels-1', 'neuropixels-2', 'tetrode', 'utah'
        #[arg(short, long, default_value = "neuropixels-1")]
        model: String,
    },

    /// Open any recording dsp-io understands and print its layout, probe, and read throughput
    Open {
        /// Recording path (SpikeGLX .bin/.cbin, raw .bin + JSON .meta, Zarr store)
        path: PathBuf,

        /// Seconds of data to read for the throughput test
        #[arg(long, default_value_t = 10.0)]
        read_sec: f64,
    },

    /// Inspect hardware topology, core counts, and dynamic launch geometry
    Inspect {
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

        /// Number of channels
        #[arg(short, long, default_value_t = 384)]
        channels: usize,

        /// Sample rate in Hz
        #[arg(short = 'r', long, default_value_t = 30000.0)]
        sample_rate: f64,
    },

    /// Generate a synthetic multi-channel recording with configurable noise and optional spikes
    #[command(alias = "mock-signal")]
    Generate(commands::generate::GenerateArgs),

    /// Benchmark real-time DSP pipeline performance across channels
    Benchmark {
        /// Number of channels (default: 384)
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

        /// Run automated multi-channel sweep (1, 4, 16, 32, 64, 128, 384, 1024 channels)
        #[arg(long, default_value_t = false)]
        sweep: bool,

        /// Time every kernel family and the streaming sorter; write a JSON report to --report-dir
        #[arg(long, default_value_t = false)]
        suite: bool,

        /// Where --suite writes its report (`<unix time>-<runtime>.json`)
        #[arg(long, default_value = "playground/benchmarks")]
        report_dir: PathBuf,
    },

    /// Serve a dataset (.bin) or continuous synthetic signal over high-throughput QUIC
    #[command(alias = "stream-server")]
    Serve {
        /// UDP socket address to bind the QUIC server on
        #[arg(short, long, default_value = "127.0.0.1:50051")]
        bind: std::net::SocketAddr,

        /// Path to recording dataset file (.bin) to stream (omitting streams live synthetic signals)
        #[arg(short, long, alias = "input")]
        file: Option<PathBuf>,

        /// Number of channels (overrides sidecar metadata or synthetic default)
        #[arg(short, long)]
        channels: Option<usize>,

        /// Sample rate in Hz (overrides sidecar metadata or synthetic default)
        #[arg(short = 'r', long)]
        sample_rate: Option<f64>,

        /// Samples per streaming frame chunk
        #[arg(short = 's', long, default_value_t = 500)]
        chunk_size: usize,

        /// Do not loop dataset when reaching end of file
        #[arg(long, default_value_t = false)]
        no_loop: bool,

        /// Disable real-time pacing and push at maximum wire line rate (stress test)
        #[arg(long, default_value_t = false, alias = "stress")]
        no_realtime: bool,

        /// Optional path to export the server TLS certificate DER
        #[arg(long)]
        cert_out: Option<PathBuf>,

        /// Send int16 payloads (half the bandwidth) with this many µV per step
        #[arg(long)]
        int16_gain_uv: Option<f32>,
    },

    /// Connect to a QUIC stream server and benchmark transmission speed, latency, jitter, and packet loss
    #[command(alias = "benchmark-net", alias = "test-stream")]
    Receive {

        /// QUIC server socket address
        #[arg(short, long, default_value = "127.0.0.1:50051")]
        addr: std::net::SocketAddr,

        /// TLS server name
        #[arg(long, default_value = "localhost")]
        server_name: String,

        /// Server TLS certificate DER (written by `serve --cert-out`)
        #[arg(long)]
        cert: Option<PathBuf>,

        /// Skip server certificate verification (trusted networks only)
        #[arg(long, default_value_t = false)]
        insecure: bool,

        /// Benchmark test duration in seconds (0 for unlimited / until Ctrl+C)
        #[arg(short, long, default_value_t = 5.0)]
        duration: f64,

        /// Maximum frames to receive (0 for unlimited)
        #[arg(short = 'n', long, default_value_t = 0)]
        max_frames: u64,

        /// Optional destination to save received continuous signal stream (.bin)
        #[arg(short = 'o', long)]
        save: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    let target = match &cli.runtime {
        Some(name) => dsp_core::ComputeTarget::parse(name)?.checked()?,
        None => dsp_core::ComputeTarget::from_env()?,
    };

    match cli.command {
        Commands::Components => {
            commands::components::run_components();
        }
        Commands::Probe { model } => {
            commands::probe::run_probe(&model);
        }
        Commands::Inspect { channels, samples } => {
            commands::inspect::run_inspect(target, channels, samples)?;
        }
        Commands::Stream {
            mode,
            buckets,
            channels,
            sample_rate,
        } => {
            commands::stream::run_stream(&mode, buckets, channels, sample_rate)?;
        }
        Commands::Open { path, read_sec } => {
            commands::info::run_info(&path, read_sec)?;
        }
        Commands::Generate(args) => {
            commands::generate::run_generate(&args)?;
        }
        Commands::Benchmark {
            channels,
            samples,
            iterations,
            save,
            format,
            output,
            sweep,
            suite,
            report_dir,
        } => {
            if suite {
                commands::benchmark::run_benchmark_suite(target, channels, samples, iterations, &report_dir)?;
            } else if sweep {
                commands::benchmark::run_benchmark_sweep(target, samples, iterations)?;
            } else {
                commands::benchmark::run_benchmark_pipeline(
                    target,
                    channels,
                    samples,
                    iterations,
                    save,
                    &format,
                    &output,
                )?;
            }
        }
        Commands::Serve {
            bind,
            file,
            channels,
            sample_rate,
            chunk_size,
            no_loop,
            no_realtime,
            cert_out,
            int16_gain_uv,
        } => {
            let loop_stream = !no_loop;
            let realtime = !no_realtime;
            commands::serve::run_serve(
                bind,
                file,
                channels,
                sample_rate,
                chunk_size,
                loop_stream,
                realtime,
                cert_out,
                int16_gain_uv,
            )
            .await?;
        }
        Commands::Receive {
            addr,
            server_name,
            cert,
            insecure,
            duration,
            max_frames,
            save,
        } => {
            commands::receive::run_receive(
                addr,
                server_name,
                cert,
                insecure,
                duration,
                max_frames,
                save,
            )
            .await?;
        }
    }

    Ok(())
}

