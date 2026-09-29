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
    Generate {
        /// Number of sensor channels (e.g. 1, 4, 32, 64, 384, 1024)
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

        /// Output file/directory path
        #[arg(short, long, default_value = "playground/data/mock_signal_384ch.bin")]
        output: PathBuf,
    },

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

        /// Path to custom server TLS certificate DER (if omitted, accepts self-signed cert)
        #[arg(long)]
        cert: Option<PathBuf>,

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

    match cli.command {
        Commands::Components => {
            commands::components::run_components();
        }
        Commands::Probe { model } => {
            commands::probe::run_probe(&model);
        }
        Commands::Inspect { channels, samples } => {
            commands::inspect::run_inspect(channels, samples);
        }
        Commands::Stream {
            mode,
            buckets,
            channels,
            sample_rate,
        } => {
            commands::stream::run_stream(&mode, buckets, channels, sample_rate)?;
        }
        Commands::Generate {
            channels,
            samples,
            sample_rate,
            noise_uv,
            line_noise_uv,
            spikes,
            format,
            output,
        } => {
            commands::generate::run_generate(
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
        Commands::Benchmark {
            channels,
            samples,
            iterations,
            save,
            format,
            output,
            sweep,
        } => {
            if sweep {
                commands::benchmark::run_benchmark_sweep(samples, iterations)?;
            } else {
                commands::benchmark::run_benchmark_pipeline(
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
            )
            .await?;
        }
        Commands::Receive {
            addr,
            server_name,
            cert,
            duration,
            max_frames,
            save,
        } => {
            commands::receive::run_receive(
                addr,
                server_name,
                cert,
                duration,
                max_frames,
                save,
            )
            .await?;
        }
    }

    Ok(())
}

