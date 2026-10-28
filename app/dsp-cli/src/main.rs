//! `dsp-cli`: a thin command line over the dsp-kitchen crates. Each command lives in its own
//! module with its own arguments; this file only parses and dispatches.

mod commands;
mod progress;

use clap::{Parser, Subcommand};
use dsp_core::ComputeTarget;

#[derive(Parser, Debug)]
#[command(name = "dsp-cli", version, about = "dsp-kitchen from the command line: devices, recordings, detection, sorting, benchmarks, streaming, model hub")]
struct Cli {
    /// Compute runtime: wgpu, cpu, cuda or hip, among those compiled in (default: the first
    /// compiled in, GPUs first; `doctor` shows which work on this machine)
    #[arg(long, global = true, env = dsp_core::compute::RUNTIME_ENV)]
    runtime: Option<String>,

    /// Show debug logs (runtimes, transport)
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// What this build can do here: version, features, runtimes and their devices
    Doctor,
    /// Launch geometries the selected runtime gives for a channels × samples buffer
    Inspect(commands::inspect::InspectArgs),
    /// Open a recording (any format dsp-io reads) and report its layout and read speed
    Open(commands::open::OpenArgs),
    /// Show a probe layout preset
    Probe(commands::probe::ProbeArgs),
    /// Detect spikes in a whole recording and write them as a sorting (one unit per channel)
    Detect(commands::detect::DetectArgs),
    /// Spike-sort a recording with Kilosort4 or EMUsort and write the sorting
    Sort(commands::sort::SortArgs),
    /// Convert a sorting between formats (Phy, .sorting.zarr, NWB units)
    Convert(commands::convert::ConvertArgs),
    /// Write a synthetic recording (noise, line noise, drifting units)
    Generate(commands::generate::GenerateArgs),
    /// Time processing on the selected runtime
    Benchmark(commands::benchmark::BenchmarkArgs),
    /// Serve a recording over QUIC (signal and views)
    Serve(commands::net::serve::ServeArgs),
    /// Receive a served recording: throughput, latency, optional save
    Receive(commands::net::receive::ReceiveArgs),
    /// Published model and sorter artifacts: list, info, pull, verify, remove, clean
    #[cfg(feature = "hub")]
    Hub(commands::hub::HubArgs),
}

impl Cli {
    /// The runtime asked for (checked to be compiled in), else the first compiled-in one.
    fn target(&self) -> anyhow::Result<ComputeTarget> {
        Ok(match &self.runtime {
            Some(name) => ComputeTarget::parse(name)?.checked()?,
            None => ComputeTarget::from_env()?,
        })
    }
}

/// Log level when `--verbose` is not given: warnings only, so runtime chatter (e.g. GPU adapter
/// details) does not bury the command's output.
const DEFAULT_LOG_LEVEL: tracing::Level = tracing::Level::WARN;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt().with_max_level(if cli.verbose { tracing::Level::DEBUG } else { DEFAULT_LOG_LEVEL }).init();
    match &cli.command {
        Command::Doctor => commands::doctor::run(),
        Command::Inspect(args) => commands::inspect::run(cli.target()?, args),
        Command::Open(args) => commands::open::run(args),
        Command::Probe(args) => commands::probe::run(args),
        Command::Detect(args) => commands::detect::run(cli.target()?, args),
        Command::Sort(args) => commands::sort::run(cli.target()?, args),
        Command::Convert(args) => commands::convert::run(args),
        Command::Generate(args) => commands::generate::run(args),
        Command::Benchmark(args) => commands::benchmark::run(cli.target()?, args),
        Command::Serve(args) => commands::net::serve::run(args).await,
        Command::Receive(args) => commands::net::receive::run(args).await,
        #[cfg(feature = "hub")]
        Command::Hub(args) => commands::hub::run(args),
    }
}
