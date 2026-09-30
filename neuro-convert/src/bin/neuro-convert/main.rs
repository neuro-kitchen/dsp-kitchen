//! `neuro-convert`: inspect recordings and convert them to publication formats.

mod commands;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "neuro-convert", version, about = "Read neurophysiology recordings and convert them to publication formats")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List the input formats and versions this build reads
    Formats,
    /// Show what a recording contains: streams, events, metadata, warnings
    Inspect {
        /// Recording path (e.g. a TDT block folder)
        path: PathBuf,
        /// Also time reading this many seconds of every channel of the largest stream
        #[arg(long)]
        read_sec: Option<f64>,
        /// Print a machine-readable summary (JSON) instead
        #[arg(long)]
        json: bool,
    },
    /// Convert a recording to NWB (Zarr), using a metadata file for what the source lacks
    #[cfg(feature = "nwb")]
    Convert(commands::convert::ConvertArgs),
    /// Check an NWB-Zarr store's structure (references, lengths, required fields)
    #[cfg(feature = "nwb")]
    Validate {
        /// Store to check (`.nwb.zarr`)
        path: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Formats => commands::formats::run(),
        Command::Inspect { path, read_sec, json: true } => commands::inspect::json(&path, read_sec),
        Command::Inspect { path, read_sec, json: false } => commands::inspect::run(&path, read_sec),
        #[cfg(feature = "nwb")]
        Command::Convert(args) => commands::convert::run(&args),
        #[cfg(feature = "nwb")]
        Command::Validate { path } => commands::convert::validate(&path),
    }
}
