//! `dspk-sorters`: every spike sorter of dspk in one program — Kilosort4 and EMUsort (GPL-3.0),
//! MountainSort 5, SpyKING CIRCUS 2 and Tridesclous 2. Because it contains Kilosort4 and EMUsort,
//! this program is distributed under GPL-3.0 as a whole (`dspk-sorters licenses` lists every
//! sorter's licence). The `dspk` command line runs it as a separate program (`dspk sort …`,
//! installed with `dspk sorters install`), so dspk itself stays MIT OR Apache-2.0.

mod sort;

use clap::{Parser, Subcommand};
use dspk_cli::style;
use dspk_core::ComputeTarget;

#[derive(Parser, Debug)]
#[command(
    name = "dspk-sorters",
    version,
    about = "Spike sorters of dspk on the GPU or CPU: Kilosort4, EMUsort, MountainSort 5, SpyKING CIRCUS 2, Tridesclous 2 (GPL-3.0 as a whole)",
    styles = style::clap_styles()
)]
struct Cli {
    /// Compute runtime: wgpu, cpu, cuda or hip, among those compiled in (default: the first
    /// compiled in, GPUs first)
    #[arg(long, global = true, env = dspk_core::compute::RUNTIME_ENV)]
    runtime: Option<String>,

    /// Show debug logs (runtimes, transport)
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    #[command(flatten)]
    Sort(sort::SortCommand),
    /// Licences of the sorters in this program, and where their sources are
    Licenses,
}

/// The licence texts that travel with this program: the program's own (GPL-3.0) and those of the
/// permissively licensed code in it.
const LICENSE_TEXTS: [(&str, &str); 4] = [
    ("GNU General Public License v3 (this program; Kilosort4, EMUsort)", include_str!("../LICENSE")),
    ("Apache License 2.0 (MountainSort 5, isosplit6; dspk code, also under MIT)", include_str!("../../../synapse/dspk-synapse/LICENSE-APACHE")),
    ("MIT License (dspk code, also under Apache-2.0)", include_str!("../../../synapse/dspk-synapse/LICENSE-MIT")),
    ("MIT License of SpikeInterface (SpyKING CIRCUS 2, Tridesclous 2, shared components)", include_str!("../../../synapse/dspk-synapse/LICENSE-SPIKEINTERFACE")),
];

/// Log level when `--verbose` is not given (as `dspk`).
const DEFAULT_LOG_LEVEL: tracing::Level = tracing::Level::WARN;

fn main() {
    let cli = Cli::parse();
    tracing_subscriber::fmt().with_max_level(if cli.verbose { tracing::Level::DEBUG } else { DEFAULT_LOG_LEVEL }).init();
    if let Err(e) = dispatch(&cli) {
        anstream::eprintln!("{} {e}", style::paint(style::ERROR, "error:"));
        for cause in e.chain().skip(1) {
            anstream::eprintln!("  {} {cause}", style::paint(style::DIM, "caused by:"));
        }
        std::process::exit(1);
    }
}

fn dispatch(cli: &Cli) -> anyhow::Result<()> {
    match &cli.command {
        Command::Licenses => {
            print!("{}", include_str!("../NOTICE.md"));
            for (title, text) in LICENSE_TEXTS {
                println!("\n---\n\n{title}\n\n{text}");
            }
            Ok(())
        }
        Command::Sort(sorter) => {
            let target = match &cli.runtime {
                Some(name) => ComputeTarget::parse(name)?.checked()?,
                None => ComputeTarget::from_env()?,
            };
            sort::run(target, sorter)
        }
    }
}
