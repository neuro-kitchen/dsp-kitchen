//! `sorters`: the `dspk-sorters` program `dspk sort` runs — install this version's build from the
//! GitHub release (verified against the release manifest's SHA-256 and size) or show which one is
//! used.

use clap::{Args, Subcommand};
use dspk_synapse_hub::sorters;

use crate::style::{self, paint};

#[derive(Subcommand, Debug)]
enum SortersCommand {
    /// Download and verify dspk-sorters for this version and platform (GPL-3.0)
    Install {
        /// Download again even when it is installed
        #[arg(long)]
        force: bool,
    },
    /// Which dspk-sorters `dspk sort` runs, and where installs go
    Status,
}

#[derive(Args, Debug)]
pub struct SortersArgs {
    #[command(subcommand)]
    command: SortersCommand,
}

pub fn run(args: &SortersArgs) -> anyhow::Result<()> {
    let version = env!("CARGO_PKG_VERSION");
    match &args.command {
        SortersCommand::Install { force } => {
            if let (Some(path), false) = (sorters::installed(version), *force) {
                println!("{} dspk-sorters {version} is installed: {}", paint(style::OK, "ok"), path.display());
                return Ok(());
            }
            println!("downloading dspk-sorters {version} from {} …", sorters::release_url(version));
            let path = sorters::install(version)?;
            println!("{} installed {}", paint(style::OK, "ok"), path.display());
            println!(
                "{}",
                paint(style::DIM, "dspk-sorters is GPL-3.0 (it contains Kilosort4 and EMUsort); `dspk-sorters licenses` lists every sorter's licence")
            );
        }
        SortersCommand::Status => {
            let caller = std::env::current_exe().ok();
            match sorters::locate(version, caller.as_deref()) {
                Some(path) => println!("{} {}", paint(style::HEADING, "dspk sort runs"), path.display()),
                None => println!("{} not installed: `dspk sorters install`", paint(style::WARN, "dspk-sorters")),
            }
            println!("{} {}", paint(style::HEADING, "installs go to"), sorters::install_dir(version).display());
        }
    }
    Ok(())
}
