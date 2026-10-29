//! `sort`: runs the `dspk-sorters` program (every spike sorter; GPL-3.0 as a whole, since it
//! contains Kilosort4 and EMUsort) as a separate process with the arguments given here, so this
//! command line stays MIT OR Apache-2.0. `dspk sorters install` downloads the program; a source
//! build finds the one built next to `dspk`.

use std::ffi::OsString;
use std::process::Command;

use anyhow::{bail, Context};
use clap::Args;
use dspk_synapse_hub::sorters;

#[derive(Args, Debug)]
pub struct SortArgs {
    /// The sorter (kilosort4, emusort, mountainsort5, spykingcircus2, tridesclous2) and its
    /// arguments, passed to dspk-sorters (`dspk sort kilosort4 --help` lists them)
    #[arg(trailing_var_arg = true, allow_hyphen_values = true, required = true, value_name = "SORTER ARGS")]
    args: Vec<OsString>,
}

pub fn run(runtime: Option<&str>, verbose: bool, args: &SortArgs) -> anyhow::Result<()> {
    let version = env!("CARGO_PKG_VERSION");
    let caller = std::env::current_exe().ok();
    let Some(program) = sorters::locate(version, caller.as_deref()) else {
        bail!(
            "the sorters are a separate program, dspk-sorters {version} (GPL-3.0), not installed here: \
             run `dspk sorters install` to download it, or set {} to an executable",
            sorters::SORTERS_BIN_ENV
        );
    };
    let mut command = Command::new(&program);
    if let Some(runtime) = runtime {
        command.arg("--runtime").arg(runtime);
    }
    if verbose {
        command.arg("--verbose");
    }
    let status = command.args(&args.args).status().with_context(|| format!("failed to run {}", program.display()))?;
    if !status.success() {
        // The sorter has reported its own error; keep its exit code
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}
