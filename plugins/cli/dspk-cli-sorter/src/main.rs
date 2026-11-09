//! `dspk-cli-sorter`: every spike sorter of dspk in one program — Kilosort4 and EMUsort (GPL-3.0),
//! MountainSort 5, SpyKING CIRCUS 2 and Tridesclous 2. Because it contains Kilosort4 and EMUsort,
//! this program is distributed under GPL-3.0 as a whole (`dspk-cli-sorter licenses` lists every
//! sorter's licence). The `dspk` command line runs it as a separate program (`dspk sort …`,
//! installed with `dspk sorters install`), so dspk itself stays MIT OR Apache-2.0.

mod sort;

use clap::{Parser, Subcommand};
use dspk_cli::style;
use dspk_core::ComputeTarget;

#[derive(Parser, Debug)]
#[command(
    name = "dspk-cli-sorter",
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

    /// How progress is reported: `bar` (on stderr) or `json` (one JSON object per line on
    /// stdout: `progress` events, then `done` or `error`; for programs running this one)
    #[arg(long, global = true, value_enum, default_value_t = ProgressMode::Bar)]
    progress: ProgressMode,

    #[command(subcommand)]
    command: Command,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum ProgressMode {
    Bar,
    Json,
}

static JSON_PROGRESS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Whether progress goes out as JSON lines (`--progress json`).
pub fn json_progress() -> bool {
    *JSON_PROGRESS.get().unwrap_or(&false)
}

#[derive(Subcommand, Debug)]
enum Command {
    #[command(flatten)]
    Sort(sort::SortCommand),
    /// Licences of the sorters in this program, and where their sources are
    Licenses,
    /// This program as a plugin: its sorters and every parameter (name, help, default, choices),
    /// as JSON, so a front end builds its sort form from it
    Describe,
}

/// The licence texts that travel with this program: the program's own (GPL-3.0) and those of the
/// permissively licensed code in it.
const LICENSE_TEXTS: [(&str, &str); 5] = [
    ("GNU General Public License v3 (this program; Kilosort4, EMUsort)", include_str!("../LICENSE")),
    ("Apache License 2.0 (MountainSort 5 and isosplit6, dspk-sorter-ms5)", include_str!("../../../../synapse/sorters/ms5/LICENSE")),
    ("Apache License 2.0 (dspk code, also under MIT)", include_str!("../../../../synapse/dspk-synapse/LICENSE-APACHE")),
    ("MIT License (dspk code, also under Apache-2.0)", include_str!("../../../../synapse/dspk-synapse/LICENSE-MIT")),
    ("MIT License of SpikeInterface (SpyKING CIRCUS 2, Tridesclous 2, their isosplit, shared components)", include_str!("../../../../synapse/dspk-synapse/LICENSE-SPIKEINTERFACE")),
];

/// Log level when `--verbose` is not given (as `dspk`).
const DEFAULT_LOG_LEVEL: tracing::Level = tracing::Level::WARN;

fn main() {
    let cli = Cli::parse();
    let _ = JSON_PROGRESS.set(cli.progress == ProgressMode::Json);
    tracing_subscriber::fmt().with_max_level(if cli.verbose { tracing::Level::DEBUG } else { DEFAULT_LOG_LEVEL }).init();
    let mut status = 0;
    if let Err(e) = dispatch(&cli) {
        if json_progress() {
            dspk_cli::progress::json_line(&serde_json::json!({"event": "error", "message": format!("{e:#}")}));
        }
        anstream::eprintln!("{} {e}", style::paint(style::ERROR, "error:"));
        for cause in e.chain().skip(1) {
            anstream::eprintln!("  {} {cause}", style::paint(style::DIM, "caused by:"));
        }
        status = 1;
    }
    // Results are written: no device teardown, which can crash or deadlock in the GPU driver
    dspk_core::compute::exit_now(status);
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
        Command::Describe => {
            println!("{}", serde_json::to_string_pretty(&describe()).expect("JSON"));
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

/// Arguments of the recording / output / run, not sorter parameters (the front end sets them).
const RUN_ARGS: [&str; 12] = ["recording", "source", "probe", "probe_file", "channel_map", "pitch_um", "start_sec", "duration_sec", "output", "format", "templates", "show_config"];

/// The plugin description: program, version, licence, and per sorter its parameters (from the
/// command line definition, so it never drifts from what the program accepts).
fn describe() -> serde_json::Value {
    use clap::CommandFactory;
    let cmd = Cli::command();
    let sorters: Vec<serde_json::Value> = cmd
        .get_subcommands()
        .filter(|s| !matches!(s.get_name(), "licenses" | "describe" | "help"))
        .map(|s| {
            let params: Vec<serde_json::Value> = s
                .get_arguments()
                .filter(|a| !a.is_global_set() && !RUN_ARGS.contains(&a.get_id().as_str()) && a.get_long().is_some())
                .map(|a| {
                    let defaults: Vec<String> = a.get_default_values().iter().map(|v| v.to_string_lossy().into_owned()).collect();
                    let choices: Vec<String> = a.get_possible_values().iter().map(|v| v.get_name().to_string()).collect();
                    serde_json::json!({
                        "name": a.get_id().as_str(),
                        "flag": format!("--{}", a.get_long().unwrap_or_default()),
                        "help": a.get_help().map(|h| h.to_string()).unwrap_or_default(),
                        "default": match defaults.len() { 0 => serde_json::Value::Null, 1 => serde_json::json!(defaults[0]), _ => serde_json::json!(defaults) },
                        "choices": choices,
                        "list": a.get_value_delimiter().is_some(),
                    })
                })
                .collect();
            serde_json::json!({"name": s.get_name(), "about": s.get_about().map(|h| h.to_string()).unwrap_or_default(), "parameters": params})
        })
        .collect();
    serde_json::json!({
        "program": "dspk-cli-sorter",
        "version": env!("CARGO_PKG_VERSION"),
        "license": "GPL-3.0-only",
        "interface": 1,
        "run": "dspk-cli-sorter --progress json <sorter> <recording> [--source ID] [--probe PRESET] -o <out>.sorting.zarr [--<parameter> VALUE …]",
        "sorters": sorters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_describe_lists_every_sorter_with_parameters() {
        let d = describe();
        let names: Vec<&str> = d["sorters"].as_array().unwrap().iter().map(|s| s["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["kilosort4", "emusort", "mountainsort5", "spykingcircus2", "tridesclous2"]);
        let ks4 = &d["sorters"][0]["parameters"];
        assert!(ks4.as_array().unwrap().iter().any(|p| p["name"] == "th_universal" && !p["default"].is_null()));
        assert!(!ks4.as_array().unwrap().iter().any(|p| p["name"] == "recording"));
    }
}
