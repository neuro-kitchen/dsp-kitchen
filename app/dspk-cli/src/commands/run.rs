//! `run`: a pipeline spec (JSON or TOML) through dspk-pipeline, with its inputs and output taken
//! from the spec or given here. Shows the plan, draws progress per input, and cancels cleanly on
//! Ctrl+C (the unfinished output is removed).

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Context;
use clap::{Args, ValueEnum};
use dspk_core::{ComputeTarget, DspResult, ProgressEvent, ProgressSink, RecordingSource};
use dspk_pipeline::{ExecMode, ExecutedMode, Input, MergePart, PipelineError, PipelineSpec, Plan, RunReport};

use crate::progress::TerminalProgress;
use crate::style::{bytes, count, hz, paint, DIM, HEADING, OK, VALUE, WARN};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Mode {
    /// Stream where the stages allow it, else independent windows
    Auto,
    /// Windows in order with filter state carried (equal to the whole recording at once)
    Stream,
    /// Independent windows with halos of context
    Windows,
}

#[derive(Args, Debug)]
pub struct RunArgs {
    /// Pipeline spec, `.json` or `.toml`; paths in it are relative to its folder
    spec: PathBuf,
    /// Input recordings, replacing the spec's: files or globs (quote a glob to let dspk expand it,
    /// sorted); several inputs run one after the other
    #[arg(short, long = "input", value_name = "PATH", num_args = 1..)]
    inputs: Vec<PathBuf>,
    /// Recordings read as one input, channels side by side (same rate and length), replacing the
    /// spec's inputs
    #[arg(long, value_name = "PATH", num_args = 2.., conflicts_with = "inputs")]
    merge: Vec<PathBuf>,
    /// Output path, replacing the spec's; with several inputs it needs `{stem}` or `{index}`
    #[arg(short, long, value_name = "PATH")]
    output: Option<PathBuf>,
    /// Execution mode, replacing the spec's
    #[arg(long, value_enum)]
    mode: Option<Mode>,
    /// Samples per channel of each window, replacing the spec's
    #[arg(long, value_name = "N")]
    chunk_samples: Option<usize>,
    /// Show what would run (inputs, outputs, mode, windows, bytes) and stop
    #[arg(short = 'n', long)]
    dry_run: bool,
    /// Print the plan (with --dry-run) or the reports as JSON on stdout, nothing else
    #[arg(long)]
    json: bool,
}

/// Opens every format dspk reads, the neural ones included.
fn open(path: &Path, source: Option<&str>) -> DspResult<Box<dyn RecordingSource>> {
    match source {
        Some(id) => dspk_synapse_io::open_source(path, id),
        None => dspk_synapse_io::open(path),
    }
}

/// `path` from the working directory (paths in the spec stay relative to the spec).
fn from_here(path: &Path) -> anyhow::Result<PathBuf> {
    std::path::absolute(path).with_context(|| format!("{}", path.display()))
}

impl RunArgs {
    /// The spec with this command line's replacements.
    fn spec(&self, runtime: Option<&str>) -> anyhow::Result<PipelineSpec> {
        let mut spec = PipelineSpec::load(&self.spec)?;
        if !self.inputs.is_empty() {
            spec.set_inputs(self.inputs.iter().map(|p| from_here(p).map(Input::Path)).collect::<anyhow::Result<_>>()?)?;
        }
        if !self.merge.is_empty() {
            let merge = self.merge.iter().map(|p| Ok(MergePart { path: from_here(p)?, source: None, channels: None })).collect::<anyhow::Result<_>>()?;
            spec.set_inputs(vec![Input::Merge { merge }])?;
        }
        if let Some(out) = &self.output {
            spec.set_output(from_here(out)?)?;
        }
        if let Some(mode) = self.mode {
            spec.execution.mode = match mode {
                Mode::Auto => ExecMode::Auto,
                Mode::Stream => ExecMode::Stream,
                Mode::Windows => ExecMode::Windows,
            };
        }
        if let Some(n) = self.chunk_samples {
            spec.execution.chunk_samples = n;
        }
        if let Some(name) = runtime {
            spec.execution.runtime = Some(ComputeTarget::parse(name)?.checked()?.name().to_string());
        }
        Ok(spec)
    }
}

/// Progress of the engine drawn as the CLI's bar, one line per input.
struct Bars {
    sink: TerminalProgress,
    labels: Vec<String>,
}

pub async fn run(runtime: Option<&str>, args: &RunArgs) -> anyhow::Result<()> {
    let spec = args.spec(runtime)?;
    let base_dir = std::path::absolute(&args.spec)?.parent().map(Path::to_path_buf).unwrap_or_default();
    let plans = dspk_pipeline::plan_with(&spec, &base_dir, open)?;
    let runtime_name = match &spec.execution.runtime {
        Some(name) => name.clone(),
        None => ComputeTarget::from_env()?.name().to_string(),
    };

    if args.dry_run {
        if args.json {
            println!("{}", serde_json::to_string_pretty(&plans.iter().map(plan_json).collect::<Vec<_>>())?);
        } else {
            print_plan(&args.spec, &runtime_name, &plans, true);
        }
        return Ok(());
    }
    if !args.json {
        print_plan(&args.spec, &runtime_name, &plans, false);
    }

    let cancel = Arc::new(AtomicBool::new(false));
    let on_signal = Arc::clone(&cancel);
    let signal = tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            on_signal.store(true, Ordering::Relaxed);
        }
    });
    let bars = (!args.json && std::io::stderr().is_terminal())
        .then(|| Bars { sink: TerminalProgress::default(), labels: plans.iter().map(|p| short(&p.input)).collect() });
    let reports = tokio::task::spawn_blocking(move || {
        dspk_pipeline::run_with(&spec, &base_dir, open, |p| {
            if let Some(bars) = &bars {
                bars.sink.report(&ProgressEvent {
                    stage: &bars.labels[p.input],
                    step: p.input + 1,
                    steps: p.inputs,
                    done: p.windows_done as u64,
                    total: p.windows as u64,
                    unit: "windows",
                });
            }
            !cancel.load(Ordering::Relaxed)
        })
    })
    .await?;
    signal.abort();

    let reports = match reports {
        Err(PipelineError::Cancelled) => {
            anstream::eprintln!("{} the unfinished output was removed", paint(WARN, "cancelled:"));
            std::process::exit(130);
        }
        other => other?,
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&reports.iter().map(report_json).collect::<Vec<_>>())?);
    } else {
        print_reports(&reports);
    }
    Ok(())
}

/// `path` relative to the working directory when it is inside it, as given otherwise.
fn shown(path: &Path) -> String {
    let here = std::env::current_dir().ok();
    here.and_then(|d| path.strip_prefix(d).ok().map(|p| p.display().to_string())).unwrap_or_else(|| path.display().to_string())
}

/// A path's last two components, enough to tell inputs apart on one line.
fn short(path: &str) -> String {
    let p = Path::new(path);
    let parts: Vec<_> = p.components().rev().take(2).collect();
    parts.into_iter().rev().map(|c| c.as_os_str().to_string_lossy().to_string()).collect::<Vec<_>>().join("/")
}

fn mode_text(mode: ExecutedMode) -> String {
    match mode {
        ExecutedMode::Stream => "stream".into(),
        ExecutedMode::Windows { left, right } => format!("windows, halo {left}+{right}"),
    }
}

fn print_plan(spec: &Path, runtime: &str, plans: &[Plan], dry_run: bool) {
    let what = if dry_run { "plan" } else { "run" };
    anstream::println!(
        "{} {}  {} on {}",
        paint(HEADING, "dspk"),
        paint(HEADING, what),
        paint(VALUE, spec.display()),
        paint(VALUE, runtime)
    );
    let wide = plans.len() > 1;
    for (i, p) in plans.iter().enumerate() {
        let index = if wide { format!("{:>3} ", i + 1) } else { "  ".into() };
        let input = p.input.split(" + ").map(|part| shown(Path::new(part))).collect::<Vec<_>>().join(" + ");
        anstream::println!("{}{} {} {}", paint(DIM, &index), paint(VALUE, input), paint(DIM, "→"), paint(VALUE, shown(&p.output)));
        let rate = if p.output_rate_hz != p.input_rate_hz {
            format!("{} → {} samples at {}", hz(p.input_rate_hz), count(p.output_samples), hz(p.output_rate_hz))
        } else {
            format!("at {}", hz(p.input_rate_hz))
        };
        let upload = if p.stored_upload { " as stored integers" } else { "" };
        let detail = format!(
            "{} channels · {} samples {rate} · {} · {} windows · reads {}, uploads {}{upload}",
            p.channels,
            count(p.input_samples),
            mode_text(p.mode),
            p.windows,
            bytes(p.bytes_read),
            bytes(p.bytes_uploaded),
        );
        anstream::println!("{}{}", " ".repeat(index.len() + 1), paint(DIM, detail));
    }
}

fn print_reports(reports: &[RunReport]) {
    let mut seconds = 0.0;
    for r in reports {
        seconds += r.seconds;
        let rate = r.bytes.read as f64 / r.seconds.max(1e-9);
        anstream::println!(
            "{} {}  {}",
            paint(OK, "✓"),
            paint(VALUE, shown(&r.output)),
            paint(DIM, format!("{:.2} s · {}/s read · {} written", r.seconds, bytes(rate as u64), bytes(r.bytes.written)))
        );
    }
    if reports.len() > 1 {
        anstream::println!("{} {} inputs in {seconds:.2} s", paint(OK, "done"), reports.len());
    }
}

fn mode_json(mode: ExecutedMode) -> serde_json::Value {
    match mode {
        ExecutedMode::Stream => serde_json::json!({"mode": "stream", "halo": [0, 0]}),
        ExecutedMode::Windows { left, right } => serde_json::json!({"mode": "windows", "halo": [left, right]}),
    }
}

fn plan_json(p: &Plan) -> serde_json::Value {
    let mut v = serde_json::json!({
        "input": p.input, "output": p.output, "stored_upload": p.stored_upload, "channels": p.channels,
        "input_samples": p.input_samples, "input_rate": p.input_rate_hz,
        "output_samples": p.output_samples, "output_rate": p.output_rate_hz,
        "windows": p.windows, "max_window_samples": p.max_window_samples,
        "bytes": {"read": p.bytes_read, "uploaded": p.bytes_uploaded},
    });
    v.as_object_mut().expect("object").extend(mode_json(p.mode).as_object().expect("object").clone());
    v
}

fn report_json(r: &RunReport) -> serde_json::Value {
    let mut v = serde_json::json!({
        "runtime": r.runtime, "stored_upload": r.stored_upload, "channels": r.channels,
        "samples": r.samples, "sample_rate": r.sample_rate_hz, "windows": r.windows,
        "bytes": {"read": r.bytes.read, "uploaded": r.bytes.uploaded, "downloaded": r.bytes.downloaded, "written": r.bytes.written},
        "output": r.output, "provenance": r.provenance, "seconds": r.seconds,
    });
    v.as_object_mut().expect("object").extend(mode_json(r.mode).as_object().expect("object").clone());
    v
}
