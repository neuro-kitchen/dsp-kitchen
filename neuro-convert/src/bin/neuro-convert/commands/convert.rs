use std::io::Write;
use std::path::PathBuf;

use clap::Args;
use neuro_convert::metadata::{Level, MetadataFile};
use neuro_convert::outputs::nwb::{self, NwbOptions, NwbPlan};

#[derive(Args)]
pub struct ConvertArgs {
    /// Recording to convert (e.g. a TDT block folder)
    input: PathBuf,
    /// Metadata file (YAML): session, subject, electrode groups, per-stream mapping
    #[arg(short, long)]
    metadata: Option<PathBuf>,
    /// Output store (`.nwb.zarr`)
    #[arg(short, long)]
    output: PathBuf,
    /// gzip level 1-9 for datasets
    #[arg(long, default_value_t = 1)]
    gzip: u32,
    /// Write uncompressed (fastest, ~30 % larger)
    #[arg(long)]
    no_compression: bool,
    /// Worker threads for copying data (default: all cores)
    #[arg(long)]
    threads: Option<usize>,
    /// Replace the output if it exists
    #[arg(long)]
    overwrite: bool,
    /// Print the plan and exit without writing
    #[arg(long)]
    dry_run: bool,
    #[command(flatten)]
    open: super::OpenArgs,
}

pub fn run(a: &ConvertArgs) -> anyhow::Result<()> {
    let session = neuro_convert::open(&a.input, &a.open.options())?;
    let meta = match &a.metadata {
        Some(p) => MetadataFile::load(p)?,
        None => MetadataFile::default(),
    };
    let plan = nwb::resolve(&session, &meta, || uuid_like());
    print_plan(&plan, &session);
    if a.dry_run {
        return Ok(());
    }
    if plan.has_errors() {
        anyhow::bail!("fix the errors above in the metadata file, then run again");
    }

    let gzip = (!a.no_compression).then_some(a.gzip);
    let mut options = NwbOptions { gzip, overwrite: a.overwrite, ..Default::default() };
    if let Some(t) = a.threads {
        options.threads = t;
    }
    println!("\nWriting {} ({} threads, {}) …", a.output.display(), options.threads, gzip.map_or("uncompressed".into(), |l| format!("gzip {l}")));
    let summary = nwb::write(&session, &plan, &a.output, &options, &|p| {
        let pct = if p.total > 0 { p.done as f64 * 100.0 / p.total as f64 } else { 100.0 };
        let rate = p.done as f64 * 4.0 / 1e6 / p.elapsed.as_secs_f64().max(1e-9);
        print!("\r  {pct:5.1}%  {rate:6.0} MB/s  {:5.0} s", p.elapsed.as_secs_f64());
        let _ = std::io::stdout().flush();
    })?;
    println!("\nDone: {} series, {:.2} G samples in {:.1} s", summary.series, summary.samples as f64 / 1e9, summary.seconds);

    // Conversion report next to the output
    let report = serde_json::json!({ "summary": summary, "plan": plan, "provenance": session.provenance });
    let report_path = a.output.with_extension("report.json");
    std::fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    println!("Report: {}", report_path.display());
    Ok(())
}

fn print_plan(plan: &NwbPlan, session: &neuro_convert::Session) {
    let f = &plan.file;
    println!("=== NWB plan ===");
    println!("identifier:   {}", f.identifier);
    println!("start time:   {}", f.start_time);
    println!("description:  {}", f.description);
    println!("subject:      {} ({})", plan.subject.id.as_deref().unwrap_or("?"), plan.subject.species.as_deref().unwrap_or("species?"));
    for g in &plan.groups {
        println!("electrodes:   group {} at {} on {}", g.name, g.location, g.device);
    }
    for s in &plan.series {
        let i = session.recordings[s.recording].info();
        let kind = if s.electrode_group.is_some() { "ElectricalSeries" } else { "TimeSeries" };
        println!("acquisition/{:8} ← {:6} {kind:16} {:>3} ch  {:>10.1} Hz  {}", s.name, s.source, i.channel_count(), i.sample_rate, s.unit);
    }
    for e in &plan.events {
        let place = if e.table { "events" } else { "acquisition" };
        println!("{place}/{:8} ← {:6} ({} events)", e.name, e.source, session.events[e.event].len());
    }
    for t in &plan.tables {
        println!("analysis/{} ← table", t.name);
    }
    for s in &plan.skipped {
        println!("skipped:      {s}");
    }
    for i in &plan.issues {
        println!("{} {}", if i.level == Level::Error { "ERROR:  " } else { "warning:" }, i.message);
    }
}

pub fn validate(path: &std::path::Path) -> anyhow::Result<()> {
    let issues = nwb::validate::validate(path)?;
    let errors = issues.iter().filter(|i| i.level == Level::Error).count();
    for i in &issues {
        println!("{} {}", if i.level == Level::Error { "ERROR:  " } else { "warning:" }, i.message);
    }
    println!("{}: {errors} errors, {} warnings", path.display(), issues.len() - errors);
    if errors > 0 {
        anyhow::bail!("{} is not a valid NWB-Zarr store", path.display());
    }
    Ok(())
}

/// Random UUID v4 text for the NWB identifier.
fn uuid_like() -> String {
    // The library generates object ids with `uuid`; reuse it through a throwaway plan field
    neuro_convert::outputs::nwb::types::typed("core", "x")["object_id"].as_str().unwrap_or_default().to_string()
}
