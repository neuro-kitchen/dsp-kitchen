//! Neurodata Without Borders (NWB 2.11) output.
//!
//! [`mapping::resolve`] turns a [`Session`] plus the user's metadata file into an [`NwbPlan`];
//! [`write`] writes that plan through a storage [`backend`] (Zarr today), streaming continuous
//! data in parallel chunks.

pub mod backend;
pub mod mapping;
pub mod schema;
pub mod types;
pub mod validate;

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub use mapping::{resolve, NwbPlan};

use crate::error::{Error, Result};
use crate::model::Session;
use backend::zarr::ZarrBackend;
use backend::Backend;

/// Chunk length along time for continuous data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkPolicy {
    /// A fixed duration for every series (default 1 s).
    Seconds(f64),
    /// About [`AUTO_CHUNK_BYTES`] per chunk: the length follows each series' rate, channel count
    /// and sample size, so every series gets similar-sized chunks.
    Auto,
}

/// Target chunk size of [`ChunkPolicy::Auto`]: 10 MB, NeuroConv's default (`chunk_mb=10.0`).
pub const AUTO_CHUNK_BYTES: u64 = 10_000_000;

impl ChunkPolicy {
    /// Rows (samples) per chunk for a series of `channels` values of `sample_bytes` at `rate` Hz.
    pub fn rows(self, rate: f64, channels: usize, sample_bytes: usize) -> u64 {
        match self {
            ChunkPolicy::Seconds(s) => ((s * rate).round() as u64).max(1),
            ChunkPolicy::Auto => (AUTO_CHUNK_BYTES / (channels.max(1) * sample_bytes.max(1)) as u64).max(1),
        }
    }
}

impl std::str::FromStr for ChunkPolicy {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        if s.eq_ignore_ascii_case("auto") {
            return Ok(ChunkPolicy::Auto);
        }
        match s.trim_end_matches('s').parse::<f64>() {
            Ok(v) if v > 0.0 && v.is_finite() => Ok(ChunkPolicy::Seconds(v)),
            _ => Err(format!("chunk must be a positive number of seconds or `auto`, got {s:?}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NwbOptions {
    /// gzip level 1–9 for datasets; `None` writes uncompressed (fastest). Default 1: on the
    /// 47 min TDT test block, 24 % smaller for ~35 % more write time.
    pub gzip: Option<u32>,
    /// Chunk length along time for continuous data.
    pub chunks: ChunkPolicy,
    pub threads: usize,
    pub overwrite: bool,
}

impl Default for NwbOptions {
    fn default() -> Self {
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        Self { gzip: Some(1), chunks: ChunkPolicy::Seconds(1.0), threads, overwrite: false }
    }
}

/// Progress of a write: samples copied so far out of the total.
#[derive(Debug, Clone, Copy)]
pub struct Progress {
    pub done: u64,
    pub total: u64,
    pub elapsed: Duration,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct WriteSummary {
    pub path: String,
    pub series: usize,
    pub samples: u64,
    pub seconds: f64,
}

/// Writes `plan` for `session` to a new NWB-Zarr store at `dest`. `progress` is called from a
/// monitor thread about twice a second while continuous data is copied.
pub fn write(
    session: &Session,
    plan: &NwbPlan,
    dest: &Path,
    options: &NwbOptions,
    progress: &(dyn Fn(Progress) + Sync),
) -> Result<WriteSummary> {
    if plan.has_errors() {
        let msgs: Vec<&str> = plan.issues.iter().filter(|i| i.level == crate::metadata::Level::Error).map(|i| i.message.as_str()).collect();
        return Err(Error::Unsupported(format!("the NWB plan has errors:\n  - {}", msgs.join("\n  - "))));
    }
    let started = Instant::now();
    let b = ZarrBackend::create(dest, options.gzip, options.overwrite)?;
    let b: &dyn Backend = &b;

    types::nwbfile::write_root(b, plan)?;
    types::subject::write(b, &plan.subject)?;
    types::devices::write(b, &plan.devices)?;
    types::electrodes::write(b, plan, session)?;

    for t in &plan.tables {
        types::tables::write(b, t, &session.tables[t.table])?;
    }
    if plan.events.iter().any(|e| e.table) {
        types::events::write_group(b)?;
    }
    for e in &plan.events {
        let ev = &session.events[e.event];
        if e.table {
            types::events::write(b, e, ev)?;
        } else {
            types::series::write_events(b, e, ev)?;
        }
    }
    for p in &plan.snippets {
        types::snippets::write(b, p, &session.snippets[p.snippet])?;
    }
    let stores: Vec<_> = plan.snippets.iter().map(|p| (p, &session.snippets[p.snippet])).collect();
    types::snippets::write_units(b, &stores)?;

    // Continuous data, with a monitor thread reporting progress
    let total: u64 = plan.series.iter().map(|s| {
        let i = session.recordings[s.recording].info();
        i.samples * i.channel_count() as u64
    }).sum();
    let done = AtomicU64::new(0);
    let finished = std::sync::atomic::AtomicBool::new(false);
    let result = std::thread::scope(|scope| {
        scope.spawn(|| {
            while !finished.load(Ordering::Relaxed) {
                progress(Progress { done: done.load(Ordering::Relaxed), total, elapsed: started.elapsed() });
                std::thread::sleep(Duration::from_millis(500));
            }
        });
        let r = plan.series.iter().try_for_each(|s| {
            types::series::write_continuous(b, s, session.recordings[s.recording].as_ref(), options.chunks, options.threads, &done)
        });
        finished.store(true, Ordering::Relaxed);
        r
    });
    result?;
    progress(Progress { done: total, total, elapsed: started.elapsed() });

    // Schema last, so a crash midway never leaves a store that looks complete
    types::nwbfile::write_specifications(b)?;
    Ok(WriteSummary { path: dest.display().to_string(), series: plan.series.len(), samples: total, seconds: started.elapsed().as_secs_f64() })
}
