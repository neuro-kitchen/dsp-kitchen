//! Fills a source's [`MinMaxSummary`] in the background, once, for every reader of it.
//!
//! One thread per source reads the recording a chunk at a time, nearest the window being looked
//! at first and then outward (a new focus re-orders what is left), and reports each chunk done.
//! Readers only draw from the summary, never fill it, so two views of one source never read the
//! same samples twice, and a zoomed-out view fills in as the reading advances.

use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use dsp_core::RecordingSource;

use super::summary::MinMaxSummary;

/// Most values (channels × samples) one read holds.
const BLOCK_VALUES: usize = 1 << 22;
/// Seconds of recording per chunk (between progress reports and focus checks).
const CHUNK_SEC: f64 = 2.0;
/// No focus requested.
const NONE: u64 = u64::MAX;

/// Progress of a summary: the samples just summarized, and how many are summarized of all.
#[derive(Debug, Clone, PartialEq)]
pub struct Progress {
    pub filled: Range<u64>,
    pub done: u64,
    pub total: u64,
}

pub type OnProgress = Arc<dyn Fn(Progress) + Send + Sync>;

/// The background filler of one source's summary.
pub struct Summarizer {
    focus: Arc<AtomicU64>,
    started: AtomicBool,
}

impl Default for Summarizer {
    fn default() -> Self {
        Self { focus: Arc::new(AtomicU64::new(NONE)), started: AtomicBool::new(false) }
    }
}

impl Summarizer {
    /// Starts filling (once), nearest sample `focus` first; later calls only move the focus.
    pub fn run(&self, source: Arc<dyn RecordingSource>, summary: Arc<MinMaxSummary>, focus: u64, cancel: Arc<AtomicBool>, on_progress: OnProgress) {
        self.focus.store(focus, Ordering::Relaxed);
        if self.started.swap(true, Ordering::Relaxed) {
            return;
        }
        let focus = self.focus.clone();
        let spawned = std::thread::Builder::new().name("minmax-summary".into()).spawn(move || fill(source.as_ref(), &summary, &focus, &cancel, &*on_progress));
        if let Err(e) = spawned {
            tracing::warn!("could not start summarizing: {e}");
        }
    }
}

/// Chunks of `total` samples, nearest `focus` first, alternating after / before it.
fn order(total: u64, chunk: u64, focus: u64) -> Vec<Range<u64>> {
    let n = total.div_ceil(chunk);
    if n == 0 {
        return Vec::new();
    }
    let at = (focus / chunk).min(n - 1);
    let mut index = vec![at];
    for d in 1..n {
        index.extend([at + d].into_iter().filter(|&i| i < n));
        index.extend(at.checked_sub(d));
    }
    index.into_iter().map(|i| i * chunk..((i + 1) * chunk).min(total)).collect()
}

fn fill(source: &dyn RecordingSource, summary: &MinMaxSummary, focus: &AtomicU64, cancel: &AtomicBool, on_progress: &dyn Fn(Progress)) {
    let info = source.info();
    let total = info.samples;
    let chunk = ((info.sample_rate_hz() * CHUNK_SEC) as u64).max(1);
    let done_of = |plan: &[Range<u64>]| plan.iter().filter(|r| summary.covers(r.start, r.end)).map(|r| r.end - r.start).sum::<u64>();
    let mut current = focus.load(Ordering::Relaxed);
    let mut plan = order(total, chunk, if current == NONE { 0 } else { current });
    loop {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        // A new focus: what is left, nearest it first
        let f = focus.load(Ordering::Relaxed);
        if f != current && f != NONE {
            current = f;
            plan = order(total, chunk, f);
        }
        let Some(next) = plan.iter().find(|r| !summary.covers(r.start, r.end)).cloned() else { return };
        let moved = || cancel.load(Ordering::Relaxed);
        match summary.fill(source, next.start, next.end, BLOCK_VALUES, moved) {
            Ok(true) => on_progress(Progress { filled: next, done: done_of(&plan), total }),
            Ok(false) => return,
            Err(e) => {
                tracing::warn!("summarizing stopped: {e}");
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::Duration;

    #[test]
    fn test_order_starts_at_the_focus_and_goes_outward() {
        let starts: Vec<u64> = order(100, 10, 45).iter().map(|r| r.start).collect();
        assert_eq!(starts, vec![40, 50, 30, 60, 20, 70, 10, 80, 0, 90]);
        assert_eq!(order(25, 10, 0).last().unwrap(), &(20..25), "the last chunk is short");
        assert_eq!(order(100, 10, 1000).first().unwrap().start, 90, "focus past the end: the end");
    }

    #[test]
    fn test_fills_everything_once_and_reports_progress() {
        // 4 channels, 9 s at 1 kHz
        let data: Vec<f32> = (0..4 * 9000).map(|i| ((i * 31) % 97) as f32).collect();
        let source: Arc<dyn RecordingSource> = Arc::new(dsp_core::MemoryRecording::new("t", data, 4, 1000.0).unwrap());
        let summary = Arc::new(MinMaxSummary::new(source.as_ref()));
        let reports = Arc::new(Mutex::new(Vec::new()));
        let r = reports.clone();
        let s = Summarizer::default();
        s.run(source.clone(), summary.clone(), 5000, Arc::new(AtomicBool::new(false)), Arc::new(move |p| r.lock().unwrap().push(p)));
        for _ in 0..500 {
            if reports.lock().unwrap().last().is_some_and(|p: &Progress| p.done == p.total) {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let reports = reports.lock().unwrap();
        assert!(summary.covers(0, 9000));
        assert_eq!(reports.first().unwrap().filled, 4000..6000, "the focus first");
        assert_eq!(reports.last().unwrap().done, 9000);
        assert_eq!(reports.len(), 5, "each chunk once");
    }
}
