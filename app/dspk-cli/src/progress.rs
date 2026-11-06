//! Terminal progress bar for long runs: one line per stage on stderr,
//! `[2/4] Finding clips  ████░░░░  12/40 windows  0:05 < 0:12`, the time left estimated from the
//! rate so far. Any run reporting through [`dspk_core::ProgressSink`] can draw it. When stderr
//! is not a terminal (container logs, journald, a file), plain lines instead, a few per stage.

use std::io::{IsTerminal as _, Write};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use dspk_core::{ProgressEvent, ProgressSink};

/// Characters of the bar.
const BAR_WIDTH: usize = 24;
/// Least time between two redraws of a stage's line.
const REDRAW: Duration = Duration::from_millis(100);
/// Least time between two lines of a stage when stderr is not a terminal (logs).
const LOG_EVERY: Duration = Duration::from_secs(10);
const SECS_PER_MINUTE: u64 = 60;
const SECS_PER_HOUR: u64 = 3600;

#[derive(Default)]
struct Line {
    /// `(step, stage)` drawn now.
    key: Option<(usize, String)>,
    start: Option<Instant>,
    drawn: Option<Instant>,
    width: usize,
}

/// Draws progress on stderr, one line per stage, ending each line when its stage completes; not
/// on a terminal, a plain line every [`LOG_EVERY`] and the last.
pub struct TerminalProgress {
    line: Mutex<Line>,
    terminal: bool,
}

impl Default for TerminalProgress {
    fn default() -> Self {
        Self { line: Mutex::new(Line::default()), terminal: std::io::stderr().is_terminal() }
    }
}

fn clock(d: Duration) -> String {
    let s = d.as_secs();
    let (h, m, s) = (s / SECS_PER_HOUR, s % SECS_PER_HOUR / SECS_PER_MINUTE, s % SECS_PER_MINUTE);
    if h > 0 { format!("{h}:{m:02}:{s:02}") } else { format!("{m}:{s:02}") }
}

impl ProgressSink for TerminalProgress {
    fn report(&self, e: &ProgressEvent<'_>) {
        let mut line = self.line.lock().expect("progress lock");
        let now = Instant::now();
        let key = (e.step, e.stage.to_string());
        if line.key.as_ref() != Some(&key) {
            if line.key.is_some() && self.terminal {
                eprintln!();
            }
            *line = Line { key: Some(key), start: Some(now), drawn: None, width: 0 };
        }
        let finished = e.total > 0 && e.done >= e.total;
        let every = if self.terminal { REDRAW } else { LOG_EVERY };
        if !finished && line.drawn.is_some_and(|t| now - t < every) {
            return;
        }
        line.drawn = Some(now);
        let elapsed = now - line.start.unwrap_or(now);
        let label = format!("[{}/{}] {}", e.step, e.steps, e.stage);
        let text = if e.total > 0 {
            let filled = (BAR_WIDTH as u64 * e.done.min(e.total) / e.total) as usize;
            let bar = format!("{}{}", "█".repeat(filled), "░".repeat(BAR_WIDTH - filled));
            let left = (e.done > 0 && e.done < e.total).then(|| format!(" < {}", clock(elapsed.mul_f64((e.total - e.done) as f64 / e.done as f64))));
            format!("{label}  {bar}  {}/{} {}  {}{}", e.done, e.total, e.unit, clock(elapsed), left.unwrap_or_default())
        } else {
            format!("{label}  {} {}  {}", e.done, e.unit, clock(elapsed))
        };
        let mut err = std::io::stderr().lock();
        if !self.terminal {
            let _ = writeln!(err, "{text}");
            if finished {
                line.key = None;
            }
            return;
        }
        let width = text.chars().count();
        // Pad over a longer previous line
        let _ = write!(err, "\r{text:<0$}", line.width.max(width));
        let _ = err.flush();
        line.width = width;
        if finished {
            let _ = writeln!(err);
            line.key = None;
        }
    }
}

/// Progress as JSON lines on stdout, for programs that run this one (the app runs
/// `dspk-cli-sorter --progress json`): `{"event": "progress", "stage": …, "step": 2, "steps": 4,
/// "done": 12, "total": 40, "unit": "windows"}`, at most one line per stage every
/// [`REDRAW`] (the last of a stage always).
#[derive(Default)]
pub struct JsonProgress {
    last: Mutex<Option<((usize, String), Instant)>>,
}

impl ProgressSink for JsonProgress {
    fn report(&self, e: &ProgressEvent<'_>) {
        let mut last = self.last.lock().expect("progress lock");
        let now = Instant::now();
        let key = (e.step, e.stage.to_string());
        let finished = e.total > 0 && e.done >= e.total;
        if !finished && last.as_ref().is_some_and(|(k, t)| *k == key && now - *t < REDRAW) {
            return;
        }
        *last = Some((key, now));
        json_line(&serde_json::json!({
            "event": "progress", "stage": e.stage, "step": e.step, "steps": e.steps,
            "done": e.done, "total": e.total, "unit": e.unit,
        }));
    }
}

/// One JSON object on its own stdout line (flushed: a reader sees it at once).
pub fn json_line(value: &serde_json::Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{value}");
    let _ = out.flush();
}

/// How a run reports progress: a bar on stderr, or JSON lines on stdout.
pub enum RunProgress {
    Bar(TerminalProgress),
    Json(JsonProgress),
}

impl RunProgress {
    pub fn new(json: bool) -> Self {
        if json { Self::Json(JsonProgress::default()) } else { Self::Bar(TerminalProgress::default()) }
    }
}

impl ProgressSink for RunProgress {
    fn report(&self, e: &ProgressEvent<'_>) {
        match self {
            Self::Bar(p) => p.report(e),
            Self::Json(p) => p.report(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_formats_minutes_and_hours() {
        assert_eq!(clock(Duration::from_secs(65)), "1:05");
        assert_eq!(clock(Duration::from_secs(3 * 3600 + 7)), "3:00:07");
    }
}
