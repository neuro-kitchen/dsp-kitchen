//! Long work started from the UI, on its own thread, with progress and cancel ([`JobMessage`],
//! [`Cancel`]): writing a view (its recording through its op chain, dspk-pipeline). Plugins run
//! theirs the same way (electrophysiology: sorting through the separate `dspk-cli-sorter`).

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use dspk_pipeline::{OutputFormat, PipelineSpec};

use crate::domain::session::ViewDoc;

/// A message from a running job.
#[derive(Debug, Clone)]
pub enum JobMessage {
    /// Progress line for the status bar (and the fraction done when known).
    Progress(String, Option<f32>),
    /// Finished: the file written.
    Done(PathBuf),
    Failed(String),
}

/// Cancels a running job.
#[derive(Clone, Default)]
pub struct Cancel {
    flag: Arc<AtomicBool>,
    child: Arc<Mutex<Option<Child>>>,
}

impl Cancel {
    /// A handle that `kill`s `child` on cancel (programs a job runs).
    pub fn with_child(&self, child: Child) {
        *self.child.lock().unwrap() = Some(child);
    }

    /// Waits for the child given with [`Self::with_child`].
    pub fn wait_child(&self) -> Option<std::io::Result<std::process::ExitStatus>> {
        self.child.lock().unwrap().as_mut().map(|c| c.wait())
    }

    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
        if let Some(child) = self.child.lock().unwrap().as_mut() {
            let _ = child.kill();
        }
    }

    pub fn cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }
}

/// The processed recording of `view` written to `output` (Zarr: compressed, parallel; float32).
pub fn write_view(view: &ViewDoc, output: PathBuf, send: impl Fn(JobMessage) + Send + 'static) -> Cancel {
    let spec: PipelineSpec = view.pipeline(output.clone(), OutputFormat::Zarr);
    let cancel = Cancel::default();
    let c = cancel.clone();
    std::thread::spawn(move || {
        let base = output.parent().map(Path::to_path_buf).unwrap_or_default();
        let open = |p: &Path, source: Option<&str>| crate::plugin::open(p, source);
        let result = dspk_pipeline::run_with(&spec, &base, open, |p| {
            let f = p.windows_done as f32 / p.windows.max(1) as f32;
            send(JobMessage::Progress(format!("Writing {} · {:.0} %", file_name(&output), f * 100.0), Some(f)));
            !c.cancelled()
        });
        send(match result {
            Ok(_) => JobMessage::Done(output),
            Err(e) => JobMessage::Failed(format!("Write failed: {e}")),
        });
    });
    cancel
}

/// The events of `view` (its chain ends with a `crossings` op) written as the Zarr group `output`
/// (inside the recording's `.dspk.zarr`).
pub fn write_events(view: &ViewDoc, output: PathBuf, send: impl Fn(JobMessage) + Send + 'static) -> Cancel {
    let spec: PipelineSpec = view.pipeline(output.clone(), OutputFormat::Events);
    let cancel = Cancel::default();
    let c = cancel.clone();
    std::thread::spawn(move || {
        let base = output.parent().map(Path::to_path_buf).unwrap_or_default();
        let open = |p: &Path, source: Option<&str>| crate::plugin::open(p, source);
        let result = dspk_pipeline::run_with(&spec, &base, open, |p| {
            let f = p.windows_done as f32 / p.windows.max(1) as f32;
            send(JobMessage::Progress(format!("Finding events · {:.0} %", f * 100.0), Some(f)));
            !c.cancelled()
        });
        send(match result {
            Ok(_) => JobMessage::Done(output),
            Err(e) => JobMessage::Failed(format!("Events failed: {e}")),
        });
    });
    cancel
}

/// Whether `view`'s chain gives events (its last enabled op is `crossings`).
pub fn gives_events(view: &ViewDoc) -> bool {
    view.ops.iter().rev().find(|o| o.enabled).is_some_and(|o| matches!(o.node, dspk_pipeline::Node::Crossings { .. }))
}

/// Writes several chains of one recording in one read of it (a branched spec: chains that start
/// with the same ops share those nodes in the spec); `outputs[i]` is chain `i`'s file.
pub fn write_branches(chains: Vec<ViewDoc>, outputs: Vec<PathBuf>, send: impl Fn(JobMessage) + Send + 'static) -> Cancel {
    let cancel = Cancel::default();
    let Some(first) = chains.first().cloned() else {
        send(JobMessage::Failed("nothing to write".into()));
        return cancel;
    };
    let mut nodes = std::collections::BTreeMap::new();
    let mut edges = Vec::new();
    nodes.insert(
        "read".to_string(),
        dspk_pipeline::Node::Read { path: Some(first.recording.path.clone()), inputs: None, source: first.recording.source.clone(), channels: first.recording.channels.clone(), start: None, stop: None },
    );
    // Shared first ops become shared nodes (keyed by the ops so far)
    let mut ids: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for (i, (v, out)) in chains.iter().zip(&outputs).enumerate() {
        let (mut last, mut key) = ("read".to_string(), String::new());
        for op in v.ops.iter().filter(|o| o.enabled) {
            key.push_str(&serde_json::to_string(&op.node).unwrap_or_default());
            key.push('|');
            let id = match ids.get(&key) {
                Some(id) => id.clone(),
                None => {
                    let id = format!("n{}", ids.len() + 1);
                    nodes.insert(id.clone(), op.node.clone());
                    edges.push((last.clone(), id.clone()));
                    ids.insert(key.clone(), id.clone());
                    id
                }
            };
            last = id;
        }
        let w = format!("write{}", i + 1);
        nodes.insert(w.clone(), dspk_pipeline::Node::Write { path: out.clone(), format: OutputFormat::Zarr, compression: None, chunk_samples: None, parallel: None });
        edges.push((last, w));
    }
    let spec = PipelineSpec { dspk_pipeline: dspk_pipeline::SPEC_VERSION, name: Some(format!("{} branches", chains.len())), nodes, edges, execution: Default::default() };
    let c = cancel.clone();
    let done = outputs[0].clone();
    std::thread::spawn(move || {
        let base = done.parent().map(Path::to_path_buf).unwrap_or_default();
        let open = |p: &Path, source: Option<&str>| crate::plugin::open(p, source);
        let n = outputs.len();
        let result = dspk_pipeline::run_with(&spec, &base, open, |p| {
            let f = p.windows_done as f32 / p.windows.max(1) as f32;
            send(JobMessage::Progress(format!("Writing {n} outputs in one read · {:.0} %", f * 100.0), Some(f)));
            !c.cancelled()
        });
        send(match result {
            Ok(_) => JobMessage::Done(done),
            Err(e) => JobMessage::Failed(format!("Write failed: {e}")),
        });
    });
    cancel
}

/// Runs a program (e.g. a plugin's `dspk-cli-sorter`) that writes `output`: its output lines are
/// progress (bars redrawn with `\r` split, colour codes removed), success is a zero exit status.
pub fn run_program(mut cmd: Command, output: PathBuf, send: impl Fn(JobMessage) + Send + Sync + 'static) -> Cancel {
    let cancel = Cancel::default();
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            send(JobMessage::Failed(format!("Could not start {}: {e}", cmd.get_program().to_string_lossy())));
            return cancel;
        }
    };
    let (stdout, stderr) = (child.stdout.take(), child.stderr.take());
    cancel.with_child(child);
    let send: Arc<dyn Fn(JobMessage) + Send + Sync> = Arc::new(send);
    let last = Arc::new(Mutex::new(String::new()));
    let follow = |pipe: Option<Box<dyn std::io::Read + Send>>, send: Arc<dyn Fn(JobMessage) + Send + Sync>, last: Arc<Mutex<String>>| {
        std::thread::spawn(move || {
            let Some(pipe) = pipe else { return };
            let mut reader = BufReader::new(pipe);
            let mut buf = Vec::new();
            while reader.read_until(b'\n', &mut buf).map(|n| n > 0).unwrap_or(false) {
                for piece in String::from_utf8_lossy(&buf).split('\r') {
                    let line = strip_ansi(piece.trim());
                    if line.is_empty() {
                        continue;
                    }
                    // A plugin program's JSON line (`--progress json`), else plain text
                    match json_progress(&line) {
                        Some(JsonLine::Progress(text, fraction)) => send(JobMessage::Progress(text, fraction)),
                        Some(JsonLine::Error(message)) => *last.lock().unwrap() = message,
                        Some(JsonLine::Other) => {}
                        None => {
                            *last.lock().unwrap() = line.clone();
                            send(JobMessage::Progress(line, None));
                        }
                    }
                }
                buf.clear();
            }
        })
    };
    let t1 = follow(stdout.map(|p| Box::new(p) as Box<dyn std::io::Read + Send>), send.clone(), last.clone());
    let t2 = follow(stderr.map(|p| Box::new(p) as Box<dyn std::io::Read + Send>), send.clone(), last.clone());
    let c = cancel.clone();
    std::thread::spawn(move || {
        let _ = (t1.join(), t2.join());
        let msg = match c.wait_child() {
            _ if c.cancelled() => JobMessage::Failed("Cancelled".into()),
            Some(Ok(s)) if s.success() => JobMessage::Done(output),
            Some(Ok(s)) => JobMessage::Failed(format!("Failed ({s}): {}", last.lock().unwrap())),
            Some(Err(e)) => JobMessage::Failed(format!("Failed: {e}")),
            None => JobMessage::Failed("Failed".into()),
        };
        send(msg);
    });
    cancel
}

/// A JSON line of a plugin program (`dspk-cli-sorter --progress json`).
#[derive(Debug, PartialEq)]
enum JsonLine {
    /// `[2/4] Finding clips 12/40 windows`, and the fraction of the whole run.
    Progress(String, Option<f32>),
    Error(String),
    Other,
}

fn json_progress(line: &str) -> Option<JsonLine> {
    if !line.starts_with('{') {
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    Some(match v.get("event")?.as_str()? {
        "progress" => {
            let n = |k: &str| v.get(k).and_then(serde_json::Value::as_u64).unwrap_or(0);
            let (step, steps, done, total) = (n("step"), n("steps"), n("done"), n("total"));
            let stage = v.get("stage").and_then(serde_json::Value::as_str).unwrap_or("");
            let unit = v.get("unit").and_then(serde_json::Value::as_str).unwrap_or("");
            let text = if total > 0 { format!("[{step}/{steps}] {stage} {done}/{total} {unit}") } else { format!("[{step}/{steps}] {stage}") };
            // Steps count equally: the fraction of the whole run
            let fraction = (steps > 0).then(|| {
                let within = if total > 0 { done.min(total) as f32 / total as f32 } else { 0.0 };
                ((step.saturating_sub(1)) as f32 + within) / steps as f32
            });
            JsonLine::Progress(text, fraction)
        }
        "error" => JsonLine::Error(v.get("message").and_then(serde_json::Value::as_str).unwrap_or("error").to_string()),
        _ => JsonLine::Other,
    })
}

/// `rec.bin` → `rec.<view id>.processed.zarr` next to it.
pub fn processed_path(recording: &Path, view: &str) -> PathBuf {
    let stem = recording.file_stem().map_or_else(|| "recording".into(), |n| n.to_string_lossy().into_owned());
    recording.with_file_name(format!("{stem}.{view}.processed.zarr"))
}

pub fn file_name(p: &Path) -> String {
    p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned())
}

/// Removes terminal colour codes (`ESC [ … m`) from a progress line.
pub fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for d in chars.by_ref() {
                if d.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paths_and_ansi() {
        assert_eq!(processed_path(Path::new("/d/rec.bin"), "v2"), PathBuf::from("/d/rec.v2.processed.zarr"));
        assert_eq!(strip_ansi("\u{1b}[32mdone\u{1b}[0m 3 units"), "done 3 units");
        let line = r#"{"event":"progress","stage":"Finding clips","step":2,"steps":4,"done":10,"total":40,"unit":"windows"}"#;
        assert_eq!(json_progress(line), Some(JsonLine::Progress("[2/4] Finding clips 10/40 windows".into(), Some(0.3125))));
        assert_eq!(json_progress(r#"{"event":"error","message":"no probe"}"#), Some(JsonLine::Error("no probe".into())));
        assert_eq!(json_progress("plain text"), None);
    }
}
