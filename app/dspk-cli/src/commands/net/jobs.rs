//! How `serve` runs its clients' jobs: the separate `dspk-cli-sorter` program (GPL-3.0, never
//! linked), with progress as JSON lines on its stdout and its messages on stderr as the job's log.

use std::io::{BufRead as _, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use dspk_stream::jobs::{JobProgress, JobReport, JobRunner, JobSpec};

/// The sorters `dspk-cli-sorter` runs.
pub const SORTERS: [&str; 5] = ["kilosort4", "emusort", "mountainsort5", "spykingcircus2", "tridesclous2"];
/// How often a running sort checks whether it was asked to stop.
const POLL: Duration = Duration::from_millis(200);

/// Runs sorts with the `dspk-cli-sorter` at `program`. The compute runtime is the server's
/// (`DSPK_RUNTIME` is inherited).
pub struct SorterRunner {
    pub program: PathBuf,
}

impl JobRunner for SorterRunner {
    fn sorters(&self) -> Vec<String> {
        SORTERS.iter().map(|s| s.to_string()).collect()
    }

    fn run(&self, job: &JobSpec, report: &dyn JobReport, cancel: &AtomicBool) -> Result<String, String> {
        let mut cmd = Command::new(&self.program);
        cmd.arg("--progress").arg("json").arg(&job.sorter).arg(&job.recording).arg("--output").arg(&job.output);
        if let Some(s) = &job.source {
            cmd.arg("--source").arg(s);
        }
        if let Some(p) = &job.probe {
            cmd.arg("--probe").arg(p);
        }
        if let Some(t) = job.start_sec {
            cmd.arg("--start-sec").arg(t.to_string());
        }
        if let Some(d) = job.duration_sec {
            cmd.arg("--duration-sec").arg(d.to_string());
        }
        // Names were checked by the server (plain flag names, none reserved); values are one
        // argument each, never through a shell
        for (name, value) in &job.params {
            cmd.arg(format!("--{name}")).arg(value);
        }
        cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        report.log(&format!("{} {} → {}", job.sorter, job.recording.display(), job.output.display()));
        let mut child = cmd.spawn().map_err(|e| format!("starting {}: {e}", self.program.display()))?;
        let (stdout, stderr) = (child.stdout.take().expect("piped"), child.stderr.take().expect("piped"));
        let (tx, rx) = mpsc::channel::<String>();

        std::thread::scope(|scope| {
            scope.spawn(move || {
                for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
            });
            scope.spawn(|| {
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    report.log(&line);
                }
            });
            let mut outcome: Option<Result<String, String>> = None;
            loop {
                if cancel.load(Ordering::Relaxed) {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err("stopped".to_string());
                }
                match rx.recv_timeout(POLL) {
                    Ok(line) => {
                        if let Some(o) = read_event(&line, report) {
                            outcome = Some(o);
                        }
                    }
                    // The program closed its stdout: it is ending
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                }
            }
            let status = child.wait().map_err(|e| e.to_string())?;
            match outcome {
                Some(o) => o,
                None if status.success() => Ok("done".into()),
                None => Err(format!("dspk-cli-sorter exited with {status}")),
            }
        })
    }
}

/// One stdout line of `dspk-cli-sorter --progress json`: progress is reported; `done` and
/// `error` end the job; anything else goes to the log.
fn read_event(line: &str, report: &dyn JobReport) -> Option<Result<String, String>> {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
        report.log(line);
        return None;
    };
    let text = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or_default().to_string();
    let num = |k: &str| v.get(k).and_then(|x| x.as_u64()).unwrap_or(0);
    match v.get("event").and_then(|e| e.as_str()) {
        Some("progress") => {
            report.progress(JobProgress { stage: text("stage"), step: num("step") as u32, steps: num("steps") as u32, done: num("done"), total: num("total"), unit: text("unit") });
            None
        }
        Some("done") => {
            let seconds = v.get("seconds").and_then(|x| x.as_f64()).unwrap_or(0.0);
            let summary = format!("{} units, {} spikes in {seconds:.1} s", num("units"), num("spikes"));
            report.log(&summary);
            Some(Ok(summary))
        }
        Some("error") => {
            let message = text("message");
            report.log(&format!("error: {message}"));
            Some(Err(message))
        }
        _ => {
            report.log(line);
            None
        }
    }
}
