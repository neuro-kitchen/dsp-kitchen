//! The jobs a server runs for its clients (sorts), one at a time.
//!
//! The device's state follows Synapse's model: Stopped ⇄ Running, Error. A client with the
//! control permission configures a job (only while nothing runs), starts it, and may stop it;
//! any client reads the status and the log. Every request is checked before anything runs:
//! - the sorter is one the runner has;
//! - the recording is under the data root;
//! - the output is a new path under the data root (never overwritten);
//! - parameter names are plain flag names, none of the ones the server sets itself, and values
//!   are one line.
//!
//! Each control request (allowed or refused) is appended to an audit log (JSON lines, mode
//! 0600). How a job runs is the caller's ([`JobRunner`]): the command line runs the separate
//! `dspk-cli-sorter` program.

use std::collections::VecDeque;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use super::auth::Peer;
use super::files::{kind_of, DataRoot};
use crate::error::{StreamError, StreamResult};
use crate::protocol::wire;

/// Log lines a server keeps of the current or last job.
pub const MAX_LOG_LINES: usize = 2_000;
/// Longest parameter value (characters).
pub const MAX_PARAM_CHARS: usize = 256;
/// Most parameters one job may set.
pub const MAX_PARAMS: usize = 64;
/// Parameter names the server sets itself: a client cannot set them through `params`.
/// `probe-file` and `channel-map` name files: a client could point them outside the data root.
pub const RESERVED_PARAMS: [&str; 11] = ["output", "o", "progress", "source", "probe", "probe-file", "channel-map", "start-sec", "duration-sec", "help", "version"];
/// What a sorting's path ends with.
pub const SORTING_SUFFIX: &str = ".sorting.zarr";

/// A checked job: absolute paths under the data root.
#[derive(Debug, Clone, PartialEq)]
pub struct JobSpec {
    pub sorter: String,
    pub recording: PathBuf,
    pub output: PathBuf,
    pub source: Option<String>,
    pub probe: Option<String>,
    /// `(flag name without "--", value)`.
    pub params: Vec<(String, String)>,
    pub start_sec: Option<f64>,
    pub duration_sec: Option<f64>,
}

/// Progress of a running job.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JobProgress {
    pub stage: String,
    pub step: u32,
    pub steps: u32,
    pub done: u64,
    pub total: u64,
    pub unit: String,
}

/// Where a running job reports to.
pub trait JobReport: Send + Sync {
    fn progress(&self, progress: JobProgress);
    fn log(&self, line: &str);
}

/// Runs jobs (the caller's: the command line runs `dspk-cli-sorter`).
pub trait JobRunner: Send + Sync {
    /// The sorters this server can run.
    fn sorters(&self) -> Vec<String>;
    /// Runs `job` to the end (blocking), reporting as it goes; stops early once `cancel` is set.
    /// `Ok`: a one-line summary of the result.
    fn run(&self, job: &JobSpec, report: &dyn JobReport, cancel: &AtomicBool) -> Result<String, String>;
}

struct JobState {
    state: wire::DeviceState,
    job: Option<JobSpec>,
    description: String,
    progress: JobProgress,
    message: String,
    started_by: String,
    started_unix: u64,
    logs: VecDeque<String>,
    cancel: Arc<AtomicBool>,
}

/// The server's jobs: one configured or running at a time.
pub struct Jobs {
    runner: Arc<dyn JobRunner>,
    root: DataRoot,
    audit: Option<PathBuf>,
    state: Mutex<JobState>,
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn refuse(why: impl Into<String>) -> StreamError {
    StreamError::Server(why.into())
}

struct Reporter(Arc<Jobs>);

impl JobReport for Reporter {
    fn progress(&self, progress: JobProgress) {
        self.0.lock().progress = progress;
    }

    fn log(&self, line: &str) {
        // Paths relative to the data root: clients never learn where it is on the server's disk
        let root = format!("{}/", self.0.root.path().display());
        let line = line.replace(&root, "");
        let mut s = self.0.lock();
        if s.logs.len() >= MAX_LOG_LINES {
            s.logs.pop_front();
        }
        s.logs.push_back(line);
    }
}

impl Jobs {
    /// Jobs run by `runner` on files under `root`; control requests are appended to `audit`
    /// (JSON lines, created with mode 0600) when given.
    pub fn new(runner: Arc<dyn JobRunner>, root: DataRoot, audit: Option<PathBuf>) -> Self {
        let state = JobState {
            state: wire::DeviceState::Stopped,
            job: None,
            description: String::new(),
            progress: JobProgress::default(),
            message: String::new(),
            started_by: String::new(),
            started_unix: 0,
            logs: VecDeque::new(),
            cancel: Arc::new(AtomicBool::new(false)),
        };
        Self { runner, root, audit, state: Mutex::new(state) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, JobState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn state(&self) -> wire::DeviceState {
        self.lock().state
    }

    /// The device's state and its job.
    pub fn status(&self) -> wire::Status {
        let s = self.lock();
        wire::Status {
            state: s.state as i32,
            job: s.description.clone(),
            stage: s.progress.stage.clone(),
            step: s.progress.step,
            steps: s.progress.steps,
            done: s.progress.done,
            total: s.progress.total,
            unit: s.progress.unit.clone(),
            message: s.message.clone(),
            output: s.job.as_ref().map(|j| self.root.relative(&j.output)).unwrap_or_default(),
            started_by: s.started_by.clone(),
            started_unix: s.started_unix,
        }
    }

    /// The last `n` log lines (0 = all kept).
    pub fn logs(&self, n: usize) -> Vec<String> {
        let s = self.lock();
        let skip = if n == 0 { 0 } else { s.logs.len().saturating_sub(n) };
        s.logs.iter().skip(skip).cloned().collect()
    }

    /// Appends one control request to the audit log (failures to write are logged, not fatal).
    fn audit(&self, peer: &Peer, action: &str, detail: &str, result: &Result<(), String>) {
        let Some(path) = &self.audit else { return };
        let line = serde_json::json!({
            "unix": now_unix(), "client": peer.name, "permission": peer.permission.to_string(),
            "action": action, "detail": detail, "ok": result.is_ok(), "error": result.as_ref().err(),
        });
        let mut options = std::fs::OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        if let Err(e) = options.open(path).and_then(|mut f| writeln!(f, "{line}")) {
            tracing::warn!("audit log {}: {e}", path.display());
        }
    }

    fn require_control(peer: &Peer) -> StreamResult<()> {
        if peer.can_control() { Ok(()) } else { Err(refuse(format!("{} may only read (enrolled without --control)", peer.name))) }
    }

    /// Checks and sets the job (only while nothing runs); clears a previous error.
    pub fn configure(&self, peer: &Peer, request: wire::Configure) -> StreamResult<wire::Status> {
        let detail = format!("{} {} → {}", request.sorter, request.recording, request.output);
        let result = Self::require_control(peer).and_then(|()| self.check(&request)).and_then(|job| {
            let mut s = self.lock();
            if s.state == wire::DeviceState::Running {
                return Err(refuse("a job is running: stop it first"));
            }
            s.description = format!("{} on {}", job.sorter, self.root.relative(&job.recording));
            s.job = Some(job);
            s.state = wire::DeviceState::Stopped;
            s.progress = JobProgress::default();
            s.message = "configured".into();
            Ok(())
        });
        self.audit(peer, "configure", &detail, &result.as_ref().map(|_| ()).map_err(|e| e.reason()));
        result.map(|()| self.status())
    }

    /// The job a request asks for, checked (see the module docs).
    fn check(&self, r: &wire::Configure) -> StreamResult<JobSpec> {
        if !self.runner.sorters().iter().any(|s| s == &r.sorter) {
            return Err(refuse(format!("unknown sorter {:?}: this server has {}", r.sorter, self.runner.sorters().join(", "))));
        }
        let recording = self.root.resolve(&r.recording)?;
        if kind_of(&recording, recording.is_dir()) != wire::FileKind::Recording {
            return Err(refuse(format!("{:?} is not a recording", r.recording)));
        }
        let output_rel = if r.output.is_empty() {
            let name = recording.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let stem = name.split('.').take_while(|p| !["bin", "cbin", "dat", "raw", "nwb", "h5", "zarr"].contains(p)).collect::<Vec<_>>().join(".");
            let parent = self.root.relative(recording.parent().unwrap_or(self.root.path()));
            let file = format!("{}.{}{SORTING_SUFFIX}", if stem.is_empty() { "recording" } else { &stem }, r.sorter);
            if parent.is_empty() { file } else { format!("{parent}/{file}") }
        } else {
            r.output.clone()
        };
        if !output_rel.ends_with(SORTING_SUFFIX) {
            return Err(refuse(format!("the output must end with {SORTING_SUFFIX}")));
        }
        let output = self.root.resolve_new(&output_rel)?;
        if r.params.len() > MAX_PARAMS {
            return Err(refuse(format!("{} parameters: at most {MAX_PARAMS}", r.params.len())));
        }
        let mut params: Vec<(String, String)> = Vec::new();
        for (name, value) in &r.params {
            let plain = !name.is_empty() && name.len() <= 64 && name.starts_with(|c: char| c.is_ascii_lowercase()) && name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
            if !plain {
                return Err(refuse(format!("parameter {name:?}: a flag name (lowercase letters, digits, '-')")));
            }
            if RESERVED_PARAMS.contains(&name.as_str()) {
                return Err(refuse(format!("parameter {name:?} is set by the server")));
            }
            if value.chars().count() > MAX_PARAM_CHARS || value.contains(['\n', '\r', '\0']) {
                return Err(refuse(format!("parameter {name:?}: one line of at most {MAX_PARAM_CHARS} characters")));
            }
            params.push((name.clone(), value.clone()));
        }
        params.sort();
        let plain_word = |s: &str| s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
        for (what, v) in [("source", &r.source), ("probe", &r.probe)] {
            if v.len() > 64 || !plain_word(v) {
                return Err(refuse(format!("{what} {v:?}: letters, digits, '-', '_', '.'")));
            }
        }
        for (what, v) in [("start", r.start_sec), ("duration", r.duration_sec)] {
            if !v.is_finite() || v < 0.0 {
                return Err(refuse(format!("{what} {v}: seconds ≥ 0")));
            }
        }
        Ok(JobSpec {
            sorter: r.sorter.clone(),
            recording,
            output,
            source: Some(r.source.clone()).filter(|s| !s.is_empty()),
            probe: Some(r.probe.clone()).filter(|s| !s.is_empty()),
            params,
            start_sec: Some(r.start_sec).filter(|&t| t > 0.0),
            duration_sec: Some(r.duration_sec).filter(|&t| t > 0.0),
        })
    }

    /// Starts the configured job in the background.
    pub fn start(self: &Arc<Self>, peer: &Peer) -> StreamResult<wire::Status> {
        let result = Self::require_control(peer).and_then(|()| {
            let mut s = self.lock();
            if s.state == wire::DeviceState::Running {
                return Err(refuse("a job is running already"));
            }
            let job = s.job.clone().ok_or_else(|| refuse("nothing configured: configure a job first"))?;
            // The output must still be new (another job may have written it since)
            if job.output.symlink_metadata().is_ok() {
                return Err(refuse(format!("{} exists already: configure another output", self.root.relative(&job.output))));
            }
            s.state = wire::DeviceState::Running;
            s.progress = JobProgress::default();
            s.message = "running".into();
            s.started_by = peer.name.clone();
            s.started_unix = now_unix();
            s.logs.clear();
            s.cancel = Arc::new(AtomicBool::new(false));
            let (cancel, jobs) = (s.cancel.clone(), self.clone());
            std::thread::spawn(move || {
                let reporter = Reporter(jobs.clone());
                let outcome = jobs.runner.run(&job, &reporter, &cancel);
                // A stopped or failed job leaves no partial output: it was new when the job started
                // (checked), so it is this job's, and a rerun would be refused while it is there
                if outcome.is_err() && job.output.symlink_metadata().is_ok() {
                    let removed = if job.output.is_dir() { std::fs::remove_dir_all(&job.output) } else { std::fs::remove_file(&job.output) };
                    match removed {
                        Ok(()) => reporter.log(&format!("removed the partial output {}", job.output.display())),
                        Err(e) => reporter.log(&format!("could not remove the partial output {}: {e}", job.output.display())),
                    }
                }
                let mut s = jobs.lock();
                (s.state, s.message) = match outcome {
                    Ok(summary) => (wire::DeviceState::Stopped, summary),
                    Err(_) if cancel.load(Ordering::Relaxed) => (wire::DeviceState::Stopped, "stopped".into()),
                    Err(why) => (wire::DeviceState::Error, why),
                };
            });
            Ok(())
        });
        self.audit(peer, "start", &self.lock().description.clone(), &result.as_ref().map(|_| ()).map_err(|e| e.reason()));
        result.map(|()| self.status())
    }

    /// Asks the running job to stop (it ends at its next check).
    pub fn stop(&self, peer: &Peer) -> StreamResult<wire::Status> {
        let result = Self::require_control(peer).and_then(|()| {
            let s = self.lock();
            if s.state != wire::DeviceState::Running {
                return Err(refuse("no job is running"));
            }
            s.cancel.store(true, Ordering::Relaxed);
            Ok(())
        });
        self.audit(peer, "stop", &self.lock().description.clone(), &result.as_ref().map(|_| ()).map_err(|e| e.reason()));
        result.map(|()| self.status())
    }

    /// Stops the running job, if any, and waits up to `timeout` for it to end (the server shutting
    /// down: no job outlives it). `true` when nothing runs any more.
    pub fn shutdown(&self, timeout: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        self.lock().cancel.store(true, Ordering::Relaxed);
        while self.state() == wire::DeviceState::Running {
            if std::time::Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        true
    }

    pub fn data_root(&self) -> &DataRoot {
        &self.root
    }

    pub fn audit_path(&self) -> Option<&Path> {
        self.audit.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::auth::Permission;

    /// Writes the output folder after a few progress reports, or waits to be cancelled.
    struct FakeRunner {
        block: bool,
    }

    impl JobRunner for FakeRunner {
        fn sorters(&self) -> Vec<String> {
            vec!["kilosort4".into()]
        }
        fn run(&self, job: &JobSpec, report: &dyn JobReport, cancel: &AtomicBool) -> Result<String, String> {
            report.log("starting");
            for i in 0..3 {
                report.progress(JobProgress { stage: "Sorting".into(), step: 1, steps: 1, done: i, total: 3, unit: "steps".into() });
            }
            while self.block && !cancel.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            report.log(&format!("writing {}", job.output.display()));
            std::fs::create_dir_all(&job.output).map_err(|e| e.to_string())?;
            Ok("3 units".into())
        }
    }

    /// Writes part of its output, then fails (or waits to be stopped).
    struct PartialRunner {
        block: bool,
    }

    impl JobRunner for PartialRunner {
        fn sorters(&self) -> Vec<String> {
            vec!["kilosort4".into()]
        }
        fn run(&self, job: &JobSpec, _: &dyn JobReport, cancel: &AtomicBool) -> Result<String, String> {
            std::fs::create_dir_all(job.output.join("spikes")).map_err(|e| e.to_string())?;
            while self.block && !cancel.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            Err("out of memory".into())
        }
    }

    /// A stopped or failed job removes its partial output, so a rerun is not refused; shutdown
    /// stops a running job.
    #[test]
    fn partial_outputs_are_removed_and_shutdown_stops_the_job() {
        let base = std::env::temp_dir().join(format!("dspk-stream-partial-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("data");
        std::fs::create_dir_all(root.join("m1")).unwrap();
        std::fs::write(root.join("m1/rec.bin"), [0u8; 8]).unwrap();
        let ctl = Peer { name: "lab".into(), permission: Permission::Control };
        let job = || wire::Configure { sorter: "kilosort4".into(), recording: "m1/rec.bin".into(), ..Default::default() };
        let output = root.join("m1/rec.kilosort4.sorting.zarr");

        let failing = Arc::new(Jobs::new(Arc::new(PartialRunner { block: false }), DataRoot::new(&root).unwrap(), None));
        failing.configure(&ctl, job()).unwrap();
        failing.start(&ctl).unwrap();
        wait_until(&failing, wire::DeviceState::Error);
        assert!(!output.exists(), "the failed job's partial output is removed");
        assert!(failing.logs(0).iter().any(|l| l.contains("removed the partial output")));
        assert!(failing.configure(&ctl, job()).is_ok(), "a rerun is not refused");

        let running = Arc::new(Jobs::new(Arc::new(PartialRunner { block: true }), DataRoot::new(&root).unwrap(), None));
        running.configure(&ctl, job()).unwrap();
        running.start(&ctl).unwrap();
        assert!(running.shutdown(std::time::Duration::from_secs(5)), "shutdown stops the running job");
        assert!(!output.exists(), "and removes its partial output");
        let _ = std::fs::remove_dir_all(&base);
    }

    fn wait_until(jobs: &Jobs, state: wire::DeviceState) {
        for _ in 0..400 {
            if jobs.state() == state {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("state stayed {:?}", jobs.state());
    }

    #[test]
    fn configure_start_stop_with_checks_and_audit() {
        let base = std::env::temp_dir().join(format!("dspk-stream-jobs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("data");
        std::fs::create_dir_all(root.join("m1")).unwrap();
        std::fs::write(root.join("m1/rec.ap.bin"), [0u8; 8]).unwrap();
        let audit = base.join("audit.jsonl");
        let ctl = Peer { name: "lab".into(), permission: Permission::Control };
        let reader = Peer { name: "alice".into(), permission: Permission::Read };
        let ok = |sorter: &str| wire::Configure { sorter: sorter.into(), recording: "m1/rec.ap.bin".into(), ..Default::default() };

        let jobs = Arc::new(Jobs::new(Arc::new(FakeRunner { block: false }), DataRoot::new(&root).unwrap(), Some(audit.clone())));
        assert!(jobs.configure(&reader, ok("kilosort4")).is_err(), "read-only clients cannot configure");
        assert!(jobs.start(&ctl).is_err(), "nothing configured");
        assert!(jobs.configure(&ctl, ok("rm -rf")).is_err(), "unknown sorter");
        let mut bad = ok("kilosort4");
        bad.params.insert("output".into(), "/etc".into());
        assert!(jobs.configure(&ctl, bad).is_err(), "reserved parameter");
        let mut bad = ok("kilosort4");
        bad.params.insert("--x; rm".into(), "1".into());
        assert!(jobs.configure(&ctl, bad).is_err(), "not a flag name");
        let mut bad = ok("kilosort4");
        bad.output = "../out.sorting.zarr".into();
        assert!(jobs.configure(&ctl, bad).is_err(), "output outside the root");
        let mut bad = ok("kilosort4");
        bad.recording = "../outside.bin".into();
        assert!(jobs.configure(&ctl, bad).is_err(), "recording outside the root");

        let mut good = ok("kilosort4");
        good.params.insert("th-universal".into(), "9".into());
        let status = jobs.configure(&ctl, good).unwrap();
        assert_eq!(status.output, "m1/rec.ap.kilosort4.sorting.zarr");
        assert!(jobs.start(&reader).is_err(), "read-only clients cannot start");
        jobs.start(&ctl).unwrap();
        wait_until(&jobs, wire::DeviceState::Stopped);
        assert_eq!(jobs.status().message, "3 units");
        assert!(root.join("m1/rec.ap.kilosort4.sorting.zarr").is_dir());
        assert_eq!(jobs.logs(0), ["starting", "writing m1/rec.ap.kilosort4.sorting.zarr"], "paths relative to the root");
        assert!(jobs.start(&ctl).is_err(), "the output exists now: never overwritten");

        // A job that runs until stopped
        let blocking = Arc::new(Jobs::new(Arc::new(FakeRunner { block: true }), DataRoot::new(&root).unwrap(), None));
        let mut other = ok("kilosort4");
        other.output = "m1/second.sorting.zarr".into();
        blocking.configure(&ctl, other.clone()).unwrap();
        blocking.start(&ctl).unwrap();
        assert!(blocking.configure(&ctl, other).is_err(), "no configuring while running");
        assert!(blocking.stop(&reader).is_err());
        blocking.stop(&ctl).unwrap();
        wait_until(&blocking, wire::DeviceState::Stopped);
        assert_eq!(blocking.status().message, "stopped");

        let lines = std::fs::read_to_string(&audit).unwrap();
        assert!(lines.lines().count() >= 9, "every control request is audited:\n{lines}");
        assert!(lines.contains("\"client\":\"alice\"") && lines.contains("\"ok\":false"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(std::fs::metadata(&audit).unwrap().permissions().mode() & 0o777, 0o600);
        }
        let _ = std::fs::remove_dir_all(&base);
    }
}
