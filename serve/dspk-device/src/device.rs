//! [`LiveDevice`]: what runs on a device. Its sources (each with a raw tap); a chain checked for
//! live input, run on one source's blocks; a thread per source that acquires, keeps the recent data
//! in a ring, records it (unless a preview), and hands every tap's blocks to the clients subscribed.
//! A lost connection never stops it.

use std::collections::{HashMap, VecDeque};
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use dspk_core::{ComputeTarget, DspError, DspResult, MemoryOrder, RecordingInfo, RecordingSource, SampleFormat};
use dspk_io::live::{LiveBlock, LiveEvent, LiveLayout, LiveRing, LiveSource, LiveZarrWriter, SessionLog, RAW_ZARR};
use dspk_io::ZarrOptions;
use dspk_pipeline::live::{chain_of_ops, check_live, with_parameter, LiveChain, LiveRunner, LiveTapKind, Refusal, TapOutput};
use dspk_stream::live::{wire, Caller, ControlOutcome, Device, Director, Lifecycle, LiveError, LiveResult, Subscription};
use dspk_stream::protocol::{header_from_info, wire as session};
use dspk_stream::transport::{Authority, Permission};
use serde_json::json;
use tokio::sync::broadcast;

use crate::config::{DeviceConfig, SourceConfig};
use crate::sources::open_source;
use crate::taps::{TapData, TapHub, ViewProcessor};

/// Log entries a device keeps in memory.
const LOG_KEPT: usize = 2_000;
/// Device events queued per client.
const EVENT_QUEUE: usize = 1_024;
/// How long a run thread waits for a block before checking the stop flag and the source's health.
const SOURCE_WAIT: Duration = Duration::from_millis(100);
/// A source silent for this long is reported disconnected.
const SILENT: Duration = Duration::from_secs(1);
/// Status pushed to clients at least this often while running.
const STATUS_EVERY: Duration = Duration::from_secs(1);
/// A heartbeat line in the session log this often.
const HEARTBEAT_EVERY: Duration = Duration::from_secs(10);
/// Block latencies kept for the median.
const LATENCIES_KEPT: usize = 512;
/// The window a source's rate is measured over.
const RATE_WINDOW: Duration = Duration::from_secs(2);
/// A client's clock estimate is kept in the session log at most this often.
const CLOCK_LOG_EVERY: Duration = Duration::from_secs(1);
/// Seconds of a derived tap kept in its ring (range reads, views of the last seconds).
const DERIVED_RING_SECONDS: f64 = 30.0;
/// Before a past range is viewed through live ops, the chain runs on this much signal before it, so
/// its filters have settled (s).
const VIEW_WARMUP_SECONDS: f64 = 1.0;
/// Most values one range read or view reads.
const MAX_READ_VALUES: u64 = 8 << 20;

/// A source the device has, and what it delivers.
struct SourceSlot {
    config: SourceConfig,
    layout: LiveLayout,
}

/// One source during a run.
struct SourceRun {
    layout: LiveLayout,
    ring: LiveRing,
    status: Mutex<wire::SourceStatus>,
    /// Its raw data on disk (a recording), relative to the device's data folder.
    raw_path: Option<PathBuf>,
    /// Min / max columns from the run's start: zoomed-out views of any length, at once.
    overview: Mutex<dspk_io::live::overview::LiveOverview>,
}

/// What a run shares with the device while it runs (and after, for range reads).
struct RunShared {
    run: Mutex<wire::LiveRun>,
    sources: Vec<SourceRun>,
    health: Mutex<wire::DeviceHealth>,
    /// The session log of a recording.
    log: Mutex<Option<SessionLog>>,
    /// The recording's folder, absolute (`None`: a preview).
    folder: Option<PathBuf>,
    /// Signal taps of the chain: their last seconds.
    derived: Mutex<HashMap<String, Arc<LiveRing>>>,
    /// A chain with a parameter changed, for the run thread to take at its next block.
    pending_chain: Mutex<Option<LiveChain>>,
    /// The chain running now.
    chain: Mutex<LiveChain>,
}

struct Run {
    /// The GPU held for the live chain (`None` on the CPU runtime): no sort starts beside it in
    /// this process (`dspk_core::lease`).
    _gpu: Option<dspk_core::lease::Lease>,
    stop: Arc<AtomicBool>,
    /// Who asked it to stop (for the log).
    stopped_by: Arc<Mutex<String>>,
    threads: Vec<JoinHandle<()>>,
    shared: Arc<RunShared>,
}

struct Inner {
    lifecycle: Lifecycle,
    director: Director,
    chain: Option<LiveChain>,
    run: Option<Run>,
    /// The last run's shared state (kept for range reads after a stop).
    last: Option<Arc<RunShared>>,
    /// Connections per client name.
    connected: HashMap<String, usize>,
}

pub struct LiveDevice {
    config: DeviceConfig,
    id: String,
    sources: Vec<SourceSlot>,
    target: ComputeTarget,
    events: broadcast::Sender<wire::DeviceEvent>,
    taps: Arc<TapHub>,
    log: Arc<Mutex<VecDeque<wire::LogEntry>>>,
    inner: Mutex<Inner>,
    /// Set when a run thread ends by itself (a source ended, the chain failed): the reason.
    ended: Arc<Mutex<Option<String>>>,
    /// The device's authority (who may connect): set when it serves, for pairing codes.
    authority: OnceLock<Arc<Authority>>,
    /// When each client's clock estimate was last logged.
    clock_logged: Mutex<HashMap<String, Instant>>,
    /// The newest view asked for per (client, plot): older ones still being made are dropped.
    views: Mutex<HashMap<(String, u64), u64>>,
    /// This device's own `Arc`, weakly.
    me: std::sync::Weak<LiveDevice>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn header_of(layout: &LiveLayout) -> session::Header {
    let mut info = RecordingInfo::new(layout.name.clone(), layout.channel_count(), 0, layout.sample_rate, layout.format, MemoryOrder::ChannelMajor);
    info.channels = layout.channels.clone();
    header_from_info(&info)
}

/// Where a source's raw data goes in a recording's folder: `raw.zarr` with one source, else
/// `raw-<source>.zarr`.
pub fn raw_store_name(sources: usize, name: &str) -> String {
    if sources == 1 { RAW_ZARR.to_string() } else { format!("raw-{}.zarr", crate::recordings::safe_label(name)) }
}

/// A viewer's ops on a live view: a chain of its own, run on the tap's values.
struct OpsView {
    runner: LiveRunner,
    /// The tap's channels it reads, in order.
    channels: Vec<usize>,
}

impl ViewProcessor for OpsView {
    fn process(&mut self, values: &[f32], channels: usize, samples: usize, first_sample: u64) -> Option<(u64, usize, usize, Vec<f32>)> {
        let mut stored = Vec::with_capacity(self.channels.len() * samples * 4);
        for &c in &self.channels {
            if c >= channels {
                return None;
            }
            for v in &values[c * samples..(c + 1) * samples] {
                stored.extend_from_slice(&v.to_le_bytes());
            }
        }
        let outputs = self.runner.process(&stored, samples, first_sample).ok()?;
        outputs.into_iter().find_map(|o| match o {
            TapOutput::Samples { first_sample, samples, channels, values, .. } => Some((first_sample, samples, channels, values)),
            TapOutput::Events { .. } => None,
        })
    }
}

impl LiveDevice {
    /// Opens the device: its id, its sources' layouts (each opened once to learn it), the compute
    /// target, recordings a crash left open.
    pub fn open(config: DeviceConfig) -> DspResult<Arc<Self>> {
        config.check()?;
        std::fs::create_dir_all(&config.data)?;
        let id = device_id(&config.data)?;
        let mut sources = Vec::new();
        for source in &config.sources {
            let layout = match source.sensor_layout() {
                Some(layout) => layout?,
                None => open_source(source)?.layout().clone(),
            };
            sources.push(SourceSlot { config: source.clone(), layout });
        }
        let target = match &config.runtime {
            Some(name) => ComputeTarget::parse(name).map_err(|e| DspError::InvalidConfig(e.to_string()))?,
            None => ComputeTarget::from_env().map_err(|e| DspError::InvalidConfig(e.to_string()))?,
        };
        let (events, _) = broadcast::channel(EVENT_QUEUE);
        let device = Arc::new_cyclic(|me| Self {
            config,
            id,
            sources,
            target,
            events,
            taps: Arc::new(TapHub::default()),
            log: Arc::default(),
            inner: Mutex::new(Inner { lifecycle: Lifecycle::default(), director: Director::default(), chain: None, run: None, last: None, connected: HashMap::new() }),
            ended: Arc::default(),
            authority: OnceLock::new(),
            clock_logged: Mutex::default(),
            views: Mutex::default(),
            me: me.clone(),
        });
        for folder in crate::recordings::recover(&device.config.recordings_dir()) {
            device.log(wire::LogLevel::Warn, "recorder", format!("{} ended without a stop (a crash?): marked recovered", folder.display()));
        }
        device.taps.set_taps(device.raw_taps());
        lock(&device.inner).lifecycle.ready();
        for s in &device.sources {
            device.log(wire::LogLevel::Info, "device", format!("source {} ({}): {} channels at {} Hz, {}", s.layout.name, s.config.kind(), s.layout.channel_count(), s.layout.sample_rate.rate_hz(), s.layout.format.name()));
        }
        device.log(wire::LogLevel::Info, "device", format!("ready; chains compute on {}", device.target.name()));
        Ok(device)
    }

    pub fn config(&self) -> &DeviceConfig {
        &self.config
    }

    /// The device's authority, so it can make pairing codes (set by [`crate::serve`]).
    pub fn set_authority(&self, authority: Arc<Authority>) {
        let _ = self.authority.set(authority);
    }

    fn source_index(&self, name: &str) -> Option<usize> {
        self.sources.iter().position(|s| s.layout.name == name)
    }

    fn raw_tap_of(layout: &LiveLayout) -> wire::TapInfo {
        let h = header_of(layout);
        wire::TapInfo {
            name: layout.name.clone(),
            endpoint: String::new(),
            kind: wire::TapKind::Samples as i32,
            direction: wire::TapDirection::Out as i32,
            node: String::new(),
            channels: h.channels,
            rate_hz: h.sample_rate_hz,
            format: h.format,
            waveform_samples: 0,
            tensor_shape: Vec::new(),
        }
    }

    fn raw_taps(&self) -> Vec<wire::TapInfo> {
        self.sources.iter().map(|s| Self::raw_tap_of(&s.layout)).collect()
    }

    /// The source a chain reads.
    fn chain_source(&self, chain: &LiveChain) -> usize {
        chain.source.as_deref().and_then(|n| self.source_index(n)).unwrap_or(0)
    }

    /// The taps of `chain`, besides the raw ones.
    fn chain_taps(&self, chain: &LiveChain) -> Vec<wire::TapInfo> {
        let raw = Self::raw_tap_of(&self.sources[self.chain_source(chain)].layout);
        // The chain's channels: those it reads, in its order (reorders after processing permute them)
        let mut channels: Vec<session::Channel> = chain.input_channels.iter().map(|&c| raw.channels[c].clone()).collect();
        for ch in &mut channels {
            (ch.gain, ch.offset) = (1.0, 0.0);
        }
        let mut by_node: HashMap<String, Vec<session::Channel>> = HashMap::new();
        for step in &chain.steps {
            if let Some(dspk_ops::PipelineStage::ReorderChannels { order }) = &step.stage {
                channels = order.iter().map(|&i| channels[i].clone()).collect();
            }
            by_node.insert(step.node.clone(), channels.clone());
        }
        chain
            .taps
            .iter()
            .map(|t| {
                let events = t.kind == LiveTapKind::Events;
                wire::TapInfo {
                    name: t.name.clone(),
                    endpoint: String::new(),
                    kind: if events { wire::TapKind::Events } else { wire::TapKind::Samples } as i32,
                    direction: wire::TapDirection::Out as i32,
                    node: t.node.clone(),
                    channels: by_node.get(&t.node).cloned().unwrap_or_else(|| channels.clone()),
                    rate_hz: raw.rate_hz,
                    format: session::SampleFormat::F32 as i32,
                    waveform_samples: 0,
                    tensor_shape: Vec::new(),
                }
            })
            .collect()
    }

    /// Checks a chain for this device: the live rule, and a source it has.
    fn check_chain(&self, text: &str) -> Result<LiveChain, Vec<Refusal>> {
        // The channel count is the source's: find the source first
        let named = dspk_pipeline::live::parse_live_spec(text).map(|(spec, _, settings)| {
            settings.source.or_else(|| spec.nodes.values().find_map(|n| if let dspk_pipeline::Node::Read { source, .. } = n { source.clone() } else { None }))
        });
        let index = match named {
            Ok(Some(n)) => self.source_index(&n).ok_or_else(|| {
                let have: Vec<&str> = self.sources.iter().map(|s| s.layout.name.as_str()).collect();
                vec![Refusal { node: String::new(), op: "read".into(), reason: format!("source {n}: this device's sources are {}", have.join(", ")) }]
            })?,
            Ok(None) => 0,
            Err(e) => return Err(vec![Refusal { node: String::new(), op: String::new(), reason: e }]),
        };
        check_live(text, self.sources[index].layout.channel_count())
    }

    /// Adds a log entry: kept in memory, sent to clients, and written to a recording's session log
    /// from `warn` up.
    pub fn log(&self, level: wire::LogLevel, source: &str, message: impl Into<String>) {
        let entry = wire::LogEntry { unix_ns: dspk_stream::live::unix_now_ns(), level: level as i32, source: source.into(), message: message.into() };
        match level {
            wire::LogLevel::Error => tracing::error!("{source}: {}", entry.message),
            wire::LogLevel::Warn => tracing::warn!("{source}: {}", entry.message),
            _ => tracing::info!("{source}: {}", entry.message),
        }
        {
            let mut log = lock(&self.log);
            if log.len() == LOG_KEPT {
                log.pop_front();
            }
            log.push_back(entry.clone());
        }
        if level as i32 >= wire::LogLevel::Warn as i32 {
            self.session_log("log", json!({"level": level.as_str_name(), "source": source, "message": entry.message}));
        }
        let _ = self.events.send(wire::DeviceEvent { kind: Some(wire::device_event::Kind::Log(entry)) });
    }

    /// A line in the running recording's session log, if one is recording.
    fn session_log(&self, kind: &str, fields: serde_json::Value) {
        let shared = lock(&self.inner).run.as_ref().map(|r| r.shared.clone());
        if let Some(shared) = shared
            && let Some(log) = lock(&shared.log).as_mut()
            && let Err(e) = log.log(kind, fields)
        {
            tracing::warn!("session log: {e}");
        }
    }

    fn push_status(&self) {
        let _ = self.events.send(wire::DeviceEvent { kind: Some(wire::device_event::Kind::Status(self.status())) });
    }

    /// Ends a run whose thread ended by itself (a source ended, the chain failed).
    fn reap(&self) {
        let reason = lock(&self.ended).take();
        if let Some(reason) = reason {
            let mut inner = lock(&self.inner);
            if let Some(run) = inner.run.take() {
                run.stop.store(true, Ordering::Relaxed);
                for t in run.threads {
                    let _ = t.join();
                }
                inner.last = Some(run.shared);
                let _ = inner.lifecycle.stop(reason.clone());
            }
            drop(inner);
            self.log(wire::LogLevel::Warn, "device", format!("the run ended: {reason}"));
            self.push_status();
        }
    }

    fn require_director(inner: &mut Inner, caller: &Caller) -> LiveResult<()> {
        let now = Instant::now();
        if inner.director.is_director(&caller.name, now) {
            return Ok(());
        }
        Err(LiveError::not_director(inner.director.holder(now).unwrap_or("")))
    }

    fn zarr_options(&self, rate_hz: f64) -> ZarrOptions {
        let chunk = ((self.config.recording.chunk_seconds * rate_hz).round() as usize).max(1);
        ZarrOptions { chunk_samples: chunk, compression: self.config.recording.zarr_compression(), ..Default::default() }
    }

    fn runner_for(&self, chain: &LiveChain) -> DspResult<LiveRunner> {
        let layout = &self.sources[self.chain_source(chain)].layout;
        let mut runner = LiveRunner::new(
            chain.clone(),
            self.target,
            layout.sample_rate.rate_hz(),
            layout.format,
            layout.channels.iter().map(|c| c.gain).collect(),
            layout.channels.iter().map(|c| c.offset).collect(),
        )?;
        // Kernels compiled and buffers made now, not on the first real block; the buffers kept
        // apart from the pools (persistent), so nothing releases them during the session
        runner.warm_persistent()?;
        Ok(runner)
    }

    fn start_run(self: &Arc<Self>, caller: &Caller, request: &wire::Start) -> LiveResult<()> {
        let mut inner = lock(&self.inner);
        Self::require_director(&mut inner, caller)?;
        let chain = inner.chain.clone().ok_or_else(|| LiveError::wrong_state("no chain is configured: configure one first"))?;
        inner.lifecycle.start()?;
        let fail = |inner: &mut Inner, e: String| {
            let _ = inner.lifecycle.stop(format!("could not start: {e}"));
            LiveError::internal(e)
        };
        let mut sources = Vec::new();
        for slot in &self.sources {
            match open_source(&slot.config) {
                Ok(s) => sources.push(s),
                Err(e) => return Err(fail(&mut inner, format!("source {}: {e}", slot.layout.name))),
            }
        }
        let chain_source = self.chain_source(&chain);
        // The GPU for the live chain while it runs (a GPU runtime): refused while a batch job
        // (a sort) holds it in this process
        let gpu = match self.target {
            ComputeTarget::Cpu => None,
            _ => match dspk_core::lease::live() {
                Ok(l) => Some(l),
                Err(e) => return Err(fail(&mut inner, e)),
            },
        };
        let runner = match self.runner_for(&chain) {
            Ok(r) => r,
            Err(e) => return Err(fail(&mut inner, e.to_string())),
        };
        let started = dspk_stream::live::unix_now_ns();

        // A recording: its folder, each source's raw store, the session log's first line
        let n = self.sources.len();
        let mut recorders: Vec<Option<LiveZarrWriter>> = (0..n).map(|_| None).collect();
        let (folder, log) = if request.record {
            let made = (|| -> DspResult<(PathBuf, SessionLog)> {
                let folder = crate::recordings::new_folder(&self.config.recordings_dir(), &request.label)?;
                for (i, slot) in self.sources.iter().enumerate() {
                    let path = folder.join(raw_store_name(n, &slot.layout.name));
                    recorders[i] = Some(LiveZarrWriter::create(&path, &slot.layout, &self.zarr_options(slot.layout.sample_rate.rate_hz()))?);
                }
                let mut log = SessionLog::create(&folder)?;
                let sources: Vec<serde_json::Value> = self
                    .sources
                    .iter()
                    .map(|s| json!({"name": s.layout.name, "kind": s.config.kind(), "channels": s.layout.channels, "sample_rate_hz": s.layout.sample_rate.rate_hz(), "format": s.layout.format, "raw": raw_store_name(n, &s.layout.name)}))
                    .collect();
                log.log(
                    "start",
                    json!({
                        "device": {"id": self.id, "name": self.config.name, "dspk": env!("CARGO_PKG_VERSION"), "runtime": self.target.name()},
                        "director": caller.name,
                        "label": request.label,
                        "notes": request.notes,
                        "config": self.config,
                        "chain": serde_json::from_str::<serde_json::Value>(&chain.spec_json).unwrap_or_default(),
                        "chain_source": self.sources[chain_source].layout.name,
                        "block_ms": chain.block_ms,
                        "delay_samples": chain.delay_samples,
                        "taps": chain.taps,
                        "sources": sources,
                    }),
                )?;
                Ok((folder, log))
            })();
            match made {
                Ok((f, l)) => (Some(f), Some(l)),
                Err(e) => return Err(fail(&mut inner, e.to_string())),
            }
        } else {
            (None, None)
        };
        let rel = |p: &PathBuf| p.strip_prefix(&self.config.data).unwrap_or(p).to_path_buf();
        let records = folder.as_ref().map(|p| rel(p).display().to_string()).unwrap_or_default();
        let shared = Arc::new(RunShared {
            run: Mutex::new(wire::LiveRun {
                recording: request.record,
                path: records,
                label: request.label.clone(),
                notes: request.notes.clone(),
                started_unix_ns: started,
                started_source_time_ns: 0,
                head_sample: 0,
                lost_samples: 0,
                running: true,
            }),
            sources: self
                .sources
                .iter()
                .map(|slot| {
                    let chunk = self.zarr_options(slot.layout.sample_rate.rate_hz()).chunk_samples as u64;
                    let capacity = LiveRing::capacity_for(&slot.layout, self.config.ring.seconds, self.config.ring.max_mb << 20, 2 * chunk);
                    SourceRun {
                        layout: slot.layout.clone(),
                        ring: LiveRing::new(slot.layout.clone(), capacity),
                        status: Mutex::new(wire::SourceStatus { name: slot.layout.name.clone(), ..Default::default() }),
                        raw_path: folder.as_ref().map(|f| rel(&f.join(raw_store_name(n, &slot.layout.name)))),
                        overview: Mutex::new(dspk_io::live::overview::LiveOverview::new(slot.layout.channel_count())),
                    }
                })
                .collect(),
            health: Mutex::new(wire::DeviceHealth::default()),
            log: Mutex::new(log),
            folder,
            derived: Mutex::default(),
            pending_chain: Mutex::new(None),
            chain: Mutex::new(chain),
        });
        let stop = Arc::new(AtomicBool::new(false));
        let stopped_by = Arc::new(Mutex::new(String::new()));
        let mut threads = Vec::new();
        let mut runner = Some(runner);
        for (i, (source, recorder)) in sources.into_iter().zip(recorders).enumerate() {
            let (device, shared, stop, stopped_by) = (self.clone(), shared.clone(), stop.clone(), stopped_by.clone());
            let runner = if i == chain_source { runner.take() } else { None };
            let thread = std::thread::Builder::new()
                .name(format!("dspk-device-{}", self.sources[i].layout.name))
                .spawn(move || device.run_loop(i, source, runner, recorder, shared, stop, stopped_by))
                .map_err(|e| LiveError::internal(e.to_string()))?;
            threads.push(thread);
        }
        inner.run = Some(Run { _gpu: gpu, stop, stopped_by, threads, shared });
        Ok(())
    }

    /// One source's run thread; the chain's source also runs the chain.
    #[allow(clippy::too_many_arguments)]
    fn run_loop(
        self: Arc<Self>,
        index: usize,
        mut source: Box<dyn LiveSource>,
        mut runner: Option<LiveRunner>,
        mut recorder: Option<LiveZarrWriter>,
        shared: Arc<RunShared>,
        stop: Arc<AtomicBool>,
        stopped_by: Arc<Mutex<String>>,
    ) {
        let layout = self.sources[index].layout.clone();
        let is_chain = runner.is_some();
        let gains: Vec<f32> = layout.channels.iter().map(|c| c.gain).collect();
        let offsets: Vec<f32> = layout.channels.iter().map(|c| c.offset).collect();
        let mut state = ChainState::default();
        if let Some(r) = &runner {
            state.reset(&self, r.chain(), &shared);
        }
        let mut latencies: VecDeque<u64> = VecDeque::with_capacity(LATENCIES_KEPT);
        let mut arrivals: VecDeque<(Instant, u64)> = VecDeque::new();
        let (mut last_data, mut last_status, mut last_heartbeat) = (None::<Instant>, Instant::now(), Instant::now());
        let started = Instant::now();
        let mut ended: Option<String> = None;

        while !stop.load(Ordering::Relaxed) {
            // A changed parameter: the new chain from the next block on (its filters start again)
            if let Some(r) = runner.as_mut() {
                let pending = lock(&shared.pending_chain).take();
                if let Some(chain) = pending {
                    match self.runner_for(&chain) {
                        Ok(new) => {
                            *r = new;
                            state.reset(&self, &chain, &shared);
                            *lock(&shared.chain) = chain;
                        }
                        Err(e) => self.log(wire::LogLevel::Error, "chain", format!("the changed chain could not start ({e}); the old one runs on")),
                    }
                }
            }
            match source.next(SOURCE_WAIT) {
                Ok(LiveEvent::Block(block)) => {
                    last_data = Some(Instant::now());
                    arrivals.push_back((Instant::now(), block.samples as u64));
                    if is_chain && lock(&shared.run).head_sample == 0 {
                        lock(&shared.run).started_source_time_ns = block.source_time_ns;
                    }
                    if let Err(e) = self.handle_block(index, &block, &shared, runner.as_mut(), recorder.as_mut(), &mut state, &gains, &offsets) {
                        ended = Some(format!("{}: {e}", if is_chain { "the chain failed" } else { "the recorder failed" }));
                        break;
                    }
                    let latency = dspk_stream::live::unix_now_ns().saturating_sub(block.received_unix_ns) / 1000;
                    if latencies.len() == LATENCIES_KEPT {
                        latencies.pop_front();
                    }
                    latencies.push_back(latency);
                }
                Ok(LiveEvent::Idle) => {}
                Ok(LiveEvent::Ended) => {
                    ended = Some(format!("source {} ended", layout.name));
                    break;
                }
                Err(e) => {
                    ended = Some(format!("source {} failed: {e}", layout.name));
                    break;
                }
            }
            // This source's health, every pass
            {
                let now = Instant::now();
                while arrivals.front().is_some_and(|(t, _)| now.duration_since(*t) > RATE_WINDOW) {
                    arrivals.pop_front();
                }
                let window = now.duration_since(started).min(RATE_WINDOW).as_secs_f64();
                let rate = if window > 0.0 { arrivals.iter().map(|(_, n)| *n).sum::<u64>() as f64 / window } else { 0.0 };
                let counters = source.counters();
                let silent = last_data.map(|t| t.elapsed());
                let connected = silent.is_some_and(|d| d <= SILENT);
                let was = lock(&shared.sources[index].status).connected;
                if was && !connected {
                    self.log(wire::LogLevel::Warn, &format!("source:{}", layout.name), format!("no data for {:.1} s", silent.unwrap_or_default().as_secs_f64()));
                }
                let mut s = lock(&shared.sources[index].status);
                s.connected = connected;
                s.measured_rate_hz = rate;
                s.checksum_errors = counters.checksum_errors;
                s.lost_samples = counters.lost_samples;
                s.last_data_unix_ns = match silent {
                    Some(d) => dspk_stream::live::unix_now_ns().saturating_sub(d.as_nanos() as u64),
                    None => 0,
                };
            }
            if !is_chain {
                continue;
            }
            {
                let mut h = lock(&shared.health);
                let mut sorted: Vec<u64> = latencies.iter().copied().collect();
                sorted.sort_unstable();
                h.block_latency_median_us = sorted.get(sorted.len() / 2).copied().unwrap_or(0) as u32;
                h.block_latency_max_us = sorted.last().copied().unwrap_or(0) as u32;
                h.ring_use = shared.sources[index].ring.stats().use_fraction();
                h.frames_dropped = self.taps.dropped();
            }
            if last_status.elapsed() >= STATUS_EVERY {
                last_status = Instant::now();
                self.push_status();
            }
            if last_heartbeat.elapsed() >= HEARTBEAT_EVERY {
                last_heartbeat = Instant::now();
                let run = lock(&shared.run).clone();
                let health = lock(&shared.health).clone();
                let sources: Vec<serde_json::Value> = shared
                    .sources
                    .iter()
                    .map(|s| {
                        let s = lock(&s.status);
                        json!({"name": s.name, "connected": s.connected, "rate_hz": s.measured_rate_hz, "lost": s.lost_samples, "checksum_errors": s.checksum_errors})
                    })
                    .collect();
                let stored = recorder.as_ref().map(|r| r.stored_samples());
                if let Some(log) = lock(&shared.log).as_mut() {
                    let _ = log.log(
                        "heartbeat",
                        json!({"head_sample": run.head_sample, "samples_stored": stored, "lost_samples": run.lost_samples, "ring_use": health.ring_use,
                               "block_latency_median_us": health.block_latency_median_us, "block_latency_max_us": health.block_latency_max_us,
                               "frames_dropped": health.frames_dropped, "sources": sources, "events": state.events_written}),
                    );
                }
            }
        }

        // End: the last chunks stored, the stop line (written by the chain's thread)
        let stored = recorder.map(|r| r.finish());
        for (_, writer) in state.derived.drain() {
            let _ = writer.finish();
        }
        for (_, mut file) in state.events.drain() {
            let _ = file.flush();
        }
        if let Some(Err(e)) = &stored {
            self.log(wire::LogLevel::Error, "recorder", format!("{}: the last chunk: {e}", layout.name));
        }
        if let Some(why) = &ended {
            // The other sources stop too: a run is whole
            stop.store(true, Ordering::Relaxed);
            *lock(&self.ended) = Some(why.clone());
        }
        if is_chain {
            lock(&shared.run).running = false;
            let run = lock(&shared.run).clone();
            let by = lock(&stopped_by).clone();
            let reason = ended.clone().unwrap_or_else(|| format!("stopped by {by}"));
            if let Some(log) = lock(&shared.log).as_mut() {
                let stored = match &stored {
                    Some(Ok(n)) => json!(n),
                    Some(Err(e)) => json!(format!("error: {e}")),
                    None => json!(null),
                };
                let _ = log.log("stop", json!({"reason": reason, "head_sample": run.head_sample, "samples_stored": stored, "lost_samples": run.lost_samples, "events": state.events_written}));
            }
        }
        if ended.is_some() {
            // Tell the device from another thread: reaping joins this one
            let device = self.clone();
            std::thread::spawn(move || device.reap());
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_block(
        &self,
        index: usize,
        block: &LiveBlock,
        shared: &RunShared,
        runner: Option<&mut LiveRunner>,
        recorder: Option<&mut LiveZarrWriter>,
        state: &mut ChainState,
        gains: &[f32],
        offsets: &[f32],
    ) -> DspResult<()> {
        let layout = &self.sources[index].layout;
        block.check(layout)?;
        let is_chain = runner.is_some();
        if block.lost_before > 0 {
            if let Some(log) = lock(&shared.log).as_mut() {
                let _ = log.log("gap", json!({"sample": block.first_sample, "lost_samples": block.lost_before, "source": layout.name}));
            }
            if is_chain {
                lock(&shared.run).lost_samples += block.lost_before;
            }
        }
        let block = Arc::new(block.clone());
        shared.sources[index].ring.push(block.clone());
        // Its min / max columns, for views zoomed out over the run
        let all: Vec<usize> = (0..layout.channel_count()).collect();
        if let Some(values) = shared.sources[index].ring.read(&all, block.first_sample..block.end_sample()) {
            lock(&shared.sources[index].overview).push(block.first_sample, &values, block.samples);
        }
        if let Some(rec) = recorder {
            rec.push(&block)?;
        }
        self.taps.publish(
            &layout.name,
            layout.channel_count(),
            block.first_sample,
            block.samples,
            block.source_time_ns,
            block.lost_before,
            TapData::Stored { bytes: &block.data, format: layout.format, gains, offsets },
        );
        let Some(runner) = runner else { return Ok(()) };
        let recording = lock(&shared.run).recording;
        let outputs = runner.process(&block.data, block.samples, block.first_sample)?;
        for output in outputs {
            match output {
                TapOutput::Samples { tap, first_sample, samples, channels, values } => {
                    let name = state.names[tap].clone();
                    self.taps.publish(&name, channels, first_sample, samples, block.source_time_ns, block.lost_before, TapData::Values(&values));
                    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let derived = LiveBlock { first_sample, samples, source_time_ns: block.source_time_ns, received_unix_ns: block.received_unix_ns, lost_before: 0, data: bytes };
                    if let Some(ring) = state.rings.get(&tap) {
                        ring.push(Arc::new(derived.clone()));
                    }
                    if recording && runner.chain().taps[tap].record {
                        self.record_derived(state, tap, shared, &derived)?;
                    }
                }
                TapOutput::Events { tap, events } => {
                    if events.is_empty() {
                        continue;
                    }
                    let pairs: Vec<(u64, u32)> = events.iter().map(|e| (e.onset, e.channel as u32)).collect();
                    let name = state.names[tap].clone();
                    let channels = state.taps[tap].channels.len();
                    self.taps.publish(&name, channels, block.first_sample, block.samples, block.source_time_ns, 0, TapData::Events(&pairs));
                    if recording
                        && runner.chain().taps[tap].record
                        && let Some(folder) = &shared.folder
                    {
                        // Events in a file of their own, one JSON object per line
                        let file = match state.events.entry(tap) {
                            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                            std::collections::hash_map::Entry::Vacant(v) => {
                                let path = folder.join(format!("{}.events.jsonl", crate::recordings::safe_label(&name)));
                                v.insert(std::io::BufWriter::new(std::fs::OpenOptions::new().create(true).append(true).open(&path)?))
                            }
                        };
                        for e in &events {
                            let line = json!({"channel": e.channel, "onset": e.onset, "offset": e.offset, "peak_sample": e.peak_sample, "peak_value": e.peak_value});
                            serde_json::to_writer(&mut *file, &line).map_err(|e| DspError::Io(e.to_string()))?;
                            file.write_all(b"\n")?;
                        }
                        file.flush()?;
                        state.events_written += events.len() as u64;
                    }
                }
            }
        }
        lock(&shared.run).head_sample = block.end_sample();
        Ok(())
    }

    /// A derived tap recorded next to the raw data (`<tap>.zarr`, float32).
    fn record_derived(&self, state: &mut ChainState, tap: usize, shared: &RunShared, block: &LiveBlock) -> DspResult<()> {
        if !state.derived.contains_key(&tap) {
            let Some(folder) = &shared.folder else { return Ok(()) };
            let layout = state.layouts[tap].clone();
            let name = crate::recordings::safe_label(&layout.name);
            let writer = LiveZarrWriter::create(&folder.join(format!("{name}.zarr")), &layout, &self.zarr_options(layout.sample_rate.rate_hz()))?;
            state.derived.insert(tap, writer);
        }
        state.derived.get_mut(&tap).expect("made above").push(block)
    }

    fn current_shared(&self) -> Option<Arc<RunShared>> {
        let inner = lock(&self.inner);
        inner.run.as_ref().map(|r| r.shared.clone()).or_else(|| inner.last.clone())
    }

    /// Values of `tap` over `range`, scaled: a source's raw data from its ring or its recording, a
    /// signal tap of the chain from its ring or its recorded store.
    fn read_tap(&self, tap: &str, channels: &[usize], range: std::ops::Range<u64>) -> LiveResult<Vec<f32>> {
        let shared = self.current_shared().ok_or_else(|| LiveError::wrong_state("no run yet"))?;
        let run = lock(&shared.run).clone();
        let n = (range.end - range.start) as usize;
        let from_disk = |rel: &std::path::Path| -> LiveResult<Vec<f32>> {
            let rec = dspk_io::ZarrRecording::open(&self.config.data.join(rel)).map_err(|e| LiveError::internal(e.to_string()))?;
            let mut out = vec![0.0f32; channels.len() * n];
            rec.read(channels, range.clone(), &mut out).map_err(|e| LiveError::invalid(e.to_string()))?;
            Ok(out)
        };
        if let Some(i) = self.source_index(tap) {
            let src = &shared.sources[i];
            check_channels(channels, src.layout.channel_count())?;
            if let Some(values) = src.ring.read(channels, range.clone()) {
                return Ok(values);
            }
            let raw = src.raw_path.clone().ok_or_else(|| LiveError::invalid("a preview keeps only its last seconds (the ring): this range is gone"))?;
            return from_disk(&raw);
        }
        let ring = lock(&shared.derived).get(tap).cloned().ok_or_else(|| LiveError::not_found(format!("no signal tap {tap} in this run")))?;
        check_channels(channels, ring.layout().channel_count())?;
        if let Some(values) = ring.read(channels, range.clone()) {
            return Ok(values);
        }
        let recorded = lock(&shared.chain).taps.iter().any(|t| t.name == tap && t.record);
        if !(run.recording && recorded) {
            return Err(LiveError::invalid(format!("tap {tap} keeps only its last seconds (it is not recorded): this range is gone")));
        }
        from_disk(&PathBuf::from(&run.path).join(format!("{}.zarr", crate::recordings::safe_label(tap))))
    }

    /// Stops a run because the device is shutting down (a signal): the recording ends cleanly.
    pub fn shutdown(&self, why: &str) {
        let run = {
            let mut inner = lock(&self.inner);
            let _ = inner.lifecycle.stop(format!("the device stopped: {why}"));
            inner.run.take()
        };
        if let Some(run) = run {
            *lock(&run.stopped_by) = format!("the device ({why})");
            run.stop.store(true, Ordering::Relaxed);
            for t in run.threads {
                let _ = t.join();
            }
            lock(&self.inner).last = Some(run.shared);
            self.log(wire::LogLevel::Info, "device", format!("the run was stopped: {why}"));
        }
    }

    fn chain_info(&self, chain: &LiveChain) -> wire::ChainInfo {
        let layout = &self.sources[self.chain_source(chain)].layout;
        wire::ChainInfo {
            spec_json: chain.spec_json.clone(),
            block_samples: ((chain.block_ms / 1000.0 * layout.sample_rate.rate_hz()).round() as u32).max(1),
            delay_samples: chain.delay_samples,
            taps: self.raw_taps().into_iter().chain(self.chain_taps(chain)).collect(),
        }
    }

    /// A runner for a viewer's ops over `channels` channels of a tap at `rate`.
    fn ops_runner(&self, ops: &[String], channels: usize, rate: f64) -> LiveResult<LiveRunner> {
        let block_ms = lock(&self.inner).chain.as_ref().map_or(dspk_pipeline::live::DEFAULT_BLOCK_MS, |c| c.block_ms);
        let chain = chain_of_ops(ops, channels, block_ms).map_err(|r| LiveError::invalid(r.iter().map(|r| r.reason.clone()).collect::<Vec<_>>().join("; ")))?;
        LiveRunner::new(chain, self.target, rate, SampleFormat::F32, vec![1.0; channels], vec![0.0; channels]).map_err(|e| LiveError::invalid(e.to_string()))
    }

    /// This device as an `Arc`, for the run threads (the [`Device`] trait takes `&self`; a device
    /// is only made by [`Self::open`], inside an `Arc`).
    fn arc(&self) -> Arc<Self> {
        self.me.upgrade().expect("a LiveDevice lives in the Arc LiveDevice::open made")
    }
}

fn check_channels(channels: &[usize], total: usize) -> LiveResult<()> {
    match channels.iter().find(|&&c| c >= total) {
        Some(bad) => Err(LiveError::invalid(format!("channel {bad} does not exist ({total} channels)"))),
        None => Ok(()),
    }
}

fn rate_of(info: &wire::TapInfo) -> f64 {
    info.rate_hz.as_ref().map_or(0.0, |r| r.numerator as f64 / r.denominator.max(1) as f64)
}

/// What the chain's thread keeps about the chain's taps.
#[derive(Default)]
struct ChainState {
    names: Vec<String>,
    taps: Vec<wire::TapInfo>,
    layouts: Vec<LiveLayout>,
    rings: HashMap<usize, Arc<LiveRing>>,
    derived: HashMap<usize, LiveZarrWriter>,
    events: HashMap<usize, std::io::BufWriter<std::fs::File>>,
    events_written: u64,
}

impl ChainState {
    /// For `chain` (a new one: rings, names and layouts made again; recorded stores stay open).
    fn reset(&mut self, device: &LiveDevice, chain: &LiveChain, shared: &RunShared) {
        let src = &device.sources[device.chain_source(chain)].layout;
        self.names = chain.taps.iter().map(|t| t.name.clone()).collect();
        self.taps = device.chain_taps(chain);
        let unit = src.channels.first().map(|c| c.unit.clone()).unwrap_or_default();
        self.layouts = self
            .taps
            .iter()
            .map(|t| LiveLayout {
                name: t.name.clone(),
                channels: t.channels.iter().map(|c| dspk_core::ChannelInfo { name: c.name.clone(), gain: 1.0, offset: 0.0, unit: unit.clone() }).collect(),
                sample_rate: src.sample_rate,
                format: SampleFormat::F32,
            })
            .collect();
        self.rings = chain
            .taps
            .iter()
            .enumerate()
            .filter(|(_, t)| t.kind == LiveTapKind::Samples)
            .map(|(i, _)| {
                let capacity = LiveRing::capacity_for(&self.layouts[i], DERIVED_RING_SECONDS, device.config.ring.max_mb << 20, 1);
                (i, Arc::new(LiveRing::new(self.layouts[i].clone(), capacity)))
            })
            .collect();
        let mut derived = lock(&shared.derived);
        derived.clear();
        for (i, ring) in &self.rings {
            derived.insert(self.names[*i].clone(), ring.clone());
        }
    }
}

/// The device's id: made once, kept in its data folder.
fn device_id(data: &std::path::Path) -> DspResult<String> {
    let path = data.join("device-id");
    if let Ok(id) = std::fs::read_to_string(&path) {
        let id = id.trim().to_string();
        if !id.is_empty() {
            return Ok(id);
        }
    }
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    h.write_u32(std::process::id());
    let id = format!("{:016x}", h.finish());
    std::fs::write(&path, &id)?;
    Ok(id)
}

/// Min / max columns of channel-major `values` (`rows × n`) in `width` columns.
fn columns(values: &[f32], rows: usize, n: usize, width: usize) -> session::Columns {
    let width = width.min(n).max(1);
    let mut min_max = vec![0.0f32; 2 * rows * width];
    for r in 0..rows {
        let row = &values[r * n..(r + 1) * n];
        for col in 0..width {
            let (a, b) = (col * n / width, ((col + 1) * n / width).max(col * n / width + 1).min(n));
            let at = 2 * (r * width + col);
            min_max[at] = row[a..b].iter().copied().fold(f32::INFINITY, f32::min);
            min_max[at + 1] = row[a..b].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        }
    }
    session::Columns { min_max, complete: true }
}

impl Device for LiveDevice {
    fn info(&self) -> wire::DeviceInfo {
        wire::DeviceInfo {
            id: self.id.clone(),
            name: self.config.name.clone(),
            version: env!("CARGO_PKG_VERSION").into(),
            runtime: self.target.name().into(),
            sources: self
                .sources
                .iter()
                .map(|s| {
                    let h = header_of(&s.layout);
                    wire::SourceInfo { name: s.layout.name.clone(), kind: s.config.kind().into(), channels: h.channels, sample_rate_hz: h.sample_rate_hz, format: h.format }
                })
                .collect(),
            status: Some(self.status()),
        }
    }

    fn status(&self) -> wire::DeviceStatus {
        let mut inner = lock(&self.inner);
        let now = Instant::now();
        let director = inner.director.holder(now).unwrap_or("").to_string();
        let kept = inner.director.kept_until().map_or(0, |until| {
            let left = until.saturating_duration_since(now);
            dspk_stream::live::unix_now_ns() + left.as_nanos() as u64
        });
        let shared = inner.run.as_ref().map(|r| r.shared.clone()).or_else(|| inner.last.clone());
        let chain = inner.chain.as_ref().map(|c| {
            let mut info = self.chain_info(c);
            info.taps = self.taps.taps();
            info
        });
        wire::DeviceStatus {
            state: inner.lifecycle.state() as i32,
            message: inner.lifecycle.message().to_string(),
            director,
            director_kept_until_unix_ns: kept,
            chain,
            run: shared.as_ref().map(|s| lock(&s.run).clone()),
            sources: shared.as_ref().map(|s| s.sources.iter().map(|src| lock(&src.status).clone()).collect()).unwrap_or_default(),
            health: shared.as_ref().map(|s| lock(&s.health).clone()),
        }
    }

    fn events(&self) -> broadcast::Receiver<wire::DeviceEvent> {
        self.events.subscribe()
    }

    fn connected(&self, caller: &Caller) {
        {
            let mut inner = lock(&self.inner);
            *inner.connected.entry(caller.name.clone()).or_default() += 1;
            inner.director.connected(&caller.name, Instant::now());
        }
        self.session_log("client", json!({"client": caller.name, "event": "connected"}));
        self.push_status();
    }

    fn disconnected(&self, caller: &Caller) {
        {
            let mut inner = lock(&self.inner);
            let gone = match inner.connected.get_mut(&caller.name) {
                Some(n) if *n > 1 => {
                    *n -= 1;
                    false
                }
                _ => {
                    inner.connected.remove(&caller.name);
                    true
                }
            };
            if gone {
                inner.director.disconnected(&caller.name, Instant::now());
            }
        }
        lock(&self.clock_logged).remove(&caller.name);
        self.session_log("client", json!({"client": caller.name, "event": "disconnected"}));
        self.push_status();
    }

    fn configure(&self, caller: &Caller, request: wire::ConfigureChain) -> LiveResult<wire::ConfigureResult> {
        self.reap();
        let checked = self.check_chain(&request.spec_json);
        let refusals = |r: &Vec<Refusal>| r.iter().map(|r| wire::Refusal { node: r.node.clone(), op: r.op.clone(), reason: r.reason.clone() }).collect::<Vec<_>>();
        if request.dry_run {
            let (refused, chain) = match &checked {
                Ok(chain) => (Vec::new(), Some(self.chain_info(chain))),
                Err(r) => (refusals(r), None),
            };
            return Ok(wire::ConfigureResult { refusals: refused, chain, status: Some(self.status()) });
        }
        {
            let mut inner = lock(&self.inner);
            Self::require_director(&mut inner, caller)?;
            inner.lifecycle.begin_configure()?;
            match &checked {
                Ok(chain) => {
                    self.taps.set_taps(self.raw_taps().into_iter().chain(self.chain_taps(chain)).collect());
                    inner.chain = Some(chain.clone());
                    inner.lifecycle.end_configure(true, format!("chain configured by {}", caller.name));
                }
                Err(r) => inner.lifecycle.end_configure(false, format!("chain refused: {} node(s)", r.len())),
            }
        }
        match &checked {
            Ok(chain) => self.log(
                wire::LogLevel::Info,
                "device",
                format!("{} configured a chain: {} steps, taps {:?}, delay {} samples", caller.name, chain.steps.len(), chain.taps.iter().map(|t| &t.name).collect::<Vec<_>>(), chain.delay_samples),
            ),
            Err(r) => self.log(wire::LogLevel::Info, "device", format!("{} sent a chain that was refused ({} node(s))", caller.name, r.len())),
        }
        self.push_status();
        let (refused, chain) = match &checked {
            Ok(chain) => (Vec::new(), Some(self.chain_info(chain))),
            Err(r) => (refusals(r), None),
        };
        Ok(wire::ConfigureResult { refusals: refused, chain, status: Some(self.status()) })
    }

    fn start(&self, caller: &Caller, request: wire::Start) -> LiveResult<wire::DeviceStatus> {
        self.reap();
        self.arc().start_run(caller, &request)?;
        let what = if request.record { "recording" } else { "a preview" };
        self.log(wire::LogLevel::Info, "device", format!("{} started {what}", caller.name));
        self.push_status();
        Ok(self.status())
    }

    fn stop(&self, caller: &Caller) -> LiveResult<wire::DeviceStatus> {
        self.reap();
        let run = {
            let mut inner = lock(&self.inner);
            Self::require_director(&mut inner, caller)?;
            inner.lifecycle.stop(format!("stopped by {}", caller.name))?;
            inner.run.take()
        };
        if let Some(run) = run {
            *lock(&run.stopped_by) = caller.name.clone();
            run.stop.store(true, Ordering::Relaxed);
            for t in run.threads {
                let _ = t.join();
            }
            lock(&self.inner).last = Some(run.shared);
        }
        self.log(wire::LogLevel::Info, "device", format!("{} stopped the run", caller.name));
        self.push_status();
        Ok(self.status())
    }

    fn set_parameter(&self, caller: &Caller, request: wire::SetParameter) -> LiveResult<wire::DeviceStatus> {
        let (old_chain, shared) = {
            let mut inner = lock(&self.inner);
            Self::require_director(&mut inner, caller)?;
            let chain = inner.chain.clone().ok_or_else(|| LiveError::wrong_state("no chain is configured"))?;
            (chain, inner.run.as_ref().map(|r| r.shared.clone()))
        };
        let text = with_parameter(&old_chain.spec_json, &request.node, &request.parameter, &request.value_json).map_err(LiveError::invalid)?;
        let chain = self.check_chain(&text).map_err(|r| LiveError::invalid(r.iter().map(|r| format!("{}: {}", r.node, r.reason)).collect::<Vec<_>>().join("; ")))?;
        // The shape stays: the same taps, so clients keep their subscriptions
        let shape = |c: &LiveChain| c.taps.iter().map(|t| (t.name.clone(), t.kind, t.channels)).collect::<Vec<_>>();
        if shape(&chain) != shape(&old_chain) {
            return Err(LiveError::invalid("this change alters the chain's taps: configure the new chain (stopped)"));
        }
        let old_value = serde_json::from_str::<serde_json::Value>(&old_chain.spec_json).ok().and_then(|v| v["nodes"][&request.node].get(&request.parameter).cloned());
        lock(&self.inner).chain = Some(chain.clone());
        if let Some(shared) = shared {
            *lock(&shared.pending_chain) = Some(chain);
        }
        self.session_log("parameter", json!({"node": request.node, "parameter": request.parameter, "old": old_value, "new": serde_json::from_str::<serde_json::Value>(&request.value_json).ok(), "by": caller.name}));
        self.log(wire::LogLevel::Info, "device", format!("{} set {}.{} = {} (the chain's filters start again)", caller.name, request.node, request.parameter, request.value_json));
        self.push_status();
        Ok(self.status())
    }

    fn request_control(&self, caller: &Caller) -> LiveResult<wire::DeviceStatus> {
        if !caller.may_control {
            return Err(LiveError::new(wire::ErrorCode::NoPermission, format!("{} may watch only (paired for reading)", caller.name)));
        }
        let outcome = lock(&self.inner).director.request(&caller.name, Instant::now());
        match outcome {
            ControlOutcome::Granted => {
                self.session_log("control", json!({"to": caller.name, "how": "granted"}));
                self.log(wire::LogLevel::Info, "device", format!("{} directs the device", caller.name));
            }
            ControlOutcome::Asked { director } => {
                let _ = self.events.send(wire::DeviceEvent { kind: Some(wire::device_event::Kind::ControlRequested(wire::ControlRequested { by: caller.name.clone() })) });
                self.log(wire::LogLevel::Info, "device", format!("{} asked {director} for control", caller.name));
            }
        }
        self.push_status();
        Ok(self.status())
    }

    fn hand_over(&self, caller: &Caller, to: &str) -> LiveResult<wire::DeviceStatus> {
        {
            let mut inner = lock(&self.inner);
            if !inner.connected.contains_key(to) {
                return Err(LiveError::not_found(format!("{to} is not connected")));
            }
            if !inner.director.hand_over(&caller.name, to, Instant::now()) {
                return Err(LiveError::not_director(inner.director.holder(Instant::now()).unwrap_or("")));
            }
        }
        self.session_log("control", json!({"from": caller.name, "to": to, "how": "handed over"}));
        self.log(wire::LogLevel::Info, "device", format!("{} handed control to {to}", caller.name));
        self.push_status();
        Ok(self.status())
    }

    fn release_control(&self, caller: &Caller) -> LiveResult<wire::DeviceStatus> {
        {
            let mut inner = lock(&self.inner);
            if !inner.director.release(&caller.name, Instant::now()) {
                return Err(LiveError::not_director(inner.director.holder(Instant::now()).unwrap_or("")));
            }
        }
        self.session_log("control", json!({"from": caller.name, "how": "released"}));
        self.push_status();
        Ok(self.status())
    }

    fn mark(&self, caller: &Caller, mut mark: wire::Mark) -> LiveResult<wire::DeviceStatus> {
        let shared = lock(&self.inner).run.as_ref().map(|r| r.shared.clone()).ok_or_else(|| LiveError::wrong_state("nothing is running to mark"))?;
        if mark.sample == 0 {
            mark.sample = lock(&shared.run).head_sample;
        }
        if let Some(log) = lock(&shared.log).as_mut() {
            let _ = log.log("mark", json!({"sample": mark.sample, "label": mark.label, "by": caller.name}));
        }
        let _ = self.events.send(wire::DeviceEvent { kind: Some(wire::device_event::Kind::Mark(mark)) });
        Ok(self.status())
    }

    fn list_taps(&self) -> Vec<wire::TapInfo> {
        self.taps.taps()
    }

    fn subscribe(&self, _caller: &Caller, request: wire::Subscribe) -> LiveResult<Subscription> {
        self.taps.subscribe(&request)
    }

    fn subscribe_view(&self, _caller: &Caller, request: wire::SubscribeView) -> LiveResult<Subscription> {
        let processor: Option<(Box<dyn ViewProcessor>, usize)> = match &request.processing {
            Some(p) if !p.ops.is_empty() => {
                let info = self.taps.tap(&request.tap).ok_or_else(|| LiveError::not_found(format!("no tap {}", request.tap)))?;
                let channels: Vec<usize> = if p.channels.is_empty() { (0..info.channels.len()).collect() } else { p.channels.iter().map(|&c| c as usize).collect() };
                check_channels(&channels, info.channels.len())?;
                let mut runner = self.ops_runner(&p.ops, channels.len(), rate_of(&info))?;
                let rows = runner.chain().taps.first().map_or(channels.len(), |t| t.channels);
                runner.warm().map_err(|e| LiveError::internal(e.to_string()))?;
                Some((Box::new(OpsView { runner, channels }), rows))
            }
            _ => None,
        };
        self.taps.subscribe_view(&request, processor)
    }

    fn read_range(&self, _caller: &Caller, request: wire::ReadRange) -> LiveResult<wire::TapFrame> {
        let info = self.taps.tap(&request.tap).ok_or_else(|| LiveError::not_found(format!("no tap {}", request.tap)))?;
        if info.kind != wire::TapKind::Samples as i32 {
            return Err(LiveError::invalid(format!("tap {} carries events: they are in the recording's {}.events.jsonl", request.tap, request.tap)));
        }
        let channels: Vec<usize> = if request.channels.is_empty() { (0..info.channels.len()).collect() } else { request.channels.iter().map(|&c| c as usize).collect() };
        if channels.len() as u64 * request.count as u64 > MAX_READ_VALUES {
            return Err(LiveError::invalid(format!("at most {MAX_READ_VALUES} values per read: ask for fewer samples or channels")));
        }
        let range = request.first_sample..request.first_sample + request.count as u64;
        // A source's raw tap keeps its stored type (as its frames do); signal taps are float32
        let data = match self.source_index(&request.tap) {
            Some(i) => {
                let shared = self.current_shared().ok_or_else(|| LiveError::wrong_state("no run yet"))?;
                let src = &shared.sources[i];
                check_channels(&channels, src.layout.channel_count())?;
                match src.ring.read_stored(&channels, range.clone()) {
                    Some(bytes) => bytes,
                    None => {
                        let raw = src.raw_path.clone().ok_or_else(|| LiveError::invalid("a preview keeps only its last seconds (the ring): this range is gone"))?;
                        let rec = dspk_io::ZarrRecording::open(&self.config.data.join(raw)).map_err(|e| LiveError::internal(e.to_string()))?;
                        let mut out = vec![0u8; channels.len() * request.count as usize * src.layout.format.bytes()];
                        rec.read_stored(&channels, range, &mut out).map_err(|e| LiveError::invalid(e.to_string()))?;
                        out
                    }
                }
            }
            None => self.read_tap(&request.tap, &channels, range)?.iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        Ok(wire::TapFrame { tap: 0, sequence: 0, first_sample: request.first_sample, count: request.count, source_time_ns: 0, sent_unix_ns: dspk_stream::live::unix_now_ns(), lost_before: 0, data })
    }

    fn view(&self, caller: &Caller, request: wire::TapView) -> LiveResult<session::EnvelopeFrame> {
        let view = request.view.ok_or_else(|| LiveError::invalid("a view without its range"))?;
        // The newest view per plot wins: an older one still being made is dropped
        let key = (caller.name.clone(), request.plot_key);
        let generation = {
            let mut views = lock(&self.views);
            let g = views.entry(key.clone()).or_default();
            *g += 1;
            *g
        };
        let superseded = || request.plot_key != 0 && lock(&self.views).get(&key).is_some_and(|&g| g != generation);
        let info = self.taps.tap(&request.tap).ok_or_else(|| LiveError::not_found(format!("no tap {}", request.tap)))?;
        let total = info.channels.len();
        let channels: Vec<usize> = if view.channels.is_empty() { (0..total).collect() } else { view.channels.iter().map(|&c| c as usize).collect() };
        if view.end <= view.start || view.width == 0 {
            return Err(LiveError::invalid("a view needs start < end and a width"));
        }
        let ops: Vec<String> = view.processing.as_ref().map(|p| p.ops.clone()).unwrap_or_default();
        let rate = rate_of(&info);
        // Zoomed out, unprocessed, a source: from its overview (built as blocks arrived)
        if ops.is_empty()
            && let Some(i) = self.source_index(&request.tap)
            && let Some(shared) = self.current_shared()
            && let Some(min_max) = lock(&shared.sources[i].overview).columns(&channels, view.start..view.end, view.width as usize)
        {
            return Ok(session::EnvelopeFrame { content: Some(session::envelope_frame::Content::Columns(dspk_stream::protocol::wire::Columns { min_max, complete: true })) });
        }
        // With viewer ops: the chain runs from a little before the range, so its filters settle
        let warmup = if ops.is_empty() { 0 } else { ((VIEW_WARMUP_SECONDS * rate) as u64).min(view.start) };
        let read_from = view.start - warmup;
        let n = (view.end - read_from) as usize;
        if channels.len() as u64 * n as u64 > MAX_READ_VALUES {
            return Err(LiveError::invalid("this view reads too much at once: zoom in"));
        }
        if superseded() {
            return Err(LiveError::new(wire::ErrorCode::Superseded, "a newer view of this plot came first"));
        }
        let values = self.read_tap(&request.tap, &channels, read_from..view.end)?;
        let (values, rows, first) = if ops.is_empty() {
            (values, channels.len(), read_from)
        } else {
            let mut runner = self.ops_runner(&ops, channels.len(), rate)?;
            let stored: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let out = runner.process(&stored, n, read_from).map_err(|e| LiveError::internal(e.to_string()))?;
            match out.into_iter().find_map(|o| match o {
                TapOutput::Samples { first_sample, channels, values, .. } => Some((values, channels, first_sample)),
                TapOutput::Events { .. } => None,
            }) {
                Some(v) => v,
                None => return Err(LiveError::invalid("the viewer's ops give no signal for this range")),
            }
        };
        if superseded() {
            return Err(LiveError::new(wire::ErrorCode::Superseded, "a newer view of this plot came first"));
        }
        // The asked range only (the warm-up and any delay cut)
        let len = values.len() / rows.max(1);
        let skip = (view.start.saturating_sub(first) as usize).min(len);
        let keep = ((view.end - view.start) as usize).min(len - skip);
        if keep == 0 {
            return Err(LiveError::invalid("nothing in this range yet"));
        }
        let mut cut = Vec::with_capacity(rows * keep);
        for r in 0..rows {
            cut.extend_from_slice(&values[r * len + skip..r * len + skip + keep]);
        }
        Ok(session::EnvelopeFrame { content: Some(session::envelope_frame::Content::Columns(columns(&cut, rows, keep, view.width as usize))) })
    }

    fn log(&self, request: wire::GetLog) -> wire::LogEntries {
        let log = lock(&self.log);
        let kept: Vec<wire::LogEntry> = log.iter().filter(|e| e.level >= request.min_level).cloned().collect();
        let skip = if request.entries == 0 { 0 } else { kept.len().saturating_sub(request.entries as usize) };
        wire::LogEntries { entries: kept[skip..].to_vec() }
    }

    fn new_pairing_code(&self, caller: &Caller, control: bool) -> LiveResult<wire::PairingCode> {
        {
            let mut inner = lock(&self.inner);
            Self::require_director(&mut inner, caller)?;
        }
        let authority = self.authority.get().ok_or_else(|| LiveError::internal("this device does not pair (no authority)"))?;
        let permission = if control { Permission::Control } else { Permission::Read };
        let code = authority.new_pairing_code(permission).map_err(|e| LiveError::internal(e.to_string()))?;
        self.log(wire::LogLevel::Info, "device", format!("{} made a pairing code ({})", caller.name, if control { "may direct" } else { "watch only" }));
        Ok(wire::PairingCode { code: code.display(), expires_unix: code.expires_unix })
    }

    fn clock_reported(&self, caller: &Caller, offset_ns: i64, round_trip_ns: u64) {
        {
            let mut logged = lock(&self.clock_logged);
            if logged.get(&caller.name).is_some_and(|t| t.elapsed() < CLOCK_LOG_EVERY) {
                return;
            }
            logged.insert(caller.name.clone(), Instant::now());
        }
        self.session_log("clock", json!({"client": caller.name, "offset_ns": offset_ns, "round_trip_ns": round_trip_ns}));
    }
}
