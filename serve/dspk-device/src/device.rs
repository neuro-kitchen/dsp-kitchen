//! [`LiveDevice`]: what runs on a device. One source; a chain checked for live input; a run thread
//! that acquires, keeps the recent data in a ring, records it (unless a preview), runs the chain
//! and hands every tap's blocks to the clients subscribed. A lost connection never stops it.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use dspk_core::{ComputeTarget, DspError, DspResult, MemoryOrder, RecordingInfo, RecordingSource, SampleFormat};
use dspk_io::live::{LiveBlock, LiveEvent, LiveLayout, LiveRing, LiveSource, LiveZarrWriter, SessionLog, RAW_ZARR};
use dspk_io::ZarrOptions;
use dspk_pipeline::live::{check_live, LiveChain, LiveRunner, LiveTapKind, TapOutput};
use dspk_stream::live::{wire, Caller, ControlOutcome, Device, Director, Lifecycle, LiveError, LiveResult, Subscription};
use dspk_stream::protocol::{header_from_info, wire as session};
use serde_json::json;
use tokio::sync::broadcast;

use crate::config::DeviceConfig;
use crate::sources::open_source;
use crate::taps::{TapData, TapHub};

/// Log entries a device keeps in memory.
const LOG_KEPT: usize = 2_000;
/// Device events queued per client.
const EVENT_QUEUE: usize = 1_024;
/// How long the run thread waits for a block before checking the stop flag and the source's health.
const SOURCE_WAIT: Duration = Duration::from_millis(100);
/// A source silent for this long is reported disconnected.
const SILENT: Duration = Duration::from_secs(1);
/// Status pushed to clients at least this often while running.
const STATUS_EVERY: Duration = Duration::from_secs(1);
/// A heartbeat line in the session log this often.
const HEARTBEAT_EVERY: Duration = Duration::from_secs(10);
/// Block latencies kept for the median.
const LATENCIES_KEPT: usize = 512;

/// What a run shares with the device while it runs.
struct RunShared {
    ring: LiveRing,
    run: Mutex<wire::LiveRun>,
    source: Mutex<wire::SourceStatus>,
    health: Mutex<wire::DeviceHealth>,
    /// The session log of a recording.
    log: Mutex<Option<SessionLog>>,
}

struct Run {
    stop: Arc<AtomicBool>,
    /// Who asked it to stop (for the log).
    stopped_by: Arc<Mutex<String>>,
    thread: Option<JoinHandle<()>>,
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
    layout: LiveLayout,
    target: ComputeTarget,
    events: broadcast::Sender<wire::DeviceEvent>,
    taps: Arc<TapHub>,
    log: Arc<Mutex<VecDeque<wire::LogEntry>>>,
    inner: Mutex<Inner>,
    /// Set when the run thread ends by itself (the source ended, the chain failed): the reason.
    ended: Arc<Mutex<Option<String>>>,
    /// This device's own `Arc`, weakly.
    me: std::sync::Weak<LiveDevice>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl LiveDevice {
    /// Opens the device: its id, the source's layout (the source is opened once to learn it),
    /// the compute target, recordings a crash left open.
    pub fn open(config: DeviceConfig) -> DspResult<Arc<Self>> {
        config.check()?;
        std::fs::create_dir_all(&config.data)?;
        let id = device_id(&config.data)?;
        let layout = match config.sources[0].sensor_layout() {
            Some(layout) => layout?,
            None => open_source(&config.sources[0])?.layout().clone(),
        };
        let target = match &config.runtime {
            Some(name) => ComputeTarget::parse(name).map_err(|e| DspError::InvalidConfig(e.to_string()))?,
            None => ComputeTarget::from_env().map_err(|e| DspError::InvalidConfig(e.to_string()))?,
        };
        let (events, _) = broadcast::channel(EVENT_QUEUE);
        let device = Arc::new_cyclic(|me| Self {
            config,
            id,
            layout,
            target,
            events,
            taps: Arc::new(TapHub::default()),
            log: Arc::default(),
            inner: Mutex::new(Inner { lifecycle: Lifecycle::default(), director: Director::default(), chain: None, run: None, last: None, connected: HashMap::new() }),
            ended: Arc::default(),
            me: me.clone(),
        });
        for folder in crate::recordings::recover(&device.config.recordings_dir()) {
            device.log(wire::LogLevel::Warn, "recorder", format!("{} ended without a stop (a crash?): marked recovered", folder.display()));
        }
        device.taps.set_taps(vec![device.raw_tap()]);
        lock(&device.inner).lifecycle.ready();
        device.log(wire::LogLevel::Info, "device", format!("ready: source {} ({} channels at {} Hz), compute on {}", device.layout.name, device.layout.channel_count(), device.layout.sample_rate.rate_hz(), device.target.name()));
        Ok(device)
    }

    pub fn config(&self) -> &DeviceConfig {
        &self.config
    }

    /// The source's description as the protocol carries it.
    fn header(&self) -> session::Header {
        let mut info = RecordingInfo::new(self.layout.name.clone(), self.layout.channel_count(), 0, self.layout.sample_rate, self.layout.format, MemoryOrder::ChannelMajor);
        info.channels = self.layout.channels.clone();
        header_from_info(&info)
    }

    fn raw_tap(&self) -> wire::TapInfo {
        let h = self.header();
        wire::TapInfo {
            name: self.layout.name.clone(),
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

    /// The taps of `chain`, besides the raw one.
    fn chain_taps(&self, chain: &LiveChain) -> Vec<wire::TapInfo> {
        let raw = self.raw_tap();
        // The chain's channels: those it reads, in its order (reorders after processing permute them)
        let mut channels: Vec<session::Channel> = chain.input_channels.iter().map(|&c| raw.channels[c].clone()).collect();
        for ch in &mut channels {
            (ch.gain, ch.offset) = (1.0, 0.0);
        }
        let mut by_node: HashMap<String, Vec<session::Channel>> = HashMap::new();
        for step in &chain.steps {
            if let Some(stage) = &step.stage
                && let Some(order) = reorder_of(stage)
            {
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

    /// Ends a run the thread ended by itself (the source ended, the chain failed).
    fn reap(&self) {
        let reason = lock(&self.ended).take();
        if let Some(reason) = reason {
            let mut inner = lock(&self.inner);
            if let Some(mut run) = inner.run.take() {
                if let Some(t) = run.thread.take() {
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

    fn start_run(self: &Arc<Self>, caller: &Caller, request: &wire::Start) -> LiveResult<()> {
        let mut inner = lock(&self.inner);
        Self::require_director(&mut inner, caller)?;
        let chain = inner.chain.clone().ok_or_else(|| LiveError::wrong_state("no chain is configured: configure one first"))?;
        inner.lifecycle.start()?;

        let fail = |inner: &mut Inner, e: String| {
            let _ = inner.lifecycle.stop(format!("could not start: {e}"));
            LiveError::internal(e)
        };
        let source = match open_source(&self.config.sources[0]) {
            Ok(s) => s,
            Err(e) => return Err(fail(&mut inner, e.to_string())),
        };
        let chunk = ((self.config.recording.chunk_seconds * self.layout.sample_rate.rate_hz()).round() as u64).max(1);
        let capacity = LiveRing::capacity_for(&self.layout, self.config.ring.seconds, self.config.ring.max_mb << 20, 2 * chunk);
        let started = dspk_stream::live::unix_now_ns();

        // A recording: its folder, the raw data's store, the session log's first line
        let (path, recorder, log) = if request.record {
            let made = (|| -> DspResult<(PathBuf, LiveZarrWriter, SessionLog)> {
                let folder = crate::recordings::new_folder(&self.config.recordings_dir(), &request.label)?;
                let options = ZarrOptions { chunk_samples: chunk as usize, ..Default::default() };
                let writer = LiveZarrWriter::create(&folder.join(RAW_ZARR), &self.layout, &options)?;
                let mut log = SessionLog::create(&folder)?;
                log.log(
                    "start",
                    json!({
                        "device": {"id": self.id, "name": self.config.name, "dspk": env!("CARGO_PKG_VERSION"), "runtime": self.target.name()},
                        "director": caller.name,
                        "label": request.label,
                        "notes": request.notes,
                        "config": self.config,
                        "chain": serde_json::from_str::<serde_json::Value>(&chain.spec_json).unwrap_or_default(),
                        "block_ms": chain.block_ms,
                        "taps": chain.taps,
                        "source": {"name": self.layout.name, "channels": self.layout.channels, "sample_rate_hz": self.layout.sample_rate.rate_hz(), "format": self.layout.format},
                    }),
                )?;
                Ok((folder, writer, log))
            })();
            match made {
                Ok((p, w, l)) => (Some(p), Some(w), Some(l)),
                Err(e) => return Err(fail(&mut inner, e.to_string())),
            }
        } else {
            (None, None, None)
        };
        let runner = match LiveRunner::new(
            chain.clone(),
            self.target,
            self.layout.sample_rate.rate_hz(),
            self.layout.format,
            self.layout.channels.iter().map(|c| c.gain).collect(),
            self.layout.channels.iter().map(|c| c.offset).collect(),
        ) {
            Ok(r) => r,
            Err(e) => return Err(fail(&mut inner, e.to_string())),
        };
        let records = path.as_ref().map(|p| p.strip_prefix(&self.config.data).unwrap_or(p).display().to_string()).unwrap_or_default();
        let shared = Arc::new(RunShared {
            ring: LiveRing::new(self.layout.clone(), capacity),
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
            source: Mutex::new(wire::SourceStatus { name: self.layout.name.clone(), ..Default::default() }),
            health: Mutex::new(wire::DeviceHealth::default()),
            log: Mutex::new(log),
        });
        let stop = Arc::new(AtomicBool::new(false));
        let stopped_by = Arc::new(Mutex::new(String::new()));
        let thread = {
            let (device, shared, stop, stopped_by) = (self.clone(), shared.clone(), stop.clone(), stopped_by.clone());
            std::thread::Builder::new()
                .name("dspk-device-run".into())
                .spawn(move || device.run_loop(source, runner, recorder, shared, stop, stopped_by))
                .map_err(|e| LiveError::internal(e.to_string()))?
        };
        inner.run = Some(Run { stop, stopped_by, thread: Some(thread), shared });
        Ok(())
    }

    /// The run thread.
    fn run_loop(self: Arc<Self>, mut source: Box<dyn LiveSource>, mut runner: LiveRunner, mut recorder: Option<LiveZarrWriter>, shared: Arc<RunShared>, stop: Arc<AtomicBool>, stopped_by: Arc<Mutex<String>>) {
        let gains: Vec<f32> = self.layout.channels.iter().map(|c| c.gain).collect();
        let offsets: Vec<f32> = self.layout.channels.iter().map(|c| c.offset).collect();
        let chain_taps = self.chain_taps(runner.chain());
        let tap_names: Vec<String> = runner.chain().taps.iter().map(|t| t.name.clone()).collect();
        let mut derived: HashMap<usize, LiveZarrWriter> = HashMap::new();
        let mut latencies: VecDeque<u64> = VecDeque::with_capacity(LATENCIES_KEPT);
        let (mut last_data, mut last_status, mut last_heartbeat) = (Instant::now(), Instant::now(), Instant::now());
        let mut first = true;
        let mut ended: Option<String> = None;

        while !stop.load(Ordering::Relaxed) {
            match source.next(SOURCE_WAIT) {
                Ok(LiveEvent::Block(block)) => {
                    last_data = Instant::now();
                    if first {
                        lock(&shared.run).started_source_time_ns = block.source_time_ns;
                        first = false;
                    }
                    if let Err(e) = self.handle_block(&block, &shared, &mut runner, recorder.as_mut(), &mut derived, &chain_taps, &tap_names, &gains, &offsets) {
                        ended = Some(format!("the chain failed: {e}"));
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
                    ended = Some("the source ended".into());
                    break;
                }
                Err(e) => {
                    ended = Some(format!("the source failed: {e}"));
                    break;
                }
            }
            // Health, every pass
            {
                let counters = source.counters();
                let mut s = lock(&shared.source);
                let silent = last_data.elapsed();
                if s.connected && silent > SILENT {
                    let name = s.name.clone();
                    drop(s);
                    self.log(wire::LogLevel::Warn, &format!("source:{name}"), format!("no data for {:.1} s", silent.as_secs_f64()));
                    s = lock(&shared.source);
                }
                s.connected = silent <= SILENT;
                s.checksum_errors = counters.checksum_errors;
                s.lost_samples = counters.lost_samples.max(lock(&shared.run).lost_samples);
                let mut h = lock(&shared.health);
                let mut sorted: Vec<u64> = latencies.iter().copied().collect();
                sorted.sort_unstable();
                h.block_latency_median_us = sorted.get(sorted.len() / 2).copied().unwrap_or(0) as u32;
                h.block_latency_max_us = sorted.last().copied().unwrap_or(0) as u32;
                h.ring_use = shared.ring.stats().use_fraction();
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
                let stored = recorder.as_ref().map(|r| r.stored_samples());
                if let Some(log) = lock(&shared.log).as_mut() {
                    let _ = log.log(
                        "heartbeat",
                        json!({"head_sample": run.head_sample, "samples_stored": stored, "lost_samples": run.lost_samples, "ring_use": health.ring_use,
                               "block_latency_median_us": health.block_latency_median_us, "block_latency_max_us": health.block_latency_max_us, "frames_dropped": health.frames_dropped}),
                    );
                }
            }
        }

        // End: the last chunk stored, the stop line
        let stored = recorder.map(|r| r.finish());
        for (_, writer) in derived {
            let _ = writer.finish();
        }
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
            let _ = log.log("stop", json!({"reason": reason, "head_sample": run.head_sample, "samples_stored": stored, "lost_samples": run.lost_samples}));
        }
        if let Some(Err(e)) = stored {
            self.log(wire::LogLevel::Error, "recorder", format!("the recording's last chunk: {e}"));
        }
        if let Some(why) = ended {
            *lock(&self.ended) = Some(why);
            // Tell the device from another thread: reaping joins this one
            let device = self.clone();
            std::thread::spawn(move || device.reap());
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_block(
        &self,
        block: &LiveBlock,
        shared: &RunShared,
        runner: &mut LiveRunner,
        recorder: Option<&mut LiveZarrWriter>,
        derived: &mut HashMap<usize, LiveZarrWriter>,
        chain_taps: &[wire::TapInfo],
        tap_names: &[String],
        gains: &[f32],
        offsets: &[f32],
    ) -> DspResult<()> {
        block.check(&self.layout)?;
        if block.lost_before > 0 {
            if let Some(log) = lock(&shared.log).as_mut() {
                let _ = log.log("gap", json!({"sample": block.first_sample, "lost_samples": block.lost_before, "source": self.layout.name}));
            }
            lock(&shared.run).lost_samples += block.lost_before;
        }
        let block = Arc::new(block.clone());
        shared.ring.push(block.clone());
        if let Some(rec) = recorder {
            rec.push(&block)?;
        }
        let channels = self.layout.channel_count();
        self.taps.publish(
            &self.layout.name,
            channels,
            block.first_sample,
            block.samples,
            block.source_time_ns,
            block.lost_before,
            TapData::Stored { bytes: &block.data, format: self.layout.format, gains, offsets },
        );
        for output in runner.process(&block.data, block.samples, block.first_sample)? {
            match output {
                TapOutput::Samples { tap, first_sample, samples, channels, values } => {
                    let name = &tap_names[tap];
                    self.taps.publish(name, channels, first_sample, samples, block.source_time_ns, block.lost_before, TapData::Values(&values));
                    if runner.chain().taps[tap].record && lock(&shared.run).recording {
                        self.record_derived(derived, tap, &chain_taps[tap], shared, first_sample, samples, channels, &values)?;
                    }
                }
                TapOutput::Events { tap, events } => {
                    if events.is_empty() {
                        continue;
                    }
                    let pairs: Vec<(u64, u32)> = events.iter().map(|e| (e.onset, e.channel as u32)).collect();
                    let name = &tap_names[tap];
                    let channels = chain_taps[tap].channels.len();
                    self.taps.publish(name, channels, block.first_sample, block.samples, block.source_time_ns, 0, TapData::Events(&pairs));
                    if runner.chain().taps[tap].record
                        && let Some(log) = lock(&shared.log).as_mut()
                    {
                        for e in &events {
                            let _ = log.log("event", json!({"tap": name, "channel": e.channel, "onset": e.onset, "offset": e.offset, "peak_sample": e.peak_sample, "peak_value": e.peak_value}));
                        }
                    }
                }
            }
        }
        lock(&shared.run).head_sample = block.end_sample();
        Ok(())
    }

    /// A derived tap recorded next to the raw data (`<tap>.zarr`, float32).
    #[allow(clippy::too_many_arguments)]
    fn record_derived(&self, derived: &mut HashMap<usize, LiveZarrWriter>, tap: usize, info: &wire::TapInfo, shared: &RunShared, first_sample: u64, samples: usize, channels: usize, values: &[f32]) -> DspResult<()> {
        if !derived.contains_key(&tap) {
            let run = lock(&shared.run).clone();
            let folder = self.config.data.join(&run.path);
            let layout = LiveLayout {
                name: info.name.clone(),
                channels: info.channels.iter().map(|c| dspk_core::ChannelInfo { name: c.name.clone(), gain: 1.0, offset: 0.0, unit: self.layout.channels.first().map(|c| c.unit.clone()).unwrap_or_default() }).collect(),
                sample_rate: self.layout.sample_rate,
                format: SampleFormat::F32,
            };
            let chunk = ((self.config.recording.chunk_seconds * self.layout.sample_rate.rate_hz()).round() as usize).max(1);
            let name = crate::recordings::safe_label(&info.name);
            derived.insert(tap, LiveZarrWriter::create(&folder.join(format!("{name}.zarr")), &layout, &ZarrOptions { chunk_samples: chunk, ..Default::default() })?);
        }
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let block = LiveBlock { first_sample, samples, source_time_ns: 0, received_unix_ns: 0, lost_before: 0, data: bytes };
        let _ = channels;
        derived.get_mut(&tap).expect("made above").push(&block)
    }

    fn current_shared(&self) -> Option<Arc<RunShared>> {
        let inner = lock(&self.inner);
        inner.run.as_ref().map(|r| r.shared.clone()).or_else(|| inner.last.clone())
    }

    /// Stored values of the raw tap over `range`: from the ring, else from the recording on disk.
    fn read_raw(&self, channels: &[usize], range: std::ops::Range<u64>) -> LiveResult<Vec<u8>> {
        let shared = self.current_shared().ok_or_else(|| LiveError::wrong_state("no run yet"))?;
        if let Some(bytes) = shared.ring.read_stored(channels, range.clone()) {
            return Ok(bytes);
        }
        let run = lock(&shared.run).clone();
        if !run.recording {
            return Err(LiveError::invalid("a preview keeps only its last seconds (the ring): this range is gone"));
        }
        let rec = dspk_io::ZarrRecording::open(&self.config.data.join(&run.path).join(RAW_ZARR)).map_err(|e| LiveError::internal(e.to_string()))?;
        let mut out = vec![0u8; channels.len() * (range.end - range.start) as usize * self.layout.format.bytes()];
        rec.read_stored(channels, range, &mut out).map_err(|e| LiveError::invalid(e.to_string()))?;
        Ok(out)
    }
}

/// The order of a reorder stage.
fn reorder_of(stage: &dspk_ops::PipelineStage) -> Option<&[usize]> {
    match stage {
        dspk_ops::PipelineStage::ReorderChannels { order } => Some(order),
        _ => None,
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

impl Device for LiveDevice {
    fn info(&self) -> wire::DeviceInfo {
        let h = self.header();
        wire::DeviceInfo {
            id: self.id.clone(),
            name: self.config.name.clone(),
            version: env!("CARGO_PKG_VERSION").into(),
            runtime: self.target.name().into(),
            sources: vec![wire::SourceInfo { name: self.layout.name.clone(), kind: self.config.sources[0].kind().into(), channels: h.channels, sample_rate_hz: h.sample_rate_hz, format: h.format }],
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
        let chain = inner.chain.as_ref().map(|c| wire::ChainInfo {
            spec_json: c.spec_json.clone(),
            block_samples: ((c.block_ms / 1000.0 * self.layout.sample_rate.rate_hz()).round() as u32).max(1),
            delay_samples: c.delay_samples,
            taps: self.taps.taps(),
        });
        wire::DeviceStatus {
            state: inner.lifecycle.state() as i32,
            message: inner.lifecycle.message().to_string(),
            director,
            director_kept_until_unix_ns: kept,
            chain,
            run: shared.as_ref().map(|s| lock(&s.run).clone()),
            sources: shared.as_ref().map(|s| vec![lock(&s.source).clone()]).unwrap_or_default(),
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
        self.session_log("client", json!({"client": caller.name, "event": "disconnected"}));
        self.push_status();
    }

    fn configure(&self, caller: &Caller, request: wire::ConfigureChain) -> LiveResult<wire::ConfigureResult> {
        self.reap();
        let checked = check_live(&request.spec_json, self.layout.channel_count());
        // The chain must read this device's source
        let checked = checked.and_then(|chain| match &chain.source {
            Some(s) if *s != self.layout.name => Err(vec![dspk_pipeline::live::Refusal { node: String::new(), op: "read".into(), reason: format!("source {s}: this device's source is {}", self.layout.name) }]),
            _ => Ok(chain),
        });
        let refusals = |r: &Vec<dspk_pipeline::live::Refusal>| r.iter().map(|r| wire::Refusal { node: r.node.clone(), op: r.op.clone(), reason: r.reason.clone() }).collect::<Vec<_>>();
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
                    self.taps.set_taps(std::iter::once(self.raw_tap()).chain(self.chain_taps(chain)).collect());
                    inner.chain = Some(chain.clone());
                    inner.lifecycle.end_configure(true, format!("chain configured by {}", caller.name));
                }
                Err(r) => inner.lifecycle.end_configure(false, format!("chain refused: {} node(s)", r.len())),
            }
        }
        match &checked {
            Ok(chain) => self.log(wire::LogLevel::Info, "device", format!("{} configured a chain: {} steps, taps {:?}", caller.name, chain.steps.len(), chain.taps.iter().map(|t| &t.name).collect::<Vec<_>>())),
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
        // Needs `Arc<Self>` for the run thread: the device is always held in one
        let me = self.arc();
        me.start_run(caller, &request)?;
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
        if let Some(mut run) = run {
            *lock(&run.stopped_by) = caller.name.clone();
            run.stop.store(true, Ordering::Relaxed);
            if let Some(t) = run.thread.take() {
                let _ = t.join();
            }
            lock(&self.inner).last = Some(run.shared);
        }
        self.log(wire::LogLevel::Info, "device", format!("{} stopped the run", caller.name));
        self.push_status();
        Ok(self.status())
    }

    fn set_parameter(&self, caller: &Caller, request: wire::SetParameter) -> LiveResult<wire::DeviceStatus> {
        let mut inner = lock(&self.inner);
        Self::require_director(&mut inner, caller)?;
        Err(LiveError::invalid(format!(
            "{}.{}: changing a parameter while running is planned; stop, configure the changed chain, start",
            request.node, request.parameter
        )))
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
        self.taps.subscribe_view(&request)
    }

    fn read_range(&self, _caller: &Caller, request: wire::ReadRange) -> LiveResult<wire::TapFrame> {
        if request.tap != self.layout.name {
            return Err(LiveError::not_found(format!("tap {}: only the raw data ({}) can be read back in this version", request.tap, self.layout.name)));
        }
        const MAX_READ_VALUES: u64 = 8 << 20;
        let channels: Vec<usize> = if request.channels.is_empty() { (0..self.layout.channel_count()).collect() } else { request.channels.iter().map(|&c| c as usize).collect() };
        if let Some(bad) = channels.iter().find(|&&c| c >= self.layout.channel_count()) {
            return Err(LiveError::invalid(format!("channel {bad} does not exist")));
        }
        if channels.len() as u64 * request.count as u64 > MAX_READ_VALUES {
            return Err(LiveError::invalid(format!("at most {MAX_READ_VALUES} values per read: ask for fewer samples or channels")));
        }
        let range = request.first_sample..request.first_sample + request.count as u64;
        let data = self.read_raw(&channels, range)?;
        Ok(wire::TapFrame { tap: 0, sequence: 0, first_sample: request.first_sample, count: request.count, source_time_ns: 0, sent_unix_ns: dspk_stream::live::unix_now_ns(), lost_before: 0, data })
    }

    fn view(&self, _caller: &Caller, request: wire::TapView) -> LiveResult<session::EnvelopeFrame> {
        let view = request.view.ok_or_else(|| LiveError::invalid("a view without its range"))?;
        if request.tap != self.layout.name {
            return Err(LiveError::not_found(format!("tap {}: views of past ranges are of the raw data ({}) in this version", request.tap, self.layout.name)));
        }
        if view.processing.as_ref().is_some_and(|p| !p.ops.is_empty()) {
            return Err(LiveError::invalid("viewer ops on past ranges of a run are planned"));
        }
        let channels: Vec<usize> = if view.channels.is_empty() { (0..self.layout.channel_count()).collect() } else { view.channels.iter().map(|&c| c as usize).collect() };
        if view.end <= view.start || view.width == 0 {
            return Err(LiveError::invalid("a view needs start < end and a width"));
        }
        let n = view.end - view.start;
        if channels.len() as u64 * n > dspk_stream::MAX_VIEW_VALUES * 8 {
            return Err(LiveError::invalid("this view reads too much at once in this version: zoom in"));
        }
        let stored = self.read_raw(&channels, view.start..view.end)?;
        let n = n as usize;
        let width = (view.width as usize).min(n);
        let row = n * self.layout.format.bytes();
        let mut min_max = vec![0.0f32; 2 * channels.len() * width];
        let mut values = vec![0.0f32; n];
        for (r, &c) in channels.iter().enumerate() {
            let ch = &self.layout.channels[c];
            self.layout.format.decode(&stored[r * row..(r + 1) * row], &mut values, ch.gain, ch.offset);
            for col in 0..width {
                let (a, b) = (col * n / width, ((col + 1) * n / width).max(col * n / width + 1));
                let slice = &values[a..b.min(n)];
                let at = 2 * (r * width + col);
                min_max[at] = slice.iter().copied().fold(f32::INFINITY, f32::min);
                min_max[at + 1] = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            }
        }
        Ok(session::EnvelopeFrame { content: Some(session::envelope_frame::Content::Columns(session::Columns { min_max, complete: true })) })
    }

    fn log(&self, request: wire::GetLog) -> wire::LogEntries {
        let log = lock(&self.log);
        let kept: Vec<wire::LogEntry> = log.iter().filter(|e| e.level >= request.min_level).cloned().collect();
        let skip = if request.entries == 0 { 0 } else { kept.len().saturating_sub(request.entries as usize) };
        wire::LogEntries { entries: kept[skip..].to_vec() }
    }
}

impl LiveDevice {
    fn chain_info(&self, chain: &LiveChain) -> wire::ChainInfo {
        wire::ChainInfo {
            spec_json: chain.spec_json.clone(),
            block_samples: ((chain.block_ms / 1000.0 * self.layout.sample_rate.rate_hz()).round() as u32).max(1),
            delay_samples: chain.delay_samples,
            taps: std::iter::once(self.raw_tap()).chain(self.chain_taps(chain)).collect(),
        }
    }

    /// This device as an `Arc`, for the run thread (the [`Device`] trait takes `&self`; a device
    /// is only made by [`Self::open`], inside an `Arc`).
    fn arc(&self) -> Arc<Self> {
        self.me.upgrade().expect("a LiveDevice lives in the Arc LiveDevice::open made")
    }
}
