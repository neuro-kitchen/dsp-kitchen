//! A recording on a server as a viewer sees it, over any connection: [`RemoteSignal`] is a
//! [`SignalBackend`] whose views are answered by the server (from its own pyramid, or processed
//! there when the view carries its ops), so a viewer draws a remote recording as it draws a local
//! one and nothing is copied whole.
//!
//! The connection is a [`Link`]: a QUIC session on a computer (`transport`), the page's WebSocket
//! in a browser (`web_client`). This module is the same on every target.

use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use dspk_core::{DspError, DspResult, RecordingInfo, RecordingSource};
use dspk_view::{Envelope, OnProgress, Progress, SignalBackend, View};

use crate::error::{StreamError, StreamResult};
use crate::protocol::{convert, wire, MAX_VIEW_COLUMNS, MAX_VIEW_VALUES};

/// How often an incomplete remote pyramid is asked about again.
const PYRAMID_POLL: Duration = Duration::from_secs(1);
/// Columns of the small view that asks whether the server's pyramid is complete.
const PROBE_COLUMNS: usize = 256;

/// A connection to a server, as the viewer's types need it.
pub trait Link: Send + Sync {
    /// Sends `kind` and waits for the server's answer; a refusal on the server is an error.
    /// Blocking: viewers call it from worker threads (never from a browser page's main thread).
    fn request(&self, kind: wire::request::Kind) -> StreamResult<wire::response::Kind>;

    /// Runs `work` on a thread of its own (a computer's thread, a browser's web worker).
    fn spawn(&self, name: &str, work: Box<dyn FnOnce() + Send>) -> StreamResult<()>;
}

/// Locks `mutex` without waiting on a browser page's main thread (which may not wait): the locks
/// here are held for a few instructions.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    #[cfg(target_family = "wasm")]
    loop {
        match mutex.try_lock() {
            Ok(guard) => return guard,
            Err(std::sync::TryLockError::Poisoned(e)) => return e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => std::hint::spin_loop(),
        }
    }
    #[cfg(not(target_family = "wasm"))]
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

fn dsp(e: StreamError) -> DspError {
    DspError::Io(format!("remote: {e}"))
}

/// The server's answer to `view` of its open recording, processed there when `processing` says.
pub fn view_over(link: &dyn Link, view: &View, processing: Option<&wire::Processing>) -> StreamResult<Envelope> {
    let mut request = convert::request_from_view(view)?;
    request.processing = processing.cloned();
    match link.request(wire::request::Kind::View(request))? {
        wire::response::Kind::Envelope(frame) => convert::envelope_from_frame(frame),
        _ => Err(StreamError::protocol("unexpected answer to a view")),
    }
}

/// Opens `path` under the server's data root as the connection's recording (blocking).
pub fn open_over(link: &dyn Link, path: &str, source: Option<&str>) -> StreamResult<RecordingInfo> {
    let open = wire::Open { path: path.to_string(), source: source.unwrap_or_default().to_string() };
    match link.request(wire::request::Kind::Open(open))? {
        wire::response::Kind::Header(header) => convert::info_from_header(&header),
        _ => Err(StreamError::protocol("unexpected answer to an open")),
    }
}

/// The entries of directory `dir` under the server's data root ("" = the root), blocking.
pub fn list_over(link: &dyn Link, dir: &str) -> StreamResult<wire::FileList> {
    match link.request(wire::request::Kind::ListFiles(wire::ListFiles { dir: dir.into() }))? {
        wire::response::Kind::Files(list) => Ok(list),
        _ => Err(StreamError::protocol("unexpected answer to a listing")),
    }
}

/// Calls `method` of the server's plugin `plugin` with `body` (blocking); its reply's bytes.
pub fn plugin_call(link: &dyn Link, plugin: &str, method: &str, body: Vec<u8>) -> StreamResult<Vec<u8>> {
    let call = wire::PluginCall { plugin: plugin.into(), method: method.into(), body };
    match link.request(wire::request::Kind::Plugin(call))? {
        wire::response::Kind::Plugin(reply) => Ok(reply.body),
        _ => Err(StreamError::protocol("unexpected answer to a plugin call")),
    }
}

/// The recording a connection has open, as a [`RecordingSource`]: samples (scaled `f32`) read on
/// the server as sample views, in pieces under its limits. For small reads (a window of
/// waveforms, the value under the cursor); a whole recording belongs on the server.
pub struct RemoteSource {
    link: Arc<dyn Link>,
    info: RecordingInfo,
}

impl RemoteSource {
    pub fn new(link: Arc<dyn Link>, info: RecordingInfo) -> Self {
        Self { link, info }
    }

    /// Samples of `channels` channels one request reads: a sample view of c channels and n
    /// samples is c × n values and n columns, both under the server's limits.
    fn samples_per_request(channels: usize) -> u64 {
        (MAX_VIEW_VALUES / channels.max(1) as u64).min(u64::from(MAX_VIEW_COLUMNS)).max(1)
    }
}

impl RecordingSource for RemoteSource {
    fn info(&self) -> &RecordingInfo {
        &self.info
    }

    /// Samples of `channels`, channel-major, fetched as sample views and reassembled.
    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        let n = (samples.end.saturating_sub(samples.start)) as usize;
        if out.len() != channels.len() * n {
            return Err(DspError::ShapeMismatch { expected: vec![channels.len(), n], actual: vec![out.len()] });
        }
        let step = Self::samples_per_request(channels.len());
        let mut start = samples.start;
        while start < samples.end {
            let end = (start + step).min(samples.end);
            let piece = (end - start) as usize;
            let view = View { channels: channels.to_vec(), start, end, width: piece };
            let values = match view_over(self.link.as_ref(), &view, None).map_err(dsp)? {
                Envelope::Samples(values) => values,
                Envelope::Columns { .. } => return Err(DspError::Io("remote: the server answered columns for a sample read".into())),
            };
            let offset = (start - samples.start) as usize;
            for (c, row) in values.chunks_exact(piece).enumerate() {
                out[c * n + offset..c * n + offset + piece].copy_from_slice(row);
            }
            start = end;
        }
        Ok(())
    }
}

/// The recording a connection has open, behind [`SignalBackend`]; with `processing`, its views
/// are the view's ops run on the server ([`Self::processed`]).
pub struct RemoteSignal {
    link: Arc<dyn Link>,
    source: Arc<RemoteSource>,
    processing: Option<wire::Processing>,
    /// Whether the server's pyramid answered complete views (until then views may hold NaN).
    complete: Arc<AtomicBool>,
    /// Samples of the server's pyramid built so far, as the last probe saw (the overview builds
    /// from the start: the probe's columns that hold values, as a share of the whole).
    built: Arc<std::sync::atomic::AtomicU64>,
    /// Which of the probe's columns held values (the server's overview, part by part).
    built_map: Arc<Mutex<Vec<bool>>>,
    listeners: Arc<Mutex<Vec<OnProgress>>>,
    polling: Arc<AtomicBool>,
}

impl RemoteSignal {
    /// The recording `link`'s server has open, described by `info`.
    pub fn over(link: Arc<dyn Link>, info: RecordingInfo) -> Self {
        let source = Arc::new(RemoteSource::new(link.clone(), info));
        Self { link, source, processing: None, complete: Arc::default(), built: Arc::default(), built_map: Arc::default(), listeners: Arc::default(), polling: Arc::default() }
    }

    /// Opens `path` on `link`'s server (blocking) and returns it as a signal.
    pub fn open(link: Arc<dyn Link>, path: &str) -> StreamResult<Self> {
        let info = open_over(link.as_ref(), path, None)?;
        Ok(Self::over(link, info))
    }

    /// The same recording as a [`RecordingSource`] (sample reads), sharing the connection.
    pub fn source(&self) -> Arc<RemoteSource> {
        self.source.clone()
    }

    /// The connection this signal is read through.
    pub fn link(&self) -> &Arc<dyn Link> {
        &self.link
    }

    /// The same recording with `ops` (pipeline nodes in JSON, in order) run by the server over
    /// `channels` (none = all): for a client that does not compute (a browser page). Samples
    /// are read as stored.
    pub fn processed(&self, channels: Vec<usize>, ops: Vec<String>) -> Self {
        let processing = wire::Processing { channels: channels.into_iter().map(|c| c as u32).collect(), ops };
        Self { link: self.link.clone(), source: self.source.clone(), processing: Some(processing), complete: self.complete.clone(), built: self.built.clone(), built_map: self.built_map.clone(), listeners: self.listeners.clone(), polling: self.polling.clone() }
    }

    fn tell(listeners: &Mutex<Vec<OnProgress>>, done: u64, total: u64) {
        for listener in lock(listeners).iter() {
            listener(Progress { filled: 0..0, done, total });
        }
    }

    /// Asks the server again, every [`PYRAMID_POLL`], until its pyramid answers complete views,
    /// telling the listeners each time (viewers then ask for their views again).
    fn poll_until_complete(&self) {
        if self.polling.swap(true, Ordering::Relaxed) {
            return;
        }
        let (link, complete, built, built_map, listeners, polling) = (self.link.clone(), self.complete.clone(), self.built.clone(), self.built_map.clone(), self.listeners.clone(), self.polling.clone());
        let total = self.source.info.samples;
        let work = Box::new(move || {
            let probe = View { channels: vec![0], start: 0, end: total, width: PROBE_COLUMNS };
            while !complete.load(Ordering::Relaxed) {
                std::thread::sleep(PYRAMID_POLL);
                let done = match view_over(link.as_ref(), &probe, None) {
                    Ok(Envelope::Columns { complete: true, .. }) | Ok(Envelope::Samples(_)) => {
                        complete.store(true, Ordering::Relaxed);
                        total
                    }
                    // Columns not built yet hold NaN
                    Ok(Envelope::Columns { values, .. }) => {
                        let map: Vec<bool> = values.iter().map(|p| p[0].is_finite()).collect();
                        let finite = map.iter().filter(|&&b| b).count() as u64;
                        let done = (finite * total / map.len().max(1) as u64).min(total);
                        built.store(done, Ordering::Relaxed);
                        *lock(&built_map) = map;
                        done
                    }
                    // The connection is gone: stop asking
                    Err(_) => break,
                };
                Self::tell(&listeners, done, total);
            }
            polling.store(false, Ordering::Relaxed);
        });
        if self.link.spawn("dspk-remote-pyramid", work).is_err() {
            self.polling.store(false, Ordering::Relaxed);
        }
    }
}

impl SignalBackend for RemoteSignal {
    fn info(&self) -> &RecordingInfo {
        &self.source.info
    }

    fn view(&self, view: &View, out: &mut Envelope) -> DspResult<()> {
        let envelope = view_over(self.link.as_ref(), view, self.processing.as_ref()).map_err(dsp)?;
        match &envelope {
            Envelope::Columns { complete: false, .. } => {
                self.complete.store(false, Ordering::Relaxed);
                self.poll_until_complete();
            }
            // The first complete answer ends "Building overview" for whoever follows it (the
            // polling above only runs after an incomplete one)
            Envelope::Columns { complete: true, .. } => {
                if !self.complete.swap(true, Ordering::Relaxed) {
                    let total = self.source.info.samples;
                    Self::tell(&self.listeners, total, total);
                }
            }
            Envelope::Samples(_) => {}
        }
        *out = envelope;
        Ok(())
    }

    /// Samples of `channels`, channel-major (through [`RemoteSource`]; as stored).
    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        self.source.read(channels, samples, out)
    }

    /// The server builds its own pyramid: nothing to steer from here.
    fn focus(&self, _sample: u64) {}

    fn prepare(&self, _start: u64, _end: u64) -> DspResult<()> {
        Ok(())
    }

    fn progress(&self) -> Progress {
        let total = self.source.info.samples;
        Progress { filled: 0..0, done: if self.complete.load(Ordering::Relaxed) { total } else { self.built.load(Ordering::Relaxed) }, total }
    }

    fn watch(&self, on_progress: OnProgress) {
        lock(&self.listeners).push(on_progress);
    }

    fn built(&self, columns: usize) -> Vec<bool> {
        if self.complete.load(Ordering::Relaxed) {
            return vec![true; columns];
        }
        let map = lock(&self.built_map);
        if map.is_empty() {
            return vec![false; columns];
        }
        // Each column takes the probe columns it spans: built when all of them are
        (0..columns)
            .map(|x| {
                let (a, b) = (x * map.len() / columns, ((x + 1) * map.len()).div_ceil(columns).max(x * map.len() / columns + 1));
                map[a.min(map.len() - 1)..b.min(map.len())].iter().all(|&v| v)
            })
            .collect()
    }
}
