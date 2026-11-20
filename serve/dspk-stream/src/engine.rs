//! What a server does with a recording, without a network or an async runtime: the recording a
//! session has open ([`OpenRecording`]: its source, pyramid and processed chains), the checks of
//! a view request, and the answers to views ([`answer_view`]). The same on every target: the QUIC
//! and WebSocket servers (`transport`) call it from their runtime, and a browser's engine worker
//! calls it directly, so a page and `dspk serve` answer a view with one code path.

use std::path::Path;
use std::sync::{Arc, Mutex};

use dspk_core::{RecordingInfo, RecordingSource, SampleFormat};
use dspk_view::{Envelope, Pyramid, SignalBackend, View};

use crate::error::{StreamError, StreamResult};
use crate::protocol::{view_from_request, wire, MAX_VIEW_COLUMNS, MAX_VIEW_VALUES};

/// What the signal stream carries: stored words when the source can read them, else scaled
/// `f32` (and the header then says `f32`, gain 1, offset 0).
fn streamed_info(source: &dyn RecordingSource) -> (RecordingInfo, bool) {
    let info = source.info().clone();
    let all: Vec<usize> = (0..info.channel_count()).collect();
    // One sample of every channel tells whether stored reads are supported
    let mut probe = vec![0u8; all.len() * info.format.bytes()];
    let stored = info.samples == 0 || source.read_stored(&all, 0..1, &mut probe).is_ok();
    if stored {
        return (info, true);
    }
    let mut scaled = info;
    scaled.format = SampleFormat::F32;
    for c in &mut scaled.channels {
        (c.gain, c.offset) = (1.0, 0.0);
    }
    (scaled, false)
}

/// A recording a session serves: its source, pyramid (views), and what the signal carries.
pub struct OpenRecording {
    pub(crate) source: Arc<dyn RecordingSource>,
    pub(crate) pyramid: Option<Arc<Pyramid>>,
    info: RecordingInfo,
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    pub(crate) stored: bool,
    processor: Option<Arc<dyn ViewProcessor>>,
    /// Moves where the pyramid's builder works next (a view looked at a sample), while it builds.
    refocus: Option<Arc<dyn Fn(u64) + Send + Sync>>,
    /// The processed signals clients asked views of, latest last (a few: each holds device memory),
    /// and whether each runs on the CPU (made while a batch job held the GPU).
    processed: Arc<Mutex<Vec<(wire::Processing, bool, Arc<ChainSlot>)>>>,
}

type Kept = Mutex<Vec<(wire::Processing, bool, Arc<ChainSlot>)>>;

/// Every open recording's kept chains, so a batch job taking the GPU drops those on it.
fn every_kept() -> &'static Mutex<Vec<std::sync::Weak<Kept>>> {
    static ALL: std::sync::OnceLock<Mutex<Vec<std::sync::Weak<Kept>>>> = std::sync::OnceLock::new();
    ALL.get_or_init(|| {
        dspk_core::lease::on_release(release_processed);
        Mutex::default()
    })
}

/// Drops the processed chains kept on the GPU, in every open recording (a batch job takes the
/// GPU): the next view makes its chain again.
pub fn release_processed() {
    let all = every_kept().lock().unwrap_or_else(|e| e.into_inner()).clone();
    for kept in all.iter().filter_map(std::sync::Weak::upgrade) {
        kept.lock().unwrap_or_else(|e| e.into_inner()).retain(|(_, cpu, _)| *cpu);
    }
}

/// One chain's processed signal: made once, by the first view that asks; later views of the
/// chain wait for it there, views of other chains do not.
type ChainSlot = std::sync::OnceLock<Result<Arc<dyn SignalBackend>, String>>;

/// Processed signals kept per open recording.
const PROCESSED_KEPT: usize = 2;

/// Runs a view's ops on the server for clients that cannot compute (pages). The ops and the
/// device are the caller's (dspk-stream knows neither): the command line runs dspk-pipeline.
pub trait ViewProcessor: Send + Sync {
    /// The signal of `source` with `ops` (pipeline nodes in JSON, in order) applied over
    /// `channels` (none = all). May take seconds (a whitening is fitted): called off the runtime.
    fn processed(&self, source: Arc<dyn RecordingSource>, pyramid: Option<Arc<Pyramid>>, channels: &[usize], ops: &[String]) -> StreamResult<Arc<dyn SignalBackend>>;

    /// Makes `view` of the chain made by [`Self::processed`] ready without blocking on the
    /// device, so the answer that follows ([`answer_view`]) is served from what this computed.
    /// A browser cannot wait for its GPU, so its engine must; a server may (the default does
    /// nothing, and the answer waits).
    fn warm<'a>(&'a self, channels: &'a [usize], ops: &'a [String], view: &'a View) -> WarmFuture<'a> {
        let _ = (channels, ops, view);
        Box::pin(std::future::ready(Ok(())))
    }
}

/// A view's processing by dspk-pipeline's chains: what `dspk serve` and a browser page's engine
/// run (the stored signal and its pyramid answer what a chain does not process).
#[cfg(feature = "pipeline")]
impl ViewProcessor for dspk_pipeline::view::ProcessedChains {
    fn processed(&self, source: Arc<dyn RecordingSource>, pyramid: Option<Arc<Pyramid>>, channels: &[usize], ops: &[String]) -> StreamResult<Arc<dyn SignalBackend>> {
        let raw: Arc<dyn SignalBackend> = Arc::new(dspk_view::StoredSignal::new(source.clone(), pyramid));
        Ok(self.make(raw, source, channels, ops).map_err(StreamError::Server)?)
    }

    fn warm<'a>(&'a self, channels: &'a [usize], ops: &'a [String], view: &'a View) -> WarmFuture<'a> {
        Box::pin(async move { dspk_pipeline::view::ProcessedChains::warm(self, channels, ops, view).await.map_err(|e| StreamError::Server(e.to_string())) })
    }
}

/// What [`ViewProcessor::warm`] returns: sent between threads on a server's runtime; a browser's
/// futures stay on their thread.
#[cfg(not(target_family = "wasm"))]
pub type WarmFuture<'a> = std::pin::Pin<Box<dyn std::future::Future<Output = StreamResult<()>> + Send + 'a>>;
#[cfg(target_family = "wasm")]
pub type WarmFuture<'a> = std::pin::Pin<Box<dyn std::future::Future<Output = StreamResult<()>> + 'a>>;

impl OpenRecording {
    pub fn new(source: Arc<dyn RecordingSource>, pyramid: Option<Arc<Pyramid>>) -> Self {
        let (info, stored) = streamed_info(source.as_ref());
        let processed: Arc<Kept> = Arc::default();
        let mut all = every_kept().lock().unwrap_or_else(|e| e.into_inner());
        all.retain(|w| w.strong_count() > 0);
        all.push(Arc::downgrade(&processed));
        drop(all);
        Self { source, pyramid, info, stored, processor: None, refocus: None, processed }
    }

    /// Views move the pyramid's builder to where they look ([`Self::look_at`]): `refocus` gets the
    /// sample at a view's centre.
    pub fn with_refocus(mut self, refocus: Arc<dyn Fn(u64) + Send + Sync>) -> Self {
        self.refocus = Some(refocus);
        self
    }

    /// A view of samples `start..end` was asked: while the pyramid builds, it builds there next.
    /// Views of most of the recording (a client asking how far the build is) move nothing.
    pub fn look_at(&self, start: u64, end: u64) {
        let total = self.info.samples;
        let Some(refocus) = self.refocus.as_ref().filter(|_| self.pyramid.as_ref().is_some_and(|p| !p.is_complete())) else { return };
        if end > start && (end - start) * 2 < total {
            refocus(start + (end - start) / 2);
        }
    }

    /// Views with processing are answered through `processor` (without one they are refused).
    pub fn with_processor(mut self, processor: Arc<dyn ViewProcessor>) -> Self {
        self.processor = Some(processor);
        self
    }

    /// The signal `processing` describes: kept from an earlier view, or made now (blocking). The
    /// list is locked only to find or add the chain's slot; the chain is made outside the lock,
    /// so a chain being made (a whitening fitted: seconds) holds up only views of that chain.
    pub fn processed(&self, processing: &wire::Processing) -> StreamResult<Arc<dyn SignalBackend>> {
        self.processed_on(processing, false)
    }

    /// [`Self::processed`] on the CPU (`on_cpu`: a batch job holds the GPU) or the default device.
    pub fn processed_on(&self, processing: &wire::Processing, on_cpu: bool) -> StreamResult<Arc<dyn SignalBackend>> {
        let processor = self.processor.as_ref().ok_or_else(|| StreamError::protocol("this server does not process views"))?;
        let channels: Vec<usize> = processing.channels.iter().map(|&c| c as usize).collect();
        if channels.iter().any(|&c| c >= self.info.channel_count()) {
            return Err(StreamError::protocol("processing asks for a channel the recording does not have"));
        }
        let slot = {
            let mut kept = self.processed.lock().unwrap_or_else(|e| e.into_inner());
            match kept.iter().position(|(p, cpu, _)| p == processing && *cpu == on_cpu) {
                Some(at) => {
                    // Most recently used last
                    let entry = kept.remove(at);
                    let slot = entry.2.clone();
                    kept.push(entry);
                    slot
                }
                None => {
                    if kept.len() >= PROCESSED_KEPT {
                        kept.remove(0);
                    }
                    let slot = Arc::new(ChainSlot::new());
                    kept.push((processing.clone(), on_cpu, slot.clone()));
                    slot
                }
            }
        };
        let make = || processor.processed(self.source.clone(), self.pyramid.clone(), &channels, &processing.ops).map_err(|e| e.reason());
        let made = slot.get_or_init(|| if on_cpu { dspk_core::device::with_target(dspk_core::ComputeTarget::Cpu, make) } else { make() });
        match made {
            Ok(signal) => Ok(signal.clone()),
            Err(e) => {
                // Not kept: the next view of this chain tries again
                self.processed.lock().unwrap_or_else(|e| e.into_inner()).retain(|(_, _, s)| !Arc::ptr_eq(s, &slot));
                Err(StreamError::Server(e.clone()))
            }
        }
    }

    /// What the signal stream carries (the header sent to clients).
    pub fn info(&self) -> &RecordingInfo {
        &self.info
    }
}

/// Opens the recordings clients ask for, by path under the data root. The formats are the
/// caller's (dspk-stream knows none): the command line opens through dspk-synapse.
pub trait RecordingOpener: Send + Sync {
    /// Opens `path` (`source`: one of the file's sources, `None` = its default).
    fn open(&self, path: &Path, source: Option<&str>) -> StreamResult<OpenRecording>;

    /// The sources `path` holds (an NWB store's series); none when the format has one only, or
    /// the opener does not list them.
    fn sources(&self, _path: &Path) -> StreamResult<Vec<crate::protocol::wire::SourceInfo>> {
        Ok(Vec::new())
    }
}

/// The view a client asks for, refused when it is outside the recording or larger than the
/// server answers: channels in range and each once, at most [`MAX_VIEW_COLUMNS`] columns and
/// [`MAX_VIEW_VALUES`] values (channels × the columns, or the samples when fewer).
pub fn checked_view(request: &wire::ViewRequest, info: &RecordingInfo) -> StreamResult<View> {
    let view = view_from_request(request);
    view.validate(info.channel_count(), info.samples).map_err(|e| StreamError::protocol(e.to_string()))?;
    let mut seen = vec![false; info.channel_count()];
    for &c in &view.channels {
        if std::mem::replace(&mut seen[c], true) {
            return Err(StreamError::protocol(format!("channel {c} asked for twice")));
        }
    }
    if request.width > MAX_VIEW_COLUMNS {
        return Err(StreamError::protocol(format!("{} columns asked for: at most {MAX_VIEW_COLUMNS}", request.width)));
    }
    let values = view.channels.len() as u64 * (view.end - view.start).min(u64::from(request.width));
    if values > MAX_VIEW_VALUES {
        return Err(StreamError::protocol(format!("a view of {values} values: at most {MAX_VIEW_VALUES} (fewer channels or columns)")));
    }
    Ok(view)
}

/// Makes a processed view of `recording` ready without blocking on the device: its chain made
/// (or found), then warmed by the recording's processor. Blocking where the chain is made (a
/// whitening's fit): a server calls this from a blocking task, a browser's engine refuses those
/// chains. Views without ops need nothing.
pub async fn warm_view(recording: &OpenRecording, request: &wire::ViewRequest) -> StreamResult<()> {
    let Some(processing) = request.processing.as_ref().filter(|p| !p.ops.is_empty()) else { return Ok(()) };
    // A batch job holds the GPU, or the GPU was lost: the view is answered on the CPU, nothing to
    // warm on the device
    if dspk_core::lease::batch_running() || gpu_lost() {
        return Ok(());
    }
    let view = checked_view(request, recording.info())?;
    recording.processed(processing)?;
    let channels: Vec<usize> = processing.channels.iter().map(|&c| c as usize).collect();
    match &recording.processor {
        Some(processor) => processor.warm(&channels, &processing.ops, &view).await,
        None => Ok(()),
    }
}

/// Whether the server's GPU was lost (wgpu poisons a device that runs out of memory or resets:
/// every later launch on it fails until the process restarts). Processed views then run on the
/// CPU, the same chains, instead of failing.
static GPU_LOST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether `error` says the device is gone, not that one view failed.
fn device_lost(error: &str) -> bool {
    // What a poisoned wgpu device answers: the out-of-memory itself, then every later launch's
    // missing buffers
    ["Out of Memory", "device runner", "poison", "Device lost", "DeviceLost", "Memory location was never initialized", "couldn't find resource for that handle"].iter().any(|m| error.contains(m))
}

/// Notes a view's error: when it says the GPU is gone, processed views move to the CPU from now
/// on (true: the caller answers again, on the CPU).
pub fn note_device_error(error: &str) -> bool {
    if !device_lost(error) {
        return false;
    }
    if !GPU_LOST.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::error!("the GPU was lost ({error}): processed views run on the CPU until the server restarts");
    }
    true
}

/// Whether processed views run on the CPU because the GPU was lost ([`GPU_LOST`]).
pub fn gpu_lost() -> bool {
    GPU_LOST.load(std::sync::atomic::Ordering::Relaxed)
}

/// The answer to view `request` of `recording`: from its pyramid or its samples, or processed
/// when the request carries ops (blocking: a chain may take seconds to make).
pub fn answer_view(recording: &OpenRecording, request: &wire::ViewRequest) -> StreamResult<Envelope> {
    let view = checked_view(request, recording.info())?;
    recording.look_at(view.start, view.end);
    match request.processing.as_ref().filter(|p| !p.ops.is_empty()) {
        Some(processing) => {
            // The GPU unless a batch job (a sort) holds it: then the CPU, the same chain
            let gpu = dspk_core::lease::interactive();
            let on_cpu = gpu.is_none() || gpu_lost();
            let answer = |on_cpu: bool| -> StreamResult<Envelope> {
                let mut out = Envelope::Samples(Vec::new());
                recording.processed_on(processing, on_cpu)?.view(&view, &mut out)?;
                Ok(out)
            };
            let cpu = || dspk_core::device::with_target(dspk_core::ComputeTarget::Cpu, || answer(true));
            if on_cpu {
                return cpu();
            }
            match answer(false) {
                Err(e) if note_device_error(&e.reason()) => cpu(),
                other => other,
            }
        }
        None => Ok(view.read(recording.source.as_ref(), recording.pyramid.as_deref())?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    /// Makes chains as their first op says: "slow" takes a while, "bad" fails; counts each make.
    #[derive(Default)]
    struct Maker {
        made: Mutex<Vec<String>>,
        slow_started: AtomicUsize,
    }

    impl ViewProcessor for Maker {
        fn processed(&self, source: Arc<dyn RecordingSource>, pyramid: Option<Arc<Pyramid>>, _channels: &[usize], ops: &[String]) -> StreamResult<Arc<dyn SignalBackend>> {
            let op = ops.first().cloned().unwrap_or_default();
            self.made.lock().unwrap().push(op.clone());
            match op.as_str() {
                "slow" => {
                    self.slow_started.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(1500));
                }
                "bad" => return Err(StreamError::protocol("an op that is not a pipeline node")),
                _ => {}
            }
            Ok(Arc::new(dspk_view::StoredSignal::new(source, pyramid)))
        }
    }

    fn chain(op: &str) -> wire::Processing {
        wire::Processing { channels: vec![], ops: vec![op.into()] }
    }

    /// A chain being made holds up only views of that chain: another chain is answered at once,
    /// the same chain is made once, and a chain that failed is tried again rather than kept.
    #[test]
    fn a_slow_chain_holds_up_only_its_own_views() {
        let recording = Arc::new(dspk_core::MemoryRecording::new("r", vec![0.0; 4 * 100], 4, 1000.0).unwrap());
        let maker = Arc::new(Maker::default());
        let open = Arc::new(OpenRecording::new(recording, None).with_processor(maker.clone()));

        let o = open.clone();
        let first = std::thread::spawn(move || o.processed(&chain("slow")).is_ok());
        while maker.slow_started.load(Ordering::SeqCst) == 0 {
            std::thread::yield_now();
        }
        let t0 = Instant::now();
        assert!(open.processed(&chain("fast")).is_ok());
        assert!(t0.elapsed() < Duration::from_millis(500), "another chain waited for the slow one: {:?}", t0.elapsed());
        // The same chain waits for the make under way instead of making it again
        assert!(open.processed(&chain("slow")).is_ok());
        assert!(first.join().unwrap());
        assert_eq!(maker.made.lock().unwrap().iter().filter(|op| *op == "slow").count(), 1, "made once");

        assert!(matches!(open.processed(&chain("bad")), Err(StreamError::Server(_))));
        assert!(open.processed(&chain("bad")).is_err());
        assert_eq!(maker.made.lock().unwrap().iter().filter(|op| *op == "bad").count(), 2, "a failed chain is tried again");
        assert_eq!(open.processed.lock().unwrap().len(), 2, "failed chains are not kept");
    }
}

