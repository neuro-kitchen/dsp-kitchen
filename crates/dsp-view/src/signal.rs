//! A signal as a viewer sees it ([`SignalBackend`]), wherever it lives: [`LocalSignal`] is a
//! recording in this process with its pyramid and background builder; a remote session
//! (dsp-stream) answers the same [`View`]s over the network. A viewer (an app) only asks for views
//! and draws the [`Envelope`]s it gets.

use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use dsp_core::{DspResult, RecordingInfo, RecordingSource};

use crate::pyramid::{pyramid_path, OnProgress, Progress, Pyramid, PyramidBuilder, PyramidIdentity, FILE_BASE, MEMORY_BASE};
use crate::view::{Envelope, View, RAW_BLOCK_VALUES};

/// Recordings with at least this many stored bytes keep their pyramid in a file next to them
/// (built once, nearest the view first, and reused by every later open); smaller ones keep it in
/// memory for the session.
pub const FILE_PYRAMID_MIN_BYTES: u64 = 64 << 20;

/// What a viewer needs from a signal. Calls block; viewers call them from worker threads.
pub trait SignalBackend: Send + Sync {
    /// The recording's description (channels, samples, exact rate and start, units).
    fn info(&self) -> &RecordingInfo;

    /// The contents of `view` into `out` (its buffer is reused). Zoomed-out columns not built yet
    /// are NaN and `complete` is false: ask again after [`Self::watch`] reports progress.
    fn view(&self, view: &View, out: &mut Envelope) -> DspResult<()>;

    /// Samples `samples` of `channels`, channel-major (e.g. the value under the cursor).
    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()>;

    /// The viewer looks at sample `sample`: zoomed-out data there is prepared first.
    fn focus(&self, sample: u64);

    /// Prepares the zoomed-out data of samples `start..end` now, blocking until it is ready (for
    /// a single exact image, e.g. a snapshot; viewers use [`Self::focus`]).
    fn prepare(&self, start: u64, end: u64) -> DspResult<()>;

    /// Samples whose zoomed-out data is ready, of all (`done == total` once complete).
    fn progress(&self) -> Progress;

    /// Calls `on_progress` whenever more zoomed-out data is ready.
    fn watch(&self, on_progress: OnProgress);
}

/// A recording in this process, its min/max [`Pyramid`] (a file for recordings of at least
/// [`FILE_PYRAMID_MIN_BYTES`], else memory) and the builder that fills it in the background,
/// nearest the focus first. The recording is read once to build the pyramid, while views already
/// draw from it.
pub struct LocalSignal {
    source: Arc<dyn RecordingSource>,
    pyramid: Arc<Pyramid>,
    builder: PyramidBuilder,
    cancel: Arc<AtomicBool>,
    /// Samples built so far.
    done: Arc<AtomicU64>,
    listeners: Arc<Mutex<Vec<OnProgress>>>,
}

impl LocalSignal {
    /// `source`, with the pyramid of source `id` of the recording file `path` when given: the
    /// complete file next to it, else (large recordings) a new file there, else memory.
    pub fn open(source: Arc<dyn RecordingSource>, recording: Option<(&Path, &str)>) -> DspResult<Self> {
        let pyramid = Self::choose_pyramid(source.as_ref(), recording)?;
        let total = source.info().samples;
        let done = if pyramid.is_complete() { total } else { 0 };
        Ok(Self {
            source,
            pyramid: Arc::new(pyramid),
            builder: PyramidBuilder::default(),
            cancel: Arc::new(AtomicBool::new(false)),
            done: Arc::new(AtomicU64::new(done)),
            listeners: Arc::new(Mutex::new(Vec::new())),
        })
    }

    fn choose_pyramid(source: &dyn RecordingSource, recording: Option<(&Path, &str)>) -> DspResult<Pyramid> {
        let Some((path, id)) = recording else { return Pyramid::in_memory(source, MEMORY_BASE) };
        let identity = PyramidIdentity::of(path, id, source)?;
        let file = pyramid_path(path, id);
        if let Some(complete) = Pyramid::open_file(&file, &identity, FILE_BASE)? {
            return Ok(complete);
        }
        if source.info().data_bytes() < FILE_PYRAMID_MIN_BYTES {
            return Pyramid::in_memory(source, MEMORY_BASE);
        }
        match Pyramid::create_file(&file, &identity, FILE_BASE) {
            Ok(pyramid) => Ok(pyramid),
            Err(e) => {
                tracing::warn!("cannot write {}: {e}; keeping the min/max pyramid in memory", file.display());
                Pyramid::in_memory(source, MEMORY_BASE)
            }
        }
    }

    /// The recording.
    pub fn source(&self) -> &Arc<dyn RecordingSource> {
        &self.source
    }

    /// The pyramid (complete or filling).
    pub fn pyramid(&self) -> &Arc<Pyramid> {
        &self.pyramid
    }
}

impl SignalBackend for LocalSignal {
    fn info(&self) -> &RecordingInfo {
        self.source.info()
    }

    fn view(&self, view: &View, out: &mut Envelope) -> DspResult<()> {
        view.read_into(self.source.as_ref(), Some(&self.pyramid), out)
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        self.source.read(channels, samples, out)
    }

    fn focus(&self, sample: u64) {
        if self.pyramid.is_complete() {
            return;
        }
        let (done, listeners) = (self.done.clone(), self.listeners.clone());
        let on_progress: OnProgress = Arc::new(move |p: Progress| {
            done.store(p.done, Ordering::Relaxed);
            for listener in listeners.lock().expect("listeners lock").iter() {
                listener(p.clone());
            }
        });
        // Starts once; later calls only move the focus
        self.builder.run(self.source.clone(), self.pyramid.clone(), sample, self.cancel.clone(), on_progress);
    }

    fn prepare(&self, start: u64, end: u64) -> DspResult<()> {
        let end = end.min(self.source.info().samples);
        if start < end {
            self.pyramid.fill(self.source.as_ref(), start, end, RAW_BLOCK_VALUES, || false)?;
        }
        Ok(())
    }

    fn progress(&self) -> Progress {
        Progress { filled: 0..0, done: self.done.load(Ordering::Relaxed), total: self.source.info().samples }
    }

    fn watch(&self, on_progress: OnProgress) {
        self.listeners.lock().expect("listeners lock").push(on_progress);
    }
}

impl Drop for LocalSignal {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::MemoryRecording;

    #[test]
    fn small_recordings_keep_the_pyramid_in_memory_and_fill_it() {
        let data: Vec<f32> = (0..2 * 50_000).map(|i| ((i * 7919) % 101) as f32).collect();
        let source: Arc<dyn RecordingSource> = Arc::new(MemoryRecording::new("s", data, 2, 1000.0).unwrap());
        let signal = LocalSignal::open(source, None).unwrap();
        assert!(signal.pyramid().path().is_none());
        assert_eq!(signal.progress().done, 0);
        let (tx, rx) = std::sync::mpsc::channel();
        signal.watch(Arc::new(move |p: Progress| {
            if p.done == p.total {
                let _ = tx.send(());
            }
        }));
        signal.focus(25_000);
        rx.recv_timeout(std::time::Duration::from_secs(30)).expect("pyramid filled");
        assert!(signal.pyramid().is_complete());
        let mut out = Envelope::Samples(Vec::new());
        signal.view(&View { channels: vec![1], start: 0, end: 50_000, width: 10 }, &mut out).unwrap();
        assert!(matches!(out, Envelope::Columns { complete: true, .. }));
    }
}
