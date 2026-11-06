//! A recording on a server as a viewer sees it: [`RemoteSignal`] is a [`SignalBackend`] whose
//! views are answered by the server (from its own pyramid), so a viewer draws a remote recording
//! as it draws a local one, and nothing is copied whole. Calls block (as the trait says: viewers
//! call them from worker threads) on the session's async runtime.

use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dspk_core::{DspError, DspResult, RecordingInfo};
use dspk_view::{Envelope, OnProgress, Progress, SignalBackend, View};

use super::client::Session;
use super::server::{MAX_VIEW_COLUMNS, MAX_VIEW_VALUES};
use crate::error::StreamError;

/// How often an incomplete remote pyramid is asked about again.
const PYRAMID_POLL: Duration = Duration::from_secs(1);
/// Columns of the small view that asks whether the server's pyramid is complete.
const PROBE_COLUMNS: usize = 64;

fn dsp(e: StreamError) -> DspError {
    DspError::Io(format!("remote: {e}"))
}

/// The recording a [`Session`] has open, behind [`SignalBackend`].
pub struct RemoteSignal {
    handle: tokio::runtime::Handle,
    session: Arc<Session>,
    info: RecordingInfo,
    /// Whether the server's pyramid answered complete views (until then views may hold NaN).
    complete: Arc<AtomicBool>,
    listeners: Arc<Mutex<Vec<OnProgress>>>,
    polling: Arc<AtomicBool>,
}

impl RemoteSignal {
    /// The recording `session` has open; calls run on `handle` (the runtime that drives the
    /// session's connection).
    pub fn new(session: Arc<Session>, handle: tokio::runtime::Handle) -> Result<Self, StreamError> {
        let info = session.recording().cloned().ok_or_else(|| StreamError::protocol("no recording is open in this session"))?;
        Ok(Self {
            handle,
            session,
            info,
            complete: Arc::new(AtomicBool::new(false)),
            listeners: Arc::new(Mutex::new(Vec::new())),
            polling: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Samples of `channels` channels one request reads: a sample view of c channels and n
    /// samples is c × n values and n columns, both under the server's limits.
    fn samples_per_request(channels: usize) -> u64 {
        (MAX_VIEW_VALUES / channels.max(1) as u64).min(u64::from(MAX_VIEW_COLUMNS)).max(1)
    }

    /// Asks the server again, every [`PYRAMID_POLL`], until its pyramid answers complete views,
    /// telling the listeners each time (viewers then ask for their views again).
    fn poll_until_complete(&self) {
        if self.polling.swap(true, Ordering::Relaxed) {
            return;
        }
        let (handle, session, complete, listeners, polling) = (self.handle.clone(), self.session.clone(), self.complete.clone(), self.listeners.clone(), self.polling.clone());
        let total = self.info.samples;
        std::thread::spawn(move || {
            let probe = View { channels: vec![0], start: 0, end: total, width: PROBE_COLUMNS };
            while !complete.load(Ordering::Relaxed) {
                std::thread::sleep(PYRAMID_POLL);
                let done = match handle.block_on(session.view(&probe)) {
                    Ok(Envelope::Columns { complete: true, .. }) | Ok(Envelope::Samples(_)) => {
                        complete.store(true, Ordering::Relaxed);
                        total
                    }
                    Ok(_) => 0,
                    // The session is gone: stop asking
                    Err(_) => break,
                };
                for listener in listeners.lock().unwrap_or_else(|e| e.into_inner()).iter() {
                    listener(Progress { filled: 0..0, done, total });
                }
            }
            polling.store(false, Ordering::Relaxed);
        });
    }
}

impl SignalBackend for RemoteSignal {
    fn info(&self) -> &RecordingInfo {
        &self.info
    }

    fn view(&self, view: &View, out: &mut Envelope) -> DspResult<()> {
        let envelope = self.handle.block_on(self.session.view(view)).map_err(dsp)?;
        match &envelope {
            Envelope::Columns { complete: false, .. } => {
                self.complete.store(false, Ordering::Relaxed);
                self.poll_until_complete();
            }
            Envelope::Columns { complete: true, .. } => self.complete.store(true, Ordering::Relaxed),
            Envelope::Samples(_) => {}
        }
        *out = envelope;
        Ok(())
    }

    /// Samples of `channels`, channel-major, fetched as sample views in pieces the server
    /// answers (its limits on columns and values).
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
            let values = match self.handle.block_on(self.session.view(&view)).map_err(dsp)? {
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

    /// The server builds its own pyramid: nothing to steer from here.
    fn focus(&self, _sample: u64) {}

    fn prepare(&self, _start: u64, _end: u64) -> DspResult<()> {
        Ok(())
    }

    fn progress(&self) -> Progress {
        let total = self.info.samples;
        Progress { filled: 0..0, done: if self.complete.load(Ordering::Relaxed) { total } else { 0 }, total }
    }

    fn watch(&self, on_progress: OnProgress) {
        self.listeners.lock().unwrap_or_else(|e| e.into_inner()).push(on_progress);
    }
}
