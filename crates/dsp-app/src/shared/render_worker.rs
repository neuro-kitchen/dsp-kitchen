//! Background render thread shared by every module.
//!
//! Keeps rasterization off the UI thread. Jobs are closures keyed by (module, view); for each
//! key the worker renders only the newest job and drops any it has fallen behind on, so a
//! burst of input costs at most one frame per view. A new job also flags the one in progress
//! for its key as cancelled; jobs that can stop early (filling the min/max summary) give up.
//! Long jobs can show preview frames while they work.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use slint::{Rgba8Pixel, SharedPixelBuffer};

/// (module id, view id)
pub type RenderKey = (u8, u32);

pub type Frame = SharedPixelBuffer<Rgba8Pixel>;

/// A rendered frame plus the amplitude scale it was drawn with (auto-scaled time plots).
pub struct Rendered {
    pub frame: Frame,
    pub scale: Option<f32>,
}

impl From<Frame> for Rendered {
    fn from(frame: Frame) -> Self {
        Self { frame, scale: None }
    }
}

pub struct RenderJob {
    pub key: RenderKey,
    /// Samples per pixel (time plots), for the status readout.
    pub samples_per_px: Option<f64>,
    /// Renders the frame, or `None` if it stopped because a newer job exists.
    pub render: Box<dyn FnOnce(&RenderContext) -> Option<Rendered> + Send>,
}

/// What a running job can use: its cancel flag and a way to show preview frames.
pub struct RenderContext<'a> {
    /// Set when a newer job for the same key was requested.
    pub cancel: &'a AtomicBool,
    /// Shows an intermediate frame (delivered like a finished one).
    pub preview: &'a dyn Fn(Rendered),
}

/// A finished frame and what produced it.
pub struct FrameInfo {
    pub key: RenderKey,
    pub elapsed: Duration,
    pub samples_per_px: Option<f64>,
    /// Amplitude scale chosen by the renderer (units that fill a lane at gain 1).
    pub scale: Option<f32>,
}

pub struct RenderWorker {
    tx: Sender<(RenderJob, Arc<AtomicBool>)>,
    /// Cancel flag of the latest job per key (UI thread only).
    latest: RefCell<HashMap<RenderKey, Arc<AtomicBool>>>,
}

impl RenderWorker {
    /// Spawns the worker. `on_frame` runs on the worker thread and must forward the frame to
    /// the UI thread (e.g. `invoke_from_event_loop`).
    pub fn spawn<F>(on_frame: F) -> Self
    where
        F: Fn(Frame, FrameInfo) + Send + 'static,
    {
        let (tx, rx) = mpsc::channel::<(RenderJob, Arc<AtomicBool>)>();
        thread::Builder::new()
            .name("dsp-app-render".into())
            .spawn(move || {
                // Exits when the sender (owned by the UI) is dropped
                while let Ok(first) = rx.recv() {
                    let mut pending = BTreeMap::new();
                    pending.insert(first.0.key, first);
                    while let Ok(newer) = rx.try_recv() {
                        pending.insert(newer.0.key, newer);
                    }
                    for (job, cancel) in pending.into_values() {
                        if cancel.load(Ordering::Relaxed) {
                            continue;
                        }
                        let t0 = Instant::now();
                        let (key, samples_per_px) = (job.key, job.samples_per_px);
                        let deliver = |out: Rendered| {
                            on_frame(out.frame, FrameInfo { key, elapsed: t0.elapsed(), samples_per_px, scale: out.scale });
                        };
                        let Some(out) = (job.render)(&RenderContext { cancel: &cancel, preview: &deliver }) else { continue };
                        deliver(out);
                    }
                }
            })
            .expect("failed to spawn render thread");
        Self { tx, latest: RefCell::default() }
    }

    /// Queues `job`, cancelling the previous job of its key.
    pub fn request(&self, job: RenderJob) {
        let cancel = Arc::new(AtomicBool::new(false));
        if let Some(previous) = self.latest.borrow_mut().insert(job.key, cancel.clone()) {
            previous.store(true, Ordering::Relaxed);
        }
        // A send error means the worker died; the UI keeps its last frames
        let _ = self.tx.send((job, cancel));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(key: RenderKey, width: u32) -> RenderJob {
        RenderJob { key, samples_per_px: None, render: Box::new(move |_| Some(Frame::new(width, 1).into())) }
    }

    #[test]
    fn test_worker_renders_latest_job_per_key() {
        let (done_tx, done_rx) = mpsc::channel();
        let worker = RenderWorker::spawn(move |frame, info| {
            done_tx.send((info.key, frame.width())).unwrap();
        });

        // Bursts for two views of two modules: frames may be skipped, the newest always renders
        for w in [100u32, 200, 300, 400] {
            worker.request(job((0, 1), w));
            worker.request(job((1, 1), w + 1));
        }

        let mut last = BTreeMap::new();
        while let Ok((key, w)) = done_rx.recv_timeout(Duration::from_secs(5)) {
            last.insert(key, w);
            if last.get(&(0, 1)) == Some(&400) && last.get(&(1, 1)) == Some(&401) {
                break;
            }
        }
        assert_eq!(last.get(&(0, 1)), Some(&400));
        assert_eq!(last.get(&(1, 1)), Some(&401));
    }

    #[test]
    fn test_newer_job_cancels_the_one_in_progress() {
        let (done_tx, done_rx) = mpsc::channel();
        let worker = RenderWorker::spawn(move |frame, info| done_tx.send((info.key, frame.width())).unwrap());
        let (started_tx, started_rx) = mpsc::channel();
        // A slow job that stops when cancelled
        worker.request(RenderJob {
            key: (0, 1),
            samples_per_px: None,
            render: Box::new(move |ctx| {
                started_tx.send(()).unwrap();
                while !ctx.cancel.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(1));
                }
                None
            }),
        });
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        worker.request(job((0, 1), 7));
        assert_eq!(done_rx.recv_timeout(Duration::from_secs(5)).unwrap(), ((0, 1), 7), "only the newer frame is delivered");
    }
}
