//! Background render thread shared by every module.
//!
//! Keeps rasterization off the UI thread. Jobs are closures keyed by (module, view); for each
//! key the worker renders only the newest job and drops any it has fallen behind on, so a
//! burst of input costs at most one frame per view.

use std::collections::BTreeMap;
use std::sync::mpsc::{self, Sender};
use std::thread;
use std::time::{Duration, Instant};

use slint::{Rgba8Pixel, SharedPixelBuffer};

/// (module id, view id)
pub type RenderKey = (u8, u32);

pub type Frame = SharedPixelBuffer<Rgba8Pixel>;

pub struct RenderJob {
    pub key: RenderKey,
    /// Samples per pixel (time plots), for the status readout.
    pub samples_per_px: Option<f64>,
    pub render: Box<dyn FnOnce() -> Frame + Send>,
}

/// A finished frame and what produced it.
pub struct FrameInfo {
    pub key: RenderKey,
    pub elapsed: Duration,
    pub samples_per_px: Option<f64>,
}

pub struct RenderWorker {
    tx: Sender<RenderJob>,
}

impl RenderWorker {
    /// Spawns the worker. `on_frame` runs on the worker thread and must forward the frame to
    /// the UI thread (e.g. `invoke_from_event_loop`).
    pub fn spawn<F>(on_frame: F) -> Self
    where
        F: Fn(Frame, FrameInfo) + Send + 'static,
    {
        let (tx, rx) = mpsc::channel::<RenderJob>();
        thread::Builder::new()
            .name("croc-render".into())
            .spawn(move || {
                // Exits when the sender (owned by the UI) is dropped
                while let Ok(first) = rx.recv() {
                    let mut pending = BTreeMap::new();
                    pending.insert(first.key, first);
                    while let Ok(newer) = rx.try_recv() {
                        pending.insert(newer.key, newer);
                    }
                    for job in pending.into_values() {
                        let t0 = Instant::now();
                        let frame = (job.render)();
                        on_frame(frame, FrameInfo { key: job.key, elapsed: t0.elapsed(), samples_per_px: job.samples_per_px });
                    }
                }
            })
            .expect("failed to spawn render thread");
        Self { tx }
    }

    pub fn request(&self, job: RenderJob) {
        // A send error means the worker died; the UI keeps its last frames
        let _ = self.tx.send(job);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(key: RenderKey, width: u32) -> RenderJob {
        RenderJob { key, samples_per_px: None, render: Box::new(move || Frame::new(width, 1)) }
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
}
