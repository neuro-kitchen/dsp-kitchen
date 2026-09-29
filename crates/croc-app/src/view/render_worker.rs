//! Background render thread.
//!
//! Keeps decimation and rasterization off the UI thread. Requests are queued over a channel;
//! for every view the worker renders only the newest request and drops any it has fallen
//! behind on, so a burst of input costs at most one frame per view.

use std::collections::BTreeMap;
use std::sync::mpsc::{self, Sender};
use std::thread;
use std::time::{Duration, Instant};

use slint::{Rgba8Pixel, SharedPixelBuffer};

use super::renderer::{RenderRequest, WaveformRenderer};

pub struct RenderWorker {
    tx: Sender<RenderRequest>,
}

impl RenderWorker {
    /// Spawns the worker. `on_frame` runs on the worker thread with the finished frame, its
    /// render time, and the request it answers; it must forward the buffer to the UI thread
    /// (e.g. `invoke_from_event_loop`).
    pub fn spawn<F>(on_frame: F) -> Self
    where
        F: Fn(SharedPixelBuffer<Rgba8Pixel>, Duration, &RenderRequest) + Send + 'static,
    {
        let (tx, rx) = mpsc::channel::<RenderRequest>();
        thread::Builder::new()
            .name("croc-render".into())
            .spawn(move || {
                let mut renderer = WaveformRenderer::default();
                // Exits when the sender (owned by the UI) is dropped
                while let Ok(first) = rx.recv() {
                    // Latest request per view wins
                    let mut pending = BTreeMap::new();
                    pending.insert(first.view_id, first);
                    while let Ok(newer) = rx.try_recv() {
                        pending.insert(newer.view_id, newer);
                    }
                    for req in pending.into_values() {
                        let t0 = Instant::now();
                        let frame = renderer.render(&req);
                        on_frame(frame, t0.elapsed(), &req);
                    }
                }
            })
            .expect("failed to spawn render thread");
        Self { tx }
    }

    pub fn request(&self, req: RenderRequest) {
        // A send error means the worker died; the UI keeps its last frames
        let _ = self.tx.send(req);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::model::{Dataset, SpikeEventStore};
    use crate::view::ViewMode;

    fn req(view_id: u32, width: u32, ds: &Arc<Dataset>, events: &Arc<SpikeEventStore>) -> RenderRequest {
        RenderRequest {
            view_id,
            source: ds.clone(),
            events: events.clone(),
            width,
            height: 50,
            scale: 1.0,
            mode: ViewMode::Traces,
            channels: vec![0, 1],
            window_start_sec: 0.0,
            window_sec: 0.1,
            amplitude_scale: 1.0,
            grid_times: Vec::new(),
            scale_bar_uv: 0.0,
        }
    }

    #[test]
    fn test_worker_renders_latest_request_per_view() {
        let ds = Arc::new(Dataset::generate_synthetic(4, 10_000.0, 0.5));
        let events = Arc::new(SpikeEventStore::detect(ds.as_ref()));
        let (done_tx, done_rx) = mpsc::channel();
        let worker = RenderWorker::spawn(move |frame, _, req| {
            done_tx.send((req.view_id, frame.width(), req.width)).unwrap();
        });

        // Bursts for two views: frames may be skipped, but each view's newest is rendered
        for w in [100u32, 200, 300, 400] {
            worker.request(req(1, w, &ds, &events));
            worker.request(req(2, w + 1, &ds, &events));
        }

        let mut last = BTreeMap::new();
        while let Ok((view, frame_w, req_w)) = done_rx.recv_timeout(Duration::from_secs(5)) {
            assert_eq!(frame_w, req_w);
            last.insert(view, frame_w);
            if last.get(&1) == Some(&400) && last.get(&2) == Some(&401) {
                break;
            }
        }
        assert_eq!(last.get(&1), Some(&400));
        assert_eq!(last.get(&2), Some(&401));
    }
}
