//! Hover readouts, read off the UI thread through the signal's backend.
//!
//! Requests that arrive while a read is running are collapsed to the newest. Each request carries
//! its `reply`, so the readout goes back to the view that asked. Reading one sample of a chunked,
//! compressed source decodes its chunk once: opened sources keep decoded chunks in their chunk
//! cache (`dsp_io::CachedRecording`), so moving along a trace does not decode a chunk again.

use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::thread;

use crate::engine::data::Dataset;

use super::view::fmt_amount;

/// Receives `(seq, text)` on the reader thread (forward it to the UI).
pub type HoverReply = Arc<dyn Fn(u64, String) + Send + Sync>;

pub struct HoverRequest {
    /// Bumped per pointer move; the UI keeps only the answer to its latest request.
    pub seq: u64,
    pub dataset: Arc<Dataset>,
    pub channel: usize,
    pub sample: u64,
    /// Text before the value (`Ch 3  ·  1.2345 s  ·  `).
    pub prefix: String,
    pub unit: String,
    pub reply: HoverReply,
}

/// The value under the cursor, or `None` when it cannot be read.
fn read(req: &HoverRequest) -> Option<f32> {
    let mut value = [0.0f32];
    req.dataset.signal().read(&[req.channel], req.sample..req.sample + 1, &mut value).ok()?;
    Some(value[0])
}

pub struct HoverReader {
    tx: Sender<HoverRequest>,
}

impl HoverReader {
    /// Spawns the reader thread.
    pub fn spawn() -> Self {
        let (tx, rx) = mpsc::channel::<HoverRequest>();
        thread::Builder::new()
            .name("dsp-app-hover".into())
            .spawn(move || {
                // Exits when the sender (owned by the UI) is dropped
                while let Ok(mut req) = rx.recv() {
                    while let Ok(newer) = rx.try_recv() {
                        req = newer;
                    }
                    let text = match read(&req) {
                        Some(v) => format!("{}{} {}", req.prefix, fmt_amount(v), req.unit),
                        None => format!("{}read failed", req.prefix),
                    };
                    (req.reply)(req.seq, text);
                }
            })
            .expect("failed to spawn hover thread");
        Self { tx }
    }

    pub fn request(&self, req: HoverRequest) {
        // A send error means the reader died; readouts stop updating
        let _ = self.tx.send(req);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_reader_answers_with_the_sample_value() {
        let ds = Arc::new(Dataset::from_samples("h", (0..20).map(|v| v as f32).collect(), 2, 1000.0));
        let (done_tx, done_rx) = mpsc::channel();
        let done_tx = std::sync::Mutex::new(done_tx);
        let reply: HoverReply = Arc::new(move |seq, text| done_tx.lock().unwrap().send((seq, text)).unwrap());
        let reader = HoverReader::spawn();
        let req = |seq, channel, sample| HoverRequest { seq, dataset: ds.clone(), channel, sample, prefix: "x · ".into(), unit: "µV".into(), reply: reply.clone() };
        reader.request(req(1, 1, 3));
        assert_eq!(done_rx.recv_timeout(Duration::from_secs(5)).unwrap(), (1, format!("x · {} µV", fmt_amount(13.0))));
        reader.request(req(2, 0, 4));
        assert_eq!(done_rx.recv_timeout(Duration::from_secs(5)).unwrap().1, format!("x · {} µV", fmt_amount(4.0)));
    }
}
