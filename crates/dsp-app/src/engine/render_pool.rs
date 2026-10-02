//! Background render threads shared by every view.
//!
//! Keeps rasterization off the UI thread. Jobs are closures keyed by view; for each key only the
//! newest job waiting is kept (a burst of input costs at most one frame per view), and a new job
//! flags the one in progress for its key as cancelled (jobs that can stop early, such as filling
//! the min/max summary, give up). One key renders on one thread at a time, so its frames arrive in
//! order; different keys render in parallel. Long jobs can show preview frames while they work.
//! Each job carries its own `deliver` callback: a view receives only its own frames.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use super::canvas::Frame;

/// Identifies what a job draws (a view); newer jobs replace older ones with the same key.
pub type RenderKey = u64;

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

/// What a running job can use: its cancel flag and a way to show preview frames.
pub struct RenderContext<'a> {
    /// Set when a newer job for the same key was requested.
    pub cancel: &'a AtomicBool,
    /// Shows an intermediate frame (delivered like a finished one, with `preview` set).
    pub preview: &'a dyn Fn(Rendered),
}

/// A delivered frame and what produced it.
pub struct FrameInfo {
    /// Time from the job's start to this frame.
    pub elapsed: Duration,
    /// An intermediate frame of a job still working.
    pub preview: bool,
}

pub type Deliver = Arc<dyn Fn(Rendered, FrameInfo) + Send + Sync>;

/// Draws a frame, or gives up (`None`) because a newer job of its key exists.
pub type RenderFn = Box<dyn FnOnce(&RenderContext) -> Option<Rendered> + Send>;

pub struct RenderJob {
    pub key: RenderKey,
    pub render: RenderFn,
    /// Receives the frames (runs on the render thread: forward them to the UI).
    pub deliver: Deliver,
}

struct Queued {
    job: RenderJob,
    cancel: Arc<AtomicBool>,
}

#[derive(Default)]
struct Queue {
    /// Newest waiting job per key.
    waiting: HashMap<RenderKey, Queued>,
    /// Keys with a waiting job, oldest request first.
    order: VecDeque<RenderKey>,
    /// Keys a thread is rendering.
    busy: HashSet<RenderKey>,
    closed: bool,
}

impl Queue {
    /// The oldest waiting job whose key is not being rendered.
    fn take(&mut self) -> Option<(RenderKey, Queued)> {
        let at = self.order.iter().position(|k| !self.busy.contains(k))?;
        let key = self.order.remove(at)?;
        let queued = self.waiting.remove(&key)?;
        self.busy.insert(key);
        Some((key, queued))
    }
}

struct Shared {
    queue: Mutex<Queue>,
    ready: Condvar,
}

pub struct RenderPool {
    shared: Arc<Shared>,
    /// Cancel flag of the latest job per key.
    latest: Mutex<HashMap<RenderKey, Arc<AtomicBool>>>,
}

impl RenderPool {
    /// Spawns `threads` render threads (at least one).
    pub fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared { queue: Mutex::new(Queue::default()), ready: Condvar::new() });
        for i in 0..threads.max(1) {
            let shared = shared.clone();
            thread::Builder::new().name(format!("dsp-app-render-{i}")).spawn(move || work(&shared)).expect("failed to spawn a render thread");
        }
        Self { shared, latest: Mutex::default() }
    }

    /// Threads for this machine: half the cores, 2 to 4.
    pub fn default_threads() -> usize {
        thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 4))
    }

    /// Queues `job`, replacing a waiting job of its key and cancelling the one in progress.
    pub fn request(&self, job: RenderJob) {
        let cancel = Arc::new(AtomicBool::new(false));
        if let Some(previous) = self.latest.lock().expect("render pool lock").insert(job.key, cancel.clone()) {
            previous.store(true, Ordering::Relaxed);
        }
        let mut q = self.shared.queue.lock().expect("render pool lock");
        let key = job.key;
        if q.waiting.insert(key, Queued { job, cancel }).is_none() {
            q.order.push_back(key);
        }
        drop(q);
        self.shared.ready.notify_one();
    }

    /// Cancels and forgets the jobs of `key` (a view that closed).
    pub fn forget(&self, key: RenderKey) {
        if let Some(flag) = self.latest.lock().expect("render pool lock").remove(&key) {
            flag.store(true, Ordering::Relaxed);
        }
        let mut q = self.shared.queue.lock().expect("render pool lock");
        if q.waiting.remove(&key).is_some() {
            q.order.retain(|k| *k != key);
        }
    }
}

impl Drop for RenderPool {
    fn drop(&mut self) {
        self.shared.queue.lock().expect("render pool lock").closed = true;
        self.shared.ready.notify_all();
    }
}

fn work(shared: &Shared) {
    loop {
        let (key, Queued { job, cancel }) = {
            let mut q = shared.queue.lock().expect("render pool lock");
            loop {
                if q.closed {
                    return;
                }
                if let Some(next) = q.take() {
                    break next;
                }
                q = shared.ready.wait(q).expect("render pool lock");
            }
        };
        if !cancel.load(Ordering::Relaxed) {
            let t0 = Instant::now();
            let deliver = job.deliver.clone();
            let preview = |out: Rendered| deliver(out, FrameInfo { elapsed: t0.elapsed(), preview: true });
            if let Some(out) = (job.render)(&RenderContext { cancel: &cancel, preview: &preview }) {
                (job.deliver)(out, FrameInfo { elapsed: t0.elapsed(), preview: false });
            }
        }
        let mut q = shared.queue.lock().expect("render pool lock");
        q.busy.remove(&key);
        // A job of this key may have been waiting for this thread
        let more = q.waiting.contains_key(&key);
        drop(q);
        if more {
            shared.ready.notify_one();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::mpsc;

    fn job(key: RenderKey, width: u32, tx: mpsc::Sender<(RenderKey, u32)>) -> RenderJob {
        let tx = Mutex::new(tx);
        RenderJob {
            key,
            render: Box::new(move |_| Some(Frame::new(width, 1).into())),
            deliver: Arc::new(move |out, _| tx.lock().unwrap().send((key, out.frame.width)).unwrap()),
        }
    }

    #[test]
    fn test_pool_renders_the_latest_job_per_key_in_order() {
        let (tx, rx) = mpsc::channel();
        let pool = RenderPool::new(3);
        // Bursts for two views: frames may be skipped, never reordered; the newest always renders
        for w in [100u32, 200, 300, 400] {
            pool.request(job(1, w, tx.clone()));
            pool.request(job(2, w + 1, tx.clone()));
        }
        let mut last = BTreeMap::new();
        while let Ok((key, w)) = rx.recv_timeout(Duration::from_secs(5)) {
            if let Some(prev) = last.insert(key, w) {
                assert!(w > prev, "frames of one key arrive in request order");
            }
            if last.get(&1) == Some(&400) && last.get(&2) == Some(&401) {
                break;
            }
        }
        assert_eq!(last.get(&1), Some(&400));
        assert_eq!(last.get(&2), Some(&401));
    }

    #[test]
    fn test_newer_job_cancels_the_one_in_progress() {
        let (tx, rx) = mpsc::channel();
        let pool = RenderPool::new(2);
        let (started_tx, started_rx) = mpsc::channel();
        let slow_tx = Mutex::new(tx.clone());
        // A slow job that stops when cancelled
        pool.request(RenderJob {
            key: 1,
            render: Box::new(move |ctx| {
                started_tx.send(()).unwrap();
                while !ctx.cancel.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(1));
                }
                None
            }),
            deliver: Arc::new(move |out, _| slow_tx.lock().unwrap().send((1, out.frame.width)).unwrap()),
        });
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        pool.request(job(1, 7, tx));
        assert_eq!(rx.recv_timeout(Duration::from_secs(5)).unwrap(), (1, 7), "only the newer frame is delivered");
    }

    #[test]
    fn test_forget_drops_waiting_jobs() {
        let (tx, rx) = mpsc::channel();
        let pool = RenderPool::new(1);
        let (gate_tx, gate_rx) = mpsc::channel::<()>();
        let gate = Mutex::new(gate_rx);
        // Occupy the only thread, queue a job for key 2, forget it, release the thread
        pool.request(RenderJob {
            key: 1,
            render: Box::new(move |_| {
                gate.lock().unwrap().recv().unwrap();
                Some(Frame::new(1, 1).into())
            }),
            deliver: Arc::new(|_, _| {}),
        });
        pool.request(job(2, 5, tx.clone()));
        pool.forget(2);
        pool.request(job(3, 9, tx));
        gate_tx.send(()).unwrap();
        assert_eq!(rx.recv_timeout(Duration::from_secs(5)).unwrap(), (3, 9));
        assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
    }
}
