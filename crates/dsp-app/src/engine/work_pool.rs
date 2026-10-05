//! Background threads shared by every view: renders and curation computations.
//!
//! Keeps work off the UI thread. A job is a closure under a [`JobKey`] (a view and one of its
//! kinds of work); for each key only the newest job waiting is kept (a burst of input costs at
//! most one run per key), and a new job flags the one in progress for its key as cancelled (jobs
//! that can stop early check the flag; any result they deliver after it is stale). One key runs on
//! one thread at a time, so its results arrive in order; different keys run in parallel. Each job
//! delivers its own result, so a view receives only its own.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

/// What a job is for: the view (or other owner) and which of its kinds of work. A newer job with
/// the same key replaces an older one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobKey {
    pub owner: u64,
    pub slot: u32,
}

impl JobKey {
    pub const fn new(owner: u64, slot: u32) -> Self {
        Self { owner, slot }
    }
}

/// A job: runs on a pool thread, given its cancel flag (set once a newer job for its key exists).
pub type Work = Box<dyn FnOnce(&AtomicBool) + Send>;

struct Queued {
    work: Work,
    cancel: Arc<AtomicBool>,
}

#[derive(Default)]
struct Queue {
    /// Newest waiting job per key.
    waiting: HashMap<JobKey, Queued>,
    /// Keys with a waiting job, oldest request first.
    order: VecDeque<JobKey>,
    /// Keys a thread is running.
    busy: HashSet<JobKey>,
    closed: bool,
}

impl Queue {
    /// The oldest waiting job whose key is not running.
    fn take(&mut self) -> Option<(JobKey, Queued)> {
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

pub struct WorkPool {
    shared: Arc<Shared>,
    /// Cancel flag of the latest job per key.
    latest: Mutex<HashMap<JobKey, Arc<AtomicBool>>>,
}

impl WorkPool {
    /// Spawns `threads` threads (at least one).
    pub fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared { queue: Mutex::new(Queue::default()), ready: Condvar::new() });
        for i in 0..threads.max(1) {
            let shared = shared.clone();
            thread::Builder::new().name(format!("dsp-app-work-{i}")).spawn(move || work(&shared)).expect("failed to spawn a work thread");
        }
        Self { shared, latest: Mutex::default() }
    }

    /// Threads for this machine: half the cores, 2 to 4.
    pub fn default_threads() -> usize {
        thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 4))
    }

    /// Queues `work` under `key`, replacing a waiting job of that key and cancelling the one in
    /// progress.
    pub fn request(&self, key: JobKey, work: Work) {
        let cancel = Arc::new(AtomicBool::new(false));
        if let Some(previous) = self.latest.lock().expect("work pool lock").insert(key, cancel.clone()) {
            previous.store(true, Ordering::Relaxed);
        }
        let mut q = self.shared.queue.lock().expect("work pool lock");
        if q.waiting.insert(key, Queued { work, cancel }).is_none() {
            q.order.push_back(key);
        }
        drop(q);
        self.shared.ready.notify_one();
    }

    /// Cancels and forgets every job of `owner` (a view that closed).
    pub fn forget(&self, owner: u64) {
        self.latest.lock().expect("work pool lock").retain(|k, flag| {
            let mine = k.owner == owner;
            if mine {
                flag.store(true, Ordering::Relaxed);
            }
            !mine
        });
        let mut q = self.shared.queue.lock().expect("work pool lock");
        q.waiting.retain(|k, _| k.owner != owner);
        q.order.retain(|k| k.owner != owner);
    }
}

impl Drop for WorkPool {
    fn drop(&mut self) {
        self.shared.queue.lock().expect("work pool lock").closed = true;
        self.shared.ready.notify_all();
    }
}

fn work(shared: &Shared) {
    loop {
        let (key, Queued { work, cancel }) = {
            let mut q = shared.queue.lock().expect("work pool lock");
            loop {
                if q.closed {
                    return;
                }
                if let Some(next) = q.take() {
                    break next;
                }
                q = shared.ready.wait(q).expect("work pool lock");
            }
        };
        if !cancel.load(Ordering::Relaxed) {
            work(&cancel);
        }
        let mut q = shared.queue.lock().expect("work pool lock");
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
    use std::time::Duration;

    /// A job of `key` that reports `(key, value)`.
    fn job(value: u32, key: JobKey, tx: &mpsc::Sender<(JobKey, u32)>) -> Work {
        let tx = Mutex::new(tx.clone());
        Box::new(move |_| tx.lock().unwrap().send((key, value)).unwrap())
    }

    #[test]
    fn test_runs_the_latest_job_per_key_in_order() {
        let (tx, rx) = mpsc::channel();
        let pool = WorkPool::new(3);
        let (a, b) = (JobKey::new(1, 0), JobKey::new(1, 1));
        // Bursts for two keys of one owner: jobs may be skipped, never reordered; the newest runs
        for v in [100u32, 200, 300, 400] {
            pool.request(a, job(v, a, &tx));
            pool.request(b, job(v + 1, b, &tx));
        }
        let mut last = BTreeMap::new();
        while let Ok((key, v)) = rx.recv_timeout(Duration::from_secs(5)) {
            if let Some(prev) = last.insert(key, v) {
                assert!(v > prev, "results of one key arrive in request order");
            }
            if last.get(&a) == Some(&400) && last.get(&b) == Some(&401) {
                break;
            }
        }
        assert_eq!((last.get(&a), last.get(&b)), (Some(&400), Some(&401)));
    }

    #[test]
    fn test_newer_job_cancels_the_one_in_progress() {
        let (tx, rx) = mpsc::channel();
        let pool = WorkPool::new(2);
        let key = JobKey::new(1, 0);
        let (started_tx, started_rx) = mpsc::channel();
        let slow_tx = Mutex::new(tx.clone());
        // A slow job that stops when cancelled (and so reports nothing)
        pool.request(
            key,
            Box::new(move |cancel| {
                started_tx.send(()).unwrap();
                while !cancel.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(1));
                }
                let _ = &slow_tx;
            }),
        );
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        pool.request(key, job(7, key, &tx));
        assert_eq!(rx.recv_timeout(Duration::from_secs(5)).unwrap(), (key, 7), "only the newer job reports");
    }

    #[test]
    fn test_forget_drops_an_owners_waiting_jobs() {
        let (tx, rx) = mpsc::channel();
        let pool = WorkPool::new(1);
        let (gate_tx, gate_rx) = mpsc::channel::<()>();
        let gate = Mutex::new(gate_rx);
        // Occupy the only thread, queue two jobs of owner 2, forget it, release the thread
        pool.request(JobKey::new(1, 0), Box::new(move |_| gate.lock().unwrap().recv().unwrap()));
        pool.request(JobKey::new(2, 0), job(5, JobKey::new(2, 0), &tx));
        pool.request(JobKey::new(2, 1), job(6, JobKey::new(2, 1), &tx));
        pool.forget(2);
        pool.request(JobKey::new(3, 0), job(9, JobKey::new(3, 0), &tx));
        gate_tx.send(()).unwrap();
        assert_eq!(rx.recv_timeout(Duration::from_secs(5)).unwrap(), (JobKey::new(3, 0), 9));
        assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
    }
}
