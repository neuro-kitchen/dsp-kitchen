//! Background compute service and invalidation cache for derived curation data (Step 5d).
//!
//! Newest request per `(view, kind)` wins (unstarted requests for the same view and kind are
//! replaced); computed values are cached by `(clusters, kind, params)` and invalidated per cluster
//! when a cluster is merged or split.

use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};

pub type ClusterId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeKind {
    Waveforms,
    Features,
    Correlogram,
    Amplitudes,
    Isi,
    FiringRate,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Clusters whose spikes contributed to this derived value (1 for single-cluster metrics, 2 for
    /// cross-correlograms, etc.). Invalidating any cluster in `clusters` drops this entry.
    pub clusters: Vec<ClusterId>,
    pub kind: ComputeKind,
    pub params: u64,
}

impl CacheKey {
    pub fn single(cluster: ClusterId, kind: ComputeKind, params: u64) -> Self {
        Self { clusters: vec![cluster], kind, params }
    }

    pub fn pair(a: ClusterId, b: ClusterId, kind: ComputeKind, params: u64) -> Self {
        Self { clusters: vec![a, b], kind, params }
    }

    pub fn touches(&self, cluster: ClusterId) -> bool {
        self.clusters.contains(&cluster)
    }
}

/// Thread-safe cache of derived per-cluster / per-pair data, with per-cluster invalidation.
#[derive(Clone, Default)]
pub struct ComputeCache {
    entries: Arc<Mutex<HashMap<CacheKey, Arc<dyn Any + Send + Sync>>>>,
}

impl ComputeCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get<T: Send + Sync + 'static>(&self, key: &CacheKey) -> Option<Arc<T>> {
        let map = self.entries.lock().unwrap();
        map.get(key)?.clone().downcast::<T>().ok()
    }

    pub fn insert<T: Send + Sync + 'static>(&self, key: CacheKey, value: Arc<T>) {
        self.entries.lock().unwrap().insert(key, value);
    }

    /// Gets a cached value or computes and caches it synchronously.
    pub fn get_or_compute<T: Send + Sync + 'static>(&self, key: CacheKey, compute: impl FnOnce() -> T) -> Arc<T> {
        if let Some(found) = self.get::<T>(&key) {
            return found;
        }
        let val = Arc::new(compute());
        self.insert(key, val.clone());
        val
    }

    /// Drops every cached entry that involves `cluster`.
    pub fn invalidate_cluster(&self, cluster: ClusterId) {
        self.entries.lock().unwrap().retain(|k, _| !k.touches(cluster));
    }

    /// Drops every cached entry that involves any cluster in `clusters`.
    pub fn invalidate_clusters(&self, clusters: &[ClusterId]) {
        self.entries.lock().unwrap().retain(|k, _| !clusters.iter().any(|&c| k.touches(c)));
    }

    pub fn clear(&self) {
        self.entries.lock().unwrap().clear();
    }

    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

struct PendingJob {
    slot: (u64, ComputeKind),
    seq: u64,
    work: Box<dyn FnOnce() + Send + 'static>,
}

struct Queue {
    waiting: VecDeque<PendingJob>,
    latest_seq: HashMap<(u64, ComputeKind), u64>,
    next_seq: u64,
    closed: bool,
}

/// Background worker pool where the newest request per `(view_id, ComputeKind)` replaces any
/// unstarted request for that same slot.
#[derive(Clone)]
pub struct ComputeService {
    pub cache: ComputeCache,
    inner: Arc<(Mutex<Queue>, Condvar)>,
}

impl Default for ComputeService {
    fn default() -> Self {
        Self::new(2)
    }
}

impl ComputeService {
    pub fn new(threads: usize) -> Self {
        let inner = Arc::new((
            Mutex::new(Queue { waiting: VecDeque::new(), latest_seq: HashMap::new(), next_seq: 1, closed: false }),
            Condvar::new(),
        ));
        for i in 0..threads.max(1) {
            let shared = inner.clone();
            let _ = std::thread::Builder::new().name(format!("dsp-compute-{i}")).spawn(move || worker(shared));
        }
        Self { cache: ComputeCache::new(), inner }
    }

    /// Submits a background computation for `(view_id, kind)`. If an older job for the same
    /// `(view_id, kind)` is still waiting in the queue, it is replaced; if it is already running,
    /// its result is discarded when a newer job has been submitted.
    pub fn request<T, F, D>(&self, view_id: u64, kind: ComputeKind, compute: F, deliver: D)
    where
        T: Send + 'static,
        F: FnOnce(&ComputeCache) -> T + Send + 'static,
        D: FnOnce(T) + Send + 'static,
    {
        let (lock, cvar) = &*self.inner;
        let mut q = lock.lock().unwrap();
        let seq = q.next_seq;
        q.next_seq += 1;
        let slot = (view_id, kind);
        q.latest_seq.insert(slot, seq);
        q.waiting.retain(|j| j.slot != slot);

        let cache = self.cache.clone();
        let inner_check = self.inner.clone();
        let work = Box::new(move || {
            let out = compute(&cache);
            let still_latest = inner_check.0.lock().unwrap().latest_seq.get(&slot).copied() == Some(seq);
            if still_latest {
                deliver(out);
            }
        });
        q.waiting.push_back(PendingJob { slot, seq, work });
        cvar.notify_one();
    }

    pub fn invalidate_cluster(&self, cluster: ClusterId) {
        self.cache.invalidate_cluster(cluster);
    }

    pub fn invalidate_clusters(&self, clusters: &[ClusterId]) {
        self.cache.invalidate_clusters(clusters);
    }

    pub fn clear(&self) {
        self.cache.clear();
    }
}

fn worker(shared: Arc<(Mutex<Queue>, Condvar)>) {
    let (lock, cvar) = &*shared;
    loop {
        let job = {
            let mut q = lock.lock().unwrap();
            loop {
                if q.closed {
                    return;
                }
                if let Some(j) = q.waiting.pop_front() {
                    break j;
                }
                q = cvar.wait(q).unwrap();
            }
        };
        (job.work)();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn test_cache_invalidation_drops_single_and_pair_entries() {
        let cache = ComputeCache::new();
        cache.insert(CacheKey::single(1, ComputeKind::Waveforms, 0), Arc::new(vec![1.0f32, 2.0]));
        cache.insert(CacheKey::single(2, ComputeKind::Waveforms, 0), Arc::new(vec![3.0f32, 4.0]));
        cache.insert(CacheKey::pair(1, 2, ComputeKind::Correlogram, 50), Arc::new(vec![10u64, 20]));
        cache.insert(CacheKey::pair(2, 3, ComputeKind::Correlogram, 50), Arc::new(vec![30u64, 40]));
        assert_eq!(cache.len(), 4);

        cache.invalidate_cluster(1);
        assert!(cache.get::<Vec<f32>>(&CacheKey::single(1, ComputeKind::Waveforms, 0)).is_none());
        assert!(cache.get::<Vec<u64>>(&CacheKey::pair(1, 2, ComputeKind::Correlogram, 50)).is_none());
        assert!(cache.get::<Vec<f32>>(&CacheKey::single(2, ComputeKind::Waveforms, 0)).is_some());
        assert!(cache.get::<Vec<u64>>(&CacheKey::pair(2, 3, ComputeKind::Correlogram, 50)).is_some());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_compute_service_delivers_latest_per_slot() {
        let svc = ComputeService::new(1);
        let (tx, rx) = mpsc::channel();
        let tx2 = tx.clone();
        svc.request(
            10,
            ComputeKind::Isi,
            |cache| *cache.get_or_compute(CacheKey::single(5, ComputeKind::Isi, 1), || 42usize),
            move |val| {
                let _ = tx2.send(val);
            },
        );
        assert_eq!(rx.recv_timeout(Duration::from_secs(2)).unwrap(), 42);
        assert_eq!(*svc.cache.get::<usize>(&CacheKey::single(5, ComputeKind::Isi, 1)).unwrap(), 42);
    }
}
