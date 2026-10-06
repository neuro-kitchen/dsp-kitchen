//! Cache of derived curation data, and background requests for it.
//!
//! Requests run on the shared [`WorkPool`] (the newest per `(view, kind)` wins); computed values
//! are cached by `(clusters, kind, params)` and dropped per cluster when a cluster is merged or
//! split.

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::curation::ClusterId;
use super::work_pool::{JobKey, WorkPool};


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeKind {
    Waveforms,
    Features,
    Correlogram,
    Amplitudes,
    Isi,
    FiringRate,
}

impl ComputeKind {
    /// Work-pool slot (0 is a view's frame renders).
    pub fn slot(self) -> u32 {
        1 + self as u32
    }
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

impl ComputeCache {
    /// Computes in the background on `pool` (the newest request per `(owner, kind)` wins) and
    /// hands the result to `deliver`, unless a newer request replaced it meanwhile.
    pub fn request<T, F, D>(&self, pool: &WorkPool, owner: u64, kind: ComputeKind, compute: F, deliver: D)
    where
        T: Send + 'static,
        F: FnOnce(&ComputeCache) -> T + Send + 'static,
        D: FnOnce(T) + Send + 'static,
    {
        let cache = self.clone();
        pool.request(
            JobKey::new(owner, kind.slot()),
            Box::new(move |cancel: &AtomicBool| {
                let out = compute(&cache);
                if !cancel.load(Ordering::Relaxed) {
                    deliver(out);
                }
            }),
        );
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
    fn test_requests_run_on_the_pool_and_fill_the_cache() {
        let pool = WorkPool::new(1);
        let cache = ComputeCache::new();
        let (tx, rx) = mpsc::channel();
        cache.request(&pool, 10, ComputeKind::Isi, |c| *c.get_or_compute(CacheKey::single(5, ComputeKind::Isi, 1), || 42usize), move |v| tx.send(v).unwrap());
        assert_eq!(rx.recv_timeout(Duration::from_secs(2)).unwrap(), 42);
        assert_eq!(*cache.get::<usize>(&CacheKey::single(5, ComputeKind::Isi, 1)).unwrap(), 42);
    }
}
