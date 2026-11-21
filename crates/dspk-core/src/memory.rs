//! The RAM the app may use for what it keeps to be fast (decoded recording chunks, frozen
//! processed summaries, spike windows): one cache with one limit for every recording open, as a
//! DAW's or a browser's memory setting. Everything in it can be read or computed again, so the
//! cache only ever trades speed for memory: past the limit, the least recently used entries go.
//!
//! **The limit** ([`host_cache`]`().limit()`): the app's setting (`[compute] memory` in
//! `settings.toml`, the Settings screen), else `DSPK_HOST_MEMORY` (`8GiB`, `512MiB`, bytes), else
//! a quarter of the computer's RAM (1 GiB in a browser, where the page sets it).
//!
//! **The guarantee**: an entry is stored only after room is made for it, and one larger than a
//! quarter of the limit is not stored at all (the caller keeps it while it uses it, then it goes).
//! So the cache never holds more than its limit, whatever the number of recordings.

use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

struct Entry {
    key: String,
    value: Arc<dyn Any + Send + Sync>,
    bytes: u64,
    used: u64,
}

/// The app's cache in RAM (see the module).
pub struct HostCache {
    limit: AtomicU64,
    entries: Mutex<Vec<Entry>>,
    clock: AtomicU64,
}

/// Bytes like `6GiB`, `512MiB`, `800MB` or a plain number (`None`: not a size, or zero).
pub fn parse_bytes(s: &str) -> Option<u64> {
    let s = s.trim();
    let split = s.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(s.len());
    let (num, unit) = s.split_at(split);
    let value: f64 = num.parse().ok()?;
    let scale = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "kib" | "k" => 1024.0,
        "mib" | "m" => 1024.0 * 1024.0,
        "gib" | "g" => 1024.0 * 1024.0 * 1024.0,
        "kb" => 1e3,
        "mb" => 1e6,
        "gb" => 1e9,
        _ => return None,
    };
    Some((value * scale) as u64).filter(|&b| b > 0)
}

/// `bytes` for people: `1.5 GiB`, `300 MiB`.
pub fn format_bytes(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= (1u64 << 30) as f64 {
        format!("{:.1} GiB", b / (1u64 << 30) as f64)
    } else {
        format!("{:.0} MiB", b / (1u64 << 20) as f64)
    }
}

/// The computer's RAM in bytes, where it can be read without a dependency (Linux); `None` elsewhere
/// (the app sets the limit from its own reading).
fn total_ram() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kib: u64 = text.lines().find_map(|l| l.strip_prefix("MemTotal:"))?.trim().trim_end_matches("kB").trim().parse().ok()?;
        Some(kib * 1024)
    }
    #[cfg(not(target_os = "linux"))]
    None
}

/// The limit when nothing sets one: `DSPK_HOST_MEMORY`, else a quarter of the RAM (2 GiB when it
/// cannot be read; 1 GiB in a browser).
pub fn default_limit() -> u64 {
    if let Some(b) = std::env::var("DSPK_HOST_MEMORY").ok().as_deref().and_then(parse_bytes) {
        return b;
    }
    if cfg!(target_family = "wasm") {
        return 1 << 30;
    }
    total_ram().map_or(512 << 20, |r| (r / 4).min(512 << 20))
}

/// The process's cache.
pub fn host_cache() -> &'static HostCache {
    static CACHE: OnceLock<HostCache> = OnceLock::new();
    CACHE.get_or_init(|| HostCache::with_limit(default_limit()))
}

impl HostCache {
    /// A cache of `limit` bytes ([`host_cache`] is the process's).
    pub fn with_limit(limit: u64) -> Self {
        Self { limit: AtomicU64::new(limit), entries: Mutex::new(Vec::new()), clock: AtomicU64::new(0) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Entry>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn tick(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed)
    }

    /// The bytes it may hold.
    pub fn limit(&self) -> u64 {
        self.limit.load(Ordering::Relaxed)
    }

    /// Sets the limit (the app's memory setting), dropping the least recently used entries past it.
    pub fn set_limit(&self, limit: u64) {
        self.limit.store(limit.max(64 << 20), Ordering::Relaxed);
        let mut entries = self.lock();
        Self::shrink(&mut entries, self.limit(), 0);
    }

    /// The bytes held now.
    pub fn used(&self) -> u64 {
        self.lock().iter().map(|e| e.bytes).sum()
    }

    /// The entry under `key`, if kept and of type `T` (it was used now).
    pub fn get<T: Any + Send + Sync>(&self, key: &str) -> Option<Arc<T>> {
        let now = self.tick();
        let mut entries = self.lock();
        let e = entries.iter_mut().find(|e| e.key == key)?;
        e.used = now;
        e.value.clone().downcast::<T>().ok()
    }

    /// Keeps `value` (`bytes` of RAM) under `key`, the least recently used entries dropped first to
    /// make room. Not kept when larger than a quarter of the limit (false).
    pub fn put<T: Any + Send + Sync>(&self, key: String, value: Arc<T>, bytes: u64) -> bool {
        let limit = self.limit();
        let now = self.tick();
        let mut entries = self.lock();
        entries.retain(|e| e.key != key);
        if bytes > limit / 4 {
            return false;
        }
        Self::shrink(&mut entries, limit, bytes);
        entries.push(Entry { key, value, bytes, used: now });
        true
    }

    /// Drops the entries whose key starts with `prefix` (a recording closed, a chain edited).
    pub fn forget_prefix(&self, prefix: &str) {
        self.lock().retain(|e| !e.key.starts_with(prefix));
    }

    fn shrink(entries: &mut Vec<Entry>, limit: u64, room: u64) {
        let mut used: u64 = entries.iter().map(|e| e.bytes).sum();
        while used + room > limit && !entries.is_empty() {
            let oldest = entries.iter().enumerate().min_by_key(|(_, e)| e.used).map(|(i, _)| i).expect("not empty");
            used -= entries.swap_remove(oldest).bytes;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cache_never_holds_more_than_its_limit_and_drops_the_oldest_first() {
        let cache = HostCache::with_limit(1000);
        for i in 0..20u64 {
            assert!(cache.put(format!("rec{}:{i}", i % 3), Arc::new(i), 200));
            assert!(cache.used() <= 1000, "{} after {i}", cache.used());
        }
        // The five most recent are kept
        assert_eq!(cache.get::<u64>("rec1:19").as_deref(), Some(&19));
        assert_eq!(cache.get::<u64>("rec0:18").as_deref(), Some(&18));
        assert!(cache.get::<u64>("rec0:0").is_none(), "the oldest went first");
        // A recent read keeps an entry: the oldest of the rest goes instead
        let _ = cache.get::<u64>("rec0:15");
        cache.put("new".into(), Arc::new(0u64), 200);
        assert!(cache.get::<u64>("rec0:15").is_some());
        // Too large for the cache: not kept
        assert!(!cache.put("big".into(), Arc::new(0u64), 600));
        // A lower limit drops what no longer fits
        cache.set_limit(64 << 20);
        cache.forget_prefix("rec");
        assert!(cache.get::<u64>("rec0:18").is_none());
        assert!(cache.get::<String>("new").is_none(), "another type is not this entry");
    }

    #[test]
    fn sizes_parse_and_print() {
        assert_eq!(parse_bytes("6GiB"), Some(6 << 30));
        assert_eq!(parse_bytes("512 MiB"), Some(512 << 20));
        assert_eq!(parse_bytes("nope"), None);
        assert_eq!(format_bytes(3 << 29), "1.5 GiB");
        assert_eq!(format_bytes(300 << 20), "300 MiB");
    }
}
