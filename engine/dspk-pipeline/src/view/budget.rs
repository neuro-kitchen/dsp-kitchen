//! One device-memory budget for everything the views keep on the device: processed blocks,
//! pipeline workspaces, uploaded input windows. The device is shared, so the budget is too (one
//! per process, [`DeviceBudget::global`]).
//!
//! **Why a budget and not counts.** wgpu poisons the device on an out-of-memory error: every
//! later launch fails and the views go blank until the process restarts. Nothing recovers from
//! that, so the views must never ask for more than the device holds. Counts of chains kept say
//! nothing about how much memory each one holds (a block of 32 channels over 20 s is 256 MiB).
//!
//! **How it works** (texture streaming pools in games, voices in audio engines):
//! - **The limit follows the device**: half of what the runtime reports it may hold
//!   (`max_memory`: Vulkan's largest heap, CUDA, Metal), or `DSPK_DEVICE_MEMORY`; where the runtime
//!   reports nothing (a browser's WebGPU), four of its largest buffers, within 256 MiB to 2 GiB.
//! - **Every holder is charged** what it keeps ([`Self::charge`]), and **makes room before it
//!   allocates** ([`Self::make_room`]): holders are released least recently used first, those not
//!   used in the last second (off screen) before those that are; a holder's workspace goes before
//!   its block (a chain off screen keeps only its result).
//! - **A released holder stays** (a virtual voice): its plan, its fitted whitening and its key are
//!   kept, only its device memory goes; the next view that asks for it makes it again.
//! - After a release, the freed pages go back to the driver (`memory_cleanup`).
//!
//! Device buffers are allocated in size classes ([`size_class`]) so the pool reuses them across
//! zoom steps instead of reserving a new page per window size.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use cubecl::prelude::Client;
use web_time::Instant;

/// Something that keeps device memory for the views and can give it back.
pub(crate) trait Holder: Send + Sync {
    /// Frees what it keeps: only its workspace (`full == false`) or everything. The bytes freed,
    /// or `None` when it is busy (being made or read now: it is not released then).
    fn release(&self, full: bool) -> Option<u64>;
}

struct Member {
    id: usize,
    holder: Weak<dyn Holder>,
    /// Bytes it keeps (as it last said), and how many of them are its workspace.
    bytes: u64,
    workspace: u64,
    used: Instant,
}

/// The views' share of the device's memory, and who holds what of it.
pub struct DeviceBudget {
    /// Set by [`Self::with_limit`], or read from the device on first use ([`Self::global`]: a
    /// browser's engine makes its chains before WebGPU opens).
    limit: OnceLock<u64>,
    members: Mutex<Vec<Member>>,
    released: AtomicU64,
}

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

/// A holder not used for this long is off screen: released before those that are.
const ON_SCREEN_SECS: f64 = 1.0;

/// A fresh holder id.
pub(crate) fn next_id() -> usize {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// `n` rounded up to a size class: `2^k × {1, 1.25, 1.5, 1.75}` (at most a quarter more), so buffers
/// of nearby sizes share one class and the device's pool reuses freed pages for them.
pub fn size_class(n: usize) -> usize {
    if n <= 64 {
        return n;
    }
    let top = usize::BITS - 1 - n.leading_zeros();
    let quarter = 1usize << (top - 2);
    n.div_ceil(quarter) * quarter
}

/// Bytes like `6GiB`, `512MiB`, `800MB` or a plain number.
fn parse_bytes(s: &str) -> Option<u64> {
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

/// The views' budget on `client`'s device: `DSPK_DEVICE_MEMORY` when set, else half of what the
/// runtime says the device may hold, else (no figure: a browser) four of its largest buffers
/// within 256 MiB to 2 GiB.
pub fn device_limit(client: &Client) -> u64 {
    if let Some(bytes) = std::env::var("DSPK_DEVICE_MEMORY").ok().as_deref().and_then(parse_bytes) {
        return bytes;
    }
    let memory = &client.properties().memory;
    match memory.max_memory().or_else(device_total) {
        Some(total) => total / 2,
        None => (4 * memory.max_page_size).clamp(256 << 20, 2 << 30),
    }
}

/// The GPU's memory where the runtime reports none (wgpu on Vulkan without the extension CubeCL
/// reads it from): NVIDIA's `nvidia-smi`, AMD's `mem_info_vram_total` (Linux). `None` in a browser
/// or where neither answers.
fn device_total() -> Option<u64> {
    #[cfg(not(target_family = "wasm"))]
    {
        static TOTAL: OnceLock<Option<u64>> = OnceLock::new();
        *TOTAL.get_or_init(|| {
            let nvidia = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .and_then(|o| String::from_utf8_lossy(&o.stdout).lines().next()?.trim().parse::<u64>().ok())
                .map(|mib| mib << 20);
            nvidia.or_else(|| {
                std::fs::read_dir("/sys/class/drm").ok()?.flatten().filter_map(|e| std::fs::read_to_string(e.path().join("device/mem_info_vram_total")).ok()).filter_map(|t| t.trim().parse::<u64>().ok()).max()
            })
        })
    }
    #[cfg(target_family = "wasm")]
    None
}

impl DeviceBudget {
    /// A budget of `limit` bytes (tests; [`Self::global`] sizes it from the device).
    pub fn with_limit(limit: u64) -> Arc<Self> {
        let budget = Self { limit: OnceLock::new(), members: Mutex::new(Vec::new()), released: AtomicU64::new(0) };
        let _ = budget.limit.set(limit);
        Arc::new(budget)
    }

    /// The process's budget, sized from the default compute target's device when first asked
    /// ([`Self::limit`]; 1 GiB when no device opens: nothing is kept then anyway).
    pub fn global() -> Arc<Self> {
        static GLOBAL: OnceLock<Arc<DeviceBudget>> = OnceLock::new();
        GLOBAL.get_or_init(|| Arc::new(Self { limit: OnceLock::new(), members: Mutex::new(Vec::new()), released: AtomicU64::new(0) })).clone()
    }

    /// The bytes the views may keep.
    pub fn limit(&self) -> u64 {
        *self.limit.get_or_init(|| dspk_core::ComputeTarget::from_env().and_then(dspk_core::ComputeTarget::client).map(|c| device_limit(&c)).unwrap_or(1 << 30))
    }

    /// The bytes the holders keep now (as they said).
    pub fn charged(&self) -> u64 {
        self.lock().iter().filter(|m| m.holder.strong_count() > 0).map(|m| m.bytes).sum()
    }

    /// Bytes released to make room so far.
    pub fn released(&self) -> u64 {
        self.released.load(Ordering::Relaxed)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Member>> {
        self.members.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Adds `holder` (its id from [`next_id`]), keeping nothing yet.
    pub(crate) fn register(&self, id: usize, holder: Weak<dyn Holder>) {
        let mut members = self.lock();
        members.retain(|m| m.holder.strong_count() > 0 && m.id != id);
        members.push(Member { id, holder, bytes: 0, workspace: 0, used: Instant::now() });
    }

    /// Holder `id` keeps `bytes` now, `workspace` of them in its pipeline workspace (it was used now).
    pub(crate) fn charge(&self, id: usize, bytes: u64, workspace: u64) {
        if let Some(m) = self.lock().iter_mut().find(|m| m.id == id) {
            (m.bytes, m.workspace, m.used) = (bytes, workspace.min(bytes), Instant::now());
        }
    }

    /// Holder `id` was used now (it is on screen).
    pub(crate) fn touch(&self, id: usize) {
        if let Some(m) = self.lock().iter_mut().find(|m| m.id == id) {
            m.used = Instant::now();
        }
    }

    /// What the device's memory pools reserve now (every allocation: the views', their short-lived
    /// buffers, the curation views', a freeze pass's, and pages freed but not given back), and the
    /// most it may reserve: three quarters of what the device holds (the rest is the driver's and
    /// other programs'). `None` in a browser (no blocking report) or without a client.
    fn device_use(client: Option<&Client>, limit: u64) -> Option<(u64, u64)> {
        #[cfg(not(target_family = "wasm"))]
        {
            let client = client?;
            let total = client.properties().memory.max_memory().or_else(device_total).unwrap_or(2 * limit);
            let reserved = client.memory_report(cubecl::MemoryScope::Device).usage().bytes_reserved;
            Some((reserved, total / 4 * 3))
        }
        #[cfg(target_family = "wasm")]
        {
            let _ = (client, limit);
            None
        }
    }

    /// Releases other holders until `need` more bytes fit for holder `id` (what it keeps now
    /// included in what it will keep: pass the bytes it will hold in all), counted two ways: what
    /// the holders say they keep against the budget, and what the device's pools really reserve
    /// against three quarters of the device ([`Self::device_use`]: short-lived buffers and pages
    /// freed but kept count there). Pages freed but kept go back to the driver first; then holders
    /// are released: off screen first (their workspaces, then their blocks, least recently used
    /// first), then on screen. When even all of that does not make room, as much as could be is
    /// freed (the caller runs; a holder alone over the budget is the device's limit, not the
    /// budget's).
    pub(crate) fn make_room(&self, id: usize, need: u64, client: Option<&Client>) {
        let limit = self.limit();
        let charged_over = |members: &[Member]| members.iter().filter(|m| m.id != id && m.holder.strong_count() > 0).map(|m| m.bytes).sum::<u64>() + need > limit;
        let device_over = || Self::device_use(client, limit).is_some_and(|(reserved, most)| reserved + need > most);
        let cleanup = || {
            #[cfg(not(target_family = "wasm"))]
            if let Some(client) = client {
                let _ = client.memory_cleanup();
            }
        };
        let mut real_over = device_over();
        if real_over {
            // Pages freed and still reserved go back first: often that is room enough
            cleanup();
            real_over = device_over();
        }
        if !real_over && !charged_over(&self.lock()) {
            return;
        }
        let mut freed = 0u64;
        // (off screen only, workspace only): the cheapest to make again first
        'passes: for (off_screen, full) in [(true, false), (true, true), (false, false), (false, true)] {
            // Holders tried in this pass (busy ones stay; each is asked once per pass)
            let mut tried: Vec<usize> = Vec::new();
            loop {
                // The least recently used candidate not tried yet in this pass
                let pick = {
                    let members = self.lock();
                    if !real_over && !charged_over(&members) {
                        break 'passes;
                    }
                    members
                        .iter()
                        .filter(|m| m.id != id && m.holder.strong_count() > 0 && !tried.contains(&m.id))
                        .filter(|m| if full { m.bytes > 0 } else { m.workspace > 0 })
                        .filter(|m| !off_screen || m.used.elapsed().as_secs_f64() >= ON_SCREEN_SECS)
                        .min_by_key(|m| m.used)
                        .map(|m| (m.id, m.holder.clone()))
                };
                let Some((other, holder)) = pick else { break };
                tried.push(other);
                // Released without the members' lock held: a holder may charge itself as it frees
                let released = holder.upgrade().and_then(|h| h.release(full));
                {
                    let mut members = self.lock();
                    if let (Some(m), Some(bytes)) = (members.iter_mut().find(|m| m.id == other), released) {
                        freed += bytes;
                        if full {
                            (m.bytes, m.workspace) = (0, 0);
                        } else {
                            m.bytes = m.bytes.saturating_sub(m.workspace);
                            m.workspace = 0;
                        }
                    }
                }
                if real_over && released.is_some_and(|b| b > 0) {
                    cleanup();
                    real_over = device_over();
                }
            }
        }
        if freed > 0 {
            self.released.fetch_add(freed, Ordering::Relaxed);
            // The freed pages back to the driver (a browser frees them as their buffers drop)
            cleanup();
        }
    }

    /// Whether `need` more bytes fit on the device now without releasing anything (a freeze pass
    /// asks before each segment, and waits when they do not: views first).
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    pub(crate) fn fits(&self, need: u64, client: Option<&Client>) -> bool {
        let limit = self.limit();
        self.charged() + need <= limit && !Self::device_use(client, limit).is_some_and(|(reserved, most)| reserved + need > most)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "prints the device's memory limits"]
    fn device_memory_limits() {
        let client = dspk_core::ComputeTarget::from_env().and_then(dspk_core::ComputeTarget::client).unwrap();
        let m = &client.properties().memory;
        println!("{}: max page {} MiB, max memory {:?} MiB, budget {} MiB", client.name(), m.max_page_size >> 20, m.max_memory().map(|b| b >> 20), device_limit(&client) >> 20);
    }

    #[test]
    fn size_classes_round_up_by_at_most_a_quarter() {
        for n in [1usize, 63, 64, 65, 100, 1000, 4097, 1 << 20, (1 << 26) + 1, 123_456_789] {
            let c = size_class(n);
            assert!(c >= n && (c as f64) <= n as f64 * 1.25 + 1.0, "{n} → {c}");
            assert_eq!(size_class(c), c, "a class is its own class");
        }
        // Nearby sizes share a class
        assert_eq!(size_class(1_000_000), size_class(1_040_000));
    }

    #[test]
    fn byte_sizes_parse() {
        assert_eq!(parse_bytes("6GiB"), Some(6 << 30));
        assert_eq!(parse_bytes("512MiB"), Some(512 << 20));
        assert_eq!(parse_bytes("800MB"), Some(800_000_000));
        assert_eq!(parse_bytes("1024"), Some(1024));
        assert_eq!(parse_bytes("lots"), None);
        assert_eq!(parse_bytes("0"), None);
    }

    struct Fake {
        bytes: Mutex<(u64, u64)>,
        busy: bool,
    }

    impl Holder for Fake {
        fn release(&self, full: bool) -> Option<u64> {
            if self.busy {
                return None;
            }
            let mut b = self.bytes.lock().unwrap();
            let freed = if full { b.0 } else { b.1 };
            *b = if full { (0, 0) } else { (b.0 - b.1, 0) };
            Some(freed)
        }
    }

    #[test]
    fn room_is_made_off_screen_first_workspaces_before_blocks_busy_ones_kept() {
        let budget = DeviceBudget::with_limit(1000);
        let make = |bytes: u64, ws: u64, busy: bool| Arc::new(Fake { bytes: Mutex::new((bytes, ws)), busy });
        let (old, busy, recent) = (make(400, 100, false), make(300, 0, true), make(200, 50, false));
        for (id, h) in [(1, old.clone()), (2, busy.clone()), (3, recent.clone())] {
            let weak: Weak<dyn Holder> = Arc::downgrade(&(h.clone() as Arc<dyn Holder>));
            budget.register(id, weak);
            let b = *h.bytes.lock().unwrap();
            budget.charge(id, b.0, b.1);
        }
        // 900 kept; 150 more for holder 4: the oldest's workspace (100) is enough
        budget.make_room(4, 150, None);
        assert_eq!(*old.bytes.lock().unwrap(), (300, 0));
        assert_eq!(budget.charged(), 800);
        // 600 more: the whole of the oldest goes, the busy one stays, then the most recent one
        budget.make_room(4, 600, None);
        assert_eq!(*old.bytes.lock().unwrap(), (0, 0));
        assert_eq!(*busy.bytes.lock().unwrap(), (300, 0));
        assert_eq!(*recent.bytes.lock().unwrap(), (0, 0));
        assert!(budget.charged() + 600 <= 1000 || budget.charged() == 300);
    }
}
