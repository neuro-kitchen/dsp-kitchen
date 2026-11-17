//! Who uses the GPU now, in one process: a live chain, interactive views, or a batch job. A GPU
//! is not preempted and its memory is not shared between programs that do not know of each
//! other, so the uses are told apart:
//!
//! | Use | Who | The rule |
//! |---|---|---|
//! | [`Use::Live`] | a device's live chain (a recording session) | the GPU is its own: no batch job starts while it runs |
//! | [`Use::Interactive`] | a view's processing, Curation's read of a signal | best effort: while a batch job holds the GPU, done on the CPU |
//! | [`Use::Batch`] | a sort, a processed write | exclusive: waits for the views in progress, then the views' device memory is released ([`on_release`]) |
//!
//! Live chains are causal only (the live rule), so a sort is never part of one: a sort *beside*
//! a live session on the same GPU is refused until the session ends.

use std::sync::{Condvar, Mutex};
use std::time::Duration;

/// What the GPU is used for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Use {
    Live,
    Interactive,
    Batch,
}

#[derive(Default)]
struct State {
    live: usize,
    interactive: usize,
    batch: bool,
}

static STATE: Mutex<State> = Mutex::new(State { live: 0, interactive: 0, batch: false });
static CHANGED: Condvar = Condvar::new();
type Hook = Box<dyn Fn() + Send + Sync>;
static HOOKS: Mutex<Vec<Hook>> = Mutex::new(Vec::new());

/// How long a batch job waits for the views in progress before taking the GPU anyway.
pub const BATCH_WAIT: Duration = Duration::from_secs(30);

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// The GPU held for one use; given back when dropped.
#[derive(Debug)]
pub struct Lease {
    pub kind: Use,
}

impl Drop for Lease {
    fn drop(&mut self) {
        let mut s = state();
        match self.kind {
            Use::Live => s.live = s.live.saturating_sub(1),
            Use::Interactive => s.interactive = s.interactive.saturating_sub(1),
            Use::Batch => s.batch = false,
        }
        CHANGED.notify_all();
    }
}

/// Adds what to release when a batch job takes the GPU (the views' kept chains): run, then the
/// device's unused memory is handed back to the driver.
pub fn on_release(hook: impl Fn() + Send + Sync + 'static) {
    HOOKS.lock().unwrap_or_else(|e| e.into_inner()).push(Box::new(hook));
}

/// The GPU for a view, unless a batch job holds it (`None`: process it on the CPU).
pub fn interactive() -> Option<Lease> {
    let mut s = state();
    if s.batch {
        return None;
    }
    s.interactive += 1;
    Some(Lease { kind: Use::Interactive })
}

/// The GPU for a live chain. Refused while a batch job holds it (a session waits for no sort,
/// but cannot share the GPU with one).
pub fn live() -> Result<Lease, String> {
    let mut s = state();
    if s.batch {
        return Err("a batch job (a sort, a write) is using the GPU: start the live session when it ends".into());
    }
    s.live += 1;
    Ok(Lease { kind: Use::Live })
}

/// Whether a live chain holds the GPU.
pub fn live_running() -> bool {
    state().live > 0
}

/// Whether a batch job holds the GPU.
pub fn batch_running() -> bool {
    state().batch
}

/// The GPU for a batch job: refused while a live chain runs or another batch job holds it;
/// waits (up to [`BATCH_WAIT`]) for the views in progress, then releases what the views keep
/// ([`on_release`]) and the device memory nobody uses.
pub fn batch() -> Result<Lease, String> {
    {
        let mut s = state();
        if s.live > 0 {
            return Err("the GPU is recording live: sort when the session ends, or on another computer or GPU".into());
        }
        if s.batch {
            return Err("another job is using the GPU".into());
        }
        s.batch = true;
        let (mut s, _) = CHANGED.wait_timeout_while(s, BATCH_WAIT, |s| s.interactive > 0).unwrap_or_else(|e| e.into_inner());
        s.batch = true;
    }
    for hook in HOOKS.lock().unwrap_or_else(|e| e.into_inner()).iter() {
        hook();
    }
    #[cfg(feature = "compute")]
    crate::compute::release_device_memory();
    Ok(Lease { kind: Use::Batch })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rules: views give way to a batch job (and wait for none), a batch job is refused
    /// beside a live chain, a live chain beside a batch job; leases are given back when dropped.
    #[test]
    fn live_excludes_batch_and_batch_sends_views_to_the_cpu() {
        let view = interactive().expect("the GPU is free");
        drop(view);
        let released = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let r = released.clone();
        on_release(move || {
            r.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        });
        let job = batch().unwrap();
        assert!(released.load(std::sync::atomic::Ordering::SeqCst) >= 1, "what views keep is released");
        assert!(interactive().is_none(), "views go to the CPU during a batch job");
        assert!(live().is_err() && batch().is_err());
        drop(job);
        let session = live().unwrap();
        assert!(live_running() && batch().unwrap_err().contains("live"));
        assert!(interactive().is_some(), "views stay best effort beside a live chain");
        drop(session);
        assert!(!live_running() && !batch_running());
    }
}
