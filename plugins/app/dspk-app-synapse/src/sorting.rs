//! The sorting the views work on is `dspk_synapse::curation::sorting` (read from `.sorting.zarr`,
//! spikes indexed by time, extents, waveforms, PCA, comparison): re-exported here. This file keeps
//! only what belongs to the app: the windows of the signal a view shows, read once and kept.

use std::sync::{Arc, Mutex};

use dspk_view::SignalBackend;

pub use dspk_synapse::curation::sorting::*;
pub use dspk_synapse::curation::waveforms::*;

/// The windows of `signal` over the span `sorting` covers, read the first time and kept (the two
/// most recent: the preview is a new signal after each edit of the ops).
pub fn windows_of(signal: &Arc<dyn SignalBackend>, sorting: &SortingData) -> Arc<Windows> {
    type Kept = Vec<((usize, Option<std::ops::Range<u64>>), Arc<Windows>)>;
    static CACHE: Mutex<Kept> = Mutex::new(Vec::new());
    let span = sorting.span();
    let key = (Arc::as_ptr(signal) as *const () as usize, span.clone());
    if let Some(w) = CACHE.lock().ok().and_then(|c| c.iter().find(|(k, _)| *k == key).map(|(_, w)| w.clone())) {
        return w;
    }
    let w = Arc::new(Windows::read_with(signal.info(), span, |channels, samples, out| signal.read(channels, samples, out)));
    if let Ok(mut c) = CACHE.lock() {
        c.retain(|(k, _)| *k != key);
        c.push((key, w.clone()));
        if c.len() > 2 {
            c.remove(0);
        }
    }
    w
}
