//! Live events: the Crossings op a view's chain ends with, run on the signal the view shows
//! (its preview: the chain before Crossings) over a stretch of time, on the host, so ticks and
//! the Epochs view follow the level while it is tuned, before any Write. The same tracker as the
//! pipeline (`dspk_ops::crossings::CrossingTracker`), started a margin before the stretch so the
//! hysteresis and refractory state are right at its start.

use std::ops::Range;
use std::sync::Arc;

use dspk_core::{DspError, DspResult};
use dspk_ops::crossings::{CrossingOptions, CrossingTracker};
use dspk_pipeline::Node;
use dspk_view::SignalBackend;

use crate::data::events::Event;
use crate::domain::session::ViewDoc;

/// Most values read for one live detection (channels × samples): a wider window says to zoom in.
pub const MAX_LIVE_VALUES: usize = 16_000_000;
/// Context read before the stretch, at least (s).
pub const MIN_MARGIN_S: f64 = 1.0;

/// The Crossings op `view`'s chain ends with (its last enabled op), if any.
pub fn crossings_of(view: &ViewDoc) -> Option<&Node> {
    view.ops.iter().rev().find(|o| o.enabled).map(|o| &o.node).filter(|n| matches!(n, Node::Crossings { .. }))
}

/// The tracker options of a crossings node at `rate`.
pub fn options_of(node: &Node, rate: f64) -> Option<CrossingOptions> {
    let Node::Crossings { level, release, direction, min_duration_ms, refractory_ms } = node else { return None };
    let samples = |ms: f64| (ms * 1e-3 * rate).round().max(0.0) as u64;
    Some(CrossingOptions {
        levels: vec![*level],
        releases: release.map(|r| vec![r]).unwrap_or_default(),
        direction: (*direction).into(),
        min_duration_samples: samples(*min_duration_ms),
        refractory_samples: samples(*refractory_ms),
    })
}

/// What a live detection was made for (a finished one for other settings is dropped).
#[derive(Debug, Clone, PartialEq)]
pub struct LiveKey {
    /// The chain (ops as JSON): any change of an op or a value changes it.
    pub chain: String,
    pub channels: Vec<usize>,
    /// Samples of the stretch.
    pub samples: Range<u64>,
    /// The signal read (its address): the raw recording until the preview is built, then the
    /// preview.
    pub signal: usize,
}

impl LiveKey {
    pub fn new(view: &ViewDoc, channels: &[usize], samples: Range<u64>, signal: &Arc<dyn SignalBackend>) -> Self {
        Self { chain: serde_json::to_string(&view.ops).unwrap_or_default(), channels: channels.to_vec(), samples, signal: Arc::as_ptr(signal) as *const () as usize }
    }
}

/// The events of `channels` of `signal` with their onset in `samples`, found by `node` (a
/// crossings node). Reads a margin before the stretch (the refractory time, ten minimum
/// durations, at least [`MIN_MARGIN_S`]); `Err` when that is more than [`MAX_LIVE_VALUES`].
pub fn detect(signal: &Arc<dyn SignalBackend>, channels: &[usize], samples: Range<u64>, node: &Node) -> DspResult<Vec<Event>> {
    let info = signal.info();
    let rate = info.sample_rate_hz();
    let options = options_of(node, rate).ok_or_else(|| DspError::InvalidConfig("not a crossings node".into()))?;
    let margin_s = match node {
        Node::Crossings { min_duration_ms, refractory_ms, .. } => (refractory_ms * 1e-3 + 10.0 * min_duration_ms * 1e-3).max(MIN_MARGIN_S),
        _ => MIN_MARGIN_S,
    };
    let end = samples.end.min(info.samples);
    let start = samples.start.min(end);
    let from = start.saturating_sub((margin_s * rate) as u64);
    let n = (end - from) as usize;
    if n * channels.len() > MAX_LIVE_VALUES {
        return Err(DspError::InvalidConfig("zoom in to see live events".into()));
    }
    if n == 0 || channels.is_empty() {
        return Ok(Vec::new());
    }
    let mut rows = vec![0.0f32; channels.len() * n];
    for (i, &ch) in channels.iter().enumerate() {
        signal.read(&[ch], from..end, &mut rows[i * n..(i + 1) * n])?;
    }
    let mut tracker = CrossingTracker::new(channels.len(), &options).map_err(|e| DspError::InvalidConfig(e.to_string()))?;
    tracker.push_host_window(&rows, n, 0..n, from);
    let secs = |s: u64| s as f64 / rate;
    Ok(tracker
        .finish()
        .into_iter()
        .filter(|e| e.onset >= start)
        .map(|e| Event { channel: channels[e.channel] as u32, onset_s: secs(e.onset), offset_s: e.offset.map(secs), peak_value: e.peak_value })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_follow_the_node() {
        let node: Node = serde_json::from_value(serde_json::json!({"op": "crossings", "level": 80.0, "release": 70.0, "min_duration_ms": 30.0, "refractory_ms": 80.0})).unwrap();
        let o = options_of(&node, 1000.0).unwrap();
        assert_eq!((o.levels, o.releases, o.min_duration_samples, o.refractory_samples), (vec![80.0], vec![70.0], 30, 80));
        let scale: Node = serde_json::from_value(serde_json::json!({"op": "scale", "alpha": 2.0})).unwrap();
        assert!(options_of(&scale, 1000.0).is_none());
    }
}
