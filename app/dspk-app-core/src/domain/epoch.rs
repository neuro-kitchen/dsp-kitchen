//! Epochs: named spans of a recording (a baseline, a stimulus, an artefact…) with a type and
//! notes, made by dragging on the timeline; and where recordings sit on the session's time
//! (placements). Both live in the session document; each recording's epochs are also written next
//! to it (`<recording>.dspk/epochs.parquet`, see `data::epochs`).

use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Epoch types offered, each with its colour (hue, degrees).
pub const EPOCH_KINDS: [(&str, f32); 6] = [("baseline", 210.0), ("stimulus", 32.0), ("response", 140.0), ("artefact", 0.0), ("sleep", 265.0), ("other", 180.0)];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EpochDoc {
    /// Unique in the session (`e1`, `e2`, …).
    pub id: String,
    /// The recording it marks.
    pub recording: PathBuf,
    pub name: String,
    /// One of [`EPOCH_KINDS`] or the user's own.
    #[serde(default = "other")]
    pub kind: String,
    /// Seconds from the recording's start.
    pub start_sec: f64,
    pub end_sec: f64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub notes: String,
}

fn other() -> String {
    "other".into()
}

impl EpochDoc {
    pub fn duration_sec(&self) -> f64 {
        self.end_sec - self.start_sec
    }
}

/// The hue of an epoch type (unknown types: a hue from their name).
pub fn kind_hue(kind: &str) -> f32 {
    EPOCH_KINDS.iter().find(|(k, _)| *k == kind).map_or_else(|| kind.bytes().fold(0u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32)) as f32 % 360.0, |(_, h)| *h)
}

/// A set of events of a recording (a crossings output, a selection), drawn as ticks over its
/// traces. The events themselves are in `table` (Parquet, `data::events`); the session keeps how
/// they look.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EventSetDoc {
    /// Unique in the session (`ev1`, `ev2`, …).
    pub id: String,
    /// The recording the events are on.
    pub recording: PathBuf,
    pub name: String,
    /// The events table (usually `<recording>.dspk/events/<name>.parquet`).
    pub table: PathBuf,
    /// Tick colour (hue, degrees).
    #[serde(default = "tick_hue")]
    pub hue: f32,
    /// Tick height, as a fraction of a lane (0.05–1).
    #[serde(default = "tick_height")]
    pub tick_height: f32,
    /// Tick thickness in logical pixels (0.5–8).
    #[serde(default = "tick_width")]
    pub tick_width: f32,
    /// Offsets drawn too (a fainter tick), so each event shows as a span.
    #[serde(default)]
    pub show_offsets: bool,
    #[serde(default = "shown")]
    pub visible: bool,
}

/// Defaults of a new event set's ticks.
pub const DEFAULT_TICK_HUE: f32 = 50.0;
pub const DEFAULT_TICK_HEIGHT: f32 = 0.6;
pub const DEFAULT_TICK_WIDTH: f32 = 1.5;

fn tick_hue() -> f32 {
    DEFAULT_TICK_HUE
}
fn tick_height() -> f32 {
    DEFAULT_TICK_HEIGHT
}
fn tick_width() -> f32 {
    DEFAULT_TICK_WIDTH
}
fn shown() -> bool {
    true
}

impl EventSetDoc {
    /// A new set with the default look.
    pub fn new(recording: PathBuf, name: impl Into<String>, table: PathBuf) -> Self {
        Self { id: String::new(), recording, name: name.into(), table, hue: DEFAULT_TICK_HUE, tick_height: DEFAULT_TICK_HEIGHT, tick_width: DEFAULT_TICK_WIDTH, show_offsets: false, visible: true }
    }
}

/// Where a recording starts on the session's time (s).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Placement {
    pub path: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    pub start_sec: f64,
}
