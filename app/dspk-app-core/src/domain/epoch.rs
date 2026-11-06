//! Epochs: named spans of a recording (a baseline, a stimulus, an artefact…) with a type and
//! notes, made by dragging on the timeline; and where recordings sit on the session's time
//! (placements). Both live in the session document; each recording's epochs are also written next
//! to it (`<recording>.epochs.zarr`, see `data::epochs`).

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

/// Where a recording starts on the session's time (s).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Placement {
    pub path: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    pub start_sec: f64,
}
