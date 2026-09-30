use serde::{Deserialize, Serialize};

/// Physical layout of recording contacts (µm).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProbeGeometry {
    pub name: String,
    /// Recording whose channels these contacts belong to.
    pub recording: String,
    pub contacts: Vec<Contact>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Contact {
    /// Channel index in the recording.
    pub channel: usize,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub shank: usize,
}
