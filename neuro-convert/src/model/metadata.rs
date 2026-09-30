use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Session-level metadata. Inputs fill what their files carry; the user's metadata file fills
/// the rest before exporting (NWB requires e.g. `start_time`, `description`, subject species).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub identifier: Option<String>,
    pub description: Option<String>,
    /// ISO 8601 local time as recorded (time zone added from user metadata when known).
    pub start_time: Option<String>,
    pub stop_time: Option<String>,
    pub experiment: Option<String>,
    pub experimenters: Vec<String>,
    pub lab: Option<String>,
    pub institution: Option<String>,
    pub subject: Subject,
    pub devices: Vec<Device>,
    /// Free text notes recorded during the session.
    pub notes: Vec<String>,
    /// Anything else worth keeping (software versions, rig names, …).
    pub extra: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Subject {
    pub id: Option<String>,
    pub species: Option<String>,
    pub sex: Option<String>,
    /// ISO 8601 duration, e.g. `P90D`.
    pub age: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Device {
    pub name: String,
    pub description: String,
    pub manufacturer: Option<String>,
    /// Model of the device (e.g. `RZ2`), written as an NWB `DeviceModel` with the manufacturer.
    #[serde(default)]
    pub model: Option<String>,
}
