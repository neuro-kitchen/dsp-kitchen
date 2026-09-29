use serde::{Deserialize, Serialize};
use crate::error::{DspError, DspResult};

/// Physical coordinates of a single sensor recording site (in micrometers or unit space).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position3D {
    pub x_um: f32,
    pub y_um: f32,
    pub z_um: f32,
}

impl Position3D {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x_um: x, y_um: y, z_um: z }
    }
}

/// Metadata and physical specifications for an individual sensor site.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorSite {
    pub channel_id: usize,
    pub device_index: usize,
    pub group_id: usize,
    pub shank_id: usize,
    pub position: Position3D,
    pub enabled: bool,
}

/// Multi-channel sensor array geometry definition (e.g. electrode probes, microphone arrays, seismic grids).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorLayout {
    pub name: String,
    pub contacts: Vec<SensorSite>,
}

impl SensorLayout {
    pub fn new(name: impl Into<String>, contacts: Vec<SensorSite>) -> Self {
        Self {
            name: name.into(),
            contacts,
        }
    }

    pub fn total_channels(&self) -> usize {
        self.contacts.len()
    }

    pub fn active_channels(&self) -> usize {
        self.contacts.iter().filter(|c| c.enabled).count()
    }

    pub fn get_site(&self, channel_id: usize) -> DspResult<&SensorSite> {
        self.contacts
            .iter()
            .find(|c| c.channel_id == channel_id)
            .ok_or(DspError::InvalidChannel {
                channel: channel_id,
                total: self.contacts.len(),
            })
    }

    pub fn get_contact(&self, channel_id: usize) -> DspResult<&SensorSite> {
        self.get_site(channel_id)
    }

    pub fn sites(&self) -> &[SensorSite] {
        &self.contacts
    }

    /// Factory for 384-channel Neuropixels 1.0 (canonical neural catalog lives in dsp-synapse).
    pub fn neuropixels_1_0_standard() -> Self {
        let mut contacts = Vec::with_capacity(384);
        for i in 0..384 {
            let col = i % 2;
            let row = i / 2;
            let x = if col == 0 { 11.0 } else { 27.0 } + (if (row % 2) == 1 { 16.0 } else { 0.0 });
            let y = (row * 20) as f32;

            contacts.push(SensorSite {
                channel_id: i,
                device_index: i,
                group_id: 0,
                shank_id: 0,
                position: Position3D::new(x, y, 0.0),
                enabled: true,
            });
        }

        Self {
            name: "Neuropixels 1.0 (Standard 384)".into(),
            contacts,
        }
    }
}

// Backward compatibility type aliases
pub type ContactPosition = Position3D;
pub type ChannelContact = SensorSite;
pub type ProbeLayout = SensorLayout;
