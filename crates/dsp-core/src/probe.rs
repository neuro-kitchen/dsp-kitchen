use serde::{Deserialize, Serialize};
use crate::error::{DspError, DspResult};

/// Physical coordinates of a single electrode recording site (in micrometers).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContactPosition {
    pub x_um: f32,
    pub y_um: f32,
    pub z_um: f32,
}

/// Metadata and physical specifications for an individual contact on a neural probe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelContact {
    pub channel_id: usize,
    pub device_index: usize,
    pub shank_id: usize,
    pub position: ContactPosition,
    pub enabled: bool,
}

/// Multi-channel electrode probe geometry definition (e.g. Neuropixels 1.0, 2.0, Utah array).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeLayout {
    pub name: String,
    pub contacts: Vec<ChannelContact>,
}

impl ProbeLayout {
    pub fn new(name: impl Into<String>, contacts: Vec<ChannelContact>) -> Self {
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

    pub fn get_contact(&self, channel_id: usize) -> DspResult<&ChannelContact> {
        self.contacts
            .iter()
            .find(|c| c.channel_id == channel_id)
            .ok_or(DspError::InvalidChannel {
                channel: channel_id,
                total: self.contacts.len(),
            })
    }

    /// Generates a standard Neuropixels 1.0 layout (384 active readout channels, staggered checkerboard).
    pub fn neuropixels_1_0_standard() -> Self {
        let mut contacts = Vec::with_capacity(384);
        for i in 0..384 {
            // Neuropixels 1.0: 4 columns, 20 um horizontal pitch, 20 um vertical pitch
            let col = i % 2;
            let row = i / 2;
            let x = if col == 0 { 11.0 } else { 27.0 } + (if (row % 2) == 1 { 16.0 } else { 0.0 });
            let y = (row * 20) as f32;

            contacts.push(ChannelContact {
                channel_id: i,
                device_index: i,
                shank_id: 0,
                position: ContactPosition {
                    x_um: x,
                    y_um: y,
                    z_um: 0.0,
                },
                enabled: true,
            });
        }

        Self {
            name: "Neuropixels 1.0 (Standard 384)".into(),
            contacts,
        }
    }
}
