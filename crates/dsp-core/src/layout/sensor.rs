use serde::{Deserialize, Serialize};
use super::site::SensorSite;
use crate::error::{DspError, DspResult};

/// Physical sensor array geometry layout (e.g. multi-electrode arrays, acoustic arrays).
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
        self.contacts.iter().filter(|c| c.is_enabled).count()
    }

    pub fn get_contact(&self, channel_id: usize) -> DspResult<&SensorSite> {
        self.contacts
            .iter()
            .find(|c| c.channel_id == channel_id)
            .ok_or(DspError::InvalidChannel {
                channel: channel_id,
                total: self.contacts.len(),
            })
    }
}
