use serde::{Deserialize, Serialize};

/// Representation of active, disabled, or bad channels across a multi-channel stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelMask {
    total_channels: usize,
    mask: Vec<bool>,
}

impl ChannelMask {
    /// Creates a mask where all channels are active.
    pub fn all_active(total_channels: usize) -> Self {
        Self {
            total_channels,
            mask: vec![true; total_channels],
        }
    }

    /// Creates a mask from an existing boolean vector.
    pub fn from_bools(mask: Vec<bool>) -> Self {
        let total = mask.len();
        Self {
            total_channels: total,
            mask,
        }
    }

    /// Creates a mask where only specified channel indices are enabled.
    pub fn from_active_indices(total_channels: usize, active_indices: &[usize]) -> Self {
        let mut mask = vec![false; total_channels];
        for &idx in active_indices {
            if idx < total_channels {
                mask[idx] = true;
            }
        }
        Self {
            total_channels,
            mask,
        }
    }

    /// Returns true if the channel is active, false otherwise.
    pub fn is_active(&self, channel_id: usize) -> bool {
        self.mask.get(channel_id).copied().unwrap_or(false)
    }

    /// Sets the active state for a given channel.
    pub fn set_active(&mut self, channel_id: usize, active: bool) {
        if channel_id < self.total_channels {
            self.mask[channel_id] = active;
        }
    }

    /// Total number of channels represented by the mask.
    pub fn total_channels(&self) -> usize {
        self.total_channels
    }

    /// Count of currently active channels.
    pub fn active_count(&self) -> usize {
        self.mask.iter().filter(|&&v| v).count()
    }

    /// Returns a vector of active channel indices.
    pub fn to_indices(&self) -> Vec<usize> {
        self.mask
            .iter()
            .enumerate()
            .filter_map(|(idx, &active)| if active { Some(idx) } else { None })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_mask() {
        let mut mask = ChannelMask::from_active_indices(8, &[0, 2, 4]);
        assert_eq!(mask.total_channels(), 8);
        assert_eq!(mask.active_count(), 3);
        assert!(mask.is_active(0));
        assert!(!mask.is_active(1));
        assert!(mask.is_active(2));

        mask.set_active(1, true);
        assert_eq!(mask.active_count(), 4);
        assert_eq!(mask.to_indices(), vec![0, 1, 2, 4]);
    }
}
