use serde::{Deserialize, Serialize};

/// Bitset/Boolean mask indicating valid (active) vs invalid (disabled/bad) channels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelMask {
    total_channels: usize,
    mask: Vec<bool>,
}

impl ChannelMask {
    pub fn all_enabled(total_channels: usize) -> Self {
        Self {
            total_channels,
            mask: vec![true; total_channels],
        }
    }

    pub fn from_bools(mask: Vec<bool>) -> Self {
        let total = mask.len();
        Self {
            total_channels: total,
            mask,
        }
    }

    pub fn from_disabled_indices(total_channels: usize, disabled: &[usize]) -> Self {
        let mut mask = vec![true; total_channels];
        for &ch in disabled {
            if ch < total_channels {
                mask[ch] = false;
            }
        }
        Self {
            total_channels,
            mask,
        }
    }

    pub fn is_enabled(&self, channel: usize) -> bool {
        self.mask.get(channel).copied().unwrap_or(false)
    }

    pub fn set_enabled(&mut self, channel: usize, enabled: bool) {
        if channel < self.total_channels {
            self.mask[channel] = enabled;
        }
    }

    pub fn total_channels(&self) -> usize {
        self.total_channels
    }

    pub fn active_count(&self) -> usize {
        self.mask.iter().filter(|&&enabled| enabled).count()
    }

    pub fn active_indices(&self) -> Vec<usize> {
        self.mask
            .iter()
            .enumerate()
            .filter_map(|(i, &enabled)| if enabled { Some(i) } else { None })
            .collect()
    }

    pub fn as_slice(&self) -> &[bool] {
        &self.mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_mask() {
        let mut mask = ChannelMask::from_disabled_indices(8, &[2, 5]);
        assert_eq!(mask.total_channels(), 8);
        assert_eq!(mask.active_count(), 6);
        assert!(mask.is_enabled(0));
        assert!(!mask.is_enabled(2));
        assert_eq!(mask.active_indices(), vec![0, 1, 3, 4, 6, 7]);

        mask.set_enabled(2, true);
        assert!(mask.is_enabled(2));
        assert_eq!(mask.active_count(), 7);
    }
}
