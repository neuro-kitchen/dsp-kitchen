//! Spatial multi-channel referencing and local subtraction primitives.

use dsp_core::ChannelMask;

/// Configuration for spatial referencing operations across electrode arrays.
#[derive(Debug, Clone)]
pub struct SpatialReferenceConfig {
    pub mask: Option<ChannelMask>,
    pub exclude_bad_channels: bool,
}

impl Default for SpatialReferenceConfig {
    fn default() -> Self {
        Self {
            mask: None,
            exclude_bad_channels: true,
        }
    }
}
