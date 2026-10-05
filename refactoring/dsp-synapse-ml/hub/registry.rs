//! Preset model architectures and hyperparameters for standard electrophysiology probes.

use serde::{Deserialize, Serialize};

/// Standard hardware probe presets supported by the `dsp-synapse-ml` model zoo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProbePreset {
    Neuropixels1,
    Neuropixels2,
    UtahArray,
    Tetrode,
}

/// Canonical model hyperparameters for a given probe geometry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelPresetConfig {
    pub preset: ProbePreset,
    pub k_neighbors: usize,
    pub snippet_samples: usize,
    pub pre_samples: usize,
    pub post_samples: usize,
    pub latent_dim: usize,
    pub base_channels: usize,
    pub sample_rate_hz: f64,
}

impl ModelPresetConfig {
    pub fn for_preset(preset: ProbePreset) -> Self {
        match preset {
            ProbePreset::Neuropixels1 => Self {
                preset,
                k_neighbors: 7,
                snippet_samples: 64,
                pre_samples: 20,
                post_samples: 44,
                latent_dim: 8,
                base_channels: 16,
                sample_rate_hz: 30_000.0,
            },
            ProbePreset::Neuropixels2 => Self {
                preset,
                k_neighbors: 8,
                snippet_samples: 64,
                pre_samples: 20,
                post_samples: 44,
                latent_dim: 8,
                base_channels: 16,
                sample_rate_hz: 30_000.0,
            },
            ProbePreset::UtahArray => Self {
                preset,
                k_neighbors: 4,
                snippet_samples: 48,
                pre_samples: 16,
                post_samples: 32,
                latent_dim: 6,
                base_channels: 12,
                sample_rate_hz: 30_000.0,
            },
            ProbePreset::Tetrode => Self {
                preset,
                k_neighbors: 4,
                snippet_samples: 40,
                pre_samples: 14,
                post_samples: 26,
                latent_dim: 4,
                base_channels: 8,
                sample_rate_hz: 32_000.0,
            },
        }
    }
}
