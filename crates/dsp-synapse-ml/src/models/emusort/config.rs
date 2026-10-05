//! EMUsort and Myomatrix configuration presets and parameters.

use serde::{Deserialize, Serialize};

/// Supported EMUsort / Myomatrix electrode array geometries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmusortProbeKind {
    /// 8-channel flexible intramuscular thread (e.g. 1 shank, 8 contacts).
    Thread8,
    /// 32-channel high-density surface/intramuscular grid (e.g. 4x8 array, 4.0 mm pitch).
    Grid32,
    /// 64-channel high-density grid (e.g. 8x8 array, 4.0 mm pitch).
    Grid64,
    /// Custom multi-channel electrode configuration.
    Custom,
}

/// Specialized configuration parameters for EMUsort / Myomatrix spike sorting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmusortSortConfig {
    /// Number of temporal samples per template/snippet (typically 128..151 samples, default 150).
    pub template_samples: usize,
    /// Detection threshold in units of estimated noise standard deviations (default: 6.5σ).
    pub threshold_sigma: f32,
    /// Minimum refractory period between successive spikes in samples (default: 60 samples ~ 2.5 ms at 24.4 kHz).
    pub refractory_samples: usize,
    /// Temporal window in samples for spatial deduplication across channels (default: 36 samples ~ 1.5 ms).
    pub dedup_window_samples: usize,
    /// Spatial radius in micrometers for cross-channel event linking/deduplication (default: 6000.0 µm).
    pub spatial_radius_um: f32,
    /// Minimum plausible motor unit conduction velocity in m/s (default: 2.5 m/s).
    pub min_conduction_velocity: f32,
    /// Maximum plausible motor unit conduction velocity in m/s (default: 6.0 m/s).
    pub max_conduction_velocity: f32,
    /// Number of temporal principal components for muscle basis embedding (default: 12).
    pub num_temporal_pcs: usize,
    /// Minimum number of clusters to evaluate for GMM/decomposition (default: 3).
    pub min_clusters: usize,
    /// Maximum number of clusters to evaluate for GMM/decomposition (default: 10).
    pub max_clusters: usize,
    /// Electrode probe geometry category.
    pub probe_kind: EmusortProbeKind,
}

impl Default for EmusortSortConfig {
    fn default() -> Self {
        Self {
            template_samples: 150,
            threshold_sigma: 6.5,
            refractory_samples: 60,
            dedup_window_samples: 36,
            spatial_radius_um: 6000.0,
            min_conduction_velocity: 2.5,
            max_conduction_velocity: 6.0,
            num_temporal_pcs: 12,
            min_clusters: 3,
            max_clusters: 10,
            probe_kind: EmusortProbeKind::Grid32,
        }
    }
}

impl EmusortSortConfig {
    /// Preset configuration optimized for 32-channel Myomatrix / HD-EMG grids (4x8, pitch 4.0 mm).
    pub fn preset_32ch_grid() -> Self {
        Self {
            probe_kind: EmusortProbeKind::Grid32,
            spatial_radius_um: 6000.0,
            ..Self::default()
        }
    }

    /// Preset configuration optimized for 64-channel Myomatrix grids (8x8).
    pub fn preset_64ch_grid() -> Self {
        Self {
            probe_kind: EmusortProbeKind::Grid64,
            spatial_radius_um: 6000.0,
            max_clusters: 16,
            ..Self::default()
        }
    }

    /// Preset configuration optimized for 8-channel Myomatrix single-thread intramuscular arrays.
    pub fn preset_8ch_thread() -> Self {
        Self {
            probe_kind: EmusortProbeKind::Thread8,
            spatial_radius_um: 1500.0,
            max_clusters: 6,
            ..Self::default()
        }
    }
}

/// Backward-compatible type alias for [`EmusortProbeKind`].
pub type MyomatrixProbeKind = EmusortProbeKind;
/// Backward-compatible type alias for [`EmusortSortConfig`].
pub type MyomatrixSortConfig = EmusortSortConfig;
