pub mod comparison;
pub mod correlogram;
pub mod evoked;
pub mod firing;
pub mod isolation;
pub mod rate;

pub use crate::core::template::{UnitQualityLabel, WaveformTemplate, compute_mean_template};
pub use comparison::{
    PairwiseTrainMatch, SortingComparison, UnitMatchSummary, compare_sortings,
    compare_spike_trains,
};
pub use correlogram::{Correlogram, compute_autocorrelogram, compute_crosscorrelogram};
pub use evoked::{
    MepMetrics, PsthResult, StimulusTriggeredAverage, compute_psth,
    compute_stimulus_triggered_average, quantify_mep,
};
pub use firing::{
    IsiMetrics, compute_amplitude_cutoff, compute_amplitude_cutoff_with, compute_isi_violations,
    compute_llobet_contamination, compute_presence_ratio, count_refractory_violations, isi_histogram,
};
pub use isolation::{
    compute_d_prime, compute_isolation_distance, compute_silhouette_score, compute_snr,
};
pub use rate::{
    BurstEpoch, FiringRateCurve, compute_instantaneous_firing_rate, detect_burst_epochs,
};

/// ISI-violation refractory threshold (ms), SpikeInterface `isi_violation(isi_threshold_ms=1.5)`.
pub const DEFAULT_ISI_THRESHOLD_MS: f64 = 1.5;
/// Censored period below which ISIs are duplicates (ms), SpikeInterface `min_isi_ms=0`.
pub const DEFAULT_MIN_ISI_MS: f64 = 0.0;
/// Presence-ratio bin (s) and minimum rate ratio, SpikeInterface `presence_ratio(bin_duration_s=60,
/// mean_fr_ratio_thresh=0.0)`.
pub const DEFAULT_PRESENCE_BIN_SEC: f64 = 60.0;
pub const DEFAULT_PRESENCE_MEAN_FR_RATIO: f64 = 0.0;
/// Correlogram bin and half window (ms), SpikeInterface `compute_correlograms(bin_ms=1.0,
/// window_ms=50.0)`.
pub const DEFAULT_CORRELOGRAM_BIN_MS: f32 = 1.0;
pub const DEFAULT_CORRELOGRAM_WINDOW_MS: f32 = 50.0;
/// Spike-matching tolerance (ms) and unit agreement threshold for sorting comparison,
/// SpikeInterface `compare_two_sorters(delta_time=0.4, match_score=0.5)`.
pub const DEFAULT_MATCH_DELTA_MS: f64 = 0.4;
pub const DEFAULT_AGREEMENT_THRESHOLD: f32 = 0.5;

/// Configurable Allen / IBL / Phy quality metric parameters and automated unit classification
/// thresholds (eliminating hardcoded refractory periods and SNR / ISI cutoffs across sorters,
/// storage loaders, and curation).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct QualityCriteria {
    /// Refractory period in milliseconds (default `1.5` ms for IBL/Allen, `2.0` ms in Phy).
    pub refractory_ms: f64,
    /// Censored / dead-time window in milliseconds (default `0.0` ms).
    pub censored_ms: f64,
    /// Minimum spikes required before a unit can be classified above `Noise` (default `3`).
    pub min_spikes: usize,
    /// SNR below which a unit is classified as `Noise` when SNR is finite (default `1.5`).
    pub noise_snr_threshold: f32,
    /// Minimum SNR required for automated `SingleUnit` (`good`) classification (default `3.0`).
    pub sua_snr_threshold: f32,
    /// Maximum Hill et al. ISI violation ratio for `SingleUnit` classification (default `0.5`).
    pub max_isi_violation_ratio: f64,
}

impl Default for QualityCriteria {
    fn default() -> Self {
        Self {
            refractory_ms: DEFAULT_ISI_THRESHOLD_MS,
            censored_ms: DEFAULT_MIN_ISI_MS,
            min_spikes: 3,
            noise_snr_threshold: 1.5,
            sua_snr_threshold: 3.0,
            max_isi_violation_ratio: 0.5,
        }
    }
}

impl QualityCriteria {
    /// Standard Phy curation defaults (`2.0` ms refractory period).
    pub const PHY: Self = Self {
        refractory_ms: 2.0,
        censored_ms: 0.0,
        min_spikes: 3,
        noise_snr_threshold: 1.5,
        sua_snr_threshold: 3.0,
        max_isi_violation_ratio: 0.5,
    };

    /// Automated quality classification from spike count, SNR, and ISI violation ratio.
    /// If `snr` is `NaN` (no noise floor known), leaves non-empty units as `Unsorted`.
    pub fn classify(&self, num_spikes: usize, snr: f32, isi_violation_ratio: f64) -> UnitQualityLabel {
        if num_spikes < self.min_spikes {
            return UnitQualityLabel::Noise;
        }
        if !snr.is_finite() {
            return UnitQualityLabel::Unsorted;
        }
        if snr < self.noise_snr_threshold {
            UnitQualityLabel::Noise
        } else if (isi_violation_ratio.is_nan() || isi_violation_ratio < self.max_isi_violation_ratio)
            && snr >= self.sua_snr_threshold
        {
            UnitQualityLabel::SingleUnit
        } else {
            UnitQualityLabel::MultiUnit
        }
    }
}

