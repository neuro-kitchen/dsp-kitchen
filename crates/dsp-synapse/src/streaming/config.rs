//! Configuration and dynamic halo computation for out-of-core spike sorting.

use dsp_base::pipeline::Pipeline;

/// Half-window lobe margin for Lanczos/sinc fractional resampling (`W = 8` samples).
pub const SINC_RESAMPLE_MARGIN: usize = 8;

/// Configuration for [`super::StreamingSpikeRunner`].
#[derive(Debug, Clone)]
pub struct StreamingSortConfig {
    /// Duration of each valid interior streaming batch in seconds (default: `10.0` s).
    pub batch_duration_sec: f64,
    /// Duration of the initial noise-floor calibration window in seconds (default: `5.0` s).
    pub calibration_duration_sec: f64,
    /// Negative threshold multiplier in units of Quiroga $\sigma_n$ (default: `5.0`).
    pub threshold_factor: f32,
    /// Refractory period in milliseconds (default: `1.0` ms).
    pub refractory_ms: f64,
    /// Spatial deduplication radius in micrometers (default: `150.0` $\mu\text{m}$).
    pub spatial_radius_um: f32,
    /// Number of $K$-nearest neighbor channels extracted per waveform snippet (default: `4`).
    pub k_neighbors: usize,
    /// Pre-trough snippet window in milliseconds (default: `1.0` ms).
    pub pre_ms: f64,
    /// Post-trough snippet window in milliseconds (default: `2.0` ms).
    pub post_ms: f64,
    /// Whether to apply continuous sub-sample Lanczos/sinc realignment (default: `true`).
    pub apply_sinc_shift: bool,
}

impl Default for StreamingSortConfig {
    fn default() -> Self {
        Self {
            batch_duration_sec: 10.0,
            calibration_duration_sec: 5.0,
            threshold_factor: 5.0,
            refractory_ms: 1.0,
            spatial_radius_um: 150.0,
            k_neighbors: 4,
            pre_ms: 1.0,
            post_ms: 2.0,
            apply_sinc_shift: true,
        }
    }
}

impl StreamingSortConfig {
    /// Refractory period in samples at `sample_rate` Hz.
    #[inline]
    pub fn refractory_samples(&self, sample_rate: f64) -> usize {
        ((self.refractory_ms.max(0.1) * 1e-3) * sample_rate.max(1.0)).round() as usize
    }

    /// Pre-trough snippet length in samples at `sample_rate` Hz.
    #[inline]
    pub fn pre_samples(&self, sample_rate: f64) -> usize {
        ((self.pre_ms.max(0.1) * 1e-3) * sample_rate.max(1.0)).round() as usize
    }

    /// Post-trough snippet length in samples at `sample_rate` Hz.
    #[inline]
    pub fn post_samples(&self, sample_rate: f64) -> usize {
        ((self.post_ms.max(0.1) * 1e-3) * sample_rate.max(1.0)).round() as usize
    }

    /// Total snippet length (`pre_samples + post_samples`) at `sample_rate` Hz.
    #[inline]
    pub fn snippet_samples(&self, sample_rate: f64) -> usize {
        self.pre_samples(sample_rate) + self.post_samples(sample_rate)
    }

    /// Valid interior batch size in samples at `sample_rate` Hz.
    #[inline]
    pub fn batch_samples(&self, sample_rate: f64) -> u64 {
        ((self.batch_duration_sec.max(0.5) * sample_rate.max(1.0)).round() as u64).max(1)
    }

    /// Computes the required `(left_halo, right_halo)` in samples at `sample_rate` Hz
    /// given the filter stages in `pipeline`:
    ///
    /// - `left_halo = pipeline.settling_samples(fs) + pre_samples(fs) + refractory_samples(fs) + SINC_RESAMPLE_MARGIN`
    /// - `right_halo = post_samples(fs) + refractory_samples(fs) + SINC_RESAMPLE_MARGIN`
    pub fn compute_halos(&self, sample_rate: f64, pipeline: &Pipeline) -> (u64, u64) {
        let settling = pipeline.settling_samples(sample_rate);
        let pre = self.pre_samples(sample_rate);
        let post = self.post_samples(sample_rate);
        let refrac = self.refractory_samples(sample_rate);

        let left = (settling + pre + refrac + SINC_RESAMPLE_MARGIN) as u64;
        let right = (post + refrac + SINC_RESAMPLE_MARGIN) as u64;
        (left, right)
    }
}
