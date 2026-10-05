//! Configuration and dynamic halo computation for out-of-core streaming detection.

use dsp_base::peaks::DistanceRule;
use dsp_base::pipeline::Pipeline;
use dsp_core::{DspError, DspResult};

use crate::detection::SpikePolarity;

/// Samples of halo the sub-sample realignment needs beyond a snippet: the windowed-sinc radius of
/// extraction (one source of truth, [`crate::extraction::SINC_KERNEL_RADIUS`]).
pub const SINC_RESAMPLE_MARGIN: usize = crate::extraction::SINC_KERNEL_RADIUS;

/// Shortest refractory period and snippet sides (ms), shortest batch and calibration (s).
const MIN_WINDOW_MS: f64 = 0.1;
const MIN_BATCH_SEC: f64 = 0.5;
const MIN_CALIBRATION_SEC: f64 = 0.1;
/// Sample rate floor used in the conversions (Hz; a real rate is far above it).
const MIN_SAMPLE_RATE_HZ: f64 = 1.0;

/// Configuration for [`super::StreamingDetector`].
#[derive(Debug, Clone)]
pub struct StreamingDetectionConfig {
    /// Duration of each valid interior streaming batch in seconds (default: `10.0` s).
    pub batch_duration_sec: f64,
    /// Total duration of the noise-floor calibration in seconds (default: `5.0` s), split into
    /// `calibration_chunks` chunks spread evenly over the recording.
    pub calibration_duration_sec: f64,
    /// Number of calibration chunks; σ per channel is the median of the per-chunk estimates
    /// (default: `5`).
    pub calibration_chunks: usize,
    /// Threshold multiplier in units of Quiroga $\sigma_n$ (default: `5.0`).
    pub threshold_factor: f32,
    /// Refractory period in milliseconds (default: `1.0` ms).
    pub refractory_ms: f64,
    /// Which extrema are spikes (default: negative troughs).
    pub polarity: SpikePolarity,
    /// How crossings nearer than the refractory period are resolved (default:
    /// [`DistanceRule::LocallyExclusive`], for which streaming equals whole-recording detection;
    /// [`DistanceRule::Scipy`] can differ at batch edges).
    pub distance_rule: DistanceRule,
    /// Spatial deduplication radius in micrometers (default: `150.0` $\mu\text{m}$).
    pub spatial_radius_um: f32,
    /// Number of $K$-nearest neighbor channels extracted per waveform snippet (default: `4`).
    pub k_neighbors: usize,
    /// Pre-trough snippet window in milliseconds (default: `1.0` ms).
    pub pre_ms: f64,
    /// Post-trough snippet window in milliseconds (default: `2.0` ms).
    pub post_ms: f64,
    /// Whether to apply sub-sample windowed-sinc (Blackman-Harris) realignment (default: `true`).
    pub apply_sinc_shift: bool,
}

impl Default for StreamingDetectionConfig {
    fn default() -> Self {
        Self {
            batch_duration_sec: 10.0,
            calibration_duration_sec: 5.0,
            calibration_chunks: 5,
            threshold_factor: 5.0,
            refractory_ms: 1.0,
            polarity: SpikePolarity::Negative,
            distance_rule: DistanceRule::LocallyExclusive,
            spatial_radius_um: 150.0,
            k_neighbors: 4,
            pre_ms: 1.0,
            post_ms: 2.0,
            apply_sinc_shift: true,
        }
    }
}

impl StreamingDetectionConfig {
    /// Refractory period in samples at `sample_rate` Hz.
    #[inline]
    pub fn refractory_samples(&self, sample_rate: f64) -> usize {
        ((self.refractory_ms.max(MIN_WINDOW_MS) * 1e-3) * sample_rate.max(MIN_SAMPLE_RATE_HZ)).round() as usize
    }

    /// Pre-trough snippet length in samples at `sample_rate` Hz.
    #[inline]
    pub fn pre_samples(&self, sample_rate: f64) -> usize {
        ((self.pre_ms.max(MIN_WINDOW_MS) * 1e-3) * sample_rate.max(MIN_SAMPLE_RATE_HZ)).round() as usize
    }

    /// Post-trough snippet length in samples at `sample_rate` Hz.
    #[inline]
    pub fn post_samples(&self, sample_rate: f64) -> usize {
        ((self.post_ms.max(MIN_WINDOW_MS) * 1e-3) * sample_rate.max(MIN_SAMPLE_RATE_HZ)).round() as usize
    }

    /// Total snippet length (`pre_samples + post_samples`) at `sample_rate` Hz.
    #[inline]
    pub fn snippet_samples(&self, sample_rate: f64) -> usize {
        self.pre_samples(sample_rate) + self.post_samples(sample_rate)
    }

    /// Valid interior batch size in samples at `sample_rate` Hz.
    #[inline]
    pub fn batch_samples(&self, sample_rate: f64) -> u64 {
        ((self.batch_duration_sec.max(MIN_BATCH_SEC) * sample_rate.max(MIN_SAMPLE_RATE_HZ)).round() as u64).max(1)
    }

    /// Global sample range where spikes are detected: a full snippet plus the realignment margin
    /// fits inside the recording.
    pub fn detection_range(&self, sample_rate: f64, total_samples: u64) -> std::ops::Range<u64> {
        let margin = crate::extraction::extraction_margin(self.apply_sinc_shift);
        let start = (self.pre_samples(sample_rate) + margin) as u64;
        let end = total_samples.saturating_sub((self.post_samples(sample_rate) + margin) as u64);
        start..end.max(start)
    }

    /// Calibration chunks: `calibration_chunks` ranges totalling `calibration_duration_sec`,
    /// spread evenly across `0..total_samples` (fewer / shorter for short recordings).
    pub fn calibration_chunks(&self, sample_rate: f64, total_samples: u64) -> Vec<std::ops::Range<u64>> {
        let k = self.calibration_chunks.max(1) as u64;
        let total_len = ((self.calibration_duration_sec.max(MIN_CALIBRATION_SEC) * sample_rate.max(MIN_SAMPLE_RATE_HZ)).round() as u64).max(k);
        let len = (total_len / k).min(total_samples);
        if len == 0 {
            return Vec::new();
        }
        let span = total_samples - len;
        (0..k)
            .map(|i| {
                let start = if k == 1 { 0 } else { span * i / (k - 1) };
                start..start + len
            })
            .collect()
    }

    /// Computes the required `(left_halo, right_halo)` in samples at `sample_rate` Hz
    /// given the filter stages in `pipeline`:
    ///
    /// - `left_halo = settling_left + pre_samples(fs) + refractory_samples(fs) + SINC_RESAMPLE_MARGIN`
    /// - `right_halo = settling_right + post_samples(fs) + refractory_samples(fs) + SINC_RESAMPLE_MARGIN`
    ///
    /// where `(settling_left, settling_right) = pipeline.settling(fs)` (forward-backward filters need
    /// both sides).
    pub fn compute_halos(&self, sample_rate: f64, pipeline: &Pipeline) -> DspResult<(u64, u64)> {
        let (settle_left, settle_right) = pipeline
            .settling(sample_rate)
            .map_err(|e| DspError::InvalidConfig(e.to_string()))?;
        let pre = self.pre_samples(sample_rate);
        let post = self.post_samples(sample_rate);
        let refrac = self.refractory_samples(sample_rate);

        let left = (settle_left + pre + refrac + SINC_RESAMPLE_MARGIN) as u64;
        let right = (settle_right + post + refrac + SINC_RESAMPLE_MARGIN) as u64;
        Ok((left, right))
    }
}
