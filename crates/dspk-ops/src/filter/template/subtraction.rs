//! Template alignment, dynamic scaling, and artifact subtraction: the filter's settings.
//!
//! Removes recurring stereotypical waveforms (e.g. ECG cardiac artifacts in EMG,
//! electrical stimulation pulses, or optical transient artifacts) from continuous signals.
//! The one implementation runs on the device ([`TemplateFilter::apply_device`]); the host version
//! in `reference.rs` exists only as the tests' oracle.

/// Default search radius (samples, each side) for aligning a template to an event.
pub const DEFAULT_MAX_LAG: usize = 8;

/// Templates with less energy (sum of squares) than this are treated as empty and not subtracted.
pub const MIN_TEMPLATE_ENERGY: f32 = 1e-12;

/// Configurable template subtraction filter.
#[derive(Debug, Clone)]
pub struct TemplateFilter {
    /// Prototype waveform template, `[samples]` or `[channels, samples]`.
    pub template: Vec<f32>,
    /// Number of channels in the template (1 for single-channel).
    pub template_channels: usize,
    /// Number of samples in the template.
    pub template_samples: usize,
    /// Anchor point within the template (e.g., peak/trigger offset from start).
    pub center_offset: usize,
    /// Maximum search radius in samples for cross-correlation lag alignment (+/- max_lag).
    pub max_lag: usize,
    /// Whether to dynamically scale the template via least-squares dot product:
    /// alpha = (signal . template) / (template . template).
    pub dynamic_scaling: bool,
}

impl TemplateFilter {
    /// Creates a 1D template filter.
    pub fn new_1d(template: Vec<f32>, center_offset: usize) -> Self {
        let len = template.len();
        Self {
            template,
            template_channels: 1,
            template_samples: len,
            center_offset,
            max_lag: DEFAULT_MAX_LAG,
            dynamic_scaling: true,
        }
    }

    /// Creates a multi-channel template filter [channels, samples].
    pub fn new_multichannel(
        template: Vec<f32>,
        channels: usize,
        samples: usize,
        center_offset: usize,
    ) -> Self {
        assert_eq!(template.len(), channels * samples);
        Self {
            template,
            template_channels: channels,
            template_samples: samples,
            center_offset,
            max_lag: DEFAULT_MAX_LAG,
            dynamic_scaling: true,
        }
    }

    /// Sets the maximum lag search radius in samples.
    pub fn with_max_lag(mut self, max_lag: usize) -> Self {
        self.max_lag = max_lag;
        self
    }

    /// Sets whether dynamic least-squares amplitude scaling is enabled.
    pub fn with_dynamic_scaling(mut self, enabled: bool) -> Self {
        self.dynamic_scaling = enabled;
        self
    }
}
