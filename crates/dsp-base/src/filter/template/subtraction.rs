//! Template alignment, dynamic scaling, and artifact subtraction.
//!
//! Removes recurring stereotypical waveforms (e.g. ECG cardiac artifacts in EMG,
//! electrical stimulation pulses, or optical transient artifacts) from continuous signals.

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

    /// Applies in-place template subtraction to a 1D continuous signal.
    pub fn apply_1d(&self, signal: &mut [f32], event_indices: &[u64]) {
        assert_eq!(
            self.template_channels, 1,
            "apply_1d requires a 1D template"
        );
        subtract_template_1d(
            signal,
            &self.template,
            event_indices,
            self.center_offset,
            self.max_lag,
            self.dynamic_scaling,
        );
    }

    /// Applies in-place template subtraction to a multi-channel continuous signal [channels, samples].
    pub fn apply_multichannel(
        &self,
        signal: &mut [f32],
        channels: usize,
        samples: usize,
        event_indices: &[u64],
    ) {
        subtract_template_multichannel(
            signal,
            channels,
            samples,
            &self.template,
            self.template_channels,
            self.template_samples,
            event_indices,
            self.center_offset,
            self.max_lag,
            self.dynamic_scaling,
        );
    }
}

/// Subtracts a 1D template from `signal` at each event index with cross-correlation alignment and scaling.
pub fn subtract_template_1d(
    signal: &mut [f32],
    template: &[f32],
    event_indices: &[u64],
    center_offset: usize,
    max_lag: usize,
    dynamic_scaling: bool,
) {
    let t_len = template.len();
    if t_len == 0 || signal.is_empty() || event_indices.is_empty() {
        return;
    }

    let t_energy: f32 = template.iter().map(|&v| v * v).sum();
    if t_energy <= MIN_TEMPLATE_ENERGY {
        return;
    }

    let num_samples = signal.len() as isize;
    let t_len_i = t_len as isize;
    let center_i = center_offset as isize;
    let max_lag_i = max_lag as isize;

    for &ev in event_indices {
        let ev_i = ev as isize;
        let base_start = ev_i - center_i;

        // Step 1: Find best lag via cross-correlation in [-max_lag, +max_lag]
        let mut best_lag = 0isize;
        let mut max_corr = f32::NEG_INFINITY;

        for lag in -max_lag_i..=max_lag_i {
            let start = base_start + lag;
            let end = start + t_len_i;

            if start >= 0 && end <= num_samples {
                let seg = &signal[start as usize..end as usize];
                let corr: f32 = seg.iter().zip(template.iter()).map(|(&s, &t)| s * t).sum();
                if corr > max_corr {
                    max_corr = corr;
                    best_lag = lag;
                }
            }
        }

        let start = base_start + best_lag;
        let end = start + t_len_i;
        if start < 0 || end > num_samples {
            continue;
        }

        let seg = &mut signal[start as usize..end as usize];

        // Step 2: Compute scaling factor alpha = (seg . template) / (template . template)
        let alpha = if dynamic_scaling {
            let dot: f32 = seg.iter().zip(template.iter()).map(|(&s, &t)| s * t).sum();
            (dot / t_energy).max(0.0) // Non-negative scaling for artifact cancellation
        } else {
            1.0
        };

        // Step 3: In-place subtraction: seg -= alpha * template
        for (s, &t) in seg.iter_mut().zip(template.iter()) {
            *s -= alpha * t;
        }
    }
}

/// Subtracts a multi-channel template from multi-channel `signal` [channels, samples].
pub fn subtract_template_multichannel(
    signal: &mut [f32],
    channels: usize,
    samples: usize,
    template: &[f32],
    template_channels: usize,
    template_samples: usize,
    event_indices: &[u64],
    center_offset: usize,
    max_lag: usize,
    dynamic_scaling: bool,
) {
    assert_eq!(signal.len(), channels * samples);
    assert_eq!(template.len(), template_channels * template_samples);
    assert_eq!(
        channels, template_channels,
        "Channel count must match between signal and template"
    );

    if template_samples == 0 || samples == 0 || event_indices.is_empty() {
        return;
    }

    let t_len_i = template_samples as isize;
    let center_i = center_offset as isize;
    let max_lag_i = max_lag as isize;
    let num_samples_i = samples as isize;

    // Precalculate energy per channel
    let mut t_energies = vec![0.0f32; channels];
    for ch in 0..channels {
        let offset = ch * template_samples;
        t_energies[ch] = template[offset..offset + template_samples]
            .iter()
            .map(|&v| v * v)
            .sum();
    }

    for &ev in event_indices {
        let ev_i = ev as isize;
        let base_start = ev_i - center_i;

        // Step 1: Joint cross-correlation across all channels to find global optimal lag
        let mut best_lag = 0isize;
        let mut max_corr = f32::NEG_INFINITY;

        for lag in -max_lag_i..=max_lag_i {
            let start = base_start + lag;
            let end = start + t_len_i;

            if start >= 0 && end <= num_samples_i {
                let mut joint_corr = 0.0f32;
                for ch in 0..channels {
                    let sig_offset = ch * samples + (start as usize);
                    let tmpl_offset = ch * template_samples;
                    let seg = &signal[sig_offset..sig_offset + template_samples];
                    let tmpl = &template[tmpl_offset..tmpl_offset + template_samples];
                    joint_corr += seg.iter().zip(tmpl.iter()).map(|(&s, &t)| s * t).sum::<f32>();
                }

                if joint_corr > max_corr {
                    max_corr = joint_corr;
                    best_lag = lag;
                }
            }
        }

        let start = base_start + best_lag;
        let end = start + t_len_i;
        if start < 0 || end > num_samples_i {
            continue;
        }

        // Step 2 & 3: Scale and subtract on each channel
        for ch in 0..channels {
            let energy = t_energies[ch];
            if energy <= MIN_TEMPLATE_ENERGY {
                continue;
            }

            let sig_offset = ch * samples + (start as usize);
            let tmpl_offset = ch * template_samples;
            let tmpl = &template[tmpl_offset..tmpl_offset + template_samples];
            let seg = &mut signal[sig_offset..sig_offset + template_samples];

            let alpha = if dynamic_scaling {
                let dot: f32 = seg.iter().zip(tmpl.iter()).map(|(&s, &t)| s * t).sum();
                (dot / energy).max(0.0)
            } else {
                1.0
            };

            for (s, &t) in seg.iter_mut().zip(tmpl.iter()) {
                *s -= alpha * t;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_subtraction_exact() {
        let template = vec![1.0, 5.0, 10.0, -8.0, 2.0];
        let mut signal = vec![0.0f32; 50];

        // Inject template at sample 20 (center_offset = 2 so peak 10.0 is at sample 20)
        let center_offset = 2;
        let event_sample = 20u64;
        let start = (event_sample as usize) - center_offset;
        for i in 0..template.len() {
            signal[start + i] += template[i];
        }

        // Before subtraction, peak is 10.0
        assert_eq!(signal[20], 10.0);

        let filter = TemplateFilter::new_1d(template.clone(), center_offset)
            .with_max_lag(3)
            .with_dynamic_scaling(true);

        filter.apply_1d(&mut signal, &[event_sample]);

        // After subtraction, signal should be 0.0 everywhere
        for &val in &signal {
            assert!(val.abs() < 1e-5, "Expected residual near 0, got {}", val);
        }
    }

    #[test]
    fn test_template_subtraction_with_lag_and_amplitude_scale() {
        let template = vec![1.0, 4.0, 12.0, -6.0, 2.0];
        let mut signal = vec![0.0f32; 60];

        let center_offset = 2;
        let nominal_event = 25u64;
        // Inject with +2 samples lag and 2.5x amplitude scaling!
        let actual_start = ((nominal_event as usize) - center_offset) + 2;
        let scale = 2.5f32;
        for i in 0..template.len() {
            signal[actual_start + i] += scale * template[i];
        }

        let filter = TemplateFilter::new_1d(template, center_offset)
            .with_max_lag(5)
            .with_dynamic_scaling(true);

        filter.apply_1d(&mut signal, &[nominal_event]);

        // Cross-correlation should find the +2 lag and scale 2.5, canceling the artifact!
        for &val in &signal {
            assert!(val.abs() < 1e-4, "Expected residual near 0, got {}", val);
        }
    }

    #[test]
    fn test_multichannel_template_subtraction() {
        let ch0_template = vec![0.0, 10.0, 0.0];
        let ch1_template = vec![0.0, 5.0, 0.0];
        let mut template = ch0_template;
        template.extend(ch1_template);

        let channels = 2;
        let samples = 20;
        let mut signal = vec![0.0f32; channels * samples];

        let event = 10u64;
        let center = 1;
        // Inject at sample 10 on both channels
        signal[0 * samples + 10] = 10.0;
        signal[1 * samples + 10] = 5.0;

        let filter = TemplateFilter::new_multichannel(template, channels, 3, center)
            .with_max_lag(2)
            .with_dynamic_scaling(true);

        filter.apply_multichannel(&mut signal, channels, samples, &[event]);

        assert!(signal[0 * samples + 10].abs() < 1e-5);
        assert!(signal[1 * samples + 10].abs() < 1e-5);
    }
}
