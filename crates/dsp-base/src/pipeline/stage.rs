/// Individual processing stage within an in-VRAM DSP pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineStage {
    /// Linear scaling and offset: $y = \alpha x + \beta$
    Scale { alpha: f32, beta: f32 },
    /// Constant baseline subtraction: $y = x - \text{baseline}$
    SubtractBaseline { baseline_uv: f32 },
    /// Sample clamping / rectification: $\text{clamp}(x, \min, \max)$
    Clamp { min: f32, max: f32 },
    /// 2nd-order narrow-band notch filter (e.g. 50 Hz or 60 Hz electrical hum)
    Notch { freq_hz: f64, q: f64 },
    /// 4th-order cascaded Butterworth bandpass filter
    Bandpass { low_hz: f64, high_hz: f64 },
    /// Common Average Referencing across all channels
    CommonAverageReference,
    /// 9-point branchless sorting network median filter
    Median9p,
    /// Discrete Teager-Kaiser Energy Operator: $\Psi[x_t] = x_t^2 - x_{t-1}x_{t+1}$
    TeagerKaiser,
}

impl PipelineStage {
    /// Returns the number of boundary settling / lookback samples required by this stage
    /// at `sample_rate` Hz to prevent edge transients at chunk boundaries.
    ///
    /// - `Bandpass { low_hz, .. }`: 4 periods of the high-pass cutoff ($\lceil 4 \cdot f_s / f_{\text{low}} \rceil$),
    ///   matching SpikeInterface's auto margin for 4th-order Butterworth transient decay ($< 0.1\%$).
    /// - `Notch { freq_hz, .. }`: 3 periods of the notch center frequency ($\lceil 3 \cdot f_s / f_0 \rceil$).
    /// - `Median9p`: 4 samples (half-stencil of 9-point window).
    /// - `TeagerKaiser`: 1 sample ($\Psi[x_t] = x_t^2 - x_{t-1}x_{t+1}$).
    /// - Pointwise / spatial stages (`Scale`, `SubtractBaseline`, `Clamp`, `CommonAverageReference`): 0 samples.
    pub fn settling_samples(&self, sample_rate: f64) -> usize {
        let fs = sample_rate.max(1.0);
        match self {
            PipelineStage::Bandpass { low_hz, .. } => {
                let f_low = low_hz.max(0.5);
                ((4.0 / f_low) * fs).ceil() as usize
            }
            PipelineStage::Notch { freq_hz, .. } => {
                let f0 = freq_hz.max(1.0);
                ((3.0 / f0) * fs).ceil() as usize
            }
            PipelineStage::Median9p => 4,
            PipelineStage::TeagerKaiser => 1,
            PipelineStage::Scale { .. }
            | PipelineStage::SubtractBaseline { .. }
            | PipelineStage::Clamp { .. }
            | PipelineStage::CommonAverageReference => 0,
        }
    }
}

