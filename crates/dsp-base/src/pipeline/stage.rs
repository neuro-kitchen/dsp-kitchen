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
