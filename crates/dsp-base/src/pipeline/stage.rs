use crate::filter::design::{FilterError, FilterSpec, Sos};

/// Individual processing stage within an in-VRAM DSP pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineStage {
    /// Linear scaling and offset: $y = \alpha x + \beta$
    Scale { alpha: f32, beta: f32 },
    /// Constant baseline subtraction: $y = x - \text{baseline}$
    SubtractBaseline { baseline_uv: f32 },
    /// Sample clamping / rectification: $\text{clamp}(x, \min, \max)$
    Clamp { min: f32, max: f32 },
    /// IIR filter (Butterworth of any order and band, notch, or explicit sections), run forward or
    /// forward-backward. Build with [`PipelineStage::bandpass`], [`PipelineStage::highpass`], ….
    Filter(FilterSpec),
    /// Common Average Referencing across all channels
    CommonAverageReference,
    /// 9-point branchless sorting network median filter
    Median9p,
    /// Discrete Teager-Kaiser Energy Operator: $\Psi[x_t] = x_t^2 - x_{t-1}x_{t+1}$
    TeagerKaiser,
}

impl PipelineStage {
    /// Butterworth band-pass, default order 5, zero phase.
    pub fn bandpass(low_hz: f64, high_hz: f64) -> Self {
        Self::Filter(FilterSpec::bandpass(low_hz, high_hz))
    }

    /// Butterworth high-pass, default order 5, zero phase (e.g. `highpass(300.0)` for spikes).
    pub fn highpass(cutoff_hz: f64) -> Self {
        Self::Filter(FilterSpec::highpass(cutoff_hz))
    }

    /// Butterworth low-pass, default order 5, zero phase (e.g. `lowpass(300.0)` for LFP).
    pub fn lowpass(cutoff_hz: f64) -> Self {
        Self::Filter(FilterSpec::lowpass(cutoff_hz))
    }

    /// Butterworth band-stop, default order 5, zero phase.
    pub fn bandstop(low_hz: f64, high_hz: f64) -> Self {
        Self::Filter(FilterSpec::bandstop(low_hz, high_hz))
    }

    /// Second-order notch (`iirnotch`), zero phase.
    pub fn notch(freq_hz: f64, q: f64) -> Self {
        Self::Filter(FilterSpec::notch(freq_hz, q))
    }

    /// Explicit second-order sections, zero phase.
    pub fn sos(sos: Sos) -> Self {
        Self::Filter(FilterSpec::sos(sos))
    }

    /// `(left, right)` samples of context this stage needs around a chunk at `sample_rate` Hz so
    /// the chunk interior matches whole-recording processing.
    ///
    /// - `Filter`: from the designed filter's pole radii ([`FilterSpec::settling`]); forward-backward
    ///   needs both sides.
    /// - `Median9p`: 4 each side (half of the 9-point window); `TeagerKaiser`: 1 each side.
    /// - Pointwise / spatial stages: none.
    pub fn settling(&self, sample_rate: f64) -> Result<(usize, usize), FilterError> {
        Ok(match self {
            PipelineStage::Filter(spec) => spec.settling(sample_rate)?,
            PipelineStage::Median9p => (4, 4),
            PipelineStage::TeagerKaiser => (1, 1),
            PipelineStage::Scale { .. }
            | PipelineStage::SubtractBaseline { .. }
            | PipelineStage::Clamp { .. }
            | PipelineStage::CommonAverageReference => (0, 0),
        })
    }
}
