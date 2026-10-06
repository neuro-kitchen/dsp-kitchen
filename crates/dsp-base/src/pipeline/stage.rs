use crate::filter::design::{FilterError, FilterSpec, Sos};
use crate::core::EdgeMode;
use crate::filter::fir::gaussian::{gaussian_radius, GAUSSIAN_DEFAULT_EDGE, GAUSSIAN_TRUNCATE};
use crate::filter::non_linear::{MEDIAN9_RADIUS, MEDIAN_DEFAULT_EDGE, TEAGER_KAISER_DEFAULT_EDGE};
use crate::spatial::{SpatialWhitening, SurfaceLaplacian};

/// Individual processing stage within an in-VRAM DSP pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineStage {
    /// Linear scaling and offset: $y = \alpha x + \beta$
    Scale { alpha: f32, beta: f32 },
    /// Constant baseline subtraction: $y = x - \text{baseline}$ (in the signal's unit)
    SubtractBaseline { baseline: f32 },
    /// Sample clamping / rectification: $\text{clamp}(x, \min, \max)$
    Clamp { min: f32, max: f32 },
    /// IIR filter (Butterworth of any order and band, notch, or explicit sections), run forward or
    /// forward-backward. Build with [`PipelineStage::bandpass`], [`PipelineStage::highpass`], ….
    Filter(FilterSpec),
    /// Common Average Referencing across all channels
    CommonAverageReference,
    /// Spatial whitening ($\mathbf{W}_{\text{ZCA}}$ or Local $K$-NN) across channels
    SpatialWhitening(SpatialWhitening),
    /// 2D Surface Laplacian (double-differential spatial filter) across channels
    SurfaceLaplacian(SurfaceLaplacian),
    /// Zero-phase 1D Gaussian temporal smoothing with standard deviation `sigma_samples`, cut at
    /// `GAUSSIAN_TRUNCATE` σ; samples past the ends come from `edge`.
    GaussianSmooth { sigma_samples: f32, edge: EdgeMode },
    /// Running median over an odd `width` (≤ `MAX_MEDIAN_WIDTH`); width 9 runs the branch-free
    /// `med9` network. Samples past the ends come from `edge`.
    Median { width: usize, edge: EdgeMode },
    /// Discrete Teager-Kaiser Energy Operator: $\Psi[x_t] = x_t^2 - x_{t-1}x_{t+1}$; the neighbours
    /// of the end samples come from `edge`.
    TeagerKaiser { edge: EdgeMode },
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

    /// Common Average Referencing across all channels.
    pub fn common_average_reference() -> Self {
        Self::CommonAverageReference
    }

    /// Explicit second-order sections, zero phase.
    pub fn sos(sos: Sos) -> Self {
        Self::Filter(FilterSpec::sos(sos))
    }

    /// Zero-phase Gaussian smoothing with reflected edges (`scipy.ndimage.gaussian_filter1d`).
    pub fn gaussian_smooth(sigma_samples: f32) -> Self {
        Self::GaussianSmooth { sigma_samples, edge: GAUSSIAN_DEFAULT_EDGE }
    }

    /// Running median over an odd `width` with zero-padded edges (`scipy.signal.medfilt`).
    pub fn median(width: usize) -> Self {
        Self::Median { width, edge: MEDIAN_DEFAULT_EDGE }
    }

    /// 9-point running median (`scipy.signal.medfilt(x, 9)`).
    pub fn median9() -> Self {
        Self::median(2 * MEDIAN9_RADIUS + 1)
    }

    /// Teager-Kaiser energy with reflected neighbours at the ends.
    pub fn teager_kaiser() -> Self {
        Self::TeagerKaiser { edge: TEAGER_KAISER_DEFAULT_EDGE }
    }

    /// `(left, right)` samples of context this stage needs around a chunk at `sample_rate` Hz so
    /// the chunk interior matches whole-recording processing.
    ///
    /// - `Filter`: from the designed filter's pole radii ([`FilterSpec::settling`]); forward-backward
    ///   needs both sides.
    /// - `GaussianSmooth`: the kernel radius ([`gaussian_radius`] at [`GAUSSIAN_TRUNCATE`]) each side.
    /// - `Median`: `width / 2` each side; `TeagerKaiser`: 1 each side.
    /// - Pointwise / spatial stages: none.
    pub fn settling(&self, sample_rate: f64) -> Result<(usize, usize), FilterError> {
        Ok(match self {
            PipelineStage::Filter(spec) => spec.settling(sample_rate)?,
            PipelineStage::GaussianSmooth { sigma_samples, .. } => {
                let r = gaussian_radius(*sigma_samples, GAUSSIAN_TRUNCATE);
                (r, r)
            }
            PipelineStage::Median { width, .. } => (width / 2, width / 2),
            PipelineStage::TeagerKaiser { .. } => (1, 1),
            PipelineStage::Scale { .. }
            | PipelineStage::SubtractBaseline { .. }
            | PipelineStage::Clamp { .. }
            | PipelineStage::CommonAverageReference
            | PipelineStage::SpatialWhitening(_)
            | PipelineStage::SurfaceLaplacian(_) => (0, 0),
        })
    }
}
