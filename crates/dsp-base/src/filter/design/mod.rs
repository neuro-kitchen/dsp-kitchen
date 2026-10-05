//! Host-side (f64) IIR filter design.
//!
//! Filters are described by a [`FilterSpec`] (what to build and how to run it) and designed into a
//! cascade of second-order sections ([`Sos`]) that the SOS kernel executes. Designs are equivalent
//! to `scipy.signal.butter(..., output="sos")` and `scipy.signal.iirnotch`.

mod butterworth;
mod notch;
mod sos;

pub use butterworth::{butterworth_sos, chebyshev1_sos};
pub use notch::notch_sos;
pub use sos::{Section, Sos, DEFAULT_SETTLING_TOLERANCE};

use thiserror::Error;

/// Frequency band of a Butterworth filter, cutoffs in Hz.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterBand {
    Lowpass(f64),
    Highpass(f64),
    Bandpass(f64, f64),
    Bandstop(f64, f64),
}

/// How a filter is applied along time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterMode {
    /// Causal, one pass (`sosfilt`). Usable on live streams with carried state.
    Forward,
    /// Zero phase: forward then backward pass (`sosfiltfilt`); squares the magnitude response.
    /// Needs future samples, so chunked use requires a right halo.
    #[default]
    ForwardBackward,
}

/// Where a [`FilterMode::Forward`] pass starts. Forward-backward passes always start from the steady
/// state of their first sample, as `sosfiltfilt` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterStart {
    /// As if the input had been zero before the first sample (`sosfilt` without `zi`): a DC level
    /// produces a step transient.
    #[default]
    Rest,
    /// From the steady state of the first sample (`sosfilt` with `zi = sosfilt_zi · x[0]`): no step
    /// transient, so halo windows over DC-heavy data settle with less context.
    SteadyState,
}

/// Filter design.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterDesign {
    /// Butterworth of any order (per edge for band filters, as in scipy).
    Butterworth { order: usize, band: FilterBand },
    /// Chebyshev type I of any order with `ripple_db` of pass-band ripple (`cheby1`).
    Chebyshev1 { order: usize, ripple_db: f64, band: FilterBand },
    /// Second-order notch at `freq_hz` with quality factor `q` (`iirnotch`).
    Notch { freq_hz: f64, q: f64 },
    /// Explicit second-order sections, already designed for the target sample rate.
    Sos(Sos),
}

/// A filter to design, the direction(s) it runs in, and where forward passes start.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterSpec {
    pub design: FilterDesign,
    pub mode: FilterMode,
    pub start: FilterStart,
}

/// Default Butterworth order, matching SpikeInterface's `bandpass_filter` / `highpass_filter`.
pub const DEFAULT_BUTTERWORTH_ORDER: usize = 5;

impl FilterSpec {
    pub fn butterworth(order: usize, band: FilterBand) -> Self {
        Self { design: FilterDesign::Butterworth { order, band }, mode: FilterMode::default(), start: FilterStart::default() }
    }

    pub fn bandpass(low_hz: f64, high_hz: f64) -> Self {
        Self::butterworth(DEFAULT_BUTTERWORTH_ORDER, FilterBand::Bandpass(low_hz, high_hz))
    }

    pub fn highpass(cutoff_hz: f64) -> Self {
        Self::butterworth(DEFAULT_BUTTERWORTH_ORDER, FilterBand::Highpass(cutoff_hz))
    }

    pub fn lowpass(cutoff_hz: f64) -> Self {
        Self::butterworth(DEFAULT_BUTTERWORTH_ORDER, FilterBand::Lowpass(cutoff_hz))
    }

    pub fn bandstop(low_hz: f64, high_hz: f64) -> Self {
        Self::butterworth(DEFAULT_BUTTERWORTH_ORDER, FilterBand::Bandstop(low_hz, high_hz))
    }

    pub fn chebyshev1(order: usize, ripple_db: f64, band: FilterBand) -> Self {
        Self { design: FilterDesign::Chebyshev1 { order, ripple_db, band }, mode: FilterMode::default(), start: FilterStart::default() }
    }

    pub fn notch(freq_hz: f64, q: f64) -> Self {
        Self { design: FilterDesign::Notch { freq_hz, q }, mode: FilterMode::default(), start: FilterStart::default() }
    }

    pub fn sos(sos: Sos) -> Self {
        Self { design: FilterDesign::Sos(sos), mode: FilterMode::default(), start: FilterStart::default() }
    }

    /// Returns the same filter with `order` (Butterworth / Chebyshev; ignored for notch / explicit SOS).
    pub fn with_order(mut self, new_order: usize) -> Self {
        if let FilterDesign::Butterworth { order, .. } | FilterDesign::Chebyshev1 { order, .. } = &mut self.design {
            *order = new_order;
        }
        self
    }

    pub fn with_mode(mut self, mode: FilterMode) -> Self {
        self.mode = mode;
        self
    }

    /// Returns the same filter starting forward passes at `start`.
    pub fn with_start(mut self, start: FilterStart) -> Self {
        self.start = start;
        self
    }

    /// Designs the second-order sections for `sample_rate` Hz, validating the parameters.
    pub fn design(&self, sample_rate: f64) -> Result<Sos, FilterError> {
        match &self.design {
            FilterDesign::Butterworth { order, band } => butterworth_sos(*order, *band, sample_rate),
            FilterDesign::Chebyshev1 { order, ripple_db, band } => chebyshev1_sos(*order, *ripple_db, *band, sample_rate),
            FilterDesign::Notch { freq_hz, q } => notch_sos(*freq_hz, *q, sample_rate),
            FilterDesign::Sos(sos) => {
                sos.validate()?;
                Ok(sos.clone())
            }
        }
    }

    /// `(left, right)` samples needed around a chunk for its interior to match whole-recording
    /// filtering (impulse response decayed below [`DEFAULT_SETTLING_TOLERANCE`]).
    pub fn settling(&self, sample_rate: f64) -> Result<(usize, usize), FilterError> {
        let s = self.design(sample_rate)?.settling_samples(DEFAULT_SETTLING_TOLERANCE);
        Ok(match self.mode {
            FilterMode::Forward => (s, 0),
            FilterMode::ForwardBackward => (s, s),
        })
    }
}

/// Invalid filter parameters.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum FilterError {
    #[error("sample rate must be positive and finite, got {0}")]
    InvalidSampleRate(f64),
    #[error("filter order must be at least 1")]
    InvalidOrder,
    #[error("cutoff {cutoff_hz} Hz must lie in (0, {nyquist_hz}) Hz")]
    CutoffOutOfRange { cutoff_hz: f64, nyquist_hz: f64 },
    #[error("band edges must satisfy low < high, got {low_hz} Hz and {high_hz} Hz")]
    InvalidBand { low_hz: f64, high_hz: f64 },
    #[error("quality factor must be positive and finite, got {0}")]
    InvalidQ(f64),
    #[error("pass-band ripple must be positive and finite dB, got {0}")]
    InvalidRipple(f64),
    #[error("second-order sections must be non-empty, finite, with a0 = 1 and stable poles")]
    InvalidSections,
    #[error("forward-backward filters need future samples and cannot run on a live stream")]
    ForwardBackwardOnLiveStream,
}

pub(crate) fn check_sample_rate(fs: f64) -> Result<(), FilterError> {
    if fs.is_finite() && fs > 0.0 { Ok(()) } else { Err(FilterError::InvalidSampleRate(fs)) }
}

pub(crate) fn check_cutoff(cutoff_hz: f64, fs: f64) -> Result<(), FilterError> {
    let nyquist_hz = fs / 2.0;
    if cutoff_hz.is_finite() && cutoff_hz > 0.0 && cutoff_hz < nyquist_hz {
        Ok(())
    } else {
        Err(FilterError::CutoffOutOfRange { cutoff_hz, nyquist_hz })
    }
}
