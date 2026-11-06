//! The one error type of dspk-ops.

use dspk_core::DspError;
use thiserror::Error;

/// `Result` of dspk-ops.
pub type OpsResult<T> = Result<T, OpsError>;

/// Why a dspk-ops operation could not run: invalid filter parameters, or an error from dspk-core
/// (shapes, formats).
#[derive(Debug, Clone, PartialEq, Error)]
pub enum OpsError {
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
    #[error("channel order: {0}")]
    InvalidChannelOrder(String),
    #[error(transparent)]
    Core(#[from] DspError),
}

/// Lets `?` pass an [`OpsError`] up through code that returns dspk-core's `DspResult` (readers,
/// sorters): core errors come back as they were, the others as `InvalidConfig`.
impl From<OpsError> for DspError {
    fn from(e: OpsError) -> Self {
        match e {
            OpsError::Core(e) => e,
            other => DspError::InvalidConfig(other.to_string()),
        }
    }
}
