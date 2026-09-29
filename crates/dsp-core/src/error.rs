use thiserror::Error;

pub type DspResult<T> = Result<T, DspError>;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum DspError {
    #[error("Invalid channel index: requested {channel}, total channels {total}")]
    InvalidChannel { channel: usize, total: usize },

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },

    #[error("Invalid sample rate: {rate_hz} Hz (must be positive and non-zero)")]
    InvalidSampleRate { rate_hz: f64 },

    #[error("Buffer overrun: capacity {capacity}, attempted {requested}")]
    BufferOverrun { capacity: usize, requested: usize },

    #[error("Buffer underrun: available {available}, attempted {requested}")]
    BufferUnderrun { available: usize, requested: usize },

    #[error("Device compute error: {0}")]
    ComputeError(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}
