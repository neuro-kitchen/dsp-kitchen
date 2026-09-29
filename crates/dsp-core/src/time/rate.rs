use serde::{Deserialize, Serialize};
use std::time::Duration;
use crate::error::{DspError, DspResult};

/// Authoritative sampling rate definition.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SampleRate {
    rate_hz: f64,
}

impl SampleRate {
    pub const STANDARD_AUDIO: f64 = 48_000.0;

    pub fn new(rate_hz: f64) -> DspResult<Self> {
        if rate_hz <= 0.0 || rate_hz.is_nan() || rate_hz.is_infinite() {
            return Err(DspError::InvalidSampleRate { rate_hz });
        }
        Ok(Self { rate_hz })
    }

    pub fn rate_hz(&self) -> f64 {
        self.rate_hz
    }

    pub fn nyquist_hz(&self) -> f64 {
        self.rate_hz / 2.0
    }

    pub fn sample_period_seconds(&self) -> f64 {
        1.0 / self.rate_hz
    }

    pub fn duration_for_samples(&self, num_samples: usize) -> Duration {
        Duration::from_secs_f64(num_samples as f64 / self.rate_hz)
    }

    pub fn samples_for_duration(&self, duration: Duration) -> usize {
        (duration.as_secs_f64() * self.rate_hz).round() as usize
    }
}
