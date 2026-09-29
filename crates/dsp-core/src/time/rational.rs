use num_rational::Ratio;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Duration;
use crate::error::{DspError, DspResult};

/// Exact rational representation of time (seconds as numerator/denominator).
/// Prevents floating-point accumulation drift over long electrophysiology recordings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RationalTime {
    seconds: Ratio<u64>,
}

impl RationalTime {
    pub const ZERO: Self = Self {
        seconds: Ratio::new_raw(0, 1),
    };

    pub fn new(numerator: u64, denominator: u64) -> DspResult<Self> {
        if denominator == 0 {
            return Err(DspError::InvalidConfig("Denominator cannot be 0".into()));
        }
        Ok(Self {
            seconds: Ratio::new(numerator, denominator),
        })
    }

    pub fn from_samples(sample_index: u64, sample_rate_hz: u64) -> DspResult<Self> {
        Self::new(sample_index, sample_rate_hz)
    }

    pub fn as_seconds_f64(&self) -> f64 {
        *self.seconds.numer() as f64 / *self.seconds.denom() as f64
    }

    pub fn as_duration(&self) -> Duration {
        Duration::from_secs_f64(self.as_seconds_f64())
    }

    pub fn to_sample_index(&self, sample_rate_hz: u64) -> u64 {
        let sample_ratio = self.seconds * Ratio::from_integer(sample_rate_hz);
        sample_ratio.to_integer()
    }
}

impl fmt::Display for RationalTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.6}s ({}/{})",
            self.as_seconds_f64(),
            self.seconds.numer(),
            self.seconds.denom()
        )
    }
}
