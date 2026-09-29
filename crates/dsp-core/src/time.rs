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
        write!(f, "{:.6}s ({}/{})", self.as_seconds_f64(), self.seconds.numer(), self.seconds.denom())
    }
}

/// Authoritative sampling rate definition.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SampleRate {
    rate_hz: f64,
}

impl SampleRate {
    pub const NEUROPIXELS_AP: f64 = 30_000.0;
    pub const NEUROPIXELS_LFP: f64 = 2_500.0;
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

/// A continuous interval of time between start and end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: RationalTime,
    pub end: RationalTime,
}

impl TimeRange {
    pub fn new(start: RationalTime, end: RationalTime) -> DspResult<Self> {
        if start > end {
            return Err(DspError::InvalidConfig(format!(
                "TimeRange start ({}) cannot be greater than end ({})",
                start, end
            )));
        }
        Ok(Self { start, end })
    }

    pub fn contains(&self, time: RationalTime) -> bool {
        time >= self.start && time <= self.end
    }
}
