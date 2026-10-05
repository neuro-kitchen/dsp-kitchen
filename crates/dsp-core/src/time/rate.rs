use num_rational::Ratio;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use crate::error::{DspError, DspResult};

/// `value` as the exact fraction of its shortest decimal form (`30000.12` → `750003/25`); a
/// continued-fraction approximation when that does not fit in `u64`. `value` must be finite and
/// non-negative.
pub(crate) fn decimal_ratio(value: f64) -> Ratio<u64> {
    let text = format!("{value}");
    let (int, frac) = text.split_once('.').unwrap_or((&text, ""));
    let digits = format!("{int}{frac}");
    match (digits.parse::<u64>(), 10u64.checked_pow(frac.len() as u32)) {
        (Ok(numer), Some(denom)) => Ratio::new(numer, denom),
        _ => Ratio::<i64>::approximate_float(value)
            .map(|r| Ratio::new(*r.numer() as u64, *r.denom() as u64))
            .unwrap_or_else(|| Ratio::from_integer(value.round() as u64)),
    }
}

/// Sampling rate, stored as an exact fraction of hertz for drift-free sample ↔ time conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SampleRate {
    hz: Ratio<u64>,
}

impl SampleRate {
    /// The rate as the exact fraction of the decimal value it is given as (`30000.12` →
    /// `750003/25`, `24414.0625` → `390625/16`).
    pub fn new(rate_hz: f64) -> DspResult<Self> {
        if rate_hz <= 0.0 || rate_hz.is_nan() || rate_hz.is_infinite() {
            return Err(DspError::InvalidSampleRate { rate_hz });
        }
        let hz = decimal_ratio(rate_hz);
        Self::from_ratio(*hz.numer(), *hz.denom())
    }

    /// The rate `numer / denom` Hz, exactly.
    pub fn from_ratio(numer: u64, denom: u64) -> DspResult<Self> {
        if numer == 0 || denom == 0 {
            return Err(DspError::InvalidSampleRate { rate_hz: if denom == 0 { f64::NAN } else { 0.0 } });
        }
        Ok(Self { hz: Ratio::new(numer, denom) })
    }

    pub fn rate_hz(&self) -> f64 {
        *self.hz.numer() as f64 / *self.hz.denom() as f64
    }

    /// The rate as an exact fraction of hertz.
    pub fn as_ratio(&self) -> Ratio<u64> {
        self.hz
    }

    pub fn nyquist_hz(&self) -> f64 {
        self.rate_hz() / 2.0
    }

    pub fn sample_period_seconds(&self) -> f64 {
        1.0 / self.rate_hz()
    }

    pub fn duration_for_samples(&self, num_samples: usize) -> Duration {
        Duration::from_secs_f64(num_samples as f64 / self.rate_hz())
    }

    pub fn samples_for_duration(&self, duration: Duration) -> usize {
        (duration.as_secs_f64() * self.rate_hz()).round() as usize
    }
}
