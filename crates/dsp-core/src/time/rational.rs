use num_rational::Ratio;
use num_traits::CheckedAdd;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Duration;
use crate::error::{DspError, DspResult};
use super::rate::{decimal_ratio, SampleRate};

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

/// Exact rational representation of time (seconds as numerator/denominator).
/// Prevents floating-point accumulation drift over long recordings.
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

    /// Time from a decimal number of seconds, exact for its shortest decimal form (`12.5` → `25/2`).
    pub fn from_seconds_f64(seconds: f64) -> DspResult<Self> {
        if seconds < 0.0 || !seconds.is_finite() {
            return Err(DspError::InvalidConfig(format!("time {seconds} s must be finite and non-negative")));
        }
        Ok(Self { seconds: decimal_ratio(seconds) })
    }

    /// `self + other`, or `None` on `u64` overflow.
    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        self.seconds.checked_add(&other.seconds).map(|seconds| Self { seconds })
    }

    /// Exact time of `sample_index` at `rate` (fractional rates included, no drift).
    pub fn from_samples(sample_index: u64, rate: SampleRate) -> DspResult<Self> {
        let r = rate.as_ratio();
        // seconds = sample · denom / numer, reduced in 128-bit arithmetic
        let (n, d) = (sample_index as u128 * *r.denom() as u128, *r.numer() as u128);
        let g = gcd(n, d);
        let (n, d) = (n / g, d / g);
        match (u64::try_from(n), u64::try_from(d)) {
            (Ok(n), Ok(d)) => Self::new(n, d),
            _ => Err(DspError::InvalidConfig(format!("time of sample {sample_index} at {} Hz overflows", rate.rate_hz()))),
        }
    }

    pub fn as_seconds_f64(&self) -> f64 {
        *self.seconds.numer() as f64 / *self.seconds.denom() as f64
    }

    pub fn as_duration(&self) -> Duration {
        Duration::from_secs_f64(self.as_seconds_f64())
    }

    /// Index of the sample at or before this time at `rate` (exact).
    pub fn to_sample_index(&self, rate: SampleRate) -> u64 {
        let r = rate.as_ratio();
        let n = *self.seconds.numer() as u128 * *r.numer() as u128;
        let d = *self.seconds.denom() as u128 * *r.denom() as u128;
        (n / d) as u64
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractional_rates_round_trip_exactly() {
        for hz in [24_414.0625, 30_000.12, 2_500.0, 30_000.0] {
            let rate = SampleRate::new(hz).unwrap();
            let one_hour = (hz * 3600.0).round() as u64;
            for s in [0, 1, 12_345, one_hour - 1, one_hour, 10 * one_hour + 7] {
                let t = RationalTime::from_samples(s, rate).unwrap();
                assert_eq!(t.to_sample_index(rate), s, "{hz} Hz sample {s}");
            }
        }
        // 1 h at the TDT rate is exactly 3600 s (the old integer rate drifted ~9 ms).
        let tdt = SampleRate::new(24_414.0625).unwrap();
        assert_eq!(RationalTime::from_samples(87_890_625, tdt).unwrap(), RationalTime::new(3600, 1).unwrap());
        assert_eq!(SampleRate::new(30_000.12).unwrap().as_ratio(), Ratio::new(750_003, 25));
    }
}
