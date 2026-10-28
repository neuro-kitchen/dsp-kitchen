use serde::{Deserialize, Serialize};
use super::rational::RationalTime;
use crate::error::{DspError, DspResult};

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
