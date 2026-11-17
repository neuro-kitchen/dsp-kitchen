//! The offset between a client's clock and a device's, from four timestamps (as NTP): the client
//! sends at `t0`, the device receives at `t1` and answers at `t2`, the client receives at `t3`.
//! offset = ((t1 − t0) + (t2 − t3)) / 2, round trip = (t3 − t0) − (t2 − t1). The sample with the
//! shortest round trip of the recent ones is the most accurate (least queueing).

use std::collections::VecDeque;

/// One exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClockSample {
    /// Device clock minus client clock (ns).
    pub offset_ns: i64,
    pub round_trip_ns: u64,
}

impl ClockSample {
    pub fn from_exchange(client_sent: u64, device_received: u64, device_sent: u64, client_received: u64) -> Self {
        let (t0, t1, t2, t3) = (client_sent as i128, device_received as i128, device_sent as i128, client_received as i128);
        let offset = ((t1 - t0) + (t2 - t3)) / 2;
        let round_trip = ((t3 - t0) - (t2 - t1)).max(0);
        Self { offset_ns: offset as i64, round_trip_ns: round_trip as u64 }
    }
}

/// The best of the last few samples.
#[derive(Debug, Clone, Default)]
pub struct ClockEstimate {
    samples: VecDeque<ClockSample>,
}

/// Samples kept.
const KEPT: usize = 16;

impl ClockEstimate {
    pub fn add(&mut self, sample: ClockSample) {
        if self.samples.len() == KEPT {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    /// The sample with the shortest round trip (`None` before the first).
    pub fn best(&self) -> Option<ClockSample> {
        self.samples.iter().min_by_key(|s| s.round_trip_ns).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_and_round_trip() {
        // Device 1 000 ns ahead; 200 ns each way; 50 ns inside the device
        let s = ClockSample::from_exchange(10_000, 11_200, 11_250, 10_450);
        assert_eq!(s.offset_ns, 1_000);
        assert_eq!(s.round_trip_ns, 400);
        let mut e = ClockEstimate::default();
        e.add(ClockSample { offset_ns: 900, round_trip_ns: 5_000 });
        e.add(s);
        assert_eq!(e.best(), Some(s));
    }
}
