//! Action potential / spike event detection for the timeline event track.

use dsp_synapse::detect_spikes_multichannel;

use super::source::SignalSource;

/// Threshold in robust noise standard deviations (Quiroga MAD estimate in `dsp-synapse`).
const THRESHOLD_FACTOR: f32 = 4.5;
/// Refractory period after a detection.
const REFRACTORY_SEC: f64 = 0.002;

/// Collection of detected spike events across channels.
#[derive(Debug, Clone, Default)]
pub struct SpikeEventStore {
    /// All event times, sorted ascending (for the overview track).
    pub times_sec: Vec<f64>,
    /// Per-channel sorted event times (for windowed lookups while rendering).
    by_channel: Vec<Vec<f64>>,
}

impl SpikeEventStore {
    pub fn len(&self) -> usize {
        self.times_sec.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.times_sec.is_empty()
    }

    /// Number of events detected on `channel`.
    pub fn count(&self, channel: usize) -> usize {
        self.by_channel.get(channel).map_or(0, Vec::len)
    }

    /// Event times of `channel` within `[t0, t1]`, found by binary search.
    pub fn in_window(&self, channel: usize, t0: f64, t1: f64) -> &[f64] {
        let Some(times) = self.by_channel.get(channel) else { return &[] };
        let lo = times.partition_point(|&t| t < t0);
        let hi = times.partition_point(|&t| t <= t1);
        &times[lo..hi.max(lo)]
    }

    /// Detects negative threshold crossings per channel with `dsp-synapse`
    /// (`-4.5 * sigma_n`, local-minimum peak, 2 ms refractory).
    pub fn detect(source: &dyn SignalSource) -> Self {
        let sample_rate = source.sample_rate();
        let refractory_samples = (sample_rate * REFRACTORY_SEC) as usize;
        let samples = source.samples();

        let mut by_channel = Vec::with_capacity(source.channels());
        let mut all: Vec<f64> = Vec::new();

        for ch in 0..source.channels() {
            // Channels are detected independently so the refractory state never leaks across them
            let events = detect_spikes_multichannel(
                source.channel(ch),
                1,
                samples,
                THRESHOLD_FACTOR,
                refractory_samples,
            );
            let times: Vec<f64> = events
                .iter()
                .map(|e| e.sample_index as f64 / sample_rate)
                .collect();
            all.extend_from_slice(&times);
            by_channel.push(times);
        }

        all.sort_by(f64::total_cmp);
        Self { times_sec: all, by_channel }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;

    #[test]
    fn test_spike_event_detection() {
        let sample_rate = 10_000.0;
        let samples = 2_000;
        let mut data = vec![1.0f32; 2 * samples];

        // Inject two clear spikes on ch 0 separated by > 2ms (20 samples)
        data[500] = -100.0;
        data[505] = -120.0; // Within 2ms refractory -> ignored
        data[600] = -100.0; // Outside refractory -> detected

        // Inject one clear spike on ch 1
        data[samples + 1000] = -110.0;
        let ds = Dataset::from_samples("test", data, 2, sample_rate);

        let store = SpikeEventStore::detect(&ds);
        assert_eq!(store.len(), 3);
        assert!((store.times_sec[0] - 0.050).abs() < 1e-6);
        assert!((store.times_sec[1] - 0.060).abs() < 1e-6);
        assert!((store.times_sec[2] - 0.100).abs() < 1e-6);

        assert_eq!(store.in_window(0, 0.0, 1.0), &[0.050, 0.060]);
        assert_eq!(store.in_window(1, 0.0, 1.0), &[0.100]);
        assert_eq!(store.in_window(0, 0.055, 1.0), &[0.060]);
        assert_eq!(store.in_window(1, 0.0, 0.05), &[] as &[f64]);
        assert_eq!(store.in_window(5, 0.0, 1.0), &[] as &[f64]);
    }
}
