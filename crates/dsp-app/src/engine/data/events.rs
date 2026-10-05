//! Spike events for the timeline event track.
//!
//! Events come only from a spike extraction the user runs with their own parameters; nothing
//! is detected when a recording opens. Until then the store is empty.

/// Collection of spike events across channels.
#[derive(Debug, Clone, Default)]
pub struct SpikeEventStore {
    /// All event times, sorted ascending.
    #[cfg_attr(not(test), allow(dead_code))]
    pub times_sec: Vec<f64>,
    /// Per-channel sorted event times (for windowed lookups while rendering).
    by_channel: Vec<Vec<f64>>,
}

impl SpikeEventStore {
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.times_sec.len()
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

    /// Builds a [`SpikeEventStore`] from a loaded sorting, assigning each spike to its unit's
    /// peak channel (`ClusterMeta::ch`).
    pub fn from_sorting(data: &crate::engine::curation::SortingData) -> Self {
        let sr = data.sample_rate().max(1.0);
        let n_ch = data
            .sorting
            .channel_map
            .iter()
            .copied()
            .max()
            .map_or(0, |c| c + 1)
            .max(data.clusters.values().map(|c| c.ch + 1).max().unwrap_or(0));
        let mut by_channel = vec![Vec::new(); n_ch];
        let mut all = Vec::with_capacity(data.sorting.spike_times.len());
        for (&sample, &cid) in data.sorting.spike_times.iter().zip(&data.sorting.spike_clusters) {
            let t = sample as f64 / sr;
            all.push(t);
            let ch = data.clusters.get(&cid).map_or(0, |m| m.ch);
            if ch >= by_channel.len() {
                by_channel.resize(ch + 1, Vec::new());
            }
            by_channel[ch].push(t);
        }
        all.sort_by(f64::total_cmp);
        for ch_times in &mut by_channel {
            ch_times.sort_by(f64::total_cmp);
        }
        Self { times_sec: all, by_channel }
    }

    /// Negative threshold crossings per channel with `dsp-synapse` (`-4.5 * sigma_n`,
    /// local-minimum peak, 2 ms refractory), reading whole channels. Test fixture only: the app
    /// never detects on its own; user-run extraction replaces it.
    #[cfg(test)]
    pub fn detect(source: &dyn dsp_core::RecordingSource) -> Self {
        use dsp_synapse::detect_spikes_multichannel;
        const THRESHOLD_FACTOR: f32 = 4.5;
        const REFRACTORY_SEC: f64 = 0.002;
        let info = source.info();
        let sample_rate = info.sample_rate_hz();
        let refractory_samples = (sample_rate * REFRACTORY_SEC) as usize;
        let samples = info.samples as usize;

        let mut by_channel = Vec::with_capacity(info.channel_count());
        let mut all: Vec<f64> = Vec::new();
        let mut trace = vec![0.0f32; samples];

        for ch in 0..info.channel_count() {
            if source.read(&[ch], 0..samples as u64, &mut trace).is_err() {
                by_channel.push(Vec::new());
                continue;
            }
            // Channels are detected independently so the refractory state never leaks across them
            let events = detect_spikes_multichannel(
                &trace,
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
    use crate::engine::data::Dataset;

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
