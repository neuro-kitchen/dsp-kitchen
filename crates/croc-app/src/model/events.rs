//! Action potential / spike event detection for the timeline event track.

/// Collection of detected spike events across channels.
#[derive(Debug, Clone, Default)]
pub struct SpikeEventStore {
    pub times_sec: Vec<f64>,
    pub channels: Vec<usize>,
}

impl SpikeEventStore {
    pub fn len(&self) -> usize {
        self.times_sec.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.times_sec.is_empty()
    }

    /// Scans multi-channel signal buffer for negative threshold crossings (-4.5 * noise_std)
    /// with a 2.0 ms refractory period.
    pub fn detect_from_raw(
        raw_data: &[f32],
        total_channels: usize,
        total_samples: usize,
        sample_rate: f64,
    ) -> Self {
        let mut times_sec = Vec::new();
        let mut channels = Vec::new();

        if total_channels == 0 || total_samples == 0 || raw_data.len() < total_channels * total_samples {
            return Self { times_sec, channels };
        }

        let refractory_samples = (sample_rate * 0.002) as usize; // 2.0 ms

        for ch in 0..total_channels {
            let ch_offset = ch * total_samples;
            let channel_data = &raw_data[ch_offset..ch_offset + total_samples];

            let mut noise_std = 15.0f32;
            let calc_len = channel_data.len().min(1000);
            if calc_len > 0 {
                let mut sum_sq = 0.0f64;
                for &val in &channel_data[0..calc_len] {
                    sum_sq += (val as f64) * (val as f64);
                }
                noise_std = ((sum_sq / calc_len as f64).sqrt() as f32).max(5.0);
            }

            let threshold = -4.5 * noise_std;
            let mut in_refractory = 0usize;

            for (i, &sample) in channel_data.iter().enumerate() {
                if in_refractory > 0 {
                    in_refractory -= 1;
                    continue;
                }
                if sample < threshold {
                    let t_sec = i as f64 / sample_rate;
                    times_sec.push(t_sec);
                    channels.push(ch);
                    in_refractory = refractory_samples;
                }
            }
        }

        Self { times_sec, channels }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_event_detection() {
        let sample_rate = 10_000.0;
        let samples = 2_000;
        let channels = 2;
        let mut data = vec![1.0f32; channels * samples];

        // Inject two clear spikes on ch 0 separated by > 2ms (20 samples)
        data[500] = -100.0;
        data[505] = -120.0; // Within 2ms refractory -> ignored
        data[600] = -100.0; // Outside refractory -> detected

        // Inject one clear spike on ch 1
        data[samples + 1000] = -110.0;

        let store = SpikeEventStore::detect_from_raw(&data, channels, samples, sample_rate);
        assert_eq!(store.len(), 3);
        assert_eq!(store.channels, vec![0, 0, 1]);
        assert!((store.times_sec[0] - 0.050).abs() < 1e-6);
        assert!((store.times_sec[1] - 0.060).abs() < 1e-6);
        assert!((store.times_sec[2] - 0.100).abs() < 1e-6);
    }
}
