use serde::{Deserialize, Serialize};

/// Detected action potential event (spike).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpikeEvent {
    pub channel_id: usize,
    pub sample_index: u64,
    pub peak_amplitude_uv: f32,
}

/// Simple voltage threshold detector (e.g. 5x standard deviation).
pub fn detect_threshold_crossings(
    signal: &[f32],
    num_samples: usize,
    channel_id: usize,
    threshold_uv: f32,
    refractory_samples: usize,
) -> Vec<SpikeEvent> {
    let mut spikes = Vec::new();
    let mut last_spike_sample = 0usize;

    for i in 1..num_samples {
        let val = signal[i];
        if val < -threshold_uv && (spikes.is_empty() || i > last_spike_sample + refractory_samples) {
            spikes.push(SpikeEvent {
                channel_id,
                sample_index: i as u64,
                peak_amplitude_uv: val,
            });
            last_spike_sample = i;
        }
    }

    spikes
}
