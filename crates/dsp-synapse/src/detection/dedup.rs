use dsp_core::SensorLayout;
use super::threshold::SpikeEvent;

/// Deduplicated multi-channel spike event.
#[derive(Debug, Clone, PartialEq)]
pub struct DeduplicatedSpike {
    /// The primary channel with the deepest negative trough.
    pub primary_channel: usize,
    /// Sample index of the peak on the primary channel.
    pub sample_index: u64,
    /// Peak amplitude in microvolts on the primary channel.
    pub peak_amplitude_uv: f32,
    /// Neighboring channels that also co-detected this action potential.
    pub participating_channels: Vec<usize>,
}

/// Deduplicates multi-channel spike events across space and time.
///
/// When a single action potential fires, multiple nearby electrode sites detect the voltage deflection.
/// This algorithm suppresses duplicate detections within a spatial radius (`radius_um`) and temporal
/// window (`window_samples`), electing the contact with the deepest negative trough as the primary channel.
pub fn deduplicate_spikes_spatial(
    spikes: &[SpikeEvent],
    layout: &SensorLayout,
    radius_um: f32,
    window_samples: u64,
) -> Vec<DeduplicatedSpike> {
    if spikes.is_empty() {
        return Vec::new();
    }

    // Sort spikes chronologically
    let mut sorted = spikes.to_vec();
    sorted.sort_by_key(|s| s.sample_index);

    let mut suppressed = vec![false; sorted.len()];
    let mut deduplicated = Vec::new();

    for i in 0..sorted.len() {
        if suppressed[i] {
            continue;
        }

        let ref_spike = &sorted[i];
        let ref_site = match layout.get_site(ref_spike.channel_id) {
            Ok(site) => site,
            Err(_) => continue,
        };

        let mut cluster_indices = vec![i];

        // Search forward within temporal window
        for j in (i + 1)..sorted.len() {
            let other_spike = &sorted[j];
            if other_spike.sample_index > ref_spike.sample_index + window_samples {
                break;
            }

            if suppressed[j] {
                continue;
            }

            if let Ok(other_site) = layout.get_site(other_spike.channel_id) {
                let dist = ref_site.position.distance_to(&other_site.position);
                if dist <= radius_um {
                    cluster_indices.push(j);
                }
            }
        }

        // Search backward within temporal window (for transitive events)
        for j in (0..i).rev() {
            let other_spike = &sorted[j];
            if ref_spike.sample_index > other_spike.sample_index + window_samples {
                break;
            }

            if suppressed[j] {
                continue;
            }

            if let Ok(other_site) = layout.get_site(other_spike.channel_id) {
                let dist = ref_site.position.distance_to(&other_site.position);
                if dist <= radius_um {
                    cluster_indices.push(j);
                }
            }
        }

        // Elect primary channel: deepest negative peak (minimum value)
        let primary_idx = *cluster_indices
            .iter()
            .min_by(|&&a, &&b| {
                sorted[a]
                    .peak_amplitude_uv
                    .partial_cmp(&sorted[b].peak_amplitude_uv)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        let primary_spike = &sorted[primary_idx];
        let mut participating: Vec<usize> = cluster_indices
            .iter()
            .map(|&idx| sorted[idx].channel_id)
            .collect();
        participating.sort_unstable();
        participating.dedup();

        deduplicated.push(DeduplicatedSpike {
            primary_channel: primary_spike.channel_id,
            sample_index: primary_spike.sample_index,
            peak_amplitude_uv: primary_spike.peak_amplitude_uv,
            participating_channels: participating,
        });

        // Mark all clustered spikes as suppressed
        for idx in cluster_indices {
            suppressed[idx] = true;
        }
    }

    // Sort final events chronologically
    deduplicated.sort_by_key(|s| s.sample_index);
    deduplicated
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::layout::{Position3D, SensorSite};

    #[test]
    fn test_spatial_deduplication() {
        let contacts = vec![
            SensorSite::new(0, Position3D::new(0.0, 0.0, 0.0), 0),
            SensorSite::new(1, Position3D::new(20.0, 0.0, 0.0), 0),
            SensorSite::new(2, Position3D::new(100.0, 0.0, 0.0), 0), // Far channel
        ];
        let layout = SensorLayout::new("Test", contacts);

        let spikes = vec![
            // Two nearby detections of the same spike at sample 100
            SpikeEvent { channel_id: 0, sample_index: 100, peak_amplitude_uv: -120.0 },
            SpikeEvent { channel_id: 1, sample_index: 102, peak_amplitude_uv: -65.0 },
            // Independent spike on far channel 2 at sample 101
            SpikeEvent { channel_id: 2, sample_index: 101, peak_amplitude_uv: -80.0 },
        ];

        let deduped = deduplicate_spikes_spatial(&spikes, &layout, 50.0, 5);
        assert_eq!(deduped.len(), 2);

        // First event should be primary channel 0 with participating [0, 1]
        let ev0 = &deduped[0];
        assert_eq!(ev0.primary_channel, 0);
        assert_eq!(ev0.peak_amplitude_uv, -120.0);
        assert_eq!(ev0.participating_channels, vec![0, 1]);

        // Second event should be independent channel 2
        let ev1 = &deduped[1];
        assert_eq!(ev1.primary_channel, 2);
        assert_eq!(ev1.peak_amplitude_uv, -80.0);
    }
}
