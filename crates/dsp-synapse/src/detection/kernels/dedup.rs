use cubecl::prelude::*;

/// CubeCL parallel locally-exclusive spatial deduplication survival kernel.
///
/// Given sorted spike events `(sample_indices, channel_ids, peak_magnitudes)` (`|amplitude|`) of length `num_spikes`
/// and a flattened `[num_channels, num_channels]` pairwise distance matrix `dist_matrix_um`,
/// each unit `i = ABSOLUTE_POS` checks neighboring spikes within `window_samples` and `radius_um`
/// and sets `survives[i] = 1` if no stronger crossing (larger magnitude, then earlier, then lower
/// channel) beats event `i`, or `0` otherwise.
/// Dispatched via [`dsp_core::compute::LaunchGeometry::elementwise`].
#[cube(launch)]
pub fn spatial_dedup_survival_kernel(
    sample_indices: &Array<u32>,
    channel_ids: &Array<u32>,
    peak_magnitudes: &Array<f32>,
    dist_matrix_um: &Array<f32>,
    survives: &mut Array<u32>,
    num_spikes: usize,
    num_channels: u32,
    radius_um: f32,
    window_samples: u32,
) {
    let i = ABSOLUTE_POS;
    if i < num_spikes {
        let t_i = sample_indices[i];
        let ch_i = channel_ids[i];
        let amp_i = peak_magnitudes[i];

        let mut keep = 1u32;

        // Scan backward while within window_samples
        let mut j = i;
        while j > 0usize {
            j = j - 1usize;
            let t_j = sample_indices[j];
            if t_i > t_j + window_samples {
                break;
            }
            let ch_j = channel_ids[j];
            let d = dist_matrix_um[(ch_i * num_channels + ch_j) as usize];
            if d <= radius_um {
                let amp_j = peak_magnitudes[j];
                let j_beats_i = amp_j > amp_i
                    || (amp_j == amp_i && (t_j < t_i || (t_j == t_i && ch_j < ch_i)));
                if j_beats_i {
                    keep = 0u32;
                }
            }
        }

        // Scan forward while within window_samples
        let mut k = i + 1usize;
        while k < num_spikes {
            let t_k = sample_indices[k];
            if t_k > t_i + window_samples {
                break;
            }
            let ch_k = channel_ids[k];
            let d = dist_matrix_um[(ch_i * num_channels + ch_k) as usize];
            if d <= radius_um {
                let amp_k = peak_magnitudes[k];
                let k_beats_i = amp_k > amp_i
                    || (amp_k == amp_i && (t_k < t_i || (t_k == t_i && ch_k < ch_i)));
                if k_beats_i {
                    keep = 0u32;
                }
            }
            k = k + 1usize;
        }

        survives[i] = keep;
    }
}
