use cubecl::prelude::*;

/// `neighbour_slot` of a channel outside the radius.
pub const NO_SLOT: u32 = u32::MAX;

/// Slot of `channel` in channel `ch`'s neighbour row (`nbr_offsets[ch]..nbr_offsets[ch + 1]` of
/// `nbr_channels`), or [`NO_SLOT`] when it is not within the radius.
#[cube]
fn neighbour_slot(nbr_offsets: &[u32], nbr_channels: &[u32], ch: u32, channel: u32) -> u32 {
    let start: u32 = nbr_offsets[ch as usize];
    let end: u32 = nbr_offsets[(ch + 1u32) as usize];
    let mut slot = NO_SLOT.runtime();
    let mut k = start;
    while k < end {
        if nbr_channels[k as usize] == channel {
            slot = k - start;
        }
        k += 1u32;
    }
    slot
}

/// Locally exclusive spatial deduplication, one unit per crossing `i`.
///
/// Crossings are sorted by `(sample, channel)`; `peak_magnitudes` holds `|amplitude|`. Every
/// crossing `j` within `window_samples` of `i` whose channel is a radius neighbour of `i`'s
/// (the CSR table `nbr_offsets` / `nbr_channels`, built once per probe) sets bit `slot` of
/// `participating[i · mask_words ..]`. `survives[i] = 0` when such a `j` is stronger (larger
/// magnitude, then earlier, then lower channel), else `1`.
/// Dispatched via [`dsp_core::compute::LaunchGeometry::elementwise`].
#[cube(launch)]
pub fn spatial_dedup_survival_kernel(
    sample_indices: &[u32],
    channel_ids: &[u32],
    peak_magnitudes: &[f32],
    nbr_offsets: &[u32],
    nbr_channels: &[u32],
    survives: &mut [u32],
    participating: &mut [u32],
    num_spikes: u32,
    mask_words: u32,
    window_samples: u32,
) {
    let i = ABSOLUTE_POS_X;
    if i < num_spikes {
        let t_i = sample_indices[i as usize];
        let ch_i = channel_ids[i as usize];
        let amp_i = peak_magnitudes[i as usize];
        let mask_base = i * mask_words;
        let mut w = 0u32;
        while w < mask_words {
            participating[(mask_base + w) as usize] = 0u32;
            w += 1u32;
        }

        // First crossing within the window before `i`
        let mut j = i;
        while j > 0u32 && sample_indices[(j - 1u32) as usize] + window_samples >= t_i {
            j -= 1u32;
        }

        let mut keep = 1u32;
        while j < num_spikes && sample_indices[j as usize] <= t_i + window_samples {
            let ch_j = channel_ids[j as usize];
            let slot = neighbour_slot(nbr_offsets, nbr_channels, ch_i, ch_j);
            if slot != NO_SLOT {
                let word = (mask_base + slot / 32u32) as usize;
                participating[word] = participating[word] | (1u32 << (slot % 32u32));
                if j != i {
                    let t_j = sample_indices[j as usize];
                    let amp_j = peak_magnitudes[j as usize];
                    let j_beats_i = amp_j > amp_i || (amp_j == amp_i && (t_j < t_i || (t_j == t_i && ch_j < ch_i)));
                    if j_beats_i {
                        keep = 0u32;
                    }
                }
            }
            j += 1u32;
        }
        survives[i as usize] = keep;
    }
}
