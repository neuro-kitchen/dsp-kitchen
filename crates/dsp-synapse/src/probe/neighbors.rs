use dsp_core::layout::SensorLayout;

/// Finds the `k` nearest neighboring channel IDs to `target_channel` using Euclidean distance.
pub fn find_k_nearest_neighbors(
    layout: &SensorLayout,
    target_channel: usize,
    k: usize,
) -> Vec<usize> {
    let target_site = match layout.get_site(target_channel) {
        Ok(site) => site,
        Err(_) => return Vec::new(),
    };

    let tx = target_site.position.x_um;
    let ty = target_site.position.y_um;
    let tz = target_site.position.z_um;

    let mut dists: Vec<(usize, f32)> = layout
        .contacts
        .iter()
        .filter(|s| s.enabled)
        .map(|s| {
            let dx = s.position.x_um - tx;
            let dy = s.position.y_um - ty;
            let dz = s.position.z_um - tz;
            let dist_sq = dx * dx + dy * dy + dz * dz;
            (s.channel_id, dist_sq)
        })
        .collect();

    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    dists.iter().take(k).map(|(ch, _)| *ch).collect()
}

/// Precomputes a flat `[num_channels * k]` table of the `k` nearest enabled sites (recording
/// channel ids, `u32`) for every recording channel `0..num_channels`: row `ch` serves spikes whose
/// primary channel is `ch`. Rows are padded with the channel itself when fewer than `k` sites exist,
/// and channels absent from `layout` map only to themselves, so every entry is a valid row of the
/// recording buffer.
pub fn precompute_knn_table(layout: &SensorLayout, num_channels: usize, k: usize) -> Vec<u32> {
    let k = k.max(1);
    let mut table = vec![0u32; num_channels * k];
    for ch in 0..num_channels {
        let nbrs: Vec<usize> = find_k_nearest_neighbors(layout, ch, k).into_iter().filter(|&c| c < num_channels).collect();
        let row = &mut table[ch * k..(ch + 1) * k];
        for (idx, slot) in row.iter_mut().enumerate() {
            *slot = nbrs.get(idx).copied().unwrap_or(ch) as u32;
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::neuropixels::neuropixels_1_0;

    #[test]
    fn test_nearest_neighbors() {
        let probe = neuropixels_1_0();
        let neighbors = find_k_nearest_neighbors(&probe, 0, 4);

        assert_eq!(neighbors.len(), 4);
        assert_eq!(neighbors[0], 0); // Nearest is always self (dist 0)

        let table = precompute_knn_table(&probe, 384, 4);
        assert_eq!(table.len(), 384 * 4);
        assert_eq!(table[0], 0);
    }

    #[test]
    fn test_table_for_non_contiguous_layout() {
        use dsp_core::layout::{Position3D, SensorSite};
        // Sites for channels 0, 2 and 5 only, in a recording of 6 channels.
        let layout = SensorLayout::new(
            "sparse",
            [0usize, 2, 5].iter().map(|&c| SensorSite::new(c, Position3D::new(0.0, c as f32 * 10.0, 0.0), 0)).collect(),
        );
        let table = precompute_knn_table(&layout, 6, 2);
        assert_eq!(table.len(), 12);
        assert_eq!(&table[5 * 2..6 * 2], &[5, 2]); // row keyed by channel id 5, not by position
        assert_eq!(&table[1 * 2..2 * 2], &[1, 1]); // channel without a site: itself only
        assert!(table.iter().all(|&c| c < 6));
    }
}
