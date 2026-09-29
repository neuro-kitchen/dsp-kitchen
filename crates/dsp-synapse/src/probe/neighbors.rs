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
    }
}
