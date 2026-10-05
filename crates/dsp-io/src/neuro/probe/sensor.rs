use serde::{Deserialize, Serialize};
use super::site::{Position3D, SensorSite};
use dsp_core::{DspError, DspResult};

/// Physical sensor array geometry layout (e.g. multi-electrode arrays, acoustic arrays).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorLayout {
    pub name: String,
    pub contacts: Vec<SensorSite>,
}

impl SensorLayout {
    pub fn new(name: impl Into<String>, contacts: Vec<SensorSite>) -> Self {
        Self {
            name: name.into(),
            contacts,
        }
    }

    pub fn total_channels(&self) -> usize {
        self.contacts.len()
    }

    pub fn active_channels(&self) -> usize {
        self.contacts.iter().filter(|c| c.enabled).count()
    }

    /// Site of recording channel `channel_id`. O(1) when sites are stored in channel order (the
    /// usual case), a scan otherwise.
    pub fn get_site(&self, channel_id: usize) -> DspResult<&SensorSite> {
        if let Some(site) = self.contacts.get(channel_id).filter(|c| c.channel_id == channel_id) {
            return Ok(site);
        }
        self.contacts
            .iter()
            .find(|c| c.channel_id == channel_id)
            .ok_or(DspError::InvalidChannel {
                channel: channel_id,
                total: self.contacts.len(),
            })
    }

    pub fn get_contact(&self, channel_id: usize) -> DspResult<&SensorSite> {
        self.get_site(channel_id)
    }

    pub fn sites(&self) -> &[SensorSite] {
        &self.contacts
    }

    /// Layout restricted to `channels` (recording channel ids of the parent), renumbered so that
    /// the site of `channels[i]` gets channel id `i` — the layout of a channel subset.
    pub fn select_channels(&self, channels: &[usize]) -> Self {
        let contacts = channels
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| self.get_site(c).ok().map(|s| SensorSite { channel_id: i, ..s.clone() }))
            .collect();
        Self { name: self.name.clone(), contacts }
    }

    /// Flat `(channel_map, channel_positions_xy, channel_shanks)` arrays as used by Phy / Kilosort
    /// and Zarr stores.
    pub fn to_channel_arrays(&self) -> (Vec<usize>, Vec<[f32; 2]>, Vec<usize>) {
        let map = self.contacts.iter().map(|c| c.channel_id).collect();
        let pos = self.contacts.iter().map(|c| [c.position.x_um, c.position.y_um]).collect();
        let shanks = self.contacts.iter().map(|c| c.shank_id).collect();
        (map, pos, shanks)
    }

    /// Reconstructs a `SensorLayout` from flat `(channel_map, channel_positions_xy, channel_shanks)`
    /// arrays (`None` when `positions` is empty). Missing `channel_map` or `shanks` entries fall back
    /// to identity (`i`) and shank `0`.
    pub fn from_channel_arrays(
        name: impl Into<String>,
        channel_map: &[usize],
        positions: &[[f32; 2]],
        shanks: &[usize],
    ) -> Option<Self> {
        if positions.is_empty() {
            return None;
        }
        let contacts = positions
            .iter()
            .enumerate()
            .map(|(i, p)| {
                SensorSite::new(
                    channel_map.get(i).copied().unwrap_or(i),
                    Position3D::new(p[0], p[1], 0.0),
                    shanks.get(i).copied().unwrap_or(0),
                )
            })
            .collect();
        Some(Self::new(name, contacts))
    }

    /// Neuropixels 1.0 (384 channels, one shank): 20 µm rows, staggered columns at x = 43 / 11 µm on
    /// even rows and 59 / 27 µm on odd rows (probeinterface `NP1010`).
    pub fn neuropixels_1_0_standard() -> Self {
        const X: [f32; 4] = [43.0, 11.0, 59.0, 27.0];
        let contacts = (0..384)
            .map(|i| SensorSite {
                channel_id: i,
                device_index: i,
                group_id: 0,
                shank_id: 0,
                position: Position3D::new(X[i % 4], (i / 2) as f32 * 20.0, 0.0),
                enabled: true,
            })
            .collect();
        Self { name: "Neuropixels 1.0 (Standard 384)".into(), contacts }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neuropixels_1_0_matches_probeinterface() {
        let np = SensorLayout::neuropixels_1_0_standard();
        let xy: Vec<(f32, f32)> = np.contacts[..6].iter().map(|c| (c.position.x_um, c.position.y_um)).collect();
        assert_eq!(xy, vec![(43.0, 0.0), (11.0, 0.0), (59.0, 20.0), (27.0, 20.0), (43.0, 40.0), (11.0, 40.0)]);
        assert_eq!(np.get_site(383).unwrap().position.y_um, 3820.0);
    }

    #[test]
    fn select_channels_renumbers_sites() {
        let np = SensorLayout::neuropixels_1_0_standard();
        let sub = np.select_channels(&[10, 3, 200]);
        assert_eq!(sub.contacts.len(), 3);
        assert_eq!(sub.get_site(1).unwrap().position, np.get_site(3).unwrap().position);
        assert_eq!(sub.get_site(2).unwrap().position, np.get_site(200).unwrap().position);
    }
}
