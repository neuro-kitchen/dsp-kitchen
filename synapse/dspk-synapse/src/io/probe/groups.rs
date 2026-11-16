//! A probe's channels as lists show them: grouped by shank, in depth order, each with a label
//! saying where its site is.

use super::SensorLayout;

/// The channels of one shank, along the shank (by depth, then across).
#[derive(Debug, Clone, PartialEq)]
pub struct ShankGroup {
    pub shank: usize,
    pub channels: Vec<usize>,
}

impl SensorLayout {
    /// The shanks in ascending order, each with its channels along the shank.
    pub fn shank_groups(&self) -> Vec<ShankGroup> {
        let mut shanks: Vec<usize> = self.contacts.iter().map(|s| s.shank_id).collect();
        shanks.sort_unstable();
        shanks.dedup();
        shanks
            .into_iter()
            .map(|shank| {
                let mut sites: Vec<_> = self.contacts.iter().filter(|s| s.shank_id == shank).collect();
                sites.sort_by(|a, b| a.position.y_um.total_cmp(&b.position.y_um).then(a.position.x_um.total_cmp(&b.position.x_um)));
                ShankGroup { shank, channels: sites.iter().map(|s| s.channel_id).collect() }
            })
            .collect()
    }

    /// A label per channel (index = channel id; `None` for a channel without a site): the site's
    /// position, `16,40 µm`, with its shank first on a probe of several, `S1 · 16,40 µm`.
    pub fn site_labels(&self) -> Vec<Option<String>> {
        let channels = self.contacts.iter().map(|s| s.channel_id + 1).max().unwrap_or(0);
        let multi = self.shank_groups().len() > 1;
        (0..channels)
            .map(|c| {
                let site = self.get_site(c).ok()?;
                let pos = format!("{:.0},{:.0} µm", site.position.x_um, site.position.y_um);
                Some(if multi { format!("S{} · {pos}", site.shank_id) } else { pos })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::presets::neuropixels_2_0;

    #[test]
    fn test_groups_follow_the_shanks_and_labels_name_the_sites() {
        let probe = neuropixels_2_0();
        let groups = probe.shank_groups();
        let total: usize = groups.iter().map(|g| g.channels.len()).sum();
        assert_eq!(total, probe.contacts.len(), "every contact is in one group");
        assert!(groups.windows(2).all(|w| w[0].shank < w[1].shank), "shanks ascending");
        for g in &groups {
            let y: Vec<f32> = g.channels.iter().map(|&c| probe.get_site(c).unwrap().position.y_um).collect();
            assert!(y.windows(2).all(|w| w[0] <= w[1]), "along the shank");
        }
        let labels = probe.site_labels();
        let first = labels.iter().flatten().next().unwrap();
        assert!(first.ends_with("µm"), "{first}");
        assert_eq!(first.starts_with('S'), groups.len() > 1, "the shank is named only on a probe of several");
    }
}
