//! Curation of a spike sorting, after phy's template GUI.
//!
//! The sorting itself is a [`PhySorting`] (read and written by `dsp-synapse`: phy / Kilosort
//! folders, `.sorting.zarr`, NWB `/units`). This module adds what curation needs on top: one
//! summary row per cluster (the Clusters table), the edit history ([`history`]: merge, split,
//! labels, undo / redo) and the data the views draw ([`derived`]: waveforms, features,
//! correlograms, amplitudes, ISI, firing rate), computed from the sorting and, where raw data is
//! needed, the recording. Nothing is invented: without a recording, views that need raw samples
//! have nothing to show.

pub mod derived;
pub mod filter;
pub mod history;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use dsp_synapse::{compute_isi_violations, load_spikes, PhySorting};
use serde::{Deserialize, Serialize};

pub use derived::{AmplitudeMode, AmplitudePlotData, ClusterWaveforms, CorrelogramMatrix, FeatureGridData, FeatureSource, SpikePoint};
pub use dsp_synapse::storage::phy_sorting::ClusterId;
pub use filter::matches_filter;
pub use history::CurationCommand;

/// Refractory period for the contamination estimate when the sorter saved none (ms).
const REFRACTORY_MS: f64 = 2.0;

/// Quality group assigned during curation (phy's `cluster_group.tsv` values).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub enum ClusterGroup {
    Good,
    Mua,
    Noise,
    #[default]
    Unsorted,
}

impl ClusterGroup {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Good => "good",
            Self::Mua => "mua",
            Self::Noise => "noise",
            Self::Unsorted => "unsorted",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "good" | "singleunit" => Self::Good,
            "mua" | "multiunit" => Self::Mua,
            "noise" => Self::Noise,
            _ => Self::Unsorted,
        }
    }
}

/// Sortable columns of the Clusters table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterSortColumn {
    Id,
    Channel,
    Depth,
    Spikes,
    FiringRate,
    Amplitude,
    ContamPct,
    KsLabel,
    Group,
}

/// One row of the Clusters table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterMeta {
    pub id: ClusterId,
    /// Best channel (sorted-channel index).
    pub ch: usize,
    pub depth: f32,
    pub sh: usize,
    pub n_spikes: usize,
    pub fr: f32,
    pub amp: f32,
    pub contam_pct: f32,
    pub ks_label: String,
    pub group: ClusterGroup,
    pub custom: BTreeMap<String, String>,
}

/// `cluster_info.tsv` columns this table computes itself (others are custom labels).
const BUILTIN_COLUMNS: &[&str] =
    &["cluster_id", "id", "ch", "channel", "depth", "sh", "shank", "n_spikes", "fr", "firing_rate", "amp", "amplitude", "contam_pct", "contampct", "kslabel", "ks_label", "group"];

/// An open sorting and its curation state.
#[derive(Clone)]
pub struct SortingData {
    pub sorting: PhySorting,
    pub name: String,
    /// Recording length (s): the attached recording's, else up to the last spike.
    pub duration_sec: f64,
    pub clusters: BTreeMap<ClusterId, ClusterMeta>,
    /// Custom label columns (from `cluster_info.tsv`, and added during curation).
    pub custom_keys: Vec<String>,
    pub(crate) next_cluster_id: ClusterId,
    pub undo_stack: Vec<CurationCommand>,
    pub redo_stack: Vec<CurationCommand>,
    /// History depth when last saved (or opened).
    saved_depth: usize,
}

impl SortingData {
    /// Opens a phy / Kilosort folder (or a file in one), `.sorting.zarr`, or NWB `/units`.
    pub fn open(path: &Path) -> Result<Self> {
        let sorting = load_spikes(path).with_context(|| format!("Could not open the sorting {}", path.display()))?;
        Ok(Self::from_sorting(sorting, path))
    }

    pub fn from_sorting(sorting: PhySorting, path: &Path) -> Self {
        let name = path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
        let rate = sorting.sample_rate().max(f64::MIN_POSITIVE);
        let duration_sec = (sorting.spike_times.iter().copied().max().unwrap_or(0) + 1) as f64 / rate;
        let custom_keys = sorting.tables.info_columns.iter().filter(|c| !BUILTIN_COLUMNS.contains(&c.to_ascii_lowercase().as_str())).cloned().collect();
        let next_cluster_id = sorting.spike_clusters.iter().copied().max().map_or(0, |m| m + 1).max(sorting.templates.as_ref().map_or(0, |t| t.count as ClusterId));
        let mut data = Self { sorting, name, duration_sec, clusters: BTreeMap::new(), custom_keys, next_cluster_id, undo_stack: Vec::new(), redo_stack: Vec::new(), saved_depth: 0 };
        data.rebuild_clusters();
        data
    }

    /// The recording this sorting was made from is attached: firing rates use its length.
    pub fn set_recording_duration(&mut self, duration_sec: f64) {
        if duration_sec > 0.0 && (duration_sec - self.duration_sec).abs() > 1e-9 {
            self.duration_sec = duration_sec;
            for m in self.clusters.values_mut() {
                m.fr = (m.n_spikes as f64 / duration_sec) as f32;
            }
        }
    }

    /// The recording `params.py` names, if it is found.
    pub fn recording_path(&self) -> Option<PathBuf> {
        self.sorting.recording_path()
    }

    /// Edits since the last save.
    pub fn dirty(&self) -> bool {
        self.undo_stack.len() != self.saved_depth
    }

    pub fn sample_rate(&self) -> f64 {
        self.sorting.sample_rate()
    }

    /// Rows for every cluster, with the sorter's tables where they say more than the spikes do.
    fn rebuild_clusters(&mut self) {
        let mut ids: Vec<ClusterId> = self.sorting.spike_clusters.clone();
        ids.sort_unstable();
        ids.dedup();
        let tables = self.sorting.tables.clone();
        self.clusters = ids
            .into_iter()
            .map(|id| {
                let mut m = self.compute_meta_for(id, None);
                if let Some(a) = tables.amplitude.get(&id) {
                    m.amp = *a;
                }
                if let Some(c) = tables.contam_pct.get(&id) {
                    m.contam_pct = *c;
                }
                m.ks_label = tables.ks_label.get(&id).or_else(|| tables.group.get(&id)).cloned().unwrap_or_else(|| "unsorted".into());
                m.group = tables.group.get(&id).map_or(ClusterGroup::Unsorted, |g| ClusterGroup::parse(g));
                if let Some(row) = tables.info.get(&id) {
                    m.custom = self.custom_keys.iter().filter_map(|k| Some((k.clone(), row.get(k)?.clone()))).collect();
                }
                (id, m)
            })
            .collect();
    }

    /// A fresh row for cluster `cid` from its spikes (labels carried over from `prev`).
    pub(crate) fn compute_meta_for(&self, cid: ClusterId, prev: Option<&ClusterMeta>) -> ClusterMeta {
        let s = &self.sorting;
        let indices = self.spike_indices(cid);
        let n_spikes = indices.len();
        let ch = self.best_channel(cid);
        let mean_amp = if n_spikes > 0 { indices.iter().filter_map(|&i| s.amplitudes.get(i)).map(|a| a.abs() as f64).sum::<f64>() / n_spikes as f64 } else { 0.0 };
        let amp = match self.representative_template(cid) {
            Some(t) => s.templates.as_ref().map_or(0.0, |tm| tm.amplitude(t)),
            None => mean_amp as f32,
        };
        let samples: Vec<u64> = indices.iter().map(|&i| s.spike_times[i]).collect();
        let contam_pct = compute_isi_violations(&samples, self.sample_rate(), self.duration_sec, REFRACTORY_MS, 0.0).violation_rate_pct;
        ClusterMeta {
            id: cid,
            ch,
            depth: s.channel_positions.get(ch).map_or(0.0, |p| p[1]),
            sh: s.channel_shanks.get(ch).copied().unwrap_or(0),
            n_spikes,
            fr: (n_spikes as f64 / self.duration_sec.max(1e-9)) as f32,
            amp,
            contam_pct,
            ks_label: prev.map_or_else(|| "unsorted".into(), |p| p.ks_label.clone()),
            group: ClusterGroup::Unsorted,
            custom: prev.map(|p| p.custom.clone()).unwrap_or_default(),
        }
    }

    /// The template that stands for cluster `cid`: its own id when the sorter has that template
    /// (unchanged clusters), else the template most of its spikes were matched with (merged or
    /// split clusters).
    pub fn representative_template(&self, cid: ClusterId) -> Option<usize> {
        let count = self.sorting.templates.as_ref()?.count;
        let mut votes: BTreeMap<ClusterId, usize> = BTreeMap::new();
        for i in self.spike_indices(cid) {
            *votes.entry(self.sorting.spike_templates[i]).or_default() += 1;
        }
        if votes.is_empty() {
            return ((cid as usize) < count).then_some(cid as usize);
        }
        // Unchanged cluster: every spike on its own template
        if votes.len() == 1 && votes.contains_key(&cid) && (cid as usize) < count {
            return Some(cid as usize);
        }
        votes.into_iter().max_by_key(|&(t, n)| (n, std::cmp::Reverse(t))).map(|(t, _)| t as usize).filter(|&t| t < count)
    }

    /// Best channel of cluster `cid` (largest template), 0 without templates.
    pub fn best_channel(&self, cid: ClusterId) -> usize {
        self.top_channels(cid, 1).first().copied().unwrap_or(0)
    }

    /// The `k` channels where cluster `cid`'s template is largest; without templates, the
    /// channels nearest its spikes' mean position; else the first `k`.
    pub fn top_channels(&self, cid: ClusterId, k: usize) -> Vec<usize> {
        let s = &self.sorting;
        let channels = s.channels();
        let k = k.clamp(1, channels.max(1));
        if let (Some(t), Some(tm)) = (self.representative_template(cid), s.templates.as_ref()) {
            return tm.top_channels(t, k);
        }
        let idx = self.spike_indices(cid);
        let pos: Vec<[f32; 2]> = idx.iter().filter_map(|&i| s.spike_positions.get(i).copied()).collect();
        if !pos.is_empty() && s.channel_positions.len() == channels {
            let (mx, my) = pos.iter().fold((0.0, 0.0), |(x, y), p| (x + p[0] / pos.len() as f32, y + p[1] / pos.len() as f32));
            let mut order: Vec<usize> = (0..channels).collect();
            let d = |c: usize| (s.channel_positions[c][0] - mx).powi(2) + (s.channel_positions[c][1] - my).powi(2);
            order.sort_by(|&a, &b| d(a).total_cmp(&d(b)));
            order.truncate(k);
            return order;
        }
        (0..k).collect()
    }

    /// Indices (into the spike arrays) of cluster `cid`'s spikes, in file order.
    pub fn spike_indices(&self, cid: ClusterId) -> Vec<usize> {
        self.sorting.spike_clusters.iter().enumerate().filter_map(|(i, &c)| (c == cid).then_some(i)).collect()
    }

    /// Sample times of cluster `cid`'s spikes, ascending.
    pub fn spike_samples(&self, cid: ClusterId) -> Vec<u64> {
        let mut s: Vec<u64> = self.spike_indices(cid).into_iter().map(|i| self.sorting.spike_times[i]).collect();
        s.sort_unstable();
        s
    }

    /// Other clusters by decreasing template similarity to `best` (phy's Similar view).
    pub fn similar_to(&self, best: ClusterId) -> Vec<(ClusterId, f32)> {
        let Some(a) = self.representative_template(best) else { return Vec::new() };
        let mut out: Vec<(ClusterId, f32)> = self
            .clusters
            .keys()
            .filter(|&&c| c != best)
            .filter_map(|&c| Some((c, self.sorting.similarity(a, self.representative_template(c)?))))
            .collect();
        out.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
        out
    }

    /// Writes the curation into `dir` (what phy reads back: `spike_clusters.npy`,
    /// `cluster_group.tsv`, `cluster_info.tsv`; originals kept as `.bak`).
    pub fn save_to(&mut self, dir: &Path) -> Result<()> {
        let groups: BTreeMap<ClusterId, String> = self.clusters.iter().map(|(&id, m)| (id, m.group.as_str().to_string())).collect();
        let mut columns: Vec<String> = ["cluster_id", "ch", "depth", "sh", "n_spikes", "fr", "amp", "contam_pct", "KSLabel", "group"].map(String::from).to_vec();
        columns.extend(self.custom_keys.iter().cloned());
        let rows = self
            .clusters
            .iter()
            .map(|(&id, m)| {
                let mut row: BTreeMap<String, String> = [
                    ("ch", m.ch.to_string()),
                    ("depth", format!("{:.1}", m.depth)),
                    ("sh", m.sh.to_string()),
                    ("n_spikes", m.n_spikes.to_string()),
                    ("fr", format!("{:.2}", m.fr)),
                    ("amp", format!("{:.2}", m.amp)),
                    ("contam_pct", format!("{:.2}", m.contam_pct)),
                    ("KSLabel", m.ks_label.clone()),
                    ("group", m.group.as_str().to_string()),
                ]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
                row.extend(m.custom.clone());
                (id, row)
            })
            .collect();
        PhySorting::save_curation(dir, &self.sorting.spike_clusters, &groups, &columns, &rows).with_context(|| format!("Could not save the curation in {}", dir.display()))?;
        self.sorting.folder = Some(dir.to_path_buf());
        self.saved_depth = self.undo_stack.len();
        Ok(())
    }

    /// Saves into the folder the sorting came from.
    pub fn save(&mut self) -> Result<()> {
        let dir = self.sorting.folder.clone().context("This sorting was not opened from a phy folder: choose where to save it")?;
        self.save_to(&dir)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use dsp_synapse::{SortedUnit, SortingOutput, UnitQualityLabel, WaveformTemplate};

    pub fn scratch_dir(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("dsp-app-curation-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// Six units on 8 channels over 10 s at 30 kHz, each with a template peaking on channel 2u+1
    /// (mod 8): a test sorting, saved as a phy folder.
    pub fn synthetic_folder(name: &str) -> PathBuf {
        let (rate, total, channels, samples) = (30_000.0, 300_000u64, 8usize, 61usize);
        let units = (0..6)
            .map(|u| {
                let step = (rate / (8.0 + u as f64 * 4.5)) as u64;
                let times: Vec<u64> = (0..).map(|i| 150 * (u as u64 + 1) + i * step).take_while(|&t| t + 40 < total).collect();
                let amps = times.iter().enumerate().map(|(i, _)| 45.0 + u as f32 * 18.0 + (i % 7) as f32).collect();
                let best = (u * 2 + 1) % channels;
                let mean: Vec<f32> = (0..channels).flat_map(|c| (0..samples).map(move |s| if c == best && s == 20 { -60.0 - 10.0 * u as f32 } else { 0.0 })).collect();
                let template = WaveformTemplate::with_count((0..channels).collect(), samples, times.len(), mean.clone(), vec![1.0; mean.len()]);
                let mut unit = SortedUnit::from_spikes(u, best, times, amps, Vec::new(), Some(template), rate, total, 5.0);
                unit.quality_label = [UnitQualityLabel::SingleUnit, UnitQualityLabel::MultiUnit, UnitQualityLabel::Noise][u % 3];
                unit
            })
            .collect();
        let so = SortingOutput::new("synthetic", rate, total, None, units, None);
        let dir = scratch_dir(name);
        dsp_synapse::save_phy_folder(&so, &dir).unwrap();
        dir
    }

    #[test]
    fn test_clusters_table_from_a_folder() {
        let dir = synthetic_folder("table");
        let data = SortingData::open(&dir).unwrap();
        assert_eq!(data.clusters.len(), 6);
        assert!(!data.dirty());
        let m = &data.clusters[&2];
        assert_eq!(m.ch, 5, "best channel from the template");
        assert_eq!(m.group, ClusterGroup::Noise);
        assert!(m.n_spikes > 100 && m.fr > 10.0);
        assert_eq!(data.top_channels(2, 3)[0], 5);
        // Similar: every other cluster, best first
        let sim = data.similar_to(0);
        assert_eq!(sim.len(), 5);
        assert!(sim.windows(2).all(|w| w[0].1 >= w[1].1));
    }

    #[test]
    fn test_merge_split_label_undo_redo_and_save_round_trip() {
        let dir = synthetic_folder("roundtrip");
        std::fs::write(dir.join("untouched_marker.txt"), "keep me").unwrap();
        let mut data = SortingData::open(&dir).unwrap();
        let (n0, n1) = (data.spike_indices(0).len(), data.spike_indices(1).len());

        let (merged, _) = data.merge(&[0, 1], &[0, 1]).unwrap();
        assert!(data.dirty());
        assert!(!data.clusters.contains_key(&0) && !data.clusters.contains_key(&1));
        assert_eq!(data.clusters[&merged].n_spikes, n0 + n1);
        assert_eq!(data.representative_template(merged), Some(1), "the template most spikes were matched with");

        let inside: Vec<usize> = data.spike_indices(merged).into_iter().filter(|&i| data.sorting.spike_templates[i] == 0).collect();
        let (a, b, _) = data.split(merged, &inside, &[merged]).unwrap();
        assert_eq!((data.clusters[&a].n_spikes, data.clusters[&b].n_spikes), (n0, n1));

        assert_eq!(data.undo().unwrap().1, Some(vec![merged]));
        assert_eq!(data.undo().unwrap().1, Some(vec![0, 1]));
        assert!(!data.dirty(), "back where it was opened");
        assert_eq!(data.clusters[&0].n_spikes, n0);

        data.redo().unwrap();
        assert!(data.set_group(&[merged], ClusterGroup::Good, &[merged], &[2]));
        assert!(data.set_custom_label(&[merged], "brain_area", "V1"));
        data.save().unwrap();
        assert!(!data.dirty());
        assert!(dir.join("spike_clusters.npy.bak").exists());
        assert_eq!(std::fs::read_to_string(dir.join("untouched_marker.txt")).unwrap(), "keep me");

        let reopened = SortingData::open(&dir).unwrap();
        assert_eq!(reopened.clusters[&merged].group, ClusterGroup::Good);
        assert_eq!(reopened.clusters[&merged].n_spikes, n0 + n1);
        assert_eq!(reopened.clusters[&merged].custom.get("brain_area").map(String::as_str), Some("V1"));
    }

    /// The user's Kilosort4 output: `DSP_KS4_FOLDER=<saved_results>`.
    #[test]
    #[ignore]
    fn test_real_kilosort4_folder() {
        let Some(dir) = std::env::var_os("DSP_KS4_FOLDER") else { return };
        let data = SortingData::open(Path::new(&dir)).unwrap();
        assert!(data.clusters.len() > 200);
        assert!(!data.similar_to(0).is_empty());
        assert_eq!(data.compute_correlograms(&[0, 1], 1.0, 25.0, 2.0).cells.len(), 2);
    }
}
