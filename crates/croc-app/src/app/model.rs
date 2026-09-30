//! Parent app model: data shared by every module (recording, events, spike sorting, cluster
//! selection), recent files, and which module is active.

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::data::{Dataset, SpikeEventStore};
use crate::modules::spikes::plots::selection_color;
use crate::modules::spikes::sorting::Sorting;
use crate::modules::time::module::SpikeMarks;

pub const MAX_RECENT: usize = 8;

/// Child module shown under the parent tabs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Module {
    #[default]
    Time,
    Spikes,
}

impl Module {
    pub const ALL: [Module; 2] = [Module::Time, Module::Spikes];

    pub fn index(self) -> usize {
        Self::ALL.iter().position(|&m| m == self).unwrap_or(0)
    }
}

/// Where the sorting stands.
#[derive(Debug, Clone, PartialEq)]
pub enum SortStatus {
    Idle,
    Running,
    Done { spikes: usize, clusters: usize, millis: u128 },
}

/// One row of the cluster panel.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterRow {
    pub id: u32,
    /// Position in the selection (its color), if selected.
    pub selection_index: Option<usize>,
    pub peak_channel: usize,
    pub spikes: usize,
    pub amplitude: f32,
    pub firing_rate_hz: f32,
    pub snr: f32,
    pub isi_violation_pct: f32,
}

pub struct AppModel {
    pub dataset: Arc<Dataset>,
    pub dataset_path: Option<PathBuf>,
    pub events: Arc<SpikeEventStore>,
    /// False when the recording was too long to detect events at open time.
    pub events_detected: bool,
    pub recent: Vec<PathBuf>,
    pub active: Module,
    pub sorting: Option<Arc<Sorting>>,
    pub sort_status: SortStatus,
    /// Selected clusters, in selection order (their colors follow this order).
    pub selected_clusters: Vec<u32>,
    /// Bumped whenever the dataset changes, so stale sorting results can be dropped.
    pub generation: u64,
}

impl AppModel {
    pub fn new(dataset: Dataset, dataset_path: Option<PathBuf>) -> Self {
        let (events, events_detected) = detect_events(&dataset);
        Self {
            dataset: Arc::new(dataset),
            dataset_path,
            events: Arc::new(events),
            events_detected,
            recent: Vec::new(),
            active: Module::Time,
            sorting: None,
            sort_status: SortStatus::Idle,
            selected_clusters: Vec::new(),
            generation: 0,
        }
    }

    pub fn load_dataset(&mut self, dataset: Dataset, path: Option<PathBuf>) {
        let (events, detected) = detect_events(&dataset);
        self.events = Arc::new(events);
        self.events_detected = detected;
        self.dataset = Arc::new(dataset);
        self.generation += 1;
        self.sorting = None;
        self.sort_status = SortStatus::Idle;
        self.selected_clusters.clear();
        if let Some(p) = &path {
            self.push_recent(p.clone());
        }
        self.dataset_path = path;
    }

    pub fn push_recent(&mut self, path: PathBuf) {
        self.recent.retain(|p| p != &path);
        self.recent.insert(0, path);
        self.recent.truncate(MAX_RECENT);
    }

    // ------------------------------------------------------------------------
    // Sorting
    // ------------------------------------------------------------------------

    /// Starts a sorting run: what the background thread needs, or `None` if one is running.
    pub fn begin_sorting(&mut self) -> Option<(Arc<Dataset>, u64)> {
        if self.sort_status == SortStatus::Running {
            return None;
        }
        self.sort_status = SortStatus::Running;
        Some((self.dataset.clone(), self.generation))
    }

    /// Installs a finished sorting; returns false (and ignores it) if the dataset changed.
    /// Keeps selected ids that still exist, else selects the largest cluster.
    pub fn finish_sorting(&mut self, generation: u64, sorting: Sorting, millis: u128) -> bool {
        if generation != self.generation {
            return false;
        }
        self.sort_status = SortStatus::Done { spikes: sorting.num_spikes(), clusters: sorting.clusters.len(), millis };
        self.selected_clusters.retain(|&id| sorting.cluster(id).is_some());
        if self.selected_clusters.is_empty() {
            if let Some(c) = sorting.clusters.iter().max_by_key(|c| c.spikes.len()) {
                self.selected_clusters.push(c.id);
            }
        }
        self.sorting = Some(Arc::new(sorting));
        true
    }

    pub fn sort_status_text(&self) -> String {
        match &self.sort_status {
            SortStatus::Idle => "Not sorted yet".into(),
            SortStatus::Running => "Sorting…".into(),
            SortStatus::Done { spikes, clusters, millis } => {
                let mut text = format!("{spikes} spikes · {clusters} clusters · {millis} ms");
                if let Some(s) = self.sorting.as_ref().filter(|s| s.duration_sec + 1e-6 < self.dataset.total_duration_sec()) {
                    text.push_str(&format!(" · first {:.1} s", s.duration_sec));
                }
                text
            }
        }
    }

    // ------------------------------------------------------------------------
    // Cluster selection (shared: the spike module edits it, the time module can overlay it)
    // ------------------------------------------------------------------------

    /// Click: replace the selection, or toggle the cluster in (`additive`, Ctrl/Shift).
    pub fn select_cluster(&mut self, id: u32, additive: bool) {
        if additive {
            if let Some(i) = self.selected_clusters.iter().position(|&c| c == id) {
                self.selected_clusters.remove(i);
            } else {
                self.selected_clusters.push(id);
            }
        } else {
            self.selected_clusters = vec![id];
        }
    }

    /// Keyboard: move the (single) selection to the next / previous cluster.
    pub fn step_cluster(&mut self, delta: i32) {
        let Some(s) = &self.sorting else { return };
        let n = s.clusters.len() as i64;
        if n == 0 {
            return;
        }
        let current = self.selected_clusters.first().map_or(-1, |&c| c as i64);
        let next = (current + delta as i64).rem_euclid(n) as u32;
        self.select_cluster(next, false);
    }

    pub fn cluster_rows(&self) -> Vec<ClusterRow> {
        let Some(s) = &self.sorting else { return Vec::new() };
        s.clusters
            .iter()
            .map(|c| ClusterRow {
                id: c.id,
                selection_index: self.selected_clusters.iter().position(|&x| x == c.id),
                peak_channel: c.peak_channel,
                spikes: c.spikes.len(),
                amplitude: c.amplitude,
                firing_rate_hz: c.firing_rate_hz,
                snr: c.snr,
                isi_violation_pct: c.isi_violation_pct,
            })
            .collect()
    }

    /// Spikes of the selected clusters inside `[t0, t1]`, colored by selection order.
    pub fn spike_marks(&self, (t0, t1): (f64, f64)) -> SpikeMarks {
        let Some(s) = &self.sorting else { return Vec::new() };
        self.selected_clusters
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| {
                let c = s.cluster(id)?;
                let lo = c.spikes.partition_point(|&k| s.times_sec[k] < t0);
                let hi = c.spikes.partition_point(|&k| s.times_sec[k] <= t1);
                let spikes = c.spikes[lo..hi].iter().map(|&k| (s.times_sec[k], s.batch.primary_channels[k])).collect();
                Some((selection_color(i), spikes))
            })
            .collect()
    }

    /// Sort with `params` synchronously (tests).
    #[cfg(test)]
    pub fn sort_now(&mut self, params: &crate::modules::spikes::sorting::SortParams) {
        let (ds, generation) = self.begin_sorting().expect("idle");
        let s = Sorting::run(&ds, params);
        self.finish_sorting(generation, s, 0);
    }
}

/// Events for the timeline, detected now only when the recording fits the budget.
fn detect_events(dataset: &Dataset) -> (SpikeEventStore, bool) {
    if SpikeEventStore::fits_budget(dataset) {
        (SpikeEventStore::detect(dataset), true)
    } else {
        (SpikeEventStore::default(), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::spikes::sorting::tests::synthetic_params;

    #[test]
    fn test_sorting_selection_and_marks() {
        let mut app = AppModel::new(Dataset::generate_synthetic(8, 30_000.0, 2.0), None);
        app.sort_now(&synthetic_params(8));
        assert!(matches!(app.sort_status, SortStatus::Done { .. }));
        assert_eq!(app.selected_clusters.len(), 1);

        let rows = app.cluster_rows();
        if let Some(other) = rows.iter().find(|r| r.selection_index.is_none()).map(|r| r.id) {
            app.select_cluster(other, true);
            assert_eq!(app.cluster_rows().iter().find(|r| r.id == other).unwrap().selection_index, Some(1));
            app.select_cluster(other, false);
            assert_eq!(app.selected_clusters, vec![other]);
        }
        let marks = app.spike_marks((0.0, 2.0));
        assert_eq!(marks.len(), 1);
        assert!(!marks[0].1.is_empty());

        app.step_cluster(1);
        assert_eq!(app.selected_clusters.len(), 1);
    }

    #[test]
    fn test_stale_sorting_is_ignored() {
        let mut app = AppModel::new(Dataset::generate_synthetic(8, 30_000.0, 1.0), None);
        let (ds, generation) = app.begin_sorting().unwrap();
        app.load_dataset(Dataset::generate_synthetic(8, 30_000.0, 1.0), None);
        assert!(!app.finish_sorting(generation, Sorting::run(&ds, &synthetic_params(8)), 1));
        assert!(app.sorting.is_none());
    }
}
