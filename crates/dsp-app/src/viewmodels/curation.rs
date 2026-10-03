//! The Curation workspace's state: the open sorting, the selected clusters (the first is the
//! "best" one, as in phy; the others are compared with it), how the Clusters table is sorted and
//! filtered, and the view data computed for the selection.
//!
//! Opening reads the sorting on a background thread; view data (waveforms, correlograms) is
//! computed on the shared work pool, the newest request per view winning, and comes back over
//! this view model's channel. A result for a selection that has since changed is dropped.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use gpui_kit::{AppContext as _, Context, Entity, EventEmitter, Subscription, Task};

use crate::engine::curation::{matches_filter, ClusterId, ClusterMeta, ClusterSortColumn, ClusterWaveforms, CorrelogramMatrix, SortingData};
use crate::engine::data::Dataset;
use crate::engine::work_pool::JobKey;
use crate::store::{AppEvent, Store};
use crate::workspace::Workspace;

use super::services::Services;

/// Work-pool owner of the Curation views (time views use their view ids, which start at 1).
const OWNER: u64 = u64::MAX - 1;
const WAVEFORMS_SLOT: u32 = 1;
const CORRELOGRAMS_SLOT: u32 = 2;
/// Clusters overlaid in Waveforms; spikes read per cluster; channels shown.
const WAVEFORM_CLUSTERS: usize = 4;
const WAVEFORM_SPIKES: usize = 100;
const WAVEFORM_CHANNELS: usize = 12;
/// Correlogram bin and window (ms), refractory period marked (ms).
pub const CCG_BIN_MS: f32 = 0.5;
pub const CCG_WINDOW_MS: f32 = 50.0;
pub const REFRACTORY_MS: f32 = 2.0;

/// Whether `path` is a standalone sorting folder (Phy/Kilosort, `.sorting.zarr`, or NWB `/units`).
pub fn is_sorting(path: &Path) -> bool {
    let dir = if path.is_dir() { Some(path) } else { path.parent() };
    dir.is_some_and(|d| {
        d.join("spike_times.npy").is_file()
            || dsp_synapse::storage::zarr_store::has_array(d, "/spikes/times")
            || (dsp_synapse::storage::zarr_store::has_array(d, "/spike_times")
                && dsp_synapse::storage::zarr_store::has_array(d, "/spike_times_index"))
            || (dsp_synapse::storage::zarr_store::has_array(d, "/units/spike_times")
                && !d.join("acquisition").is_dir())
    })
}

/// Whether `path` is a recording container (e.g. `.nwb.zarr`) that also has an embedded sorting.
pub fn has_embedded_sorting(path: &Path) -> bool {
    path.is_dir() && dsp_synapse::storage::zarr_store::has_array(path, "/units/spike_times")
}

enum Computed {
    Waveforms(u64, Vec<ClusterWaveforms>),
    Correlograms(u64, CorrelogramMatrix),
}

pub enum CurationEvent {
    /// A sorting was opened (or failed to).
    Opened,
    /// The selection changed.
    Selection,
    /// New view data arrived.
    Data,
}

pub struct CurationVm {
    store: Entity<Store>,
    pub data: Option<Arc<SortingData>>,
    /// What is being opened, while it opens.
    pub opening: Option<String>,
    pub error: Option<String>,
    /// Selected clusters, best first.
    pub selected: Vec<ClusterId>,
    /// Bumped on every selection change; results for an older one are dropped.
    generation: u64,
    pub sort: (ClusterSortColumn, bool),
    pub filter: String,
    pub waveforms: Arc<Vec<ClusterWaveforms>>,
    pub correlograms: Option<CorrelogramMatrix>,
    results: async_channel::Sender<Computed>,
    _tasks: Vec<Task<()>>,
    _store: Subscription,
}

impl EventEmitter<CurationEvent> for CurationVm {}

impl CurationVm {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let (tx, rx) = async_channel::unbounded::<Computed>();
        let task = cx.spawn(async move |this, cx| {
            while let Ok(r) = rx.recv().await {
                if this.update(cx, |vm, cx| vm.on_result(r, cx)).is_err() {
                    break;
                }
            }
        });
        // The recording arriving (or changing) makes raw waveforms possible, or loads its embedded sorting
        let sub = cx.subscribe(&store, |vm, s, e: &AppEvent, cx| {
            if *e == AppEvent::RecordingChanged {
                let embedded_path = s
                    .read(cx)
                    .recording
                    .as_ref()
                    .and_then(|r| r.path.clone())
                    .filter(|p| has_embedded_sorting(p));
                let already_open = embedded_path.as_ref().is_some_and(|p| {
                    vm.data.as_ref().is_some_and(|d| d.sorting.folder.as_deref() == Some(p.as_path()))
                });
                if let Some(p) = embedded_path
                    && !already_open
                    && vm.opening.is_none()
                {
                    vm.open_with_mode(p, false, cx);
                } else {
                    vm.attach_recording(cx);
                    vm.compute(cx);
                }
            }
        });
        Self {
            store,
            data: None,
            opening: None,
            error: None,
            selected: Vec::new(),
            generation: 0,
            sort: (ClusterSortColumn::Id, false),
            filter: String::new(),
            waveforms: Arc::new(Vec::new()),
            correlograms: None,
            results: tx,
            _tasks: vec![task],
            _store: sub,
        }
    }

    /// The recording open in Explore, when it is the one this sorting was made from (same rate,
    /// enough channels for the channel map).
    pub fn recording(&self, cx: &gpui_kit::App) -> Option<Arc<Dataset>> {
        let data = self.data.as_ref()?;
        let ds = self.store.read(cx).sources()?.default_dataset();
        let needed = data.sorting.channel_map.iter().copied().max().map_or(0, |m| m + 1);
        ((ds.sample_rate - data.sample_rate()).abs() < 1e-6 && ds.total_channels >= needed).then_some(ds)
    }

    fn attach_recording(&mut self, cx: &mut Context<Self>) {
        let duration = self.recording(cx).map(|ds| ds.total_samples as f64 / ds.sample_rate);
        if let (Some(d), Some(data)) = (duration, self.data.as_mut()) {
            Arc::make_mut(data).set_recording_duration(d);
            let events = Arc::new(crate::engine::data::SpikeEventStore::from_sorting(data));
            self.store.update(cx, |s, cx| s.set_events(events, cx));
            cx.notify();
        }
    }

    /// Reads the sorting at `path` in the background; on success shows it in Curation and opens
    /// its recording in Explore (from `params.py` or Zarr metadata) when none is open.
    pub fn open(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        self.open_with_mode(path, true, cx);
    }

    fn open_with_mode(&mut self, path: PathBuf, switch_workspace: bool, cx: &mut Context<Self>) {
        let name = path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
        self.opening = Some(name.clone());
        self.error = None;
        if switch_workspace {
            self.store.update(cx, |s, cx| {
                s.set_workspace(Workspace::Curation, cx);
                s.set_status(format!("Opening the sorting {name}…"), cx);
            });
        }
        cx.notify();
        let task = cx.background_spawn({
            let path = path.clone();
            async move { SortingData::open(&path) }
        });
        cx.spawn(async move |this, cx| {
            let result = task.await;
            let _ = this.update(cx, |vm, cx| vm.opened(path, result, switch_workspace, cx));
        })
        .detach();
    }

    fn opened(&mut self, path: PathBuf, result: anyhow::Result<SortingData>, switch_workspace: bool, cx: &mut Context<Self>) {
        self.opening = None;
        match result {
            Ok(data) => {
                let summary = format!("{} clusters, {} spikes", data.clusters.len(), data.sorting.spike_times.len());
                let recording = data.recording_path();
                self.selected = data.clusters.keys().next().copied().into_iter().collect();
                self.data = Some(Arc::new(data));
                self.store.update(cx, |s, cx| {
                    s.push_recent_sorting(&path);
                    if switch_workspace {
                        s.set_status(format!("Opened the sorting {} ({summary})", path.display()), cx);
                    } else if let Some(rec) = &s.recording {
                        s.set_status(format!("Opened {} ({}) · {summary}", rec.name, rec.summary()), cx);
                    }
                    if s.recording.is_none()
                        && let Some(rec) = recording
                    {
                        s.open(rec, cx);
                    }
                });
                self.attach_recording(cx);
                self.generation += 1;
                self.compute(cx);
            }
            Err(e) => {
                self.error = Some(format!("{e:#}"));
                if switch_workspace {
                    self.store.update(cx, |s, cx| s.set_status(format!("Could not open the sorting: {e:#}"), cx));
                }
            }
        }
        cx.emit(CurationEvent::Opened);
        cx.notify();
    }

    // ------------------------------------------------------------------------
    // Clusters table
    // ------------------------------------------------------------------------

    /// Rows of the Clusters table, filtered and sorted.
    pub fn rows(&self) -> Vec<ClusterMeta> {
        let Some(data) = &self.data else { return Vec::new() };
        let mut rows: Vec<ClusterMeta> = data.clusters.values().filter(|c| matches_filter(&self.filter, c)).cloned().collect();
        let (column, descending) = self.sort;
        rows.sort_by(|a, b| {
            let o = match column {
                ClusterSortColumn::Id => a.id.cmp(&b.id),
                ClusterSortColumn::Channel => a.ch.cmp(&b.ch),
                ClusterSortColumn::Depth => a.depth.total_cmp(&b.depth),
                ClusterSortColumn::Spikes => a.n_spikes.cmp(&b.n_spikes),
                ClusterSortColumn::FiringRate => a.fr.total_cmp(&b.fr),
                ClusterSortColumn::Amplitude => a.amp.total_cmp(&b.amp),
                ClusterSortColumn::ContamPct => a.contam_pct.total_cmp(&b.contam_pct),
                ClusterSortColumn::KsLabel => a.ks_label.cmp(&b.ks_label),
                ClusterSortColumn::Group => a.group.cmp(&b.group),
            };
            if descending { o.reverse() } else { o }.then(a.id.cmp(&b.id))
        });
        rows
    }

    /// Sorts by `column`; the same column again flips the order.
    pub fn sort_by(&mut self, column: ClusterSortColumn, cx: &mut Context<Self>) {
        self.sort = if self.sort.0 == column { (column, !self.sort.1) } else { (column, false) };
        cx.notify();
    }

    pub fn set_filter(&mut self, filter: String, cx: &mut Context<Self>) {
        self.filter = filter;
        cx.notify();
    }

    /// Clusters by decreasing similarity to the best selected one.
    pub fn similar(&self) -> Vec<(ClusterId, f32)> {
        match (&self.data, self.selected.first()) {
            (Some(d), Some(&best)) => d.similar_to(best),
            _ => Vec::new(),
        }
    }

    // ------------------------------------------------------------------------
    // Selection
    // ------------------------------------------------------------------------

    /// Click: only `cid`. Ctrl / Shift click (`add`): `cid` added to or removed from the selection.
    pub fn select(&mut self, cid: ClusterId, add: bool, cx: &mut Context<Self>) {
        let next = if add {
            let mut s = self.selected.clone();
            match s.iter().position(|&c| c == cid) {
                Some(i) if s.len() > 1 => {
                    s.remove(i);
                }
                Some(_) => {}
                None => s.push(cid),
            }
            s
        } else {
            vec![cid]
        };
        self.set_selection(next, cx);
    }

    /// A similar cluster clicked: the best one and it (Ctrl / Shift: added to the others).
    pub fn select_similar(&mut self, cid: ClusterId, add: bool, cx: &mut Context<Self>) {
        if add {
            return self.select(cid, true, cx);
        }
        let best = self.selected.first().copied();
        let next = best.into_iter().filter(|&b| b != cid).chain([cid]).collect();
        self.set_selection(next, cx);
    }

    pub fn set_selection(&mut self, selected: Vec<ClusterId>, cx: &mut Context<Self>) {
        if selected != self.selected && !selected.is_empty() {
            self.selected = selected;
            self.generation += 1;
            cx.emit(CurationEvent::Selection);
            self.compute(cx);
            cx.notify();
        }
    }

    // ------------------------------------------------------------------------
    // View data
    // ------------------------------------------------------------------------

    /// Asks the work pool for the selection's waveforms and correlograms.
    fn compute(&mut self, cx: &mut Context<Self>) {
        let Some(data) = self.data.clone() else { return };
        if self.selected.is_empty() {
            return;
        }
        let selected = self.selected.clone();
        let generation = self.generation;
        let recording = self.recording(cx);
        let pool = &cx.global::<Services>().pool;

        let (data_w, sel_w, tx) = (data.clone(), selected.clone(), self.results.clone());
        pool.request(
            JobKey::new(OWNER, WAVEFORMS_SLOT),
            Box::new(move |cancel| {
                // Every cluster on the best one's channels, so they overlay
                let channels = data_w.top_channels(sel_w[0], WAVEFORM_CHANNELS);
                let rec = recording.as_deref().map(|d| d as &dyn dsp_core::RecordingSource);
                let mut out = Vec::new();
                for &c in sel_w.iter().take(WAVEFORM_CLUSTERS) {
                    if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                        return;
                    }
                    out.push(data_w.waveforms_on(c, channels.clone(), rec, WAVEFORM_SPIKES));
                }
                let _ = tx.try_send(Computed::Waveforms(generation, out));
            }),
        );
        let tx = self.results.clone();
        pool.request(
            JobKey::new(OWNER, CORRELOGRAMS_SLOT),
            Box::new(move |cancel| {
                let ccg = data.compute_correlograms(&selected, CCG_BIN_MS, CCG_WINDOW_MS, REFRACTORY_MS);
                if !cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    let _ = tx.try_send(Computed::Correlograms(generation, ccg));
                }
            }),
        );
    }

    fn on_result(&mut self, r: Computed, cx: &mut Context<Self>) {
        match r {
            Computed::Waveforms(g, w) if g == self.generation => self.waveforms = Arc::new(w),
            Computed::Correlograms(g, c) if g == self.generation => self.correlograms = Some(c),
            _ => return,
        }
        cx.emit(CurationEvent::Data);
        cx.notify();
    }
}
