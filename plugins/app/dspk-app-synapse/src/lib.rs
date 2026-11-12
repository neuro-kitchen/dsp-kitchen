//! DSP App plugin for electrophysiology. Adds to the core app:
//! - formats with probe geometry (SpikeGLX, NWB through `dspk_synapse::io`), the probe as a
//!   channel layout (shanks, sites) — [`opener`];
//! - synthetic recordings (noise, hum, drifting units);
//! - sortings (`.sorting.zarr`) linked to a view (`views[i].plugins.synapse.sorting`), loaded
//!   once and indexed by time — [`sorting`];
//! - the spike highlight over traces: the trace faint, each spike in its unit's colour over its
//!   template's extent (samples and channels), unit toggles;
//! - sorting from the app through the separate `dspk-cli-sorter` program — [`sort`];
//! - the Sorting panel of Explore — [`panel`];
//! - the Sorting workspace (the installed sorters, their settings, runs on a part of the
//!   recording, the result as unit cards and clusters per channel) — [`sorting_ws`];
//! - the Curation workspace (Phy style: unit table, waveforms, features, correlograms,
//!   similarity; labels and merges kept as a log in the store) — [`curation_ws`], [`curation`];
//! - the bottom strip of both: a simple trace view or raster over the shared timeline — [`strip`].

pub mod aligned;
pub mod chart;
pub mod curation;
pub mod curation_ws;
pub mod opener;
pub mod panel;
pub mod plots;
pub mod sort;
pub mod sorting;
pub mod sorting_ws;
pub mod strip;
pub mod table;
pub mod templates;

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

use dspk_app_core::data::store::{AppEvent, Store};
use dspk_app_core::domain::session::ViewDoc;
use dspk_app_core::navigation::route::Workspace;
use dspk_app_core::plugin::{AppPlugin, Layers, OverlayMark, OverlayUnit, Raster, RasterRow, SourceOpener, TraceOverlay};
use dspk_core::RecordingSource;
use dspk_synapse::io::{SyntheticParams, SyntheticRecording};
use gpui_kit::{AnyElement, AnyView, App, AppContext as _, Context, Entity, EventEmitter, Global, Subscription, Window};

use crate::curation::{Curation, CurationOp};
use crate::sorting::{Snippets, SortingData};

/// The plugin's key in the session document (`views[i].plugins.synapse`).
pub const ID: &str = "synapse";
/// Spikes are highlighted while the window is at most this long (s): longer, they would cover
/// the traces.
pub const OVERLAY_MAX_SEC: f64 = 30.0;
/// Channels per spiking unit of a synthetic recording (each unit spreads over a few channels).
const SYNTHETIC_CHANNELS_PER_UNIT: usize = 4;
/// Most spiking units of a synthetic recording.
const SYNTHETIC_MAX_UNITS: usize = 64;

/// The `.sorting.zarr` linked to `view`, if any.
pub fn sorting_of(view: &ViewDoc) -> Option<PathBuf> {
    view.plugins.get(ID)?.get("sorting")?.as_str().map(PathBuf::from)
}

/// The sortings made for `view` from the Sorting workspace (its runs), oldest first.
pub fn runs_of(view: &ViewDoc) -> Vec<PathBuf> {
    view.plugins.get(ID).and_then(|d| d.get("runs")).and_then(|r| r.as_array()).map(|a| a.iter().filter_map(|p| p.as_str().map(PathBuf::from)).collect()).unwrap_or_default()
}

/// `view`'s plugin data with `sorting` linked (`None`: unlinked) and `run` added to its runs.
pub fn plugin_data(view: &ViewDoc, sorting: Option<&std::path::Path>, run: Option<&std::path::Path>) -> serde_json::Value {
    let mut runs = runs_of(view);
    if let Some(r) = run {
        if !runs.iter().any(|p| p == r) {
            runs.push(r.to_path_buf());
        }
    }
    let mut data = serde_json::Map::new();
    if let Some(s) = sorting {
        data.insert("sorting".into(), serde_json::json!(s));
    }
    if !runs.is_empty() {
        data.insert("runs".into(), serde_json::json!(runs));
    }
    if data.is_empty() { serde_json::Value::Null } else { serde_json::Value::Object(data) }
}

/// The plugin's state: loaded sortings (as the sorter wrote them) and their curation (the log
/// and the curated result the views show), hidden units, the Sort choices.
pub struct SynapseState {
    pub store: Entity<Store>,
    pub sortings: HashMap<PathBuf, Arc<SortingData>>,
    pub curations: HashMap<PathBuf, Curation>,
    curated: HashMap<PathBuf, Arc<SortingData>>,
    loading: HashSet<PathBuf>,
    /// Units (by index) not highlighted, per sorting.
    pub hidden: HashMap<PathBuf, HashSet<usize>>,
    /// Units a workspace is focused on (Sorting, Curation): its strip marks only those.
    pub focus: HashMap<Workspace, Vec<usize>>,
    /// Index into [`sort::SORTERS`] and [`sort::PROBES`].
    pub sorter: usize,
    pub probe: usize,
    _store: Subscription,
}

/// What changed in the plugin's state.
pub enum SynapseEvent {
    SortingLoaded(PathBuf),
    UnitsChanged,
    /// The curation of a sorting changed (labels, merges, undo).
    Curated(PathBuf),
}

impl EventEmitter<SynapseEvent> for SynapseState {}

struct Synapse(Entity<SynapseState>);
impl Global for Synapse {}

/// The plugin's state (once installed).
pub fn state(cx: &App) -> Option<Entity<SynapseState>> {
    cx.try_global::<Synapse>().map(|s| s.0.clone())
}

impl SynapseState {
    fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let sub = cx.subscribe(&store, |this, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::Document(_) | AppEvent::RecordingChanged) {
                this.load_active(cx);
            }
        });
        let mut this = Self {
            store,
            sortings: HashMap::new(),
            curations: HashMap::new(),
            curated: HashMap::new(),
            loading: HashSet::new(),
            hidden: HashMap::new(),
            focus: HashMap::new(),
            sorter: 0,
            probe: 0,
            _store: sub,
        };
        this.load_active(cx);
        this
    }

    /// The active view's sorting (curated), loaded once.
    pub fn active_sorting(&self, cx: &App) -> Option<Arc<SortingData>> {
        let path = sorting_of(self.store.read(cx).active_view()?)?;
        self.sorting(&path)
    }

    /// A loaded sorting as curated.
    pub fn sorting(&self, path: &PathBuf) -> Option<Arc<SortingData>> {
        self.curated.get(path).or_else(|| self.sortings.get(path)).cloned()
    }

    /// Loads the active view's sorting in the background when it is not loaded yet.
    pub fn load_active(&mut self, cx: &mut Context<Self>) {
        if let Some(path) = self.store.read(cx).active_view().and_then(sorting_of) {
            self.load(path, cx);
        }
    }

    /// Loads the sorting at `path` (and its curation log) in the background, once.
    pub fn load(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        if self.sortings.contains_key(&path) || !self.loading.insert(path.clone()) {
            return;
        }
        let task = cx.background_spawn({
            let path = path.clone();
            async move { SortingData::load(&path).map(|d| (d, Curation::load(&path))) }
        });
        cx.spawn(async move |this, cx| {
            let result = task.await;
            let _ = this.update(cx, |s, cx| {
                s.loading.remove(&path);
                match result {
                    Ok((data, log)) => {
                        let summary = data.summary();
                        s.sortings.insert(path.clone(), Arc::new(data));
                        match log {
                            Ok(log) => {
                                s.curations.insert(path.clone(), log);
                            }
                            Err(e) => s.store.update(cx, |st, cx| st.set_status(format!("Curation not applied: {e:#}"), cx)),
                        }
                        s.recurate(&path);
                        s.store.update(cx, |st, cx| {
                            st.set_status(format!("Sorting loaded: {summary}"), cx);
                            st.redraw(cx);
                        });
                        cx.emit(SynapseEvent::SortingLoaded(path));
                    }
                    Err(e) => s.store.update(cx, |st, cx| st.set_status(format!("{e:#}"), cx)),
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// The curated sorting of `path` from its log.
    fn recurate(&mut self, path: &PathBuf) {
        let (Some(base), Some(log)) = (self.sortings.get(path), self.curations.get(path)) else { return };
        if log.ops.is_empty() {
            self.curated.remove(path);
        } else {
            self.curated.insert(path.clone(), Arc::new(curation::apply(base, &log.ops)));
        }
    }

    fn curated_changed(&mut self, path: &PathBuf, cx: &mut Context<Self>) {
        self.recurate(path);
        self.store.update(cx, |s, cx| s.redraw(cx));
        cx.emit(SynapseEvent::Curated(path.clone()));
        cx.notify();
    }

    /// Adds a curation edit to `path`'s log.
    pub fn curate(&mut self, path: &PathBuf, op: CurationOp, cx: &mut Context<Self>) {
        self.curations.entry(path.clone()).or_default().push(op);
        self.curated_changed(path, cx);
    }

    pub fn undo(&mut self, path: &PathBuf, cx: &mut Context<Self>) {
        if self.curations.get_mut(path).is_some_and(Curation::undo) {
            self.curated_changed(path, cx);
        }
    }

    pub fn redo(&mut self, path: &PathBuf, cx: &mut Context<Self>) {
        if self.curations.get_mut(path).is_some_and(Curation::redo) {
            self.curated_changed(path, cx);
        }
    }

    /// Writes `path`'s curation log into the store (`curation.json`).
    pub fn save_curation(&mut self, path: &PathBuf, cx: &mut Context<Self>) {
        let Some(log) = self.curations.get_mut(path) else { return };
        let status = match log.save(path) {
            Ok(file) => format!("Curation saved: {} ({} edits)", dspk_app_core::domain::paths::short_path(&file), log.ops.len()),
            Err(e) => format!("Curation not saved: {e:#}"),
        };
        self.store.update(cx, |s, cx| s.set_status(status, cx));
        cx.emit(SynapseEvent::Curated(path.clone()));
        cx.notify();
    }

    /// The units workspace `ws` is focused on (its trace strip marks only those).
    pub fn set_focus(&mut self, ws: Workspace, units: Vec<usize>, cx: &mut Context<Self>) {
        if self.focus.get(&ws) == Some(&units) {
            return;
        }
        self.focus.insert(ws, units);
        self.store.update(cx, |s, cx| s.redraw(cx));
    }

    /// Whether unit `u` is marked on the traces of the workspace in front.
    fn shown_in(&self, ws: Workspace, path: &PathBuf, u: usize) -> bool {
        match self.focus.get(&ws) {
            Some(f) => f.contains(&u),
            None => !self.is_hidden(path, u),
        }
    }

    pub fn is_hidden(&self, sorting: &PathBuf, unit: usize) -> bool {
        self.hidden.get(sorting).is_some_and(|h| h.contains(&unit))
    }

    /// Shows or hides `unit` of `sorting` on the traces.
    pub fn toggle_unit(&mut self, sorting: &PathBuf, unit: usize, cx: &mut Context<Self>) {
        let h = self.hidden.entry(sorting.clone()).or_default();
        if !h.remove(&unit) {
            h.insert(unit);
        }
        self.units_changed(cx);
    }

    /// Shows every unit (`all`) or none.
    pub fn show_all(&mut self, sorting: &PathBuf, all: bool, cx: &mut Context<Self>) {
        let n = self.sorting(sorting).map_or(0, |s| s.units.len());
        self.hidden.insert(sorting.clone(), if all { HashSet::new() } else { (0..n).collect() });
        self.units_changed(cx);
    }

    /// A Sort choice changed (the panel redraws).
    pub fn choices_changed(&mut self, cx: &mut Context<Self>) {
        self.store.update(cx, |s, cx| s.plugin_changed(cx));
        cx.notify();
    }

    fn units_changed(&mut self, cx: &mut Context<Self>) {
        self.store.update(cx, |s, cx| s.redraw(cx));
        cx.emit(SynapseEvent::UnitsChanged);
        cx.notify();
    }
}

pub struct SynapsePlugin {
    opener: opener::SynapseOpener,
    /// Analytics: the last units-aligned-to-events cards.
    aligned: aligned::Cache,
}

impl SynapsePlugin {
    pub fn new() -> Self {
        Self { opener: opener::SynapseOpener, aligned: aligned::Cache::default() }
    }
}

impl Default for SynapsePlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl AppPlugin for SynapsePlugin {
    fn id(&self) -> &'static str {
        ID
    }

    fn name(&self) -> &'static str {
        "Electrophysiology"
    }

    fn opener(&self) -> Option<&dyn SourceOpener> {
        Some(&self.opener)
    }

    fn synthetic(&self, channels: usize, sample_rate: f64, duration_sec: f64) -> Option<anyhow::Result<Arc<dyn RecordingSource>>> {
        let units = (channels / SYNTHETIC_CHANNELS_PER_UNIT).clamp(1, SYNTHETIC_MAX_UNITS);
        let params = SyntheticParams { channels, sample_rate_hz: sample_rate, duration_sec, units, ..Default::default() };
        Some(SyntheticRecording::new(params).map(|r| Arc::new(r) as Arc<dyn RecordingSource>).map_err(Into::into))
    }

    fn install(&self, store: &Entity<Store>, cx: &mut App) {
        let state = cx.new(|cx| SynapseState::new(store.clone(), cx));
        cx.set_global(Synapse(state));
        // The device and its PCA kernels get ready while the user opens a recording, so the
        // first unit clicked in Curation does not wait for them (seconds, cold)
        cx.background_spawn(async {
            let snippets = Snippets { channel: 0, before: 2, after: 2, rows: (0..8).map(|k| (None, k as u64, vec![k as f32, 1.0, -(k as f32), 0.5])).collect() };
            let _ = snippets.pca2();
        })
        .detach();
    }

    /// The active view's sorting over `source`: every spike near the window, on its unit's
    /// extent channels, over its template's samples; the traces faint.
    fn trace_overlay(&self, store: &Store, view: &ViewDoc, window_sec: Range<f64>, cx: &App) -> Option<TraceOverlay> {
        if window_sec.end - window_sec.start > OVERLAY_MAX_SEC {
            return None;
        }
        let source = store.view_source(view);
        let source = source.as_str();
        let state = state(cx)?;
        let state = state.read(cx);
        let path = sorting_of(view)?;
        let sorting = state.sorting(&path)?;
        let start = store.sources()?.get(source).start_time_sec;
        let rate = sorting.sample_rate;
        // A view of a channel subset sorted on its own output: the sorting's channel i is the
        // view's i-th channel
        let map: Option<&Vec<usize>> = view.recording.channels.as_ref().filter(|c| sorting.channels == Some(c.len()));
        let palette = store.palette();
        let units = sorting
            .units
            .iter()
            .enumerate()
            .map(|(i, u)| {
                let mut channels: Vec<usize> = u.extent.channels.iter().filter_map(|&c| match map {
                    Some(m) => m.get(c).copied(),
                    None => Some(c),
                }).collect();
                channels.sort_unstable();
                OverlayUnit { color: palette.unit(i), channels }
            })
            .collect();
        let (max_before, max_after) = sorting.units.iter().fold((0, 0), |(b, a), u| (b.max(u.extent.before), a.max(u.extent.after)));
        let s0 = (((window_sec.start - start) * rate).max(0.0) as u64).saturating_sub(max_after);
        let s1 = ((window_sec.end - start) * rate).max(0.0) as u64 + max_before;
        let ws = store.workspace;
        let marks = sorting
            .in_range(s0, s1)
            .filter_map(|k| {
                let u = sorting.unit_of[k]? as usize;
                if !state.shown_in(ws, &path, u) {
                    return None;
                }
                let e = &sorting.units[u].extent;
                let t = sorting.times[k];
                Some(OverlayMark {
                    start_sec: start + t.saturating_sub(e.before) as f64 / rate,
                    end_sec: start + (t + e.after) as f64 / rate,
                    unit: u as u32,
                })
            })
            .collect();
        Some(TraceOverlay { dim: true, units, marks })
    }

    /// The active view's sorting as raster rows: units by primary channel (depth along the
    /// probe), hidden units left out, spikes in the window.
    fn layers(&self, _store: &Store, view: &ViewDoc, _cx: &App) -> Layers {
        // A linked sorting gives both: spike times on the traces, units in a raster
        let linked = sorting_of(view).is_some();
        Layers { spikes: linked, units: linked }
    }

    fn raster(&self, store: &Store, view: &ViewDoc, window_sec: Range<f64>, cx: &App) -> Option<Raster> {
        let source = store.view_source(view);
        let source = source.as_str();
        let state = state(cx)?;
        let state = state.read(cx);
        let path = sorting_of(view)?;
        let sorting = state.sorting(&path)?;
        let start = store.sources()?.get(source).start_time_sec;
        let rate = sorting.sample_rate;
        let palette = store.palette();
        let ws = store.workspace;
        let mut order: Vec<usize> = sorting.alive().filter(|&u| state.shown_in(ws, &path, u)).collect();
        order.sort_by_key(|&u| (sorting.units[u].info.primary_channel, u));
        let row_of: std::collections::HashMap<usize, usize> = order.iter().enumerate().map(|(r, &u)| (u, r)).collect();
        let mut rows: Vec<RasterRow> = order
            .iter()
            .map(|&u| RasterRow { color: palette.unit(u), label: format!("Unit {}", sorting.units[u].info.unit_id), times: Vec::new() })
            .collect();
        let s0 = ((window_sec.start - start) * rate).max(0.0) as u64;
        let s1 = ((window_sec.end - start) * rate).max(0.0) as u64;
        for k in sorting.in_range(s0, s1) {
            if let Some(&r) = sorting.unit_of[k].and_then(|u| row_of.get(&(u as usize))) {
                rows[r].times.push(start + sorting.times[k] as f64 / rate);
            }
        }
        Some(Raster { rows })
    }

    fn provides(&self, ws: Workspace) -> bool {
        matches!(ws, Workspace::Sorting | Workspace::Curation)
    }

    fn workspace(&self, ws: Workspace, store: &Entity<Store>, window: &mut Window, cx: &mut App) -> Option<AnyView> {
        match ws {
            Workspace::Sorting => Some(cx.new(|cx| sorting_ws::SortingWorkspace::new(store.clone(), window, cx)).into()),
            Workspace::Curation => Some(cx.new(|cx| curation_ws::CurationWorkspace::new(store.clone(), window, cx)).into()),
            _ => None,
        }
    }

    fn recording_epochs(&self, path: &std::path::Path, start_sec: f64) -> Vec<dspk_app_core::domain::epoch::EpochDoc> {
        if !dspk_synapse::io::nwb::is_nwb_zarr(path) {
            return Vec::new();
        }
        let intervals = dspk_synapse::io::nwb::read_intervals(path).unwrap_or_else(|e| {
            tracing::warn!("NWB intervals of {}: {e}", path.display());
            Vec::new()
        });
        intervals
            .into_iter()
            .filter(|i| i.stop_sec > start_sec)
            .map(|i| {
                // `epochs` → epoch, `trials` → trial: the type; named after the table and row
                let kind = i.table.strip_suffix('s').unwrap_or(&i.table).to_string();
                dspk_app_core::domain::epoch::EpochDoc {
                    id: String::new(),
                    recording: path.to_path_buf(),
                    name: format!("{kind} {}", i.row + 1),
                    kind,
                    start_sec: (i.start_sec - start_sec).max(0.0),
                    end_sec: i.stop_sec - start_sec,
                    notes: format!("NWB /intervals/{} row {}", i.table, i.row),
                }
            })
            .collect()
    }

    fn aligned(&self, store: &Entity<Store>, events_s: &[f64], pre_ms: f64, post_ms: f64, _: &mut Window, cx: &mut App) -> Option<AnyElement> {
        aligned::section(&self.aligned, store, events_s, pre_ms, post_ms, cx)
    }

    fn panel(&self) -> Option<(&'static str, std::rc::Rc<dyn Fn() -> gpui_kit::component::Icon>)> {
        Some(("Sorting", std::rc::Rc::new(|| gpui_kit::component::Icon::new(gpui_kit::assets::IconName::Microscope))))
    }

    fn view_section(&self, store: &Entity<Store>, view: &ViewDoc, window: &mut Window, cx: &mut App) -> Option<AnyElement> {
        Some(panel::section(store, view, window, cx))
    }
}
