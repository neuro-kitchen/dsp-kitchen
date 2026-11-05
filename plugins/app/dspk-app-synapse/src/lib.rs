//! DSP App plugin for electrophysiology. Adds to the core app:
//! - formats with probe geometry (SpikeGLX, NWB through `dspk_synapse::io`), the probe as a
//!   channel layout (shanks, sites) — [`opener`];
//! - synthetic recordings (noise, hum, drifting units);
//! - sortings (`.sorting.zarr`) linked to a view (`views[i].plugins.synapse.sorting`), loaded
//!   once and indexed by time — [`sorting`];
//! - the spike highlight over traces: the trace faint, each spike in its unit's colour over its
//!   template's extent (samples and channels), unit toggles;
//! - sorting from the app through the separate `dspk-cli-sorter` program — [`sort`];
//! - the Processing panel's Sorting section — [`panel`] — and the Sorting workspace —
//!   [`workspace`].

pub mod opener;
pub mod panel;
pub mod plots;
pub mod sort;
pub mod sorting;
pub mod table;
pub mod workspace;

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

use dspk_app_core::data::store::{AppEvent, Store};
use dspk_app_core::domain::session::ViewDoc;
use dspk_app_core::navigation::route::Workspace;
use dspk_app_core::plugin::{AppPlugin, OverlayMark, OverlayUnit, Raster, RasterRow, SourceOpener, TraceOverlay};
use dspk_core::RecordingSource;
use dspk_synapse::io::{SyntheticParams, SyntheticRecording};
use gpui_kit::{AnyElement, AnyView, App, AppContext as _, Context, Entity, EventEmitter, Global, Subscription, Window};

use crate::sorting::SortingData;

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

/// The plugin's state: loaded sortings, hidden units, the Sort choices, the unit selected in the
/// Sorting workspace.
pub struct SynapseState {
    pub store: Entity<Store>,
    pub sortings: HashMap<PathBuf, Arc<SortingData>>,
    loading: HashSet<PathBuf>,
    /// Units (by index) not highlighted, per sorting.
    pub hidden: HashMap<PathBuf, HashSet<usize>>,
    /// Index into [`sort::SORTERS`] and [`sort::PROBES`].
    pub sorter: usize,
    pub probe: usize,
    _store: Subscription,
}

/// What changed in the plugin's state.
pub enum SynapseEvent {
    SortingLoaded(PathBuf),
    UnitsChanged,
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
        let mut this = Self { store, sortings: HashMap::new(), loading: HashSet::new(), hidden: HashMap::new(), sorter: 0, probe: 0, _store: sub };
        this.load_active(cx);
        this
    }

    /// The active view's sorting, loaded once.
    pub fn active_sorting(&self, cx: &App) -> Option<Arc<SortingData>> {
        let path = sorting_of(self.store.read(cx).active_view()?)?;
        self.sortings.get(&path).cloned()
    }

    /// Loads the active view's sorting in the background when it is not loaded yet.
    pub fn load_active(&mut self, cx: &mut Context<Self>) {
        let Some(path) = self.store.read(cx).active_view().and_then(sorting_of) else { return };
        if self.sortings.contains_key(&path) || !self.loading.insert(path.clone()) {
            return;
        }
        let task = cx.background_spawn({
            let path = path.clone();
            async move { SortingData::load(&path) }
        });
        cx.spawn(async move |this, cx| {
            let result = task.await;
            let _ = this.update(cx, |s, cx| {
                s.loading.remove(&path);
                match result {
                    Ok(data) => {
                        let summary = data.summary();
                        s.sortings.insert(path.clone(), Arc::new(data));
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
        let n = self.sortings.get(sorting).map_or(0, |s| s.units.len());
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
}

impl SynapsePlugin {
    pub fn new() -> Self {
        Self { opener: opener::SynapseOpener }
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
    }

    /// The active view's sorting over `source`: every spike near the window, on its unit's
    /// extent channels, over its template's samples; the traces faint.
    fn trace_overlay(&self, store: &Store, source: &str, window_sec: Range<f64>, cx: &App) -> Option<TraceOverlay> {
        if window_sec.end - window_sec.start > OVERLAY_MAX_SEC {
            return None;
        }
        let view = store.active_view()?;
        if store.view_source(view) != source {
            return None;
        }
        let state = state(cx)?;
        let state = state.read(cx);
        let path = sorting_of(view)?;
        let sorting = state.sortings.get(&path)?;
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
        let hidden = state.hidden.get(&path);
        let marks = sorting
            .in_range(s0, s1)
            .filter_map(|k| {
                let u = sorting.unit_of[k]? as usize;
                if hidden.is_some_and(|h| h.contains(&u)) {
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
    fn raster(&self, store: &Store, source: &str, window_sec: Range<f64>, cx: &App) -> Option<Raster> {
        let view = store.active_view()?;
        if store.view_source(view) != source {
            return None;
        }
        let state = state(cx)?;
        let state = state.read(cx);
        let path = sorting_of(view)?;
        let sorting = state.sortings.get(&path)?;
        let start = store.sources()?.get(source).start_time_sec;
        let rate = sorting.sample_rate;
        let palette = store.palette();
        let hidden = state.hidden.get(&path);
        let mut order: Vec<usize> = (0..sorting.units.len()).filter(|u| !hidden.is_some_and(|h| h.contains(u))).collect();
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
        ws == Workspace::Sorting
    }

    fn workspace(&self, ws: Workspace, store: &Entity<Store>, window: &mut Window, cx: &mut App) -> Option<AnyView> {
        (ws == Workspace::Sorting).then(|| cx.new(|cx| workspace::SortingWorkspace::new(store.clone(), window, cx)).into())
    }

    fn panel(&self) -> Option<(&'static str, std::rc::Rc<dyn Fn() -> gpui_kit::component::Icon>)> {
        Some(("Sorting", std::rc::Rc::new(|| gpui_kit::component::Icon::new(gpui_kit::assets::IconName::Microscope))))
    }

    fn view_section(&self, store: &Entity<Store>, view: &ViewDoc, window: &mut Window, cx: &mut App) -> Option<AnyElement> {
        Some(panel::section(store, view, window, cx))
    }
}
