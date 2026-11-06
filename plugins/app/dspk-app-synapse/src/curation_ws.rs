//! The Curation workspace (Phy style): the active view's sorting in views the user arranges, its
//! units judged and edited. Every view is a dock panel (drag a tab to snap it beside, under or
//! into another; close it), the Views menu toggles each one, the Layout menu applies a grid.
//! Views: the unit table (a Polars frame: sort by any column, filter by label), the selected
//! units' waveforms, their features (PC1 vs PC2), autocorrelogram, ISI, amplitude over time, the
//! lead unit's template, and the units most similar to it (template similarity: merge
//! candidates). Under them the bottom strip: the lead unit's channels over the shared timeline
//! (or a raster), previous / next spike.
//!
//! Editing, as in Phy: label the selected units good / mua / noise (Alt+G, Alt+M, Alt+N), merge
//! them (G), undo / redo (Ctrl+Z, Ctrl+Shift+Z), save (Ctrl+S) — a log inside the
//! `.sorting.zarr` ([`crate::curation`]); the sorter's arrays are never rewritten.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use dspk_app_core::data::store::{AppEvent, Store};
use dspk_app_core::screens::explore::trace::trace_panel::color;
use dspk_app_core::ui::plot::canvas::Pixel;
use dspk_app_core::ui::theme::palette::Palette;
use dspk_app_core::ui::widgets::{icon_button, EmptyState};
use gpui_kit::base::dock::{DockArea, DockLayout, Panel as BasePanel, PanelEvent};
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::dock::{panel_handle, DockSkin, Panel};
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenu, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Disableable as _, Icon, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, img, px, uniform_list, AnyElement, App, AppContext as _, Context, Entity, EventEmitter, FocusHandle, Focusable, FontWeight, InteractiveElement as _, IntoElement,
    KeyDownEvent, ParentElement as _, Render, RenderImage, SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, Task, WeakEntity, Window,
};

use crate::curation::{next_unit_id, similar, CurationOp, LABELS};
use crate::plots;
use crate::sorting::{Snippets, SortingData};
use crate::strip::{view_signal, BottomStrip};
use crate::table::{UnitTable, COLUMNS};
use crate::templates::TemplateGrid;
use crate::{sorting_of, state, SynapseEvent};

/// Spikes per selected unit drawn and projected (and as many unsorted ones).
pub const SPIKES_PER_UNIT: usize = 300;
/// Plot sizes (physical pixels; shown scaled to their panels).
const PLOT_W: u32 = 720;
const PLOT_H: u32 = 420;
/// Units listed by the similarity view.
const SIMILAR_MAX: usize = 20;
/// Selected units in the correlogram grid (auto on the diagonal, cross elsewhere).
const CCG_MAX_UNITS: usize = 4;
/// Autocorrelogram ±50 ms in 1 ms bins; ISI 0–50 ms in 0.5 ms bins; refractory period marked.
const ACG_HALF_MS: f64 = 50.0;
const ACG_BIN_MS: f64 = 1.0;
const ISI_MAX_MS: f64 = 50.0;
const ISI_BIN_MS: f64 = 0.5;
const REFRACTORY_MS: f64 = 1.5;
/// Spikes per unit in the amplitude-over-time plot.
const AMPLITUDE_POINTS: usize = 5000;
/// Table row height and the table's width in layouts (logical px).
const ROW_H: f32 = 24.;
const TABLE_W: f32 = 560.;

/// A view of the workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewKind {
    Table,
    Waveforms,
    Features,
    Acg,
    Isi,
    Amplitude,
    Template,
    Similarity,
    Correlograms,
    Templates,
    Probe,
}

impl ViewKind {
    pub const ALL: [ViewKind; 11] =
        [Self::Table, Self::Waveforms, Self::Features, Self::Acg, Self::Isi, Self::Amplitude, Self::Template, Self::Similarity, Self::Correlograms, Self::Templates, Self::Probe];

    pub fn title(self) -> &'static str {
        match self {
            Self::Table => "Units",
            Self::Waveforms => "Waveforms",
            Self::Features => "Features (PC1 · PC2)",
            Self::Acg => "Autocorrelogram",
            Self::Isi => "ISI",
            Self::Amplitude => "Amplitude over time (lasso: split)",
            Self::Template => "Template",
            Self::Similarity => "Similar units",
            Self::Correlograms => "Correlograms",
            Self::Templates => "All templates",
            Self::Probe => "Probe",
        }
    }
}

/// How the views are arranged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grid {
    /// Columns = √(plots).
    Auto,
    Columns(usize),
}

#[derive(Default, Clone)]
struct Plots {
    images: HashMap<ViewKind, Arc<RenderImage>>,
    note: String,
    /// The amplitude plot's axes: (recording length s, lowest amplitude, amplitude span).
    amp_axes: Option<(f32, f32, f32)>,
}

pub struct CurationWorkspace {
    store: Entity<Store>,
    dock: Entity<DockArea>,
    strip: Entity<BottomStrip>,
    templates: Entity<TemplateGrid>,
    focus: FocusHandle,
    panels: HashMap<ViewKind, Entity<CurationPanel>>,
    /// Views shown, in layout order.
    visible: Vec<ViewKind>,
    grid: Grid,
    /// Units selected in the table (indices), the first one leads (its channel, its spikes).
    selected: Vec<usize>,
    /// The table of the loaded sorting (built per curated state), its order (sorted, filtered),
    /// and how.
    table: Option<(*const SortingData, Arc<UnitTable>)>,
    order: Vec<usize>,
    sort_by: &'static str,
    descending: bool,
    label: Option<String>,
    plots: Plots,
    computing: bool,
    /// Bumped by every new selection: older work sees it and stops.
    generation: Arc<AtomicU64>,
    /// The keyboard help shows.
    show_keys: bool,
    _work: Option<Task<()>>,
    _subs: Vec<Subscription>,
}

impl CurationWorkspace {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let dock = cx.new(|cx| {
            let skin = DockSkin::new(cx);
            DockArea::new("curation", Some(1), window, cx).with_renderer(skin)
        });
        let strip = cx.new(|cx| BottomStrip::new(store.clone(), cx));
        let me = cx.weak_entity();
        let templates = cx.new(|_| {
            TemplateGrid::new(std::rc::Rc::new(move |u, add, _, cx| {
                let _ = me.update(cx, |w, cx| w.pick_unit(u, add, cx));
            }))
        });
        let mut subs = vec![cx.subscribe(&store, |this, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::Document(_) | AppEvent::RecordingChanged | AppEvent::PaletteChanged) {
                this.selected.clear();
                this.plots = Plots::default();
                this.refresh(cx);
            }
        })];
        if let Some(st) = state(cx) {
            subs.push(cx.subscribe(&st, |this, _, e: &SynapseEvent, cx| match e {
                SynapseEvent::SortingLoaded(_) => {
                    this.selected.clear();
                    this.refresh(cx);
                }
                SynapseEvent::Curated(_) => {
                    // Merged-away units leave the selection
                    if let Some(s) = this.sorting(cx) {
                        this.selected.retain(|&u| s.units.get(u).is_some_and(|u| u.merged_into.is_none()));
                    }
                    this.refresh(cx);
                }
                SynapseEvent::UnitsChanged => {}
            }));
        }
        let visible = vec![ViewKind::Table, ViewKind::Waveforms, ViewKind::Features, ViewKind::Correlograms, ViewKind::Isi, ViewKind::Amplitude, ViewKind::Template, ViewKind::Similarity];
        let mut this = Self {
            store,
            dock,
            strip,
            templates,
            focus: cx.focus_handle(),
            panels: HashMap::new(),
            visible,
            grid: Grid::Auto,
            selected: Vec::new(),
            table: None,
            order: Vec::new(),
            sort_by: "spikes",
            descending: true,
            label: None,
            plots: Plots::default(),
            computing: false,
            generation: Arc::new(AtomicU64::new(0)),
            show_keys: false,
            _work: None,
            _subs: subs,
        };
        this.apply_layout(window, cx);
        this.refresh(cx);
        this
    }

    fn sorting(&self, cx: &App) -> Option<Arc<SortingData>> {
        state(cx)?.read(cx).active_sorting(cx)
    }

    /// The table of the loaded sorting (built once per sorting), its order, a default selection.
    fn refresh(&mut self, cx: &mut Context<Self>) {
        let Some(sorting) = self.sorting(cx) else {
            self.table = None;
            self.order.clear();
            self.notify_panels(cx);
            return;
        };
        if self.table.as_ref().is_none_or(|(p, _)| *p != Arc::as_ptr(&sorting)) {
            match UnitTable::new(&sorting) {
                Ok(t) => self.table = Some((Arc::as_ptr(&sorting), Arc::new(t))),
                Err(e) => tracing::warn!("unit table: {e}"),
            }
        }
        self.reorder(cx);
        if self.selected.is_empty() {
            if let Some(&u) = self.order.first() {
                self.selected.push(u);
            }
        }
        self.compute(cx);
    }

    fn reorder(&mut self, cx: &mut Context<Self>) {
        if let Some((_, t)) = &self.table {
            match t.order(self.sort_by, self.descending, self.label.as_deref()) {
                Ok(o) => self.order = o,
                Err(e) => tracing::warn!("unit table: {e}"),
            }
        }
        self.notify_panels(cx);
    }

    fn sort_by(&mut self, column: &'static str, cx: &mut Context<Self>) {
        if self.sort_by == column {
            self.descending = !self.descending;
        } else {
            self.sort_by = column;
            self.descending = column != "unit" && column != "label";
        }
        self.reorder(cx);
    }

    fn set_label(&mut self, label: Option<String>, cx: &mut Context<Self>) {
        self.label = label;
        self.reorder(cx);
    }

    fn toggle_unit(&mut self, unit: usize, cx: &mut Context<Self>) {
        match self.selected.iter().position(|&u| u == unit) {
            Some(i) => {
                self.selected.remove(i);
            }
            None => self.selected.push(unit),
        }
        self.compute(cx);
    }

    /// Selects `unit` alone (a click without Ctrl / Shift, as in Phy).
    fn pick_unit(&mut self, unit: usize, add: bool, cx: &mut Context<Self>) {
        if add {
            self.toggle_unit(unit, cx);
        } else {
            self.selected = vec![unit];
            self.compute(cx);
        }
    }

    fn sorting_path(&self, cx: &App) -> Option<std::path::PathBuf> {
        sorting_of(self.store.read(cx).active_view()?)
    }

    /// The next (`dir` 1) or previous (−1) unit of the table's order, alone.
    fn step_unit(&mut self, dir: i64, cx: &mut Context<Self>) {
        if self.order.is_empty() {
            return;
        }
        let at = self.selected.first().and_then(|u| self.order.iter().position(|o| o == u));
        let next = match at {
            Some(i) => (i as i64 + dir).clamp(0, self.order.len() as i64 - 1) as usize,
            None => 0,
        };
        self.selected = vec![self.order[next]];
        self.compute(cx);
    }

    /// The lead unit with the next (or previous) unit most like it, for a merge decision
    /// (Phy's space / shift+space).
    fn step_similar(&mut self, dir: i64, cx: &mut Context<Self>) {
        let (Some(sorting), Some(&lead)) = (self.sorting(cx), self.selected.first()) else { return };
        let list = similar(&sorting, lead, SIMILAR_MAX);
        if list.is_empty() {
            return;
        }
        let at = self.selected.get(1).and_then(|u| list.iter().position(|(o, _)| o == u));
        let next = match at {
            Some(i) => (i as i64 + dir).clamp(0, list.len() as i64 - 1) as usize,
            None => 0,
        };
        self.selected = vec![lead, list[next].0];
        self.compute(cx);
    }

    /// Splits the lead unit: its spikes inside `polygon` (fractions of the amplitude plot)
    /// become a new unit, which joins the selection.
    fn split_lasso(&mut self, polygon: &[(f32, f32)], cx: &mut Context<Self>) {
        let (Some(path), Some(sorting), Some(&lead)) = (self.sorting_path(cx), self.sorting(cx), self.selected.first()) else { return };
        let Some((duration, lo, span)) = self.plots.amp_axes else { return };
        if polygon.len() < 3 || sorting.amplitudes.is_empty() {
            return;
        }
        let spikes: Vec<u64> = sorting.by_unit[lead]
            .iter()
            .map(|&k| k as usize)
            .filter(|&k| {
                let p = ((sorting.times[k] as f64 / sorting.sample_rate) as f32 / duration.max(1e-6), 1.0 - (sorting.amplitudes[k] - lo) / span.max(1e-6));
                inside(p, polygon)
            })
            .map(|k| sorting.times[k])
            .collect();
        let n = spikes.len();
        if n == 0 || n == sorting.by_unit[lead].len() {
            self.store.update(cx, |s, cx| s.set_status(if n == 0 { "The lasso holds none of the unit's spikes" } else { "The lasso holds every spike of the unit: nothing to split" }, cx));
            return;
        }
        let (unit, into, new_index) = (sorting.units[lead].info.unit_id, next_unit_id(&sorting), sorting.units.len());
        let Some(st) = state(cx) else { return };
        st.update(cx, |s, cx| s.curate(&path, CurationOp::Split { unit, spikes, into }, cx));
        self.store.update(cx, |s, cx| s.set_status(format!("Unit {unit}: {n} spikes split off as unit {into}"), cx));
        self.selected = vec![lead, new_index];
        self.compute(cx);
    }

    /// Labels the selected units.
    fn label_selected(&mut self, label: &str, cx: &mut Context<Self>) {
        let (Some(path), Some(sorting)) = (self.sorting_path(cx), self.sorting(cx)) else { return };
        let units: Vec<usize> = self.selected.iter().map(|&u| sorting.units[u].info.unit_id).collect();
        if units.is_empty() {
            return;
        }
        let Some(st) = state(cx) else { return };
        st.update(cx, |s, cx| s.curate(&path, CurationOp::Label { units, label: label.to_string() }, cx));
    }

    /// Merges the selected units into a new one, which is selected.
    fn merge_selected(&mut self, cx: &mut Context<Self>) {
        let (Some(path), Some(sorting)) = (self.sorting_path(cx), self.sorting(cx)) else { return };
        if self.selected.len() < 2 {
            return;
        }
        let units: Vec<usize> = self.selected.iter().map(|&u| sorting.units[u].info.unit_id).collect();
        let into = next_unit_id(&sorting);
        let new_index = sorting.units.len();
        let Some(st) = state(cx) else { return };
        st.update(cx, |s, cx| s.curate(&path, CurationOp::Merge { units, into }, cx));
        self.selected = vec![new_index];
        self.compute(cx);
    }

    fn undo(&mut self, redo: bool, cx: &mut Context<Self>) {
        let (Some(path), Some(st)) = (self.sorting_path(cx), state(cx)) else { return };
        st.update(cx, |s, cx| if redo { s.redo(&path, cx) } else { s.undo(&path, cx) });
    }

    fn save(&mut self, cx: &mut Context<Self>) {
        let (Some(path), Some(st)) = (self.sorting_path(cx), state(cx)) else { return };
        st.update(cx, |s, cx| s.save_curation(&path, cx));
    }

    fn on_key(&mut self, e: &KeyDownEvent, _: &mut Window, cx: &mut Context<Self>) {
        let k = &e.keystroke;
        let (cmd, alt, shift) = (k.modifiers.control || k.modifiers.platform, k.modifiers.alt, k.modifiers.shift);
        match k.key.as_str() {
            "g" if alt => self.label_selected("good", cx),
            "m" if alt => self.label_selected("mua", cx),
            "n" if alt => self.label_selected("noise", cx),
            "u" if alt => self.label_selected("unsorted", cx),
            "down" | "j" if !cmd => self.step_unit(1, cx),
            "up" | "k" if !cmd => self.step_unit(-1, cx),
            "space" if shift => self.step_similar(-1, cx),
            "space" => self.step_similar(1, cx),
            "?" | "h" if !cmd => {
                self.show_keys = !self.show_keys;
                cx.notify();
            }
            "escape" if self.show_keys => {
                self.show_keys = false;
                cx.notify();
            }
            "g" if !cmd => self.merge_selected(cx),
            "z" if cmd && shift => self.undo(true, cx),
            "z" if cmd => self.undo(false, cx),
            "y" if cmd => self.undo(true, cx),
            "s" if cmd => self.save(cx),
            _ => return,
        }
        cx.stop_propagation();
    }

    /// Shows or hides `kind`, then arranges the views again.
    fn toggle_view(&mut self, kind: ViewKind, window: &mut Window, cx: &mut Context<Self>) {
        match self.visible.iter().position(|&k| k == kind) {
            Some(i) => {
                self.visible.remove(i);
            }
            None => {
                self.visible.push(kind);
                self.visible.sort_by_key(|k| ViewKind::ALL.iter().position(|a| a == k));
            }
        }
        self.apply_layout(window, cx);
        self.compute(cx);
    }

    /// A view's tab was closed.
    fn closed(&mut self, kind: ViewKind, cx: &mut Context<Self>) {
        self.visible.retain(|&k| k != kind);
        cx.notify();
    }

    fn panel(&mut self, kind: ViewKind, cx: &mut Context<Self>) -> Entity<CurationPanel> {
        let ws = cx.weak_entity();
        self.panels.entry(kind).or_insert_with(|| cx.new(|cx| CurationPanel { ws, kind, focus: cx.focus_handle(), lasso: Vec::new(), plot: Default::default() })).clone()
    }

    /// The visible views in the grid: the table on the left, plots in rows of `columns`.
    fn apply_layout(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let plot_kinds: Vec<ViewKind> = self.visible.iter().copied().filter(|k| *k != ViewKind::Table).collect();
        let cols = match self.grid {
            Grid::Auto => (plot_kinds.len() as f32).sqrt().ceil().max(1.0) as usize,
            Grid::Columns(n) => n.max(1),
        };
        let mut grid = DockLayout::v_split();
        let mut any = false;
        for chunk in plot_kinds.chunks(cols) {
            let mut row = DockLayout::h_split();
            for &k in chunk {
                let p = self.panel(k, cx);
                row = row.child(DockLayout::tabs().panel_view(panel_handle(p), cx), None);
            }
            grid = grid.child(row, None);
            any = true;
        }
        let root = if self.visible.contains(&ViewKind::Table) {
            let table = self.panel(ViewKind::Table, cx);
            let mut root = DockLayout::h_split().child(DockLayout::tabs().panel_view(panel_handle(table), cx), Some(px(TABLE_W)));
            if any {
                root = root.child(grid, None);
            }
            root
        } else {
            grid
        };
        self.dock.update(cx, |d, cx| d.set_center(root, window, cx));
        cx.notify();
    }

    fn set_grid(&mut self, grid: Grid, window: &mut Window, cx: &mut Context<Self>) {
        self.grid = grid;
        self.apply_layout(window, cx);
    }

    /// A subset view sorted on its own output: the sorting's channel i is the view's i-th.
    fn channel_map(&self, sorting: &SortingData, cx: &App) -> impl Fn(usize) -> Option<usize> + Send + 'static {
        let map = self.store.read(cx).active_view().and_then(|v| v.recording.channels.clone()).filter(|c| sorting.channels == Some(c.len()));
        move |c: usize| map.as_ref().map_or(Some(c), |m| m.get(c).copied())
    }

    /// Every plot of the selected units, on a background thread (a newer selection replaces it).
    fn compute(&mut self, cx: &mut Context<Self>) {
        let sorting = self.sorting(cx);
        let (palette, selected) = (self.store.read(cx).palette(), self.selected.clone());
        self.templates.update(cx, |g, cx| g.set(sorting.clone(), selected, palette, cx));
        let Some(sorting) = sorting else { return };
        let Some(&lead) = self.selected.first() else {
            self.plots = Plots::default();
            self.notify_panels(cx);
            return;
        };
        // The last plots stay until the new ones replace them (no blank "Computing…" panels)
        let Some((signal, dataset, source)) = view_signal(self.store.read(cx)) else { return };
        let to_view = self.channel_map(&sorting, cx);
        let palette = self.store.read(cx).palette();
        let Some(channel) = to_view(sorting.units[lead].info.primary_channel) else { return };
        let units = self.selected.clone();
        let lead_colour = palette.unit(lead);
        let template_rows = sorting.units[lead].template.clone();
        let duration = (dataset.total_samples as f64 / dataset.sample_rate) as f32;
        // The strip follows the selection: the lead unit's channels
        let strip_channels: Vec<usize> = sorting.units[lead].extent.channels.iter().filter_map(|&c| to_view(c)).collect();
        let strip_units = units.clone();
        self.strip.update(cx, |s, cx| s.set_focus(strip_units, strip_channels, cx));
        let processed = self.store.read(cx).signal_for(&source).is_some();
        let note = format!("Ch {channel} · ≤{SPIKES_PER_UNIT} spikes per unit · {}", if processed { "processed" } else { "as stored" });
        let wanted: Vec<ViewKind> = self.visible.clone();
        // A newer selection makes this work stale: it stops at its next step
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let current = self.generation.clone();
        let stale = move || current.load(Ordering::SeqCst) != generation;
        self.computing = true;
        self.notify_panels(cx);
        // Fast: everything from the spike times alone (shown at once)
        let fast = cx.background_spawn({
            let (sorting, units, palette, stale, wanted) = (sorting.clone(), units.clone(), palette, stale.clone(), wanted.clone());
            async move {
                let mut images = HashMap::new();
                if stale() {
                    return (images, None);
                }
                let times = sorting.unit_times(lead);
                let acg = crate::sorting::autocorrelogram(&times, sorting.sample_rate, ACG_HALF_MS, ACG_BIN_MS);
                let isi = crate::sorting::isi_histogram(&times, sorting.sample_rate, ISI_MAX_MS, ISI_BIN_MS);
                let mut put = |k: ViewKind, f: Option<Arc<RenderImage>>| {
                    if let Some(i) = f {
                        images.insert(k, i);
                    }
                };
                put(ViewKind::Acg, plots::to_image(plots::bars(&acg, lead_colour, Some((ACG_HALF_MS / ACG_BIN_MS) as f32), &palette, PLOT_W, PLOT_H)));
                put(ViewKind::Isi, plots::to_image(plots::bars(&isi, lead_colour, Some((REFRACTORY_MS / ISI_BIN_MS) as f32), &palette, PLOT_W, PLOT_H)));
                put(ViewKind::Template, plots::to_image(plots::template_grid(&template_rows, lead_colour, &palette, PLOT_W, PLOT_H, 1.5)));
                if wanted.contains(&ViewKind::Correlograms) {
                    let shown: Vec<usize> = units.iter().copied().take(CCG_MAX_UNITS).collect();
                    let trains: Vec<Vec<u64>> = shown.iter().map(|&u| sorting.unit_times(u)).collect();
                    let grid: Vec<Vec<Vec<u32>>> = trains
                        .iter()
                        .map(|a| trains.iter().map(|b| crate::sorting::crosscorrelogram(a, b, sorting.sample_rate, ACG_HALF_MS, ACG_BIN_MS)).collect())
                        .collect();
                    let colours: Vec<_> = shown.iter().map(|&u| palette.unit(u)).collect();
                    put(ViewKind::Correlograms, plots::to_image(plots::correlogram_grid(&grid, &colours, &palette, PLOT_W, PLOT_H)));
                }
                if stale() {
                    return (images, None);
                }
                let amps: Vec<(u32, Vec<(f32, f32)>)> = units.iter().map(|&u| (u as u32, sorting.unit_amplitudes(u, AMPLITUDE_POINTS))).collect();
                let (frame, (lo, span)) = plots::amplitude_time(&amps, duration, &palette, PLOT_W, PLOT_H, 1.5);
                put(ViewKind::Amplitude, plots::to_image(frame));
                if wanted.contains(&ViewKind::Probe) {
                    let channels: Vec<(f32, f32)> = sorting.positions.iter().flatten().copied().collect();
                    let marks: Vec<((f32, f32), _, bool)> = sorting.alive().filter_map(|u| sorting.unit_position(u).map(|p| (p, palette.unit(u), units.contains(&u)))).collect();
                    put(ViewKind::Probe, plots::to_image(plots::probe_view(&channels, &marks, &palette, PLOT_W / 2, PLOT_H * 2, 1.5)));
                }
                (images, Some((duration, lo, span)))
            }
        });
        // Slow: waveforms and features from the signal's windows (read once per signal)
        let slow = cx.background_spawn(async move {
            let mut images = HashMap::new();
            if stale() || !(wanted.contains(&ViewKind::Waveforms) || wanted.contains(&ViewKind::Features)) {
                return images;
            }
            let windows = crate::sorting::windows_of(&signal);
            if stale() {
                return images;
            }
            let snippets = Snippets::read(&sorting, &windows, &units, channel, SPIKES_PER_UNIT);
            if let Some(i) = plots::to_image(plots::waveforms(&snippets, &palette, PLOT_W, PLOT_H, 1.5)) {
                images.insert(ViewKind::Waveforms, i);
            }
            let points = snippets.pca2();
            let labels: Vec<Option<u32>> = snippets.rows.iter().map(|(u, _, _)| *u).collect();
            if let Some(i) = plots::to_image(plots::scatter(&points, &labels, &palette, PLOT_W, PLOT_H, 1.5)) {
                images.insert(ViewKind::Features, i);
            }
            images
        });
        self.plots.note = note;
        self._work = Some(cx.spawn(async move |this, cx| {
            let (images, axes) = fast.await;
            let _ = this.update(cx, |w, cx| {
                if w.generation.load(Ordering::SeqCst) == generation {
                    w.plots.images.extend(images);
                    w.plots.amp_axes = axes;
                    w.notify_panels(cx);
                }
            });
            let images = slow.await;
            let _ = this.update(cx, |w, cx| {
                if w.generation.load(Ordering::SeqCst) == generation {
                    w.plots.images.extend(images);
                    w.computing = false;
                    w.notify_panels(cx);
                }
            });
        }));
    }

    fn notify_panels(&self, cx: &mut Context<Self>) {
        for p in self.panels.values() {
            p.update(cx, |_, cx| cx.notify());
        }
        cx.notify();
    }
}

impl Render for CurationWorkspace {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let (muted, border, bg) = (t.muted_foreground, t.border, t.background);
        let Some(sorting) = self.sorting(cx) else {
            return div()
                .size_full()
                .child(EmptyState::new(
                    dspk_app_core::navigation::route::Workspace::Curation.icon(),
                    "No sorting on the active view",
                    "Run a sorter in Sorting, or link a .sorting.zarr in Explore → Sorting.",
                ))
                .into_any_element();
        };
        let me = cx.weak_entity();
        let views = {
            let (me, visible) = (me.clone(), self.visible.clone());
            Button::new("curation-views").ghost().small().icon(IconName::Eye).label("Views").dropdown_menu(move |menu: PopupMenu, _, _| {
                let mut menu = menu;
                for k in ViewKind::ALL {
                    let me = me.clone();
                    menu = menu.item(PopupMenuItem::new(k.title()).checked(visible.contains(&k)).on_click(move |_, w, cx| {
                        let _ = me.update(cx, |ws, cx| ws.toggle_view(k, w, cx));
                    }));
                }
                menu
            })
        };
        let layout = {
            let (me, grid) = (me.clone(), self.grid);
            Button::new("curation-layout").ghost().small().icon(gpui_kit::assets::IconName::LayoutGrid).tooltip("Arrange the views in a grid").dropdown_menu(move |menu: PopupMenu, _, _| {
                let mut menu = menu;
                for (label, g) in [("Auto grid", Grid::Auto), ("1 column", Grid::Columns(1)), ("2 columns", Grid::Columns(2)), ("3 columns", Grid::Columns(3))] {
                    let me = me.clone();
                    menu = menu.item(PopupMenuItem::new(label).checked(grid == g).on_click(move |_, w, cx| {
                        let _ = me.update(cx, |ws, cx| ws.set_grid(g, w, cx));
                    }));
                }
                menu
            })
        };
        let palette = self.store.read(cx).palette();
        let legend = h_flex().gap_3().min_w_0().overflow_hidden().children(self.selected.iter().take(8).map(|&u| {
            h_flex().gap_1().flex_none().child(div().size(px(8.)).rounded_full().bg(color(palette.unit(u)))).child(div().text_xs().child(format!("Unit {}", sorting.units[u].info.unit_id)))
        }));
        let status = if self.computing { "Computing…".to_string() } else { self.plots.note.clone() };
        // Curation: labels, merge, undo / redo, save
        let path = self.sorting_path(cx);
        let (can_undo, can_redo, dirty, edits) = path
            .as_ref()
            .and_then(|p| state(cx).and_then(|st| st.read(cx).curations.get(p).map(|c| (!c.ops.is_empty(), !c.undone.is_empty(), c.dirty(), c.ops.len()))))
            .unwrap_or_default();
        let n = self.selected.len();
        let label_button = |label: &'static str, keys: &'static str| {
            let me = me.clone();
            Button::new(SharedString::from(format!("label-{label}")))
                .outline()
                .xsmall()
                .label(label)
                .tooltip(format!("Label the selected units {label} ({keys})"))
                .disabled(n == 0)
                .on_click(move |_, _, cx| {
                    let _ = me.update(cx, |w, cx| w.label_selected(label, cx));
                })
        };
        let act = |id: &'static str, icon: gpui_kit::assets::IconName, tip: &'static str, enabled: bool, f: fn(&mut CurationWorkspace, &mut Context<CurationWorkspace>)| {
            let me = me.clone();
            icon_button(id, Icon::new(icon), tip).small().disabled(!enabled).on_click(move |_, _, cx| {
                let _ = me.update(cx, |w, cx| f(w, cx));
            })
        };
        let keys = ["Alt+G", "Alt+M", "Alt+N", ""];
        let mut labels = h_flex().gap_1().flex_none();
        for (l, k) in LABELS.iter().zip(keys) {
            labels = labels.child(label_button(l, if k.is_empty() { "no key" } else { k }));
        }
        let edit = h_flex()
            .gap_1()
            .flex_none()
            .child(labels)
            .child({
                let me = me.clone();
                Button::new("curation-merge").outline().xsmall().icon(Icon::new(gpui_kit::assets::IconName::Merge)).label("Merge").tooltip("Merge the selected units into one (G)").disabled(n < 2).on_click(move |_, _, cx| {
                    let _ = me.update(cx, |w, cx| w.merge_selected(cx));
                })
            })
            .child(act("curation-undo", gpui_kit::assets::IconName::Undo2, "Undo (Ctrl+Z)", can_undo, |w, cx| w.undo(false, cx)))
            .child(act("curation-redo", gpui_kit::assets::IconName::Redo2, "Redo (Ctrl+Shift+Z)", can_redo, |w, cx| w.undo(true, cx)))
            .child({
                let me = me.clone();
                Button::new("curation-save")
                    .xsmall()
                    .map(|b| if dirty { b.primary() } else { b.ghost() })
                    .icon(Icon::new(gpui_kit::assets::IconName::Save))
                    .label(if dirty { format!("Save ({edits})") } else { "Saved".to_string() })
                    .tooltip("Write the curation log into the .sorting.zarr (curation.json; Ctrl+S)")
                    .disabled(!dirty)
                    .on_click(move |_, _, cx| {
                        let _ = me.update(cx, |w, cx| w.save(cx));
                    })
            });
        let toolbar = h_flex()
            .h(px(40.))
            .flex_none()
            .px_2()
            .gap_2()
            .border_b_1()
            .border_color(border)
            .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).flex_none().child(SharedString::from(sorting.summary())))
            .child(edit)
            .child(legend)
            .child(div().flex_1())
            .child(div().text_xs().text_color(muted).truncate().child(status))
            .child(views)
            .child(layout);
        v_flex()
            .id("curation-workspace")
            .track_focus(&self.focus)
            .on_key_down(cx.listener(Self::on_key))
            .on_drag_move(cx.listener(|this, e: &gpui_kit::DragMoveEvent<dspk_app_core::ui::panel_edge::PanelResize>, _, cx| {
                dspk_app_core::ui::panel_edge::follow_panel_drag(&this.store, e, cx);
                this.strip.update(cx, |_, cx| cx.notify());
            }))
            .on_mouse_down(gpui_kit::MouseButton::Left, cx.listener(|this, _, window, cx| window.focus(&this.focus, cx)))
            .size_full()
            .bg(bg)
            .child(toolbar)
            .child(div().flex_1().min_h_0().child(self.dock.clone()))
            .child(self.strip.clone())
            .when(self.show_keys, |d| d.child(keys_help(cx)))
            .into_any_element()
    }
}

/// Phy's keys, as this workspace binds them.
const KEYS: [(&str, &str); 12] = [
    ("Click / Ctrl+click", "Select a unit / add it"),
    ("↓ ↑  (J K)", "Next / previous unit of the table"),
    ("Space / Shift+Space", "Next / previous most similar unit beside the lead"),
    ("Alt+G  Alt+M  Alt+N", "Label good / mua / noise"),
    ("Alt+U", "Label unsorted"),
    ("G", "Merge the selected units"),
    ("Lasso on Amplitude", "Split the lead unit's spikes inside it"),
    ("Ctrl+Z / Ctrl+Shift+Z", "Undo / redo"),
    ("Ctrl+S", "Save the curation in the .sorting.zarr"),
    ("◀ ▶ in the strip", "The lead unit's previous / next spike"),
    ("Ctrl+wheel in the strip", "Zoom in time"),
    ("? or H", "This help"),
];

fn keys_help(cx: &App) -> impl IntoElement + use<> {
    let t = cx.theme();
    v_flex()
        .absolute()
        .top(px(48.))
        .right(px(16.))
        .w(px(420.))
        .p_3()
        .gap_1p5()
        .rounded_lg()
        .border_1()
        .border_color(t.border)
        .bg(t.popover)
        .shadow_lg()
        .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child("Curation keys (Phy)"))
        .children(KEYS.iter().map(|(k, what)| h_flex().gap_3().child(div().w(px(170.)).text_xs().font_weight(FontWeight::SEMIBOLD).child(*k)).child(div().text_xs().text_color(t.muted_foreground).child(*what))))
}

/// Whether `p` lies inside `polygon` (ray casting).
fn inside(p: (f32, f32), polygon: &[(f32, f32)]) -> bool {
    let mut on = false;
    let mut j = polygon.len() - 1;
    for i in 0..polygon.len() {
        let ((xi, yi), (xj, yj)) = (polygon[i], polygon[j]);
        if (yi > p.1) != (yj > p.1) && p.0 < (xj - xi) * (p.1 - yi) / (yj - yi) + xi {
            on = !on;
        }
        j = i;
    }
    on
}

/// One view of the Curation workspace as a dock panel.
pub struct CurationPanel {
    ws: WeakEntity<CurationWorkspace>,
    kind: ViewKind,
    focus: FocusHandle,
    /// The lasso being drawn on the amplitude plot (fractions of the plot).
    lasso: Vec<(f32, f32)>,
    plot: std::rc::Rc<std::cell::Cell<gpui_kit::Bounds<gpui_kit::Pixels>>>,
}

/// What the table draws (copied out of the workspace).
struct TableData {
    sorting: Arc<SortingData>,
    palette: Palette,
    order: Arc<Vec<usize>>,
    selected: Vec<usize>,
    sort_by: &'static str,
    descending: bool,
    labels: Vec<String>,
    label: Option<String>,
}

/// `12.3`, `—` for NaN.
fn num(v: f64, digits: usize) -> String {
    if v.is_finite() { format!("{v:.digits$}") } else { "—".into() }
}

impl CurationPanel {
    fn table(&self, d: TableData, cx: &mut Context<Self>) -> AnyElement {
        let t = cx.theme();
        let (muted, border, accent) = (t.muted_foreground, t.border, t.accent);
        let me = self.ws.clone();
        let header = h_flex().h(px(ROW_H + 4.)).px_2().gap_1().flex_none().border_b_1().border_color(border).children(COLUMNS.iter().map(|&(key, name, w, _)| {
            let me = me.clone();
            let arrow = if d.sort_by == key { if d.descending { " ↓" } else { " ↑" } } else { "" };
            div()
                .id(SharedString::from(format!("sort-{key}")))
                .w(px(w))
                .flex_none()
                .text_xs()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(muted)
                .cursor_pointer()
                .child(format!("{name}{arrow}"))
                .on_click(move |_, _, cx| {
                    let _ = me.update(cx, |ws, cx| ws.sort_by(key, cx));
                })
        }));
        let filter_menu = {
            let (me, labels, current) = (me.clone(), d.labels.clone(), d.label.clone());
            Button::new("label-filter").ghost().xsmall().label(current.clone().unwrap_or_else(|| "All labels".into())).dropdown_menu(move |menu: PopupMenu, _, _| {
                let mut menu = menu;
                for l in std::iter::once(None).chain(labels.iter().cloned().map(Some)) {
                    let me = me.clone();
                    let name = l.clone().unwrap_or_else(|| "All labels".into());
                    menu = menu.item(PopupMenuItem::new(name).checked(l == current).on_click(move |_, _, cx| {
                        let l = l.clone();
                        let _ = me.update(cx, |ws, cx| ws.set_label(l, cx));
                    }));
                }
                menu
            })
        };
        let count = format!("{} units · {} selected", d.order.len(), d.selected.len());
        let (order, selected, sorting, palette) = (d.order.clone(), d.selected.clone(), d.sorting.clone(), d.palette);
        let rows = uniform_list("unit-rows", order.len(), move |range, _, _| {
            range
                .map(|i| {
                    let u = order[i];
                    let info = &sorting.units[u].info;
                    let on = selected.contains(&u);
                    let me = me.clone();
                    let cells = [
                        info.unit_id.to_string(),
                        info.quality_label.clone(),
                        info.num_spikes.to_string(),
                        num(info.firing_rate_hz, 2),
                        num(info.snr as f64, 1),
                        num(info.isi_violation_ratio, 3),
                        num(info.presence_ratio, 2),
                        num(info.amplitude_cutoff, 3),
                        num(info.composite_score, 2),
                    ];
                    let dot: Pixel = palette.unit(u);
                    h_flex()
                        .id(("unit-row", u))
                        .h(px(ROW_H))
                        .px_2()
                        .gap_1()
                        .cursor_pointer()
                        .when(on, |el| el.bg(accent))
                        .children(cells.into_iter().zip(COLUMNS.iter()).enumerate().map(move |(c, (text, &(_, _, w, _)))| {
                            let cell = h_flex().w(px(w)).flex_none().gap_1().text_xs().overflow_hidden();
                            if c == 0 {
                                cell.child(div().size(px(8.)).flex_none().rounded_full().bg(color(dot))).child(text)
                            } else {
                                cell.child(text)
                            }
                        }))
                        .on_click(move |e, _, cx| {
                            let m = e.modifiers();
                            let add = m.control || m.platform || m.shift;
                            let _ = me.update(cx, |ws, cx| ws.pick_unit(u, add, cx));
                        })
                })
                .collect()
        })
        .flex_1()
        .min_h_0();
        v_flex()
            .size_full()
            .child(h_flex().h(px(32.)).flex_none().px_2().gap_2().child(filter_menu).child(div().flex_1()).child(div().text_xs().text_color(muted).child(count)))
            .child(header)
            .child(rows)
            .into_any_element()
    }
}

impl Render for CurationPanel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let Some(ws) = self.ws.upgrade() else { return div().into_any_element() };
        let muted = cx.theme().muted_foreground;
        if self.kind == ViewKind::Table {
            let w = ws.read(cx);
            let Some(sorting) = w.sorting(cx) else { return div().into_any_element() };
            let data = TableData {
                sorting,
                palette: w.store.read(cx).palette(),
                order: Arc::new(w.order.clone()),
                selected: w.selected.clone(),
                sort_by: w.sort_by,
                descending: w.descending,
                labels: w.table.as_ref().map(|(_, t)| t.labels.clone()).unwrap_or_default(),
                label: w.label.clone(),
            };
            return self.table(data, cx);
        }
        if self.kind == ViewKind::Similarity {
            return self.similarity(&ws, cx);
        }
        if self.kind == ViewKind::Templates {
            return div().size_full().p_1().child(ws.read(cx).templates.clone()).into_any_element();
        }
        let (image, computing, background) = {
            let w = ws.read(cx);
            (w.plots.images.get(&self.kind).cloned(), w.computing, w.store.read(cx).palette().background)
        };
        let body = match image {
            Some(i) if self.kind == ViewKind::Amplitude => self.lasso_plot(i, cx),
            Some(i) => div().size_full().child(img(i).size_full()).into_any_element(),
            None => div()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .text_xs()
                .text_color(muted)
                .child(if computing { "Computing…" } else { "Select units in the table" })
                .into_any_element(),
        };
        v_flex().size_full().bg(color(background)).child(div().flex_1().min_h_0().child(body)).into_any_element()
    }
}

impl CurationPanel {
    /// The amplitude plot with a lasso: press and drag around spikes, release to split them off
    /// the lead unit.
    fn lasso_plot(&self, image: Arc<RenderImage>, cx: &mut Context<Self>) -> AnyElement {
        let frac = |this: &Self, p: gpui_kit::Point<gpui_kit::Pixels>| {
            let b = this.plot.get();
            (((p.x - b.origin.x).as_f32() / b.size.width.as_f32().max(1.0)).clamp(0.0, 1.0), ((p.y - b.origin.y).as_f32() / b.size.height.as_f32().max(1.0)).clamp(0.0, 1.0))
        };
        let cell = self.plot.clone();
        let lasso = self.lasso.clone();
        let line = cx.theme().primary;
        div()
            .id("amplitude-lasso")
            .relative()
            .size_full()
            .cursor_crosshair()
            .on_mouse_down(gpui_kit::MouseButton::Left, cx.listener(move |this, e: &gpui_kit::MouseDownEvent, _, cx| {
                this.lasso = vec![frac(this, e.position)];
                cx.notify();
            }))
            .on_mouse_move(cx.listener(move |this, e: &gpui_kit::MouseMoveEvent, _, cx| {
                if e.pressed_button == Some(gpui_kit::MouseButton::Left) && !this.lasso.is_empty() {
                    this.lasso.push(frac(this, e.position));
                    cx.notify();
                }
            }))
            .on_mouse_up(gpui_kit::MouseButton::Left, cx.listener(|this, _, _, cx| {
                let polygon = std::mem::take(&mut this.lasso);
                if polygon.len() >= 3 {
                    let _ = this.ws.update(cx, |w, cx| w.split_lasso(&polygon, cx));
                }
                cx.notify();
            }))
            .child(img(image).size_full())
            .child(
                gpui_kit::canvas(
                    move |b, _, _| cell.set(b),
                    move |b, _, window, _| {
                        if lasso.len() < 2 {
                            return;
                        }
                        let at = |(x, y): (f32, f32)| gpui_kit::point(b.origin.x + b.size.width * x, b.origin.y + b.size.height * y);
                        let mut path = gpui_kit::PathBuilder::stroke(px(1.5));
                        path.move_to(at(lasso[0]));
                        for &p in &lasso[1..] {
                            path.line_to(at(p));
                        }
                        path.line_to(at(lasso[0]));
                        if let Ok(p) = path.build() {
                            window.paint_path(p, line);
                        }
                    },
                )
                .absolute()
                .size_full(),
            )
            .into_any_element()
    }

    /// The units most like the lead unit (template similarity): a click adds one to the
    /// selection, to compare or merge.
    fn similarity(&self, ws: &Entity<CurationWorkspace>, cx: &mut Context<Self>) -> AnyElement {
        let t = cx.theme();
        let (muted, accent) = (t.muted_foreground, t.accent);
        let w = ws.read(cx);
        let (Some(sorting), Some(&lead)) = (w.sorting(cx), w.selected.first()) else {
            return div().size_full().p_3().text_xs().text_color(muted).child("Select a unit: the units whose templates look most like it are listed here.").into_any_element();
        };
        let palette = w.store.read(cx).palette();
        let selected = w.selected.clone();
        let list = similar(&sorting, lead, SIMILAR_MAX);
        let me = self.ws.clone();
        v_flex()
            .id("similar-units")
            .size_full()
            .overflow_y_scroll()
            .p_1()
            .child(div().px_1().pb_1().text_xs().text_color(muted).child(format!("Like unit {} (template similarity); click to add to the selection", sorting.units[lead].info.unit_id)))
            .children(list.into_iter().map(|(u, score)| {
                let info = &sorting.units[u].info;
                let on = selected.contains(&u);
                let me = me.clone();
                h_flex()
                    .id(("similar", u))
                    .h(px(ROW_H))
                    .px_2()
                    .gap_2()
                    .rounded_sm()
                    .cursor_pointer()
                    .when(on, |d| d.bg(accent))
                    .child(div().size(px(8.)).flex_none().rounded_full().bg(color(palette.unit(u))))
                    .child(div().w(px(56.)).text_xs().child(format!("Unit {}", info.unit_id)))
                    .child(div().w(px(44.)).text_xs().font_weight(FontWeight::SEMIBOLD).child(format!("{score:.2}")))
                    .child(div().h(px(4.)).w(px(80. * score.clamp(0.0, 1.0))).rounded_sm().bg(color(palette.unit(u))))
                    .child(div().flex_1().text_xs().text_color(muted).truncate().child(format!("{} · {} spikes · ch {}", info.quality_label, info.num_spikes, info.primary_channel)))
                    .on_click(move |_, _, cx| {
                        let _ = me.update(cx, |ws, cx| {
                            if !ws.selected.contains(&u) {
                                ws.toggle_unit(u, cx);
                            }
                        });
                    })
            }))
            .into_any_element()
    }
}

impl EventEmitter<PanelEvent> for CurationPanel {}

impl Focusable for CurationPanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl BasePanel for CurationPanel {
    fn panel_name(&self) -> &'static str {
        "dspk-app.curation-view"
    }

    fn on_removed(&mut self, _: &mut Window, cx: &mut Context<Self>) {
        let kind = self.kind;
        let _ = self.ws.update(cx, |ws, cx| ws.closed(kind, cx));
    }
}

impl Panel for CurationPanel {
    fn title(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.kind.title()
    }
}

/// Timing of a unit click on real data (the Kilosort4 test recording and its sorting):
/// `cargo test --profile validate -p dspk-app-synapse bench_curation_click -- --ignored --nocapture`.
#[cfg(test)]
mod bench {
    use std::time::Instant;

    use dspk_view::{LocalSignal, SignalBackend};

    use super::*;

    #[test]
    #[ignore = "needs .data/kilosort4"]
    fn bench_curation_click() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.data/kilosort4");
        let rec = root.join("ZFM-02370_mini.imec0.ap.short.bin");
        let store = root.join("ZFM-02370_mini.imec0.ap.short.kilosort4.sorting.zarr");
        let t = Instant::now();
        let sorting = SortingData::load(&store).unwrap();
        println!("load sorting: {:.0} ms ({})", t.elapsed().as_secs_f64() * 1e3, sorting.summary());
        let list = dspk_synapse::io::sources(&rec).unwrap();
        let id = dspk_io::default_source(&list).unwrap().id.clone();
        let source: Arc<dyn dspk_core::RecordingSource> = Arc::from(dspk_synapse::io::open_source(&rec, &id).unwrap());
        let signal: Arc<dyn SignalBackend> = Arc::new(LocalSignal::open(source, None).unwrap());
        let t = Instant::now();
        let windows = crate::sorting::windows_of(&signal);
        println!("windows (once per signal): {:.0} ms, {} × {} samples × {} channels", t.elapsed().as_secs_f64() * 1e3, windows.starts.len(), windows.len, windows.channels);
        let palette = dspk_app_core::ui::theme::palette::Palette::of(true);
        let lead = sorting.alive().max_by_key(|&u| sorting.units[u].info.num_spikes).unwrap();
        for _ in 0..3 {
            let t = Instant::now();
            let times = sorting.unit_times(lead);
            let acg = crate::sorting::autocorrelogram(&times, sorting.sample_rate, ACG_HALF_MS, ACG_BIN_MS);
            let _ = plots::to_image(plots::bars(&acg, palette.unit(lead), None, &palette, PLOT_W, PLOT_H));
            let fast = t.elapsed();
            let ch = sorting.units[lead].info.primary_channel;
            let snippets = Snippets::read(&sorting, &windows, &[lead], ch, SPIKES_PER_UNIT);
            let _ = plots::to_image(plots::waveforms(&snippets, &palette, PLOT_W, PLOT_H, 1.5));
            let pts = snippets.pca2();
            let labels: Vec<Option<u32>> = snippets.rows.iter().map(|(u, _, _)| *u).collect();
            let _ = plots::to_image(plots::scatter(&pts, &labels, &palette, PLOT_W, PLOT_H, 1.5));
            let sim = crate::curation::similar(&sorting, lead, SIMILAR_MAX);
            let ch = sorting.units[lead].info.primary_channel;
            let tp = sorting.templates.as_ref().unwrap();
            let row = sorting.units[lead].template_row.unwrap();
            let r = tp.row(row);
            let peak = (0..tp.channels).max_by(|&a, &b| { let p = |c: usize| r[c * tp.samples..(c + 1) * tp.samples].iter().fold(0f32, |m, v| m.max(v.abs())); p(a).total_cmp(&p(b)) }).unwrap();
            let nonzero = r.iter().filter(|v| v.abs() > 1e-6).count();
            let neighbours: Vec<(u32, f32)> = sim.iter().map(|&(u, sc)| (sorting.units[u].info.primary_channel as u32, sc)).take(5).collect();
            println!("unit id {} primary ch {ch}, template peak ch {peak}, nonzero {nonzero}/{}, top: {neighbours:?}", sorting.units[lead].info.unit_id, r.len());
            println!("click: fast plots {:.1} ms, all {:.1} ms ({} snippets; best match {:?})", fast.as_secs_f64() * 1e3, t.elapsed().as_secs_f64() * 1e3, snippets.rows.len(), sim.first());
        }
    }
}
