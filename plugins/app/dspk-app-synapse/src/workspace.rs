//! The Sorting workspace: the active view's sorting in views the user arranges. Every view is a
//! dock panel (drag a tab to snap it beside, under or into another; close it), the Views menu
//! toggles each one, the Layout menu applies a grid (auto, or 1–3 columns, the table on the
//! left). Views: the unit table (a Polars frame: sort by any column, filter by label), the
//! selected units' waveforms (spikes, unsorted grey, means), their features (PC1 vs PC2),
//! autocorrelogram, ISI, amplitude over time, the lead unit's template on its channels, a
//! raster of the selected units over the recording, and the trace around one of the lead unit's
//! spikes (previous / next). Waveforms and features are computed on demand from the view's
//! signal (its processed preview when it has ops: what was sorted), on a background thread;
//! nothing of it is stored.

use std::collections::HashMap;
use std::sync::Arc;

use dspk_app_core::data::recording::{Dataset, SpikeEventStore};
use dspk_app_core::data::store::{AppEvent, Store};
use dspk_app_core::plugin::{AppPlugin as _, Raster, RasterRow};
use dspk_app_core::screens::explore::trace::renderer::{nominal_amplitude, RenderRequest, TimeViewKind, WaveformRenderer};
use dspk_app_core::screens::explore::trace::trace_panel::color;
use dspk_app_core::ui::plot::canvas::Pixel;
use dspk_app_core::ui::theme::palette::Palette;
use dspk_app_core::ui::widgets::{icon_button, EmptyState};
use dspk_view::SignalBackend;
use gpui_kit::base::dock::{DockArea, DockLayout, Panel as BasePanel, PanelEvent};
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::dock::{panel_handle, DockSkin, Panel};
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenu, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, img, px, uniform_list, AnyElement, App, AppContext as _, Context, Entity, EventEmitter, FocusHandle, Focusable, FontWeight, InteractiveElement as _, IntoElement,
    ParentElement as _, Render, RenderImage, SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, Task, WeakEntity, Window,
};

use crate::plots;
use crate::sorting::{Snippets, SortingData};
use crate::table::{UnitTable, COLUMNS};
use crate::{state, SynapseEvent, SynapsePlugin};

/// Spikes per selected unit drawn and projected (and as many unsorted ones).
pub const SPIKES_PER_UNIT: usize = 300;
/// Plot sizes (physical pixels; shown scaled to their panels).
const PLOT_W: u32 = 720;
const PLOT_H: u32 = 420;
const WIDE_W: u32 = 1400;
const WIDE_H: u32 = 320;
/// The trace strip's window around a spike (s).
const STRIP_SEC: f64 = 0.1;
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
    Raster,
    Trace,
}

impl ViewKind {
    pub const ALL: [ViewKind; 9] = [Self::Table, Self::Waveforms, Self::Features, Self::Acg, Self::Isi, Self::Amplitude, Self::Template, Self::Raster, Self::Trace];

    pub fn title(self) -> &'static str {
        match self {
            Self::Table => "Units",
            Self::Waveforms => "Waveforms",
            Self::Features => "Features (PC1 · PC2)",
            Self::Acg => "Autocorrelogram",
            Self::Isi => "ISI",
            Self::Amplitude => "Amplitude over time",
            Self::Template => "Template",
            Self::Raster => "Raster",
            Self::Trace => "Trace",
        }
    }

    /// Plots as wide as the grid (time along x).
    fn wide(self) -> bool {
        matches!(self, Self::Raster | Self::Trace)
    }
}

/// How the views are arranged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grid {
    /// Columns = √(plots), time plots full width below.
    Auto,
    Columns(usize),
}

#[derive(Default, Clone)]
struct Plots {
    images: HashMap<ViewKind, Arc<RenderImage>>,
    note: String,
}

pub struct SortingWorkspace {
    store: Entity<Store>,
    dock: Entity<DockArea>,
    panels: HashMap<ViewKind, Entity<SortingPanel>>,
    /// Views shown, in layout order.
    visible: Vec<ViewKind>,
    grid: Grid,
    /// Units selected in the table (indices), the first one leads (its channel, its spikes).
    selected: Vec<usize>,
    /// The table of the loaded sorting, its order (sorted, filtered), and how.
    table: Option<(std::path::PathBuf, Arc<UnitTable>)>,
    order: Vec<usize>,
    sort_by: &'static str,
    descending: bool,
    label: Option<String>,
    /// Which spike of the lead unit the trace shows.
    strip_spike: usize,
    plots: Plots,
    computing: bool,
    _work: Option<Task<()>>,
    _strip: Option<Task<()>>,
    _subs: Vec<Subscription>,
}

impl SortingWorkspace {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let dock = cx.new(|cx| {
            let skin = DockSkin::new(cx);
            DockArea::new("sorting", Some(1), window, cx).with_renderer(skin)
        });
        let mut subs = vec![cx.subscribe(&store, |this, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::Document(_) | AppEvent::RecordingChanged | AppEvent::PaletteChanged) {
                this.selected.clear();
                this.plots = Plots::default();
                this.refresh(cx);
            }
        })];
        if let Some(st) = state(cx) {
            subs.push(cx.subscribe(&st, |this, _, e: &SynapseEvent, cx| {
                if matches!(e, SynapseEvent::SortingLoaded(_)) {
                    this.selected.clear();
                    this.refresh(cx);
                }
            }));
        }
        let visible = vec![ViewKind::Table, ViewKind::Waveforms, ViewKind::Features, ViewKind::Acg, ViewKind::Isi, ViewKind::Amplitude, ViewKind::Template, ViewKind::Raster];
        let mut this = Self {
            store,
            dock,
            panels: HashMap::new(),
            visible,
            grid: Grid::Auto,
            selected: Vec::new(),
            table: None,
            order: Vec::new(),
            sort_by: "spikes",
            descending: true,
            label: None,
            strip_spike: 0,
            plots: Plots::default(),
            computing: false,
            _work: None,
            _strip: None,
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
        if self.table.as_ref().is_none_or(|(p, _)| *p != sorting.path) {
            match UnitTable::new(&sorting) {
                Ok(t) => self.table = Some((sorting.path.clone(), Arc::new(t))),
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
        self.strip_spike = 0;
        self.compute(cx);
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

    fn panel(&mut self, kind: ViewKind, cx: &mut Context<Self>) -> Entity<SortingPanel> {
        let ws = cx.weak_entity();
        self.panels.entry(kind).or_insert_with(|| cx.new(|cx| SortingPanel { ws, kind, focus: cx.focus_handle() })).clone()
    }

    /// The visible views in the grid: the table on the left, plots in rows of `columns`, time
    /// plots (raster, trace) full width under them.
    fn apply_layout(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let plot_kinds: Vec<ViewKind> = self.visible.iter().copied().filter(|k| *k != ViewKind::Table && !k.wide()).collect();
        let wide: Vec<ViewKind> = self.visible.iter().copied().filter(|k| k.wide()).collect();
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
        for &k in &wide {
            let p = self.panel(k, cx);
            grid = grid.child(DockLayout::tabs().panel_view(panel_handle(p), cx), None);
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

    /// The active view's signal (its processed preview when it has ops), dataset and source.
    fn signal(&self, cx: &App) -> Option<(Arc<dyn SignalBackend>, Arc<Dataset>, String)> {
        let store = self.store.read(cx);
        let view = store.active_view()?;
        let source = store.view_source(view);
        let dataset = store.sources()?.get(&source);
        let signal = store.signal_for(&source).unwrap_or_else(|| dataset.signal().clone());
        Some((signal, dataset, source))
    }

    /// A subset view sorted on its own output: the sorting's channel i is the view's i-th.
    fn channel_map(&self, sorting: &SortingData, cx: &App) -> impl Fn(usize) -> Option<usize> + Send + 'static {
        let map = self.store.read(cx).active_view().and_then(|v| v.recording.channels.clone()).filter(|c| sorting.channels == Some(c.len()));
        move |c: usize| map.as_ref().map_or(Some(c), |m| m.get(c).copied())
    }

    /// A request over `channels` from `start_sec`, `window_sec` long.
    fn request(&self, signal: Arc<dyn SignalBackend>, dataset: &Dataset, channels: Vec<usize>, start_sec: f64, window_sec: f64, mode: TimeViewKind, cx: &App) -> RenderRequest {
        let unit = dataset.signal().info().channels.first().map(|c| c.unit.clone()).unwrap_or(dspk_core::SignalUnit::Microvolt);
        RenderRequest {
            signal,
            events: Arc::new(SpikeEventStore::default()),
            width: WIDE_W,
            height: WIDE_H,
            scale: 1.5,
            mode,
            channels,
            window_start_sec: start_sec,
            window_sec,
            start_time_sec: dataset.start_time_sec,
            amplitude_scale: 1.0,
            nominal: nominal_amplitude(&unit),
            auto_scale: true,
            remove_dc: true,
            scale_hint: 0.0,
            grid_times: Vec::new(),
            scale_bar: mode == TimeViewKind::Traces,
            highlights: Vec::new(),
            overlay: None,
            raster: None,
            color_by: Default::default(),
            palette: self.store.read(cx).palette(),
        }
    }

    /// Every plot of the selected units, on a background thread (a newer selection replaces it).
    fn compute(&mut self, cx: &mut Context<Self>) {
        let Some(sorting) = self.sorting(cx) else { return };
        let Some(&lead) = self.selected.first() else {
            self.plots = Plots::default();
            self.notify_panels(cx);
            return;
        };
        let Some((signal, dataset, source)) = self.signal(cx) else { return };
        let to_view = self.channel_map(&sorting, cx);
        let palette = self.store.read(cx).palette();
        let Some(channel) = to_view(sorting.units[lead].info.primary_channel) else { return };
        let units = self.selected.clone();
        let lead_colour = palette.unit(lead);
        let template_rows = sorting.units[lead].template.clone();
        let duration = (dataset.total_samples as f64 / dataset.sample_rate) as f32;
        let start = dataset.start_time_sec;
        // Raster of the selected units over the whole recording
        let rows = units
            .iter()
            .map(|&u| RasterRow { color: palette.unit(u), label: String::new(), times: sorting.unit_times(u).iter().map(|&t| start + t as f64 / sorting.sample_rate).collect() })
            .collect();
        let mut raster = self.request(signal.clone(), &dataset, Vec::new(), start, duration as f64, TimeViewKind::Raster, cx);
        raster.raster = Some(Arc::new(Raster { rows }));
        let processed = self.store.read(cx).signal_for(&source).is_some();
        let note = format!("Ch {channel} · ≤{SPIKES_PER_UNIT} spikes per unit · {}", if processed { "processed" } else { "as stored" });
        let wanted: Vec<ViewKind> = self.visible.clone();
        self.computing = true;
        self.notify_panels(cx);
        let work = cx.background_spawn(async move {
            let mut images = HashMap::new();
            let mut put = |k: ViewKind, f: Option<Arc<RenderImage>>| {
                if let Some(i) = f {
                    images.insert(k, i);
                }
            };
            if wanted.contains(&ViewKind::Waveforms) || wanted.contains(&ViewKind::Features) {
                let snippets = Snippets::read(&sorting, signal.as_ref(), &units, channel, SPIKES_PER_UNIT);
                put(ViewKind::Waveforms, plots::to_image(plots::waveforms(&snippets, &palette, PLOT_W, PLOT_H, 1.5)));
                let points = snippets.pca2();
                let labels: Vec<Option<u32>> = snippets.rows.iter().map(|(u, _, _)| *u).collect();
                put(ViewKind::Features, plots::to_image(plots::scatter(&points, &labels, &palette, PLOT_W, PLOT_H, 1.5)));
            }
            let times = sorting.unit_times(lead);
            let acg = crate::sorting::autocorrelogram(&times, sorting.sample_rate, ACG_HALF_MS, ACG_BIN_MS);
            let isi = crate::sorting::isi_histogram(&times, sorting.sample_rate, ISI_MAX_MS, ISI_BIN_MS);
            put(ViewKind::Acg, plots::to_image(plots::bars(&acg, lead_colour, Some((ACG_HALF_MS / ACG_BIN_MS) as f32), &palette, PLOT_W, PLOT_H)));
            put(ViewKind::Isi, plots::to_image(plots::bars(&isi, lead_colour, Some((REFRACTORY_MS / ISI_BIN_MS) as f32), &palette, PLOT_W, PLOT_H)));
            let amps: Vec<(u32, Vec<(f32, f32)>)> = units.iter().map(|&u| (u as u32, sorting.unit_amplitudes(u, AMPLITUDE_POINTS))).collect();
            put(ViewKind::Amplitude, plots::to_image(plots::amplitude_time(&amps, duration, &palette, PLOT_W, PLOT_H, 1.5)));
            put(ViewKind::Template, plots::to_image(plots::template_grid(&template_rows, lead_colour, &palette, PLOT_W, PLOT_H, 1.5)));
            put(ViewKind::Raster, plots::to_image(WaveformRenderer::default().render_scaled(&raster).0));
            Plots { images, note }
        });
        self._work = Some(cx.spawn(async move |this, cx| {
            let plots = work.await;
            let _ = this.update(cx, |w, cx| {
                // The trace keeps its own image
                let strip = w.plots.images.remove(&ViewKind::Trace);
                w.plots = plots;
                if let Some(s) = strip {
                    w.plots.images.insert(ViewKind::Trace, s);
                }
                w.computing = false;
                w.compute_strip(cx);
                w.notify_panels(cx);
            });
        }));
    }

    /// The trace around spike `strip_spike` of the lead unit, on the unit's channels.
    fn compute_strip(&mut self, cx: &mut Context<Self>) {
        if !self.visible.contains(&ViewKind::Trace) {
            return;
        }
        let (Some(sorting), Some(&lead)) = (self.sorting(cx), self.selected.first()) else { return };
        let Some((signal, dataset, source)) = self.signal(cx) else { return };
        let times = sorting.unit_times(lead);
        if times.is_empty() {
            return;
        }
        self.strip_spike = self.strip_spike.min(times.len() - 1);
        let to_view = self.channel_map(&sorting, cx);
        let channels: Vec<usize> = sorting.units[lead].extent.channels.iter().filter_map(|&c| to_view(c)).collect();
        let t = dataset.start_time_sec + times[self.strip_spike] as f64 / sorting.sample_rate;
        let window = (t - STRIP_SEC / 2.0)..(t + STRIP_SEC / 2.0);
        let mut req = self.request(signal, &dataset, channels, window.start, STRIP_SEC, TimeViewKind::Traces, cx);
        req.overlay = SynapsePlugin::new().trace_overlay(self.store.read(cx), &source, window, cx).map(Arc::new);
        let work = cx.background_spawn(async move { plots::to_image(WaveformRenderer::default().render_scaled(&req).0) });
        self._strip = Some(cx.spawn(async move |this, cx| {
            let image = work.await;
            let _ = this.update(cx, |w, cx| {
                if let Some(i) = image {
                    w.plots.images.insert(ViewKind::Trace, i);
                }
                w.notify_panels(cx);
            });
        }));
    }

    fn step_spike(&mut self, forward: bool, cx: &mut Context<Self>) {
        self.strip_spike = if forward { self.strip_spike + 1 } else { self.strip_spike.saturating_sub(1) };
        self.compute_strip(cx);
        self.notify_panels(cx);
    }

    /// `Unit 88 · spike 12 of 4171 · 1.234 s`
    fn strip_label(&self, cx: &App) -> String {
        let (Some(sorting), Some(&lead)) = (self.sorting(cx), self.selected.first()) else { return String::new() };
        let times = sorting.unit_times(lead);
        let k = self.strip_spike.min(times.len().saturating_sub(1));
        let t = times.get(k).map_or(0.0, |&t| t as f64 / sorting.sample_rate);
        format!("Unit {} · spike {} of {} · {t:.3} s · 100 ms on its channels", sorting.units[lead].info.unit_id, k + 1, times.len())
    }

    fn notify_panels(&self, cx: &mut Context<Self>) {
        for p in self.panels.values() {
            p.update(cx, |_, cx| cx.notify());
        }
        cx.notify();
    }
}

impl Render for SortingWorkspace {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let (muted, border, bg) = (t.muted_foreground, t.border, t.background);
        let Some(sorting) = self.sorting(cx) else {
            return div()
                .size_full()
                .child(EmptyState::new(
                    dspk_app_core::navigation::route::Workspace::Sorting.icon(),
                    "No sorting on the active view",
                    "Link one in Explore → Sorting (attach a .sorting.zarr, or Sort the view).",
                ))
                .into_any_element();
        };
        let me = cx.weak_entity();
        let views = {
            let (me, visible) = (me.clone(), self.visible.clone());
            Button::new("sorting-views").ghost().small().icon(IconName::Eye).label("Views").dropdown_menu(move |menu: PopupMenu, _, _| {
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
            Button::new("sorting-layout").ghost().small().icon(gpui_kit::assets::IconName::LayoutGrid).tooltip("Arrange the views in a grid").dropdown_menu(move |menu: PopupMenu, _, _| {
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
        let toolbar = h_flex()
            .h(px(36.))
            .flex_none()
            .px_2()
            .gap_2()
            .border_b_1()
            .border_color(border)
            .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).flex_none().child(SharedString::from(sorting.summary())))
            .child(legend)
            .child(div().flex_1())
            .child(div().text_xs().text_color(muted).truncate().child(status))
            .child(views)
            .child(layout);
        v_flex().id("sorting-workspace").size_full().bg(bg).child(toolbar).child(div().flex_1().min_h_0().child(self.dock.clone())).into_any_element()
    }
}

/// One view of the Sorting workspace as a dock panel.
pub struct SortingPanel {
    ws: WeakEntity<SortingWorkspace>,
    kind: ViewKind,
    focus: FocusHandle,
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

impl SortingPanel {
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
                        .on_click(move |_, _, cx| {
                            let _ = me.update(cx, |ws, cx| ws.toggle_unit(u, cx));
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

impl Render for SortingPanel {
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
        let (image, computing, background, label) = {
            let w = ws.read(cx);
            (w.plots.images.get(&self.kind).cloned(), w.computing, w.store.read(cx).palette().background, (self.kind == ViewKind::Trace).then(|| w.strip_label(cx)))
        };
        let body = match image {
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
        let controls = label.map(|label| {
            let (me, me2) = (self.ws.clone(), self.ws.clone());
            h_flex()
                .h(px(28.))
                .flex_none()
                .px_2()
                .gap_1()
                .child(icon_button("spike-prev", IconName::ChevronLeft, "Previous spike").xsmall().on_click(move |_, _, cx| {
                    let _ = me.update(cx, |w, cx| w.step_spike(false, cx));
                }))
                .child(icon_button("spike-next", IconName::ChevronRight, "Next spike").xsmall().on_click(move |_, _, cx| {
                    let _ = me2.update(cx, |w, cx| w.step_spike(true, cx));
                }))
                .child(div().text_xs().text_color(muted).truncate().child(label))
        });
        v_flex().size_full().bg(color(background)).children(controls).child(div().flex_1().min_h_0().child(body)).into_any_element()
    }
}

impl EventEmitter<PanelEvent> for SortingPanel {}

impl Focusable for SortingPanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl BasePanel for SortingPanel {
    fn panel_name(&self) -> &'static str {
        "dspk-app.sorting-view"
    }

    fn on_removed(&mut self, _: &mut Window, cx: &mut Context<Self>) {
        let kind = self.kind;
        let _ = self.ws.update(cx, |ws, cx| ws.closed(kind, cx));
    }
}

impl Panel for SortingPanel {
    fn title(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.kind.title()
    }
}
