//! The Sorting workspace (first version; reference: Offline Sorter-style layouts): the active
//! view's sorting as a unit table (metrics), the selected units' waveforms (individual spikes in
//! their unit's colour, unsorted grey, means on top), their 2-D feature space (PC1 vs PC2) and a
//! trace strip around one of their spikes with the spikes highlighted. Waveforms and features are
//! computed on demand for a subset of spikes, from the view's signal (its processed preview when
//! it has ops: what was sorted), on a background thread; nothing of it is stored.

use std::sync::Arc;

use dspk_app_core::data::recording::SpikeEventStore;
use dspk_app_core::data::store::{AppEvent, Store};
use dspk_app_core::plugin::AppPlugin as _;
use dspk_app_core::screens::explore::trace::renderer::{nominal_amplitude, RenderRequest, TimeViewKind, WaveformRenderer};
use dspk_app_core::screens::explore::trace::trace_panel::color;
use dspk_app_core::ui::widgets::EmptyState;
use dspk_view::SignalBackend;
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, img, px, uniform_list, AppContext as _, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, RenderImage, SharedString,
    StatefulInteractiveElement as _, Styled as _, Subscription, Task, Window,
};

use crate::plots;
use crate::sorting::{SortingData, Snippets};
use crate::{state, SynapseEvent, SynapsePlugin};

/// Spikes per selected unit drawn and projected (and as many unsorted ones).
pub const SPIKES_PER_UNIT: usize = 300;
/// Plot sizes (physical pixels; shown scaled to their boxes).
const PLOT_W: u32 = 720;
const PLOT_H: u32 = 420;
const STRIP_W: u32 = 1400;
const STRIP_H: u32 = 360;
/// The trace strip's window around a spike (s).
const STRIP_SEC: f64 = 0.1;

#[derive(Default, Clone)]
struct Plots {
    waveforms: Option<Arc<RenderImage>>,
    scatter: Option<Arc<RenderImage>>,
    strip: Option<Arc<RenderImage>>,
    acg: Option<Arc<RenderImage>>,
    isi: Option<Arc<RenderImage>>,
    amplitude: Option<Arc<RenderImage>>,
    templates: Option<Arc<RenderImage>>,
    note: String,
}

/// Autocorrelogram: ±50 ms in 1 ms bins; ISI histogram: 0–50 ms in 0.5 ms bins; the refractory
/// period marked at 1.5 ms.
const ACG_HALF_MS: f64 = 50.0;
const ACG_BIN_MS: f64 = 1.0;
const ISI_MAX_MS: f64 = 50.0;
const ISI_BIN_MS: f64 = 0.5;
const REFRACTORY_MS: f64 = 1.5;
/// Spikes per unit in the amplitude-over-time plot.
const AMPLITUDE_POINTS: usize = 5000;
/// Small plots (physical pixels).
const SMALL_W: u32 = 480;
const SMALL_H: u32 = 280;

pub struct SortingWorkspace {
    store: Entity<Store>,
    /// Units selected in the table (indices), the first one leads (its channel, its spike).
    selected: Vec<usize>,
    plots: Plots,
    computing: Option<Task<()>>,
    _subs: Vec<Subscription>,
}

impl SortingWorkspace {
    pub fn new(store: Entity<Store>, _window: &mut Window, cx: &mut Context<Self>) -> Self {
        let mut subs = vec![cx.subscribe(&store, |this, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::Document(_) | AppEvent::RecordingChanged | AppEvent::PaletteChanged) {
                this.selected.clear();
                this.plots = Plots::default();
                cx.notify();
            }
        })];
        if let Some(st) = state(cx) {
            subs.push(cx.subscribe(&st, |this, _, e: &SynapseEvent, cx| {
                if matches!(e, SynapseEvent::SortingLoaded(_)) {
                    this.selected.clear();
                    this.select_default(cx);
                }
                cx.notify();
            }));
        }
        let mut this = Self { store, selected: Vec::new(), plots: Plots::default(), computing: None, _subs: subs };
        this.select_default(cx);
        this
    }

    fn sorting(&self, cx: &gpui_kit::App) -> Option<Arc<SortingData>> {
        state(cx)?.read(cx).active_sorting(cx)
    }

    /// The largest unit, when nothing is selected.
    fn select_default(&mut self, cx: &mut Context<Self>) {
        let Some(s) = self.sorting(cx) else { return };
        if self.selected.is_empty() {
            if let Some(u) = (0..s.units.len()).max_by_key(|&u| s.units[u].info.num_spikes) {
                self.selected.push(u);
                self.compute(cx);
            }
        }
    }

    fn toggle(&mut self, unit: usize, cx: &mut Context<Self>) {
        match self.selected.iter().position(|&u| u == unit) {
            Some(i) => {
                self.selected.remove(i);
            }
            None => self.selected.push(unit),
        }
        self.compute(cx);
        cx.notify();
    }

    /// Waveforms, features and the trace strip of the selected units, on a background thread.
    fn compute(&mut self, cx: &mut Context<Self>) {
        let Some(sorting) = self.sorting(cx) else { return };
        let Some(&lead) = self.selected.first() else {
            self.plots = Plots::default();
            return;
        };
        let store = self.store.read(cx);
        let Some(view) = store.active_view().cloned() else { return };
        let source = store.view_source(&view);
        let Some(sources) = store.sources().cloned() else { return };
        let dataset = sources.get(&source);
        let signal: Arc<dyn SignalBackend> = store.signal_for(&source).unwrap_or_else(|| dataset.signal().clone());
        let palette = store.palette();
        // A subset view sorted on its own output: the sorting's channel i is the view's i-th
        let map = view.recording.channels.clone().filter(|c| sorting.channels == Some(c.len()));
        let to_view = move |c: usize| map.as_ref().map_or(Some(c), |m| m.get(c).copied());
        let unit = &sorting.units[lead];
        let Some(channel) = to_view(unit.info.primary_channel) else { return };
        let strip_channels: Vec<usize> = unit.extent.channels.iter().filter_map(|&c| to_view(c)).collect();
        // The strip: a spike of the lead unit from the middle of the recording
        let times = sorting.unit_times(lead);
        let strip = times.get(times.len() / 2).map(|&t| {
            let t_sec = dataset.start_time_sec + t as f64 / sorting.sample_rate;
            let window = (t_sec - STRIP_SEC / 2.0)..(t_sec + STRIP_SEC / 2.0);
            let overlay = SynapsePlugin::new().trace_overlay(store, &source, window.clone(), cx).map(Arc::new);
            let unit_of = dataset.signal().info().channels.first().map(|c| c.unit.clone()).unwrap_or(dspk_core::SignalUnit::Microvolt);
            RenderRequest {
                signal: signal.clone(),
                events: Arc::new(SpikeEventStore::default()),
                width: STRIP_W,
                height: STRIP_H,
                scale: 1.5,
                mode: TimeViewKind::Traces,
                channels: strip_channels,
                window_start_sec: window.start,
                window_sec: STRIP_SEC,
                start_time_sec: dataset.start_time_sec,
                amplitude_scale: 1.0,
                nominal: nominal_amplitude(&unit_of),
                auto_scale: true,
                remove_dc: true,
                scale_hint: 0.0,
                grid_times: Vec::new(),
                scale_bar: true,
                highlights: Vec::new(),
                overlay,
                palette,
            }
        });
        let units = self.selected.clone();
        let lead_colour = palette.unit(lead);
        let template_rows = unit.template.clone();
        let duration = (dataset.total_samples as f64 / dataset.sample_rate) as f32;
        let note = format!("Ch {channel} · {} spikes per unit at most · {}", SPIKES_PER_UNIT, if store.signal_for(&source).is_some() { "processed preview" } else { "recording as stored" });
        let work = cx.background_spawn(async move {
            let snippets = Snippets::read(&sorting, signal.as_ref(), &units, channel, SPIKES_PER_UNIT);
            let points = snippets.pca2();
            let labels: Vec<Option<u32>> = snippets.rows.iter().map(|(u, _, _)| *u).collect();
            let waveforms = plots::to_image(plots::waveforms(&snippets, &palette, PLOT_W, PLOT_H, 1.5));
            let scatter = plots::to_image(plots::scatter(&points, &labels, &palette, PLOT_W, PLOT_H, 1.5));
            let strip = strip.and_then(|req| plots::to_image(WaveformRenderer::default().render_scaled(&req).0));
            // Lead unit: correlogram, intervals, template; every selected unit: amplitudes
            let times = sorting.unit_times(lead);
            let acg = crate::sorting::autocorrelogram(&times, sorting.sample_rate, ACG_HALF_MS, ACG_BIN_MS);
            let isi = crate::sorting::isi_histogram(&times, sorting.sample_rate, ISI_MAX_MS, ISI_BIN_MS);
            let acg = plots::to_image(plots::bars(&acg, lead_colour, Some((ACG_HALF_MS / ACG_BIN_MS) as f32), &palette, SMALL_W, SMALL_H));
            let isi = plots::to_image(plots::bars(&isi, lead_colour, Some((REFRACTORY_MS / ISI_BIN_MS) as f32), &palette, SMALL_W, SMALL_H));
            let amps: Vec<(u32, Vec<(f32, f32)>)> = units.iter().map(|&u| (u as u32, sorting.unit_amplitudes(u, AMPLITUDE_POINTS))).collect();
            let amplitude = plots::to_image(plots::amplitude_time(&amps, duration, &palette, SMALL_W, SMALL_H, 1.5));
            let templates = plots::to_image(plots::template_grid(&template_rows, lead_colour, &palette, SMALL_W, SMALL_H, 1.5));
            Plots { waveforms, scatter, strip, acg, isi, amplitude, templates, note }
        });
        self.computing = Some(cx.spawn(async move |this, cx| {
            let plots = work.await;
            let _ = this.update(cx, |w, cx| {
                w.plots = plots;
                cx.notify();
            });
        }));
    }
}

/// `12.3`, `—` for NaN.
fn num(v: f64, digits: usize) -> String {
    if v.is_finite() { format!("{v:.digits$}") } else { "—".into() }
}

impl Render for SortingWorkspace {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let (muted, border, accent, bg) = (t.muted_foreground, t.border, t.accent, t.background);
        let Some(sorting) = self.sorting(cx) else {
            return div()
                .size_full()
                .child(EmptyState::new(
                    dspk_app_core::navigation::route::Workspace::Sorting.icon(),
                    "No sorting on the active view",
                    "Link one in Explore → Processing → Sorting (attach a .sorting.zarr, or Sort the view): its units, waveforms and features show here.",
                ))
                .into_any_element();
        };
        let palette = self.store.read(cx).palette();
        let mut order: Vec<usize> = (0..sorting.units.len()).collect();
        order.sort_by_key(|&u| std::cmp::Reverse(sorting.units[u].info.num_spikes));
        let order = Arc::new(order);
        let selected = self.selected.clone();
        let this = cx.entity().downgrade();
        let cols: [(&str, f32); 9] = [("Unit", 70.), ("Label", 70.), ("Spikes", 64.), ("Rate Hz", 60.), ("SNR", 48.), ("ISI viol.", 60.), ("Presence", 64.), ("Amp. cut", 64.), ("Score", 52.)];
        let header = h_flex().h(px(26.)).px_2().gap_1().border_b_1().border_color(border).children(cols.iter().map(|(n, w)| div().w(px(*w)).text_xs().font_weight(FontWeight::SEMIBOLD).text_color(muted).child(*n)));
        let rows_sorting = sorting.clone();
        let table = uniform_list("unit-table", order.len(), move |range, _, _| {
            range
                .map(|i| {
                    let u = order[i];
                    let unit = &rows_sorting.units[u].info;
                    let on = selected.contains(&u);
                    let this = this.clone();
                    let cell = |w: f32, text: String| div().w(px(w)).text_xs().child(text);
                    h_flex()
                        .id(("unit-row", u))
                        .h(px(24.))
                        .px_2()
                        .gap_1()
                        .cursor_pointer()
                        .when(on, |d| d.bg(accent))
                        .child(h_flex().w(px(70.)).gap_1().child(div().size(px(8.)).rounded_full().bg(color(palette.unit(u)))).child(div().text_xs().child(unit.unit_id.to_string())))
                        .child(cell(70., unit.quality_label.clone()))
                        .child(cell(64., unit.num_spikes.to_string()))
                        .child(cell(60., num(unit.firing_rate_hz, 2)))
                        .child(cell(48., num(unit.snr as f64, 1)))
                        .child(cell(60., num(unit.isi_violation_ratio, 3)))
                        .child(cell(64., num(unit.presence_ratio, 2)))
                        .child(cell(64., num(unit.amplitude_cutoff, 3)))
                        .child(cell(52., num(unit.composite_score, 2)))
                        .on_click(move |_, _, cx| {
                            let _ = this.update(cx, |w, cx| w.toggle(u, cx));
                        })
                })
                .collect()
        })
        .flex_1()
        .min_h_0();

        let plot = |title: &'static str, image: Option<Arc<RenderImage>>| {
            v_flex()
                .flex_1()
                .min_w_0()
                .min_h_0()
                .gap_1()
                .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(muted).child(title))
                .child(
                    div()
                        .flex_1()
                        .min_h_0()
                        .rounded_md()
                        .border_1()
                        .border_color(border)
                        .overflow_hidden()
                        .bg(color(palette.background))
                        .map(|d| match image {
                            Some(i) => d.child(img(i).size_full()),
                            None => d.child(div().size_full().flex().items_center().justify_center().text_xs().text_color(muted).child("Select units in the table")),
                        }),
                )
        };
        let legend = h_flex().gap_3().flex_wrap().children(self.selected.iter().map(|&u| {
            h_flex().gap_1().child(div().size(px(8.)).rounded_full().bg(color(palette.unit(u)))).child(div().text_xs().child(format!("Unit {}", sorting.units[u].info.unit_id)))
        }));

        h_flex()
            .id("sorting-workspace")
            .size_full()
            .bg(bg)
            .child(
                v_flex()
                    .w(px(600.))
                    .h_full()
                    .border_r_1()
                    .border_color(border)
                    .child(h_flex().h(px(30.)).px_2().gap_2().child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(SharedString::from(sorting.summary()))))
                    .child(header)
                    .child(table),
            )
            .child(
                v_flex()
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .p_2()
                    .gap_2()
                    .child(h_flex().gap_2().child(legend).child(div().flex_1()).child(div().text_xs().text_color(muted).child(self.plots.note.clone())))
                    .child(h_flex().flex_1().min_h_0().gap_2().child(plot("Waveforms (spikes, unit means)", self.plots.waveforms.clone())).child(plot("Feature space (PC1 vs PC2)", self.plots.scatter.clone())))
                    .child(
                        h_flex()
                            .h(px(190.))
                            .gap_2()
                            .child(plot("Autocorrelogram (±50 ms)", self.plots.acg.clone()))
                            .child(plot("ISI (0–50 ms; 1.5 ms marked)", self.plots.isi.clone()))
                            .child(plot("Amplitude over time", self.plots.amplitude.clone()))
                            .child(plot("Template on its channels", self.plots.templates.clone())),
                    )
                    .child(div().h(px(220.)).flex().child(plot("Trace around a spike of the first unit (100 ms)", self.plots.strip.clone()))),
            )
            .into_any_element()
    }
}
