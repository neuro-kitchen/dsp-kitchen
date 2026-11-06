//! The bottom strip of the Sorting and Curation workspaces: a simple view of the recording over
//! the app's one timeline (moving it here moves Explore too). Two tabs: Traces (the focused
//! units' channels, their spikes coloured) and Raster (the focused units' spikes, or every
//! unit's); under them an overview of the whole recording with the window as a frame and the
//! focused units' spikes as ticks. Drag the plot to move in time, Ctrl+wheel to zoom, click the
//! overview to jump; previous / next go to the lead unit's spikes.

use std::cell::Cell;
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;

use dspk_app_core::data::recording::{Dataset, SpikeEventStore};
use dspk_app_core::data::store::{duration, AppEvent, Store};
use dspk_app_core::plugin::{AppPlugin as _, Raster, RasterRow};
use dspk_app_core::screens::explore::trace::renderer::{nominal_amplitude, RenderRequest, TimeViewKind, WaveformRenderer};
use dspk_app_core::screens::explore::trace::trace_panel::color;
use dspk_app_core::ui::widgets::icon_button;
use dspk_view::SignalBackend;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, IconName, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    canvas, div, fill, img, point, px, size, App, AppContext as _, Bounds, Context, Entity, InteractiveElement as _, IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent, ParentElement as _,
    Pixels, Render, RenderImage, ScrollEvent, Styled as _, Subscription, Task, Window,
};

use crate::{plots, state, SynapseEvent, SynapsePlugin};

/// Heights (logical px): the open strip, its header, the overview.
pub const STRIP_H: f32 = 236.;
const HEADER_H: f32 = 30.;
const OVERVIEW_H: f32 = 26.;
/// At most this many channels in the trace tab.
const MAX_CHANNELS: usize = 8;
/// Window shown around a spike reached with previous / next (s), when the window is longer.
const SPIKE_WINDOW_SEC: f64 = 0.2;
/// Ticks per unit on the overview.
const TICKS_PER_UNIT: usize = 4000;
/// Oversampling of the rendered image.
const SCALE: f32 = 1.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StripTab {
    Traces,
    Raster,
}

pub struct BottomStrip {
    store: Entity<Store>,
    pub open: bool,
    tab: StripTab,
    /// Units followed (the first leads) and the channels of the trace tab.
    units: Vec<usize>,
    channels: Vec<usize>,
    image: Option<Arc<RenderImage>>,
    plot: Rc<Cell<Bounds<Pixels>>>,
    overview: Rc<Cell<Bounds<Pixels>>>,
    /// Pointer x and window start when a drag on the plot began.
    drag: Option<(f32, f64)>,
    _render: Option<Task<()>>,
    _subs: Vec<Subscription>,
}

/// A render of `channels` of `signal` from `start_sec`, `window_sec` long, `width × height` px.
#[allow(clippy::too_many_arguments)]
pub fn request(store: &Store, signal: Arc<dyn SignalBackend>, dataset: &Dataset, channels: Vec<usize>, start_sec: f64, window_sec: f64, mode: TimeViewKind, width: u32, height: u32) -> RenderRequest {
    let unit = dataset.signal().info().channels.first().map(|c| c.unit.clone()).unwrap_or(dspk_core::SignalUnit::Microvolt);
    RenderRequest {
        signal,
        events: Arc::new(SpikeEventStore::default()),
        width,
        height,
        scale: SCALE,
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
        palette: store.palette(),
    }
}

/// The active view's signal (its processed preview when it has ops), dataset and source.
pub fn view_signal(store: &Store) -> Option<(Arc<dyn SignalBackend>, Arc<Dataset>, String)> {
    let view = store.active_view()?;
    let source = store.view_source(view);
    let dataset = store.sources()?.get(&source);
    let signal = store.signal_for(&source).unwrap_or_else(|| dataset.signal().clone());
    Some((signal, dataset, source))
}

impl BottomStrip {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let mut subs = vec![cx.subscribe(&store, |this, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::WindowMoved | AppEvent::PaletteChanged | AppEvent::RecordingChanged | AppEvent::Document(_) | AppEvent::PyramidProgress(_)) {
                this.render_plot(cx);
            }
        })];
        if let Some(st) = state(cx) {
            subs.push(cx.subscribe(&st, |this, _, _: &SynapseEvent, cx| this.render_plot(cx)));
        }
        Self {
            store,
            open: true,
            tab: StripTab::Traces,
            units: Vec::new(),
            channels: Vec::new(),
            image: None,
            plot: Rc::new(Cell::new(Bounds::default())),
            overview: Rc::new(Cell::new(Bounds::default())),
            drag: None,
            _render: None,
            _subs: subs,
        }
    }

    /// Follow `units` (the first leads), drawing `channels` in the trace tab.
    pub fn set_focus(&mut self, units: Vec<usize>, channels: Vec<usize>, cx: &mut Context<Self>) {
        if units == self.units && channels == self.channels {
            return;
        }
        self.units = units;
        self.channels = channels;
        self.render_plot(cx);
    }

    fn window(&self, cx: &App) -> (f64, f64) {
        let tl = &self.store.read(cx).timeline;
        (tl.window_start_sec, tl.visible_window_sec)
    }

    /// Raster rows of the followed units (none followed: every unit, by channel) in `window`.
    fn raster_rows(&self, window: Range<f64>, cx: &App) -> Option<Raster> {
        let st = state(cx)?;
        let sorting = st.read(cx).active_sorting(cx)?;
        let s = self.store.read(cx);
        let start = view_signal(s)?.1.start_time_sec;
        let palette = s.palette();
        let mut units = self.units.clone();
        if units.is_empty() {
            units = sorting.alive().collect();
            units.sort_by_key(|&u| (sorting.units[u].info.primary_channel, u));
        }
        let rate = sorting.sample_rate;
        let (s0, s1) = (((window.start - start) * rate).max(0.0) as u64, ((window.end - start) * rate).max(0.0) as u64);
        let row: std::collections::HashMap<usize, usize> = units.iter().enumerate().map(|(r, &u)| (u, r)).collect();
        let mut rows: Vec<RasterRow> = units.iter().map(|&u| RasterRow { color: palette.unit(u), label: format!("Unit {}", sorting.units[u].info.unit_id), times: Vec::new() }).collect();
        for k in sorting.in_range(s0, s1) {
            if let Some(&r) = sorting.unit_of[k].and_then(|u| row.get(&(u as usize))) {
                rows[r].times.push(start + sorting.times[k] as f64 / rate);
            }
        }
        Some(Raster { rows })
    }

    /// Draws the plot for the current window on a background thread (a newer one replaces it).
    pub fn render_plot(&mut self, cx: &mut Context<Self>) {
        cx.notify();
        if !self.open {
            return;
        }
        let s = self.store.read(cx);
        let Some((signal, dataset, source)) = view_signal(s) else { return };
        let (start, win) = (s.timeline.window_start_sec, s.timeline.visible_window_sec);
        let b = self.plot.get().size;
        let (w, h) = (((b.width.as_f32().max(200.0)) * SCALE) as u32, ((b.height.as_f32().max(60.0)) * SCALE) as u32);
        let req = match self.tab {
            StripTab::Traces => {
                let mut channels: Vec<usize> = if self.channels.is_empty() { s.selection(&source).to_vec() } else { self.channels.clone() };
                channels.truncate(MAX_CHANNELS);
                let mut req = request(s, signal, &dataset, channels, start, win, TimeViewKind::Traces, w, h);
                req.overlay = SynapsePlugin::new().trace_overlay(s, &source, start..start + win, cx).map(Arc::new);
                req
            }
            StripTab::Raster => {
                let mut req = request(s, signal, &dataset, Vec::new(), start, win, TimeViewKind::Raster, w, h);
                req.raster = self.raster_rows(start..start + win, cx).map(Arc::new);
                req
            }
        };
        let work = cx.background_spawn(async move { plots::to_image(WaveformRenderer::default().render_scaled(&req).0) });
        self._render = Some(cx.spawn(async move |this, cx| {
            let image = work.await;
            let _ = this.update(cx, |strip, cx| {
                if image.is_some() {
                    strip.image = image;
                }
                cx.notify();
            });
        }));
    }

    fn set_tab(&mut self, tab: StripTab, cx: &mut Context<Self>) {
        self.tab = tab;
        self.render_plot(cx);
    }

    pub fn toggle_open(&mut self, cx: &mut Context<Self>) {
        self.open = !self.open;
        self.render_plot(cx);
    }

    /// The window around the lead unit's next (or previous) spike from the window's centre.
    fn step_spike(&mut self, forward: bool, cx: &mut Context<Self>) {
        let Some(&lead) = self.units.first() else { return };
        let Some(sorting) = state(cx).and_then(|st| st.read(cx).active_sorting(cx)) else { return };
        let Some(start) = view_signal(self.store.read(cx)).map(|v| v.1.start_time_sec) else { return };
        let (w0, win) = self.window(cx);
        let centre = w0 + win / 2.0;
        let times = sorting.unit_times(lead);
        let t_of = |t: u64| start + t as f64 / sorting.sample_rate;
        let eps = 1.0 / sorting.sample_rate;
        let hit = if forward { times.iter().map(|&t| t_of(t)).find(|&t| t > centre + eps) } else { times.iter().rev().map(|&t| t_of(t)).find(|&t| t < centre - eps) };
        if let Some(t) = hit {
            let win = win.min(SPIKE_WINDOW_SEC);
            self.store.update(cx, |s, cx| s.set_window(t - win / 2.0, t + win / 2.0, cx));
        }
    }

    fn on_plot_down(&mut self, e: &MouseDownEvent, _: &mut Window, cx: &mut Context<Self>) {
        self.drag = Some((e.position.x.as_f32(), self.window(cx).0));
    }

    fn on_plot_move(&mut self, e: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        match self.drag {
            Some((x, start)) if e.pressed_button == Some(MouseButton::Left) => {
                let w = self.plot.get().size.width.as_f32().max(1.0) as f64;
                let dx = (e.position.x.as_f32() - x) as f64;
                self.store.update(cx, |s, cx| {
                    let span = s.timeline.visible_window_sec;
                    let start = start - dx / w * span;
                    s.set_window(start, start + span, cx);
                });
            }
            Some(_) => self.drag = None,
            None => {}
        }
    }

    fn on_plot_wheel(&mut self, e: &ScrollEvent, _: &mut Window, cx: &mut Context<Self>) {
        let d = e.delta.pixel_delta(px(20.));
        let (dx, dy) = (d.x.as_f32() as f64, d.y.as_f32() as f64);
        let b = self.plot.get();
        let fx = ((e.position.x - b.origin.x).as_f32() / b.size.width.as_f32().max(1.0)).clamp(0.0, 1.0) as f64;
        if e.modifiers.control || e.modifiers.platform {
            self.store.update(cx, |s, cx| s.zoom_at((dy * 0.004).exp(), fx, cx));
        } else {
            let moved = if dx != 0.0 { dx } else { dy };
            let w = b.size.width.as_f32().max(1.0) as f64;
            self.store.update(cx, |s, cx| {
                let dt = -moved / w * s.timeline.visible_window_sec;
                s.pan_time(dt, cx);
            });
        }
        cx.stop_propagation();
    }

    /// Jumps the window's centre to the overview position under the pointer.
    fn on_overview(&mut self, x: Pixels, cx: &mut Context<Self>) {
        let b = self.overview.get();
        let f = ((x - b.origin.x).as_f32() / b.size.width.as_f32().max(1.0)).clamp(0.0, 1.0) as f64;
        let s = self.store.read(cx);
        let Some((_, ds, _)) = view_signal(s) else { return };
        let total = ds.total_samples as f64 / ds.sample_rate;
        let t = ds.start_time_sec + f * total;
        let win = s.timeline.visible_window_sec;
        self.store.update(cx, |s, cx| s.set_window(t - win / 2.0, t + win / 2.0, cx));
    }

    fn overview(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let s = self.store.read(cx);
        let t = cx.theme();
        let (bar, frame) = (t.muted, t.primary);
        let ds = view_signal(s).map(|v| v.1);
        let total = ds.as_ref().map_or(1.0, |d| d.total_samples as f64 / d.sample_rate).max(1e-9);
        let start = ds.as_ref().map_or(0.0, |d| d.start_time_sec);
        let (w0, win) = (s.timeline.window_start_sec, s.timeline.visible_window_sec);
        let span = (((w0 - start) / total) as f32, ((w0 + win - start) / total) as f32);
        // Ticks of the followed units: (fraction, colour)
        let mut ticks: Vec<(f32, gpui_kit::Hsla)> = Vec::new();
        if let Some(sorting) = state(cx).and_then(|st| st.read(cx).active_sorting(cx)) {
            let palette = s.palette();
            for &u in self.units.iter().take(8) {
                let times = sorting.unit_times(u);
                let step = (times.len() / TICKS_PER_UNIT).max(1);
                let c = color(palette.unit(u));
                ticks.extend(times.iter().step_by(step).map(|&t| ((t as f64 / sorting.sample_rate / total) as f32, c)));
            }
        }
        let cell = self.overview.clone();
        let me = cx.entity().downgrade();
        let me2 = me.clone();
        div()
            .id("strip-overview")
            .h(px(OVERVIEW_H))
            .flex_none()
            .mx_2()
            .mb_1()
            .cursor_pointer()
            .on_mouse_down(MouseButton::Left, move |e, _, cx| {
                let _ = me.update(cx, |s, cx| s.on_overview(e.position.x, cx));
            })
            .on_mouse_move(move |e, _, cx| {
                if e.pressed_button == Some(MouseButton::Left) {
                    let _ = me2.update(cx, |s, cx| s.on_overview(e.position.x, cx));
                }
            })
            .child(
                canvas(
                    move |b, _, _| cell.set(b),
                    move |b, _, window, _| {
                        let (x0, y0, w, h) = (b.origin.x, b.origin.y, b.size.width.as_f32(), b.size.height.as_f32());
                        window.paint_quad(fill(b, bar.opacity(0.5)));
                        for (f, c) in &ticks {
                            window.paint_quad(fill(Bounds { origin: point(x0 + px(f * w), y0 + px(3.0)), size: size(px(1.0), px(h - 6.0)) }, c.opacity(0.8)));
                        }
                        let (a, z) = (span.0.clamp(0.0, 1.0) * w, span.1.clamp(0.0, 1.0) * w);
                        let r = Bounds { origin: point(x0 + px(a), y0), size: size(px((z - a).max(2.0)), px(h)) };
                        window.paint_quad(fill(r, frame.opacity(0.18)));
                        window.paint_quad(gpui_kit::outline(r, frame, gpui_kit::BorderStyle::Solid));
                    },
                )
                .size_full(),
            )
    }
}

impl Render for BottomStrip {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let (muted, border, bg) = (t.muted_foreground, t.border, t.background);
        let (w0, win) = self.window(cx);
        let me = cx.entity().downgrade();
        let tab = |id: &'static str, label: &'static str, which: StripTab| {
            let me = me.clone();
            Button::new(id).ghost().xsmall().label(label).selected(self.tab == which).on_click(move |_, _, cx| {
                let _ = me.update(cx, |s, cx| s.set_tab(which, cx));
            })
        };
        let act = |id: &'static str, icon: IconName, tip: &'static str, f: fn(&mut BottomStrip, &mut Context<BottomStrip>)| {
            let me = me.clone();
            icon_button(id, icon, tip).xsmall().on_click(move |_, _, cx| {
                let _ = me.update(cx, |s, cx| f(s, cx));
            })
        };
        let header = h_flex()
            .h(px(HEADER_H))
            .flex_none()
            .px_2()
            .gap_1()
            .border_b_1()
            .border_color(border)
            .child(tab("strip-traces", "Traces", StripTab::Traces))
            .child(tab("strip-raster", "Raster", StripTab::Raster))
            .child(div().w(px(1.)).h(px(16.)).bg(border).mx_1())
            .child(act("strip-prev", IconName::ChevronLeft, "Previous spike of the lead unit", |s, cx| s.step_spike(false, cx)))
            .child(act("strip-next", IconName::ChevronRight, "Next spike of the lead unit", |s, cx| s.step_spike(true, cx)))
            .child(act("strip-zoom-out", IconName::Minus, "Zoom out", |s, cx| s.store.update(cx, |st, cx| st.zoom_at(1.5, 0.5, cx))))
            .child(act("strip-zoom-in", IconName::Plus, "Zoom in", |s, cx| s.store.update(cx, |st, cx| st.zoom_at(1.0 / 1.5, 0.5, cx))))
            .child(div().text_xs().text_color(muted).child(format!("{:.3} s · {} window", w0, duration(win))))
            .child(div().flex_1())
            .when(self.tab == StripTab::Traces, |d| {
                d.child(div().text_xs().text_color(muted).child(if self.channels.is_empty() { "Shown channels".to_string() } else { format!("Lead unit's {} channels", self.channels.len().min(MAX_CHANNELS)) }))
            })
            .child(act(
                "strip-toggle",
                if self.open { IconName::ChevronDown } else { IconName::ChevronUp },
                if self.open { "Minimise" } else { "Show the strip" },
                |s, cx| s.toggle_open(cx),
            ));
        if !self.open {
            return v_flex().id("bottom-strip").w_full().h(px(HEADER_H)).flex_none().bg(bg).border_t_1().border_color(border).child(header);
        }
        let plot_cell = self.plot.clone();
        let image = self.image.clone();
        let plot = div()
            .id("strip-plot")
            .relative()
            .flex_1()
            .min_h_0()
            .mx_2()
            .my_1()
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_plot_down))
            .on_mouse_move(cx.listener(Self::on_plot_move))
            .on_scroll(cx.listener(Self::on_plot_wheel))
            .child(canvas(move |b, _, _| plot_cell.set(b), |_, _, _, _| {}).absolute().size_full())
            .children(image.map(|i| img(i).size_full()));
        v_flex()
            .id("bottom-strip")
            .w_full()
            .h(px(STRIP_H))
            .flex_none()
            .bg(bg)
            .border_t_1()
            .border_color(border)
            .child(header)
            .child(plot)
            .child(self.overview(cx))
    }
}
