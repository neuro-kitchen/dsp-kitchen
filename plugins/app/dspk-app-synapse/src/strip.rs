//! The bottom strip of the Sorting and Curation workspaces: Explore's own trace view (the same
//! renderer, gain, channel scrolling, drag and Ctrl+wheel zoom, raster) over the app's one
//! timeline, pinned to the channels of the units the workspace is focused on, with only those
//! units' spikes coloured; under it an overview of the whole recording (the window as a frame,
//! the focused units' spikes as ticks; click to jump). Previous / next go to the lead unit's
//! spikes. It renders only while its workspace is in front.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use dspk_app_core::data::recording::Dataset;
use dspk_app_core::data::store::{duration, AppEvent, Store};
use dspk_app_core::navigation::route::Workspace;
use dspk_app_core::screens::explore::trace::renderer::TimeViewKind;
use dspk_app_core::screens::explore::trace::trace_panel::{color, TracePanel};
use dspk_app_core::screens::explore::TraceVm;
use dspk_app_core::ui::panel_edge::{panel_edge, Side};
use dspk_app_core::ui::theme::{PANE_GAP, RADIUS_LG};
use dspk_app_core::ui::widgets::icon_button;
use dspk_view::SignalBackend;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, IconName, Selectable as _, Sizable as _};
use gpui_kit::{
    canvas, div, fill, point, px, size, AppContext as _, Bounds, Context, Entity, InteractiveElement as _, IntoElement, MouseButton, ParentElement as _, Pixels, Render,
    Styled as _, Subscription, WeakEntity, Window,
};

use crate::state;

/// Heights (logical px): the open strip, its header, the overview.
pub const STRIP_H: f32 = 260.;
const HEADER_H: f32 = 32.;
const OVERVIEW_H: f32 = 24.;
/// At most this many channels pinned to the strip.
const MAX_CHANNELS: usize = 12;
/// Window shown around a spike reached with previous / next (s), when the window is longer.
const SPIKE_WINDOW_SEC: f64 = 0.2;
/// Ticks per unit on the overview.
const TICKS_PER_UNIT: usize = 4000;
/// Trace view ids of the strips (Explore numbers its views from 1).
const STRIP_VIEW_ID: u64 = 1 << 40;

pub struct BottomStrip {
    store: Entity<Store>,
    workspace: Workspace,
    vm: Entity<TraceVm>,
    panel: Entity<TracePanel>,
    pub open: bool,
    raster: bool,
    /// Units followed (the first leads) and their channels.
    units: Vec<usize>,
    channels: Vec<usize>,
    overview: Rc<Cell<Bounds<Pixels>>>,
    _subs: Vec<Subscription>,
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
    pub fn new(store: Entity<Store>, workspace: Workspace, cx: &mut Context<Self>) -> Self {
        let source = store.read(cx).sources().map(|s| s.default_entry().id.clone()).unwrap_or_default();
        let id = STRIP_VIEW_ID + workspace as u64;
        let vm = cx.new(|cx| {
            let mut vm = TraceVm::new(store.clone(), id, TimeViewKind::Traces, &source, cx);
            vm.set_home(workspace, cx);
            vm.set_spikes(true, cx);
            vm
        });
        let panel = cx.new(|cx| TracePanel::new(vm.clone(), WeakEntity::new_invalid(), cx));
        let subs = vec![cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::WindowMoved | AppEvent::RecordingChanged | AppEvent::WorkspaceChanged) {
                cx.notify();
            }
        })];
        Self { store, workspace, vm, panel, open: true, raster: false, units: Vec::new(), channels: Vec::new(), overview: Rc::new(Cell::new(Bounds::default())), _subs: subs }
    }

    /// Follow `units` (the first leads): their channels pinned, only their spikes coloured.
    pub fn set_focus(&mut self, units: Vec<usize>, channels: Vec<usize>, cx: &mut Context<Self>) {
        if units == self.units && channels == self.channels {
            return;
        }
        self.units = units.clone();
        self.channels = channels.clone();
        if let Some(st) = state(cx) {
            let ws = self.workspace;
            st.update(cx, |s, cx| s.set_focus(ws, units, cx));
        }
        let mut pinned = channels;
        pinned.truncate(MAX_CHANNELS);
        self.vm.update(cx, |vm, cx| {
            vm.set_pinned_selection(true, cx);
            if !pinned.is_empty() {
                vm.set_selection(pinned, cx);
            }
        });
        cx.notify();
    }

    fn set_raster(&mut self, raster: bool, cx: &mut Context<Self>) {
        self.raster = raster;
        self.vm.update(cx, |vm, cx| vm.set_kind(if raster { TimeViewKind::Raster } else { TimeViewKind::Traces }, cx));
        cx.notify();
    }

    pub fn toggle_open(&mut self, cx: &mut Context<Self>) {
        self.open = !self.open;
        cx.notify();
    }

    /// The window around the lead unit's next (or previous) spike from the window's centre.
    pub fn step_spike(&mut self, forward: bool, cx: &mut Context<Self>) {
        let Some(&lead) = self.units.first() else { return };
        let Some(sorting) = state(cx).and_then(|st| st.read(cx).active_sorting(cx)) else { return };
        let Some(start) = view_signal(self.store.read(cx)).map(|v| v.1.start_time_sec) else { return };
        let (w0, win) = {
            let tl = &self.store.read(cx).timeline;
            (tl.window_start_sec, tl.visible_window_sec)
        };
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
        let (me, me2) = (cx.entity().downgrade(), cx.entity().downgrade());
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
        let (w0, win) = {
            let tl = &self.store.read(cx).timeline;
            (tl.window_start_sec, tl.visible_window_sec)
        };
        let me = cx.entity().downgrade();
        let tab = |id: &'static str, label: &'static str, raster: bool| {
            let me = me.clone();
            Button::new(id).ghost().small().label(label).selected(self.raster == raster).on_click(move |_, _, cx| {
                let _ = me.update(cx, |s, cx| s.set_raster(raster, cx));
            })
        };
        let act = |id: &'static str, icon: IconName, tip: &'static str, f: fn(&mut BottomStrip, &mut Context<BottomStrip>)| {
            let me = me.clone();
            icon_button(id, icon, tip).small().on_click(move |_, _, cx| {
                let _ = me.update(cx, |s, cx| f(s, cx));
            })
        };
        let focus = match self.units.len() {
            0 => "Shown channels · every unit".to_string(),
            1 => format!("{} channels of the unit · its spikes coloured", self.channels.len().min(MAX_CHANNELS)),
            n => format!("{} channels · {n} units' spikes coloured", self.channels.len().min(MAX_CHANNELS)),
        };
        let header = h_flex()
            .h(px(HEADER_H))
            .flex_none()
            .px_2()
            .gap_1()
            .border_b_1()
            .border_color(border)
            .child(tab("strip-traces", "Traces", false))
            .child(tab("strip-raster", "Raster", true))
            .child(div().w(px(1.)).h(px(16.)).bg(border).mx_1())
            .child(act("strip-prev", IconName::ChevronLeft, "The lead unit's previous spike", |s, cx| s.step_spike(false, cx)))
            .child(act("strip-next", IconName::ChevronRight, "The lead unit's next spike", |s, cx| s.step_spike(true, cx)))
            .child(div().flex_none().text_xs().text_color(muted).child(format!("{w0:.3} s · {}", duration(win))))
            .child(div().flex_1().min_w_0().truncate().text_xs().text_color(muted).text_right().child(focus))
            .child(act(
                "strip-toggle",
                if self.open { IconName::ChevronDown } else { IconName::ChevronUp },
                if self.open { "Minimise" } else { "Show the traces" },
                |s, cx| s.toggle_open(cx),
            ));
        if !self.open {
            // A card on the workspace's chrome (kitchen-ui panes)
            return v_flex().id("bottom-strip").w_full().h(px(HEADER_H)).flex_none().rounded(px(RADIUS_LG)).overflow_hidden().bg(bg).child(header);
        }
        let height = self.store.read(cx).panel_width("strip.height", STRIP_H);
        v_flex()
            .id("bottom-strip")
            .relative()
            .w_full()
            .h(px(height))
            .flex_none()
            .rounded(px(RADIUS_LG))
            .overflow_hidden()
            .bg(bg)
            .child(header)
            .child(div().flex_1().min_h_0().child(self.panel.clone()))
            .child(self.overview(cx))
            // The workspace's padding is under the strip
            .child(panel_edge("strip.height", Side::Bottom, PANE_GAP))
    }
}
