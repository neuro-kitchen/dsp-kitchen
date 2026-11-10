//! A time view as a dock panel: channel labels, the rendered image, overlays drawn as elements
//! (playhead, cursor, hover readout, scale bar label) and the time axis.
//!
//! While a new frame renders, the last one is drawn shifted and stretched to the current window,
//! so dragging and zooming answer at once. Input: drag or swipe sideways to move in time,
//! Ctrl+wheel to zoom at the pointer, the wheel to scroll channels, Alt+wheel for gain, a
//! double-click on a heatmap to show that channel in a traces view; keys once the view has focus.

use std::cell::Cell;
use std::rc::Rc;

use gpui_kit::base::dock::{Panel as BasePanel, PanelEvent};
use gpui_kit::component::dock::{Panel, PanelControl};
use gpui_kit::component::menu::{PopupMenu, PopupMenuItem};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::TestSupportExt as _;
use gpui_kit::{
    canvas, div, point, px, relative, rgb, size, App, Bounds, Context, Corners, CursorStyle, Entity, EventEmitter, FocusHandle, Focusable, FontWeight, InteractiveElement as _,
    IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement as _, Pixels, Render, ScrollEvent, SharedString, StatefulInteractiveElement as _,
    Styled as _, Subscription, WeakEntity, Window,
};

use crate::navigation::actions::*;
use crate::ui::plot::canvas::Pixel;
use crate::screens::explore::trace::renderer::TimeViewKind;
use crate::screens::explore::{ExploreVm, TraceVm};

/// Channel label column and time axis heights (logical px).
pub const GUTTER: f32 = 64.;
pub const AXIS: f32 = 22.;

/// A dragged ruler value rounded to 3 significant digits (what is typed in the inspector).
fn nice_value(v: f32) -> f32 {
    if v == 0.0 || !v.is_finite() {
        return 0.0;
    }
    let mag = 10f32.powf(v.abs().log10().floor() - 2.0);
    (v / mag).round() * mag
}

fn fmt_value(v: f32) -> String {
    let s = format!("{:.3}", v);
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

pub fn color(p: Pixel) -> gpui_kit::Hsla {
    rgb(((p.r as u32) << 16) | ((p.g as u32) << 8) | p.b as u32).into()
}

#[derive(Debug, Clone, Copy)]
struct Drag {
    /// Pointer x and window start when the drag began.
    x: f32,
    start: f64,
    /// Height (fraction) where it began, and whether it moved: a press without movement is a
    /// click, which picks the channel under it (its processing shows in Processing).
    fy: f32,
    moved: bool,
}

/// Pointer travel (logical px) that turns a press into a drag.
const CLICK_SLOP: f32 = 3.0;
/// How close (logical px) a press must be to a ruler line to grab it.
const RULER_GRAB: f32 = 5.0;

/// The level ruler: the Crossings op the view's chain ends with, drawn as lines at its level and
/// release in every lane; dragging one sets that value.
#[derive(Debug, Clone)]
struct Ruler {
    /// Session view and op the values belong to.
    view: String,
    op: String,
    node: dspk_pipeline::Node,
    level: f32,
    release: f32,
    /// Lines on both sides of the baseline (`direction = both`).
    both: bool,
}

/// A ruler line being dragged: which one, and its value now.
#[derive(Debug, Clone, Copy)]
struct RulerDrag {
    release: bool,
    value: f32,
}

pub struct TracePanel {
    vm: Entity<TraceVm>,
    explore: WeakEntity<ExploreVm>,
    focus: FocusHandle,
    plot: Rc<Cell<Bounds<Pixels>>>,
    drag: Option<Drag>,
    /// Wheel movement not yet turned into channel steps.
    wheel: f32,
    /// Pointer in the plot (fractions), for the crosshair.
    pointer: Option<(f32, f32)>,
    ruler_drag: Option<RulerDrag>,
    _vm: Subscription,
}

impl TracePanel {
    pub fn new(vm: Entity<TraceVm>, explore: WeakEntity<ExploreVm>, cx: &mut Context<Self>) -> Self {
        let sub = cx.observe(&vm, |_, _, cx| cx.notify());
        Self { vm, explore, focus: cx.focus_handle(), plot: Rc::new(Cell::new(Bounds::default())), drag: None, wheel: 0.0, pointer: None, ruler_drag: None, _vm: sub }
    }

    /// Position of `p` in the plot as fractions (clamped).
    fn fractions(&self, p: gpui_kit::Point<Pixels>) -> (f32, f32) {
        let b = self.plot.get();
        let (w, h) = (b.size.width.as_f32().max(1.0), b.size.height.as_f32().max(1.0));
        (((p.x - b.origin.x).as_f32() / w).clamp(0.0, 1.0), ((p.y - b.origin.y).as_f32() / h).clamp(0.0, 1.0))
    }

    fn store_update(&self, cx: &mut Context<Self>, f: impl FnOnce(&mut crate::data::store::Store, &mut Context<crate::data::store::Store>)) {
        let store = self.vm.read(cx).store().clone();
        store.update(cx, f);
    }

    /// The ruler of this view, when its chain ends with an enabled Crossings op (traces only).
    fn ruler(&self, cx: &App) -> Option<Ruler> {
        let vm = self.vm.read(cx);
        if vm.view.kind != TimeViewKind::Traces {
            return None;
        }
        let s = vm.store().read(cx);
        let id = if vm.view.doc_view.is_empty() { s.doc.active()?.id.clone() } else { vm.view.doc_view.clone() };
        let v = s.doc.view(&id)?;
        let op = v.ops.iter().rev().find(|o| o.enabled)?;
        let dspk_pipeline::Node::Crossings { level, release, direction, .. } = &op.node else { return None };
        let both = *direction == dspk_pipeline::Direction::Both;
        let (level, release) = if both { (level.abs(), release.unwrap_or(*level).abs()) } else { (*level, release.unwrap_or(*level)) };
        Some(Ruler { view: id, op: op.id.clone(), node: op.node.clone(), level, release, both })
    }

    /// Pixels (physical) per signal unit in a lane, and the lane height (physical px).
    fn lane_scale(&self, cx: &App) -> (f32, f32) {
        let v = &self.vm.read(cx).view;
        let h = v.canvas_height.max(1) as f32;
        let lanes = v.drawn_channels().len().max(1) as f32;
        let lane_h = h / lanes;
        (crate::screens::explore::trace::renderer::px_per_unit(lane_h, v.gain, v.amp_scale), lane_h)
    }

    /// The ruler line near plot height `fy` (fraction), if any: whether it is the release.
    fn ruler_hit(&self, fy: f32, cx: &App) -> Option<bool> {
        let r = self.ruler(cx)?;
        if self.vm.read(cx).view.remove_dc {
            return None;
        }
        let (k, lane_h) = self.lane_scale(cx);
        let v = &self.vm.read(cx).view;
        let h = v.canvas_height.max(1) as f32;
        let y = fy * h;
        let lane = (y / lane_h).floor();
        let center = (lane + 0.5) * lane_h;
        let grab = RULER_GRAB * v.scale_factor;
        let signs: &[f32] = if r.both { &[1.0, -1.0] } else { &[1.0] };
        for &s in signs {
            if (center - s * r.level * k - y).abs() <= grab {
                return Some(false);
            }
            if (r.release - r.level).abs() > f32::EPSILON && (center - s * r.release * k - y).abs() <= grab {
                return Some(true);
            }
        }
        None
    }

    fn on_down(&mut self, e: &MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        window.focus(&self.focus, cx);
        self.vm.update(cx, |vm, cx| vm.pressed(cx));
        let (fx, fy) = self.fractions(e.position);
        if let Some(release) = self.ruler_hit(fy, cx) {
            let r = self.ruler(cx).expect("a ruler was hit");
            self.ruler_drag = Some(RulerDrag { release, value: if release { r.release } else { r.level } });
            cx.notify();
            return;
        }
        if e.click_count == 2 && self.vm.read(cx).view.kind == TimeViewKind::Heatmap {
            if let (Some(ch), Some(explore)) = (self.vm.read(cx).channel_at(fy), self.explore.upgrade()) {
                let vm = self.vm.clone();
                explore.update(cx, |ex, cx| ex.reveal_in_traces(&vm, ch, cx));
            }
            return;
        }
        if e.click_count == 2 && self.vm.read(cx).view.kind == TimeViewKind::Traces {
            if let Some(ch) = self.vm.read(cx).channel_at(fy) {
                self.vm.update(cx, |vm, cx| vm.toggle_solo(ch, cx));
            }
            return;
        }
        let _ = fx;
        let start = self.vm.read(cx).store().read(cx).timeline.window_start_sec;
        self.drag = Some(Drag { x: e.position.x.as_f32(), start, fy, moved: false });
    }

    fn on_move(&mut self, e: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        let (fx, fy) = self.fractions(e.position);
        if let Some(d) = self.ruler_drag {
            if e.pressed_button != Some(MouseButton::Left) {
                self.ruler_drag = None;
            } else {
                // The value under the pointer in its lane (magnitudes for `both`)
                let (k, lane_h) = self.lane_scale(cx);
                let h = self.vm.read(cx).view.canvas_height.max(1) as f32;
                let y = fy * h;
                let center = ((y / lane_h).floor() + 0.5) * lane_h;
                let mut value = (center - y) / k.max(f32::MIN_POSITIVE);
                if self.ruler(cx).is_some_and(|r| r.both) {
                    value = value.abs();
                }
                self.ruler_drag = Some(RulerDrag { value, ..d });
                cx.notify();
                return;
            }
        }
        match self.drag {
            Some(d) if e.pressed_button == Some(MouseButton::Left) => {
                let w = self.plot.get().size.width.as_f32().max(1.0) as f64;
                let dx = (e.position.x.as_f32() - d.x) as f64;
                if dx.abs() as f32 > CLICK_SLOP {
                    if let Some(drag) = &mut self.drag {
                        drag.moved = true;
                    }
                }
                self.store_update(cx, |s, cx| {
                    let span = s.timeline.visible_window_sec;
                    let start = d.start - dx / w * span;
                    s.set_window(start, start + span, cx);
                });
            }
            Some(_) => self.drag = None,
            None => {}
        }
        if self.plot.get().contains(&e.position) {
            self.pointer = Some((fx, fy));
            self.vm.update(cx, |vm, cx| vm.hover_at(fx, fy, cx));
            cx.notify();
        }
    }

    fn on_up(&mut self, _: &MouseUpEvent, _: &mut Window, cx: &mut Context<Self>) {
        if let Some(d) = self.ruler_drag.take() {
            // One change on release (one undo step), the level and release kept in order
            if let Some(r) = self.ruler(cx) {
                let mut node = r.node.clone();
                if let dspk_pipeline::Node::Crossings { level, release, direction, .. } = &mut node {
                    let falling = *direction == dspk_pipeline::Direction::Falling;
                    let v = nice_value(d.value);
                    if d.release {
                        let below = if falling { v.max(*level) } else { v.min(level.abs()) };
                        *release = Some(below);
                    } else {
                        *level = v;
                        if let Some(rel) = release.as_mut() {
                            *rel = if falling { rel.max(v) } else { rel.min(v.abs()) };
                        }
                    }
                }
                let cmd = crate::domain::command::SessionCommand::SetOp { view: r.view, op: r.op, node };
                let _ = self.vm.read(cx).store().clone().update(cx, |s, cx| s.execute(cmd, cx));
            }
            cx.notify();
            return;
        }
        if let Some(d) = self.drag.take() {
            if !d.moved {
                if let Some(ch) = self.vm.read(cx).channel_at(d.fy) {
                    self.store_update(cx, |s, cx| s.pick_channel(Some(ch), cx));
                }
            }
        }
    }

    fn on_wheel(&mut self, e: &ScrollEvent, _: &mut Window, cx: &mut Context<Self>) {
        let d = e.delta.pixel_delta(px(20.));
        let (dx, dy) = (d.x.as_f32(), d.y.as_f32());
        let (fx, _) = self.fractions(e.position);
        let w = self.plot.get().size.width.as_f32().max(1.0) as f64;
        let m = e.modifiers;
        if m.control || m.platform {
            // Zoom around the pointer: wheel up = closer
            self.store_update(cx, |s, cx| s.zoom_at((dy as f64 * 0.004).exp(), fx as f64, cx));
        } else if m.alt {
            self.vm.update(cx, |vm, cx| vm.zoom_gain((-dy * 0.004).exp(), cx));
        } else if dx != 0.0 || m.shift {
            // Sideways (trackpad) or Shift+wheel: move in time, the content follows the fingers
            let moved = if dx != 0.0 { dx } else { dy } as f64;
            self.store_update(cx, |s, cx| {
                let dt = -moved / w * s.timeline.visible_window_sec;
                s.pan_time(dt, cx);
            });
        } else {
            self.wheel += dy;
            let steps = (self.wheel / 20.0).trunc();
            if steps != 0.0 {
                self.wheel -= steps * 20.0;
                // One channel per notch; wheel down shows later channels
                self.vm.update(cx, |vm, cx| vm.scroll_by(-(steps as i64), cx));
            }
        }
        cx.stop_propagation();
    }

    /// The ruler's lines in every lane (or, with Remove DC on, a button to turn it off: lanes are
    /// then shifted by an offset the ruler can't know).
    fn ruler_lines(&self, cx: &App) -> Vec<gpui_kit::AnyElement> {
        let Some(r) = self.ruler(cx) else { return Vec::new() };
        if self.vm.read(cx).view.events_only {
            return Vec::new();
        }
        let vm = self.vm.read(cx);
        let palette = vm.store().read(cx).palette();
        if vm.view.remove_dc {
            let vm_entity = self.vm.clone();
            return vec![div()
                .id("ruler-dc")
                .absolute()
                .top(px(6.))
                .left(px(8.))
                .h(px(24.))
                .px_2()
                .flex()
                .items_center()
                .rounded(px(4.))
                .bg(color(palette.background))
                .text_xs()
                .text_color(color(palette.text))
                .cursor(CursorStyle::PointingHand)
                .on_mouse_down(MouseButton::Left, move |_, _, cx| {
                    cx.stop_propagation();
                    vm_entity.update(cx, |vm, cx| vm.set_remove_dc(false, cx));
                })
                .child("Level ruler: turn off Remove DC")
                .into_any_element()];
        }
        let (k, lane_h) = self.lane_scale(cx);
        let h = vm.view.canvas_height.max(1) as f32;
        let lanes = vm.view.drawn_channels().len();
        let (mut level, mut release) = (r.level, r.release);
        if let Some(d) = self.ruler_drag {
            if d.release { release = d.value } else { level = d.value }
        }
        let amber = crate::screens::explore::components::timeline_tab::tick_color(45.0, 1.0);
        let unit = vm.view.unit.clone();
        let mut out = Vec::new();
        let signs: &[f32] = if r.both { &[1.0, -1.0] } else { &[1.0] };
        for lane in 0..lanes {
            let center = (lane as f32 + 0.5) * lane_h;
            for &s in signs {
                for (value, alpha, thick) in [(level, 0.95, 1.5), (release, 0.5, 1.0)] {
                    if thick < 1.5 && (release - level).abs() <= f32::EPSILON {
                        continue;
                    }
                    let y = (center - s * value * k) / h;
                    if !(0.0..=1.0).contains(&y) {
                        continue;
                    }
                    out.push(div().absolute().left_0().right_0().top(relative(y)).mt(px(-thick / 2.0)).h(px(thick)).bg(amber.opacity(alpha)).into_any_element());
                }
            }
        }
        // The values, once, at the top lane's line
        let mut label = if (release - level).abs() > f32::EPSILON { format!("level {} · release {} {unit}", fmt_value(level), fmt_value(release)) } else { format!("level {} {unit}", fmt_value(level)) };
        if let Some((_, events)) = &vm.live {
            label.push_str(&format!(" · {} event{} here", events.len(), if events.len() == 1 { "" } else { "s" }));
        }
        let y0 = ((0.5 * lane_h - level * k) / h).clamp(0.0, 0.95);
        out.push(div().absolute().left(px(8.)).top(relative(y0)).mt(px(-18.)).px_1().rounded(px(3.)).bg(color(palette.background)).text_xs().text_color(amber).child(label).into_any_element());
        out
    }

    /// The live events of the window (the chain ends with Crossings): a tick at each onset and a
    /// faint band to its offset, in the event's lane; or why there are none.
    fn live_ticks(&self, cx: &App) -> Vec<gpui_kit::AnyElement> {
        let vm = self.vm.read(cx);
        if !vm.view.events || (vm.view.remove_dc && !vm.view.events_only) {
            return Vec::new();
        }
        let store = vm.store().read(cx);
        let palette = store.palette();
        if let Some(note) = &vm.live_note {
            return vec![div()
                .absolute()
                .top(px(6.))
                .right(px(8.))
                .h(px(24.))
                .px_2()
                .flex()
                .items_center()
                .rounded(px(4.))
                .bg(color(palette.background))
                .text_xs()
                .text_color(color(palette.text))
                .child(format!("Live events: {note}"))
                .into_any_element()];
        }
        let Some((_, events)) = &vm.live else { return Vec::new() };
        let drawn = vm.view.drawn_channels();
        let lanes = drawn.len().max(1) as f32;
        let (t0, win) = (store.timeline.window_start_sec, store.timeline.visible_window_sec.max(1e-9));
        let amber = crate::screens::explore::components::timeline_tab::tick_color(45.0, 1.0);
        let mut out = Vec::new();
        for e in events.iter().take(Self::MAX_TICKS) {
            let Some(lane) = drawn.iter().position(|&c| c == e.channel as usize) else { continue };
            let x = ((e.onset_s - t0) / win) as f32;
            let top = lane as f32 / lanes;
            let h = 1.0 / lanes;
            if let Some(off) = e.offset_s {
                let x1 = (((off - t0) / win) as f32).min(1.0);
                if x1 > x {
                    out.push(div().absolute().left(relative(x.max(0.0))).w(relative(x1 - x.max(0.0))).top(relative(top + 0.1 * h)).h(relative(0.8 * h)).bg(amber.opacity(0.10)).into_any_element());
                }
            }
            if (0.0..=1.0).contains(&x) {
                out.push(div().absolute().left(relative(x)).ml(px(-1.)).w(px(2.)).top(relative(top + 0.1 * h)).h(relative(0.8 * h)).bg(amber.opacity(0.95)).into_any_element());
            }
        }
        out
    }

    /// Most ticks drawn at once (a zoomed-out view of a dense set shows the first ones).
    const MAX_TICKS: usize = 4000;

    /// The visible event sets of the open recording, as ticks: onsets (and offsets, fainter, when
    /// the set shows them) inside the window, in the lane of their channel.
    fn event_ticks(&self, cx: &App) -> Vec<gpui_kit::AnyElement> {
        let vm = self.vm.read(cx);
        if vm.view.kind != TimeViewKind::Traces || !vm.view.events {
            return Vec::new();
        }
        let store = vm.store().read(cx);
        let Some(rec) = store.recording.as_ref().and_then(|r| r.path.clone()) else { return Vec::new() };
        let drawn = vm.view.drawn_channels().to_vec();
        let lanes = drawn.len().max(1) as f32;
        let (t0, win) = (store.timeline.window_start_sec, store.timeline.visible_window_sec.max(1e-9));
        let mut out = Vec::new();
        for set in store.doc.event_sets.iter().filter(|s| s.visible && s.recording == rec) {
            let Some(events) = store.events_of(set) else { continue };
            let tick = |t: f64, channel: u32, alpha: f32| -> Option<gpui_kit::AnyElement> {
                let x = ((t - t0) / win) as f32;
                let lane = drawn.iter().position(|&c| c == channel as usize)? as f32;
                if !(0.0..=1.0).contains(&x) {
                    return None;
                }
                let h = set.tick_height / lanes;
                let y = (lane + 0.5) / lanes - h / 2.0;
                let w = set.tick_width;
                Some(
                    div()
                        .absolute()
                        .left(relative(x))
                        .ml(px(-w / 2.0))
                        .top(relative(y))
                        .h(relative(h))
                        .w(px(w))
                        .bg(crate::screens::explore::components::timeline_tab::tick_color(set.hue, alpha))
                        .into_any_element(),
                )
            };
            let first = events.partition_point(|e| e.onset_s < t0);
            for e in events[first..].iter().take_while(|e| e.onset_s <= t0 + win) {
                if out.len() >= Self::MAX_TICKS {
                    return out;
                }
                out.extend(tick(e.onset_s, e.channel, 1.0));
            }
            if set.show_offsets {
                // An offset in the window may belong to an event that began before it
                for e in events.iter().take_while(|e| e.onset_s <= t0 + win) {
                    if out.len() >= Self::MAX_TICKS {
                        return out;
                    }
                    if let Some(off) = e.offset_s.filter(|&o| o >= t0) {
                        out.extend(tick(off, e.channel, 0.45));
                    }
                }
            }
        }
        out
    }

    fn overlays(&self, cx: &mut Context<Self>) -> Vec<gpui_kit::AnyElement> {
        let vm = self.vm.read(cx);
        let palette = vm.store().read(cx).palette();
        let tl = &vm.store().read(cx).timeline;
        // Overlays drawn over the plot: light on dark plots, dark on light ones
        let ink = |a: f32| if palette.dark { gpui_kit::hsla(0., 0., 1., a) } else { gpui_kit::hsla(0., 0., 0., a) };
        let mut out = Vec::new();
        // Playhead
        let t = (tl.current_time_sec - tl.window_start_sec) / tl.visible_window_sec.max(1e-12);
        if (0.0..=1.0).contains(&t) {
            out.push(div().absolute().top_0().bottom_0().left(relative(t as f32)).w(px(1.5)).bg(color(palette.playhead)).into_any_element());
        }
        // Crosshair
        if let Some((fx, _)) = self.pointer {
            out.push(div().absolute().top_0().bottom_0().left(relative(fx)).w(px(1.)).bg(ink(0.22)).into_any_element());
        }
        // Scale bar label beside the raster's bar (bottom right)
        let label = vm.view.scale_bar_label();
        if !label.is_empty() {
            let y = vm.view.scale_bar_center_frac();
            out.push(
                div()
                    .absolute()
                    .right(px(18.))
                    .top(relative(y))
                    .mt(px(-7.))
                    .text_xs()
                    .text_color(color(palette.text))
                    .child(label)
                    .into_any_element(),
            );
        }
        // Event sets of this recording as ticks in their channel's lane (traces only)
        out.extend(self.event_ticks(cx));
        out.extend(self.live_ticks(cx));
        out.extend(self.ruler_lines(cx));
        // One channel shown alone: say so; the chip (or double-click, or Esc) goes back
        if let Some(solo) = &vm.view.solo {
            let ch = solo.channel;
            out.push(
                div()
                    .id("solo")
                    .absolute()
                    .top(px(6.))
                    .right(px(8.))
                    .h(px(24.))
                    .px_2()
                    .flex()
                    .items_center()
                    .rounded(px(4.))
                    .bg(color(palette.background))
                    .text_xs()
                    .text_color(color(palette.text))
                    .cursor(CursorStyle::PointingHand)
                    .on_mouse_down(MouseButton::Left, cx.listener(|this, _: &MouseDownEvent, _, cx| {
                        cx.stop_propagation();
                        this.vm.update(cx, |vm, cx| vm.leave_solo(cx));
                    }))
                    .child(format!("Ch {ch} alone · Show all (Esc)"))
                    .into_any_element(),
            );
        }
        out
    }
}

impl Render for TracePanel {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let vm = self.vm.read(cx);
        let view = &vm.view;
        let tl = vm.store().read(cx).timeline.clone();
        let heatmap = view.kind == TimeViewKind::Heatmap;
        let hovered_lane = self.pointer.map(|(_, fy)| (fy * view.drawn_channels().len() as f32) as usize);
        let palette = vm.store().read(cx).palette();
        let lanes = view.lane_labels(&palette);
        let ticks = view.time_ticks(&tl);
        let shown = vm.shown.clone();
        let (canvas_w, canvas_h, canvas_scale) = (view.canvas_width, view.canvas_height, view.scale_factor);
        let empty = view.selection.is_empty();
        let events_only = view.events_only && view.kind == TimeViewKind::Traces;
        let view_id = view.id as usize;
        let muted = color(palette.text);
        let _ = window;

        let gutter = div().w(px(GUTTER)).h_full().relative().overflow_hidden().children(lanes.into_iter().enumerate().map(|(i, l)| {
            let strong = !heatmap && hovered_lane == Some(i);
            // Labels at the very top or bottom stay inside the gutter
            let nudge = if l.y_frac <= 0.04 { 0. } else if l.y_frac >= 0.96 { -16. } else { -8. };
            div()
                .absolute()
                .left(px(10.))
                .top(relative(l.y_frac))
                .mt(px(nudge))
                .text_xs()
                .text_color(color(l.color))
                .when(strong, |d| d.font_weight(FontWeight::BOLD))
                .child(SharedString::from(l.label))
        }));

        let plot_cell = self.plot.clone();
        let vm_handle = self.vm.downgrade();
        let raster = canvas(
            move |bounds, window, cx| {
                plot_cell.set(bounds);
                let scale = window.scale_factor();
                let (w, h) = ((bounds.size.width.as_f32() * scale).round() as u32, (bounds.size.height.as_f32() * scale).round() as u32);
                if (w, h, scale) != (canvas_w, canvas_h, canvas_scale) && w > 0 && h > 0 {
                    let vm = vm_handle.clone();
                    cx.defer(move |cx| {
                        let _ = vm.update(cx, |vm, cx| vm.set_plot_size(w, h, scale, cx));
                    });
                }
            },
            move |bounds, _, window, _| {
                let Some(s) = shown else { return };
                // The frame's window placed in the current one: shifted / stretched until the
                // exact frame arrives
                let span = tl.visible_window_sec.max(1e-12);
                let x = bounds.origin.x + bounds.size.width * (((s.start - tl.window_start_sec) / span) as f32);
                let w = bounds.size.width * ((s.window / span) as f32);
                let image_bounds = Bounds { origin: point(x, bounds.origin.y), size: size(w, bounds.size.height) };
                let _ = window.paint_image(bounds, image_bounds, Corners::default(), s.image, 0, false);
            },
        )
        .absolute()
        .size_full();

        let plot = div()
            .id("plot")
            .flex_1()
            .h_full()
            .relative()
            .overflow_hidden()
            .cursor(if self.drag.is_some() { CursorStyle::ClosedHand } else { CursorStyle::Crosshair })
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_down))
            .on_mouse_move(cx.listener(Self::on_move))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_up))
            .on_scroll(cx.listener(Self::on_wheel))
            .on_hover(cx.listener(|this, hovered: &bool, _, cx| {
                if !hovered {
                    this.pointer = None;
                    this.vm.update(cx, |vm, cx| vm.hover_left(cx));
                }
            }))
            // Events only: the ticks over an empty plot
            .when(!events_only, |d| d.child(raster))
            .children(self.overlays(cx))
            .when(empty, |d| {
                d.child(div().absolute().size_full().flex().items_center().justify_center().text_sm().text_color(muted).child("No channels selected: pick some in Channels"))
            });

        // Ticks at the ends align inward so they are never cut off; the hover readout sits at the
        // right of the axis (off the traces)
        let readout = (!self.vm.read(cx).hover.is_empty() && self.pointer.is_some()).then(|| {
            div().absolute().right_0().top_0().h_full().pl_3().pr_1().flex().items_center().bg(color(palette.background)).text_xs().text_color(color(palette.text)).child(self.vm.read(cx).hover.clone())
        });
        let axis = div().h(px(AXIS)).flex().child(div().w(px(GUTTER))).child(
            div()
                .flex_1()
                .h_full()
                .relative()
                .children(ticks.into_iter().map(|t| {
                    let tick = div().absolute().left(relative(t.frac)).top(px(3.)).w(px(48.)).flex().text_xs().text_color(muted).child(t.label);
                    if t.frac < 0.04 {
                        tick.justify_start()
                    } else if t.frac > 0.96 {
                        tick.ml(px(-48.)).justify_end()
                    } else {
                        tick.ml(px(-24.)).justify_center()
                    }
                }))
                .children(readout),
        );

        div()
            .id(("time-view", view_id))
            .test_support()
            .key_context(PLOT)
            .track_focus(&self.focus)
            .size_full()
            .flex()
            .flex_col()
            .bg(color(palette.background))
            .on_action(cx.listener(|this, _: &PlayPause, _, cx| this.store_update(cx, |s, cx| s.toggle_play(cx))))
            .on_action(cx.listener(|this, _: &PanBack, _, cx| this.store_update(cx, |s, cx| s.pan_fraction(-0.1, cx))))
            .on_action(cx.listener(|this, _: &PanForward, _, cx| this.store_update(cx, |s, cx| s.pan_fraction(0.1, cx))))
            .on_action(cx.listener(|this, _: &PageBack, _, cx| this.store_update(cx, |s, cx| s.pan_fraction(-1.0, cx))))
            .on_action(cx.listener(|this, _: &PageForward, _, cx| this.store_update(cx, |s, cx| s.pan_fraction(1.0, cx))))
            .on_action(cx.listener(|this, _: &ZoomIn, _, cx| this.store_update(cx, |s, cx| s.zoom_at(0.8, 0.5, cx))))
            .on_action(cx.listener(|this, _: &ZoomOut, _, cx| this.store_update(cx, |s, cx| s.zoom_at(1.25, 0.5, cx))))
            .on_action(cx.listener(|this, _: &JumpStart, _, cx| this.store_update(cx, |s, cx| s.jump_to_start(cx))))
            .on_action(cx.listener(|this, _: &JumpEnd, _, cx| this.store_update(cx, |s, cx| s.jump_to_end(cx))))
            .on_action(cx.listener(|this, _: &ToggleLoop, _, cx| this.store_update(cx, |s, cx| s.toggle_loop(cx))))
            .on_action(cx.listener(|this, _: &ChannelUp, _, cx| this.vm.update(cx, |vm, cx| vm.scroll_by(-1, cx))))
            .on_action(cx.listener(|this, _: &ChannelDown, _, cx| this.vm.update(cx, |vm, cx| vm.scroll_by(1, cx))))
            .on_action(cx.listener(|this, _: &ChannelPageUp, _, cx| this.vm.update(cx, |vm, cx| vm.page(false, cx))))
            .on_action(cx.listener(|this, _: &ChannelPageDown, _, cx| this.vm.update(cx, |vm, cx| vm.page(true, cx))))
            .on_action(cx.listener(|this, _: &GainUp, _, cx| this.vm.update(cx, |vm, cx| vm.zoom_gain(1.0 / 0.75, cx))))
            .on_action(cx.listener(|this, _: &GainDown, _, cx| this.vm.update(cx, |vm, cx| vm.zoom_gain(0.75, cx))))
            .on_action(cx.listener(|this, _: &LeaveSolo, _, cx| this.vm.update(cx, |vm, cx| vm.leave_solo(cx))))
            .child(div().flex_1().min_h_0().flex().child(gutter).child(plot))
            .child(axis)
    }
}

impl EventEmitter<PanelEvent> for TracePanel {}

impl Focusable for TracePanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl BasePanel for TracePanel {
    fn panel_name(&self) -> &'static str {
        "dspk-app.time-view"
    }

    fn set_active(&mut self, active: bool, _: &mut Window, cx: &mut Context<Self>) {
        self.vm.update(cx, |vm, cx| {
            // Its tab was just chosen: that selects the view, so the Inspector follows
            let shown_now = active && !vm.is_active();
            vm.set_active(active, cx);
            if shown_now {
                vm.pressed(cx);
            }
        });
    }

    fn on_removed(&mut self, _: &mut Window, cx: &mut Context<Self>) {
        let id = self.vm.read(cx).id();
        if let Some(explore) = self.explore.upgrade() {
            explore.update(cx, |ex, cx| ex.removed(id, cx));
        }
    }
}

impl Panel for TracePanel {
    /// The kind and the source, cut to fit; the whole title in the tooltip.
    fn title(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let view = &self.vm.read(cx).view;
        let full = SharedString::from(view.title.clone());
        div().id(("view-title", view.id as usize)).max_w(px(280.)).truncate().child(full.clone()).tooltip(move |window, cx| gpui_kit::component::tooltip::Tooltip::new(full.clone()).build(window, cx))
    }

    /// Maximise / restore as a button in the view's toolbar.
    fn zoom_control(&self, _: &App) -> Option<PanelControl> {
        Some(PanelControl::Toolbar)
    }

    fn dropdown_menu(&mut self, menu: PopupMenu, _: &mut Window, cx: &mut Context<Self>) -> PopupMenu {
        let explore = self.explore.clone();
        let add = move |kind: TimeViewKind| {
            let explore = explore.clone();
            move |_: &gpui_kit::ClickEvent, _: &mut Window, cx: &mut App| {
                if let Some(ex) = explore.upgrade() {
                    ex.update(cx, |ex, cx| {
                        ex.add(kind, cx);
                    });
                }
            }
        };
        let (explore, id) = (self.explore.clone(), self.vm.read(cx).id());
        menu.item(PopupMenuItem::new("Add traces view").on_click(add(TimeViewKind::Traces)))
            .item(PopupMenuItem::new("Add heatmap view").on_click(add(TimeViewKind::Heatmap)))
            .item({
                // A new session view of the channels shown now, with this view's chain
                let vm = self.vm.clone();
                PopupMenuItem::new("Copy with shown channels").on_click(move |_, _, cx| {
                    let (store, doc_view, source) = {
                        let v = vm.read(cx);
                        (v.store().clone(), v.shown_view(v.store().read(cx)).map(|d| d.id.clone()), v.view.source.clone())
                    };
                    let Some(view) = doc_view else { return };
                    store.update(cx, |s, cx| {
                        let sel = s.selection(&source).to_vec();
                        let channels = (!sel.is_empty()).then_some(sel);
                        let _ = s.execute(crate::domain::command::SessionCommand::CopyView { view, channels, name: None }, cx);
                    });
                    // This panel now shows (and the Inspector edits) the copy
                    if let Some(copy) = store.read(cx).doc.active_view.clone() {
                        vm.update(cx, |vm, cx| {
                            vm.view.doc_view = copy;
                            cx.notify();
                        });
                    }
                })
            })
            .separator()
            .item(PopupMenuItem::new("Close view").on_click(move |_, _, cx| {
                if let Some(ex) = explore.upgrade() {
                    ex.update(cx, |ex, cx| ex.close(id, cx));
                }
            }))
    }

    fn inner_padding(&self, _: &App) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dragged_ruler_values_are_rounded_to_three_digits() {
        assert_eq!(nice_value(52.3456), 52.3);
        assert!((nice_value(0.012345) - 0.0123).abs() < 1e-7);
        assert_eq!(nice_value(-1234.5), -1230.0);
        assert_eq!(nice_value(0.0), 0.0);
        assert_eq!(fmt_value(50.0), "50");
        assert_eq!(fmt_value(0.125), "0.125");
    }
}
