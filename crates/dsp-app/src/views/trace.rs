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
use gpui_kit::component::dock::Panel;
use gpui_kit::component::ActiveTheme as _;
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::TestSupportExt as _;
use gpui_kit::{
    canvas, div, point, px, relative, rgb, size, App, Bounds, Context, Corners, CursorStyle, Entity, EventEmitter, FocusHandle, Focusable, FontWeight, InteractiveElement as _,
    IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement as _, Pixels, Render, ScrollWheelEvent, SharedString, StatefulInteractiveElement as _,
    Styled as _, Subscription, WeakEntity, Window,
};

use crate::actions::*;
use crate::engine::canvas::Pixel;
use crate::engine::time::renderer::TimeViewKind;
use crate::viewmodels::{ExploreVm, TraceVm};

/// Channel label column and time axis heights (logical px).
pub const GUTTER: f32 = 64.;
pub const AXIS: f32 = 22.;
/// Plot background (the renderer's).
pub const PLOT_BG: u32 = 0x090d13;
const PLAYHEAD: u32 = 0xf43f5e;

pub fn color(p: Pixel) -> gpui_kit::Hsla {
    rgb(((p.r as u32) << 16) | ((p.g as u32) << 8) | p.b as u32).into()
}

#[derive(Debug, Clone, Copy)]
struct Drag {
    /// Pointer x and window start when the drag began.
    x: f32,
    start: f64,
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
    _vm: Subscription,
}

impl TracePanel {
    pub fn new(vm: Entity<TraceVm>, explore: WeakEntity<ExploreVm>, cx: &mut Context<Self>) -> Self {
        let sub = cx.observe(&vm, |_, _, cx| cx.notify());
        Self { vm, explore, focus: cx.focus_handle(), plot: Rc::new(Cell::new(Bounds::default())), drag: None, wheel: 0.0, pointer: None, _vm: sub }
    }

    /// Position of `p` in the plot as fractions (clamped).
    fn fractions(&self, p: gpui_kit::Point<Pixels>) -> (f32, f32) {
        let b = self.plot.get();
        let (w, h) = (b.size.width.as_f32().max(1.0), b.size.height.as_f32().max(1.0));
        (((p.x - b.origin.x).as_f32() / w).clamp(0.0, 1.0), ((p.y - b.origin.y).as_f32() / h).clamp(0.0, 1.0))
    }

    fn store_update(&self, cx: &mut Context<Self>, f: impl FnOnce(&mut crate::store::Store, &mut Context<crate::store::Store>)) {
        let store = self.vm.read(cx).store().clone();
        store.update(cx, f);
    }

    fn on_down(&mut self, e: &MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        window.focus(&self.focus, cx);
        self.vm.update(cx, |vm, cx| vm.pressed(cx));
        let (fx, fy) = self.fractions(e.position);
        if e.click_count == 2 && self.vm.read(cx).view.kind == TimeViewKind::Heatmap {
            if let (Some(ch), Some(explore)) = (self.vm.read(cx).channel_at(fy), self.explore.upgrade()) {
                let vm = self.vm.clone();
                explore.update(cx, |ex, cx| ex.reveal_in_traces(&vm, ch, cx));
            }
            return;
        }
        let _ = fx;
        let start = self.vm.read(cx).store().read(cx).timeline.window_start_sec;
        self.drag = Some(Drag { x: e.position.x.as_f32(), start });
    }

    fn on_move(&mut self, e: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        let (fx, fy) = self.fractions(e.position);
        match self.drag {
            Some(d) if e.pressed_button == Some(MouseButton::Left) => {
                let w = self.plot.get().size.width.as_f32().max(1.0) as f64;
                let dx = (e.position.x.as_f32() - d.x) as f64;
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

    fn on_up(&mut self, _: &MouseUpEvent, _: &mut Window, _: &mut Context<Self>) {
        self.drag = None;
    }

    fn on_wheel(&mut self, e: &ScrollWheelEvent, _: &mut Window, cx: &mut Context<Self>) {
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

    fn overlays(&self, cx: &mut Context<Self>) -> Vec<gpui_kit::AnyElement> {
        let vm = self.vm.read(cx);
        let tl = &vm.store().read(cx).timeline;
        let mut out = Vec::new();
        // Playhead
        let t = (tl.current_time_sec - tl.window_start_sec) / tl.visible_window_sec.max(1e-12);
        if (0.0..=1.0).contains(&t) {
            out.push(div().absolute().top_0().bottom_0().left(relative(t as f32)).w(px(1.5)).bg(rgb(PLAYHEAD)).into_any_element());
        }
        // Crosshair
        if let Some((fx, _)) = self.pointer {
            out.push(div().absolute().top_0().bottom_0().left(relative(fx)).w(px(1.)).bg(gpui_kit::hsla(0., 0., 1., 0.19)).into_any_element());
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
                    .text_color(gpui_kit::hsla(0., 0., 0.75, 1.))
                    .child(label)
                    .into_any_element(),
            );
        }
        // Hover readout
        if !vm.hover.is_empty() && self.pointer.is_some() {
            out.push(
                div()
                    .absolute()
                    .top(px(6.))
                    .left(px(8.))
                    .px_2()
                    .py_0p5()
                    .rounded_md()
                    .bg(gpui_kit::hsla(0., 0., 0., 0.6))
                    .text_xs()
                    .text_color(gpui_kit::white())
                    .child(vm.hover.clone())
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
        let lanes = view.lane_labels();
        let ticks = view.time_ticks(&tl);
        let shown = vm.shown.clone();
        let (canvas_w, canvas_h, canvas_scale) = (view.canvas_width, view.canvas_height, view.scale_factor);
        let empty = view.selection.is_empty();
        let view_id = view.id as usize;
        let muted = cx.theme().muted_foreground;
        let _ = window;

        let gutter = div().w(px(GUTTER)).h_full().relative().overflow_hidden().children(lanes.into_iter().enumerate().map(|(i, l)| {
            let strong = !heatmap && hovered_lane == Some(i);
            div()
                .absolute()
                .left(px(10.))
                .top(relative(l.y_frac))
                .mt(px(-8.))
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
            .on_scroll_wheel(cx.listener(Self::on_wheel))
            .on_hover(cx.listener(|this, hovered: &bool, _, cx| {
                if !hovered {
                    this.pointer = None;
                    this.vm.update(cx, |vm, cx| vm.hover_left(cx));
                }
            }))
            .child(raster)
            .children(self.overlays(cx))
            .when(empty, |d| {
                d.child(div().absolute().size_full().flex().items_center().justify_center().text_sm().text_color(muted).child("No channels selected: pick some in Channels"))
            });

        let axis = div().h(px(AXIS)).flex().child(div().w(px(GUTTER))).child(div().flex_1().h_full().relative().children(ticks.into_iter().map(|t| {
            div().absolute().left(relative(t.frac)).top(px(3.)).ml(px(-20.)).w(px(40.)).flex().justify_center().text_xs().text_color(muted).child(t.label)
        })));

        div()
            .id(("time-view", view_id))
            .test_support()
            .key_context(PLOT)
            .track_focus(&self.focus)
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(PLOT_BG))
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
        "dsp-app.time-view"
    }

    fn set_active(&mut self, active: bool, _: &mut Window, cx: &mut Context<Self>) {
        self.vm.update(cx, |vm, cx| vm.set_active(active, cx));
    }

    fn on_removed(&mut self, _: &mut Window, cx: &mut Context<Self>) {
        let id = self.vm.read(cx).id();
        if let Some(explore) = self.explore.upgrade() {
            explore.update(cx, |ex, cx| ex.removed(id, cx));
        }
    }
}

impl Panel for TracePanel {
    fn title(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        SharedString::from(self.vm.read(cx).view.title.clone())
    }

    fn inner_padding(&self, _: &App) -> bool {
        false
    }
}
