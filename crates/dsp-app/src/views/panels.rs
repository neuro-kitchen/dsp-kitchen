//! The Explore workspace's side panels, as dock panels: Channels (left), View settings (right),
//! Timeline (bottom). Each follows the focused view; their docks open and close with the toggle
//! beside the views' tabs.

use std::cell::Cell;
use std::rc::Rc;

use gpui_kit::base::dock::{DockArea, DockPlacement, Panel as BasePanel, PanelEvent};
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::dock::Panel;
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::switch::Switch;
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, IconName, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::TestSupportExt as _;
use gpui_kit::{
    canvas, div, px, relative, uniform_list, App, AppContext as _, Bounds, Context, CursorStyle, Entity, EventEmitter, FocusHandle, Focusable, InteractiveElement as _, IntoElement,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement as _, Pixels, Render, SharedString, StatefulInteractiveElement as _, Styled as _, Subscription,
    WeakEntity, Window,
};

use crate::engine::time::renderer::TimeViewKind;
use crate::store::{AppEvent, Store};
use crate::viewmodels::{ExploreEvent, ExploreVm};
use crate::views::trace::color;
use crate::widgets::{icon_button, row, Edge, MenuSelect, PanelHeader, Section};

/// The focused view's source id (else the recording's default source).
fn focused_source(explore: &Entity<ExploreVm>, cx: &App) -> Option<String> {
    let ex = explore.read(cx);
    match &ex.focused {
        Some(f) => Some(f.read(cx).view.source.clone()),
        None => ex.store().read(cx).sources().map(|s| s.default_entry().id.clone()),
    }
}

/// Opens and closes the dock a side panel lives in.
#[derive(Clone)]
pub struct DockToggle {
    pub dock: WeakEntity<DockArea>,
    pub placement: DockPlacement,
}

impl DockToggle {
    pub fn is_open(&self, cx: &App) -> bool {
        self.dock.upgrade().is_some_and(|d| d.read(cx).is_dock_open(self.placement))
    }

    pub fn toggle(&self, window: &mut Window, cx: &mut App) {
        let placement = self.placement;
        if let Some(dock) = self.dock.upgrade() {
            dock.update(cx, |d, cx| d.toggle_dock(placement, window, cx));
        }
    }

    fn edge(&self) -> Edge {
        match self.placement {
            DockPlacement::Left => Edge::Left,
            DockPlacement::Right => Edge::Right,
            _ => Edge::Bottom,
        }
    }
}

/// A side panel: its own header (title and the hide button at the window edge), its body, no tab
/// bar (and so no tab menu) while it is alone in its dock.
macro_rules! side_panel {
    ($ty:ty, $name:literal, $title:literal) => {
        impl Render for $ty {
            fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
                let toggle = self.dock.clone();
                let open = toggle.is_open(cx);
                let header = PanelHeader::new(concat!($name, ".toggle"), $title, toggle.edge(), move |window, cx| toggle.toggle(window, cx)).open(open);
                let header = match self.header_extra(cx) {
                    Some(extra) => header.extra(extra),
                    None => header,
                };
                let body = self.body(window, cx);
                v_flex().size_full().child(header).child(div().flex_1().min_h_0().child(body))
            }
        }

        impl EventEmitter<PanelEvent> for $ty {}

        impl Focusable for $ty {
            fn focus_handle(&self, _: &App) -> FocusHandle {
                self.focus.clone()
            }
        }

        impl BasePanel for $ty {
            fn panel_name(&self) -> &'static str {
                $name
            }

            fn closable(&self, _: &App) -> bool {
                false
            }

            fn zoomable(&self, _: &App) -> bool {
                false
            }
        }

        impl Panel for $ty {
            fn title(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
                $title
            }

            fn zoom_control(&self, _: &App) -> Option<gpui_kit::component::dock::PanelControl> {
                None
            }

            fn title_bar(&self, _: &App) -> bool {
                false
            }

            fn inner_padding(&self, _: &App) -> bool {
                false
            }
        }
    };
}

// ----------------------------------------------------------------------------
// Channels
// ----------------------------------------------------------------------------

pub struct ChannelsPanel {
    explore: Entity<ExploreVm>,
    dock: DockToggle,
    focus: FocusHandle,
    filter: Entity<InputState>,
    ranges: Entity<InputState>,
    range_error: Option<String>,
    _subs: Vec<Subscription>,
}

side_panel!(ChannelsPanel, "dsp-app.channels", "Channels");

impl ChannelsPanel {
    pub fn new(explore: Entity<ExploreVm>, dock: DockToggle, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let filter = cx.new(|cx| InputState::new(window, cx).placeholder("Filter, e.g. 12"));
        let ranges = cx.new(|cx| InputState::new(window, cx).placeholder("Show only, e.g. 0-31, 40"));
        let store = explore.read(cx).store().clone();
        let subs = vec![
            cx.observe(&explore, |_, _, cx| cx.notify()),
            cx.subscribe(&explore, |_, _, _: &ExploreEvent, cx| cx.notify()),
            cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
                if matches!(e, AppEvent::SelectionChanged(_) | AppEvent::RecordingChanged) {
                    cx.notify();
                }
            }),
            cx.subscribe(&filter, |_, _, e: &InputEvent, cx| {
                if matches!(e, InputEvent::Change) {
                    cx.notify();
                }
            }),
            cx.subscribe(&ranges, |this, state, e: &InputEvent, cx| {
                if let InputEvent::PressEnter { .. } = e {
                    let text = state.read(cx).value().to_string();
                    this.apply_ranges(&text, cx);
                }
            }),
        ];
        Self { explore, dock, focus: cx.focus_handle(), filter, ranges, range_error: None, _subs: subs }
    }

    fn store(&self, cx: &App) -> Entity<Store> {
        self.explore.read(cx).store().clone()
    }

    fn apply_ranges(&mut self, text: &str, cx: &mut Context<Self>) {
        let Some(source) = focused_source(&self.explore, cx) else { return };
        let result = self.store(cx).update(cx, |s, cx| s.select_ranges(&source, text, cx));
        self.range_error = result.err();
        cx.notify();
    }

    fn with_store(&self, cx: &mut Context<Self>, f: impl FnOnce(&mut Store, &str, &mut Context<Store>)) {
        if let Some(source) = focused_source(&self.explore, cx) {
            self.store(cx).update(cx, |s, cx| f(s, &source, cx));
        }
    }
}

impl ChannelsPanel {
    /// `12 of 385`
    fn header_extra(&self, cx: &App) -> Option<gpui_kit::AnyElement> {
        let source = focused_source(&self.explore, cx)?;
        let s = self.store(cx);
        let s = s.read(cx);
        let total = s.sources()?.entry(&source).channels;
        Some(div().text_xs().text_color(cx.theme().muted_foreground).child(format!("{} of {total}", s.selection(&source).len())).into_any_element())
    }

    fn body(&mut self, _: &mut Window, cx: &mut Context<Self>) -> gpui_kit::AnyElement {
        let store = self.store(cx);
        let Some(source) = focused_source(&self.explore, cx) else {
            return v_flex().size_full().p_3().text_sm().text_color(cx.theme().muted_foreground).child("No recording open.").into_any_element();
        };
        let (name, total, selected, events) = {
            let s = store.read(cx);
            let sources = s.sources().cloned();
            let entry = sources.as_ref().map(|ss| ss.entry(&source).clone());
            let events = s.recording.as_ref().map(|r| r.events.clone()).unwrap_or_default();
            (entry.as_ref().map_or_else(String::new, |e| e.name.clone()), entry.map_or(0, |e| e.channels), s.selection(&source).to_vec(), events)
        };
        let filter = self.filter.read(cx).value().trim().to_string();
        let rows: Vec<usize> = (0..total).filter(|c| filter.is_empty() || c.to_string().contains(&filter)).collect();
        let checked: Rc<Vec<bool>> = Rc::new({
            let mut v = vec![false; total];
            for &c in &selected {
                v[c] = true;
            }
            v
        });
        let palette = store.read(cx).palette();
        let t = cx.theme();
        let (muted, danger) = (t.muted_foreground, t.danger);
        let store_for_rows = store.clone();
        let source_for_rows = source.clone();
        let rows = Rc::new(rows);
        let list_rows = rows.clone();

        let list = uniform_list("channel-rows", rows.len(), move |range, _, _| {
            range
                .map(|i| {
                    let ch = list_rows[i];
                    let on = checked[ch];
                    let (store, source) = (store_for_rows.clone(), source_for_rows.clone());
                    let count = events.count(ch);
                    h_flex()
                        .id(("channel", ch))
                        .h(px(26.))
                        .px_2()
                        .gap_2()
                        .child(Checkbox::new(("channel-check", ch)).checked(on).on_click(move |_, _, cx| store.update(cx, |s, cx| s.toggle_channel(&source, ch, cx))))
                        .child(div().size(px(8.)).rounded_full().bg(color(palette.channel(ch))))
                        .child(div().flex_1().text_sm().when(!on, |d| d.text_color(muted)).child(format!("Ch {ch}")))
                        .when(count > 0, |d| d.child(div().text_xs().text_color(muted).child(count.to_string())))
                })
                .collect()
        })
        .flex_1()
        .min_h_0();

        let actions = h_flex()
            .gap_1()
            .child(Button::new("select-all").ghost().xsmall().label("All").on_click(cx.listener(|this, _, _, cx| this.with_store(cx, |s, src, cx| s.select_all(src, cx)))))
            .child(Button::new("select-none").ghost().xsmall().label("None").on_click(cx.listener(|this, _, _, cx| this.with_store(cx, |s, src, cx| s.select_none(src, cx)))))
            .child(Button::new("select-invert").ghost().xsmall().label("Invert").on_click(cx.listener(|this, _, _, cx| this.with_store(cx, |s, src, cx| s.select_invert(src, cx)))))
            .child(div().flex_1())
            .child(div().text_xs().text_color(muted).truncate().child(name));

        v_flex()
            .id("channels-panel")
            .test_support()
            .size_full()
            .gap_2()
            .p_2()
            .child(actions)
            .child(Input::new(&self.ranges).small())
            .when_some(self.range_error.clone(), |d, e| d.child(div().text_xs().text_color(danger).child(e)))
            .child(Input::new(&self.filter).small().prefix(gpui_kit::component::Icon::new(IconName::Search).small()))
            .child(list)
            .into_any_element()
    }
}

// ----------------------------------------------------------------------------
// View settings
// ----------------------------------------------------------------------------

pub struct SettingsPanel {
    explore: Entity<ExploreVm>,
    dock: DockToggle,
    focus: FocusHandle,
    _focused: Option<Subscription>,
    _subs: Vec<Subscription>,
}

side_panel!(SettingsPanel, "dsp-app.view-settings", "View settings");

impl SettingsPanel {
    pub fn new(explore: Entity<ExploreVm>, dock: DockToggle, cx: &mut Context<Self>) -> Self {
        let subs = vec![cx.subscribe(&explore, |this, _, e: &ExploreEvent, cx| {
            if matches!(e, ExploreEvent::Focus | ExploreEvent::Reset(_)) {
                this.follow(cx);
            }
        })];
        let mut this = Self { explore, dock, focus: cx.focus_handle(), _focused: None, _subs: subs };
        this.follow(cx);
        this
    }

    /// Re-renders with the focused view.
    fn follow(&mut self, cx: &mut Context<Self>) {
        self._focused = self.explore.read(cx).focused.clone().map(|vm| cx.observe(&vm, |_, _, cx| cx.notify()));
        cx.notify();
    }
}

impl SettingsPanel {
    fn header_extra(&self, _: &App) -> Option<gpui_kit::AnyElement> {
        None
    }

    fn body(&mut self, _: &mut Window, cx: &mut Context<Self>) -> gpui_kit::AnyElement {
        let muted = cx.theme().muted_foreground;
        let Some(vm) = self.explore.read(cx).focused.clone() else {
            return v_flex().size_full().p_3().text_sm().text_color(muted).child("Click a view to see its settings here.").into_any_element();
        };
        let v = vm.read(cx).view.clone();
        let sources = vm.read(cx).store().read(cx).sources().cloned();
        let traces = v.kind == TimeViewKind::Traces;

        let kind = h_flex()
            .gap_1()
            .child(
                Button::new("kind-traces")
                    .small()
                    .icon(gpui_kit::assets::IconName::AudioWaveform)
                    .label("Traces")
                    .map(|b| if traces { b.primary() } else { b.outline() })
                    .on_click({
                        let vm = vm.clone();
                        move |_, _, cx| vm.update(cx, |vm, cx| vm.set_kind(TimeViewKind::Traces, cx))
                    }),
            )
            .child(
                Button::new("kind-heatmap")
                    .small()
                    .icon(gpui_kit::assets::IconName::Flame)
                    .label("Heatmap")
                    .map(|b| if !traces { b.primary() } else { b.outline() })
                    .on_click({
                        let vm = vm.clone();
                        move |_, _, cx| vm.update(cx, |vm, cx| vm.set_kind(TimeViewKind::Heatmap, cx))
                    }),
            );

        let source = sources.as_ref().map(|ss| {
            let names: Vec<SharedString> = ss.entries().iter().map(|e| e.summary().into()).collect();
            let ids: Vec<String> = ss.entries().iter().map(|e| e.id.clone()).collect();
            let selected = ids.iter().position(|id| *id == v.source);
            let vm = vm.clone();
            MenuSelect::new("view-source", names, selected, move |i, _, cx| {
                let id = ids[i].clone();
                vm.update(cx, |vm, cx| vm.set_source(&id, cx));
            })
            .full_width()
        });

        let stepper = |id: &'static str, value: String, tip_down: &'static str, tip_up: &'static str, down: Rc<dyn Fn(&mut App)>, up: Rc<dyn Fn(&mut App)>| {
            h_flex()
                .gap_1()
                .child(icon_button(SharedString::from(format!("{id}-down")), IconName::Minus, tip_down).on_click(move |_, _, cx| down(cx)))
                .child(div().min_w(px(64.)).flex().justify_center().text_sm().child(value))
                .child(icon_button(SharedString::from(format!("{id}-up")), IconName::Plus, tip_up).on_click(move |_, _, cx| up(cx)))
        };
        let lanes = v.lanes_on_screen();
        let lanes_row = {
            let (a, b) = (vm.clone(), vm.clone());
            stepper(
                "lanes",
                lanes.to_string(),
                "Fewer channels on screen",
                "More channels on screen",
                Rc::new(move |cx| a.update(cx, |vm, cx| vm.set_lanes(lanes.saturating_sub(1).max(1), cx))),
                Rc::new(move |cx| b.update(cx, |vm, cx| vm.set_lanes(lanes + 1, cx))),
            )
        };
        let gain_row = {
            let (a, b) = (vm.clone(), vm.clone());
            stepper(
                "gain",
                v.gain_label(),
                "Lower gain ([ or Alt+wheel on the view)",
                "Higher gain (] or Alt+wheel on the view)",
                Rc::new(move |cx| a.update(cx, |vm, cx| vm.zoom_gain(0.75, cx))),
                Rc::new(move |cx| b.update(cx, |vm, cx| vm.zoom_gain(1.0 / 0.75, cx))),
            )
        };
        let auto = {
            let vm = vm.clone();
            Switch::new("auto-scale").checked(v.auto_scale).label("Fit amplitude to what is shown").on_click(move |on, _, cx| vm.update(cx, |vm, cx| vm.set_auto_scale(*on, cx)))
        };
        let dc = {
            let vm = vm.clone();
            Switch::new("remove-dc").checked(v.remove_dc).label("Remove DC offset").on_click(move |on, _, cx| vm.update(cx, |vm, cx| vm.set_remove_dc(*on, cx)))
        };

        v_flex()
            .id("settings-panel")
            .test_support()
            .size_full()
            .gap_4()
            .p_3()
            .overflow_y_scroll()
            .child(div().text_base().font_weight(gpui_kit::FontWeight::SEMIBOLD).child(v.title.clone()))
            .child(Section::new("Display").child(kind).children(source).child(row("Range", div().text_sm().child(v.range_label()), cx)))
            .when(traces, |d| d.child(Section::new("Traces").child(row("On screen", lanes_row, cx)).child(row("Gain", gain_row, cx))))
            .child(Section::new("Scaling").child(auto).child(dc))
            .into_any_element()
    }
}

/// The open recording's default source.
fn s_sources_default(store: &Entity<Store>, cx: &App) -> Option<std::sync::Arc<crate::engine::data::Dataset>> {
    Some(store.read(cx).sources()?.default_dataset())
}

// ----------------------------------------------------------------------------
// Timeline
// ----------------------------------------------------------------------------

/// Pixels of a window-box edge that resize instead of move.
const EDGE: f32 = 6.;
const SPEEDS: [f64; 5] = [0.25, 0.5, 1.0, 2.0, 4.0];

#[derive(Debug, Clone, Copy)]
enum TrackDrag {
    /// Moving the window box: seconds between its start and the pointer.
    Window { grab: f64 },
    Edge { left: bool },
    Scrub,
}

pub struct TimelinePanel {
    store: Entity<Store>,
    dock: DockToggle,
    /// Activity per overview column (and the width it was computed for); dropped when more of
    /// the recording is summarized.
    activity: Option<(usize, Rc<Vec<f32>>)>,
    focus: FocusHandle,
    track: Rc<Cell<Bounds<Pixels>>>,
    drag: Option<TrackDrag>,
    _sub: Subscription,
}

side_panel!(TimelinePanel, "dsp-app.timeline", "Timeline");

impl TimelinePanel {
    pub fn new(store: Entity<Store>, dock: DockToggle, cx: &mut Context<Self>) -> Self {
        let sub = cx.subscribe(&store, |this, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::RecordingChanged | AppEvent::SummaryProgress(_) | AppEvent::PaletteChanged) {
                this.activity = None;
            }
            if matches!(e, AppEvent::WindowMoved | AppEvent::PlaybackChanged | AppEvent::RecordingChanged | AppEvent::SummaryProgress(_) | AppEvent::PaletteChanged) {
                cx.notify();
            }
        });
        Self { store, dock, activity: None, focus: cx.focus_handle(), track: Rc::new(Cell::new(Bounds::default())), drag: None, _sub: sub }
    }

    /// Time under `x` of the track.
    fn time_at(&self, x: Pixels, cx: &App) -> f64 {
        let b = self.track.get();
        let w = b.size.width.as_f32().max(1.0);
        (((x - b.origin.x).as_f32() / w).clamp(0.0, 1.0) as f64) * self.store.read(cx).timeline.total_duration_sec
    }

    fn on_down(&mut self, e: &MouseDownEvent, _: &mut Window, cx: &mut Context<Self>) {
        let b = self.track.get();
        if !b.contains(&e.position) {
            return;
        }
        let tl = self.store.read(cx).timeline.clone();
        let total = tl.total_duration_sec.max(1e-9);
        let w = b.size.width.as_f32();
        let x = (e.position.x - b.origin.x).as_f32();
        let (x0, x1) = ((tl.window_start_sec / total) as f32 * w, ((tl.window_start_sec + tl.visible_window_sec) / total) as f32 * w);
        let t = self.time_at(e.position.x, cx);
        self.drag = Some(if (x - x0).abs() <= EDGE {
            TrackDrag::Edge { left: true }
        } else if (x - x1).abs() <= EDGE {
            TrackDrag::Edge { left: false }
        } else if x > x0 && x < x1 {
            TrackDrag::Window { grab: t - tl.window_start_sec }
        } else {
            self.store.update(cx, |s, cx| s.scrub_to(t, cx));
            TrackDrag::Scrub
        });
        cx.stop_propagation();
    }

    fn on_move(&mut self, e: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        let Some(drag) = self.drag else { return };
        if e.pressed_button != Some(MouseButton::Left) {
            self.drag = None;
            return;
        }
        let t = self.time_at(e.position.x, cx);
        self.store.update(cx, |s, cx| {
            let (a, span) = (s.timeline.window_start_sec, s.timeline.visible_window_sec);
            match drag {
                TrackDrag::Window { grab } => s.set_window(t - grab, t - grab + span, cx),
                TrackDrag::Edge { left: true } => s.set_window(t.min(a + span - 0.005), a + span, cx),
                TrackDrag::Edge { left: false } => s.set_window(a, t.max(a + 0.005), cx),
                TrackDrag::Scrub => s.scrub_to(t, cx),
            }
        });
    }

    fn on_up(&mut self, _: &MouseUpEvent, _: &mut Window, _: &mut Context<Self>) {
        self.drag = None;
    }
}

impl TimelinePanel {
    /// The time readout, also visible while the timeline is folded down to its header.
    fn header_extra(&self, cx: &App) -> Option<gpui_kit::AnyElement> {
        let s = self.store.read(cx);
        let muted = cx.theme().muted_foreground;
        let summarizing = s.summary_label().map(|l| div().id("summarizing").text_xs().text_color(cx.theme().primary).child(l));
        Some(h_flex().gap_3().child(div().text_xs().text_color(muted).child(s.timeline.format_time_readout())).children(summarizing).into_any_element())
    }

    fn body(&mut self, _: &mut Window, cx: &mut Context<Self>) -> gpui_kit::AnyElement {
        let s = self.store.read(cx);
        let tl = s.timeline.clone();
        let open = s.recording.is_some();
        let t = cx.theme();
        let (muted, border, accent, fg) = (t.muted_foreground, t.border, t.primary, t.foreground);
        let store = self.store.clone();
        let act = move |f: fn(&mut Store, &mut Context<Store>)| {
            let store = store.clone();
            move |_: &gpui_kit::ClickEvent, _: &mut Window, cx: &mut App| store.update(cx, f)
        };
        use gpui_kit::assets::IconName as Lucide;
        let speed_ix = SPEEDS.iter().position(|v| (*v - tl.playback_speed).abs() < 1e-9);
        let speed = {
            let store = self.store.clone();
            MenuSelect::new("speed", SPEEDS.iter().map(|v| SharedString::from(format!("{v}×"))).collect(), speed_ix, move |i, _, cx| store.update(cx, |s, cx| s.set_speed(SPEEDS[i], cx)))
        };
        let transport = h_flex()
            .gap_0p5()
            .child(icon_button("jump-start", Lucide::SkipBack, "Start of the recording (Home)").on_click(act(Store::jump_to_start)))
            .child(icon_button("step-back", Lucide::StepBack, "Step back").on_click(act(|s, cx| s.step(false, cx))))
            .child(
                Button::new("play")
                    .small()
                    .primary()
                    .icon(if tl.is_playing { IconName::Pause } else { IconName::Play })
                    .tooltip(if tl.is_playing { "Pause (Space)" } else { "Play (Space)" })
                    .on_click(act(Store::toggle_play)),
            )
            .child(icon_button("step-forward", Lucide::StepForward, "Step forward").on_click(act(|s, cx| s.step(true, cx))))
            .child(icon_button("jump-end", Lucide::SkipForward, "End of the recording (End)").on_click(act(Store::jump_to_end)))
            .child(
                icon_button("loop", Lucide::Repeat, if tl.loop_playback { "Looping: on (L)" } else { "Looping: off (L)" })
                    .selected(tl.loop_playback)
                    .on_click(act(Store::toggle_loop)),
            )
            .child(speed);
        // Activity of the whole recording behind the window box (columns of ~2 px)
        let columns = (self.track.get().size.width.as_f32() / 2.0).round().max(1.0) as usize;
        if self.activity.as_ref().is_none_or(|(w, _)| *w != columns) {
            let values = s_sources_default(&self.store, cx).map(|ds| crate::engine::data::dataset::activity(&ds, columns)).unwrap_or_default();
            self.activity = Some((columns, Rc::new(values)));
        }
        let activity = self.activity.as_ref().map(|(_, v)| v.clone()).unwrap_or_default();
        let palette = self.store.read(cx).palette();
        let bar = color(palette.text).opacity(0.55);

        let total = tl.total_duration_sec.max(1e-9);
        let (a, b) = ((tl.window_start_sec / total) as f32, ((tl.window_start_sec + tl.visible_window_sec) / total) as f32);
        let p = (tl.current_time_sec / total) as f32;
        let cell = self.track.clone();
        let track = div()
            .id("overview")
            .h(px(30.))
            .w_full()
            .relative()
            .rounded_md()
            .border_1()
            .border_color(border)
            .bg(color(self.store.read(cx).palette().background))
            .overflow_hidden()
            .cursor(CursorStyle::PointingHand)
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_down))
            .on_mouse_move(cx.listener(Self::on_move))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_up))
            .child(
                canvas(
                    move |bounds, _, _| cell.set(bounds),
                    move |bounds, _, window, _| {
                        let n = activity.len();
                        if n == 0 {
                            return;
                        }
                        let w = bounds.size.width / n as f32;
                        for (i, v) in activity.iter().enumerate() {
                            if !v.is_finite() || *v <= 0.0 {
                                continue;
                            }
                            let h = bounds.size.height * v.min(1.0);
                            let origin = gpui_kit::point(bounds.origin.x + w * i as f32, bounds.origin.y + bounds.size.height - h);
                            window.paint_quad(gpui_kit::fill(Bounds { origin, size: gpui_kit::size(w, h) }, bar));
                        }
                    },
                )
                .absolute()
                .size_full(),
            )
            .child(
                div()
                    .id("overview-window")
                    .absolute()
                    .top_0()
                    .bottom_0()
                    .left(relative(a))
                    .w(relative((b - a).max(0.004)))
                    .min_w(px(6.))
                    .bg(accent.opacity(0.22))
                    .border_x_2()
                    .border_color(accent)
                    .cursor(CursorStyle::ResizeLeftRight)
                    .overflow_hidden()
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_xs()
                    .text_color(fg)
                    .child(crate::store::duration(tl.visible_window_sec)),
            )
            .child(div().absolute().top_0().bottom_0().left(relative(p)).w(px(2.)).bg(color(self.store.read(cx).palette().playhead)));
        let ruler = h_flex().justify_between().children((0..=4).map(|q| div().text_xs().text_color(muted).child(crate::store::duration(total * q as f64 / 4.0))));

        v_flex()
            .id("timeline-panel")
            .test_support()
            .size_full()
            .gap_1p5()
            .px_3()
            .py_2()
            .when(!open, |d| d.opacity(0.5))
            .child(h_flex().gap_3().child(transport))
            .child(track)
            .child(ruler)
            .into_any_element()
    }
}
