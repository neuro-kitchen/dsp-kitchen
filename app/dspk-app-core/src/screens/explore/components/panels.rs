//! The Explore workspace's side panels, as dock panels: Channels (left), View settings (right),
//! Timeline (bottom). Each follows the focused view; their docks open and close with the toggle
//! beside the views' tabs.

use std::cell::Cell;
use std::rc::Rc;

use gpui_kit::base::dock::{DockPlacement, Panel as BasePanel, PanelEvent};
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::dock::Panel;
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::switch::Switch;
use gpui_kit::component::menu::DropdownMenu as _;
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, IconName, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::TestSupportExt as _;
use gpui_kit::{
    canvas, div, px, relative, uniform_list, App, AppContext as _, Bounds, Context, CursorStyle, Entity, EventEmitter, FocusHandle, Focusable, InteractiveElement as _, IntoElement,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement as _, Pixels, Render, SharedString, StatefulInteractiveElement as _, Styled as _, Subscription,
    Window,
};

use crate::screens::explore::trace::renderer::{ColorBy, TimeViewKind};
use crate::data::store::{AppEvent, Store};
use crate::screens::explore::{ExploreEvent, ExploreVm};
use crate::screens::explore::trace::trace_panel::color;
use crate::ui::widgets::{icon_button, row, Edge, MenuSelect, PanelHeader, Section};

/// The focused view's source id (else the recording's default source).
fn focused_source(explore: &Entity<ExploreVm>, cx: &App) -> Option<String> {
    let ex = explore.read(cx);
    match &ex.focused {
        Some(f) => Some(f.read(cx).view.source.clone()),
        None => ex.store().read(cx).sources().map(|s| s.default_entry().id.clone()),
    }
}

/// Opens and closes the side a panel lives in (the panel's hide button).
#[derive(Clone)]
pub struct DockToggle {
    pub placement: DockPlacement,
    pub open: Rc<dyn Fn(&App) -> bool>,
    pub toggle: Rc<dyn Fn(&mut Window, &mut App)>,
}

impl DockToggle {
    pub fn is_open(&self, cx: &App) -> bool {
        (self.open)(cx)
    }

    pub fn toggle(&self, window: &mut Window, cx: &mut App) {
        (self.toggle)(window, cx)
    }

    pub fn edge(&self) -> Edge {
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

pub(crate) use side_panel;

// ----------------------------------------------------------------------------
// Channels
// ----------------------------------------------------------------------------

pub struct ChannelsPanel {
    explore: Entity<ExploreVm>,
    dock: DockToggle,
    focus: FocusHandle,
    /// One field: a number or site filters the list; a range (`0-31, 40`) + Enter shows those.
    filter: Entity<InputState>,
    range_error: Option<String>,
    /// Groups unfolded, by index (the first one when a recording opens).
    expanded: std::collections::HashSet<usize>,
    _subs: Vec<Subscription>,
}

/// Channels per group when the recording has one group (or no layout): chunks of this many.
const CHANNEL_BLOCK: usize = 32;
/// Row height (8 pt grid).
const ROW_H: f32 = 24.;

/// A row of the channel tree.
#[derive(Debug, Clone, Copy)]
enum TreeRow {
    /// Group `g`: channels shown of it, of all, and whether it is unfolded.
    Group { g: usize, on: usize, of: usize, open: bool },
    Channel(usize),
}

side_panel!(ChannelsPanel, "dspk-app.channels", "Channels");

/// Whether `text` is a range expression (`0-31, 40`) rather than a filter.
fn is_range(text: &str) -> bool {
    text.contains('-') || text.contains(',')
}

impl ChannelsPanel {
    pub fn new(explore: Entity<ExploreVm>, dock: DockToggle, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let filter = cx.new(|cx| InputState::new(window, cx).placeholder("Find, or show 0-31, 40 ⏎"));
        let store = explore.read(cx).store().clone();
        let subs = vec![
            cx.observe(&explore, |_, _, cx| cx.notify()),
            cx.subscribe(&explore, |_, _, _: &ExploreEvent, cx| cx.notify()),
            cx.subscribe(&store, |this, _, e: &AppEvent, cx| {
                if matches!(e, AppEvent::RecordingChanged) {
                    this.expanded = [0].into_iter().collect();
                }
                if matches!(e, AppEvent::SelectionChanged(_) | AppEvent::RecordingChanged | AppEvent::Picked) {
                    cx.notify();
                }
            }),
            cx.subscribe(&filter, |this, state, e: &InputEvent, cx| match e {
                InputEvent::Change => {
                    this.range_error = None;
                    cx.notify();
                }
                InputEvent::PressEnter { .. } => {
                    let text = state.read(cx).value().to_string();
                    if is_range(&text) {
                        this.apply_ranges(&text, cx);
                    }
                }
                _ => {}
            }),
        ];
        Self { explore, dock, focus: cx.focus_handle(), filter, range_error: None, expanded: [0].into_iter().collect(), _subs: subs }
    }

    fn store(&self, cx: &App) -> Entity<Store> {
        self.explore.read(cx).store().clone()
    }

    fn pinned_vm(&self, cx: &App) -> Option<Entity<crate::screens::explore::TraceVm>> {
        self.explore.read(cx).focused.clone().filter(|vm| vm.read(cx).view.pinned_selection)
    }

    fn apply_ranges(&mut self, text: &str, cx: &mut Context<Self>) {
        if let Some(vm) = self.pinned_vm(cx) {
            self.range_error = vm.update(cx, |vm, cx| vm.select_ranges(text, cx)).err();
            cx.notify();
            return;
        }
        let Some(source) = focused_source(&self.explore, cx) else { return };
        let result = self.store(cx).update(cx, |s, cx| s.select_ranges(&source, text, cx));
        self.range_error = result.err();
        cx.notify();
    }

    fn select_all(&self, cx: &mut Context<Self>) {
        if let Some(vm) = self.pinned_vm(cx) {
            vm.update(cx, |vm, cx| vm.select_all(cx));
        } else if let Some(source) = focused_source(&self.explore, cx) {
            self.store(cx).update(cx, |s, cx| s.select_all(&source, cx));
        }
    }

    fn select_none(&self, cx: &mut Context<Self>) {
        if let Some(vm) = self.pinned_vm(cx) {
            vm.update(cx, |vm, cx| vm.select_none(cx));
        } else if let Some(source) = focused_source(&self.explore, cx) {
            self.store(cx).update(cx, |s, cx| s.select_none(&source, cx));
        }
    }

    fn select_invert(&self, cx: &mut Context<Self>) {
        if let Some(vm) = self.pinned_vm(cx) {
            vm.update(cx, |vm, cx| vm.select_invert(cx));
        } else if let Some(source) = focused_source(&self.explore, cx) {
            self.store(cx).update(cx, |s, cx| s.select_invert(&source, cx));
        }
    }
}

impl ChannelsPanel {
    /// `12 / 385` shown
    fn header_extra(&self, cx: &App) -> Option<gpui_kit::AnyElement> {
        let source = focused_source(&self.explore, cx)?;
        let s = self.store(cx);
        let s = s.read(cx);
        let total = s.sources()?.entry(&source).channels;
        let count = match self.pinned_vm(cx) {
            Some(vm) => vm.read(cx).view.selection.len(),
            None => s.selection(&source).len(),
        };
        Some(div().text_xs().text_color(cx.theme().muted_foreground).child(format!("{count} / {total}")).into_any_element())
    }

    /// Shown channels (of the pinned view, else of the source).
    fn shown(&self, source: &str, cx: &App) -> Vec<usize> {
        match self.pinned_vm(cx) {
            Some(vm) => vm.read(cx).view.selection.clone(),
            None => self.store(cx).read(cx).selection(source).to_vec(),
        }
    }

    fn set_shown(&self, source: &str, channels: Vec<usize>, cx: &mut App) {
        match self.pinned_vm(cx) {
            Some(vm) => vm.update(cx, |vm, cx| vm.set_selection(channels, cx)),
            None => self.store(cx).update(cx, |s, cx| s.set_selection(source, channels, cx)),
        }
    }

    fn body(&mut self, _: &mut Window, cx: &mut Context<Self>) -> gpui_kit::AnyElement {
        let store = self.store(cx);
        let muted = cx.theme().muted_foreground;
        let Some(source) = focused_source(&self.explore, cx) else {
            return v_flex().size_full().p_4().text_sm().text_color(muted).child("No recording open.").into_any_element();
        };
        let pinned = self.pinned_vm(cx).is_some();
        let focused_vm = self.explore.read(cx).focused.clone();
        let (total, ds, picked) = {
            let s = store.read(cx);
            let ds = s.sources().map(|ss| ss.get(&source));
            (s.sources().map_or(0, |ss| ss.entry(&source).channels), ds, s.picked_channel)
        };
        let shown_list = self.shown(&source, cx);
        let shown: Rc<Vec<bool>> = Rc::new({
            let mut v = vec![false; total];
            for &c in &shown_list {
                if c < total {
                    v[c] = true;
                }
            }
            v
        });
        let text = self.filter.read(cx).value().trim().to_lowercase();
        let filtering = !text.is_empty() && !is_range(&text);
        let layout = ds.as_ref().and_then(|d| d.layout.clone());
        let labels: Rc<Vec<Option<String>>> = Rc::new((0..total).map(|c| layout.as_ref().and_then(|l| l.label(c).map(str::to_string))).collect());
        let matches = |c: usize| !filtering || c.to_string().contains(&text) || labels[c].as_ref().is_some_and(|s| s.to_lowercase().contains(&text));
        // Groups: the layout's (shanks) when it has several, else blocks of 32 channels
        let groups: Vec<(String, Vec<usize>)> = match &layout {
            Some(l) if l.groups.len() > 1 => l.groups.iter().map(|g| (format!("{} {}", l.group_kind, g.name), g.channels.iter().copied().filter(|&c| c < total).collect())).collect(),
            _ => (0..total).step_by(CHANNEL_BLOCK).map(|a| {
                let b = (a + CHANNEL_BLOCK).min(total);
                (format!("{a}–{}", b - 1), (a..b).collect())
            }).collect(),
        };
        let mut rows: Vec<TreeRow> = Vec::new();
        for (g, (_, chans)) in groups.iter().enumerate() {
            let visible: Vec<usize> = chans.iter().copied().filter(|&c| matches(c)).collect();
            if visible.is_empty() {
                continue;
            }
            // A search unfolds what it finds
            let open = filtering || self.expanded.contains(&g);
            rows.push(TreeRow::Group { g, on: chans.iter().filter(|&&c| shown[c]).count(), of: chans.len(), open });
            if open {
                rows.extend(visible.into_iter().map(TreeRow::Channel));
            }
        }
        let t = cx.theme();
        let (accent, accent_fg, danger, fg) = (t.accent, t.accent_foreground, t.danger, t.foreground);
        let rows = Rc::new(rows);
        let groups = Rc::new(groups);
        let this = cx.entity().downgrade();
        let (list_rows, list_labels, list_groups, src) = (rows.clone(), labels.clone(), groups.clone(), source.clone());

        let list = uniform_list("channel-rows", rows.len(), move |range, _, _| {
            range
                .map(|i| {
                    let (this, src) = (this.clone(), src.clone());
                    match list_rows[i] {
                        TreeRow::Group { g, on, of, open } => {
                            let chans = list_groups[g].1.clone();
                            let all = on == of;
                            let (t2, s2) = (this.clone(), src.clone());
                            h_flex()
                                .id(("channel-group", g))
                                .h(px(ROW_H + 4.))
                                .pl_2()
                                .pr_1()
                                .gap_2()
                                .cursor_pointer()
                                .child(gpui_kit::component::Icon::new(if open { IconName::ChevronDown } else { IconName::ChevronRight }).xsmall().text_color(muted))
                                .child(div().flex_1().text_sm().font_weight(gpui_kit::FontWeight::MEDIUM).child(list_groups[g].0.clone()))
                                .child(div().text_xs().text_color(muted).child(format!("{on}/{of}")))
                                .child(
                                    icon_button(("group-eye", g), if on == 0 { IconName::EyeOff } else { IconName::Eye }, if all { "Hide this group" } else { "Show this group" })
                                        .xsmall()
                                        .on_click(move |_, _, cx| {
                                            let _ = t2.update(cx, |p, cx| {
                                                let mut next: Vec<usize> = p.shown(&s2, cx).into_iter().filter(|c| !chans.contains(c)).collect();
                                                if !all {
                                                    next.extend(chans.iter().copied());
                                                    next.sort_unstable();
                                                }
                                                p.set_shown(&s2, next, cx);
                                            });
                                        }),
                                )
                                .on_click(move |_, _, cx| {
                                    let _ = this.update(cx, |p, cx| {
                                        if !p.expanded.remove(&g) {
                                            p.expanded.insert(g);
                                        }
                                        cx.notify();
                                    });
                                })
                                .into_any_element()
                        }
                        TreeRow::Channel(ch) => {
                            let on = shown[ch];
                            let is_picked = picked == Some(ch);
                            let label = list_labels[ch].clone();
                            let t2 = this.clone();
                            h_flex()
                                .id(("channel", ch))
                                .h(px(ROW_H))
                                .pl(px(32.))
                                .pr_1()
                                .gap_2()
                                .rounded_sm()
                                .cursor_pointer()
                                .when(is_picked, |d| d.bg(accent).text_color(accent_fg))
                                .child(div().w(px(36.)).text_sm().text_color(if on { fg } else { muted }).child(ch.to_string()))
                                .child(div().flex_1().min_w_0().truncate().text_xs().text_color(muted).children(label))
                                .child(
                                    icon_button(("channel-eye", ch), if on { IconName::Eye } else { IconName::EyeOff }, if on { "Hide" } else { "Show" })
                                        .xsmall()
                                        .on_click(move |_, _, cx| {
                                            let _ = t2.update(cx, |p, cx| {
                                                let mut next = p.shown(&src, cx);
                                                match next.iter().position(|&c| c == ch) {
                                                    Some(i) => {
                                                        next.remove(i);
                                                    }
                                                    None => {
                                                        let at = next.partition_point(|&c| c < ch);
                                                        next.insert(at, ch);
                                                    }
                                                }
                                                p.set_shown(&src, next, cx);
                                            });
                                        }),
                                )
                                .on_click(move |_, _, cx| {
                                    let _ = this.update(cx, |p, cx| p.store(cx).update(cx, |s, cx| s.pick_channel(Some(ch), cx)));
                                })
                                .into_any_element()
                        }
                    }
                })
                .collect()
        })
        .flex_1()
        .min_h_0();

        // Every selection action in one menu (few visible choices)
        let menu = {
            let this = cx.entity().downgrade();
            let pin_vm = focused_vm.clone();
            Button::new("channels-menu").ghost().xsmall().icon(IconName::Ellipsis).tooltip("Show all, none, invert; pin to the view").dropdown_menu(move |menu: gpui_kit::component::menu::PopupMenu, _, _| {
                let item = |label: &'static str, f: fn(&ChannelsPanel, &mut Context<ChannelsPanel>)| {
                    let this = this.clone();
                    gpui_kit::component::menu::PopupMenuItem::new(label).on_click(move |_, _, cx| {
                        let _ = this.update(cx, |p, cx| f(p, cx));
                    })
                };
                let mut menu = menu.item(item("Show all", |p, cx| p.select_all(cx))).item(item("Show none", |p, cx| p.select_none(cx))).item(item("Invert", |p, cx| p.select_invert(cx)));
                if let Some(vm) = pin_vm.clone() {
                    menu = menu.separator().item(gpui_kit::component::menu::PopupMenuItem::new(if pinned { "Unpin from this view" } else { "Pin to this view" }).on_click(move |_, _, cx| {
                        vm.update(cx, |vm, cx| vm.set_pinned_selection(!pinned, cx))
                    }));
                }
                menu
            })
        };

        v_flex()
            .id("channels-panel")
            .test_support()
            .size_full()
            .gap_2()
            .p_2()
            .child(h_flex().gap_1().child(div().flex_1().child(Input::new(&self.filter).small().prefix(gpui_kit::component::Icon::new(IconName::Search).small()))).child(menu))
            .when_some(self.range_error.clone(), |d, e| d.child(div().text_xs().text_color(danger).child(e)))
            .when(pinned, |d| d.child(div().text_xs().text_color(muted).child("Pinned to the focused view")))
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

side_panel!(SettingsPanel, "dspk-app.view-settings", "Display");

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

        // Layers: what the view draws (one at a time)
        let layer = |id: &'static str, icon: gpui_kit::assets::IconName, label: &'static str, tip: &'static str, on: bool, kind: TimeViewKind, spikes: bool| {
            let vm = vm.clone();
            Button::new(id).xsmall().icon(icon).label(label).tooltip(tip).map(|b| if on { b.primary() } else { b.ghost() }).on_click(move |_, _, cx| {
                vm.update(cx, |vm, cx| {
                    vm.set_kind(kind, cx);
                    if kind == TimeViewKind::Traces {
                        vm.set_spikes(spikes, cx);
                    }
                })
            })
        };
        let kind = h_flex()
            .gap_1()
            .flex_wrap()
            .child(layer("kind-traces", gpui_kit::assets::IconName::AudioWaveform, "Traces", "The traces alone", traces && !v.spikes, TimeViewKind::Traces, false))
            .child(layer("kind-spikes", gpui_kit::assets::IconName::Activity, "Spikes", "Traces faint, each spike in its unit's colour (a linked sorting)", traces && v.spikes, TimeViewKind::Traces, true))
            .child(layer("kind-heatmap", gpui_kit::assets::IconName::Flame, "Heatmap", "Every channel as a row, colour = amplitude", v.kind == TimeViewKind::Heatmap, TimeViewKind::Heatmap, false))
            .child(layer("kind-raster", gpui_kit::assets::IconName::GitCommitVertical, "Raster", "Spike times, one row per unit (a linked sorting)", v.kind == TimeViewKind::Raster, TimeViewKind::Raster, false));
        let color_by = {
            let vm = vm.clone();
            let names: Vec<SharedString> = ColorBy::ALL.iter().map(|c| SharedString::from(c.name())).collect();
            let sel = ColorBy::ALL.iter().position(|c| *c == v.color_by);
            MenuSelect::new("color-by", names, sel, move |i, _, cx| vm.update(cx, |vm, cx| vm.set_color_by(ColorBy::ALL[i], cx)))
        };

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
            .child(div().text_sm().font_weight(gpui_kit::FontWeight::SEMIBOLD).truncate().child(v.title.clone()))
            .child(Section::new("Layer").child(kind).when(traces, |d| d.child(row("Colour", color_by, cx))))
            .child(Section::new("Source").children(source).child(row("Range", div().text_sm().child(v.range_label()), cx)))
            .when(traces, |d| d.child(Section::new("Traces").child(row("On screen", lanes_row, cx)).child(row("Gain", gain_row, cx))))
            .child(Section::new("Scaling").child(auto).child(dc))
            .into_any_element()
    }
}

/// The open recording's default source.
fn s_sources_default(store: &Entity<Store>, cx: &App) -> Option<std::sync::Arc<crate::data::recording::Dataset>> {
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

side_panel!(TimelinePanel, "dspk-app.timeline", "Timeline");

impl TimelinePanel {
    pub fn new(store: Entity<Store>, dock: DockToggle, cx: &mut Context<Self>) -> Self {
        let sub = cx.subscribe(&store, |this, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::RecordingChanged | AppEvent::PyramidProgress(_) | AppEvent::PaletteChanged) {
                this.activity = None;
            }
            if matches!(e, AppEvent::WindowMoved | AppEvent::PlaybackChanged | AppEvent::RecordingChanged | AppEvent::PyramidProgress(_) | AppEvent::PaletteChanged) {
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
        let summarizing = s.pyramid_label().map(|l| div().id("building-overview").text_xs().text_color(cx.theme().primary).child(l));
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
            let values = s_sources_default(&self.store, cx).map(|ds| crate::data::recording::dataset::activity(&ds, columns)).unwrap_or_default();
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
                    .child(crate::data::store::duration(tl.visible_window_sec)),
            )
            .child(div().absolute().top_0().bottom_0().left(relative(p)).w(px(2.)).bg(color(self.store.read(cx).palette().playhead)));
        let ruler = h_flex().justify_between().children((0..=4).map(|q| div().text_xs().text_color(muted).child(crate::data::store::duration(total * q as f64 / 4.0))));

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
