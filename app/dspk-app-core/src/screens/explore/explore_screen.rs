//! The Explore workspace, laid out as Zed / VS Code: the time views in the centre (a dock: tabs
//! and splits the user rearranges by dragging tabs); an icon bar at each side edge that shows
//! one panel at a time on that side (left: Channels or Timeline; right: the Inspector and the
//! plugins' panels such as Sorting), clicking the shown panel's icon hides the side; the
//! Timeline fixed at the bottom (minimised to its header, never shared).

use std::collections::HashMap;
use std::rc::Rc;

use gpui_kit::base::dock::{DockArea, DockLayout, DockPlacement, InsertTarget};
use gpui_kit::base::Placement;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::TestSupportExt as _;
use gpui_kit::component::dock::{panel_handle, DockSkin};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Icon, IconName, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, px, AnyView, App, AppContext as _, Context, DragMoveEvent, Entity, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString,
    StatefulInteractiveElement as _, Styled as _, Subscription, WeakEntity, Window,
};

use crate::data::store::{AppEvent, Store};
use crate::screens::explore::trace::renderer::TimeViewKind;
use crate::screens::explore::trace::view::ViewId;
use crate::screens::explore::{ExploreEvent, ExploreVm, TraceVm};
use crate::ui::panel_edge::{follow_panel_drag, panel_edge, PanelResize, Side};
use crate::ui::widgets::HEADER_HEIGHT;

use super::components::inspector::InspectorPanel;
use super::components::panels::{ChannelsPanel, DockToggle, TimelinePanel};
use super::components::plugin_panel::PluginPanel;
use super::components::timeline_tab::TimelineTab;
use super::trace::trace_panel::TracePanel;

/// Widths of the side panels and their icon bars, and the open timeline's height (logical px).
const LEFT_W: f32 = 264.;
const RIGHT_W: f32 = 328.;
const RAIL_W: f32 = 40.;
const TIMELINE_H: f32 = 136.;

/// The panel shown on the right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RightPanel {
    /// The selected view (source, display, processing, output) or recording source.
    Inspector,
    /// The i-th plugin panel.
    Plugin(usize),
}

impl RightPanel {
    fn key(self) -> String {
        match self {
            Self::Inspector => "inspector".into(),
            Self::Plugin(i) => format!("plugin{i}"),
        }
    }

    fn from_key(key: &str) -> Self {
        match key {
            k if k.starts_with("plugin") => k[6..].parse().map(Self::Plugin).unwrap_or(Self::Inspector),
            _ => Self::Inspector,
        }
    }
}

pub struct ExploreView {
    pub vm: Entity<ExploreVm>,
    /// The centre: the time views.
    pub dock: Entity<DockArea>,
    pub panels: HashMap<ViewId, Entity<TracePanel>>,
    store: Entity<Store>,
    channels: Entity<ChannelsPanel>,
    timeline_tab: Entity<TimelineTab>,
    /// The left side shows the timeline tab (else Channels).
    left_timeline: bool,
    inspector: Entity<InspectorPanel>,
    timeline: Entity<TimelinePanel>,
    /// Plugins' panels: title, icon, view.
    plugin_panels: Vec<(SharedString, Rc<dyn Fn() -> Icon>, Entity<PluginPanel>)>,
    left_open: bool,
    /// The right panel shown, and the last one (an icon click on a hidden side shows it again).
    right: Option<RightPanel>,
    last_right: RightPanel,
    timeline_open: bool,
    /// "Add view or container" is open over the views.
    add_dialog: bool,
    _subs: Vec<Subscription>,
}

/// How [`ExploreView::arrange`] lays out the views (the dialog's containers: the dock has no empty
/// containers, so a container is a layout of the views there are).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arrangement {
    /// One tab group holding every view.
    Tabs,
    /// Side by side, left to right.
    Horizontal,
    /// Stacked, top to bottom.
    Vertical,
    /// Rows of up to ⌈√n⌉ views.
    Grid,
}

/// Opens / closes one side of `explore` (the panels' hide buttons).
fn toggle_of(explore: WeakEntity<ExploreView>, placement: DockPlacement) -> DockToggle {
    let e = explore.clone();
    DockToggle {
        placement,
        open: Rc::new(move |cx: &App| e.upgrade().is_some_and(|e| e.read(cx).is_side_open(placement))),
        toggle: Rc::new(move |window: &mut Window, cx: &mut App| {
            let _ = explore.update(cx, |e, cx| e.toggle_dock(placement, window, cx));
        }),
    }
}

impl ExploreView {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let vm = cx.new(|cx| ExploreVm::new(store.clone(), cx));
        let mut skin = None;
        let dock = cx.new(|cx| {
            let s = DockSkin::new(cx);
            skin = Some(s.clone());
            DockArea::new("explore", Some(1), window, cx).with_renderer(s)
        });
        if let Some(skin) = skin {
            skin.set_toggle_button_visible(false, cx);
        }
        let me = cx.weak_entity();
        let channels = cx.new(|cx| ChannelsPanel::new(vm.clone(), toggle_of(me.clone(), DockPlacement::Left), window, cx));
        let timeline_tab = cx.new(|cx| TimelineTab::new(vm.clone(), toggle_of(me.clone(), DockPlacement::Left), cx));
        let inspector = cx.new(|cx| InspectorPanel::new(vm.clone(), toggle_of(me.clone(), DockPlacement::Right), cx));
        let timeline = cx.new(|cx| TimelinePanel::new(store.clone(), toggle_of(me.clone(), DockPlacement::Bottom), cx));
        let plugin_panels = crate::plugin::plugins()
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                let (title, icon) = p.panel()?;
                let toggle = toggle_of(me.clone(), DockPlacement::Right);
                let panel = cx.new(|cx| PluginPanel::new(store.clone(), i, title, toggle, cx));
                Some((SharedString::from(title), icon, panel))
            })
            .collect();
        let saved = store.read(cx).session.explore.clone();
        let last_right = RightPanel::from_key(&saved.right_panel);
        let sub = cx.subscribe_in(&vm, window, |this, _, event, window, cx| match event {
            ExploreEvent::Reset(views) => this.reset(views.clone(), window, cx),
            ExploreEvent::Added { vm, below } => this.add(vm.clone(), *below, window, cx),
            ExploreEvent::Focus => {}
            ExploreEvent::Close(id) => {
                if let Some(panel) = this.panels.remove(id) {
                    this.dock.update(cx, |d, cx| d.remove_panel(panel, window, cx));
                }
                cx.notify();
            }
        });
        // Picking a channel shows its processing
        let picked = cx.subscribe(&store, |this, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::Picked) && this.right != Some(RightPanel::Inspector) {
                this.show_right(RightPanel::Inspector, cx);
            }
            if matches!(e, AppEvent::Document(_) | AppEvent::RecordingChanged) {
                cx.notify();
            }
        });
        let mut this = Self {
            vm,
            dock,
            panels: HashMap::new(),
            store,
            channels,
            timeline_tab,
            left_timeline: false,
            inspector,
            timeline,
            plugin_panels,
            left_open: saved.left_open,
            right: saved.right_open.then_some(last_right),
            last_right,
            timeline_open: saved.bottom_open,
            add_dialog: false,
            _subs: vec![sub, picked],
        };
        let views = this.vm.read(cx).views.clone();
        if !views.is_empty() {
            this.reset(views, window, cx);
        }
        this
    }

    /// Whether a side shows a panel (the bottom: the timeline open, not minimised).
    pub fn is_side_open(&self, placement: DockPlacement) -> bool {
        match placement {
            DockPlacement::Left => self.left_open,
            DockPlacement::Right => self.right.is_some(),
            _ => self.timeline_open,
        }
    }

    fn save_sides(&self, cx: &mut Context<Self>) {
        let (l, r, b, key) = (self.left_open, self.right.is_some(), self.timeline_open, self.last_right.key());
        self.store.update(cx, |s, _| {
            s.set_explore_docks(l, r, b);
            s.set_explore_right(&key);
        });
    }

    /// The right side shows `which`; when it shows it already, the side hides.
    pub fn show_right(&mut self, which: RightPanel, cx: &mut Context<Self>) {
        self.right = if self.right == Some(which) { None } else { Some(which) };
        self.last_right = which;
        self.save_sides(cx);
        cx.notify();
    }

    fn panel(&mut self, vm: Entity<TraceVm>, cx: &mut Context<Self>) -> Entity<TracePanel> {
        let id = vm.read(cx).id();
        let explore = self.vm.downgrade();
        let panel = cx.new(|cx| TracePanel::new(vm, explore, cx));
        self.panels.insert(id, panel.clone());
        panel
    }

    /// Traces over the heatmap (65 / 35).
    fn reset(&mut self, views: Vec<Entity<TraceVm>>, window: &mut Window, cx: &mut Context<Self>) {
        self.panels.clear();
        let panels: Vec<Entity<TracePanel>> = views.into_iter().map(|vm| self.panel(vm, cx)).collect();
        let mut layout = DockLayout::v_split();
        let n = panels.len();
        for (i, p) in panels.into_iter().enumerate() {
            let size = (n > 1 && i == 0).then(|| px(420.));
            layout = layout.child(DockLayout::tabs().panel_view(panel_handle(p), cx), size);
        }
        self.dock.update(cx, |d, cx| d.set_center(layout, window, cx));
    }

    /// Lays the views out as `how` (in the order they were added).
    pub fn arrange(&mut self, how: Arrangement, window: &mut Window, cx: &mut Context<Self>) {
        let order: Vec<ViewId> = self.vm.read(cx).views.iter().map(|v| v.read(cx).id()).collect();
        let panels: Vec<Entity<TracePanel>> = order.iter().filter_map(|id| self.panels.get(id).cloned()).collect();
        if panels.is_empty() {
            return;
        }
        let group = |p: &Entity<TracePanel>, cx: &App| DockLayout::tabs().panel_view(panel_handle(p.clone()), cx);
        let layout = match how {
            Arrangement::Tabs => panels.iter().fold(DockLayout::tabs(), |l, p| l.panel_view(panel_handle(p.clone()), cx)),
            Arrangement::Horizontal => panels.iter().fold(DockLayout::h_split(), |l, p| l.child(group(p, cx), None)),
            Arrangement::Vertical => panels.iter().fold(DockLayout::v_split(), |l, p| l.child(group(p, cx), None)),
            Arrangement::Grid => {
                let cols = (panels.len() as f64).sqrt().ceil().max(1.0) as usize;
                panels.chunks(cols).fold(DockLayout::v_split(), |l, row| l.child(row.iter().fold(DockLayout::h_split(), |r, p| r.child(group(p, cx), None)), None))
            }
        };
        self.dock.update(cx, |d, cx| d.set_center(layout, window, cx));
        cx.notify();
    }

    /// Opens or closes "Add view or container".
    pub fn set_add_dialog(&mut self, open: bool, cx: &mut Context<Self>) {
        if self.add_dialog != open {
            self.add_dialog = open;
            cx.notify();
        }
    }

    /// A new view in its own group beside the focused view's group (to its right, or under it).
    fn add(&mut self, vm: Entity<TraceVm>, below: bool, window: &mut Window, cx: &mut Context<Self>) {
        if self.panels.is_empty() {
            self.reset(vec![vm], window, cx);
            cx.notify();
            return;
        }
        let focused = self.vm.read(cx).focused.as_ref().and_then(|f| self.panels.get(&f.read(cx).id())).map(|p| p.entity_id());
        let panel = self.panel(vm, cx);
        let id = gpui_kit::base::dock::PanelId::from(panel.entity_id());
        self.dock.update(cx, |d, cx| {
            let target = focused.and_then(|f| d.layout(DockPlacement::Center)?.find_panel_node(gpui_kit::base::dock::PanelId::from(f)));
            d.add_panel_view(panel_handle(panel), DockPlacement::Center, None, window, cx);
            if let Some(node) = target {
                let placement = if below { Placement::Bottom } else { Placement::Right };
                d.move_panel(id, InsertTarget::Split { node, placement, size: None }, window, cx);
            }
        });
    }

    pub fn add_view(&mut self, kind: TimeViewKind, cx: &mut Context<Self>) {
        self.vm.update(cx, |vm, cx| {
            vm.add(kind, cx);
        });
    }

    /// The left side shows the timeline tab (`timeline`) or Channels; when it shows it already,
    /// the side hides.
    fn show_left(&mut self, timeline: bool, window: &mut Window, cx: &mut Context<Self>) {
        if self.left_open && self.left_timeline == timeline {
            self.toggle_dock(DockPlacement::Left, window, cx);
            return;
        }
        self.left_timeline = timeline;
        if !self.left_open {
            self.toggle_dock(DockPlacement::Left, window, cx);
        }
        cx.notify();
    }

    pub fn toggle_dock(&mut self, placement: DockPlacement, _window: &mut Window, cx: &mut Context<Self>) {
        match placement {
            DockPlacement::Left => self.left_open = !self.left_open,
            DockPlacement::Right => self.right = if self.right.is_some() { None } else { Some(self.last_right) },
            _ => self.timeline_open = !self.timeline_open,
        }
        self.save_sides(cx);
        cx.notify();
    }

    fn right_view(&self, which: RightPanel) -> Option<AnyView> {
        Some(match which {
            RightPanel::Inspector => self.inspector.clone().into(),
            RightPanel::Plugin(i) => self.plugin_panels.get(i)?.2.clone().into(),
        })
    }
}

impl ExploreView {
    /// "Add view…": opens "Add view or container".
    fn add_view_button(&self, id: &'static str, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let me = cx.entity().downgrade();
        Button::new(id).ghost().xsmall().icon(IconName::Plus).label("Add view…").tooltip("Add a view, or arrange the views").on_click(move |_, _, cx| {
            let _ = me.update(cx, |e, cx| e.set_add_dialog(true, cx));
        })
    }

    /// "Add view or container" (as Rerun's): containers arrange the views; a view kind adds a
    /// view beside the focused one, on its source (changed in the Inspector, which opens).
    fn add_dialog(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        use gpui_kit::assets::IconName as Assets;
        let t = cx.theme();
        let (muted, border, hover, popover, fg) = (t.muted_foreground, t.border, t.accent, t.popover, t.foreground);
        let me = cx.entity().downgrade();
        // A row: icon, title, one line of description; disabled rows say why
        let row = |id: String, icon: Assets, title: &'static str, text: String, enabled: bool, on: Rc<dyn Fn(&mut Window, &mut App)>| {
            h_flex()
                .id(SharedString::from(id))
                .test_support()
                .min_h(px(48.))
                .px_2()
                .gap_3()
                .rounded_md()
                .when(enabled, |d| d.cursor_pointer().hover(move |d| d.bg(hover)).on_click(move |_, window, cx| on(window, cx)))
                .when(!enabled, |d| d.opacity(0.5))
                .child(div().flex_none().size(px(32.)).rounded_md().border_1().border_color(border).flex().items_center().justify_center().child(Icon::new(icon).text_color(fg)))
                .child(v_flex().min_w_0().child(div().text_sm().font_weight(gpui_kit::FontWeight::MEDIUM).child(title)).child(div().text_xs().text_color(muted).truncate().child(text)))
        };
        let close = {
            let me = me.clone();
            move |_: &mut Window, cx: &mut App| {
                let _ = me.update(cx, |e, cx| e.set_add_dialog(false, cx));
            }
        };
        let arrange = |how: Arrangement| -> Rc<dyn Fn(&mut Window, &mut App)> {
            let me = me.clone();
            Rc::new(move |window, cx| {
                let _ = me.update(cx, |e, cx| {
                    e.arrange(how, window, cx);
                    e.set_add_dialog(false, cx);
                });
            })
        };
        let add = |kind: TimeViewKind| -> Rc<dyn Fn(&mut Window, &mut App)> {
            let me = me.clone();
            Rc::new(move |_, cx| {
                let _ = me.update(cx, |e, cx| {
                    e.set_add_dialog(false, cx);
                    e.vm.update(cx, |vm, cx| {
                        vm.add(kind, cx);
                    });
                    e.show_right(RightPanel::Inspector, cx);
                });
            })
        };
        // The focused view's source, and whether it has units (a raster needs them)
        let s = self.store.read(cx);
        let source = self.vm.read(cx).focused.as_ref().map(|f| f.read(cx).view.source.clone()).or_else(|| s.sources().map(|ss| ss.default_entry().id.clone()));
        let source_name = source.as_ref().and_then(|id| s.sources().map(|ss| ss.entry(id).name.clone())).unwrap_or_default();
        let units = s.active_view().is_some_and(|v| crate::plugin::layers(s, v, cx).units);
        let has_views = !self.vm.read(cx).views.is_empty();
        let containers = [
            (Arrangement::Tabs, Assets::AppWindow, "Tabs", "Every view as a tab of one group"),
            (Arrangement::Horizontal, Assets::Columns2, "Horizontal", "Every view side by side"),
            (Arrangement::Vertical, Assets::Rows2, "Vertical", "Every view stacked"),
            (Arrangement::Grid, Assets::LayoutGrid, "Grid", "Every view in a grid"),
        ];
        let on = |what: &str| if source_name.is_empty() { what.to_string() } else { format!("{what} of {source_name}") };
        let views = [
            (TimeViewKind::Traces, Assets::AudioWaveform, "Traces", on("The signals"), true),
            (TimeViewKind::Heatmap, Assets::Flame, "Heatmap", on("Amplitude by channel and time"), true),
            (TimeViewKind::Raster, Assets::ChartGantt, "Raster", if units { on("Spike times by unit") } else { "Needs a linked sorting (Sorting panel)".to_string() }, units),
        ];
        let heading = |text: &'static str, hint: &'static str| v_flex().px_2().pt_3().pb_1().child(div().text_xs().font_weight(gpui_kit::FontWeight::SEMIBOLD).child(text)).child(div().text_xs().text_color(muted).child(hint));
        let card = v_flex()
            .id("add-view-dialog")
            .test_support()
            .occlude()
            .w(px(440.))
            .max_h(px(620.))
            .overflow_y_scroll()
            .p_2()
            .rounded_lg()
            .border_1()
            .border_color(border)
            .bg(popover)
            .shadow_lg()
            .on_mouse_down(gpui_kit::MouseButton::Left, |_, _, cx| cx.stop_propagation())
            .child(
                h_flex()
                    .px_2()
                    .py_1()
                    .justify_between()
                    .child(div().text_sm().font_weight(gpui_kit::FontWeight::SEMIBOLD).child("Add view or container"))
                    .child(Button::new("add-view-close").ghost().small().icon(IconName::Close).tooltip("Close").on_click({
                        let close = close.clone();
                        move |_, w, cx| close(w, cx)
                    })),
            )
            .child(heading("Containers", "Arrange the views"))
            .children(containers.into_iter().map(|(how, icon, title, text)| {
                let id = format!("add-container-{}", title.to_lowercase());
                row(id, icon, title, text.into(), has_views, arrange(how))
            }))
            .child(div().mx_2().my_2().h(px(1.)).bg(border))
            .child(heading("Views", "Added beside the focused view; change its source in the Inspector"))
            .children(views.into_iter().map(|(kind, icon, title, text, enabled)| {
                let id = format!("add-view-kind-{}", title.to_lowercase());
                row(id, icon, title, text, enabled, add(kind))
            }))
            .child(div().mx_2().my_2().h(px(1.)).bg(border))
            .child({
                let (vm, store) = (self.vm.clone(), self.store.clone());
                let (me2, me3) = (me.clone(), me.clone());
                h_flex()
                    .px_1()
                    .gap_1()
                    .child(Button::new("add-view-reset").ghost().small().label("Reset to suggested layout").on_click(move |_, _, cx| {
                        vm.update(cx, |vm, cx| vm.reset_layout(cx));
                        let _ = me2.update(cx, |e, cx| e.set_add_dialog(false, cx));
                    }))
                    .child(Button::new("add-view-pipeline").ghost().small().label("New stream: branch in Pipeline…").on_click(move |_, _, cx| {
                        store.update(cx, |s, cx| s.set_workspace(crate::navigation::route::Workspace::Pipeline, cx));
                        let _ = me3.update(cx, |e, cx| e.set_add_dialog(false, cx));
                    }))
            });
        // The backdrop over the views closes it
        div()
            .id("add-view-backdrop")
            .test_support()
            .absolute()
            .inset_0()
            .bg(gpui_kit::black().opacity(0.35))
            .flex()
            .items_start()
            .justify_center()
            .pt(px(48.))
            .on_mouse_down(gpui_kit::MouseButton::Left, move |_, w, cx| close(w, cx))
            .child(card)
    }

    /// Above the views: what the focused view shows, and "Add view…".
    fn toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let t = cx.theme();
        let (muted, border) = (t.muted_foreground, t.border);
        let s = self.store.read(cx);
        let shown = s.active_view().map(|v| {
            let ops = v.ops.iter().filter(|o| o.enabled).count();
            format!("{} · {}", v.name, if ops == 0 { "raw".to_string() } else { format!("{ops} op{}", if ops == 1 { "" } else { "s" }) })
        });
        h_flex()
            .h(px(30.))
            .flex_none()
            .px_1()
            .gap_2()
            .border_b_1()
            .border_color(border)
            .child(self.add_view_button("explore-add-view", cx))
            .when_some(shown, |d, label| d.child(div().text_xs().text_color(muted).truncate().child(label)))
    }

    /// No views: say so, and offer "Add view…" (or opening a recording).
    fn empty_state(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let muted = cx.theme().muted_foreground;
        let has_recording = self.store.read(cx).recording.is_some();
        v_flex()
            .size_full()
            .items_center()
            .justify_center()
            .gap_2()
            .child(div().text_sm().text_color(muted).child(if has_recording { "No views" } else { "No recording open" }))
            .child(div().text_xs().text_color(muted).child(if has_recording { "Add a view of a source: traces, a heatmap, or a raster when units are linked." } else { "Open a recording (File → Open, or drop it here); it is listed under Timeline → Recordings." }))
            .when(has_recording, |d| d.child(self.add_view_button("explore-empty-add-view", cx)))
    }
}

/// An icon of a side bar: selected while its panel shows.
fn rail_button(id: impl Into<SharedString>, icon: Icon, tip: SharedString, selected: bool, on: impl Fn(&mut Window, &mut App) + 'static) -> Button {
    Button::new(gpui_kit::ElementId::Name(id.into())).ghost().icon(icon).tooltip(tip).selected(selected).on_click(move |_, window, cx| on(window, cx))
}

impl Render for ExploreView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let (border, bar) = (t.border, t.sidebar);
        let me = cx.weak_entity();
        let left_rail = {
            let me = me.clone();
            v_flex()
                .w(px(RAIL_W))
                .h_full()
                .py_2()
                .gap_1()
                .items_center()
                .bg(bar)
                .border_r_1()
                .border_color(border)
                .child(rail_button("rail-channels", Icon::new(gpui_kit::assets::IconName::Rows3), "Channels".into(), self.left_open && !self.left_timeline, {
                    let me = me.clone();
                    move |w, cx| {
                        let _ = me.update(cx, |e, cx| e.show_left(false, w, cx));
                    }
                }))
                .child(rail_button("rail-timeline", Icon::new(gpui_kit::assets::IconName::ChartGantt), "Timeline: recordings and epochs".into(), self.left_open && self.left_timeline, move |w, cx| {
                    let _ = me.update(cx, |e, cx| e.show_left(true, w, cx));
                }))
        };
        let mut items: Vec<(String, Icon, SharedString, RightPanel)> = vec![
            ("rail-inspector".into(), Icon::new(gpui_kit::assets::IconName::SlidersHorizontal), "Inspector: the selected view or source".into(), RightPanel::Inspector),
        ];
        for (i, (title, icon, _)) in self.plugin_panels.iter().enumerate() {
            items.push((format!("rail-plugin-{i}"), icon(), title.clone(), RightPanel::Plugin(i)));
        }
        let right_rail = v_flex().w(px(RAIL_W)).h_full().py_2().gap_1().items_center().bg(bar).border_l_1().border_color(border).children(items.into_iter().map(|(id, icon, tip, which)| {
            let me = me.clone();
            rail_button(id, icon, tip, self.right == Some(which), move |_, cx| {
                let _ = me.update(cx, |e, cx| e.show_right(which, cx));
            })
        }));
        let right_panel = self.right.and_then(|r| self.right_view(r));
        let (left_w, right_w) = {
            let s = self.store.read(cx);
            (s.panel_width("explore.left", LEFT_W), s.panel_width("explore.right", RIGHT_W))
        };
        v_flex()
            .size_full()
            .child(
                h_flex()
                    .flex_1()
                    .min_h_0()
                    .on_drag_move(cx.listener(|this, e: &DragMoveEvent<PanelResize>, _, cx| {
                        follow_panel_drag(&this.store, e, cx);
                        cx.notify();
                    }))
                    .child(left_rail)
                    .when(self.left_open, |d| {
                        d.child(
                            div()
                                .relative()
                                .w(px(left_w))
                                .flex_none()
                                .h_full()
                                .border_r_1()
                                .border_color(border)
                                .map(|d| if self.left_timeline { d.child(self.timeline_tab.clone()) } else { d.child(self.channels.clone()) })
                                .child(panel_edge("explore.left", Side::Left, RAIL_W)),
                        )
                    })
                    .child({
                        let empty = self.vm.read(cx).views.is_empty();
                        let dialog = self.add_dialog.then(|| self.add_dialog(cx));
                        v_flex()
                            .relative()
                            .flex_1()
                            .min_w_0()
                            .h_full()
                            .child(self.toolbar(cx))
                            .child(div().flex_1().min_h_0().map(|d| if empty { d.child(self.empty_state(cx)) } else { d.child(self.dock.clone()) }))
                            .children(dialog)
                    })
                    .when_some(right_panel, |d, p| {
                        d.child(
                            div()
                                .relative()
                                .w(px(right_w))
                                .flex_none()
                                .h_full()
                                .border_l_1()
                                .border_color(border)
                                .child(p)
                                .child(panel_edge("explore.right", Side::Right, RAIL_W)),
                        )
                    })
                    .child(right_rail),
            )
            // Fixed at the bottom: open, or minimised to its header; nothing else goes here
            .child(
                div()
                    .flex_none()
                    .h(px(if self.timeline_open { TIMELINE_H } else { HEADER_HEIGHT }))
                    .overflow_hidden()
                    .border_t_1()
                    .border_color(border)
                    .child(self.timeline.clone()),
            )
    }
}
