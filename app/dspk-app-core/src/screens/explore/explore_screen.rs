//! The Explore workspace, laid out as Zed / VS Code: the time views in the centre (a dock: tabs
//! and splits the user rearranges by dragging tabs); an icon bar at each side edge that shows
//! one panel at a time on that side (left: Channels; right: Processing, Display, and the
//! plugins' panels such as Sorting), clicking the shown panel's icon hides the side; the
//! Timeline fixed at the bottom (minimised to its header, never shared).

use std::collections::HashMap;
use std::rc::Rc;

use gpui_kit::base::dock::{DockArea, DockLayout, DockPlacement, InsertTarget};
use gpui_kit::base::Placement;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::dock::{panel_handle, DockSkin};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Icon, Selectable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{div, px, AnyView, App, AppContext as _, Context, Entity, IntoElement, ParentElement as _, Render, SharedString, Styled as _, Subscription, WeakEntity, Window};

use crate::data::store::{AppEvent, Store};
use crate::screens::explore::trace::renderer::TimeViewKind;
use crate::screens::explore::trace::view::ViewId;
use crate::screens::explore::{ExploreEvent, ExploreVm, TraceVm};
use crate::ui::widgets::HEADER_HEIGHT;

use super::components::panels::{ChannelsPanel, DockToggle, SettingsPanel, TimelinePanel};
use super::components::plugin_panel::PluginPanel;
use super::components::processing::ProcessingPanel;
use super::trace::trace_panel::TracePanel;

/// Widths of the side panels and their icon bars, and the open timeline's height (logical px).
const LEFT_W: f32 = 264.;
const RIGHT_W: f32 = 328.;
const RAIL_W: f32 = 40.;
const TIMELINE_H: f32 = 136.;

/// The panel shown on the right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RightPanel {
    Processing,
    Display,
    /// The i-th plugin panel.
    Plugin(usize),
}

impl RightPanel {
    fn key(self) -> String {
        match self {
            Self::Processing => "processing".into(),
            Self::Display => "display".into(),
            Self::Plugin(i) => format!("plugin{i}"),
        }
    }

    fn from_key(key: &str) -> Self {
        match key {
            "display" => Self::Display,
            k if k.starts_with("plugin") => k[6..].parse().map(Self::Plugin).unwrap_or(Self::Processing),
            _ => Self::Processing,
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
    processing: Entity<ProcessingPanel>,
    settings: Entity<SettingsPanel>,
    timeline: Entity<TimelinePanel>,
    /// Plugins' panels: title, icon, view.
    plugin_panels: Vec<(SharedString, Rc<dyn Fn() -> Icon>, Entity<PluginPanel>)>,
    left_open: bool,
    /// The right panel shown, and the last one (an icon click on a hidden side shows it again).
    right: Option<RightPanel>,
    last_right: RightPanel,
    timeline_open: bool,
    _subs: Vec<Subscription>,
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
        let settings = cx.new(|cx| SettingsPanel::new(vm.clone(), toggle_of(me.clone(), DockPlacement::Right), cx));
        let processing = cx.new(|cx| ProcessingPanel::new(vm.clone(), toggle_of(me.clone(), DockPlacement::Right), cx));
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
        });
        // Picking a channel shows its processing
        let picked = cx.subscribe(&store, |this, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::Picked) && this.right != Some(RightPanel::Processing) {
                this.show_right(RightPanel::Processing, cx);
            }
        });
        let mut this = Self {
            vm,
            dock,
            panels: HashMap::new(),
            store,
            channels,
            processing,
            settings,
            timeline,
            plugin_panels,
            left_open: saved.left_open,
            right: saved.right_open.then_some(last_right),
            last_right,
            timeline_open: saved.bottom_open,
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

    /// A new view in its own group beside the focused view's group (to its right, or under it).
    fn add(&mut self, vm: Entity<TraceVm>, below: bool, window: &mut Window, cx: &mut Context<Self>) {
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
            RightPanel::Processing => self.processing.clone().into(),
            RightPanel::Display => self.settings.clone().into(),
            RightPanel::Plugin(i) => self.plugin_panels.get(i)?.2.clone().into(),
        })
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
                .child(rail_button("rail-channels", Icon::new(gpui_kit::assets::IconName::Rows3), "Channels".into(), self.left_open, move |w, cx| {
                    let _ = me.update(cx, |e, cx| e.toggle_dock(DockPlacement::Left, w, cx));
                }))
        };
        let mut items: Vec<(String, Icon, SharedString, RightPanel)> = vec![
            ("rail-processing".into(), Icon::new(gpui_kit::assets::IconName::Workflow), "Processing".into(), RightPanel::Processing),
            ("rail-display".into(), Icon::new(gpui_kit::assets::IconName::SlidersHorizontal), "Display".into(), RightPanel::Display),
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
        v_flex()
            .size_full()
            .child(
                h_flex()
                    .flex_1()
                    .min_h_0()
                    .child(left_rail)
                    .when(self.left_open, |d| d.child(div().w(px(LEFT_W)).flex_none().h_full().border_r_1().border_color(border).child(self.channels.clone())))
                    .child(div().flex_1().min_w_0().h_full().child(self.dock.clone()))
                    .when_some(right_panel, |d, p| d.child(div().w(px(RIGHT_W)).flex_none().h_full().border_l_1().border_color(border).child(p)))
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
