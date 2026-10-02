//! The Explore workspace: a dock with the time views in the centre (tabs and splits the user
//! rearranges by dragging tabs), Channels on the left, View settings on the right and the Timeline
//! at the bottom. Each side panel hides with the button at its window edge; a hidden one leaves a
//! rail at that edge to show it again (the timeline folds down to its header).

use std::collections::HashMap;

use gpui_kit::base::dock::{DockArea, DockEvent, DockLayout, DockPlacement, InsertTarget};
use gpui_kit::base::Placement;
use gpui_kit::component::dock::{panel_handle, DockSkin};
use gpui_kit::component::h_flex;
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{div, px, AppContext as _, Context, Entity, IntoElement, ParentElement as _, Render, Styled as _, Subscription, Window};

use crate::engine::time::renderer::TimeViewKind;
use crate::engine::time::view::ViewId;
use crate::store::Store;
use crate::viewmodels::{ExploreEvent, ExploreVm, TraceVm};

use super::panels::{ChannelsPanel, DockToggle, SettingsPanel, TimelinePanel};
use crate::widgets::{rail, Edge};
use super::trace::TracePanel;

pub struct ExploreView {
    pub vm: Entity<ExploreVm>,
    pub dock: Entity<DockArea>,
    pub panels: HashMap<ViewId, Entity<TracePanel>>,
    _subs: Vec<Subscription>,
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
        // Side panels carry their own hide button at the window edge (set once the area exists:
        // the skin redraws it)
        if let Some(skin) = skin {
            skin.set_toggle_button_visible(false, cx);
        }
        let toggle = |placement| DockToggle { dock: dock.downgrade(), placement };
        let channels = cx.new(|cx| ChannelsPanel::new(vm.clone(), toggle(DockPlacement::Left), window, cx));
        let settings = cx.new(|cx| SettingsPanel::new(vm.clone(), toggle(DockPlacement::Right), cx));
        let timeline = cx.new(|cx| TimelinePanel::new(store, toggle(DockPlacement::Bottom), cx));
        dock.update(cx, |d, cx| {
            // Panels go in through `panel_handle` so the skin draws their titles and controls
            d.set_dock(DockPlacement::Left, DockLayout::tabs().panel_view(panel_handle(channels), cx), window, cx);
            d.set_dock(DockPlacement::Right, DockLayout::tabs().panel_view(panel_handle(settings), cx), window, cx);
            d.set_dock(DockPlacement::Bottom, DockLayout::tabs().panel_view(panel_handle(timeline), cx), window, cx);
            d.set_dock_size(DockPlacement::Left, px(230.), window, cx);
            d.set_dock_size(DockPlacement::Right, px(280.), window, cx);
            d.set_dock_size(DockPlacement::Bottom, px(140.), window, cx);
        });
        let sub = cx.subscribe_in(&vm, window, |this, _, event, window, cx| match event {
            ExploreEvent::Reset(views) => this.reset(views.clone(), window, cx),
            ExploreEvent::Added { vm, below } => this.add(vm.clone(), *below, window, cx),
            ExploreEvent::Focus => {}
        });
        let layout = cx.subscribe(&dock, |_, _, e: &DockEvent, cx| {
            if matches!(e, DockEvent::LayoutChanged) {
                cx.notify();
            }
        });
        let mut this = Self { vm, dock, panels: HashMap::new(), _subs: vec![sub, layout] };
        let views = this.vm.read(cx).views.clone();
        if !views.is_empty() {
            this.reset(views, window, cx);
        }
        this
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

    pub fn toggle_dock(&mut self, placement: DockPlacement, window: &mut Window, cx: &mut Context<Self>) {
        self.dock.update(cx, |d, cx| d.toggle_dock(placement, window, cx));
    }
}

impl Render for ExploreView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let d = self.dock.read(cx);
        let (left, right) = (d.is_dock_open(DockPlacement::Left), d.is_dock_open(DockPlacement::Right));
        let show = |placement| {
            let dock = self.dock.downgrade();
            move |window: &mut Window, cx: &mut gpui_kit::App| {
                if let Some(dock) = dock.upgrade() {
                    dock.update(cx, |d, cx| d.toggle_dock(placement, window, cx));
                }
            }
        };
        h_flex()
            .size_full()
            .when(!left, |d| d.child(rail("show-channels", "Channels", Edge::Left, show(DockPlacement::Left), cx)))
            .child(div().flex_1().min_w_0().h_full().child(self.dock.clone()))
            .when(!right, |d| d.child(rail("show-settings", "View settings", Edge::Right, show(DockPlacement::Right), cx)))
    }
}
