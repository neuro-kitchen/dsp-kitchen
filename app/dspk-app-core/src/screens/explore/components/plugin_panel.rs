//! A plugin's panel on the right side (e.g. electrophysiology: Sorting): its header and the
//! plugin's section for the active view, redrawn when the session or the plugin's state changes.

use gpui_kit::base::dock::{Panel as BasePanel, PanelEvent};
use gpui_kit::component::dock::Panel;
use gpui_kit::component::{v_flex, ActiveTheme as _};
use gpui_kit::{div, App, Context, Entity, EventEmitter, FocusHandle, Focusable, InteractiveElement as _, IntoElement, ParentElement as _, Render, StatefulInteractiveElement as _, Styled as _, Subscription, Window};

use crate::data::store::{AppEvent, Store};
use crate::ui::widgets::PanelHeader;

use super::panels::DockToggle;

pub struct PluginPanel {
    store: Entity<Store>,
    /// Index of the plugin in the registry.
    plugin: usize,
    title: &'static str,
    dock: DockToggle,
    focus: FocusHandle,
    _sub: Subscription,
}

impl PluginPanel {
    pub fn new(store: Entity<Store>, plugin: usize, title: &'static str, dock: DockToggle, cx: &mut Context<Self>) -> Self {
        let sub = cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
            if !matches!(e, AppEvent::PlaybackChanged | AppEvent::PyramidProgress(_)) {
                cx.notify();
            }
        });
        Self { store, plugin, title, dock, focus: cx.focus_handle(), _sub: sub }
    }
}

impl Render for PluginPanel {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let toggle = self.dock.clone();
        let header = PanelHeader::new(gpui_kit::ElementId::Name(format!("plugin-panel-{}.toggle", self.plugin).into()), self.title, toggle.edge(), move |w, cx| toggle.toggle(w, cx));
        let muted = cx.theme().muted_foreground;
        let view = self.store.read(cx).active_view().cloned();
        let body = match (view, crate::plugin::plugins().get(self.plugin)) {
            (Some(view), Some(p)) => p.view_section(&self.store, &view, window, cx).unwrap_or_else(|| div().into_any_element()),
            _ => div().p_4().text_sm().text_color(muted).child("Open a recording first.").into_any_element(),
        };
        v_flex().size_full().bg(crate::ui::theme::color::panel()).child(header).child(div().id("plugin-panel-body").flex_1().min_h_0().overflow_y_scroll().p_3().child(body))
    }
}

impl EventEmitter<PanelEvent> for PluginPanel {}

impl Focusable for PluginPanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl BasePanel for PluginPanel {
    fn panel_name(&self) -> &'static str {
        "dspk-app.plugin-panel"
    }
}

impl Panel for PluginPanel {
    fn title(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.title
    }
}
