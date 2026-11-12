//! The Inspector, Explore's right panel: what is selected, in one place. A view: its source (the
//! one place to pick it, with what the source is), how it is drawn, its processing chain and its
//! output. A recording's source (clicked under Timeline → Recordings): what it holds.

use std::path::PathBuf;

use gpui_kit::base::dock::{Panel as BasePanel, PanelEvent};
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::dock::Panel;
use gpui_kit::component::{v_flex, ActiveTheme as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::TestSupportExt as _;
use gpui_kit::{
    div, AnyElement, App, AppContext as _, Context, Entity, EventEmitter, FocusHandle, Focusable, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render,
    StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};

use crate::data::store::{file_label, AppEvent, Floating, Store};
use crate::screens::explore::ExploreVm;
use crate::ui::widgets::{PanelHeader, Section};

use super::panels::{side_panel, source_facts, DockToggle, SettingsPanel};
use super::processing::ProcessingPanel;

pub struct InspectorPanel {
    store: Entity<Store>,
    dock: DockToggle,
    focus: FocusHandle,
    settings: Entity<SettingsPanel>,
    processing: Entity<ProcessingPanel>,
    _subs: Vec<Subscription>,
}

side_panel!(InspectorPanel, "dspk-app.inspector", "Inspector");

impl InspectorPanel {
    pub fn new(explore: Entity<ExploreVm>, dock: DockToggle, cx: &mut Context<Self>) -> Self {
        let store = explore.read(cx).store().clone();
        let settings = cx.new(|cx| SettingsPanel::new(explore.clone(), dock.clone(), cx));
        let processing = cx.new(|cx| ProcessingPanel::new(explore.clone(), dock.clone(), cx));
        let subs = vec![
            cx.observe(&settings, |_, _, cx| cx.notify()),
            cx.observe(&processing, |_, _, cx| cx.notify()),
            cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
                if matches!(e, AppEvent::Picked | AppEvent::Status | AppEvent::RecordingChanged | AppEvent::Document(_)) {
                    cx.notify();
                }
            }),
        ];
        Self { store, dock, focus: cx.focus_handle(), settings, processing, _subs: subs }
    }

    fn header_extra(&self, cx: &App) -> Option<AnyElement> {
        let s = self.store.read(cx);
        let label = match &s.inspected {
            Some((path, _)) => file_label(path),
            None => s.active_view()?.name.clone(),
        };
        Some(div().text_xs().text_color(cx.theme().muted_foreground).truncate().child(label).into_any_element())
    }

    /// A recording's source: what it holds (the loaded one's facts; Metadata… for the rest).
    fn source_sections(&self, path: &PathBuf, source: &str, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let muted = cx.theme().muted_foreground;
        let s = self.store.read(cx);
        let entry = s.recordings().find(|r| r.path.as_ref() == Some(path)).map(|r| r.sources.entry(source).clone());
        let (store, store2, p, src) = (self.store.clone(), self.store.clone(), path.clone(), source.to_string());
        vec![
            div().text_sm().font_weight(FontWeight::SEMIBOLD).truncate().child(file_label(path)).into_any_element(),
            Section::new("Source")
                .child(div().text_xs().text_color(muted).truncate().child(crate::domain::paths::short_path(&path)))
                .children(entry.as_ref().map(|e| div().text_sm().child(e.name.clone())))
                .children(entry.as_ref().map(|e| source_facts(e, muted)))
                .when_none(&entry, |d| d.child(div().text_xs().text_color(muted).child("Not loaded: open it to see its sources.")))
                .into_any_element(),
            v_flex()
                .gap_1()
                .child(Button::new("inspector-metadata").outline().small().label("Metadata and placement…").on_click(move |_, _, cx| {
                    store.update(cx, |s, cx| s.open_floating(Floating::Metadata { path: p.clone(), source: src.clone() }, cx))
                }))
                .child(Button::new("inspector-back").ghost().small().label("Back to the focused view").on_click(move |_, _, cx| store2.update(cx, |s, cx| s.inspect(None, cx))))
                .into_any_element(),
        ]
    }

    fn body(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let sections = match self.store.read(cx).inspected.clone() {
            Some((path, source)) => self.source_sections(&path, &source, cx),
            None => {
                let mut out = self.settings.update(cx, |p, cx| p.sections(window, cx));
                out.extend(self.processing.update(cx, |p, cx| p.sections(window, cx)));
                out
            }
        };
        v_flex().id("inspector-panel").test_support().size_full().gap_4().p_3().overflow_y_scroll().children(sections).into_any_element()
    }
}
