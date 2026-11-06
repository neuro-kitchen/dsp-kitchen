//! The Timeline tab of Explore's left side: the recordings on the session's time (each a bar on
//! one axis, at its placement; right-click: metadata, open, take off) and the epochs (by
//! recording, coloured by type; click: go there; right-click: edit, delete). `+` adds a recording
//! (after the last one) or an epoch (the visible window, to edit). Epochs are drawn on the bottom
//! timeline too, where Shift+drag marks a new one.

use std::path::PathBuf;

use gpui_kit::base::dock::{Panel as BasePanel, PanelEvent};
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::dock::Panel;
use gpui_kit::component::menu::{ContextMenuExt as _, DropdownMenu as _, PopupMenu, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Icon, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::TestSupportExt as _;
use gpui_kit::{
    div, px, relative, AnyElement, App, Context, Entity, EventEmitter, FocusHandle, Focusable, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _,
    PathPromptOptions, Render, StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};

use crate::data::store::{duration, file_label, AppEvent, Floating, Store};
use crate::domain::command::SessionCommand;
use crate::domain::epoch::{kind_hue, EpochDoc};
use crate::screens::explore::ExploreVm;
use crate::ui::widgets::PanelHeader;

use super::panels::{side_panel, DockToggle};

/// One recording on the session's time.
#[derive(Debug, Clone)]
struct Track {
    path: PathBuf,
    source: String,
    start: f64,
    length: f64,
    open: bool,
    placed: bool,
}

pub struct TimelineTab {
    store: Entity<Store>,
    dock: DockToggle,
    focus: FocusHandle,
    _subs: Vec<Subscription>,
}

side_panel!(TimelineTab, "dspk-app.timeline-tab", "Timeline");

/// The colour of an epoch type.
pub fn epoch_color(kind: &str) -> gpui_kit::Hsla {
    gpui_kit::hsla(kind_hue(kind) / 360.0, 0.65, 0.55, 1.0)
}

impl TimelineTab {
    pub fn new(explore: Entity<ExploreVm>, dock: DockToggle, cx: &mut Context<Self>) -> Self {
        let store = explore.read(cx).store().clone();
        let subs = vec![cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::Document(_) | AppEvent::RecordingChanged | AppEvent::WindowMoved) {
                cx.notify();
            }
        })];
        Self { store, dock, focus: cx.focus_handle(), _subs: subs }
    }

    fn header_extra(&self, cx: &App) -> Option<AnyElement> {
        let store = self.store.clone();
        let store2 = self.store.clone();
        let _ = cx;
        Some(
            Button::new("timeline-add")
                .ghost()
                .xsmall()
                .icon(IconName::Plus)
                .tooltip("Add a recording or an epoch")
                .dropdown_menu(move |m: PopupMenu, _, _| {
                    let (a, b) = (store.clone(), store2.clone());
                    m.item(PopupMenuItem::new("Recording…").on_click(move |_, _, cx| add_recording(&a, cx)))
                        .item(PopupMenuItem::new("Epoch (the visible window)").on_click(move |_, _, cx| new_epoch(&b, cx)))
                })
                .into_any_element(),
        )
    }

    /// The recordings the session knows: placed ones, the open one, the views' others.
    fn tracks(&self, cx: &App) -> Vec<Track> {
        let s = self.store.read(cx);
        let open = s.recording.as_ref().and_then(|r| r.path.clone());
        let mut tracks: Vec<Track> = Vec::new();
        let mut add = |path: PathBuf, source: String| {
            if tracks.iter().any(|t| t.path == path) {
                return;
            }
            let placed = s.doc.placements.iter().find(|p| p.path == path).map(|p| p.start_sec);
            let is_open = open.as_ref() == Some(&path);
            let length = if is_open { s.sources().map_or(0.0, |ss| ss.extent_sec()) } else { 0.0 };
            tracks.push(Track { path, source, start: placed.unwrap_or(0.0), length, open: is_open, placed: placed.is_some() });
        };
        for p in &s.doc.placements {
            add(p.path.clone(), p.source.clone().unwrap_or_default());
        }
        if let Some(p) = &open {
            add(p.clone(), s.sources().map(|ss| ss.default_entry().id.clone()).unwrap_or_default());
        }
        for v in &s.doc.views {
            add(v.recording.path.clone(), v.recording.source.clone().unwrap_or_default());
        }
        tracks
    }

    fn body(&mut self, _: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = cx.theme();
        let (muted, border, bar, accent) = (t.muted_foreground, t.border, t.primary, t.accent);
        let tracks = self.tracks(cx);
        let span = tracks.iter().map(|t| t.start + t.length.max(1.0)).fold(1.0f64, f64::max);
        let s = self.store.read(cx);
        let open = s.recording.as_ref().and_then(|r| r.path.clone());
        let window = (s.timeline.window_start_sec, s.timeline.visible_window_sec);
        let epochs: Vec<EpochDoc> = s.doc.epochs.clone();
        let rows = tracks.iter().enumerate().map(|(i, tr)| {
            let (a, w) = ((tr.start / span) as f32, ((tr.length.max(span * 0.01)) / span) as f32);
            let (store, store2, store3, store4) = (self.store.clone(), self.store.clone(), self.store.clone(), self.store.clone());
            let (p1, p2, p3, src) = (tr.path.clone(), tr.path.clone(), tr.path.clone(), tr.source.clone());
            v_flex()
                .id(("timeline-track", i))
                .p_1p5()
                .gap_1()
                .rounded_md()
                .when(tr.open, |d| d.bg(accent))
                .child(
                    h_flex()
                        .gap_1p5()
                        .child(Icon::new(gpui_kit::assets::IconName::AudioWaveform).xsmall().text_color(muted))
                        .child(div().flex_1().min_w_0().truncate().text_sm().child(file_label(&tr.path)))
                        .child(div().text_xs().text_color(muted).child(if tr.length > 0.0 { duration(tr.length) } else { "—".into() })),
                )
                .child(
                    div()
                        .relative()
                        .h(px(8.))
                        .w_full()
                        .rounded_sm()
                        .bg(border)
                        .child(div().absolute().top_0().bottom_0().left(relative(a)).w(relative(w.min(1.0 - a))).rounded_sm().bg(bar.opacity(if tr.open { 0.9 } else { 0.5 }))),
                )
                .child(div().text_xs().text_color(muted).child(if tr.placed { format!("at {} on the session's time", duration(tr.start)) } else { "not placed (right-click → Metadata to place)".to_string() }))
                .context_menu(move |m: PopupMenu, _, _| {
                    let (store, store2, store3, store4) = (store.clone(), store2.clone(), store3.clone(), store4.clone());
                    let (p1, p2, p3, src) = (p1.clone(), p2.clone(), p3.clone(), src.clone());
                    m.item(PopupMenuItem::new("Metadata…").on_click(move |_, _, cx| store.update(cx, |s, cx| s.open_floating(Floating::Metadata { path: p1.clone(), source: src.clone() }, cx))))
                        .item(PopupMenuItem::new("Open in Explore").on_click(move |_, _, cx| store2.update(cx, |s, cx| s.open(p2.clone(), cx))))
                        .item(PopupMenuItem::new("New epoch here").on_click(move |_, _, cx| new_epoch(&store4, cx)))
                        .separator()
                        .item(PopupMenuItem::new("Take off the timeline").on_click(move |_, _, cx| {
                            let _ = store3.update(cx, |s, cx| s.execute(SessionCommand::UnplaceRecording { path: p3.clone() }, cx));
                        }))
                })
        });
        // Epochs of each recording
        let mut list = v_flex().gap_0p5();
        let mut last: Option<PathBuf> = None;
        for (i, e) in epochs.iter().enumerate() {
            if last.as_ref() != Some(&e.recording) {
                list = list.child(div().pt_2().pb_0p5().px_1().text_xs().text_color(muted).truncate().child(file_label(&e.recording)));
                last = Some(e.recording.clone());
            }
            let here = open.as_ref() == Some(&e.recording);
            let inside = here && e.start_sec < window.0 + window.1 && e.end_sec > window.0;
            let (store, store2, store3, e1, e2) = (self.store.clone(), self.store.clone(), self.store.clone(), e.clone(), e.clone());
            let (rec, a, b) = (e.recording.clone(), e.start_sec, e.end_sec);
            list = list.child(
                h_flex()
                    .id(("timeline-epoch", i))
                    .h(px(26.))
                    .px_1p5()
                    .gap_2()
                    .rounded_md()
                    .cursor_pointer()
                    .when(inside, |d| d.bg(accent))
                    .child(div().size(px(8.)).flex_none().rounded_sm().bg(epoch_color(&e.kind)))
                    .child(div().flex_1().min_w_0().truncate().text_sm().child(e.name.clone()))
                    .child(div().text_xs().text_color(muted).child(format!("{} · {}", duration(e.start_sec), duration(e.duration_sec()))))
                    .on_click(move |_, _, cx| {
                        store.update(cx, |s, cx| {
                            let open = s.recording.as_ref().and_then(|r| r.path.clone());
                            if open.as_ref() != Some(&rec) {
                                s.open(rec.clone(), cx);
                            }
                            let pad = (b - a) * 0.1;
                            s.set_window(a - pad, b + pad, cx);
                        })
                    })
                    .context_menu(move |m: PopupMenu, _, _| {
                        let (store2, store3, e1, e2) = (store2.clone(), store3.clone(), e1.clone(), e2.clone());
                        m.item(PopupMenuItem::new("Edit…").on_click(move |_, _, cx| store2.update(cx, |s, cx| s.open_floating(Floating::Epoch(e1.clone()), cx))))
                            .item(PopupMenuItem::new("Delete").on_click(move |_, _, cx| {
                                let _ = store3.update(cx, |s, cx| s.execute(SessionCommand::RemoveEpoch { id: e2.id.clone() }, cx));
                            }))
                }),
            );
        }
        v_flex()
            .id("timeline-tab")
            .test_support()
            .size_full()
            .p_2()
            .gap_3()
            .overflow_y_scroll()
            .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(muted).child("RECORDINGS"))
            .children(rows)
            .when(tracks.is_empty(), |d| d.child(div().px_1().text_xs().text_color(muted).child("None yet: + → Recording…")))
            .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(muted).child("EPOCHS"))
            .when(epochs.is_empty(), |d| d.child(div().px_1().text_xs().text_color(muted).child("None yet: Shift+drag on the bottom timeline, or + → Epoch.")))
            .child(list)
            .into_any_element()
    }
}

/// Picks a recording and places it after the last one on the session's time.
fn add_recording(store: &Entity<Store>, cx: &mut App) {
    let rx = cx.prompt_for_paths(PathPromptOptions { files: true, directories: true, multiple: true, prompt: Some("Add to the timeline".into()) });
    let store = store.clone();
    cx.spawn(async move |cx| {
        if let Ok(Ok(Some(paths))) = rx.await {
            let _ = store.update(cx, |s, cx| {
                for path in paths {
                    let end = s.doc.placements.iter().map(|p| p.start_sec).fold(0.0f64, f64::max);
                    let length = if s.recording.as_ref().and_then(|r| r.path.as_ref()) == Some(&path) { s.sources().map_or(0.0, |ss| ss.extent_sec()) } else { 0.0 };
                    let start = if s.doc.placements.is_empty() { 0.0 } else { end + length.max(1.0) };
                    let _ = s.execute(SessionCommand::PlaceRecording { path: path.clone(), source: None, start_sec: start }, cx);
                    s.open_floating(Floating::Metadata { path, source: String::new() }, cx);
                }
            });
        }
    })
    .detach();
}

/// Opens the epoch editor on the visible window of the open recording.
pub fn new_epoch(store: &Entity<Store>, cx: &mut App) {
    store.update(cx, |s, cx| {
        let Some(path) = s.recording.as_ref().and_then(|r| r.path.clone()) else {
            s.set_status("Open a recording first: epochs mark a recording", cx);
            return;
        };
        let (a, w) = (s.timeline.window_start_sec.max(0.0), s.timeline.visible_window_sec);
        let epoch = EpochDoc { id: String::new(), recording: path, name: String::new(), kind: "stimulus".into(), start_sec: a, end_sec: a + w, notes: String::new() };
        s.open_floating(Floating::Epoch(epoch), cx);
    });
}
