//! The Explore workspace's views: which time views exist, which one is focused (the one the
//! settings panel edits), and where a new one goes. The dock decides where they sit; this decides
//! what they are.

use gpui_kit::{App, AppContext as _, Context, Entity, EventEmitter, Subscription};

use crate::screens::explore::trace::renderer::TimeViewKind;
use crate::screens::explore::trace::view::ViewId;
use crate::data::store::{AppEvent, Store};

use super::trace::trace_vm::{TraceEvent, TraceVm};

pub enum ExploreEvent {
    /// A recording opened: these views replace every view (default layout: traces over heatmap).
    Reset(Vec<Entity<TraceVm>>),
    /// A view to place beside the focused one (`below`: under it, else to its right).
    Added { vm: Entity<TraceVm>, below: bool },
    /// The focused view changed, or one of its settings did.
    Focus,
    /// A view to take out of the dock (its header menu's Close).
    Close(ViewId),
}

pub struct ExploreVm {
    store: Entity<Store>,
    pub views: Vec<Entity<TraceVm>>,
    pub focused: Option<Entity<TraceVm>>,
    next_id: ViewId,
    subs: Vec<(ViewId, Subscription)>,
    /// Views asked for recordings still loading beside the open one: (recording, kind).
    pending: Vec<(std::path::PathBuf, TimeViewKind)>,
    _store: Subscription,
}

impl EventEmitter<ExploreEvent> for ExploreVm {}

impl ExploreVm {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let sub = cx.subscribe(&store, |vm, _, event, cx| match event {
            AppEvent::RecordingChanged => vm.reset(cx),
            AppEvent::Document(crate::domain::command::Changed::Views) => vm.follow_active_view(cx),
            AppEvent::RecordingLoaded(path) => {
                let (now, later): (Vec<_>, Vec<_>) = std::mem::take(&mut vm.pending).into_iter().partition(|(p, _)| p == path);
                vm.pending = later;
                for (path, kind) in now {
                    vm.add_beside(kind, &path, cx);
                }
            }
            _ => {}
        });
        let mut vm = Self { store, views: Vec::new(), focused: None, next_id: 1, subs: Vec::new(), pending: Vec::new(), _store: sub };
        if vm.store.read(cx).recording.is_some() {
            vm.reset(cx);
        }
        vm
    }

    pub fn store(&self) -> &Entity<Store> {
        &self.store
    }

    fn make(&mut self, kind: TimeViewKind, source: &str, saved: Option<&crate::data::preferences::SavedTimeView>, cx: &mut Context<Self>) -> Entity<TraceVm> {
        let id = self.next_id;
        self.next_id += 1;
        let store = self.store.clone();
        let source = source.to_string();
        let saved = saved.cloned();
        let vm = cx.new(|cx| {
            let mut vm = TraceVm::new(store, id, kind, &source, cx);
            if let Some(s) = saved {
                s.apply(&mut vm.view);
            }
            vm
        });
        let sub = cx.subscribe(&vm, |this, vm, event, cx| match event {
            TraceEvent::Pressed => this.focus(vm, cx),
            TraceEvent::Released(id) => this.release(id.clone(), cx),
            TraceEvent::Changed => {
                this.persist_views(cx);
                if this.focused.as_ref() == Some(&vm) {
                    // Its source (so its session view) may have changed
                    this.select_view_of(&vm, cx);
                    cx.emit(ExploreEvent::Focus);
                }
            }
        });
        self.subs.push((id, sub));
        self.views.push(vm.clone());
        vm
    }

    fn persist_views(&self, cx: &mut Context<Self>) {
        let saved: Vec<crate::data::preferences::SavedTimeView> = self.views.iter().map(|v| crate::data::preferences::SavedTimeView::from_view(&v.read(cx).view)).collect();
        self.store.update(cx, |s, _| s.set_explore_views(saved));
    }

    /// Default views on the default source: saved views from session, or traces over a heatmap.
    fn reset(&mut self, cx: &mut Context<Self>) {
        for vm in std::mem::take(&mut self.views) {
            vm.update(cx, |vm, cx| vm.close(cx));
        }
        self.subs.clear();
        let Some(sources) = self.store.read(cx).sources().cloned() else {
            // No recording: no views (Explore shows its empty state)
            self.focused = None;
            cx.emit(ExploreEvent::Reset(Vec::new()));
            cx.emit(ExploreEvent::Focus);
            return;
        };
        let default_source = sources.default_entry().id.clone();
        let known = |id: &str| sources.entries().iter().any(|e| e.id == id);
        let saved: Vec<_> = self.store.read(cx).session.explore.views.clone();
        let created: Vec<Entity<TraceVm>> = if saved.is_empty() {
            vec![self.make(TimeViewKind::Traces, &default_source, None, cx), self.make(TimeViewKind::Heatmap, &default_source, None, cx)]
        } else {
            saved
                .iter()
                .filter_map(|s| match &s.recording {
                    // A view of another recording: kept while that recording is loaded
                    Some(path) if self.store.read(cx).sources_of(path).is_some() => {
                        let vm = self.make(s.kind, &default_source, Some(s), cx);
                        let source = s.source.clone();
                        let path = path.clone();
                        vm.update(cx, |vm, cx| vm.pin_recording(&path, &source, cx)).then_some(vm)
                    }
                    Some(_) => None,
                    None => {
                        let src = if known(&s.source) { &s.source } else { &default_source };
                        Some(self.make(s.kind, src, Some(s), cx))
                    }
                })
                .collect()
        };
        self.focused = created.first().cloned();
        self.persist_views(cx);
        cx.emit(ExploreEvent::Reset(created));
        cx.emit(ExploreEvent::Focus);
    }

    /// A new view beside the focused one, on its source (heatmaps go under it, traces to its right).
    pub fn add(&mut self, kind: TimeViewKind, cx: &mut Context<Self>) -> Option<Entity<TraceVm>> {
        let sources = self.store.read(cx).sources().cloned()?;
        let source = self.focused.as_ref().map(|f| f.read(cx).view.source.clone()).unwrap_or_else(|| sources.default_entry().id.clone());
        Some(self.add_on(kind, &source, cx))
    }

    /// A view of `kind` of the recording at `path` beside the open recording's views (the open
    /// one stays active; time aligned by the placements). Loads `path` first when needed; with
    /// no recording open, opens it instead.
    pub fn show_beside(&mut self, kind: TimeViewKind, path: std::path::PathBuf, cx: &mut Context<Self>) {
        let open = self.store.read(cx).recording.as_ref().and_then(|r| r.path.clone());
        if open.is_none() || open.as_ref() == Some(&path) {
            self.store.update(cx, |s, cx| s.open(path, cx));
            return;
        }
        self.pending.push((path.clone(), kind));
        self.store.update(cx, |s, cx| s.load_beside(path, cx));
    }

    /// Adds the view of the loaded recording `path` (its default source).
    fn add_beside(&mut self, kind: TimeViewKind, path: &std::path::Path, cx: &mut Context<Self>) {
        let Some(active) = self.store.read(cx).sources().map(|s| s.default_entry().id.clone()) else { return };
        let Some(source) = self.store.read(cx).sources_of(path).map(|s| s.default_entry().id.clone()) else { return };
        let vm = self.make(kind, &active, None, cx);
        if !vm.update(cx, |vm, cx| vm.pin_recording(path, &source, cx)) {
            return;
        }
        self.persist_views(cx);
        cx.emit(ExploreEvent::Added { vm: vm.clone(), below: true });
        self.focus(vm, cx);
    }

    /// A new view of `kind` on `source`, beside the focused one.
    pub fn add_on(&mut self, kind: TimeViewKind, source: &str, cx: &mut Context<Self>) -> Entity<TraceVm> {
        let vm = self.make(kind, source, None, cx);
        self.persist_views(cx);
        cx.emit(ExploreEvent::Added { vm: vm.clone(), below: kind == TimeViewKind::Heatmap });
        self.focus(vm.clone(), cx);
        vm
    }

    /// The active session view (picked in the Pipeline, made there, or chosen by an agent) is
    /// drawn here when no view of Explore draws it yet and it reads the open recording: by the
    /// focused view when it shows that source, else by the first view that does. Explore and the
    /// Pipeline then show the same chain (before, Explore kept the view it opened with, and a
    /// chain built in the Pipeline showed only after the app was started again).
    fn follow_active_view(&mut self, cx: &mut Context<Self>) {
        let store = self.store.read(cx);
        let Some(active) = store.doc.active() else { return };
        if self.views.iter().any(|v| v.read(cx).view.doc_view == active.id) {
            return;
        }
        let open = store.recording.as_ref().and_then(|r| r.path.clone());
        if open.as_deref() != Some(active.recording.path.as_path()) {
            return;
        }
        let (id, source) = (active.id.clone(), store.view_source(active));
        let reads = |v: &Entity<TraceVm>, cx: &App| v.read(cx).view.source == source && !v.read(cx).view.doc_view.is_empty();
        // Every view of that source drawing it unprocessed follows (traces and heatmap together);
        // a view drawing another chain keeps it, except the focused one
        let raw = |v: &Entity<TraceVm>, cx: &App| store.doc.view(&v.read(cx).view.doc_view).is_none_or(|d| d.ops.is_empty());
        let mut targets: Vec<Entity<TraceVm>> = self.views.iter().filter(|v| reads(v, cx) && raw(v, cx)).cloned().collect();
        if let Some(f) = self.focused.clone().filter(|v| reads(v, cx) && !targets.contains(v)) {
            targets.push(f);
        }
        if targets.is_empty() {
            targets.extend(self.views.iter().find(|v| reads(v, cx)).cloned());
        }
        for vm in targets {
            vm.update(cx, |vm, cx| vm.show_doc_view(&id, cx));
        }
    }

    /// `vm`'s session view becomes the active one (what Processing edits).
    fn select_view_of(&self, vm: &Entity<TraceVm>, cx: &mut Context<Self>) {
        let doc_view = vm.read(cx).view.doc_view.clone();
        if !doc_view.is_empty() && self.store.read(cx).doc.active_view.as_deref() != Some(doc_view.as_str()) {
            self.store.update(cx, |s, cx| {
                let _ = s.execute(crate::domain::command::SessionCommand::SelectView { view: doc_view }, cx);
            });
        }
    }

    /// Focuses `vm`: its session view becomes the active one (what Processing edits).
    pub fn focus(&mut self, vm: Entity<TraceVm>, cx: &mut Context<Self>) {
        self.select_view_of(&vm, cx);
        if self.store.read(cx).inspected.is_some() {
            self.store.update(cx, |s, cx| s.inspect(None, cx));
        }
        if self.focused.as_ref() != Some(&vm) {
            self.focused = Some(vm);
            cx.emit(ExploreEvent::Focus);
            cx.notify();
        }
    }

    /// The suggested layout: traces over a heatmap on the default source (saved views forgotten).
    pub fn reset_layout(&mut self, cx: &mut Context<Self>) {
        self.store.update(cx, |s, _| s.set_explore_views(Vec::new()));
        self.reset(cx);
    }

    /// Asks the dock to close view `id` (the dock then reports it [`Self::removed`]).
    pub fn close(&mut self, id: ViewId, cx: &mut Context<Self>) {
        cx.emit(ExploreEvent::Close(id));
    }

    /// A view left the dock (its tab was closed).
    pub fn removed(&mut self, id: ViewId, cx: &mut Context<Self>) {
        let Some(at) = self.views.iter().position(|v| v.read(cx).id() == id) else { return };
        let vm = self.views.remove(at);
        let doc_view = vm.read(cx).view.doc_view.clone();
        vm.update(cx, |vm, cx| vm.close(cx));
        if !doc_view.is_empty() {
            self.release(doc_view, cx);
        }
        self.subs.retain(|(v, _)| *v != id);
        if self.focused.as_ref() == Some(&vm) {
            self.focused = self.views.first().cloned();
            cx.emit(ExploreEvent::Focus);
        }
        self.persist_views(cx);
        cx.notify();
    }

    /// A view stopped showing session view `doc_view`: when no other view shows it and it holds
    /// no work, it leaves the session (the Pipeline loses its source and view nodes).
    fn release(&mut self, doc_view: String, cx: &mut Context<Self>) {
        if self.views.iter().any(|v| v.read(cx).view.doc_view == doc_view) {
            return;
        }
        self.store.update(cx, |s, cx| s.close_view_if_untouched(&doc_view, cx));
    }

    /// Double-click on a heatmap row: show `channel` in a traces view of the same source (a new one
    /// when none shows it).
    pub fn reveal_in_traces(&mut self, from: &Entity<TraceVm>, channel: usize, cx: &mut Context<Self>) {
        let source = from.read(cx).view.source.clone();
        let target = self.views.iter().find(|v| {
            let v = &v.read(cx).view;
            v.kind == TimeViewKind::Traces && v.source == source && v.selection.contains(&channel)
        });
        let target = match target {
            Some(t) => t.clone(),
            None => self.add_on(TimeViewKind::Traces, &source, cx),
        };
        target.update(cx, |vm, cx| vm.reveal(channel, cx));
        self.focus(target, cx);
    }
}
