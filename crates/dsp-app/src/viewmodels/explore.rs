//! The Explore workspace's views: which time views exist, which one is focused (the one the
//! settings panel edits), and where a new one goes. The dock decides where they sit; this decides
//! what they are.

use gpui_kit::{AppContext as _, Context, Entity, EventEmitter, Subscription};

use crate::engine::time::renderer::TimeViewKind;
use crate::engine::time::view::ViewId;
use crate::store::{AppEvent, Store};

use super::trace::{TraceEvent, TraceVm};

pub enum ExploreEvent {
    /// A recording opened: these views replace every view (default layout: traces over heatmap).
    Reset(Vec<Entity<TraceVm>>),
    /// A view to place beside the focused one (`below`: under it, else to its right).
    Added { vm: Entity<TraceVm>, below: bool },
    /// The focused view changed, or one of its settings did.
    Focus,
}

pub struct ExploreVm {
    store: Entity<Store>,
    pub views: Vec<Entity<TraceVm>>,
    pub focused: Option<Entity<TraceVm>>,
    next_id: ViewId,
    subs: Vec<(ViewId, Subscription)>,
    _store: Subscription,
}

impl EventEmitter<ExploreEvent> for ExploreVm {}

impl ExploreVm {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let sub = cx.subscribe(&store, |vm, _, event, cx| {
            if *event == AppEvent::RecordingChanged {
                vm.reset(cx);
            }
        });
        let mut vm = Self { store, views: Vec::new(), focused: None, next_id: 1, subs: Vec::new(), _store: sub };
        if vm.store.read(cx).recording.is_some() {
            vm.reset(cx);
        }
        vm
    }

    pub fn store(&self) -> &Entity<Store> {
        &self.store
    }

    fn make(&mut self, kind: TimeViewKind, source: &str, cx: &mut Context<Self>) -> Entity<TraceVm> {
        let id = self.next_id;
        self.next_id += 1;
        let store = self.store.clone();
        let source = source.to_string();
        let vm = cx.new(|cx| TraceVm::new(store, id, kind, &source, cx));
        let sub = cx.subscribe(&vm, |this, vm, event, cx| match event {
            TraceEvent::Pressed => this.focus(vm, cx),
            TraceEvent::Changed => {
                if this.focused.as_ref() == Some(&vm) {
                    cx.emit(ExploreEvent::Focus);
                }
            }
        });
        self.subs.push((id, sub));
        self.views.push(vm.clone());
        vm
    }

    /// Default views on the default source: traces over a heatmap.
    fn reset(&mut self, cx: &mut Context<Self>) {
        for vm in std::mem::take(&mut self.views) {
            vm.update(cx, |vm, cx| vm.close(cx));
        }
        self.subs.clear();
        let source = self.store.read(cx).sources().map(|s| s.default_entry().id.clone()).unwrap_or_default();
        let traces = self.make(TimeViewKind::Traces, &source, cx);
        let heatmap = self.make(TimeViewKind::Heatmap, &source, cx);
        self.focused = Some(traces.clone());
        cx.emit(ExploreEvent::Reset(vec![traces, heatmap]));
        cx.emit(ExploreEvent::Focus);
    }

    /// A new view beside the focused one, on its source (heatmaps go under it, traces to its right).
    pub fn add(&mut self, kind: TimeViewKind, cx: &mut Context<Self>) -> Option<Entity<TraceVm>> {
        let sources = self.store.read(cx).sources().cloned()?;
        let source = self.focused.as_ref().map(|f| f.read(cx).view.source.clone()).unwrap_or_else(|| sources.default_entry().id.clone());
        Some(self.add_on(kind, &source, cx))
    }

    /// A new view of `kind` on `source`, beside the focused one.
    pub fn add_on(&mut self, kind: TimeViewKind, source: &str, cx: &mut Context<Self>) -> Entity<TraceVm> {
        let vm = self.make(kind, source, cx);
        cx.emit(ExploreEvent::Added { vm: vm.clone(), below: kind == TimeViewKind::Heatmap });
        self.focus(vm.clone(), cx);
        vm
    }

    pub fn focus(&mut self, vm: Entity<TraceVm>, cx: &mut Context<Self>) {
        if self.focused.as_ref() != Some(&vm) {
            self.focused = Some(vm);
            cx.emit(ExploreEvent::Focus);
            cx.notify();
        }
    }

    /// A view left the dock (its tab was closed).
    pub fn removed(&mut self, id: ViewId, cx: &mut Context<Self>) {
        let Some(at) = self.views.iter().position(|v| v.read(cx).id() == id) else { return };
        let vm = self.views.remove(at);
        vm.update(cx, |vm, cx| vm.close(cx));
        self.subs.retain(|(v, _)| *v != id);
        if self.focused.as_ref() == Some(&vm) {
            self.focused = self.views.first().cloned();
            cx.emit(ExploreEvent::Focus);
        }
        cx.notify();
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
