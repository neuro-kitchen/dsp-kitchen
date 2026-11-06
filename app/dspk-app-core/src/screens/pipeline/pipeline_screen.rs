//! The Pipeline workspace: a menu bar (one menu per kind of operation, as in Photoshop), the
//! graph canvas in the middle, and on the right the operations tree (drag an operation onto the
//! canvas, or click it to add it after the selected node) over the inspector of the selection.
//! Every edit is a session command, so the canvas, Explore and an agent see the same graph.

use std::collections::HashSet;
use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenu, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Disableable as _, Icon, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::TestSupportExt as _;
use gpui_kit::{
    div, px, AnyElement, App, AppContext as _, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString,
    StatefulInteractiveElement as _, Styled as _, Subscription, WeakEntity, Window,
};

use crate::data::jobs::processed_path;
use crate::data::store::{AppEvent, Store};
use crate::domain::command::SessionCommand;
use crate::domain::graph::NodeKind;
use crate::domain::session::{DisplayDoc, Show};
use crate::screens::explore::components::processing::op_summary;
use crate::ui::param_editor::{Commit, ParamEditor};
use crate::ui::widgets::{row, Section};

use super::canvas::{CanvasEvent, CatalogDrag, DragGhost, GraphCanvas};
use super::catalog::{catalog, describe, Template};

/// Width of the right side (logical px).
const SIDE_W: f32 = 312.;

pub struct PipelineView {
    store: Entity<Store>,
    canvas: Entity<GraphCanvas>,
    inspector: Entity<Inspector>,
    /// Categories folded in the tree.
    folded: HashSet<&'static str>,
    _subs: Vec<Subscription>,
}

impl PipelineView {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let canvas = cx.new(|cx| GraphCanvas::new(store.clone(), cx));
        let inspector = cx.new(|cx| Inspector::new(store.clone(), canvas.clone(), window, cx));
        let subs = vec![cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::Document(_) | AppEvent::RecordingChanged | AppEvent::JobChanged) {
                cx.notify();
            }
        })];
        Self { store, canvas, inspector, folded: HashSet::new(), _subs: subs }
    }

    fn canvas_do(canvas: &WeakEntity<GraphCanvas>, cx: &mut App, f: impl FnOnce(&mut GraphCanvas, &mut Context<GraphCanvas>)) {
        let _ = canvas.update(cx, f);
    }

    /// The menu bar: File, Edit, one menu per kind of operation, View.
    fn menu_bar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let t = cx.theme();
        let canvas = self.canvas.downgrade();
        let selected = !self.canvas.read(cx).selected.is_empty();
        let writable = self.canvas.read(cx).single().and_then(|id| self.store.read(cx).doc.graph.node(id)).is_some_and(|n| matches!(n.kind, NodeKind::Write {} | NodeKind::View { .. }));
        let menu = |id: &'static str, title: &'static str, build: Rc<dyn Fn(PopupMenu) -> PopupMenu>| {
            Button::new(id).ghost().small().label(title).dropdown_menu(move |m: PopupMenu, _, _| build(m))
        };
        let item_canvas = canvas.clone();
        let item = move |label: &'static str, enabled: bool, f: fn(&mut GraphCanvas, &mut Window, &mut Context<GraphCanvas>)| {
            let canvas = item_canvas.clone();
            PopupMenuItem::new(label).disabled(!enabled).on_click(move |_, window, cx| {
                let _ = canvas.update(cx, |c, cx| f(c, window, cx));
            })
        };
        let file = {
            let item = item.clone();
            Rc::new(move |m: PopupMenu| {
                m.item(item("Add the open recording", true, |c, w, cx| c.add(&Template::OpenRecording, None, w, cx)))
                    .item(item("Add a recording from file…", true, |c, w, cx| c.add(&Template::RecordingFile, None, w, cx)))
                    .separator()
                    .item(item("Write the selected chain", writable, |c, _, cx| {
                        if let Some(id) = c.single().cloned() {
                            let store = c.store_entity();
                            store.update(cx, |s, cx| s.write_node(&id, cx));
                        }
                    }))
            }) as Rc<dyn Fn(PopupMenu) -> PopupMenu>
        };
        let edit = {
            let item = item.clone();
            Rc::new(move |m: PopupMenu| {
                m.item(item("Duplicate  (Ctrl+D)", selected, |c, _, cx| c.duplicate_selected(cx)))
                    .item(item("Delete  (Del)", selected, |c, _, cx| c.delete_selected(cx)))
                    .item(item("Disconnect input", selected, |c, _, cx| c.disconnect_selected(cx)))
                    .separator()
                    .item(item("Select all  (Ctrl+A)", true, |c, _, cx| c.select_all(cx)))
            }) as Rc<dyn Fn(PopupMenu) -> PopupMenu>
        };
        let view = {
            let item = item.clone();
            Rc::new(move |m: PopupMenu| {
                m.item(item("Zoom in  (Ctrl +)", true, |c, _, cx| c.zoom_by(1.25, None, cx)))
                    .item(item("Zoom out  (Ctrl −)", true, |c, _, cx| c.zoom_by(0.8, None, cx)))
                    .item(item("Actual size  (Ctrl+0)", true, |c, _, cx| c.zoom_to(1.0, cx)))
                    .item(item("Fit the graph  (F)", true, |c, _, cx| c.fit(cx)))
                    .separator()
                    .item(item("Dot grid", true, |c, _, cx| {
                        c.show_grid = !c.show_grid;
                        cx.notify();
                    }))
                    .item(item("Minimap", true, |c, _, cx| {
                        c.show_minimap = !c.show_minimap;
                        cx.notify();
                    }))
            }) as Rc<dyn Fn(PopupMenu) -> PopupMenu>
        };
        let mut bar = h_flex()
            .h(px(34.))
            .flex_none()
            .px_2()
            .gap_0p5()
            .border_b_1()
            .border_color(t.border)
            .child(menu("pipeline-menu-file", "File", file))
            .child(menu("pipeline-menu-edit", "Edit", edit));
        // One menu per kind of operation (Sources are in File)
        for c in catalog().into_iter().filter(|c| c.name != "Sources") {
            let canvas = canvas.clone();
            let items = c.items.clone();
            let build: Rc<dyn Fn(PopupMenu) -> PopupMenu> = Rc::new(move |m: PopupMenu| {
                let mut m = m;
                for it in &items {
                    let (canvas, template) = (canvas.clone(), it.template.clone());
                    m = m.item(PopupMenuItem::new(it.label).on_click(move |_, window, cx| {
                        Self::canvas_do(&canvas, cx, |c, cx| c.add(&template, None, window, cx));
                    }));
                }
                m
            });
            let id: &'static str = match c.name {
                "Filter" => "pipeline-menu-filter",
                "Reference" => "pipeline-menu-reference",
                "Smooth" => "pipeline-menu-smooth",
                "Transform" => "pipeline-menu-transform",
                _ => "pipeline-menu-output",
            };
            bar = bar.child(menu(id, c.name, build));
        }
        bar.child(menu("pipeline-menu-view", "View", view))
            .child(div().flex_1())
            .child(div().text_xs().text_color(t.muted_foreground).child("Wheel: pan · Ctrl+wheel: zoom · Space+drag: move · drag an output onto a node: connect"))
    }

    /// The operations tree: categories (fold with a click), items to drag or click.
    fn tree(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let t = cx.theme();
        let (muted, hover) = (t.muted_foreground, t.accent);
        let mut list = v_flex().gap_0p5();
        for c in catalog() {
            let folded = self.folded.contains(c.name);
            let name = c.name;
            list = list.child(
                h_flex()
                    .id(SharedString::from(format!("ops-cat-{}", name.to_lowercase())))
                    .h(px(26.))
                    .px_1()
                    .gap_1p5()
                    .rounded_md()
                    .cursor_pointer()
                    .hover(|d| d.bg(hover))
                    .child(Icon::new(if folded { IconName::ChevronRight } else { IconName::ChevronDown }).xsmall().text_color(muted))
                    .child(Icon::new(c.icon).small().text_color(muted))
                    .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(name))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        if !this.folded.remove(name) {
                            this.folded.insert(name);
                        }
                        cx.notify();
                    })),
            );
            if folded {
                continue;
            }
            for (i, it) in c.items.into_iter().enumerate() {
                let drag = CatalogDrag { label: it.label.into(), template: it.template.clone() };
                let canvas = self.canvas.downgrade();
                let template = it.template.clone();
                list = list.child(
                    v_flex()
                        .id(SharedString::from(format!("ops-item-{}-{i}", name.to_lowercase())))
                        .test_support()
                        .pl(px(30.))
                        .pr_1()
                        .py_0p5()
                        .rounded_md()
                        .cursor_grab()
                        .hover(|d| d.bg(hover))
                        .child(div().text_sm().truncate().child(it.label))
                        .child(div().text_xs().text_color(muted).truncate().child(it.hint))
                        .on_drag(drag, |d: &CatalogDrag, _, _, cx| cx.new(|_| DragGhost(d.label.clone())))
                        .on_click(move |_, window, cx| Self::canvas_do(&canvas, cx, |c, cx| c.add(&template, None, window, cx))),
                );
            }
        }
        v_flex()
            .id("ops-tree")
            .flex_none()
            .max_h(px(360.))
            .overflow_y_scroll()
            .p_2()
            .gap_1()
            .child(div().px_1().text_xs().text_color(muted).child("Drag onto the canvas, or click to add after the selected node"))
            .child(list)
    }
}

impl Render for PipelineView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let border = cx.theme().border;
        v_flex()
            .id("pipeline-workspace")
            .size_full()
            .child(self.menu_bar(cx))
            .child(
                h_flex()
                    .flex_1()
                    .min_h_0()
                    .child(div().flex_1().min_w_0().h_full().child(self.canvas.clone()))
                    .child(
                        v_flex()
                            .w(px(SIDE_W))
                            .flex_none()
                            .h_full()
                            .border_l_1()
                            .border_color(border)
                            .child(self.tree(cx))
                            .child(div().h(px(1.)).flex_none().bg(border))
                            .child(div().flex_1().min_h_0().child(self.inspector.clone())),
                    ),
            )
    }
}

/// The selected node: what it is, and what can be changed.
pub struct Inspector {
    store: Entity<Store>,
    canvas: Entity<GraphCanvas>,
    params: Entity<ParamEditor>,
    name: Entity<InputState>,
    /// The view node the name input was filled for.
    name_for: String,
    _subs: Vec<Subscription>,
}

impl Inspector {
    fn new(store: Entity<Store>, canvas: Entity<GraphCanvas>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let name = cx.new(|cx| InputState::new(window, cx));
        let subs = vec![
            cx.subscribe(&canvas, |_, _, _: &CanvasEvent, cx| cx.notify()),
            cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
                if matches!(e, AppEvent::Document(_) | AppEvent::JobChanged) {
                    cx.notify();
                }
            }),
            cx.subscribe(&name, |this, input, e: &InputEvent, cx| {
                if matches!(e, InputEvent::PressEnter { .. } | InputEvent::Blur) {
                    let text = input.read(cx).value().trim().to_string();
                    if !text.is_empty() && !this.name_for.is_empty() {
                        let view = this.name_for.clone();
                        let _ = this.store.update(cx, |s, cx| s.execute(SessionCommand::RenameView { view, name: text }, cx));
                    }
                }
            }),
        ];
        Self { store, canvas, params: cx.new(|_| ParamEditor::new()), name, name_for: String::new(), _subs: subs }
    }

    fn exec(&self, cmd: SessionCommand, cx: &mut App) {
        let _ = self.store.update(cx, |s, cx| s.execute(cmd, cx));
    }

    fn body(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = cx.theme();
        let muted = t.muted_foreground;
        let selected = self.canvas.read(cx).selected.clone();
        let s = self.store.read(cx);
        let g = s.doc.graph.clone();
        let job = s.job.as_ref().map(|j| (j.label.clone(), j.fraction));
        let active = s.doc.active_view.clone();
        if selected.len() > 1 {
            let canvas = self.canvas.downgrade();
            return Section::new(format!("{} nodes selected", selected.len()))
                .child(div().text_xs().text_color(muted).child("Drag one of them to move them together; Ctrl+D duplicates them with their edges."))
                .child(Button::new("inspector-delete").outline().small().label("Delete them").on_click(move |_, _, cx| {
                    let _ = canvas.update(cx, |c, cx| c.delete_selected(cx));
                }))
                .into_any_element();
        }
        let Some(node) = selected.iter().next().and_then(|id| g.node(id)).cloned() else {
            return Section::new("Nothing selected")
                .child(div().text_xs().text_color(muted).child("Click a node to see it here; its path through the graph lights up."))
                .child(div().text_xs().text_color(muted).child("A view node is a stream in Explore: connect a chain to it, then double-click it to show it."))
                .child(div().text_xs().text_color(muted).child("A node has one input; outputs fan out, so one recording can feed several branches."))
                .into_any_element();
        };
        let (title, sub, _) = describe(&node.kind);
        let path = g.path(&node.id);
        let header = v_flex()
            .gap_0p5()
            .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(title))
            .child(div().text_xs().font_family("monospace").text_color(muted).truncate().child(sub))
            .child(div().text_xs().text_color(muted).child(match &path {
                Some(p) => format!("{} · {} step{} from its recording", node.id, p.len() - 1, if p.len() == 2 { "" } else { "s" }),
                None if node.kind.has_input() => format!("{} · not connected to a recording", node.id),
                None => node.id.clone(),
            }));
        let id = node.id.clone();
        let body: AnyElement = match &node.kind {
            NodeKind::Source { recording } => {
                let open = s.recording.as_ref().and_then(|r| r.path.clone());
                let is_open = open.as_deref() == Some(recording.path.as_path());
                let store = self.store.clone();
                let path = recording.path.clone();
                v_flex()
                    .gap_2()
                    .child(row("File", div().text_xs().truncate().child(recording.path.display().to_string()), cx))
                    .child(row("Source", div().text_xs().child(recording.source.clone().unwrap_or_else(|| "default".into())), cx))
                    .child(row(
                        "Channels",
                        div().text_xs().child(recording.channels.as_ref().map_or_else(|| "all".to_string(), |c| format!("{} ({}…)", c.len(), c.iter().take(4).map(|c| c.to_string()).collect::<Vec<_>>().join(", ")))),
                        cx,
                    ))
                    .child(div().text_xs().text_color(muted).child(format!("Feeds {} node(s).", g.outputs(&id).len())))
                    .when(!is_open, |d| {
                        d.child(Button::new("inspector-open-recording").outline().small().label("Open in Explore").on_click(move |_, _, cx| {
                            store.update(cx, |s, cx| s.open(path.clone(), cx));
                        }))
                    })
                    .into_any_element()
            }
            NodeKind::Op { enabled, node: op } => {
                let commit: Commit = {
                    let (store, id) = (self.store.clone(), id.clone());
                    Rc::new(move |node, cx| store.update(cx, |s, cx| s.execute(SessionCommand::SetNode { id: id.clone(), node }, cx)).map(|_| ()).map_err(|e| e.to_string()))
                };
                self.params.update(cx, |p, cx| p.set_target(&id, op, commit, window, cx));
                let views: Vec<String> = g.descendants(&id).into_iter().filter(|d| matches!(g.node(d).map(|n| &n.kind), Some(NodeKind::View { .. }))).collect();
                let this = cx.entity().downgrade();
                let on = *enabled;
                let id2 = id.clone();
                v_flex()
                    .gap_2()
                    .child(Checkbox::new("inspector-enabled").checked(on).label("On (off: kept in the graph, skipped)").on_click(move |v: &bool, _, cx| {
                        let v = *v;
                        let _ = this.update(cx, |i, cx| i.exec(SessionCommand::EnableNode { id: id2.clone(), enabled: v }, cx));
                    }))
                    .child(Section::new(format!("Parameters · {}", op_summary(op))).child(self.params.clone()))
                    .child(div().text_xs().text_color(muted).child(if views.is_empty() {
                        "Feeds no view yet: connect it to one to see it in Explore.".to_string()
                    } else {
                        format!("Changes {}: {}", if views.len() == 1 { "view" } else { "views" }, views.join(", "))
                    }))
                    .into_any_element()
            }
            NodeKind::View { name, display, .. } => {
                if self.name_for != id {
                    self.name_for = id.clone();
                    let name = name.clone();
                    self.name.update(cx, |i, cx| i.set_value(name, window, cx));
                }
                let ops = path.as_ref().map_or(0, |p| p.len().saturating_sub(2));
                let shown = active.as_deref() == Some(id.as_str());
                let show = |which: Show, label: &'static str| {
                    let (store, id, display) = (self.store.clone(), id.clone(), display.clone());
                    Button::new(SharedString::from(format!("inspector-show-{label}")))
                        .small()
                        .label(label)
                        .map(|b| if display.show == which { b.primary() } else { b.outline() })
                        .on_click(move |_, _, cx| {
                            let display = DisplayDoc { show: which, ..display.clone() };
                            let _ = store.update(cx, |s, cx| s.execute(SessionCommand::SetDisplay { view: id.clone(), display }, cx));
                        })
                };
                let store = self.store.clone();
                let id2 = id.clone();
                v_flex()
                    .gap_2()
                    .child(row("Name", div().w(px(160.)).child(Input::new(&self.name).small()), cx))
                    .child(row("Shows", h_flex().gap_1().child(show(Show::Processed, "Processed")).child(show(Show::Raw, "Raw")), cx))
                    .child(div().text_xs().text_color(muted).child(match &path {
                        Some(_) => format!("{ops} op(s) on its path.{}", if shown { " Shown in Explore now." } else { "" }),
                        None => "Connect a chain to it (drag from an output onto it).".to_string(),
                    }))
                    .child(Button::new("inspector-show-view").primary().small().icon(Icon::new(IconName::ExternalLink)).label("Show in Explore").disabled(path.is_none()).on_click(move |_, _, cx| {
                        store.update(cx, |s, cx| s.show_view(&id2, cx));
                    }))
                    .into_any_element()
            }
            NodeKind::Write {} => {
                let out = g.chain(&id).map(|(r, _)| processed_path(&r.path, &id));
                let store = self.store.clone();
                let id2 = id.clone();
                v_flex()
                    .gap_2()
                    .child(row("Output", div().text_xs().truncate().child(out.as_ref().map_or_else(|| "connect it first".into(), |p| p.display().to_string())), cx))
                    .child(div().text_xs().text_color(muted).child("float32, Zarr, compressed; the whole recording through the enabled ops of its path."))
                    .child(match &job {
                        Some((label, fraction)) => div().text_xs().child(format!("{label}{}", fraction.map(|f| format!(" · {:.0} %", f * 100.0)).unwrap_or_default())).into_any_element(),
                        None => Button::new("inspector-write").primary().small().icon(Icon::new(IconName::Play)).label("Write").disabled(out.is_none()).on_click(move |_, _, cx| {
                            store.update(cx, |s, cx| s.write_node(&id2, cx));
                        }).into_any_element(),
                    })
                    .into_any_element()
            }
        };
        v_flex().gap_3().child(header).child(body).into_any_element()
    }
}

impl Render for Inspector {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let body = self.body(window, cx);
        v_flex().id("pipeline-inspector").size_full().p_3().gap_2().overflow_y_scroll().child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(cx.theme().muted_foreground).child("INSPECTOR")).child(body)
    }
}
