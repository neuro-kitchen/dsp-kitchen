//! The Pipeline workspace, laid out as Figma: a menu bar (one menu per family of operations, as
//! in Photoshop); on the left the layers (the graph as a tree: each recording, the ops it feeds,
//! the branches), the operations (drag one onto the canvas, or click to add it after the
//! selected node) and the recipes (the session's, the built-in ones, the user's library); the
//! canvas in the middle with a floating toolbar; on the right the inspector of the selection
//! (values, presets, a recipe's uses). Every edit is a session command, so the canvas, Explore
//! and an agent see the same graph.

use std::collections::HashSet;
use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenu, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Disableable as _, Icon, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::TestSupportExt as _;
use gpui_kit::{
    div, px, AnyElement, App, AppContext as _, Context, DragMoveEvent, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString,
    StatefulInteractiveElement as _, Styled as _, Subscription, WeakEntity, Window,
};

use crate::data::jobs::processed_path;
use crate::data::store::{AppEvent, Store};
use crate::domain::command::SessionCommand;
use crate::domain::graph::NodeKind;
use crate::domain::session::{DisplayDoc, Show};
use crate::ui::panel_edge::{follow_panel_drag, panel_edge, rail, toggle_item, PanelResize, RailItem, Side, RAIL_W};
use crate::ui::param_editor::{Commit, ParamEditor};
use crate::ui::widgets::{row, Section};

use super::canvas::{CanvasEvent, DragGhost, GraphCanvas, RecipeDrag};
use super::catalog::{catalog, describe, presets, type_name, Template};
use crate::data::preferences::Session;
use crate::domain::recipe::{builtin, library, RecipeDoc};

/// Widths of the sides (logical px).
const SIDE_W: f32 = 312.;
const LEFT_W: f32 = 260.;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeftTab {
    Layers,
    Recipes,
}

pub struct PipelineView {
    store: Entity<Store>,
    canvas: Entity<GraphCanvas>,
    inspector: Entity<Inspector>,
    left: LeftTab,
    /// Layers folded (their children hidden).
    collapsed: HashSet<String>,
    /// Built-in and library recipes (read when the Recipes tab opens), and unreadable files.
    library: Vec<(RecipeDoc, &'static str)>,
    library_errors: Vec<String>,
    _subs: Vec<Subscription>,
}

impl PipelineView {
    pub fn new(store: Entity<Store>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let canvas = cx.new(|cx| GraphCanvas::new(store.clone(), cx));
        let inspector = cx.new(|cx| Inspector::new(store.clone(), canvas.clone(), window, cx));
        let subs = vec![
            cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
                if matches!(e, AppEvent::Document(_) | AppEvent::RecordingChanged | AppEvent::JobChanged) {
                    cx.notify();
                }
            }),
            cx.subscribe(&canvas, |_, _, _: &CanvasEvent, cx| cx.notify()),
        ];
        let mut this = Self { store, canvas, inspector, left: LeftTab::Layers, collapsed: HashSet::new(), library: Vec::new(), library_errors: Vec::new(), _subs: subs };
        this.read_library();
        this
    }

    /// Adds `template` to the canvas (the palette).
    pub fn add(&mut self, template: &Template, window: &mut Window, cx: &mut Context<Self>) {
        self.canvas.update(cx, |c, cx| c.add(template, None, window, cx));
    }

    /// Adds a use of `recipe` (the palette).
    pub fn add_recipe(&mut self, recipe: RecipeDoc, cx: &mut Context<Self>) {
        self.canvas.update(cx, |c, cx| c.add_recipe(recipe, None, cx));
    }

    /// Built-in recipes and the user's library folder.
    fn read_library(&mut self) {
        self.library = builtin().into_iter().map(|r| (r, "built-in")).collect();
        self.library_errors.clear();
        if let Some(dir) = Session::recipes_dir() {
            let (found, errors) = library(&dir);
            self.library.extend(found.into_iter().map(|r| (r, "library")));
            self.library_errors = errors;
        }
    }

    /// The left side shows `tab`; its icon when shown already hides the side.
    fn set_left(&mut self, tab: LeftTab, cx: &mut Context<Self>) {
        let open = self.store.read(cx).panel_open("pipeline.left");
        if open && self.left == tab {
            self.store.update(cx, |s, cx| s.set_panel_open("pipeline.left", false, cx));
        } else {
            if tab == LeftTab::Recipes {
                self.read_library();
            }
            self.left = tab;
            if !open {
                self.store.update(cx, |s, cx| s.set_panel_open("pipeline.left", true, cx));
            }
        }
        cx.notify();
    }

    /// The graph as layers: each recording with what it feeds, branches nested; nodes without
    /// an input that are not recordings under "Not connected".
    fn layers(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let t = cx.theme();
        let (muted, accent) = (t.muted_foreground, t.accent);
        let g = self.store.read(cx).doc.graph.clone();
        let selected = self.canvas.read(cx).selected.clone();
        let mut rows: Vec<(String, usize, bool)> = Vec::new();
        fn walk(g: &crate::domain::graph::GraphDoc, id: &str, depth: usize, collapsed: &HashSet<String>, rows: &mut Vec<(String, usize, bool)>, seen: &mut HashSet<String>) {
            if !seen.insert(id.to_string()) {
                return;
            }
            let kids = g.outputs(id);
            rows.push((id.to_string(), depth, !kids.is_empty()));
            if collapsed.contains(id) {
                return;
            }
            for k in kids {
                walk(g, k, depth + 1, collapsed, rows, seen);
            }
        }
        let mut seen = HashSet::new();
        for n in g.nodes.iter().filter(|n| matches!(n.kind, NodeKind::Source { .. })) {
            walk(&g, &n.id, 0, &self.collapsed, &mut rows, &mut seen);
        }
        let loose: Vec<String> = g.nodes.iter().filter(|n| !seen.contains(&n.id) && g.input(&n.id).is_none()).map(|n| n.id.clone()).collect();
        let first_loose = rows.len();
        for id in &loose {
            walk(&g, id, 0, &self.collapsed, &mut rows, &mut seen);
        }
        let canvas = self.canvas.downgrade();
        let me = cx.entity().downgrade();
        let mut list = v_flex().gap_0p5();
        for (i, (id, depth, has_kids)) in rows.into_iter().enumerate() {
            if i == first_loose && !loose.is_empty() {
                list = list.child(div().pt_2().pb_1().px_1().text_xs().text_color(muted).child("Not connected"));
            }
            let Some(n) = g.node(&id) else { continue };
            let (title, _, icon) = describe(&n.kind);
            let on = selected.contains(&id);
            let folded = self.collapsed.contains(&id);
            let (canvas, me, id2, id3) = (canvas.clone(), me.clone(), id.clone(), id.clone());
            list = list.child(
                h_flex()
                    .id(SharedString::from(format!("layer-{id}")))
                    .test_support()
                    .h(px(26.))
                    .pl(px(4. + depth as f32 * 14.))
                    .pr_1()
                    .gap_1()
                    .rounded_md()
                    .cursor_pointer()
                    .when(on, |d| d.bg(accent))
                    .child(
                        div()
                            .id(SharedString::from(format!("layer-fold-{id}")))
                            .w(px(14.))
                            .when(has_kids, |d| d.child(Icon::new(if folded { IconName::ChevronRight } else { IconName::ChevronDown }).xsmall().text_color(muted)))
                            .on_click(move |_, _, cx| {
                                cx.stop_propagation();
                                let _ = me.update(cx, |p, cx| {
                                    if !p.collapsed.remove(&id2) {
                                        p.collapsed.insert(id2.clone());
                                    }
                                    cx.notify();
                                });
                            }),
                    )
                    .child(Icon::new(icon).small().text_color(muted))
                    .child(div().flex_1().min_w_0().truncate().text_sm().child(title))
                    .on_click(move |_, _, cx| Self::canvas_do(&canvas, cx, |c, cx| c.reveal(&id3, cx))),
            );
        }
        if g.nodes.is_empty() {
            list = list.child(div().p_2().text_xs().text_color(muted).child("No layers yet: add a recording."));
        }
        div().id("pipeline-layers").flex_1().min_h_0().overflow_y_scroll().p_1().child(list)
    }

    /// The session's recipes, the built-in ones and the library's: drag one onto the canvas or
    /// click it to add a use.
    fn recipes_panel(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let t = cx.theme();
        let (muted, hover, border) = (t.muted_foreground, t.accent, t.border);
        let session: Vec<RecipeDoc> = self.store.read(cx).doc.recipes.values().cloned().collect();
        let names: HashSet<String> = session.iter().map(|r| r.name.clone()).collect();
        let mut items: Vec<(RecipeDoc, &'static str)> = session.into_iter().map(|r| (r, "this session")).collect();
        items.extend(self.library.iter().filter(|(r, _)| !names.contains(&r.name)).cloned());
        let canvas = self.canvas.downgrade();
        let rows = items.into_iter().enumerate().map(|(i, (r, from))| {
            let drag = RecipeDrag { recipe: r.clone() };
            let (canvas, recipe) = (canvas.clone(), r.clone());
            v_flex()
                .id(("recipe-item", i))
                .p_1p5()
                .gap_0p5()
                .rounded_md()
                .cursor_grab()
                .hover(|d| d.bg(hover))
                .child(
                    h_flex()
                        .gap_1p5()
                        .child(Icon::new(IconName::Package).small().text_color(muted))
                        .child(div().flex_1().min_w_0().truncate().text_sm().child(r.name.clone()))
                        .child(div().text_xs().text_color(muted).child(format!("v{} · {from}", r.version))),
                )
                .when(!r.description.is_empty(), |d| d.child(div().text_xs().text_color(muted).line_clamp(2).child(r.description.clone())))
                .child(h_flex().flex_wrap().gap_1().children(r.labels.iter().map(|l| div().px_1().rounded_sm().border_1().border_color(border).text_xs().text_color(muted).child(l.clone()))))
                .on_drag(drag, |d: &RecipeDrag, _, _, cx| cx.new(|_| DragGhost(d.recipe.name.clone().into())))
                .on_click(move |_, _, cx| Self::canvas_do(&canvas, cx, |c, cx| c.add_recipe(recipe.clone(), None, cx)))
        });
        let dir = Session::recipes_dir().map(|d| d.display().to_string()).unwrap_or_default();
        v_flex()
            .id("pipeline-recipes")
            .flex_1()
            .min_h_0()
            .overflow_y_scroll()
            .p_2()
            .gap_1()
            .child(div().px_1().text_xs().text_color(muted).child("Select a chain of ops, right side → Make recipe. Drag one here onto the canvas, or click to add it."))
            .children(rows)
            .children(self.library_errors.iter().map(|e| div().px_1().text_xs().text_color(cx.theme().danger).child(e.clone())))
            .child(div().px_1().pt_2().text_xs().text_color(muted).truncate().child(format!("Library: {dir}")))
    }

    /// The tool bar floating at the bottom of the canvas: select or move, add a recording, an
    /// operation, an output, make a recipe.
    fn toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let t = cx.theme();
        let c = self.canvas.read(cx);
        let (hand, selected) = (c.hand, c.selected.len());
        let canvas = self.canvas.downgrade();
        let tool = |id: &'static str, icon: IconName, tip: &'static str, on: bool, f: fn(&mut GraphCanvas, &mut Context<GraphCanvas>)| {
            let canvas = canvas.clone();
            Button::new(id).ghost().small().icon(Icon::new(icon)).tooltip(tip).selected(on).on_click(move |_, _, cx| Self::canvas_do(&canvas, cx, f))
        };
        let menu = |id: &'static str, icon: IconName, label: &'static str, families: Vec<&'static str>| {
            let canvas = canvas.clone();
            Button::new(id).ghost().small().icon(Icon::new(icon)).label(label).dropdown_menu(move |m: PopupMenu, _, _| {
                let mut m = m;
                for c in catalog().into_iter().filter(|c| families.contains(&c.name)) {
                    m = m.label(c.name);
                    for it in c.items {
                        let (canvas, template) = (canvas.clone(), it.template.clone());
                        m = m.item(PopupMenuItem::new(it.label).on_click(move |_, window, cx| Self::canvas_do(&canvas, cx, |c, cx| c.add(&template, None, window, cx))));
                    }
                }
                m
            })
        };
        h_flex()
            .id("pipeline-toolbar")
            .test_support()
            .occlude()
            .absolute()
            .bottom_4()
            .left_0()
            .right_0()
            .justify_center()
            .child(
                h_flex()
                    .gap_0p5()
                    .p_1()
                    .rounded_lg()
                    .border_1()
                    .border_color(t.border)
                    .bg(t.popover)
                    .shadow_lg()
                    .child(tool("tool-select", IconName::MousePointer2, "Select (V)", !hand, |c, cx| {
                        c.hand = false;
                        cx.notify();
                    }))
                    .child(tool("tool-hand", IconName::Hand, "Move the canvas (H, or hold Space)", hand, |c, cx| {
                        c.hand = true;
                        cx.notify();
                    }))
                    .child(div().w(px(1.)).h(px(20.)).mx_1().bg(t.border))
                    .child(menu("tool-input", IconName::AudioWaveform, "Input", vec!["Sources"]))
                    .child(menu("tool-op", IconName::SlidersHorizontal, "Operation", vec!["Filter", "Reference", "Smooth", "Transform"]))
                    .child(menu("tool-output", IconName::Eye, "Output", vec!["Output"]))
                    .child(div().w(px(1.)).h(px(20.)).mx_1().bg(t.border))
                    .child({
                        let canvas = canvas.clone();
                        Button::new("tool-recipe")
                            .ghost()
                            .small()
                            .icon(Icon::new(IconName::Package))
                            .label("Make recipe")
                            .tooltip("The selected chain of ops becomes a recipe: one reusable node")
                            .disabled(selected == 0)
                            .on_click(move |_, _, cx| Self::canvas_do(&canvas, cx, |c, cx| c.make_recipe(cx)))
                    })
                    .child(tool("tool-fit", IconName::Maximize, "Fit the graph (F)", false, |c, cx| c.fit(cx))),
            )
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

}

impl Render for PipelineView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let border = cx.theme().border;
        let (side_w, left_w, left_open, right_open) = {
            let s = self.store.read(cx);
            (s.panel_width("pipeline.right", SIDE_W), s.panel_width("pipeline.left", LEFT_W), s.panel_open("pipeline.left"), s.panel_open("pipeline.right"))
        };
        let me = cx.entity().downgrade();
        let item = |which: LeftTab, id: &'static str, icon: IconName, tip: &'static str| {
            let me = me.clone();
            RailItem {
                id: id.into(),
                icon: Icon::new(icon),
                tip: tip.into(),
                selected: left_open && self.left == which,
                on_click: Box::new(move |_, cx| {
                    let _ = me.update(cx, |p, cx| p.set_left(which, cx));
                }),
            }
        };
        let left_rail = rail(vec![item(LeftTab::Layers, "rail-layers", IconName::Layers, "Layers: the graph as a tree"), item(LeftTab::Recipes, "rail-recipes", IconName::Package, "Recipes")], true, cx);
        let right_rail = rail(vec![toggle_item(&self.store, "pipeline.right", Icon::new(IconName::Inspector), "Inspector", cx)], false, cx);
        let left_body: AnyElement = match self.left {
            LeftTab::Layers => self.layers(cx).into_any_element(),
            LeftTab::Recipes => self.recipes_panel(cx).into_any_element(),
        };
        let title = match self.left {
            LeftTab::Layers => "LAYERS",
            LeftTab::Recipes => "RECIPES",
        };
        let muted = cx.theme().muted_foreground;
        v_flex()
            .id("pipeline-workspace")
            .size_full()
            .child(self.menu_bar(cx))
            .child(
                h_flex()
                    .flex_1()
                    .min_h_0()
                    .on_drag_move(cx.listener(|this, e: &DragMoveEvent<PanelResize>, _, cx| {
                        follow_panel_drag(&this.store, e, cx);
                        cx.notify();
                    }))
                    .child(left_rail)
                    .when(left_open, |d| {
                        d.child(
                            v_flex()
                                .relative()
                                .w(px(left_w))
                                .flex_none()
                                .h_full()
                                .border_r_1()
                                .border_color(border)
                                .child(h_flex().h(px(32.)).flex_none().px_3().border_b_1().border_color(border).child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(muted).child(title)))
                                .child(left_body)
                                .child(panel_edge("pipeline.left", Side::Left, RAIL_W)),
                        )
                    })
                    .child(div().relative().flex_1().min_w_0().h_full().child(self.canvas.clone()).child(self.toolbar(cx)))
                    .when(right_open, |d| {
                        d.child(
                            v_flex()
                                .relative()
                                .w(px(side_w))
                                .flex_none()
                                .h_full()
                                .border_l_1()
                                .border_color(border)
                                .child(div().flex_1().min_h_0().child(self.inspector.clone()))
                                .child(panel_edge("pipeline.right", Side::Right, RAIL_W)),
                        )
                    })
                    .child(right_rail),
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
    /// A recipe's name, labels and description inputs, and the recipe they were filled for.
    rname: Entity<InputState>,
    rlabels: Entity<InputState>,
    rdesc: Entity<InputState>,
    rinfo_for: String,
    /// One editor per op of the selected recipe.
    recipe_params: Vec<Entity<ParamEditor>>,
    /// Edits change the recipe itself (every use), not only this use.
    edit_recipe: bool,
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
        let (rname, rlabels, rdesc) = (cx.new(|cx| InputState::new(window, cx)), cx.new(|cx| InputState::new(window, cx).placeholder("filter, emg, sorter…")), cx.new(|cx| InputState::new(window, cx).placeholder("What it does")));
        let mut subs = subs;
        for input in [&rname, &rlabels, &rdesc] {
            subs.push(cx.subscribe(input, |this, _, e: &InputEvent, cx| {
                if matches!(e, InputEvent::PressEnter { .. } | InputEvent::Blur) {
                    this.save_recipe_info(cx);
                }
            }));
        }
        Self {
            store,
            canvas,
            params: cx.new(|_| ParamEditor::new()),
            name,
            name_for: String::new(),
            rname,
            rlabels,
            rdesc,
            rinfo_for: String::new(),
            recipe_params: Vec::new(),
            edit_recipe: false,
            _subs: subs,
        }
    }

    /// The recipe's name, labels and description from the inputs.
    fn save_recipe_info(&mut self, cx: &mut Context<Self>) {
        if self.rinfo_for.is_empty() {
            return;
        }
        let Some(def) = self.store.read(cx).doc.recipes.get(&self.rinfo_for).cloned() else { return };
        let name = self.rname.read(cx).value().trim().to_string();
        let labels: Vec<String> = self.rlabels.read(cx).value().split(',').map(|l| l.trim().to_lowercase()).filter(|l| !l.is_empty()).collect();
        let description = self.rdesc.read(cx).value().trim().to_string();
        if name == def.name && labels == def.labels && description == def.description {
            return;
        }
        let cmd = SessionCommand::SetRecipeInfo { recipe: def.name.clone(), name: name.clone(), labels, description };
        if self.store.update(cx, |s, cx| s.execute(cmd, cx)).is_ok() {
            self.rinfo_for = name;
        }
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
        let recipes = s.doc.recipes.clone();
        let job = s.job.as_ref().map(|j| (j.label.clone(), j.fraction));
        let active = s.doc.active_view.clone();
        if selected.len() > 1 {
            let (canvas, canvas2) = (self.canvas.downgrade(), self.canvas.downgrade());
            let all_ops = selected.iter().all(|id| matches!(g.node(id).map(|n| &n.kind), Some(NodeKind::Op { .. })));
            return Section::new(format!("{} nodes selected", selected.len()))
                .child(div().text_xs().text_color(muted).child("Drag one of them to move them together; Ctrl+D duplicates them with their edges."))
                .child(
                    Button::new("inspector-make-recipe")
                        .primary()
                        .small()
                        .icon(Icon::new(IconName::Package))
                        .label("Make recipe")
                        .tooltip("These ops become one reusable node (a chain of ops only)")
                        .disabled(!all_ops)
                        .on_click(move |_, _, cx| {
                            let _ = canvas2.update(cx, |c, cx| c.make_recipe(cx));
                        }),
                )
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
                    .child({
                        // The file's sources (when loaded): pick which one this node reads
                        let loaded = s.recordings().find(|r| r.path.as_ref() == Some(&recording.path)).map(|r| r.sources.clone());
                        match &loaded {
                            Some(ss) => {
                                let names: Vec<SharedString> = ss.entries().iter().map(|e| SharedString::from(e.summary())).collect();
                                let ids: Vec<String> = ss.entries().iter().map(|e| e.id.clone()).collect();
                                let current = recording.source.clone().unwrap_or_else(|| ss.default_entry().id.clone());
                                let at = ids.iter().position(|i| *i == current);
                                let (store, id, rec) = (self.store.clone(), id.clone(), recording.clone());
                                row(
                                    "Source",
                                    crate::ui::widgets::MenuSelect::new("inspector-source", names, at, move |i, _, cx| {
                                        // Another source: its channels differ, so the subset goes
                                        let recording = crate::domain::session::RecordingRef { path: rec.path.clone(), source: Some(ids[i].clone()), channels: None };
                                        let _ = store.update(cx, |s, cx| s.execute(SessionCommand::SetSource { id: id.clone(), recording }, cx));
                                    })
                                    .full_width(),
                                    cx,
                                )
                                .into_any_element()
                            }
                            None => row("Source", div().text_xs().child(format!("{} (open the file to choose)", recording.source.clone().unwrap_or_else(|| "default".into()))), cx).into_any_element(),
                        }
                    })
                    .children(s.recordings().find(|r| r.path.as_ref() == Some(&recording.path)).map(|r| {
                        let e = r.sources.entry(recording.source.as_deref().unwrap_or(&r.sources.default_entry().id)).clone();
                        Section::new("Metadata").child(crate::screens::explore::components::panels::source_facts(&e, muted))
                    }))
                    .child(row(
                        "Channels",
                        div().text_xs().child(recording.channels.as_ref().map_or_else(|| "all".to_string(), |c| format!("{} ({}…)", c.len(), c.iter().take(4).map(|c| c.to_string()).collect::<Vec<_>>().join(", ")))),
                        cx,
                    ))
                    .child(div().text_xs().text_color(muted).child(format!("Feeds {} node(s).", g.outputs(&id).len())))
                    .when(s.writes_of(&id).len() >= 2, |d| {
                        let (store, id) = (self.store.clone(), id.clone());
                        let n = s.writes_of(&id).len();
                        d.child(Button::new("inspector-write-all").primary().small().icon(Icon::new(IconName::Play)).label(format!("Write its {n} outputs (one read)")).on_click(move |_, _, cx| {
                            store.update(cx, |s, cx| s.write_source(&id, cx));
                        }))
                    })
                    .when(!is_open, |d| {
                        d.child(Button::new("inspector-open-recording").outline().small().label("Open in Explore").on_click(move |_, _, cx| {
                            store.update(cx, |s, cx| s.open(path.clone(), cx));
                        }))
                    })
                    .into_any_element()
            }
            NodeKind::Op { enabled, node: op @ dspk_pipeline::Node::Remap { path: map_path, .. } } => {
                // The input's channels: of the node's recording, when it is loaded
                let input = path.as_ref().and_then(|p| match g.node(&p[0]).map(|n| &n.kind) {
                    Some(NodeKind::Source { recording }) => s.recordings().find(|r| r.path.as_ref() == Some(&recording.path)).map(|r| {
                        let src = recording.source.clone().unwrap_or_else(|| r.sources.default_entry().id.clone());
                        let ds = r.sources.get(&src);
                        let read: Vec<usize> = recording.channels.clone().unwrap_or_else(|| (0..r.sources.entry(&src).channels).collect());
                        let labels: Vec<Option<String>> = read.iter().map(|&c| ds.channel_label(c).map(str::to_string)).collect();
                        (read.len(), labels)
                    }),
                    _ => None,
                });
                let (channels, labels) = match input {
                    Some((n, l)) => (Some(n), l),
                    None => (None, Vec::new()),
                };
                let on = *enabled;
                let this = cx.entity().downgrade();
                let id2 = id.clone();
                let enabled_box = Checkbox::new("inspector-enabled").checked(on).label("On").on_click(move |v: &bool, _, cx| {
                    let v = *v;
                    let _ = this.update(cx, |i, cx| i.exec(SessionCommand::EnableNode { id: id2.clone(), enabled: v }, cx));
                });
                let table: AnyElement = match map_path {
                    Some(file) => {
                        let (store, id, file) = (self.store.clone(), id.clone(), file.clone());
                        v_flex()
                            .gap_2()
                            .child(div().text_xs().text_color(muted).child(format!("Reads the map file {}", file.display())))
                            .child(Button::new("remap-edit-here").outline().small().label("Edit here (copy the file's map into the node)").on_click(move |_, _, cx| {
                                match dspk_io::read_channel_map(&file).map_err(|e| e.to_string()).and_then(|m| super::channel_table::Rows::of(&m)) {
                                    Ok(rows) => {
                                        let _ = store.update(cx, |s, cx| s.execute(SessionCommand::SetNode { id: id.clone(), node: rows.node() }, cx));
                                    }
                                    Err(e) => store.update(cx, |s, cx| s.set_status(format!("{}: {e}", file.display()), cx)),
                                }
                            }))
                            .into_any_element()
                    }
                    None => match op.channel_map(None).map_err(|e| e.to_string()).and_then(|m| m.map(|m| super::channel_table::Rows::of(&m)).transpose()) {
                        Ok(Some(rows)) => super::channel_table::editor(&self.store, &id, rows, channels, |c| labels.get(c).cloned().flatten(), muted),
                        Ok(None) => div().into_any_element(),
                        Err(e) => v_flex()
                            .gap_1()
                            .child(div().text_xs().text_color(cx.theme().danger).child(e))
                            .children(channels.map(|c| {
                                let (store, id) = (self.store.clone(), id.clone());
                                Button::new("remap-fix").outline().small().label("Start over: every channel in order").on_click(move |_, _, cx| {
                                    let _ = store.update(cx, |s, cx| s.execute(SessionCommand::SetNode { id: id.clone(), node: super::channel_table::Rows::identity(c).node() }, cx));
                                })
                            }))
                            .into_any_element(),
                    },
                };
                v_flex()
                    .gap_2()
                    .child(enabled_box)
                    .child(Section::new("Channels").child(table))
                    .child(div().text_xs().text_color(muted).child("Right after the recording it folds into the read (Explore shows the new order); later in the chain it reorders on the device."))
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
                let preset_list = presets(op);
                let preset = (!preset_list.is_empty()).then(|| {
                    let names: Vec<SharedString> = preset_list.iter().map(|(l, _)| SharedString::from(*l)).collect();
                    let current = preset_list.iter().position(|(_, v)| serde_json::from_value::<dspk_pipeline::Node>(v.clone()).ok().as_ref() == Some(op));
                    let (store, id) = (self.store.clone(), id.clone());
                    row(
                        "Preset",
                        crate::ui::widgets::MenuSelect::new("inspector-preset", names, current, move |i, _, cx| {
                            if let Ok(node) = serde_json::from_value(preset_list[i].1.clone()) {
                                let _ = store.update(cx, |s, cx| s.execute(SessionCommand::SetNode { id: id.clone(), node }, cx));
                            }
                        }),
                        cx,
                    )
                });
                v_flex()
                    .gap_2()
                    .children(preset)
                    .child(Checkbox::new("inspector-enabled").checked(on).label("On").on_click(move |v: &bool, _, cx| {
                        let v = *v;
                        let _ = this.update(cx, |i, cx| i.exec(SessionCommand::EnableNode { id: id2.clone(), enabled: v }, cx));
                    }))
                    .child(Section::new("Parameters").child(self.params.clone()))
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
                    .child(row("Name", div().w_full().child(Input::new(&self.name).small()), cx))
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
            NodeKind::Recipe { recipe, enabled, overrides } => self.recipe_body(&id, recipe, *enabled, overrides, &g, &recipes, window, cx),
            NodeKind::Write {} => {
                let out = g.chain(&id, &recipes).map(|(r, _)| processed_path(&r.path, &id));
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

impl Inspector {
    /// A recipe node: its definition (name, labels, version), its values (this use's, or the
    /// recipe's for every use), ungroup, save to the library.
    #[allow(clippy::too_many_arguments)]
    fn recipe_body(
        &mut self,
        id: &str,
        recipe: &str,
        enabled: bool,
        overrides: &std::collections::BTreeMap<String, serde_json::Value>,
        g: &crate::domain::graph::GraphDoc,
        recipes: &crate::domain::graph::Recipes,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let muted = cx.theme().muted_foreground;
        let Some(def) = recipes.get(recipe).cloned() else {
            return div().text_xs().text_color(cx.theme().danger).child(format!("The session has no recipe {recipe:?}.")).into_any_element();
        };
        if self.rinfo_for != def.name {
            self.rinfo_for = def.name.clone();
            let (n, l, d) = (def.name.clone(), def.labels.join(", "), def.description.clone());
            self.rname.update(cx, |i, cx| i.set_value(n, window, cx));
            self.rlabels.update(cx, |i, cx| i.set_value(l, window, cx));
            self.rdesc.update(cx, |i, cx| i.set_value(d, window, cx));
        }
        let uses = g.nodes.iter().filter(|n| matches!(&n.kind, NodeKind::Recipe { recipe: r, .. } if *r == def.name)).count();
        while self.recipe_params.len() < def.ops.len() {
            self.recipe_params.push(cx.new(|_| ParamEditor::new()));
        }
        let ops = def.ops_with(overrides);
        let mut sections = Vec::new();
        for (k, op) in ops.iter().enumerate() {
            let commit: Commit = if self.edit_recipe {
                let (store, recipe, op_id) = (self.store.clone(), def.name.clone(), op.id.clone());
                Rc::new(move |node, cx| store.update(cx, |s, cx| s.execute(SessionCommand::SetRecipeOp { recipe: recipe.clone(), op: op_id.clone(), node }, cx)).map(|_| ()).map_err(|e| e.to_string()))
            } else {
                let (store, id, def, op_id, current) = (self.store.clone(), id.to_string(), def.clone(), op.id.clone(), overrides.clone());
                Rc::new(move |node, cx| {
                    // This use's values for the op: what differs from the recipe; the rest goes back
                    let want = def.overrides_for(&op_id, &node);
                    let prefix = format!("{op_id}.");
                    let mut cmds: Vec<SessionCommand> = current.keys().filter(|k| k.starts_with(&prefix) && !want.contains_key(*k)).map(|k| SessionCommand::SetRecipeOverride { id: id.clone(), key: k.clone(), value: serde_json::Value::Null }).collect();
                    cmds.extend(want.into_iter().map(|(key, value)| SessionCommand::SetRecipeOverride { id: id.clone(), key, value }));
                    for c in cmds {
                        store.update(cx, |s, cx| s.execute(c, cx)).map_err(|e| e.to_string())?;
                    }
                    Ok(())
                })
            };
            let key = format!("{id}/{}/{}", op.id, self.edit_recipe);
            self.recipe_params[k].update(cx, |p, cx| p.set_target(&key, &op.node, commit, window, cx));
            let own = overrides.keys().filter(|o| o.starts_with(&format!("{}.", op.id))).count();
            let title = format!("{} · {}{}", op.id, type_name(&op.node), if own > 0 && !self.edit_recipe { format!(" · {own} own") } else { String::new() });
            sections.push(Section::new(title).child(self.recipe_params[k].clone()));
        }
        let me = cx.entity().downgrade();
        let mode = |label: &'static str, recipe_mode: bool, on: bool| {
            let me = me.clone();
            Button::new(SharedString::from(format!("recipe-mode-{recipe_mode}")))
                .xsmall()
                .label(label)
                .map(|b| if on { b.primary() } else { b.outline() })
                .on_click(move |_, _, cx| {
                    let _ = me.update(cx, |i, cx| {
                        i.edit_recipe = recipe_mode;
                        cx.notify();
                    });
                })
        };
        let (store, store2, id2, id3, save_def) = (self.store.clone(), self.store.clone(), id.to_string(), id.to_string(), def.clone());
        v_flex()
            .gap_2()
            .child(row("Name", div().w_full().child(Input::new(&self.rname).small()), cx))
            .child(row("Labels", div().w_full().child(Input::new(&self.rlabels).small()), cx))
            .child(row("About", div().w_full().child(Input::new(&self.rdesc).small()), cx))
            .child(div().text_xs().text_color(muted).child(format!(
                "Version {} · {} use{} · {} → {}{}",
                def.version,
                uses,
                if uses == 1 { "" } else { "s" },
                format!("{:?}", def.input).to_lowercase(),
                format!("{:?}", def.output).to_lowercase(),
                if def.is_sorter() { " · listed in Sorting" } else { "" }
            )))
            .child(Checkbox::new("recipe-enabled").checked(enabled).label("On").on_click(move |v: &bool, _, cx| {
                let _ = store.update(cx, |s, cx| s.execute(SessionCommand::EnableNode { id: id2.clone(), enabled: *v }, cx));
            }))
            .child(h_flex().gap_1().child(div().text_xs().text_color(muted).child("Edit")).child(mode("this use", false, !self.edit_recipe)).child(mode("the recipe (every use)", true, self.edit_recipe)))
            .children(sections)
            .child(
                h_flex()
                    .gap_1()
                    .child(Button::new("recipe-ungroup").outline().small().icon(Icon::new(IconName::Layers)).label("Ungroup").on_click(move |_, _, cx| {
                        let _ = store2.update(cx, |s, cx| s.execute(SessionCommand::Ungroup { id: id3.clone() }, cx));
                    }))
                    .child({
                        let store = self.store.clone();
                        Button::new("recipe-save").outline().small().icon(Icon::new(IconName::Save)).label("Save to library").on_click(move |_, _, cx| {
                            let status = match Session::recipes_dir().map(|d| save_def.save_in(&d)) {
                                Some(Ok(p)) => format!("Recipe saved: {}", p.display()),
                                Some(Err(e)) => format!("Recipe not saved: {e}"),
                                None => "No configuration folder for the library".to_string(),
                            };
                            store.update(cx, |s, cx| s.set_status(status, cx));
                        })
                    }),
            )
            .into_any_element()
    }
}

impl Render for Inspector {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let body = self.body(window, cx);
        v_flex().id("pipeline-inspector").size_full().p_3().gap_2().overflow_y_scroll().child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(cx.theme().muted_foreground).child("INSPECTOR")).child(body)
    }
}
