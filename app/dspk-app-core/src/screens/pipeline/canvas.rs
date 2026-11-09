//! The Pipeline canvas: the session's graph on an infinite plane, navigated as in Figma — the
//! wheel (or a two-finger swipe) pans, Ctrl / Cmd + wheel zooms at the pointer, Space + drag or
//! the middle button drags the plane; a drag on empty space selects with a rectangle.
//!
//! Nodes are cards (KotlinFlow style: rounded, a border that lights up when selected) with an
//! input handle on the left and an output handle on the right; dragging from an output to a
//! node connects them (a node has one input: the new edge replaces it), dragging an input off
//! its edge disconnects it. Pointing at a node (or selecting one) highlights its path:
//! everything upstream to its recording and downstream to every view or write it feeds, its
//! edges and ports lit; the rest fades. Edges run left to right with rounded elbows. Each node
//! shows its last run (writes, sorts): running (with the stage, and data flowing along the edges
//! into it), done (how long it took), failed (why). Drop an operation from the tree (or the
//! menus) on the plane to add it.
//!
//! Hit testing is geometric (node rectangles and handle circles in canvas units), so the cards
//! are plain elements and the plane handles every pointer event.

use std::cell::Cell;
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Icon, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::TestSupportExt as _;
use gpui_kit::{
    canvas, div, fill, point, px, size, App, Bounds, Context, Entity, EventEmitter, FocusHandle, FontWeight, Hsla, InteractiveElement as _, IntoElement,
    KeyDownEvent, KeyUpEvent, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement as _, PathBuilder, Pixels, Render, ScrollEvent, SharedString,
    Styled as _, Subscription, Window,
};

use crate::data::store::{AppEvent, NodeRun, Store};
use crate::domain::command::{NodeMove, SessionCommand};
use crate::domain::graph::{GraphDoc, NodeKind, COLUMN};
use crate::ui::widgets::icon_button;

use super::catalog::{describe, type_name, Template};

/// A node's size in canvas units.
pub const NODE_W: f32 = 216.;
pub const NODE_H: f32 = 64.;
/// Height of a recipe's tab above its card (canvas units).
const RECIPE_TAB: f32 = 18.;
/// Handle radius (screen px) and how far from it a press still grabs it.
const HANDLE_R: f32 = 5.;
const HANDLE_GRAB: f32 = 11.;
/// Dot grid spacing and node snapping (canvas units).
const GRID: f32 = 24.;
const SNAP: f32 = 8.;
const MIN_ZOOM: f32 = 0.2;
const MAX_ZOOM: f32 = 2.5;
/// Pointer travel (screen px) that turns a press on a node into a drag.
const DRAG_SLOP: f32 = 3.;
/// Minimap size (screen px).
const MAP_W: f32 = 200.;
const MAP_H: f32 = 128.;

/// An item dragged from the operations tree.
#[derive(Debug, Clone)]
pub struct CatalogDrag {
    pub label: SharedString,
    pub template: Template,
}

/// What the pointer drags on the plane.
#[derive(Debug, Clone)]
enum Drag {
    Pan { mouse: (f32, f32), pan: (f32, f32) },
    Nodes { mouse: (f32, f32), origins: Vec<(String, f32, f32)>, moved: bool },
    /// A new edge from `from`'s output, its loose end at the pointer (canvas-local px).
    Connect { from: String, to: (f32, f32) },
    /// Selection rectangle, canvas-local px.
    Marquee { a: (f32, f32), b: (f32, f32), additive: bool },
}

pub enum CanvasEvent {
    SelectionChanged,
}

pub struct GraphCanvas {
    store: Entity<Store>,
    pub focus: FocusHandle,
    /// Screen offset of the canvas origin inside the plane (logical px), and the scale.
    pan: (f32, f32),
    zoom: f32,
    pub selected: BTreeSet<String>,
    drag: Option<Drag>,
    /// Positions of nodes being dragged, until the drop writes them to the document.
    live: HashMap<String, (f32, f32)>,
    space: bool,
    /// The hand tool: a drag moves the plane (else it selects).
    pub hand: bool,
    pub show_grid: bool,
    pub show_minimap: bool,
    /// The plane's bounds in the window (set while painting).
    bounds: Rc<Cell<Bounds<Pixels>>>,
    /// Fit the graph once the plane has a size.
    fit_pending: bool,
    /// The node under the pointer (its path is highlighted while nothing is selected).
    hover: Option<String>,
    /// Recipe nodes whose ops are listed under them.
    expanded: BTreeSet<String>,
    _subs: Vec<Subscription>,
}

impl EventEmitter<CanvasEvent> for GraphCanvas {}

impl GraphCanvas {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let subs = vec![cx.subscribe(&store, |this, s, e: &AppEvent, cx| {
            if matches!(e, AppEvent::Document(_) | AppEvent::RecordingChanged | AppEvent::JobChanged) {
                let g = &s.read(cx).doc.graph;
                let before = this.selected.len();
                this.selected.retain(|id| g.contains(id));
                if this.selected.len() != before {
                    cx.emit(CanvasEvent::SelectionChanged);
                }
                cx.notify();
            }
        })];
        Self {
            store,
            focus: cx.focus_handle(),
            pan: (40.0, 40.0),
            zoom: 1.0,
            selected: BTreeSet::new(),
            drag: None,
            live: HashMap::new(),
            space: false,
            hand: false,
            show_grid: true,
            show_minimap: true,
            bounds: Rc::new(Cell::new(Bounds::default())),
            fit_pending: true,
            hover: None,
            expanded: BTreeSet::new(),
            _subs: subs,
        }
    }

    pub fn store_entity(&self) -> Entity<Store> {
        self.store.clone()
    }

    fn graph<'a>(&self, cx: &'a App) -> &'a GraphDoc {
        &self.store.read(cx).doc.graph
    }

    /// A node's place, following a drag in progress.
    fn pos(&self, id: &str, x: f32, y: f32) -> (f32, f32) {
        self.live.get(id).copied().unwrap_or((x, y))
    }

    /// Window position → canvas-local px.
    fn local(&self, p: gpui_kit::Point<Pixels>) -> (f32, f32) {
        let o = self.bounds.get().origin;
        ((p.x - o.x).as_f32(), (p.y - o.y).as_f32())
    }

    /// Canvas-local px → canvas units.
    fn world(&self, (x, y): (f32, f32)) -> (f32, f32) {
        ((x - self.pan.0) / self.zoom, (y - self.pan.1) / self.zoom)
    }

    fn screen(&self, (x, y): (f32, f32)) -> (f32, f32) {
        (x * self.zoom + self.pan.0, y * self.zoom + self.pan.1)
    }

    /// The topmost node under canvas point `p`.
    fn node_at(&self, p: (f32, f32), cx: &App) -> Option<String> {
        self.graph(cx).nodes.iter().rev().find_map(|n| {
            let (x, y) = self.pos(&n.id, n.x, n.y);
            (p.0 >= x && p.0 <= x + NODE_W && p.1 >= y && p.1 <= y + NODE_H).then(|| n.id.clone())
        })
    }

    /// The handle under canvas point `p`: (node, is its output).
    fn handle_at(&self, p: (f32, f32), cx: &App) -> Option<(String, bool)> {
        let grab = HANDLE_GRAB / self.zoom;
        self.graph(cx).nodes.iter().rev().find_map(|n| {
            let (x, y) = self.pos(&n.id, n.x, n.y);
            let cy = y + NODE_H / 2.0;
            let near = |hx: f32| (p.0 - hx).hypot(p.1 - cy) <= grab;
            if n.kind.has_output() && near(x + NODE_W) {
                Some((n.id.clone(), true))
            } else if n.kind.has_input() && near(x) {
                Some((n.id.clone(), false))
            } else {
                None
            }
        })
    }

    fn exec(&self, cmd: SessionCommand, cx: &mut Context<Self>) -> bool {
        self.store.update(cx, |s, cx| s.execute(cmd, cx)).is_ok()
    }

    pub fn select(&mut self, ids: impl IntoIterator<Item = String>, cx: &mut Context<Self>) {
        self.selected = ids.into_iter().collect();
        cx.emit(CanvasEvent::SelectionChanged);
        cx.notify();
    }

    /// The node selected alone.
    pub fn single(&self) -> Option<&String> {
        (self.selected.len() == 1).then(|| self.selected.iter().next()).flatten()
    }

    // ------------------------------------------------------------------------
    // View
    // ------------------------------------------------------------------------

    pub fn zoom_by(&mut self, factor: f32, anchor: Option<(f32, f32)>, cx: &mut Context<Self>) {
        let b = self.bounds.get().size;
        let anchor = anchor.unwrap_or((b.width.as_f32() / 2.0, b.height.as_f32() / 2.0));
        let w = self.world(anchor);
        self.zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        self.pan = (anchor.0 - w.0 * self.zoom, anchor.1 - w.1 * self.zoom);
        cx.notify();
    }

    pub fn zoom_to(&mut self, zoom: f32, cx: &mut Context<Self>) {
        self.zoom_by(zoom / self.zoom, None, cx);
    }

    /// The whole graph in view.
    pub fn fit(&mut self, cx: &mut Context<Self>) {
        let b = self.bounds.get().size;
        let (w, h) = (b.width.as_f32(), b.height.as_f32());
        if w < 10.0 || h < 10.0 {
            self.fit_pending = true;
            return;
        }
        self.fit_pending = false;
        let Some((x0, y0, x1, y1)) = extent(self.graph(cx)) else {
            self.zoom = 1.0;
            self.pan = (40.0, 40.0);
            cx.notify();
            return;
        };
        let pad = 80.0;
        self.zoom = ((w - pad) / (x1 - x0).max(1.0)).min((h - pad) / (y1 - y0).max(1.0)).clamp(MIN_ZOOM, 1.2);
        self.pan = (w / 2.0 - (x0 + x1) / 2.0 * self.zoom, h / 2.0 - (y0 + y1) / 2.0 * self.zoom);
        cx.notify();
    }

    /// Centres the plane on canvas point `p`.
    fn centre_on(&mut self, p: (f32, f32), cx: &mut Context<Self>) {
        let b = self.bounds.get().size;
        self.pan = (b.width.as_f32() / 2.0 - p.0 * self.zoom, b.height.as_f32() / 2.0 - p.1 * self.zoom);
        cx.notify();
    }

    /// Where a new node goes: right of the selected node, else the middle of the view.
    fn new_place(&self, cx: &App) -> (f32, f32) {
        if let Some(n) = self.single().and_then(|id| self.graph(cx).node(id)) {
            return (n.x + COLUMN, n.y);
        }
        let b = self.bounds.get().size;
        let (x, y) = self.world((b.width.as_f32() / 2.0, b.height.as_f32() / 2.0));
        (snap(x - NODE_W / 2.0), snap(y - NODE_H / 2.0))
    }

    // ------------------------------------------------------------------------
    // Editing
    // ------------------------------------------------------------------------

    /// Adds `template` at `at` (default: after the selected node, connected to it), selects it.
    pub fn add(&mut self, template: &Template, at: Option<(f32, f32)>, window: &mut Window, cx: &mut Context<Self>) {
        let after = if at.is_none() { self.single().filter(|id| self.graph(cx).node(id).is_some_and(|n| n.kind.has_output())).cloned() } else { None };
        let (x, y) = at.unwrap_or_else(|| self.new_place(cx));
        let kind = match template {
            Template::Op(v) => match serde_json::from_value(v.clone()) {
                Ok(node) => NodeKind::Op { enabled: true, node },
                Err(e) => {
                    self.store.update(cx, |s, cx| s.set_status(e.to_string(), cx));
                    return;
                }
            },
            Template::Remap => {
                // Every channel of the input in order: of the selected node's recording when it is
                // loaded, else of the open recording's default source
                let s = self.store.read(cx);
                let g = self.graph(cx);
                let from_path = after.as_ref().and_then(|a| g.path(a)).and_then(|p| match g.node(&p[0]).map(|n| &n.kind) {
                    Some(NodeKind::Source { recording }) => s.recordings().find(|r| r.path.as_ref() == Some(&recording.path)).map(|r| {
                        let e = r.sources.entry(recording.source.as_deref().unwrap_or(&r.sources.default_entry().id));
                        recording.channels.as_ref().map_or(e.channels, Vec::len)
                    }),
                    _ => None,
                });
                let channels = from_path.or_else(|| s.sources().map(|ss| ss.default_entry().channels)).unwrap_or(1);
                NodeKind::Op { enabled: true, node: super::channel_table::Rows::identity(channels).node() }
            }
            Template::View => {
                let n = self.graph(cx).nodes.iter().filter(|n| matches!(n.kind, NodeKind::View { .. })).count() + 1;
                crate::domain::command::view_kind(format!("View {n}"))
            }
            Template::Write => NodeKind::Write {},
            Template::OpenRecording => {
                let s = self.store.read(cx);
                let Some(path) = s.recording.as_ref().and_then(|r| r.path.clone()) else {
                    self.store.update(cx, |s, cx| s.set_status("No recording is open: use Recording from file…", cx));
                    return;
                };
                let source = s.sources().map(|ss| ss.default_entry().id.clone());
                NodeKind::Source { recording: crate::domain::session::RecordingRef { path, source, channels: None } }
            }
            Template::RecordingFile => {
                self.prompt_recording(x, y, window, cx);
                return;
            }
        };
        self.add_kind(kind, x, y, after, cx);
    }

    fn add_kind(&mut self, kind: NodeKind, x: f32, y: f32, after: Option<String>, cx: &mut Context<Self>) {
        let before: BTreeSet<String> = self.graph(cx).nodes.iter().map(|n| n.id.clone()).collect();
        if self.exec(SessionCommand::AddNode { kind, x, y, id: None, after }, cx) {
            let new: Vec<String> = self.graph(cx).nodes.iter().map(|n| n.id.clone()).filter(|id| !before.contains(id)).collect();
            self.select(new, cx);
        }
    }

    fn prompt_recording(&mut self, x: f32, y: f32, window: &mut Window, cx: &mut Context<Self>) {
        let picked = crate::ui::recording_picker::ask_and_pick("Add recording", false, window, cx);
        cx.spawn(async move |this, cx| {
            if let Some(paths) = picked.await {
                if let Some(path) = paths.into_iter().next() {
                    let _ = this.update(cx, |c, cx| {
                        let kind = NodeKind::Source { recording: crate::domain::session::RecordingRef { path, source: None, channels: None } };
                        c.add_kind(kind, x, y, None, cx);
                    });
                }
            }
        })
        .detach();
    }

    pub fn delete_selected(&mut self, cx: &mut Context<Self>) {
        if self.selected.is_empty() {
            return;
        }
        let nodes: Vec<String> = self.selected.iter().cloned().collect();
        self.exec(SessionCommand::RemoveNodes { nodes }, cx);
        self.select([], cx);
    }

    /// Copies of the selected nodes (and the edges between them), offset down-right.
    pub fn duplicate_selected(&mut self, cx: &mut Context<Self>) {
        let g = self.graph(cx).clone();
        let mut map: HashMap<String, String> = HashMap::new();
        for id in &self.selected {
            let Some(n) = g.node(id) else { continue };
            let before: BTreeSet<String> = self.graph(cx).nodes.iter().map(|n| n.id.clone()).collect();
            let mut kind = n.kind.clone();
            if let NodeKind::View { name, plugins, .. } = &mut kind {
                *name = format!("{name} (copy)");
                plugins.clear();
            }
            if self.exec(SessionCommand::AddNode { kind, x: n.x + 32.0, y: n.y + 32.0, id: None, after: None }, cx) {
                if let Some(new) = self.graph(cx).nodes.iter().map(|n| n.id.clone()).find(|i| !before.contains(i)) {
                    map.insert(id.clone(), new);
                }
            }
        }
        for e in &g.edges {
            if let (Some(a), Some(b)) = (map.get(&e.from), map.get(&e.to)) {
                self.exec(SessionCommand::Connect { from: a.clone(), to: b.clone() }, cx);
            }
        }
        self.select(map.into_values(), cx);
    }

    /// The selected ops become a recipe (named `Recipe N`, renamed in the inspector).
    pub fn make_recipe(&mut self, cx: &mut Context<Self>) {
        let nodes: Vec<String> = self.selected.iter().cloned().collect();
        let n = self.store.read(cx).doc.recipes.len() + 1;
        let name = (n..).map(|i| format!("Recipe {i}")).find(|name| !self.store.read(cx).doc.recipes.contains_key(name)).expect("names are unbounded");
        let before: BTreeSet<String> = self.graph(cx).nodes.iter().map(|n| n.id.clone()).collect();
        if self.exec(SessionCommand::MakeRecipe { nodes, name, labels: Vec::new() }, cx) {
            let new: Vec<String> = self.graph(cx).nodes.iter().map(|n| n.id.clone()).filter(|id| !before.contains(id)).collect();
            self.select(new, cx);
        }
    }

    /// Adds a use of recipe `name` (from the library when the session does not have it yet).
    pub fn add_recipe(&mut self, recipe: crate::domain::recipe::RecipeDoc, at: Option<(f32, f32)>, cx: &mut Context<Self>) {
        let name = recipe.name.clone();
        if !self.store.read(cx).doc.recipes.contains_key(&name) && !self.exec(SessionCommand::AddRecipe { recipe }, cx) {
            return;
        }
        let after = if at.is_none() { self.single().filter(|id| self.graph(cx).node(id).is_some_and(|n| n.kind.has_output())).cloned() } else { None };
        let (x, y) = at.unwrap_or_else(|| self.new_place(cx));
        self.add_kind(NodeKind::Recipe { recipe: name, enabled: true, overrides: Default::default() }, x, y, after, cx);
    }

    /// Centres the view on node `id` and selects it.
    pub fn reveal(&mut self, id: &str, cx: &mut Context<Self>) {
        if let Some(n) = self.graph(cx).node(id) {
            let p = (n.x + NODE_W / 2.0, n.y + NODE_H / 2.0);
            self.centre_on(p, cx);
        }
        self.select([id.to_string()], cx);
    }

    pub fn select_all(&mut self, cx: &mut Context<Self>) {
        let all: Vec<String> = self.graph(cx).nodes.iter().map(|n| n.id.clone()).collect();
        self.select(all, cx);
    }

    pub fn disconnect_selected(&mut self, cx: &mut Context<Self>) {
        for id in self.selected.clone() {
            if self.graph(cx).input(&id).is_some() {
                self.exec(SessionCommand::Disconnect { to: id }, cx);
            }
        }
    }

    // ------------------------------------------------------------------------
    // Input
    // ------------------------------------------------------------------------

    fn on_down(&mut self, e: &MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        window.focus(&self.focus, cx);
        let local = self.local(e.position);
        if e.button == MouseButton::Middle || self.space || self.hand || e.modifiers.alt {
            self.drag = Some(Drag::Pan { mouse: local, pan: self.pan });
            return;
        }
        let p = self.world(local);
        let additive = e.modifiers.shift || e.modifiers.control || e.modifiers.platform;
        if let Some((id, output)) = self.handle_at(p, cx) {
            if output {
                self.drag = Some(Drag::Connect { from: id, to: local });
                return;
            }
            // Pull an input off its edge: the edge follows the pointer from its source
            if let Some(from) = self.graph(cx).input(&id).map(str::to_string) {
                self.exec(SessionCommand::Disconnect { to: id }, cx);
                self.drag = Some(Drag::Connect { from, to: local });
                return;
            }
        }
        if let Some(id) = self.node_at(p, cx) {
            if e.click_count == 2 {
                self.open(&id, cx);
                return;
            }
            if additive {
                let mut s = self.selected.clone();
                if !s.remove(&id) {
                    s.insert(id.clone());
                }
                self.select(s, cx);
            } else if !self.selected.contains(&id) {
                self.select([id.clone()], cx);
            }
            let g = self.graph(cx);
            let origins = self.selected.iter().filter_map(|i| g.node(i).map(|n| (i.clone(), n.x, n.y))).collect();
            self.drag = Some(Drag::Nodes { mouse: local, origins, moved: false });
            return;
        }
        if !additive && !self.selected.is_empty() {
            self.select([], cx);
        }
        self.drag = Some(Drag::Marquee { a: local, b: local, additive });
    }

    fn on_move(&mut self, e: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if e.pressed_button.is_none() {
            if self.drag.take().is_some() {
                self.live.clear();
                cx.notify();
            }
            let hover = self.node_at(self.local(e.position), cx);
            if hover != self.hover {
                self.hover = hover;
                cx.notify();
            }
            return;
        }
        let local = self.local(e.position);
        let zoom = self.zoom;
        match &mut self.drag {
            Some(Drag::Pan { mouse, pan }) => {
                self.pan = (pan.0 + local.0 - mouse.0, pan.1 + local.1 - mouse.1);
            }
            Some(Drag::Nodes { mouse, origins, moved }) => {
                let (dx, dy) = (local.0 - mouse.0, local.1 - mouse.1);
                if !*moved && dx.hypot(dy) < DRAG_SLOP {
                    return;
                }
                *moved = true;
                self.live = origins.iter().map(|(id, x, y)| (id.clone(), (snap(x + dx / zoom), snap(y + dy / zoom)))).collect();
            }
            Some(Drag::Connect { to, .. }) => *to = local,
            Some(Drag::Marquee { b, .. }) => *b = local,
            None => return,
        }
        cx.notify();
    }

    fn on_up(&mut self, e: &MouseUpEvent, _: &mut Window, cx: &mut Context<Self>) {
        let Some(drag) = self.drag.take() else { return };
        match drag {
            Drag::Pan { .. } => {}
            Drag::Nodes { moved, .. } => {
                if moved {
                    let moves = self.live.iter().map(|(id, &(x, y))| NodeMove { id: id.clone(), x, y }).collect();
                    self.exec(SessionCommand::MoveNodes { moves }, cx);
                }
            }
            Drag::Connect { from, .. } => {
                let p = self.world(self.local(e.position));
                let target = self.handle_at(p, cx).map(|(id, _)| id).or_else(|| self.node_at(p, cx));
                if let Some(to) = target.filter(|t| *t != from) {
                    if self.exec(SessionCommand::Connect { from, to: to.clone() }, cx) {
                        self.select([to], cx);
                    }
                }
            }
            Drag::Marquee { a, b, additive } => {
                let (wa, wb) = (self.world(a), self.world(b));
                let (x0, x1, y0, y1) = (wa.0.min(wb.0), wa.0.max(wb.0), wa.1.min(wb.1), wa.1.max(wb.1));
                if (x1 - x0) * self.zoom > 2.0 || (y1 - y0) * self.zoom > 2.0 {
                    let hit: Vec<String> = self
                        .graph(cx)
                        .nodes
                        .iter()
                        .filter(|n| n.x < x1 && n.x + NODE_W > x0 && n.y < y1 && n.y + NODE_H > y0)
                        .map(|n| n.id.clone())
                        .collect();
                    let mut s = if additive { self.selected.clone() } else { BTreeSet::new() };
                    s.extend(hit);
                    self.select(s, cx);
                }
            }
        }
        self.live.clear();
        cx.notify();
    }

    fn on_wheel(&mut self, e: &ScrollEvent, _: &mut Window, cx: &mut Context<Self>) {
        let d = e.delta.pixel_delta(px(20.));
        let (dx, dy) = (d.x.as_f32(), d.y.as_f32());
        let m = e.modifiers;
        if m.control || m.platform {
            self.zoom_by((dy * 0.003).exp(), Some(self.local(e.position)), cx);
        } else if m.shift && dx == 0.0 {
            self.pan.0 += dy;
        } else {
            self.pan = (self.pan.0 + dx, self.pan.1 + dy);
        }
        cx.stop_propagation();
        cx.notify();
    }

    fn on_key(&mut self, e: &KeyDownEvent, _: &mut Window, cx: &mut Context<Self>) {
        let k = &e.keystroke;
        let cmd = k.modifiers.control || k.modifiers.platform;
        match k.key.as_str() {
            "space" => self.space = true,
            "delete" | "backspace" => self.delete_selected(cx),
            "escape" => self.select([], cx),
            "a" if cmd => self.select_all(cx),
            "d" if cmd => self.duplicate_selected(cx),
            "0" if cmd => self.zoom_to(1.0, cx),
            "=" | "+" if cmd => self.zoom_by(1.25, None, cx),
            "-" if cmd => self.zoom_by(0.8, None, cx),
            "f" if !cmd => self.fit(cx),
            "v" if !cmd => {
                self.hand = false;
                cx.notify();
            }
            "h" if !cmd => {
                self.hand = true;
                cx.notify();
            }
            "enter" => {
                if let Some(id) = self.single().cloned() {
                    self.open(&id, cx);
                }
            }
            _ => return,
        }
        cx.stop_propagation();
    }

    fn on_key_up(&mut self, e: &KeyUpEvent, _: &mut Window, _: &mut Context<Self>) {
        if e.keystroke.key == "space" {
            self.space = false;
        }
    }

    /// A double-click: a view shows in Explore, a write runs.
    pub fn open(&mut self, id: &str, cx: &mut Context<Self>) {
        let kind = self.graph(cx).node(id).map(|n| n.kind.clone());
        match kind {
            Some(NodeKind::View { .. }) => self.store.update(cx, |s, cx| s.show_view(id, cx)),
            Some(NodeKind::Write {}) => self.store.update(cx, |s, cx| s.write_node(id, cx)),
            _ => {}
        }
    }

    // ------------------------------------------------------------------------
    // Drawing
    // ------------------------------------------------------------------------

    /// Nodes on the path (upstream and downstream) of the selected node, else of the node under
    /// the pointer.
    fn highlight(&self, g: &GraphDoc) -> Option<BTreeSet<String>> {
        let id = self.single().or(self.hover.as_ref()).filter(|id| g.node(id).is_some())?;
        let mut set = g.ancestors(id);
        set.extend(g.descendants(id));
        set.insert(id.clone());
        Some(set)
    }

    #[allow(clippy::too_many_arguments)]
    fn node_card(
        &self,
        id: &str,
        kind: &NodeKind,
        (sx, sy): (f32, f32),
        selected: bool,
        faded: bool,
        lit: bool,
        active: bool,
        run: Option<NodeRun>,
        group: Option<Vec<String>>,
        cx: &App,
    ) -> impl IntoElement + use<> {
        let t = cx.theme();
        let (warning, success, danger, muted) = (t.warning, t.success, t.danger, t.muted_foreground);
        let z = self.zoom;
        let (title, sub, icon) = describe(kind);
        let enabled = !matches!(kind, NodeKind::Op { enabled: false, .. });
        let accent = kind_color(kind);
        let handle = |left: bool| {
            let r = (HANDLE_R * z.max(0.6)).max(3.0);
            div()
                .absolute()
                .top(px(NODE_H * z / 2.0 - r))
                .when(left, |d| d.left(px(-r)))
                .when(!left, |d| d.right(px(-r)))
                .size(px(r * 2.0))
                .rounded_full()
                .border_2()
                .border_color(if selected || lit { t.primary } else { t.muted_foreground })
                .bg(if lit { t.primary } else { t.background })
        };
        // The last run: a badge in the title row and a line under it
        let z0 = z;
        let badge = run.as_ref().map(|r| {
            let d = px(10.0 * z0.max(0.6));
            match r {
                NodeRun::Running { .. } => div().flex_none().size(d).rounded_full().border_2().border_color(warning).into_any_element(),
                NodeRun::Done { took } => h_flex()
                    .flex_none()
                    .gap(px(3.0 * z0))
                    .child(div().size(d).rounded_full().bg(success))
                    .child(div().text_size(px(10.0 * z0)).text_color(muted).child(duration_label(*took)))
                    .into_any_element(),
                NodeRun::Failed(_) => div().flex_none().size(d).rounded_full().bg(danger).into_any_element(),
            }
        });
        // A recipe: a group of ops, shown as one card with a tab and what it holds
        let recipe_name = match kind {
            NodeKind::Recipe { recipe, .. } => Some(recipe.clone()),
            _ => None,
        };
        let group_line = group.as_ref().map(|ops| match ops.len() {
            0 => "no ops (not defined in this session)".to_string(),
            n => format!("{n} op{}: {}", if n == 1 { "" } else { "s" }, ops.join(" · ")),
        });
        let listed = group.clone().filter(|_| self.expanded.contains(id));
        let run_line = match &run {
            Some(NodeRun::Running { line, fraction, .. }) => {
                let pct = fraction.map(|f| format!("{:.0} % · ", f * 100.0)).unwrap_or_default();
                Some((format!("{pct}{}", if line.is_empty() { "running…" } else { line }), warning))
            }
            Some(NodeRun::Failed(e)) => Some((e.clone(), danger)),
            _ => None,
        };
        let inner_radius = px((12.0 * z - if selected { 2.0 } else { 1.0 }).max(0.0));
        div()
            .id(SharedString::from(format!("node-{id}")))
            .test_support()
            .absolute()
            .left(px(sx))
            .top(px(sy))
            .w(px(NODE_W * z))
            .h(px(NODE_H * z))
            .rounded(px(12.0 * z))
            .when(selected, |d| d.border_2())
            .when(!selected, |d| d.border_1())
            .border_color(if selected { t.primary } else { t.border })
            .bg(t.background)
            .shadow_sm()
            .when(faded, |d| d.opacity(0.35))
            .when(!enabled && !faded, |d| d.opacity(0.55))
            .child(
                // The kind's colour fills the card inside its border (with the card's own rounded
                // corners: clipping is rectangular, so a bar would poke out square), and the body
                // covers all of it but a strip on the left that follows the card's curve
                h_flex()
                    .size_full()
                    .overflow_hidden()
                    .rounded(inner_radius)
                    .bg(accent)
                    .child(
                        v_flex()
                            .flex_1()
                            .h_full()
                            .min_w_0()
                            .ml(px(4.0 * z.max(0.5)))
                            .bg(t.background)
                            .rounded_r(inner_radius)
                            .rounded_l(px(8.0 * z))
                            .px(px(10.0 * z))
                            .gap(px(3.0 * z))
                            .justify_center()
                            .child(
                                h_flex()
                                    .gap(px(6.0 * z))
                                    .min_w_0()
                                    .child(Icon::new(icon).size(px(14.0 * z)).text_color(accent))
                                    .child(div().flex_1().min_w_0().truncate().text_size(px(13.0 * z)).font_weight(FontWeight::SEMIBOLD).child(title))
                                    .when(active, |d| d.child(div().flex_none().px(px(5.0 * z)).rounded(px(4.0 * z)).bg(t.accent).text_size(px(10.0 * z)).child("shown")))
                                    .when(!enabled, |d| d.child(div().flex_none().text_size(px(10.0 * z)).text_color(t.muted_foreground).child("off")))
                                    .children(badge),
                            )
                            .when(z >= 0.45, |d| match (run_line, group_line) {
                                (Some((line, color)), _) => d.child(div().truncate().text_size(px(10.5 * z)).text_color(color).child(line)),
                                (None, Some(ops)) => d.child(div().truncate().text_size(px(11.0 * z)).text_color(t.muted_foreground).child(ops)),
                                (None, None) => d.child(div().truncate().text_size(px(11.0 * z)).font_family("monospace").text_color(t.muted_foreground).child(sub)),
                            }),
                    ),
            )
            .when(kind.has_input(), |d| d.child(handle(true)))
            .when(kind.has_output(), |d| d.child(handle(false)))
            // The group's tab above the card, and its ops listed under it when expanded
            .children(recipe_name.filter(|_| z >= 0.45).map(|name| {
                div()
                    .absolute()
                    .top(px(-RECIPE_TAB * z))
                    .left(px(10.0 * z))
                    .h(px(RECIPE_TAB * z))
                    .px(px(8.0 * z))
                    .rounded_t(px(6.0 * z))
                    .border_1()
                    .border_b_0()
                    .border_color(if selected { t.primary } else { t.border })
                    .bg(t.secondary)
                    .flex()
                    .items_center()
                    .text_size(px(10.0 * z))
                    .text_color(t.muted_foreground)
                    .child(format!("Recipe: {name}"))
            }))
            .children(listed.map(|ops| {
                v_flex()
                    .absolute()
                    .top(px(NODE_H * z + 4.0))
                    .left_0()
                    .w(px(NODE_W * z))
                    .p(px(6.0 * z))
                    .gap(px(2.0 * z))
                    .rounded(px(8.0 * z))
                    .border_1()
                    .border_color(t.border)
                    .bg(t.background)
                    .shadow_sm()
                    .children(ops.into_iter().enumerate().map(|(i, op)| {
                        h_flex()
                            .gap(px(6.0 * z))
                            .text_size(px(11.0 * z))
                            .child(div().w(px(14.0 * z)).text_color(t.muted_foreground).child(format!("{}", i + 1)))
                            .child(div().truncate().child(op))
                    }))
            }))
    }

    /// Buttons over the selected node.
    fn toolbar(&self, id: &str, kind: &NodeKind, (sx, sy): (f32, f32), has_input: bool, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let t = cx.theme();
        let me = cx.entity().downgrade();
        let mut bar = h_flex().gap_0p5();
        let button = |name: &'static str, icon: IconName, tip: &'static str, f: fn(&mut GraphCanvas, &str, &mut Context<GraphCanvas>)| {
            let (me, id) = (me.clone(), id.to_string());
            icon_button(SharedString::from(format!("node-tool-{name}")), Icon::new(icon), tip).xsmall().on_click(move |_, _, cx| {
                let _ = me.update(cx, |c, cx| f(c, &id, cx));
            })
        };
        match kind {
            NodeKind::Op { enabled, .. } | NodeKind::Recipe { enabled, .. } => {
                let (icon, tip) = if *enabled { (IconName::EyeOff, "Turn off (kept, skipped)") } else { (IconName::Eye, "Turn on") };
                let on = !*enabled;
                let (me, id) = (me.clone(), id.to_string());
                bar = bar.child(icon_button("node-tool-enable", Icon::new(icon), tip).xsmall().on_click(move |_, _, cx| {
                    let _ = me.update(cx, |c, cx| c.exec(SessionCommand::EnableNode { id: id.clone(), enabled: on }, cx));
                }));
            }
            NodeKind::View { .. } => bar = bar.child(button("show", IconName::ExternalLink, "Show in Explore (double-click)", |c, id, cx| c.open(id, cx))),
            NodeKind::Write {} => bar = bar.child(button("write", IconName::Play, "Write the processed file (double-click)", |c, id, cx| c.open(id, cx))),
            NodeKind::Source { .. } => {}
        }
        if matches!(kind, NodeKind::Recipe { .. }) {
            let shown = self.expanded.contains(id);
            bar = bar.child(button(if shown { "hide-ops" } else { "show-ops" }, IconName::Menu, if shown { "Hide its ops" } else { "Show all its ops" }, |c, id, cx| {
                if !c.expanded.remove(id) {
                    c.expanded.insert(id.to_string());
                }
                cx.notify();
            }));
            bar = bar.child(button("ungroup", IconName::Layers, "Ungroup: back to its ops", |c, id, cx| {
                c.exec(SessionCommand::Ungroup { id: id.to_string() }, cx);
            }));
        }
        if has_input {
            bar = bar.child(button("unlink", IconName::Unlink, "Disconnect its input", |c, id, cx| {
                c.exec(SessionCommand::Disconnect { to: id.to_string() }, cx);
            }));
        }
        bar = bar
            .child(button("duplicate", IconName::Copy, "Duplicate (Ctrl+D)", |c, _, cx| c.duplicate_selected(cx)))
            .child(button("delete", IconName::Trash, "Delete (Del)", |c, _, cx| c.delete_selected(cx)));
        // Above a recipe's tab
        let lift = if matches!(kind, NodeKind::Recipe { .. }) { RECIPE_TAB * self.zoom } else { 0.0 };
        div()
            .absolute()
            .left(px(sx))
            .top(px(sy - 34.0 - lift))
            .p_0p5()
            .rounded_md()
            .border_1()
            .border_color(t.border)
            .bg(t.popover)
            .shadow_md()
            .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
            .child(bar)
    }

    fn controls(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let t = cx.theme();
        let me = cx.entity().downgrade();
        let b = |name: &'static str, icon: IconName, tip: &'static str, f: fn(&mut GraphCanvas, &mut Context<GraphCanvas>)| {
            let me = me.clone();
            icon_button(SharedString::from(format!("canvas-{name}")), Icon::new(icon), tip).small().on_click(move |_, _, cx| {
                let _ = me.update(cx, |c, cx| f(c, cx));
            })
        };
        h_flex()
            .absolute()
            .left_3()
            .bottom_3()
            .gap_0p5()
            .p_0p5()
            .rounded_md()
            .border_1()
            .border_color(t.border)
            .bg(t.popover)
            .shadow_md()
            .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
            .child(b("zoom-out", IconName::ZoomOut, "Zoom out (Ctrl −)", |c, cx| c.zoom_by(0.8, None, cx)))
            .child(div().w(px(44.)).text_xs().text_center().child(format!("{:.0} %", self.zoom * 100.0)))
            .child(b("zoom-in", IconName::ZoomIn, "Zoom in (Ctrl +)", |c, cx| c.zoom_by(1.25, None, cx)))
            .child(b("fit", IconName::Maximize, "Fit the graph (F)", |c, cx| c.fit(cx)))
            .child(b("grid", IconName::LayoutGrid, "Dot grid", |c, cx| {
                c.show_grid = !c.show_grid;
                cx.notify();
            }))
            .child(b("minimap", IconName::Map, "Minimap", |c, cx| {
                c.show_minimap = !c.show_minimap;
                cx.notify();
            }))
    }

    /// The whole graph small, the view as a frame; a press moves the view there.
    fn minimap(&self, g: &GraphDoc, cx: &mut Context<Self>) -> Option<impl IntoElement + use<>> {
        let (x0, y0, x1, y1) = extent(g)?;
        let t = cx.theme();
        let size_now = self.bounds.get().size;
        let (vw0, vh0) = self.world((0.0, 0.0));
        let (vw1, vh1) = self.world((size_now.width.as_f32(), size_now.height.as_f32()));
        let (ex0, ey0, ex1, ey1) = (x0.min(vw0), y0.min(vh0), x1.max(vw1), y1.max(vh1));
        let scale = ((MAP_W - 12.0) / (ex1 - ex0).max(1.0)).min((MAP_H - 12.0) / (ey1 - ey0).max(1.0));
        let rects: Vec<((f32, f32, f32, f32), Hsla)> = g
            .nodes
            .iter()
            .map(|n| {
                let (x, y) = self.pos(&n.id, n.x, n.y);
                (((x - ex0) * scale + 6.0, (y - ey0) * scale + 6.0, NODE_W * scale, NODE_H * scale), kind_color(&n.kind))
            })
            .collect();
        let view = ((vw0 - ex0) * scale + 6.0, (vh0 - ey0) * scale + 6.0, (vw1 - vw0) * scale, (vh1 - vh0) * scale);
        let frame = t.primary;
        let me = cx.entity().downgrade();
        let to_world = move |local: (f32, f32)| ((local.0 - 6.0) / scale + ex0, (local.1 - 6.0) / scale + ey0);
        let map_bounds = Rc::new(Cell::new(Bounds::<Pixels>::default()));
        let mb = map_bounds.clone();
        Some(
            div()
                .id("canvas-minimap")
                .absolute()
                .right_3()
                .bottom_3()
                .w(px(MAP_W))
                .h(px(MAP_H))
                .rounded_md()
                .border_1()
                .border_color(t.border)
                .bg(t.popover)
                .shadow_md()
                .overflow_hidden()
                .on_mouse_down(MouseButton::Left, move |e, _, cx| {
                    cx.stop_propagation();
                    let o = map_bounds.get().origin;
                    let p = to_world(((e.position.x - o.x).as_f32(), (e.position.y - o.y).as_f32()));
                    let _ = me.update(cx, |c, cx| c.centre_on(p, cx));
                })
                .child(
                    canvas(
                        move |bounds, _, _| mb.set(bounds),
                        move |bounds, _, window, _| {
                            let o = bounds.origin;
                            for ((x, y, w, h), c) in &rects {
                                window.paint_quad(fill(Bounds { origin: point(o.x + px(*x), o.y + px(*y)), size: size(px(w.max(2.0)), px(h.max(2.0))) }, c.opacity(0.85)));
                            }
                            let (x, y, w, h) = view;
                            let r = Bounds { origin: point(o.x + px(x), o.y + px(y)), size: size(px(w), px(h)) };
                            window.paint_quad(gpui_kit::outline(r, frame, gpui_kit::BorderStyle::Solid));
                        },
                    )
                    .size_full(),
                ),
        )
    }
}

/// Snaps a canvas coordinate to the node grid.
fn snap(v: f32) -> f32 {
    (v / SNAP).round() * SNAP
}

/// The colour of a node kind (its strip and minimap mark).
fn kind_color(kind: &NodeKind) -> Hsla {
    match kind {
        NodeKind::Source { .. } => gpui_kit::hsla(152. / 360., 0.55, 0.42, 1.0),
        NodeKind::Op { .. } => gpui_kit::hsla(239. / 360., 0.84, 0.67, 1.0),
        NodeKind::View { .. } => gpui_kit::hsla(271. / 360., 0.70, 0.60, 1.0),
        NodeKind::Write {} => gpui_kit::hsla(27. / 360., 0.90, 0.55, 1.0),
        NodeKind::Recipe { .. } => gpui_kit::hsla(186. / 360., 0.70, 0.45, 1.0),
    }
}

/// The graph's extent in canvas units.
fn extent(g: &GraphDoc) -> Option<(f32, f32, f32, f32)> {
    let mut it = g.nodes.iter();
    let first = it.next()?;
    let init = (first.x, first.y, first.x + NODE_W, first.y + NODE_H);
    Some(it.fold(init, |(a, b, c, d), n| (a.min(n.x), b.min(n.y), c.max(n.x + NODE_W), d.max(n.y + NODE_H))))
}

/// Radius of an edge's elbows (screen px, at most half the room there is).
const ELBOW: f32 = 12.0;

/// The corners of the edge from `a` to `b` when it runs left to right (out, across at the
/// middle, in); `None` when `b` is behind `a` (then it is a curve).
fn elbow_points(a: (f32, f32), b: (f32, f32)) -> Option<[(f32, f32); 4]> {
    (b.0 - a.0 >= 2.0 * ELBOW).then(|| {
        let mx = (a.0 + b.0) / 2.0;
        [a, (mx, a.1), (mx, b.1), b]
    })
}

/// An edge from `a` (an output) to `b` (an input), screen px: left to right with rounded elbows
/// (an S-curve when `b` is behind `a`).
fn edge_path(a: (f32, f32), b: (f32, f32), width: f32) -> Option<gpui_kit::Path<Pixels>> {
    let mut p = PathBuilder::stroke(px(width));
    let pt = |x: f32, y: f32| point(px(x), px(y));
    p.move_to(pt(a.0, a.1));
    match elbow_points(a, b) {
        Some([_, (mx, _), _, _]) if (b.1 - a.1).abs() > 0.5 => {
            let dy = b.1 - a.1;
            let r = ELBOW.min(dy.abs() / 2.0).min((b.0 - a.0) / 2.0);
            let s = dy.signum();
            p.line_to(pt(mx - r, a.1));
            p.curve_to(pt(mx, a.1 + s * r), pt(mx, a.1));
            p.line_to(pt(mx, b.1 - s * r));
            p.curve_to(pt(mx + r, b.1), pt(mx, b.1));
            p.line_to(pt(b.0, b.1));
        }
        Some(_) => p.line_to(pt(b.0, b.1)),
        None => {
            let dx = ((b.0 - a.0).abs() * 0.5).max(40.0);
            p.cubic_bezier_to(pt(b.0, b.1), pt(a.0 + dx, a.1), pt(b.0 - dx, b.1));
        }
    }
    p.build().ok()
}

/// Points every `step` px along the edge from `a` to `b`, shifted by `phase` (0..step): data
/// flowing into a running node.
fn flow_points(a: (f32, f32), b: (f32, f32), step: f32, phase: f32) -> Vec<(f32, f32)> {
    let corners = elbow_points(a, b).map(|c| c.to_vec()).unwrap_or_else(|| vec![a, b]);
    let mut out = Vec::new();
    let mut carry = phase;
    for w in corners.windows(2) {
        let (p, q) = (w[0], w[1]);
        let len = (q.0 - p.0).hypot(q.1 - p.1);
        let mut d = carry;
        while d < len {
            let f = d / len;
            out.push((p.0 + (q.0 - p.0) * f, p.1 + (q.1 - p.1) * f));
            d += step;
        }
        carry = d - len;
    }
    out
}

/// `2.3 s`, `4 min 12 s`, `1 h 05 min`.
fn duration_label(d: std::time::Duration) -> String {
    let s = d.as_secs_f64();
    if s < 60.0 {
        format!("{s:.1} s")
    } else if s < 3600.0 {
        format!("{} min {:02} s", (s / 60.0) as u64, (s % 60.0) as u64)
    } else {
        format!("{} h {:02} min", (s / 3600.0) as u64, ((s % 3600.0) / 60.0) as u64)
    }
}

/// The arrow head at an input.
fn arrow(b: (f32, f32), s: f32) -> Option<gpui_kit::Path<Pixels>> {
    let mut p = PathBuilder::fill();
    p.move_to(point(px(b.0), px(b.1)));
    p.line_to(point(px(b.0 - s * 1.6), px(b.1 - s)));
    p.line_to(point(px(b.0 - s * 1.6), px(b.1 + s)));
    p.line_to(point(px(b.0), px(b.1)));
    p.build().ok()
}

impl Render for GraphCanvas {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme().clone();
        let s = self.store.read(cx);
        let g = s.doc.graph.clone();
        let active = s.doc.active_view.clone();
        let z = self.zoom;
        let highlight = self.highlight(&g);
        let faded = |id: &str| highlight.as_ref().is_some_and(|h| !h.contains(id));

        // Edges and the dot grid, painted under the cards
        let status = s.node_status.clone();
        let recipes = s.doc.recipes.clone();
        let running = |id: &str| matches!(status.get(id), Some(NodeRun::Running { .. }));
        let mut edges: Vec<((f32, f32), (f32, f32), bool, bool, bool)> = Vec::new();
        for e in &g.edges {
            let (Some(a), Some(b)) = (g.node(&e.from), g.node(&e.to)) else { continue };
            let pa = self.pos(&a.id, a.x, a.y);
            let pb = self.pos(&b.id, b.x, b.y);
            let from = self.screen((pa.0 + NODE_W, pa.1 + NODE_H / 2.0));
            let to = self.screen((pb.0, pb.1 + NODE_H / 2.0));
            let lit = highlight.as_ref().is_some_and(|h| h.contains(&e.from) && h.contains(&e.to));
            let dim = highlight.is_some() && !lit;
            edges.push((from, to, lit, dim, running(&e.to)));
        }
        let pending = match &self.drag {
            Some(Drag::Connect { from, to }) => g.node(from).map(|n| {
                let p = self.pos(&n.id, n.x, n.y);
                (self.screen((p.0 + NODE_W, p.1 + NODE_H / 2.0)), *to)
            }),
            _ => None,
        };
        let marquee = match &self.drag {
            Some(Drag::Marquee { a, b, .. }) => Some((*a, *b)),
            _ => None,
        };
        let (pan, show_grid) = (self.pan, self.show_grid);
        let (dot, line, lit_line, sel_fill, flow_color) = (t.muted_foreground.opacity(0.35), t.muted_foreground, t.primary, t.primary.opacity(0.08), t.warning);
        let bounds_cell = self.bounds.clone();
        let plane = canvas(
            move |bounds, _, _| bounds_cell.set(bounds),
            move |bounds, _, window, _| {
                let o = bounds.origin;
                let (w, h) = (bounds.size.width.as_f32(), bounds.size.height.as_f32());
                if show_grid {
                    let mut step = GRID * z;
                    while step < 10.0 {
                        step *= 2.0;
                    }
                    let r = (1.2 * z.clamp(0.6, 1.4)).max(1.0);
                    let mut x = pan.0.rem_euclid(step);
                    while x < w {
                        let mut y = pan.1.rem_euclid(step);
                        while y < h {
                            window.paint_quad(fill(Bounds { origin: point(o.x + px(x - r / 2.0), o.y + px(y - r / 2.0)), size: size(px(r), px(r)) }, dot));
                            y += step;
                        }
                        x += step;
                    }
                }
                let at = |p: (f32, f32)| (p.0 + o.x.as_f32(), p.1 + o.y.as_f32());
                // Dimmed edges first, the highlighted path on top
                let mut order: Vec<&((f32, f32), (f32, f32), bool, bool, bool)> = edges.iter().collect();
                order.sort_by_key(|e| e.2);
                // Data flowing into running nodes: dots moving along their edges
                let phase = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| (d.as_millis() % 1000) as f32 / 1000.0);
                let mut flowing = false;
                for (a, b, _, _, flow) in order.iter().copied() {
                    if !*flow {
                        continue;
                    }
                    flowing = true;
                    let step = 14.0 * z.clamp(0.7, 1.3);
                    let r = 2.0 * z.clamp(0.7, 1.3);
                    for (x, y) in flow_points(at(*a), at(*b), step, phase * step) {
                        window.paint_quad(fill(Bounds { origin: point(px(x - r), px(y - r)), size: size(px(2.0 * r), px(2.0 * r)) }, flow_color).corner_radii(px(r)));
                    }
                }
                if flowing {
                    window.request_animation_frame();
                }
                for (a, b, lit, dim, _) in order {
                    let (c, width) = if *lit { (lit_line, 2.5) } else if *dim { (line.opacity(0.25), 1.5) } else { (line.opacity(0.8), 1.5) };
                    if let Some(p) = edge_path(at(*a), at(*b), width * z.clamp(0.7, 1.3)) {
                        window.paint_path(p, c);
                    }
                    if let Some(p) = arrow(at(*b), 4.0 * z.clamp(0.7, 1.3)) {
                        window.paint_path(p, c);
                    }
                }
                if let Some((a, b)) = pending {
                    if let Some(p) = edge_path(at(a), at(b), 2.0) {
                        window.paint_path(p, lit_line);
                    }
                }
                if let Some((a, b)) = marquee {
                    let (a, b) = (at(a), at(b));
                    let r = Bounds { origin: point(px(a.0.min(b.0)), px(a.1.min(b.1))), size: size(px((a.0 - b.0).abs()), px((a.1 - b.1).abs())) };
                    window.paint_quad(fill(r, sel_fill));
                    window.paint_quad(gpui_kit::outline(r, lit_line, gpui_kit::BorderStyle::Solid));
                }
            },
        )
        .absolute()
        .size_full();

        let cards: Vec<_> = g
            .nodes
            .iter()
            .map(|n| {
                let p = self.pos(&n.id, n.x, n.y);
                let sp = self.screen(p);
                let shown = active.as_deref() == Some(n.id.as_str());
                let lit = highlight.as_ref().is_some_and(|h| h.contains(&n.id));
                let group = match &n.kind {
                    NodeKind::Recipe { recipe, .. } => Some(recipes.get(recipe).map(|r| r.ops.iter().map(|o| type_name(&o.node).to_string()).collect()).unwrap_or_default()),
                    _ => None,
                };
                self.node_card(&n.id, &n.kind, sp, self.selected.contains(&n.id), faded(&n.id), lit, shown, status.get(&n.id).cloned(), group, cx).into_any_element()
            })
            .collect();
        let toolbar = self.single().cloned().and_then(|id| {
            let n = g.node(&id)?;
            let sp = self.screen(self.pos(&n.id, n.x, n.y));
            Some(self.toolbar(&id, &n.kind, sp, g.input(&id).is_some(), cx).into_any_element())
        });
        let empty = g.nodes.is_empty().then(|| {
            v_flex()
                .absolute()
                .top_0()
                .left_0()
                .size_full()
                .items_center()
                .justify_center()
                .gap_1()
                .child(div().text_lg().font_weight(FontWeight::SEMIBOLD).child("An empty canvas"))
                .child(div().text_sm().text_color(t.muted_foreground).child("Drag a recording and operations from the right, or open a recording in Explore."))
        });
        let minimap = if self.show_minimap { self.minimap(&g, cx).map(IntoElement::into_any_element) } else { None };
        if self.fit_pending && self.bounds.get().size.width.as_f32() > 10.0 {
            self.fit_pending = false;
            let me = cx.entity().downgrade();
            cx.defer(move |cx| {
                let _ = me.update(cx, |c, cx| c.fit(cx));
            });
        }
        let cursor = match self.drag {
            Some(Drag::Pan { .. }) => gpui_kit::CursorStyle::ClosedHand,
            _ if self.space || self.hand => gpui_kit::CursorStyle::OpenHand,
            _ => gpui_kit::CursorStyle::Arrow,
        };
        div()
            .id("graph-canvas")
            .test_support()
            .track_focus(&self.focus)
            .key_context("GraphCanvas")
            .relative()
            .size_full()
            .overflow_hidden()
            .bg(t.muted.opacity(0.35))
            .cursor(cursor)
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_down))
            .on_mouse_down(MouseButton::Middle, cx.listener(Self::on_down))
            .on_mouse_move(cx.listener(Self::on_move))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_up))
            .on_mouse_up(MouseButton::Middle, cx.listener(Self::on_up))
            .on_scroll(cx.listener(Self::on_wheel))
            .on_key_down(cx.listener(Self::on_key))
            .on_key_up(cx.listener(Self::on_key_up))
            .on_drop(cx.listener(|this, d: &CatalogDrag, window, cx| {
                let local = this.local(window.mouse_position());
                let (x, y) = this.world(local);
                this.add(&d.template, Some((snap(x - NODE_W / 2.0), snap(y - NODE_H / 2.0))), window, cx);
            }))
            .on_drop(cx.listener(|this, d: &RecipeDrag, window, cx| {
                let local = this.local(window.mouse_position());
                let (x, y) = this.world(local);
                this.add_recipe(d.recipe.clone(), Some((snap(x - NODE_W / 2.0), snap(y - NODE_H / 2.0))), cx);
            }))
            .child(plane)
            .children(cards)
            .children(toolbar)
            .children(empty)
            .child(self.controls(cx))
            .children(minimap)
    }
}

/// A recipe dragged from the library.
#[derive(Debug, Clone)]
pub struct RecipeDrag {
    pub recipe: crate::domain::recipe::RecipeDoc,
}

/// What a dragged tree item looks like under the pointer.
pub struct DragGhost(pub SharedString);

impl Render for DragGhost {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        div().px_2().py_1().rounded_md().border_1().border_color(t.primary).bg(t.background).shadow_md().text_sm().child(self.0.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_run_left_to_right_with_elbows() {
        assert_eq!(elbow_points((0.0, 0.0), (100.0, 40.0)), Some([(0.0, 0.0), (50.0, 0.0), (50.0, 40.0), (100.0, 40.0)]));
        assert_eq!(elbow_points((100.0, 0.0), (90.0, 40.0)), None, "behind: a curve");
        assert!(edge_path((0.0, 0.0), (100.0, 40.0), 2.0).is_some() && edge_path((100.0, 0.0), (10.0, 40.0), 2.0).is_some());
        // Flow: evenly spaced along the corners, shifted by the phase
        let pts = flow_points((0.0, 0.0), (100.0, 0.0), 10.0, 5.0);
        assert_eq!(pts.len(), 10);
        assert_eq!(pts[0], (5.0, 0.0));
        let bent = flow_points((0.0, 0.0), (100.0, 40.0), 10.0, 0.0);
        assert!(bent.iter().any(|&(x, y)| x == 50.0 && y > 0.0 && y < 40.0), "the middle leg");
        assert_eq!(duration_label(std::time::Duration::from_millis(2300)), "2.3 s");
        assert_eq!(duration_label(std::time::Duration::from_secs(252)), "4 min 12 s");
        assert_eq!(duration_label(std::time::Duration::from_secs(3900)), "1 h 05 min");
    }
}
