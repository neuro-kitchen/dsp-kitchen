//! The session's processing graph: what the Pipeline canvas draws and edits. Nodes are sources
//! (a recording, optionally a channel subset), ops (a dspk-pipeline node), views (a stream shown
//! in Explore: its display and what plugins keep for it) and writes (a processed file); edges
//! carry the signal from a node's output to another's input.
//!
//! Every node has at most one input and its output may feed any number of nodes, so one
//! recording can feed several branches (a spike band and an LFP band, each to its own view).
//! Each view or write is therefore one chain: the path from its source through its ops. The
//! graph is the truth; the session's `views` are compiled from it ([`GraphDoc::compile`]).

use std::collections::{BTreeMap, BTreeSet};

use dspk_pipeline::Node;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::recipe::RecipeDoc;
use super::session::{op_name, DisplayDoc, OpDoc, RecordingRef, ViewDoc};

/// The session's recipe definitions, by name.
pub type Recipes = BTreeMap<String, RecipeDoc>;

/// Spacing of nodes laid out by the app (canvas units): one column per step, one row per chain.
pub const COLUMN: f32 = 260.;
pub const ROW: f32 = 150.;

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GraphDoc {
    #[serde(default)]
    pub nodes: Vec<GraphNode>,
    #[serde(default)]
    pub edges: Vec<EdgeDoc>,
}

/// A node and where it sits on the canvas (canvas units; y grows downwards).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GraphNode {
    /// Unique in the session; a view node's id is its view's id.
    pub id: String,
    #[serde(default)]
    pub x: f32,
    #[serde(default)]
    pub y: f32,
    pub kind: NodeKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NodeKind {
    /// A recording (one source of one file, optionally a channel subset): no input.
    Source { recording: RecordingRef },
    /// Processing applied to every channel of its input.
    Op {
        /// Off: kept in the graph, skipped when processing.
        #[serde(default = "yes")]
        enabled: bool,
        node: Node,
    },
    /// A stream shown in Explore (a view of the session).
    View {
        name: String,
        #[serde(default)]
        display: DisplayDoc,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        #[schemars(with = "BTreeMap<String, serde_json::Value>")]
        plugins: BTreeMap<String, serde_json::Value>,
    },
    /// The chain written to `<recording>.<id>.processed.zarr`.
    Write {},
    /// A use of recipe `recipe` (defined in the session's `recipes`), with its own values.
    Recipe {
        recipe: String,
        #[serde(default = "yes")]
        enabled: bool,
        /// `"<op>.<key>"` → value, over the recipe's.
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        #[schemars(with = "BTreeMap<String, serde_json::Value>")]
        overrides: BTreeMap<String, serde_json::Value>,
    },
}

fn yes() -> bool {
    true
}

impl NodeKind {
    /// Whether the node takes an input (everything but a source).
    pub fn has_input(&self) -> bool {
        !matches!(self, NodeKind::Source { .. })
    }

    /// Whether the node's output can feed others (sources and ops).
    pub fn has_output(&self) -> bool {
        matches!(self, NodeKind::Source { .. } | NodeKind::Op { .. } | NodeKind::Recipe { .. })
    }
}

/// The signal of `from`'s output into `to`'s input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EdgeDoc {
    pub from: String,
    pub to: String,
}

impl GraphDoc {
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn node(&self, id: &str) -> Option<&GraphNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn node_mut(&mut self, id: &str) -> Option<&mut GraphNode> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    pub fn contains(&self, id: &str) -> bool {
        self.node(id).is_some()
    }

    /// The node feeding `id`.
    pub fn input(&self, id: &str) -> Option<&str> {
        self.edges.iter().find(|e| e.to == id).map(|e| e.from.as_str())
    }

    /// The nodes `id` feeds, in edge order.
    pub fn outputs(&self, id: &str) -> Vec<&str> {
        self.edges.iter().filter(|e| e.from == id).map(|e| e.to.as_str()).collect()
    }

    /// `id`'s chain from its source: `[source, op…, id]`; `None` when it does not reach one.
    pub fn path(&self, id: &str) -> Option<Vec<String>> {
        let mut path = vec![id.to_string()];
        let mut cur = id;
        // At most one input per node: the walk ends at a source, at a loose end, or (a cycle a
        // hand-edited file may hold) after visiting every node
        for _ in 0..=self.nodes.len() {
            if matches!(self.node(cur)?.kind, NodeKind::Source { .. }) {
                path.reverse();
                return Some(path);
            }
            cur = self.input(cur)?;
            path.push(cur.to_string());
        }
        None
    }

    /// Every node upstream of `id`.
    pub fn ancestors(&self, id: &str) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        let mut cur = id;
        while let Some(i) = self.input(cur) {
            if !out.insert(i.to_string()) {
                break;
            }
            cur = i;
        }
        out
    }

    /// Every node downstream of `id`.
    pub fn descendants(&self, id: &str) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        let mut stack = vec![id.to_string()];
        while let Some(n) = stack.pop() {
            for o in self.outputs(&n) {
                if out.insert(o.to_string()) {
                    stack.push(o.to_string());
                }
            }
        }
        out
    }

    /// An id not used yet: `base` itself, else `base1`, `base2`, …
    pub fn fresh_id(&self, base: &str) -> String {
        if !self.contains(base) && base != "read" && base != "write" {
            return base.to_string();
        }
        (1..).map(|i| format!("{base}{i}")).find(|id| !self.contains(id)).expect("ids are unbounded")
    }

    /// `prefix1`, `prefix2`, … (sources `s1`, writes `w1`).
    pub fn numbered_id(&self, prefix: &str, taken: impl Fn(&str) -> bool) -> String {
        (1..).map(|i| format!("{prefix}{i}")).find(|id| !self.contains(id) && !taken(id)).expect("ids are unbounded")
    }

    /// The y of a new row under every node.
    pub fn next_row(&self) -> f32 {
        self.nodes.iter().map(|n| n.y + ROW).fold(0.0, f32::max)
    }

    /// Connects `from`'s output to `to`'s input (replacing `to`'s input).
    pub fn connect(&mut self, from: &str, to: &str) -> Result<(), String> {
        let (Some(a), Some(b)) = (self.node(from), self.node(to)) else { return Err(format!("no node {from:?} or {to:?}")) };
        if !a.kind.has_output() {
            return Err(format!("{from} has no output (views and writes end a chain)"));
        }
        if !b.kind.has_input() {
            return Err(format!("{to} is a recording: it takes no input"));
        }
        if from == to || self.descendants(to).contains(from) {
            return Err("that connection would make a loop".into());
        }
        self.edges.retain(|e| e.to != to);
        self.edges.push(EdgeDoc { from: from.to_string(), to: to.to_string() });
        Ok(())
    }

    /// Removes `to`'s input.
    pub fn disconnect(&mut self, to: &str) {
        self.edges.retain(|e| e.to != to);
    }

    /// Removes a node and its edges.
    pub fn remove(&mut self, id: &str) {
        self.nodes.retain(|n| n.id != id);
        self.edges.retain(|e| e.from != id && e.to != id);
    }

    /// Removes a node, its input feeding what it fed.
    pub fn remove_bridging(&mut self, id: &str) {
        self.unlink_bridging(id);
        self.nodes.retain(|n| n.id != id);
    }

    /// Takes a node out of its chains (its input feeds what it fed); the node stays, unlinked.
    pub fn unlink_bridging(&mut self, id: &str) {
        let input = self.input(id).map(str::to_string);
        let outputs: Vec<String> = self.outputs(id).into_iter().map(str::to_string).collect();
        self.edges.retain(|e| e.from != id && e.to != id);
        if let Some(i) = input {
            for o in outputs {
                self.edges.push(EdgeDoc { from: i.clone(), to: o });
            }
        }
    }

    /// Puts `id` on the edge `a → b` (`a → id → b`).
    pub fn splice(&mut self, a: &str, b: &str, id: &str) {
        self.edges.retain(|e| !(e.from == a && e.to == b));
        self.edges.push(EdgeDoc { from: a.to_string(), to: id.to_string() });
        self.edges.push(EdgeDoc { from: id.to_string(), to: b.to_string() });
    }

    /// The recording and ops (enabled or not) of `id`'s chain; a recipe node gives its ops
    /// (`<node>.<op>`, its overrides applied).
    pub fn chain(&self, id: &str, recipes: &Recipes) -> Option<(RecordingRef, Vec<OpDoc>)> {
        let path = self.path(id)?;
        let NodeKind::Source { recording } = &self.node(&path[0])?.kind else { return None };
        let mut ops = Vec::new();
        for n in &path[1..path.len() - 1] {
            match &self.node(n)?.kind {
                NodeKind::Op { enabled, node } => ops.push(OpDoc { id: n.clone(), enabled: *enabled, node: node.clone() }),
                NodeKind::Recipe { recipe, enabled, overrides } => {
                    // A use of a recipe the session does not define breaks the chain
                    let def = recipes.get(recipe)?;
                    ops.extend(def.ops_with(overrides).into_iter().map(|op| OpDoc { id: format!("{n}.{}", op.id), enabled: op.enabled && *enabled, node: op.node }));
                }
                _ => {}
            }
        }
        Some((recording.clone(), ops))
    }

    /// The views: each view node that reaches a source, in node order.
    pub fn compile(&self, recipes: &Recipes) -> Vec<ViewDoc> {
        self.nodes
            .iter()
            .filter_map(|n| {
                let NodeKind::View { name, display, plugins } = &n.kind else { return None };
                let (recording, ops) = self.chain(&n.id, recipes)?;
                Some(ViewDoc { id: n.id.clone(), name: name.clone(), recording, ops, display: display.clone(), plugins: plugins.clone() })
            })
            .collect()
    }

    /// The graph of a document that has views only (written before the graph, or by hand): one
    /// row per view, its source, its ops, the view.
    pub fn from_views(views: &[ViewDoc]) -> Self {
        let mut g = GraphDoc::default();
        let view_ids: BTreeSet<&str> = views.iter().map(|v| v.id.as_str()).collect();
        for (row, v) in views.iter().enumerate() {
            let y = row as f32 * ROW;
            let s = g.numbered_id("s", |id| view_ids.contains(id));
            g.nodes.push(GraphNode { id: s.clone(), x: 0.0, y, kind: NodeKind::Source { recording: v.recording.clone() } });
            let mut last = s;
            for (k, op) in v.ops.iter().enumerate() {
                let id = if g.contains(&op.id) || view_ids.contains(op.id.as_str()) { g.fresh_id(op_name(&op.node)) } else { op.id.clone() };
                g.nodes.push(GraphNode { id: id.clone(), x: (k + 1) as f32 * COLUMN, y, kind: NodeKind::Op { enabled: op.enabled, node: op.node.clone() } });
                g.edges.push(EdgeDoc { from: last, to: id.clone() });
                last = id;
            }
            let kind = NodeKind::View { name: v.name.clone(), display: v.display.clone(), plugins: v.plugins.clone() };
            g.nodes.push(GraphNode { id: v.id.clone(), x: (v.ops.len() + 1) as f32 * COLUMN, y, kind });
            g.edges.push(EdgeDoc { from: last, to: v.id.clone() });
        }
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, kind: NodeKind) -> GraphNode {
        GraphNode { id: id.into(), x: 0.0, y: 0.0, kind }
    }

    fn view(name: &str) -> NodeKind {
        NodeKind::View { name: name.into(), display: DisplayDoc::default(), plugins: BTreeMap::new() }
    }

    /// s1 → hp → cmr → spikes, and s1 → lp → lfp
    fn branched() -> GraphDoc {
        let rec = RecordingRef { path: "/d/rec.bin".into(), source: None, channels: None };
        let hp: Node = serde_json::from_value(serde_json::json!({"op": "iir", "band": "highpass", "cutoff": [300.0]})).unwrap();
        let lp: Node = serde_json::from_value(serde_json::json!({"op": "iir", "band": "lowpass", "cutoff": [300.0]})).unwrap();
        let mut g = GraphDoc::default();
        g.nodes = vec![
            node("s1", NodeKind::Source { recording: rec }),
            node("hp", NodeKind::Op { enabled: true, node: hp }),
            node("cmr", NodeKind::Op { enabled: true, node: Node::Cmr {} }),
            node("spikes", view("Spikes")),
            node("lp", NodeKind::Op { enabled: true, node: lp }),
            node("lfp", view("LFP")),
        ];
        for (a, b) in [("s1", "hp"), ("hp", "cmr"), ("cmr", "spikes"), ("s1", "lp"), ("lp", "lfp")] {
            g.connect(a, b).unwrap();
        }
        g
    }

    #[test]
    fn test_branches_compile_to_one_view_each() {
        let views = branched().compile(&Recipes::new());
        assert_eq!(views.len(), 2);
        assert_eq!(views[0].ops.iter().map(|o| o.id.as_str()).collect::<Vec<_>>(), ["hp", "cmr"]);
        assert_eq!(views[1].ops.iter().map(|o| o.id.as_str()).collect::<Vec<_>>(), ["lp"]);
    }

    #[test]
    fn test_paths_and_highlight_sets() {
        let g = branched();
        assert_eq!(g.path("spikes").unwrap(), ["s1", "hp", "cmr", "spikes"]);
        assert_eq!(g.ancestors("cmr").into_iter().collect::<Vec<_>>(), ["hp", "s1"]);
        assert_eq!(g.descendants("s1").len(), 5);
        assert!(g.descendants("hp").contains("spikes") && !g.descendants("hp").contains("lfp"));
    }

    #[test]
    fn test_connections_are_checked() {
        let mut g = branched();
        assert!(g.connect("spikes", "lp").is_err(), "a view has no output");
        assert!(g.connect("cmr", "s1").is_err(), "a source has no input");
        assert!(g.connect("cmr", "hp").is_err(), "a loop");
        // A new input replaces the old one: lfp now hangs off the spike band
        g.connect("cmr", "lfp").unwrap();
        assert_eq!(g.input("lfp"), Some("cmr"));
        assert_eq!(g.edges.iter().filter(|e| e.to == "lfp").count(), 1);
    }

    #[test]
    fn test_removing_bridges_and_loose_views_drop_out() {
        let mut g = branched();
        g.remove_bridging("hp");
        assert_eq!(g.path("spikes").unwrap(), ["s1", "cmr", "spikes"]);
        g.disconnect("lfp");
        assert_eq!(g.compile(&Recipes::new()).len(), 1, "a view without a source is not a view yet");
    }

    #[test]
    fn test_from_views_round_trips() {
        let views = branched().compile(&Recipes::new());
        let again = GraphDoc::from_views(&views).compile(&Recipes::new());
        assert_eq!(again.iter().map(|v| (&v.id, v.ops.len())).collect::<Vec<_>>(), views.iter().map(|v| (&v.id, v.ops.len())).collect::<Vec<_>>());
    }
}
