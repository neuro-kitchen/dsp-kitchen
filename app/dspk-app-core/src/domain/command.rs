//! Commands: every change to the session document. The UI and an agent (future MCP server) send
//! the same commands, so there is one code path; each command is plain data (serializable, so an
//! agent sends it as JSON) and [`apply`] is pure (no I/O, no device), so it is tested directly.
//!
//! Two families, one graph: view commands (`add_op`, `set_op`, …) act on a view's chain (the
//! path from its recording to the view node), graph commands (`add_node`, `connect`, …) on the
//! Pipeline canvas. After every command the views are compiled from the graph again.
//!
//! Running work (preview, write, sort) is not a document change: the data layer does it and
//! records its result with a command (e.g. a sort: [`SessionCommand::SetPluginData`]).

use dspk_pipeline::Node;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::graph::{GraphNode, NodeKind, COLUMN};
use super::epoch::{EpochDoc, EventSetDoc, Placement};
use super::recipe::RecipeDoc;
use super::session::{op_name, DisplayDoc, RecordingRef, SessionDoc};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
pub enum SessionCommand {
    /// Adds a recording and a view of it (all its channels, no ops) and shows it.
    OpenView {
        recording: RecordingRef,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// Copies a view (its chain, display, plugin data) into a new one, optionally reading only
    /// `channels` (indices into the source, in order): how some channels get other processing.
    CopyView {
        view: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        channels: Option<Vec<usize>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// Removes a view and what only fed it.
    CloseView { view: String },
    RenameView { view: String, name: String },
    /// Shows a view.
    SelectView { view: String },
    /// Inserts an op at `index` (default: the end) of a view's chain.
    AddOp {
        view: String,
        node: Node,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    /// Replaces an op's parameters (the node may not change kind to `read` / `write`).
    SetOp { view: String, op: String, node: Node },
    EnableOp { view: String, op: String, enabled: bool },
    /// Removes an op (from every chain through it: its input feeds what it fed).
    RemoveOp { view: String, op: String },
    /// Moves an op to `index` in the view's chain (an op other chains share stays put).
    MoveOp { view: String, op: String, index: usize },
    SetDisplay { view: String, display: DisplayDoc },
    /// Replaces what plugin `plugin` keeps for a view (`null` removes it), e.g.
    /// `{"plugin": "synapse", "data": {"sorting": "rec.kilosort4.sorting.zarr"}}`.
    SetPluginData { view: String, plugin: String, data: serde_json::Value },
    /// Adds a node at (`x`, `y`), fed by `after` when given.
    AddNode {
        kind: NodeKind,
        #[serde(default)]
        x: f32,
        #[serde(default)]
        y: f32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        after: Option<String>,
    },
    /// Removes nodes and their edges.
    RemoveNodes { nodes: Vec<String> },
    /// Places nodes.
    MoveNodes { moves: Vec<NodeMove> },
    /// Feeds `to` from `from` (replacing `to`'s input).
    Connect { from: String, to: String },
    /// Removes `to`'s input.
    Disconnect { to: String },
    /// Replaces an op node's parameters.
    SetNode { id: String, node: Node },
    EnableNode { id: String, enabled: bool },
    /// Points a source node at another recording (or channel subset).
    SetSource { id: String, recording: RecordingRef },
    /// Adds (or replaces) a recipe definition in the session (e.g. from the library).
    AddRecipe { recipe: RecipeDoc },
    /// Turns a chain of op nodes into a recipe and one node using it, wired as they were.
    MakeRecipe {
        nodes: Vec<String>,
        name: String,
        #[serde(default)]
        labels: Vec<String>,
    },
    /// A use's own value of `key` (`"<op>.<key>"`); `null` goes back to the recipe's.
    SetRecipeOverride { id: String, key: String, value: serde_json::Value },
    /// Edits the recipe itself (every use follows): op `op`'s values become `node`'s.
    SetRecipeOp { recipe: String, op: String, node: Node },
    /// Renames a recipe and relabels it (every use follows).
    SetRecipeInfo { recipe: String, name: String, labels: Vec<String>, description: String },
    /// Replaces a recipe node by its ops (with its overrides), wired as it was.
    Ungroup { id: String },
    /// Adds an epoch, or replaces the one with its id (an empty id: a new one).
    SetEpoch { epoch: EpochDoc },
    RemoveEpoch { id: String },
    /// Adds several epochs as one change (one undo step); none when one of them is invalid.
    SetEpochs { epochs: Vec<EpochDoc> },
    /// Adds an event set (empty id: a new one) or replaces the one with its id (its look).
    SetEventSet { set: EventSetDoc },
    RemoveEventSet { id: String },
    /// Puts a recording at `start_sec` on the session's time.
    PlaceRecording {
        path: std::path::PathBuf,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<String>,
        start_sec: f64,
    },
    /// Takes a recording off the session's time.
    UnplaceRecording { path: std::path::PathBuf },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NodeMove {
    pub id: String,
    pub x: f32,
    pub y: f32,
}

/// What a command changed, so the app redraws only that.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Changed {
    /// Views added, removed or renamed, or another view shown.
    Views,
    /// The view's op chain (its processed signal changes).
    Ops(String),
    /// The view's display only.
    Display(String),
    /// What a plugin keeps for the view: (view, plugin).
    Plugin(String, String),
    /// The graph (nodes, places, edges); the views it changed are reported on their own.
    Graph,
    /// Epochs or placements.
    Timeline,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CommandError {
    #[error("no view {0:?}")]
    NoView(String),
    #[error("view {view:?} has no op {op:?}")]
    NoOp { view: String, op: String },
    #[error("no node {0:?}")]
    NoNode(String),
    #[error("{0}")]
    Invalid(String),
}

/// Applies `cmd` to `doc`; on an error `doc` is unchanged.
pub fn apply(doc: &mut SessionDoc, cmd: SessionCommand) -> Result<Changed, CommandError> {
    let mut work = doc.clone();
    let change = apply_graph(&mut work, cmd)?;
    work.compile();
    *doc = work;
    Ok(change)
}

fn apply_graph(doc: &mut SessionDoc, cmd: SessionCommand) -> Result<Changed, CommandError> {
    use SessionCommand as C;
    match cmd {
        C::OpenView { recording, name } => {
            check_channels(&recording.channels)?;
            let name = name.unwrap_or_else(|| crate::domain::session::view_name(&recording));
            let y = doc.graph.next_row();
            let s = source_id(doc);
            let v = doc.new_view_id();
            doc.graph.nodes.push(GraphNode { id: s.clone(), x: 0.0, y, kind: NodeKind::Source { recording } });
            doc.graph.nodes.push(GraphNode { id: v.clone(), x: COLUMN, y, kind: view_kind(name) });
            doc.graph.connect(&s, &v).map_err(CommandError::Invalid)?;
            doc.active_view = Some(v);
            Ok(Changed::Views)
        }
        C::CopyView { view, channels, name } => {
            let path = view_path(doc, &view)?;
            let NodeKind::Source { recording } = node(doc, &path[0])?.kind.clone() else { unreachable!("a path starts at a source") };
            let NodeKind::View { name: old, display, plugins } = node(doc, &view)?.kind.clone() else { unreachable!("checked by view_path") };
            check_channels(&channels)?;
            let y = doc.graph.next_row();
            let source = match channels {
                Some(subset) => {
                    // `subset` indexes the source; a copy of a subset view must stay inside it
                    if let Some(parent) = &recording.channels {
                        if let Some(c) = subset.iter().find(|c| !parent.contains(c)) {
                            return Err(CommandError::Invalid(format!("channel {c} is not read by view {view:?}")));
                        }
                    }
                    let s = source_id(doc);
                    let recording = RecordingRef { channels: Some(subset), ..recording };
                    doc.graph.nodes.push(GraphNode { id: s.clone(), x: 0.0, y, kind: NodeKind::Source { recording } });
                    s
                }
                None => path[0].clone(),
            };
            // Its own copy of the chain (edits to it leave the original alone)
            let mut last = source;
            let mut x = COLUMN;
            for op in &path[1..path.len() - 1] {
                let kind = node(doc, op)?.kind.clone();
                let NodeKind::Op { node: n, .. } = &kind else { continue };
                let id = doc.new_op_id(n);
                doc.graph.nodes.push(GraphNode { id: id.clone(), x, y, kind });
                doc.graph.connect(&last, &id).map_err(CommandError::Invalid)?;
                last = id;
                x += COLUMN;
            }
            let v = doc.new_view_id();
            let name = name.unwrap_or_else(|| format!("{old} (copy)"));
            doc.graph.nodes.push(GraphNode { id: v.clone(), x, y, kind: NodeKind::View { name, display, plugins } });
            doc.graph.connect(&last, &v).map_err(CommandError::Invalid)?;
            doc.active_view = Some(v);
            Ok(Changed::Views)
        }
        C::CloseView { view } => {
            view_node(doc, &view)?;
            let mut cur = doc.graph.input(&view).map(str::to_string);
            doc.graph.remove(&view);
            // What fed only this view goes with it
            while let Some(id) = cur {
                if !doc.graph.outputs(&id).is_empty() {
                    break;
                }
                cur = doc.graph.input(&id).map(str::to_string);
                doc.graph.remove(&id);
            }
            if doc.active_view.as_deref() == Some(view.as_str()) {
                let i = doc.views.iter().position(|v| v.id == view).unwrap_or(0);
                let rest: Vec<String> = doc.views.iter().filter(|v| v.id != view).map(|v| v.id.clone()).collect();
                doc.active_view = rest.get(i.min(rest.len().saturating_sub(1))).cloned();
            }
            Ok(Changed::Views)
        }
        C::RenameView { view, name } => {
            if let NodeKind::View { name: n, .. } = &mut view_node(doc, &view)?.kind {
                *n = name;
            }
            Ok(Changed::Views)
        }
        C::SelectView { view } => {
            view_path(doc, &view)?;
            doc.active_view = Some(view);
            Ok(Changed::Views)
        }
        C::AddOp { view, node: op, index, id } => {
            check_processing(&op)?;
            let path = view_path(doc, &view)?;
            let id = match id {
                Some(id) if doc.graph.contains(&id) || doc.view(&id).is_some() || id == "read" || id == "write" => {
                    return Err(CommandError::Invalid(format!("op id {id:?} is taken")));
                }
                Some(id) => id,
                None => doc.new_op_id(&op),
            };
            let ops = path.len() - 2;
            let at = index.unwrap_or(ops).min(ops);
            let (x, y) = node(doc, &path[at + 1]).map(|n| (n.x, n.y))?;
            // Room for it: what follows on this chain moves one column right
            for later in &path[at + 1..] {
                if let Some(n) = doc.graph.node_mut(later) {
                    n.x += COLUMN;
                }
            }
            doc.graph.nodes.push(GraphNode { id: id.clone(), x, y, kind: NodeKind::Op { enabled: true, node: op } });
            doc.graph.splice(&path[at], &path[at + 1], &id);
            Ok(Changed::Ops(view))
        }
        C::SetOp { view, op, node: n } => {
            check_processing(&n)?;
            if let NodeKind::Op { node, .. } = &mut chain_op(doc, &view, &op)?.kind {
                *node = n;
            }
            Ok(Changed::Ops(view))
        }
        C::EnableOp { view, op, enabled } => {
            if let NodeKind::Op { enabled: e, .. } = &mut chain_op(doc, &view, &op)?.kind {
                *e = enabled;
            }
            Ok(Changed::Ops(view))
        }
        C::RemoveOp { view, op } => {
            chain_op(doc, &view, &op)?;
            doc.graph.remove_bridging(&op);
            Ok(Changed::Ops(view))
        }
        C::MoveOp { view, op, index } => {
            chain_op(doc, &view, &op)?;
            if doc.graph.outputs(&op).len() > 1 {
                return Err(CommandError::Invalid(format!("{op} feeds other chains too: move it on the Pipeline canvas")));
            }
            doc.graph.unlink_bridging(&op);
            let path = view_path(doc, &view)?;
            let at = index.min(path.len() - 2);
            doc.graph.splice(&path[at], &path[at + 1], &op);
            Ok(Changed::Ops(view))
        }
        C::SetDisplay { view, display } => {
            if display.lanes == 0 || !(display.gain.is_finite() && display.gain > 0.0) {
                return Err(CommandError::Invalid("display: lanes ≥ 1 and gain > 0".into()));
            }
            if let NodeKind::View { display: d, .. } = &mut view_node(doc, &view)?.kind {
                *d = display;
            }
            Ok(Changed::Display(view))
        }
        C::SetPluginData { view, plugin, data } => {
            if let NodeKind::View { plugins, .. } = &mut view_node(doc, &view)?.kind {
                if data.is_null() {
                    plugins.remove(&plugin);
                } else {
                    plugins.insert(plugin.clone(), data);
                }
            }
            Ok(Changed::Plugin(view, plugin))
        }
        C::AddNode { kind, x, y, id, after } => {
            let id = match id {
                Some(id) if doc.graph.contains(&id) || doc.view(&id).is_some() || id == "read" || id == "write" => {
                    return Err(CommandError::Invalid(format!("node id {id:?} is taken")));
                }
                Some(id) => id,
                None => match &kind {
                    NodeKind::Source { .. } => source_id(doc),
                    NodeKind::Op { node, .. } => doc.new_op_id(node),
                    NodeKind::View { .. } => doc.new_view_id(),
                    NodeKind::Write {} => doc.graph.numbered_id("w", |id| doc.view(id).is_some()),
                    NodeKind::Recipe { recipe, .. } => doc.graph.fresh_id(&super::recipe::file_stem(recipe).replace('-', "_")),
                },
            };
            match &kind {
                NodeKind::Op { node, .. } => check_processing(node)?,
                NodeKind::Recipe { recipe, .. } if !doc.recipes.contains_key(recipe) => {
                    return Err(CommandError::Invalid(format!("no recipe {recipe:?} in the session")));
                }
                NodeKind::Source { recording } => check_channels(&recording.channels)?,
                _ => {}
            }
            doc.graph.nodes.push(GraphNode { id: id.clone(), x, y, kind });
            if let Some(from) = after {
                doc.graph.connect(&from, &id).map_err(CommandError::Invalid)?;
            }
            Ok(Changed::Graph)
        }
        C::RemoveNodes { nodes } => {
            for id in &nodes {
                node(doc, id)?;
            }
            for id in &nodes {
                doc.graph.remove(id);
            }
            Ok(Changed::Graph)
        }
        C::MoveNodes { moves } => {
            for m in moves {
                let n = doc.graph.node_mut(&m.id).ok_or_else(|| CommandError::NoNode(m.id.clone()))?;
                if !(m.x.is_finite() && m.y.is_finite()) {
                    return Err(CommandError::Invalid("node positions are numbers".into()));
                }
                (n.x, n.y) = (m.x, m.y);
            }
            Ok(Changed::Graph)
        }
        C::Connect { from, to } => {
            doc.graph.connect(&from, &to).map_err(CommandError::Invalid)?;
            Ok(Changed::Graph)
        }
        C::Disconnect { to } => {
            node(doc, &to)?;
            doc.graph.disconnect(&to);
            Ok(Changed::Graph)
        }
        C::SetNode { id, node: n } => {
            check_processing(&n)?;
            match &mut doc.graph.node_mut(&id).ok_or_else(|| CommandError::NoNode(id.clone()))?.kind {
                NodeKind::Op { node, .. } => *node = n,
                _ => return Err(CommandError::Invalid(format!("{id} is not an op"))),
            }
            Ok(Changed::Graph)
        }
        C::EnableNode { id, enabled } => {
            match &mut doc.graph.node_mut(&id).ok_or_else(|| CommandError::NoNode(id.clone()))?.kind {
                NodeKind::Op { enabled: e, .. } | NodeKind::Recipe { enabled: e, .. } => *e = enabled,
                _ => return Err(CommandError::Invalid(format!("{id} is not an op"))),
            }
            Ok(Changed::Graph)
        }
        C::AddRecipe { recipe } => {
            if recipe.name.trim().is_empty() {
                return Err(CommandError::Invalid("a recipe needs a name".into()));
            }
            for op in &recipe.ops {
                check_processing(&op.node)?;
            }
            doc.recipes.insert(recipe.name.clone(), recipe);
            Ok(Changed::Graph)
        }
        C::MakeRecipe { nodes, name, labels } => {
            let name = name.trim().to_string();
            if name.is_empty() || doc.recipes.contains_key(&name) {
                return Err(CommandError::Invalid(format!("recipe name {name:?} is empty or taken")));
            }
            let order = chain_order(doc, &nodes)?;
            let ops: Vec<super::session::OpDoc> = order
                .iter()
                .map(|id| match &doc.graph.node(id).expect("checked").kind {
                    NodeKind::Op { enabled, node } => super::session::OpDoc { id: id.clone(), enabled: *enabled, node: node.clone() },
                    _ => unreachable!("checked: ops only"),
                })
                .collect();
            let (head, tail) = (order[0].clone(), order[order.len() - 1].clone());
            let input = doc.graph.input(&head).map(str::to_string);
            let outputs: Vec<String> = doc.graph.outputs(&tail).into_iter().map(str::to_string).collect();
            let (x, y) = node(doc, &head).map(|n| (n.x, n.y))?;
            let id = doc.graph.fresh_id(&super::recipe::file_stem(&name).replace('-', "_"));
            let mut labels = labels;
            if labels.is_empty() {
                let mut fams: Vec<String> = ops.iter().map(|o| super::session::op_family(&o.node).to_string()).collect();
                fams.dedup();
                labels = fams;
            }
            doc.recipes.insert(name.clone(), RecipeDoc::new(name.clone(), labels, ops));
            for n in &order {
                doc.graph.remove(n);
            }
            doc.graph.nodes.push(GraphNode { id: id.clone(), x, y, kind: NodeKind::Recipe { recipe: name, enabled: true, overrides: Default::default() } });
            if let Some(i) = input {
                doc.graph.connect(&i, &id).map_err(CommandError::Invalid)?;
            }
            for o in outputs {
                doc.graph.connect(&id, &o).map_err(CommandError::Invalid)?;
            }
            Ok(Changed::Graph)
        }
        C::SetRecipeOverride { id, key, value } => {
            let recipe = match &node(doc, &id)?.kind {
                NodeKind::Recipe { recipe, .. } => recipe.clone(),
                _ => return Err(CommandError::Invalid(format!("{id} is not a recipe"))),
            };
            let def = doc.recipes.get(&recipe).ok_or_else(|| CommandError::Invalid(format!("no recipe {recipe:?}")))?.clone();
            let Some(NodeKind::Recipe { overrides, .. }) = doc.graph.node_mut(&id).map(|n| &mut n.kind) else { unreachable!() };
            if value.is_null() {
                overrides.remove(&key);
            } else {
                overrides.insert(key.clone(), value);
            }
            // The values must still make valid ops
            for op in def.ops_with(overrides) {
                check_processing(&op.node)?;
            }
            Ok(Changed::Graph)
        }
        C::SetRecipeOp { recipe, op, node: n } => {
            check_processing(&n)?;
            let def = doc.recipes.get_mut(&recipe).ok_or_else(|| CommandError::Invalid(format!("no recipe {recipe:?}")))?;
            let slot = def.ops.iter_mut().find(|o| o.id == op).ok_or_else(|| CommandError::Invalid(format!("recipe {recipe:?} has no op {op:?}")))?;
            slot.node = n;
            def.version += 1;
            Ok(Changed::Graph)
        }
        C::SetRecipeInfo { recipe, name, labels, description } => {
            let name = name.trim().to_string();
            if name.is_empty() || (name != recipe && doc.recipes.contains_key(&name)) {
                return Err(CommandError::Invalid(format!("recipe name {name:?} is empty or taken")));
            }
            let mut def = doc.recipes.remove(&recipe).ok_or_else(|| CommandError::Invalid(format!("no recipe {recipe:?}")))?;
            def.name = name.clone();
            def.labels = labels;
            def.description = description;
            def.version += 1;
            doc.recipes.insert(name.clone(), def);
            for n in &mut doc.graph.nodes {
                if let NodeKind::Recipe { recipe: r, .. } = &mut n.kind {
                    if *r == recipe {
                        *r = name.clone();
                    }
                }
            }
            Ok(Changed::Graph)
        }
        C::Ungroup { id } => {
            let (recipe, enabled, overrides, x, y) = match &node(doc, &id)?.kind {
                NodeKind::Recipe { recipe, enabled, overrides } => (recipe.clone(), *enabled, overrides.clone(), node(doc, &id)?.x, node(doc, &id)?.y),
                _ => return Err(CommandError::Invalid(format!("{id} is not a recipe"))),
            };
            let def = doc.recipes.get(&recipe).ok_or_else(|| CommandError::Invalid(format!("no recipe {recipe:?}")))?.clone();
            let input = doc.graph.input(&id).map(str::to_string);
            let outputs: Vec<String> = doc.graph.outputs(&id).into_iter().map(str::to_string).collect();
            doc.graph.remove(&id);
            let mut last = input;
            for (k, op) in def.ops_with(&overrides).into_iter().enumerate() {
                let nid = doc.new_op_id(&op.node);
                doc.graph.nodes.push(GraphNode { id: nid.clone(), x: x + k as f32 * COLUMN, y, kind: NodeKind::Op { enabled: op.enabled && enabled, node: op.node } });
                if let Some(l) = &last {
                    doc.graph.connect(l, &nid).map_err(CommandError::Invalid)?;
                }
                last = Some(nid);
            }
            if let Some(l) = last {
                for o in outputs {
                    doc.graph.connect(&l, &o).map_err(CommandError::Invalid)?;
                }
            }
            Ok(Changed::Graph)
        }
        C::SetEpoch { mut epoch } => {
            if !(epoch.start_sec.is_finite() && epoch.end_sec.is_finite() && epoch.end_sec > epoch.start_sec && epoch.start_sec >= 0.0) {
                return Err(CommandError::Invalid("an epoch ends after it starts, at or after 0 s".into()));
            }
            if epoch.name.trim().is_empty() {
                epoch.name = format!("{} {}", epoch.kind, doc.epochs.len() + 1);
            }
            if epoch.id.is_empty() {
                epoch.id = (1..).map(|i| format!("e{i}")).find(|id| !doc.epochs.iter().any(|e| e.id == *id)).expect("ids are unbounded");
            }
            match doc.epochs.iter_mut().find(|e| e.id == epoch.id) {
                Some(e) => *e = epoch,
                None => doc.epochs.push(epoch),
            }
            doc.epochs.sort_by(|a, b| a.recording.cmp(&b.recording).then(a.start_sec.total_cmp(&b.start_sec)));
            Ok(Changed::Timeline)
        }
        C::RemoveEpoch { id } => {
            let n = doc.epochs.len();
            doc.epochs.retain(|e| e.id != id);
            if doc.epochs.len() == n {
                return Err(CommandError::Invalid(format!("no epoch {id:?}")));
            }
            Ok(Changed::Timeline)
        }
        C::SetEpochs { epochs } => {
            let mut d = doc.clone();
            for epoch in epochs {
                apply(&mut d, C::SetEpoch { epoch })?;
            }
            *doc = d;
            Ok(Changed::Timeline)
        }
        C::SetEventSet { mut set } => {
            if !(set.tick_height.is_finite() && set.tick_width.is_finite() && set.hue.is_finite()) {
                return Err(CommandError::Invalid("an event set's look must be finite".into()));
            }
            set.tick_height = set.tick_height.clamp(0.05, 1.0);
            set.tick_width = set.tick_width.clamp(0.5, 8.0);
            set.hue = set.hue.rem_euclid(360.0);
            if set.name.trim().is_empty() {
                set.name = format!("events {}", doc.event_sets.len() + 1);
            }
            if set.id.is_empty() {
                set.id = (1..).map(|i| format!("ev{i}")).find(|id| !doc.event_sets.iter().any(|e| e.id == *id)).expect("ids are unbounded");
            }
            match doc.event_sets.iter_mut().find(|e| e.id == set.id) {
                Some(e) => *e = set,
                None => doc.event_sets.push(set),
            }
            Ok(Changed::Timeline)
        }
        C::RemoveEventSet { id } => {
            let n = doc.event_sets.len();
            doc.event_sets.retain(|e| e.id != id);
            if doc.event_sets.len() == n {
                return Err(CommandError::Invalid(format!("no event set {id:?}")));
            }
            Ok(Changed::Timeline)
        }
        C::PlaceRecording { path, source, start_sec } => {
            if !(start_sec.is_finite() && start_sec >= 0.0) {
                return Err(CommandError::Invalid("a recording starts at or after 0 s".into()));
            }
            match doc.placements.iter_mut().find(|p| p.path == path && p.source == source) {
                Some(p) => p.start_sec = start_sec,
                None => doc.placements.push(Placement { path, source, start_sec }),
            }
            Ok(Changed::Timeline)
        }
        C::UnplaceRecording { path } => {
            doc.placements.retain(|p| p.path != path);
            Ok(Changed::Timeline)
        }
        C::SetSource { id, recording } => {
            check_channels(&recording.channels)?;
            match &mut doc.graph.node_mut(&id).ok_or_else(|| CommandError::NoNode(id.clone()))?.kind {
                NodeKind::Source { recording: r } => *r = recording,
                _ => return Err(CommandError::Invalid(format!("{id} is not a recording"))),
            }
            Ok(Changed::Graph)
        }
    }
}

/// `nodes` (op nodes) in chain order: one enters the group from outside, each leads to the next.
fn chain_order(doc: &SessionDoc, nodes: &[String]) -> Result<Vec<String>, CommandError> {
    let bad = |m: &str| Err(CommandError::Invalid(format!("a recipe is made of a chain of ops: {m}")));
    if nodes.is_empty() {
        return bad("nothing selected");
    }
    for id in nodes {
        if !matches!(node(doc, id)?.kind, NodeKind::Op { .. }) {
            return bad(&format!("{id} is not an op"));
        }
    }
    let inside = |id: &str| nodes.iter().any(|n| n == id);
    let heads: Vec<&String> = nodes.iter().filter(|n| !doc.graph.input(n).is_some_and(inside)).collect();
    if heads.len() != 1 {
        return bad("the selection must be one chain");
    }
    let mut order = vec![heads[0].clone()];
    while order.len() < nodes.len() {
        let next: Vec<&str> = doc.graph.outputs(order.last().expect("non-empty")).into_iter().filter(|o| inside(o)).collect();
        match next.as_slice() {
            [one] if doc.graph.outputs(order.last().expect("non-empty")).len() == 1 => order.push(one.to_string()),
            _ => return bad("the selection branches or has gaps"),
        }
    }
    Ok(order)
}

/// A new view node.
pub fn view_kind(name: String) -> NodeKind {
    NodeKind::View { name, display: DisplayDoc::default(), plugins: Default::default() }
}

fn source_id(doc: &SessionDoc) -> String {
    doc.graph.numbered_id("s", |id| doc.view(id).is_some())
}

fn check_channels(channels: &Option<Vec<usize>>) -> Result<(), CommandError> {
    if channels.as_ref().is_some_and(Vec::is_empty) {
        return Err(CommandError::Invalid("a view needs at least one channel".into()));
    }
    Ok(())
}

/// Processing nodes run on the visible window and in the written output; `read` / `write` are the
/// view's recording and its write step, and `resample` changes the timeline (not in v1 views).
fn check_processing(node: &Node) -> Result<(), CommandError> {
    match node {
        Node::Read { .. } | Node::Write { .. } | Node::Resample { .. } => {
            Err(CommandError::Invalid(format!("{} is not an op of a view", op_name(node))))
        }
        _ => node.stage().map(|_| ()).map_err(|e| CommandError::Invalid(e.to_string())),
    }
}

fn node<'a>(doc: &'a SessionDoc, id: &str) -> Result<&'a GraphNode, CommandError> {
    doc.graph.node(id).ok_or_else(|| CommandError::NoNode(id.to_string()))
}

/// The view node `view`.
fn view_node<'a>(doc: &'a mut SessionDoc, view: &str) -> Result<&'a mut GraphNode, CommandError> {
    match doc.graph.node_mut(view) {
        Some(n) if matches!(n.kind, NodeKind::View { .. }) => Ok(n),
        _ => Err(CommandError::NoView(view.to_string())),
    }
}

/// `view`'s chain `[source, op…, view]`.
fn view_path(doc: &mut SessionDoc, view: &str) -> Result<Vec<String>, CommandError> {
    view_node(doc, view)?;
    doc.graph.path(view).ok_or_else(|| CommandError::Invalid(format!("view {view:?} is not connected to a recording")))
}

/// Op `op` on `view`'s chain.
fn chain_op<'a>(doc: &'a mut SessionDoc, view: &str, op: &str) -> Result<&'a mut GraphNode, CommandError> {
    let path = view_path(doc, view)?;
    if !path[1..path.len() - 1].iter().any(|n| n == op) {
        return Err(CommandError::NoOp { view: view.to_string(), op: op.to_string() });
    }
    doc.graph.node_mut(op).ok_or_else(|| CommandError::NoOp { view: view.to_string(), op: op.to_string() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::session::Show;
    use crate::domain::graph::NodeKind;
    use dspk_pipeline::{Band, Design, Mode, Start};

    fn bandpass(lo: f64, hi: f64) -> Node {
        Node::Iir { band: Band::Bandpass, cutoff: vec![lo, hi], order: 3, design: Design::Butterworth, ripple_db: None, mode: Mode::ForwardBackward, start: Start::Rest }
    }

    fn opened() -> SessionDoc {
        let mut d = SessionDoc::default();
        let rec = RecordingRef { path: "/data/rec.bin".into(), source: None, channels: None };
        assert_eq!(apply(&mut d, SessionCommand::OpenView { recording: rec, name: None }), Ok(Changed::Views));
        d
    }

    #[test]
    fn test_open_add_set_remove() {
        let mut d = opened();
        assert_eq!(d.active().unwrap().name, "rec.bin");
        apply(&mut d, SessionCommand::AddOp { view: "v1".into(), node: bandpass(300.0, 6000.0), index: None, id: None }).unwrap();
        apply(&mut d, SessionCommand::AddOp { view: "v1".into(), node: Node::Cmr {}, index: None, id: None }).unwrap();
        assert_eq!(d.views[0].ops.iter().map(|o| o.id.as_str()).collect::<Vec<_>>(), ["iir", "cmr"]);
        apply(&mut d, SessionCommand::SetOp { view: "v1".into(), op: "iir".into(), node: bandpass(500.0, 5000.0) }).unwrap();
        assert_eq!(d.views[0].ops[0].node, bandpass(500.0, 5000.0));
        apply(&mut d, SessionCommand::MoveOp { view: "v1".into(), op: "cmr".into(), index: 0 }).unwrap();
        assert_eq!(d.views[0].ops[0].id, "cmr");
        apply(&mut d, SessionCommand::RemoveOp { view: "v1".into(), op: "cmr".into() }).unwrap();
        assert_eq!(d.views[0].ops.len(), 1);
    }

    /// Views are named by their source (the file for a file of one source); a document saved with
    /// the old default names (`file · /source`) loads with the new ones.
    #[test]
    fn test_views_are_named_by_source() {
        let mut d = opened();
        let emg = RecordingRef { path: "/data/meps.nwb.zarr".into(), source: Some("/acquisition/EMG".into()), channels: None };
        apply(&mut d, SessionCommand::OpenView { recording: emg.clone(), name: None }).unwrap();
        assert_eq!(d.active().unwrap().name, "EMG");
        let main = RecordingRef { path: "/data/rec.zarr".into(), source: Some(dspk_io::sources::MAIN.into()), channels: None };
        assert_eq!(crate::domain::session::view_name(&main), "rec.zarr");
        let view = d.active_view.clone().unwrap();
        apply(&mut d, SessionCommand::RenameView { view, name: "meps.nwb.zarr · /acquisition/EMG".into() }).unwrap();
        let loaded = SessionDoc::from_json(&serde_json::to_string(&d).unwrap()).unwrap();
        assert!(loaded.views.iter().any(|v| v.name == "EMG"), "{:?}", loaded.views.iter().map(|v| &v.name).collect::<Vec<_>>());
    }

    #[test]
    fn test_invalid_commands_leave_the_document_unchanged() {
        let mut d = opened();
        let before = d.clone();
        let bad_iir = Node::Iir { band: Band::Bandpass, cutoff: vec![300.0], order: 3, design: Design::Butterworth, ripple_db: None, mode: Mode::ForwardBackward, start: Start::Rest };
        assert!(apply(&mut d, SessionCommand::AddOp { view: "v1".into(), node: bad_iir, index: None, id: None }).is_err());
        assert!(apply(&mut d, SessionCommand::AddOp { view: "v1".into(), node: Node::Car {}, index: None, id: Some("read".into()) }).is_err());
        assert!(apply(&mut d, SessionCommand::RemoveOp { view: "v1".into(), op: "nope".into() }).is_err());
        assert!(apply(&mut d, SessionCommand::SelectView { view: "v9".into() }).is_err());
        let display = DisplayDoc { lanes: 0, ..DisplayDoc::default() };
        assert!(apply(&mut d, SessionCommand::SetDisplay { view: "v1".into(), display }).is_err());
        assert_eq!(d, before);
    }

    #[test]
    fn test_copy_with_a_channel_subset() {
        let mut d = opened();
        apply(&mut d, SessionCommand::AddOp { view: "v1".into(), node: Node::Cmr {}, index: None, id: None }).unwrap();
        apply(&mut d, SessionCommand::CopyView { view: "v1".into(), channels: Some(vec![4, 5]), name: None }).unwrap();
        let copy = d.active().unwrap();
        assert_eq!((copy.id.as_str(), copy.recording.channels.as_deref()), ("v2", Some(&[4, 5][..])));
        let nodes = |ops: &[crate::domain::session::OpDoc]| ops.iter().map(|o| (o.enabled, o.node.clone())).collect::<Vec<_>>();
        assert_eq!(nodes(&copy.ops), nodes(&d.views[0].ops), "the copy starts with the same chain");
        assert_ne!(copy.ops[0].id, d.views[0].ops[0].id, "its own ops: edits to it leave the original alone");
        // A copy of a subset stays inside it
        assert!(apply(&mut d, SessionCommand::CopyView { view: "v2".into(), channels: Some(vec![6]), name: None }).is_err());
    }

    #[test]
    fn test_close_moves_the_active_view() {
        let mut d = opened();
        apply(&mut d, SessionCommand::CopyView { view: "v1".into(), channels: None, name: None }).unwrap();
        apply(&mut d, SessionCommand::CloseView { view: "v2".into() }).unwrap();
        assert_eq!(d.active_view.as_deref(), Some("v1"));
        apply(&mut d, SessionCommand::CloseView { view: "v1".into() }).unwrap();
        assert_eq!(d.active_view, None);
    }

    #[test]
    fn test_commands_are_json() {
        let cmd: SessionCommand = serde_json::from_str(
            r#"{"command": "set_op", "view": "v1", "op": "iir", "node": {"op": "iir", "band": "highpass", "cutoff": [250.0]}}"#,
        )
        .unwrap();
        let mut d = opened();
        apply(&mut d, SessionCommand::AddOp { view: "v1".into(), node: bandpass(300.0, 6000.0), index: None, id: None }).unwrap();
        assert_eq!(apply(&mut d, cmd), Ok(Changed::Ops("v1".into())));
        let display = DisplayDoc { show: Show::Raw, ..DisplayDoc::default() };
        assert_eq!(apply(&mut d, SessionCommand::SetDisplay { view: "v1".into(), display }), Ok(Changed::Display("v1".into())));
    }

    #[test]
    fn test_plugin_data_is_set_and_removed() {
        let mut d = opened();
        let data = serde_json::json!({"sorting": "/d/rec.kilosort4.sorting.zarr"});
        let cmd = SessionCommand::SetPluginData { view: "v1".into(), plugin: "synapse".into(), data: data.clone() };
        assert_eq!(apply(&mut d, cmd), Ok(Changed::Plugin("v1".into(), "synapse".into())));
        assert_eq!(d.views[0].plugins["synapse"], data);
        let back = SessionDoc::from_json(&d.to_json()).unwrap();
        assert_eq!(back, d);
        apply(&mut d, SessionCommand::SetPluginData { view: "v1".into(), plugin: "synapse".into(), data: serde_json::Value::Null }).unwrap();
        assert!(d.views[0].plugins.is_empty());
    }

    #[test]
    fn test_graph_commands_branch_a_recording() {
        let mut d = opened();
        apply(&mut d, SessionCommand::AddOp { view: "v1".into(), node: bandpass(300.0, 6000.0), index: None, id: None }).unwrap();
        // A second branch off the recording: a low-pass into a new view
        let lp: Node = serde_json::from_value(serde_json::json!({"op": "iir", "band": "lowpass", "cutoff": [300.0]})).unwrap();
        apply(&mut d, SessionCommand::AddNode { kind: NodeKind::Op { enabled: true, node: lp }, x: 260.0, y: 150.0, id: Some("lfp_band".into()), after: Some("s1".into()) }).unwrap();
        apply(&mut d, SessionCommand::AddNode { kind: view_kind("LFP".into()), x: 520.0, y: 150.0, id: None, after: Some("lfp_band".into()) }).unwrap();
        assert_eq!(d.views.len(), 2);
        assert_eq!(d.views[1].name, "LFP");
        assert_eq!(d.views[1].ops[0].id, "lfp_band");
        // Editing the shared recording's branch leaves the other alone
        apply(&mut d, SessionCommand::EnableNode { id: "iir".into(), enabled: false }).unwrap();
        assert!(!d.views[0].ops[0].enabled && d.views[1].ops[0].enabled);
        // A view cut off its source is no longer a view
        apply(&mut d, SessionCommand::Disconnect { to: "v2".into() }).unwrap();
        assert_eq!(d.views.len(), 1);
        assert!(apply(&mut d, SessionCommand::Connect { from: "v1".into(), to: "v2".into() }).is_err(), "a view has no output");
        apply(&mut d, SessionCommand::MoveNodes { moves: vec![NodeMove { id: "v2".into(), x: 10.0, y: 20.0 }] }).unwrap();
        assert_eq!((d.graph.node("v2").unwrap().x, d.graph.node("v2").unwrap().y), (10.0, 20.0));
        apply(&mut d, SessionCommand::RemoveNodes { nodes: vec!["v2".into(), "lfp_band".into()] }).unwrap();
        assert_eq!(d.graph.nodes.len(), 3);
    }

    #[test]
    fn test_epochs_and_placements() {
        let mut d = opened();
        let e = EpochDoc { id: String::new(), recording: "/data/rec.bin".into(), name: String::new(), kind: "stimulus".into(), start_sec: 2.0, end_sec: 3.5, notes: String::new() };
        apply(&mut d, SessionCommand::SetEpoch { epoch: e.clone() }).unwrap();
        assert_eq!((d.epochs[0].id.as_str(), d.epochs[0].name.as_str()), ("e1", "stimulus 1"));
        assert!(apply(&mut d, SessionCommand::SetEpoch { epoch: EpochDoc { end_sec: 1.0, ..e.clone() } }).is_err(), "ends before it starts");
        apply(&mut d, SessionCommand::SetEpoch { epoch: EpochDoc { id: "e1".into(), name: "tone".into(), ..e } }).unwrap();
        assert_eq!((d.epochs.len(), d.epochs[0].name.as_str()), (1, "tone"));
        apply(&mut d, SessionCommand::RemoveEpoch { id: "e1".into() }).unwrap();
        // Event sets: ids given, the look clamped, removed by id
        let set = EventSetDoc { tick_height: 3.0, tick_width: 0.0, ..EventSetDoc::new("/r.bin".into(), "", "/r.dspk/events/a.parquet".into()) };
        apply(&mut d, SessionCommand::SetEventSet { set }).unwrap();
        let s = &d.event_sets[0];
        assert_eq!((s.id.as_str(), s.name.as_str(), s.tick_height, s.tick_width), ("ev1", "events 1", 1.0, 0.5));
        apply(&mut d, SessionCommand::RemoveEventSet { id: "ev1".into() }).unwrap();
        assert!(d.event_sets.is_empty());
        // Several epochs at once, or none when one is invalid
        let ep = |a: f64, b: f64| EpochDoc { id: String::new(), recording: "/r.bin".into(), name: String::new(), kind: "response".into(), start_sec: a, end_sec: b, notes: String::new() };
        assert!(apply(&mut d, SessionCommand::SetEpochs { epochs: vec![ep(1.0, 2.0), ep(3.0, 2.0)] }).is_err());
        assert!(d.epochs.is_empty(), "nothing added");
        apply(&mut d, SessionCommand::SetEpochs { epochs: vec![ep(1.0, 2.0), ep(3.0, 4.0)] }).unwrap();
        assert_eq!(d.epochs.len(), 2);
        apply(&mut d, SessionCommand::PlaceRecording { path: "/data/rec.bin".into(), source: None, start_sec: 60.0 }).unwrap();
        apply(&mut d, SessionCommand::PlaceRecording { path: "/data/rec.bin".into(), source: None, start_sec: 90.0 }).unwrap();
        assert_eq!((d.placements.len(), d.placements[0].start_sec), (1, 90.0));
        let back = SessionDoc::from_json(&d.to_json()).unwrap();
        assert_eq!(back.placements, d.placements);
    }

    #[test]
    fn test_recipes_make_override_edit_ungroup() {
        let mut d = opened();
        apply(&mut d, SessionCommand::AddOp { view: "v1".into(), node: bandpass(300.0, 6000.0), index: None, id: None }).unwrap();
        apply(&mut d, SessionCommand::AddOp { view: "v1".into(), node: Node::Cmr {}, index: None, id: None }).unwrap();
        let before = d.views[0].ops.iter().map(|o| o.node.clone()).collect::<Vec<_>>();
        apply(&mut d, SessionCommand::MakeRecipe { nodes: vec!["cmr".into(), "iir".into()], name: "Spikes".into(), labels: vec![] }).unwrap();
        assert_eq!(d.graph.nodes.len(), 3, "source, recipe, view");
        assert_eq!(d.views[0].ops.iter().map(|o| o.node.clone()).collect::<Vec<_>>(), before, "the view runs the same ops");
        assert_eq!(d.recipes["Spikes"].labels, ["filter", "reference"]);
        // A use's own value, then the recipe's (every use follows)
        apply(&mut d, SessionCommand::SetRecipeOverride { id: "spikes".into(), key: "iir.cutoff".into(), value: serde_json::json!([500.0, 5000.0]) }).unwrap();
        assert_eq!(d.views[0].ops[0].node, bandpass(500.0, 5000.0));
        assert!(apply(&mut d, SessionCommand::SetRecipeOverride { id: "spikes".into(), key: "iir.cutoff".into(), value: serde_json::json!([500.0]) }).is_err(), "a band-pass needs two cutoffs");
        apply(&mut d, SessionCommand::SetRecipeOverride { id: "spikes".into(), key: "iir.cutoff".into(), value: serde_json::Value::Null }).unwrap();
        apply(&mut d, SessionCommand::SetRecipeOp { recipe: "Spikes".into(), op: "iir".into(), node: bandpass(250.0, 7000.0) }).unwrap();
        assert_eq!((d.views[0].ops[0].node.clone(), d.recipes["Spikes"].version), (bandpass(250.0, 7000.0), 2));
        apply(&mut d, SessionCommand::Ungroup { id: "spikes".into() }).unwrap();
        assert_eq!(d.graph.nodes.len(), 4);
        assert_eq!(d.views[0].ops[0].node, bandpass(250.0, 7000.0));
        // Not a chain
        assert!(apply(&mut d, SessionCommand::MakeRecipe { nodes: vec!["s1".into()], name: "x".into(), labels: vec![] }).is_err());
    }

    #[test]
    fn test_a_shared_op_is_not_moved_by_one_view() {
        let mut d = opened();
        apply(&mut d, SessionCommand::AddOp { view: "v1".into(), node: Node::Cmr {}, index: None, id: None }).unwrap();
        apply(&mut d, SessionCommand::AddOp { view: "v1".into(), node: bandpass(300.0, 6000.0), index: None, id: None }).unwrap();
        apply(&mut d, SessionCommand::AddNode { kind: view_kind("other".into()), x: 0.0, y: 0.0, id: None, after: Some("cmr".into()) }).unwrap();
        assert!(apply(&mut d, SessionCommand::MoveOp { view: "v1".into(), op: "cmr".into(), index: 1 }).is_err());
        assert_eq!(d.view("v2").unwrap().ops.len(), 1, "the other view keeps cmr");
    }
}
