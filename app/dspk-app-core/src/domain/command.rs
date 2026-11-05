//! Commands: every change to the session document. The UI and an agent (future MCP server) send
//! the same commands, so there is one code path; each command is plain data (serializable, so an
//! agent sends it as JSON) and [`apply`] is pure (no I/O, no device), so it is tested directly.
//!
//! Running work (preview, write, sort) is not a document change: the data layer does it and
//! records its result with a command (e.g. a sort: [`SessionCommand::SetPluginData`]).

use dspk_pipeline::Node;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::session::{op_name, DisplayDoc, OpDoc, RecordingRef, SessionDoc, ViewDoc};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
pub enum SessionCommand {
    /// Adds a view of a recording (all its channels, no ops) and shows it.
    OpenView {
        recording: RecordingRef,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// Copies a view (recording, ops, display, sorting) into a new one, optionally reading only
    /// `channels` (indices into the source, in order): how some channels get other processing.
    CopyView {
        view: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        channels: Option<Vec<usize>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
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
    RemoveOp { view: String, op: String },
    /// Moves an op to `index` in the chain.
    MoveOp { view: String, op: String, index: usize },
    SetDisplay { view: String, display: DisplayDoc },
    /// Replaces what plugin `plugin` keeps for a view (`null` removes it), e.g.
    /// `{"plugin": "synapse", "data": {"sorting": "rec.kilosort4.sorting.zarr"}}`.
    SetPluginData { view: String, plugin: String, data: serde_json::Value },
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
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CommandError {
    #[error("no view {0:?}")]
    NoView(String),
    #[error("view {view:?} has no op {op:?}")]
    NoOp { view: String, op: String },
    #[error("{0}")]
    Invalid(String),
}

/// Applies `cmd` to `doc`; on an error `doc` is unchanged.
pub fn apply(doc: &mut SessionDoc, cmd: SessionCommand) -> Result<Changed, CommandError> {
    use SessionCommand as C;
    match cmd {
        C::OpenView { recording, name } => {
            if recording.channels.as_ref().is_some_and(Vec::is_empty) {
                return Err(CommandError::Invalid("a view needs at least one channel".into()));
            }
            let id = doc.new_view_id();
            let name = name.unwrap_or_else(|| recording_name(&recording));
            doc.views.push(ViewDoc { id: id.clone(), name, recording, ops: Vec::new(), display: DisplayDoc::default(), plugins: Default::default() });
            doc.active_view = Some(id);
            Ok(Changed::Views)
        }
        C::CopyView { view, channels, name } => {
            let src = doc.view(&view).ok_or_else(|| CommandError::NoView(view.clone()))?;
            if channels.as_ref().is_some_and(Vec::is_empty) {
                return Err(CommandError::Invalid("a view needs at least one channel".into()));
            }
            let mut copy = src.clone();
            copy.id = doc.new_view_id();
            copy.name = name.unwrap_or_else(|| format!("{} (copy)", src.name));
            if let Some(subset) = channels {
                // `subset` indexes the source; a copy of a subset view must stay inside it
                if let Some(parent) = &src.recording.channels {
                    if let Some(c) = subset.iter().find(|c| !parent.contains(c)) {
                        return Err(CommandError::Invalid(format!("channel {c} is not read by view {view:?}")));
                    }
                }
                copy.recording.channels = Some(subset);
            }
            doc.active_view = Some(copy.id.clone());
            doc.views.push(copy);
            Ok(Changed::Views)
        }
        C::CloseView { view } => {
            let i = view_index(doc, &view)?;
            doc.views.remove(i);
            if doc.active_view.as_deref() == Some(view.as_str()) {
                doc.active_view = doc.views.get(i.min(doc.views.len().saturating_sub(1))).map(|v| v.id.clone());
            }
            Ok(Changed::Views)
        }
        C::RenameView { view, name } => {
            view_mut(doc, &view)?.name = name;
            Ok(Changed::Views)
        }
        C::SelectView { view } => {
            view_index(doc, &view)?;
            doc.active_view = Some(view);
            Ok(Changed::Views)
        }
        C::AddOp { view, node, index, id } => {
            check_processing(&node)?;
            let v = view_mut(doc, &view)?;
            let id = match id {
                Some(id) if v.op(&id).is_some() || id == "read" || id == "write" => {
                    return Err(CommandError::Invalid(format!("op id {id:?} is taken")));
                }
                Some(id) => id,
                None => v.new_op_id(&node),
            };
            let at = index.unwrap_or(v.ops.len()).min(v.ops.len());
            v.ops.insert(at, OpDoc { id, enabled: true, node });
            Ok(Changed::Ops(view))
        }
        C::SetOp { view, op, node } => {
            check_processing(&node)?;
            op_mut(doc, &view, &op)?.node = node;
            Ok(Changed::Ops(view))
        }
        C::EnableOp { view, op, enabled } => {
            op_mut(doc, &view, &op)?.enabled = enabled;
            Ok(Changed::Ops(view))
        }
        C::RemoveOp { view, op } => {
            let v = view_mut(doc, &view)?;
            let i = v.ops.iter().position(|o| o.id == op).ok_or_else(|| CommandError::NoOp { view: view.clone(), op })?;
            v.ops.remove(i);
            Ok(Changed::Ops(view))
        }
        C::MoveOp { view, op, index } => {
            let v = view_mut(doc, &view)?;
            let i = v.ops.iter().position(|o| o.id == op).ok_or_else(|| CommandError::NoOp { view: view.clone(), op })?;
            let o = v.ops.remove(i);
            let at = index.min(v.ops.len());
            v.ops.insert(at, o);
            Ok(Changed::Ops(view))
        }
        C::SetDisplay { view, display } => {
            if display.lanes == 0 || !(display.gain.is_finite() && display.gain > 0.0) {
                return Err(CommandError::Invalid("display: lanes ≥ 1 and gain > 0".into()));
            }
            view_mut(doc, &view)?.display = display;
            Ok(Changed::Display(view))
        }
        C::SetPluginData { view, plugin, data } => {
            let v = view_mut(doc, &view)?;
            if data.is_null() {
                v.plugins.remove(&plugin);
            } else {
                v.plugins.insert(plugin.clone(), data);
            }
            Ok(Changed::Plugin(view, plugin))
        }
    }
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

fn view_index(doc: &SessionDoc, view: &str) -> Result<usize, CommandError> {
    doc.views.iter().position(|v| v.id == view).ok_or_else(|| CommandError::NoView(view.to_string()))
}

fn view_mut<'a>(doc: &'a mut SessionDoc, view: &str) -> Result<&'a mut ViewDoc, CommandError> {
    doc.view_mut(view).ok_or_else(|| CommandError::NoView(view.to_string()))
}

fn op_mut<'a>(doc: &'a mut SessionDoc, view: &str, op: &str) -> Result<&'a mut OpDoc, CommandError> {
    let v = view_mut(doc, view)?;
    v.ops.iter_mut().find(|o| o.id == op).ok_or_else(|| CommandError::NoOp { view: view.to_string(), op: op.to_string() })
}

/// `rec.bin` or `rec.nwb · ecephys` for a new view.
fn recording_name(r: &RecordingRef) -> String {
    let file = r.path.file_name().map_or_else(|| r.path.display().to_string(), |n| n.to_string_lossy().into_owned());
    match &r.source {
        Some(s) => format!("{file} · {s}"),
        None => file,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::session::Show;
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
        assert_eq!(copy.ops, d.views[0].ops, "the copy starts with the same chain");
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
}
