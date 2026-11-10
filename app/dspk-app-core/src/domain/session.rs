//! The session document: the whole working session as one versioned JSON document, so the app and an agent (a future MCP server, as Paper's MCP edits the open
//! file) read and change the same thing.
//!
//! A session holds a processing graph ([`GraphDoc`]: recordings, ops, views, writes, as the
//! Pipeline canvas shows them) and the views compiled from it. Each view is one recording (file,
//! source within it, optional channel subset read as one grouped array), the op chain on its path
//! (dspk-pipeline nodes, applied to every channel of the view), how it is displayed, and what
//! plugins keep for it (e.g. electrophysiology: the linked sorting). The graph is the truth: the
//! views are rewritten from it after every command (they stay in the file for agents to read).
//!
//! The document only changes through [`crate::domain::command::SessionCommand`]s; the app
//! re-renders from it and saves it after each change.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use dspk_pipeline::{Node, OutputFormat, PipelineSpec, SPEC_VERSION};

use super::epoch::{EpochDoc, EventSetDoc, Placement};
use super::graph::{GraphDoc, Recipes};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The document version this app reads and writes (`dspk_session` field).
pub const SESSION_VERSION: u32 = 1;

/// The whole session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionDoc {
    /// Document version; must equal [`SESSION_VERSION`].
    pub dspk_session: u32,
    /// The processing graph (the Pipeline canvas).
    #[serde(default, skip_serializing_if = "GraphDoc::is_empty")]
    pub graph: GraphDoc,
    /// The recipes the graph uses, by name (copies of library recipes, or made here).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub recipes: Recipes,
    /// Marked spans of the recordings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub epochs: Vec<EpochDoc>,
    /// Where recordings sit on the session's time.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub placements: Vec<Placement>,
    /// Event sets drawn over the recordings' traces.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub event_sets: Vec<EventSetDoc>,
    /// Every view, in the order they are listed: compiled from `graph` (a document with views
    /// and no graph gets one on load).
    #[serde(default)]
    pub views: Vec<ViewDoc>,
    /// Id of the view shown (and edited by default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_view: Option<String>,
}

impl Default for SessionDoc {
    fn default() -> Self {
        Self { dspk_session: SESSION_VERSION, graph: GraphDoc::default(), recipes: Recipes::new(), epochs: Vec::new(), placements: Vec::new(), event_sets: Vec::new(), views: Vec::new(), active_view: None }
    }
}

/// One view: a recording, its op chain, its display and its results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ViewDoc {
    /// Unique within the session (`v1`, `v2`, …).
    pub id: String,
    /// Shown in tabs and lists.
    pub name: String,
    pub recording: RecordingRef,
    /// Processing applied to every channel of the view, in order (no `read` / `write`).
    #[serde(default)]
    pub ops: Vec<OpDoc>,
    #[serde(default)]
    pub display: DisplayDoc,
    /// What plugins keep for this view, by plugin id (e.g. `synapse`: the linked
    /// `.sorting.zarr`); each plugin owns its own value.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    #[schemars(with = "BTreeMap<String, serde_json::Value>")]
    pub plugins: BTreeMap<String, serde_json::Value>,
}

/// The recording of a view: one source of one file, optionally a channel subset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RecordingRef {
    pub path: PathBuf,
    /// Source id within the file; default: the file's default source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Channels of the source read by this view, in this order; default: all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channels: Option<Vec<usize>>,
}

/// One op of a view's chain: a dspk-pipeline node with an id the commands address.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OpDoc {
    /// Unique within the view (`bandpass`, `cmr`, `op3`, …).
    pub id: String,
    /// Off: kept in the chain, skipped when processing (compare with and without).
    #[serde(default = "yes")]
    pub enabled: bool,
    pub node: Node,
}

fn yes() -> bool {
    true
}

/// How a view is drawn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DisplayDoc {
    /// `processed`: the op chain's output (live preview); `raw`: the recording as stored.
    #[serde(default)]
    pub show: Show,
    /// Amplitude gain (1 = nominal).
    #[serde(default = "default_gain")]
    pub gain: f32,
    /// Traces on screen at once.
    #[serde(default = "default_lanes")]
    pub lanes: usize,
    /// Visible time window, seconds from the recording's start: `[start, end]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<[f64; 2]>,
    /// What the view is in Explore: the stream (time views), or its events' windows stacked.
    #[serde(default, skip_serializing_if = "is_default")]
    pub layout: ViewLayout,
    /// Epochs layout: the window around each event, an index into the Analytics windows
    /// (`screens::analytics::WINDOWS_MS`, default −200 … +800 ms).
    #[serde(default = "default_epochs_window")]
    pub epochs_window: usize,
    /// Epochs layout: where the events come from.
    #[serde(default, skip_serializing_if = "is_default")]
    pub epochs_from: EpochsFrom,
}

/// What a view is in Explore.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ViewLayout {
    /// Time views of the stream (traces, heatmap, raster).
    #[default]
    Stream,
    /// Each event's window stacked per channel, mean ± SD (the chain ends with Crossings).
    Epochs,
}

/// Where an Epochs view's events come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EpochsFrom {
    /// The event set a Write of the same chain saved, else the live ones of the visible window.
    #[default]
    Auto,
    /// Found now in the visible window (follows the level while it is tuned).
    Live,
    /// The saved event set of the chain's Write.
    Saved,
}

fn is_default<T: Default + PartialEq>(v: &T) -> bool {
    *v == T::default()
}

/// Default window of an Epochs view (−200 … +800 ms).
pub const DEFAULT_EPOCHS_WINDOW: usize = 2;

fn default_epochs_window() -> usize {
    DEFAULT_EPOCHS_WINDOW
}

/// Default amplitude gain of a view.
pub const DEFAULT_GAIN: f32 = 1.0;
/// Default traces on screen of a view.
pub const DEFAULT_LANES: usize = 8;

fn default_gain() -> f32 {
    DEFAULT_GAIN
}
fn default_lanes() -> usize {
    DEFAULT_LANES
}

impl Default for DisplayDoc {
    fn default() -> Self {
        Self { show: Show::default(), gain: DEFAULT_GAIN, lanes: DEFAULT_LANES, window: None, layout: ViewLayout::Stream, epochs_window: DEFAULT_EPOCHS_WINDOW, epochs_from: EpochsFrom::Auto }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Show {
    #[default]
    Processed,
    Raw,
}

/// Why a document could not be read.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("session document: {0}")]
    Parse(String),
    #[error("session document version {0} is not supported (this app reads version {SESSION_VERSION})")]
    Version(u32),
    #[error("{0}")]
    Io(String),
}

impl SessionDoc {
    pub fn from_json(text: &str) -> Result<Self, SessionError> {
        serde_json::from_str::<Self>(text).map_err(|e| SessionError::Parse(e.to_string()))?.checked()
    }

    fn checked(mut self) -> Result<Self, SessionError> {
        if self.dspk_session != SESSION_VERSION {
            return Err(SessionError::Version(self.dspk_session));
        }
        self.normalize();
        Ok(self)
    }

    /// A graph for a document with views only, then the views from the graph. Views still named
    /// the old default way (`file · /source/path`) take their source's name.
    pub fn normalize(&mut self) {
        if self.graph.is_empty() && !self.views.is_empty() {
            self.graph = GraphDoc::from_views(&self.views);
        }
        self.compile();
        let renames: Vec<(String, String)> = self.views.iter().filter(|v| v.name == legacy_view_name(&v.recording)).map(|v| (v.id.clone(), view_name(&v.recording))).collect();
        if !renames.is_empty() {
            for (id, name) in renames {
                if let Some(super::graph::NodeKind::View { name: old, .. }) = self.graph.nodes.iter_mut().find(|n| n.id == id).map(|n| &mut n.kind) {
                    *old = name;
                }
            }
            self.compile();
        }
    }

    /// The views from the graph; the active view stays when it is still one, else the first.
    pub fn compile(&mut self) {
        self.views = self.graph.compile(&self.recipes);
        if !self.active_view.as_ref().is_some_and(|a| self.views.iter().any(|v| v.id == *a)) {
            self.active_view = self.views.first().map(|v| v.id.clone());
        }
    }

    /// The document as written by [`Self::from_json`] (graph and compiled views).
    pub fn normalized(mut self) -> Self {
        self.normalize();
        self
    }

    /// Reads a document (JSON).
    pub fn load(path: &Path) -> Result<Self, SessionError> {
        let text = std::fs::read_to_string(path).map_err(|e| SessionError::Io(format!("{}: {e}", path.display())))?;
        Self::from_json(&text)
    }

    /// The canonical form: pretty JSON with every default written out.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a session always serializes")
    }

    /// Writes the canonical JSON (through a temporary file, so a reader never sees half of it).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, self.to_json())?;
        std::fs::rename(tmp, path)
    }

    /// JSON Schema of the document (for agents and editors).
    pub fn json_schema() -> String {
        serde_json::to_string_pretty(&schemars::schema_for!(SessionDoc)).expect("a schema always serializes")
    }

    pub fn view(&self, id: &str) -> Option<&ViewDoc> {
        self.views.iter().find(|v| v.id == id)
    }

    pub fn view_mut(&mut self, id: &str) -> Option<&mut ViewDoc> {
        self.views.iter_mut().find(|v| v.id == id)
    }

    pub fn active(&self) -> Option<&ViewDoc> {
        self.active_view.as_deref().and_then(|id| self.view(id))
    }

    /// A view id not used yet (`v1`, `v2`, …), by any view or node.
    pub fn new_view_id(&self) -> String {
        fresh_id("v", |id| self.view(id).is_some() || self.graph.contains(id))
    }

    /// An op id not used yet by any node, from the op's name (`iir`, `iir1`, …).
    pub fn new_op_id(&self, node: &Node) -> String {
        let id = self.graph.fresh_id(op_name(node));
        if self.view(&id).is_none() {
            return id;
        }
        fresh_id(op_name(node), |id| self.graph.contains(id) || self.view(id).is_some())
    }
}

impl ViewDoc {
    pub fn op(&self, id: &str) -> Option<&OpDoc> {
        self.ops.iter().find(|o| o.id == id)
    }

    /// An op id not used yet in this view, from the op's name (`iir`, `iir2`, …).
    pub fn new_op_id(&self, node: &Node) -> String {
        let base = op_name(node);
        if self.op(base).is_none() {
            return base.to_string();
        }
        fresh_id(base, |id| self.op(id).is_some())
    }

    /// The enabled ops as stages, in order.
    pub fn enabled_nodes(&self) -> impl Iterator<Item = &Node> {
        self.ops.iter().filter(|o| o.enabled).map(|o| &o.node)
    }

    /// The pipeline that writes this view (its recording through its enabled ops) to `output`.
    pub fn pipeline(&self, output: PathBuf, format: OutputFormat) -> PipelineSpec {
        let mut nodes = BTreeMap::new();
        let mut edges = Vec::new();
        nodes.insert(
            "read".to_string(),
            Node::Read {
                path: Some(self.recording.path.clone()),
                inputs: None,
                source: self.recording.source.clone(),
                channels: self.recording.channels.clone(),
                start: None,
                stop: None,
            },
        );
        let mut last = "read".to_string();
        for op in self.ops.iter().filter(|o| o.enabled) {
            // Op ids are unique in a view; `read` / `write` are reserved by the commands
            nodes.insert(op.id.clone(), op.node.clone());
            edges.push((last, op.id.clone()));
            last = op.id.clone();
        }
        nodes.insert("write".to_string(), Node::Write { path: output, format, compression: None, chunk_samples: None, parallel: None });
        edges.push((last, "write".to_string()));
        PipelineSpec { dspk_pipeline: SPEC_VERSION, name: Some(self.name.clone()), nodes, edges, execution: Default::default() }
    }
}

/// The family of an op (a recipe's default label).
pub fn op_family(node: &Node) -> &'static str {
    match node {
        Node::Iir { .. } | Node::Fir { .. } | Node::Sos { .. } | Node::Notch { .. } => "filter",
        Node::Car {} | Node::Cmr {} | Node::Laplacian { .. } | Node::Whiten { .. } => "reference",
        Node::Median { .. } | Node::GaussianSmooth { .. } => "smooth",
        Node::Remap { .. } => "channels",
        _ => "transform",
    }
}

/// The spec name of a node (`iir`, `cmr`, …).
pub fn op_name(node: &Node) -> &'static str {
    match node {
        Node::Read { .. } => "read",
        Node::Write { .. } => "write",
        Node::Scale { .. } => "scale",
        Node::SubtractBaseline { .. } => "subtract_baseline",
        Node::Clamp { .. } => "clamp",
        Node::Iir { .. } => "iir",
        Node::Notch { .. } => "notch",
        Node::Sos { .. } => "sos",
        Node::Fir { .. } => "fir",
        Node::Resample { .. } => "resample",
        Node::Remap { .. } => "remap",
        Node::Car {} => "car",
        Node::Cmr {} => "cmr",
        Node::GaussianSmooth { .. } => "gaussian_smooth",
        Node::Median { .. } => "median",
        Node::TeagerKaiser { .. } => "teager_kaiser",
        Node::Rectify {} => "rectify",
        Node::Square {} => "square",
        Node::Sqrt {} => "sqrt",
        Node::Log10 {} => "log10",
        Node::Laplacian { .. } => "laplacian",
        Node::Whiten { .. } => "whiten",
        Node::Crossings { .. } => "crossings",
    }
}

fn fresh_id(prefix: &str, taken: impl Fn(&str) -> bool) -> String {
    (1..).map(|i| format!("{prefix}{i}")).find(|id| !taken(id)).expect("ids are unbounded")
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspk_pipeline::{Band, Design, Mode, Start};

    fn bandpass(lo: f64, hi: f64) -> Node {
        Node::Iir { band: Band::Bandpass, cutoff: vec![lo, hi], order: 3, design: Design::Butterworth, ripple_db: None, mode: Mode::ForwardBackward, start: Start::Rest }
    }

    fn doc() -> SessionDoc {
        let view = ViewDoc {
            id: "v1".into(),
            name: "probe A".into(),
            recording: RecordingRef { path: "/data/rec.bin".into(), source: None, channels: Some(vec![0, 1, 2]) },
            ops: vec![
                OpDoc { id: "iir".into(), enabled: true, node: bandpass(300.0, 6000.0) },
                OpDoc { id: "cmr".into(), enabled: false, node: Node::Cmr {} },
            ],
            display: DisplayDoc::default(),
            plugins: BTreeMap::new(),
        };
        SessionDoc { views: vec![view], active_view: Some("v1".into()), ..SessionDoc::default() }.normalized()
    }

    #[test]
    fn test_json_round_trip() {
        let d = doc();
        assert_eq!(d.graph.nodes.len(), 4, "source, two ops, view");
        assert_eq!(SessionDoc::from_json(&d.to_json()).unwrap(), d);
    }

    #[test]
    fn test_defaults_are_filled() {
        let text = r#"{"dspk_session": 1, "active_view": "v1",
            "views": [{"id": "v1", "name": "probe A", "recording": {"path": "/data/rec.bin"},
                       "ops": [{"id": "hp", "node": {"op": "iir", "band": "highpass", "cutoff": [300.0]}}]}]}"#;
        let d = SessionDoc::from_json(text).unwrap();
        assert_eq!(d.views[0].ops[0].id, "hp");
        assert!(d.views[0].ops[0].enabled);
        assert_eq!(d.views[0].display, DisplayDoc::default());
    }

    #[test]
    fn test_unknown_version_and_fields_fail() {
        assert!(matches!(SessionDoc::from_json(r#"{"dspk_session": 2}"#), Err(SessionError::Version(2))));
        assert!(SessionDoc::from_json(r#"{"dspk_session": 1, "viewz": []}"#).is_err());
    }

    #[test]
    fn test_pipeline_of_a_view_skips_disabled_ops() {
        let spec = doc().views[0].pipeline("/out/x.zarr".into(), OutputFormat::Zarr);
        let chain = spec.chain().unwrap();
        assert_eq!(chain.stages.iter().map(|(id, _)| *id).collect::<Vec<_>>(), ["iir"]);
        match chain.read.1 {
            Node::Read { channels, .. } => assert_eq!(channels.as_deref(), Some(&[0, 1, 2][..])),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_ids() {
        let d = doc();
        assert_eq!(d.new_view_id(), "v2");
        assert_eq!(d.views[0].new_op_id(&bandpass(1.0, 2.0)), "iir1");
        assert_eq!(d.views[0].new_op_id(&Node::Car {}), "car");
        assert_eq!(d.new_op_id(&bandpass(1.0, 2.0)), "iir1");
    }

    #[test]
    fn test_schema_names_the_document() {
        assert!(SessionDoc::json_schema().contains("dspk_session"));
    }
}

/// The name of a view of `r`: its source's name (the last part of the source's path: `EMG` for
/// `/acquisition/EMG`), or the file's name for a file of one source.
pub fn view_name(r: &RecordingRef) -> String {
    match r.source.as_deref().filter(|s| !s.is_empty() && *s != dspk_io::sources::MAIN) {
        Some(s) => source_name(s),
        None => file_name(&r.path),
    }
}

/// A source's short name: the last part of its path (`EMG` for `/acquisition/EMG`).
pub fn source_name(id: &str) -> String {
    id.rsplit('/').find(|p| !p.is_empty()).unwrap_or(id).to_string()
}

fn file_name(path: &Path) -> String {
    path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned())
}

/// The default view name before views were named by source (`file · /source/path`).
fn legacy_view_name(r: &RecordingRef) -> String {
    match &r.source {
        Some(s) => format!("{} · {s}", file_name(&r.path)),
        None => file_name(&r.path),
    }
}
