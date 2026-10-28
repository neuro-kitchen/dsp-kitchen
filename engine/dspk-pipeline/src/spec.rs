//! The pipeline spec: nodes (one operation each) connected by edges, versioned.
//!
//! JSON is the canonical form (written next to every output, sent over the network, stored as
//! provenance); TOML is accepted for files written by hand. Both parse into [`PipelineSpec`].
//! Unknown fields are errors, so a misspelt option fails instead of silently taking its default.
//!
//! Spec version 1 runs a single chain: one `read`, processing nodes, one `write`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use dspk_ops::core::EdgeMode;
use dspk_ops::filter::design::DEFAULT_BUTTERWORTH_ORDER;
use dspk_ops::filter::fir::GAUSSIAN_DEFAULT_EDGE;
use dspk_ops::filter::non_linear::{MEDIAN_DEFAULT_EDGE, TEAGER_KAISER_DEFAULT_EDGE};
use dspk_ops::filter::{FilterBand, FilterDesign, FilterMode, FilterSpec, FilterStart, Section, Sos};
use dspk_ops::PipelineStage;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::PipelineError;

/// The spec version this crate reads and writes (`dspk_pipeline` field).
pub const SPEC_VERSION: u32 = 1;

/// Samples per channel in each window the executor processes (2.2 s at 30 kHz).
pub const DEFAULT_CHUNK_SAMPLES: usize = 65_536;

/// A pipeline: operations (nodes, by id) and the edges between them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PipelineSpec {
    /// Spec version; must equal [`SPEC_VERSION`].
    pub dspk_pipeline: u32,
    /// Free-form name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Every operation, by node id.
    pub nodes: BTreeMap<String, Node>,
    /// `[from, to]` node ids.
    pub edges: Vec<(String, String)>,
    #[serde(default)]
    pub execution: Execution,
}

/// One operation. `read` and `write` are the ends; the others mirror `dspk.ops.stage`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Node {
    /// A recording file (any format `dspk_io::open` recognizes). Relative paths are relative to the
    /// spec file.
    Read {
        path: PathBuf,
        /// Source id within the file (`dspk_io::sources`); default: the file's default source.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<String>,
    },
    /// The output file. Relative paths are relative to the spec file.
    Write {
        path: PathBuf,
        #[serde(default)]
        format: OutputFormat,
    },
    /// `y = alpha * x + beta`.
    Scale {
        alpha: f32,
        #[serde(default)]
        beta: f32,
    },
    /// `y = x - baseline`.
    SubtractBaseline { baseline: f32 },
    /// `y = clip(x, min, max)`.
    Clamp { min: f32, max: f32 },
    /// IIR filter designed for the recording's sample rate.
    Iir {
        band: Band,
        /// Hz: one value (lowpass, highpass) or `[low, high]` (bandpass, bandstop).
        cutoff: Vec<f64>,
        #[serde(default = "default_order")]
        order: usize,
        #[serde(default)]
        design: Design,
        /// Pass-band ripple, chebyshev1 only.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ripple_db: Option<f64>,
        #[serde(default)]
        mode: Mode,
        #[serde(default)]
        start: Start,
    },
    /// Second-order notch (`scipy.signal.iirnotch`).
    Notch {
        freq: f64,
        q: f64,
        #[serde(default)]
        mode: Mode,
        #[serde(default)]
        start: Start,
    },
    /// Explicit second-order sections `[b0, b1, b2, a0, a1, a2]` (scipy layout), already designed
    /// for the recording's sample rate.
    Sos {
        sections: Vec<[f64; 6]>,
        #[serde(default)]
        mode: Mode,
        #[serde(default)]
        start: Start,
    },
    /// Common average reference.
    Car {},
    /// Common median reference.
    Cmr {},
    /// Zero-phase Gaussian smoothing.
    GaussianSmooth {
        sigma_samples: f32,
        #[serde(default = "default_gaussian_edge")]
        edge: Edge,
    },
    /// Running median over an odd `width`.
    Median {
        width: usize,
        #[serde(default = "default_median_edge")]
        edge: Edge,
    },
    /// Teager–Kaiser energy.
    TeagerKaiser {
        #[serde(default = "default_teager_kaiser_edge")]
        edge: Edge,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    /// Raw float32, time-major, with a JSON sidecar (`.meta`).
    #[default]
    Raw,
    /// Zarr `/traces`, float32.
    Zarr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Band {
    Lowpass,
    Highpass,
    Bandpass,
    Bandstop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Design {
    #[default]
    Butterworth,
    Chebyshev1,
    Bessel,
}

/// `forward_backward`: zero phase (`sosfiltfilt`); `forward`: causal (`sosfilt`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    ForwardBackward,
    Forward,
}

/// How a forward filter starts: at rest, or from the steady state of the first sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Start {
    #[default]
    Rest,
    SteadyState,
}

/// Samples past the ends of a window (scipy names: `constant` 0, `odd`, `reflect`, `nearest`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Edge {
    Zeros,
    Odd,
    Reflect,
    Nearest,
}

/// How the executor runs the chain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Execution {
    #[serde(default)]
    pub mode: ExecMode,
    /// Samples per channel processed per window.
    #[serde(default = "default_chunk_samples")]
    pub chunk_samples: usize,
    /// `wgpu`, `cuda`, `hip`, `cpu`; default: `DSPK_RUNTIME`, else the first compiled in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<String>,
}

impl Default for Execution {
    fn default() -> Self {
        Self { mode: ExecMode::default(), chunk_samples: DEFAULT_CHUNK_SAMPLES, runtime: None }
    }
}

/// `stream`: windows in order, filter state carried (exact; forward filters, point-wise and
/// spatial stages only). `windows`: each window on its own with halos of the chain's settling
/// (zero-phase filters, stencil stages). `auto`: `stream` when the chain allows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ExecMode {
    #[default]
    Auto,
    Stream,
    Windows,
}

fn default_order() -> usize {
    DEFAULT_BUTTERWORTH_ORDER
}
fn default_chunk_samples() -> usize {
    DEFAULT_CHUNK_SAMPLES
}
fn default_gaussian_edge() -> Edge {
    Edge::from(GAUSSIAN_DEFAULT_EDGE)
}
fn default_median_edge() -> Edge {
    Edge::from(MEDIAN_DEFAULT_EDGE)
}
fn default_teager_kaiser_edge() -> Edge {
    Edge::from(TEAGER_KAISER_DEFAULT_EDGE)
}

impl From<EdgeMode> for Edge {
    fn from(edge: EdgeMode) -> Self {
        match edge {
            EdgeMode::Zeros => Edge::Zeros,
            EdgeMode::Odd => Edge::Odd,
            EdgeMode::Reflect => Edge::Reflect,
            EdgeMode::Nearest => Edge::Nearest,
        }
    }
}

impl From<Edge> for EdgeMode {
    fn from(edge: Edge) -> Self {
        match edge {
            Edge::Zeros => EdgeMode::Zeros,
            Edge::Odd => EdgeMode::Odd,
            Edge::Reflect => EdgeMode::Reflect,
            Edge::Nearest => EdgeMode::Nearest,
        }
    }
}

impl From<Mode> for FilterMode {
    fn from(mode: Mode) -> Self {
        match mode {
            Mode::ForwardBackward => FilterMode::ForwardBackward,
            Mode::Forward => FilterMode::Forward,
        }
    }
}

impl From<Start> for FilterStart {
    fn from(start: Start) -> Self {
        match start {
            Start::Rest => FilterStart::Rest,
            Start::SteadyState => FilterStart::SteadyState,
        }
    }
}

/// The chain of a valid spec, in order: the `read` node, the processing nodes, the `write` node.
#[derive(Debug, Clone)]
pub struct Chain<'a> {
    pub read: (&'a str, &'a Node),
    pub stages: Vec<(&'a str, &'a Node)>,
    pub write: (&'a str, &'a Node),
}

impl PipelineSpec {
    /// Parses JSON.
    pub fn from_json(text: &str) -> Result<Self, PipelineError> {
        serde_json::from_str(text).map_err(|e| PipelineError::Parse(e.to_string()))
    }

    /// Parses TOML.
    pub fn from_toml(text: &str) -> Result<Self, PipelineError> {
        toml::from_str(text).map_err(|e| PipelineError::Parse(e.to_string()))
    }

    /// Reads a spec file: `.toml` as TOML, anything else as JSON.
    pub fn load(path: &Path) -> Result<Self, PipelineError> {
        let text = std::fs::read_to_string(path).map_err(|e| PipelineError::Io(format!("{}: {e}", path.display())))?;
        match path.extension().and_then(|e| e.to_str()) {
            Some("toml") => Self::from_toml(&text),
            _ => Self::from_json(&text),
        }
    }

    /// The canonical form: pretty JSON with every default written out.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a spec always serializes")
    }

    /// JSON Schema of the spec (editors validate and complete spec files against it).
    pub fn json_schema() -> String {
        serde_json::to_string_pretty(&schemars::schema_for!(PipelineSpec)).expect("a schema always serializes")
    }

    /// Checks the version and the graph; returns the chain from `read` to `write`.
    pub fn chain(&self) -> Result<Chain<'_>, PipelineError> {
        let invalid = |msg: String| Err(PipelineError::Invalid(msg));
        if self.dspk_pipeline != SPEC_VERSION {
            return invalid(format!("spec version {} is not supported (this dspk reads version {SPEC_VERSION})", self.dspk_pipeline));
        }
        let of_op = |want: fn(&Node) -> bool| self.nodes.iter().filter(|(_, n)| want(n)).map(|(id, _)| id.as_str()).collect::<Vec<_>>();
        let reads = of_op(|n| matches!(n, Node::Read { .. }));
        let writes = of_op(|n| matches!(n, Node::Write { .. }));
        if reads.len() != 1 || writes.len() != 1 {
            return invalid(format!("version 1 runs one chain: exactly one read and one write node (found {} read, {} write)", reads.len(), writes.len()));
        }
        let mut next: BTreeMap<&str, &str> = BTreeMap::new();
        let mut has_input: BTreeSet<&str> = BTreeSet::new();
        for (from, to) in &self.edges {
            for id in [from, to] {
                if !self.nodes.contains_key(id) {
                    return invalid(format!("edge [{from:?}, {to:?}]: no node {id:?}"));
                }
            }
            if from == to {
                return invalid(format!("edge [{from:?}, {to:?}] connects a node to itself"));
            }
            if next.insert(from, to).is_some() {
                return invalid(format!("node {from:?} has more than one output (branching is not supported in version 1)"));
            }
            if !has_input.insert(to) {
                return invalid(format!("node {to:?} has more than one input (merging is not supported in version 1)"));
            }
        }
        let (read, write) = (reads[0], writes[0]);
        if has_input.contains(read) {
            return invalid(format!("the read node {read:?} cannot have an input"));
        }
        if next.contains_key(write) {
            return invalid(format!("the write node {write:?} cannot have an output"));
        }
        let mut order = vec![read];
        while let Some(&to) = next.get(order.last().expect("starts with read")) {
            order.push(to);
            if order.len() > self.nodes.len() {
                return invalid("the edges form a cycle".into());
            }
        }
        if order.last() != Some(&write) {
            return invalid(format!("no path of edges leads from {read:?} to {write:?}"));
        }
        if order.len() != self.nodes.len() {
            let unused: Vec<&str> = self.nodes.keys().map(String::as_str).filter(|id| !order.contains(id)).collect();
            return invalid(format!("nodes not on the chain from {read:?} to {write:?}: {unused:?}"));
        }
        let node = |id: &'_ str| (self.nodes.get_key_value(id).map(|(k, _)| k.as_str()).expect("checked"), &self.nodes[id]);
        Ok(Chain {
            read: node(read),
            stages: order[1..order.len() - 1].iter().map(|id| node(id)).collect(),
            write: node(write),
        })
    }
}

impl Node {
    /// The dspk-ops stage of a processing node (`None` for `read` and `write`).
    pub fn stage(&self) -> Result<Option<PipelineStage>, PipelineError> {
        let filter = |design: FilterDesign, mode: Mode, start: Start| {
            Ok(Some(PipelineStage::Filter(FilterSpec { design, mode: mode.into(), start: start.into() })))
        };
        match self {
            Node::Read { .. } | Node::Write { .. } => Ok(None),
            Node::Scale { alpha, beta } => Ok(Some(PipelineStage::Scale { alpha: *alpha, beta: *beta })),
            Node::SubtractBaseline { baseline } => Ok(Some(PipelineStage::SubtractBaseline { baseline: *baseline })),
            Node::Clamp { min, max } => {
                if min > max {
                    return Err(PipelineError::Invalid(format!("clamp: min {min} is above max {max}")));
                }
                Ok(Some(PipelineStage::Clamp { min: *min, max: *max }))
            }
            Node::Iir { band, cutoff, order, design, ripple_db, mode, start } => {
                let band = match (band, cutoff.as_slice()) {
                    (Band::Lowpass, [f]) => FilterBand::Lowpass(*f),
                    (Band::Highpass, [f]) => FilterBand::Highpass(*f),
                    (Band::Bandpass, [lo, hi]) => FilterBand::Bandpass(*lo, *hi),
                    (Band::Bandstop, [lo, hi]) => FilterBand::Bandstop(*lo, *hi),
                    (b, c) => return Err(PipelineError::Invalid(format!("iir {b:?}: wrong number of cutoff frequencies ({})", c.len()))),
                };
                let order = *order;
                let design = match (design, ripple_db) {
                    (Design::Butterworth, None) => FilterDesign::Butterworth { order, band },
                    (Design::Bessel, None) => FilterDesign::Bessel { order, band },
                    (Design::Chebyshev1, Some(ripple_db)) => FilterDesign::Chebyshev1 { order, ripple_db: *ripple_db, band },
                    (Design::Chebyshev1, None) => return Err(PipelineError::Invalid("iir chebyshev1 needs ripple_db".into())),
                    (d, Some(_)) => return Err(PipelineError::Invalid(format!("iir: ripple_db applies to chebyshev1 only, not {d:?}"))),
                };
                filter(design, *mode, *start)
            }
            Node::Notch { freq, q, mode, start } => filter(FilterDesign::Notch { freq_hz: *freq, q: *q }, *mode, *start),
            Node::Sos { sections, mode, start } => {
                let sos = Sos::new(sections.iter().map(|r| Section { b: [r[0], r[1], r[2]], a: [r[3], r[4], r[5]] }).collect());
                sos.validate().map_err(|e| PipelineError::Invalid(format!("sos: {e}")))?;
                filter(FilterDesign::Sos(sos), *mode, *start)
            }
            Node::Car {} => Ok(Some(PipelineStage::CommonAverageReference)),
            Node::Cmr {} => Ok(Some(PipelineStage::CommonMedianReference)),
            Node::GaussianSmooth { sigma_samples, edge } => Ok(Some(PipelineStage::GaussianSmooth { sigma_samples: *sigma_samples, edge: (*edge).into() })),
            Node::Median { width, edge } => Ok(Some(PipelineStage::Median { width: *width, edge: (*edge).into() })),
            Node::TeagerKaiser { edge } => Ok(Some(PipelineStage::TeagerKaiser { edge: (*edge).into() })),
        }
    }
}
