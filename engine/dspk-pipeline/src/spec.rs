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
use dspk_ops::filter::fir::{FIR_DEFAULT_EDGE, GAUSSIAN_DEFAULT_EDGE};
use dspk_ops::filter::non_linear::{MEDIAN_DEFAULT_EDGE, TEAGER_KAISER_DEFAULT_EDGE};
use dspk_ops::filter::{FilterBand, FilterDesign, FilterMode, FilterSpec, FilterStart, Section, Sos};
use dspk_ops::resampler::poly::{RESAMPLE_POLY_DEFAULT_EDGE, RESAMPLE_POLY_WINDOW};
use dspk_ops::resampler::FirWindow;
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
    /// The input recording(s): `path` (one file, or a glob such as `sessions/*.bin`, which runs the
    /// pipeline once per match) or `inputs` (a list of paths / globs and merges, run one after the
    /// other). A merge reads several recordings as one, channels side by side. Relative paths are
    /// relative to the spec file. With several inputs the write path needs `{stem}` or `{index}`.
    Read {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<PathBuf>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        inputs: Option<Vec<Input>>,
        /// Source id within the file (`dspk_io::sources`; with `path` only); default: the file's
        /// default source.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<String>,
        /// Channels to read, in this order; default: all.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        channels: Option<Vec<usize>>,
        /// First sample to process (samples, not seconds, so a range is exact); default: 0.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        start: Option<u64>,
        /// Sample after the last one to process; default: the end of the recording. Halos of
        /// window mode may read real data outside `start..stop`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stop: Option<u64>,
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
    /// FIR filter with explicit `taps` (design them with `dspk.ops.filter.firwin` or scipy):
    /// causal, or `centered` (odd number of taps around each sample).
    Fir {
        taps: Vec<f32>,
        #[serde(default)]
        centered: bool,
        #[serde(default = "default_fir_edge")]
        edge: Edge,
    },
    /// Resampling by `up / down` (`scipy.signal.resample_poly`: polyphase FIR, Kaiser `beta`
    /// window); must be the last node before `write`. The output's sample rate is
    /// `rate · up / down`; with a `read` range, its `start` must be a multiple of `down`.
    Resample {
        up: usize,
        down: usize,
        #[serde(default = "default_resample_beta")]
        beta: f64,
        #[serde(default = "default_resample_edge")]
        edge: Edge,
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

/// One input of a `read` node: a path (or glob), or several recordings merged side by side.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Input {
    Path(PathBuf),
    Merge {
        /// The recordings, in channel order; same sample rate and length.
        merge: Vec<MergePart>,
    },
}

/// One recording of a merge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MergePart {
    pub path: PathBuf,
    /// Source id within the file; default: the file's default source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Channels taken from this recording, in order; default: all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channels: Option<Vec<usize>>,
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
    #[serde(default)]
    pub upload: Upload,
}

/// What crosses to the device. `stored`: the recording's stored integers, scaled on the device
/// (int16 moves half the bytes of float32). `scaled`: float32 values scaled on the host. `auto`:
/// `stored` for integer recordings, `scaled` otherwise (float32 is the same either way; float64 is
/// scaled on the host).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Upload {
    #[default]
    Auto,
    Stored,
    Scaled,
}

impl Default for Execution {
    fn default() -> Self {
        Self { mode: ExecMode::default(), chunk_samples: DEFAULT_CHUNK_SAMPLES, runtime: None, upload: Upload::default() }
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
fn default_resample_beta() -> f64 {
    match RESAMPLE_POLY_WINDOW {
        FirWindow::Kaiser { beta } => beta,
        FirWindow::Hamming => unreachable!("resample_poly's default window is Kaiser"),
    }
}
fn default_resample_edge() -> Edge {
    Edge::from(RESAMPLE_POLY_DEFAULT_EDGE)
}
fn default_fir_edge() -> Edge {
    Edge::from(FIR_DEFAULT_EDGE)
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
        if let Node::Read { path, inputs, source, .. } = &self.nodes[read] {
            match (path, inputs) {
                (Some(_), None) => {}
                (None, Some(list)) if !list.is_empty() => {
                    if source.is_some() {
                        return invalid("read: `source` goes with `path`; give each merge part its own `source`".into());
                    }
                    if list.iter().any(|i| matches!(i, Input::Merge { merge } if merge.is_empty())) {
                        return invalid("read: a merge needs at least one recording".into());
                    }
                }
                (None, Some(_)) => return invalid("read: inputs is empty".into()),
                _ => return invalid("read: give exactly one of `path` and `inputs`".into()),
            }
        }
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

/// One concrete input after glob expansion: the recordings it reads (one, or a merge) and its stem.
#[derive(Debug, Clone)]
struct Concrete {
    parts: Vec<MergePart>,
    merged: bool,
}

impl PipelineSpec {
    /// One spec per input, globs expanded against `base_dir` (sorted), `{stem}` / `{index}` in the
    /// write path filled in. Each is a valid single-input spec: what the executor runs and writes
    /// next to that output as provenance.
    pub fn expand_inputs(&self, base_dir: &Path) -> Result<Vec<PipelineSpec>, PipelineError> {
        let chain = self.chain()?;
        let (read_id, write_id) = (chain.read.0.to_string(), chain.write.0.to_string());
        let Node::Read { path, inputs, source, .. } = chain.read.1 else { unreachable!("chain starts with read") };
        let Node::Write { path: out_path, .. } = chain.write.1 else { unreachable!("chain ends with write") };
        let items: Vec<Input> = match (path, inputs) {
            (Some(p), None) => vec![Input::Path(p.clone())],
            (None, Some(list)) => list.clone(),
            _ => unreachable!("checked by chain"),
        };
        let mut concrete = Vec::new();
        for item in items {
            match item {
                Input::Path(p) => {
                    for path in expand_glob(base_dir, &p)? {
                        concrete.push(Concrete { parts: vec![MergePart { path, source: source.clone(), channels: None }], merged: false });
                    }
                }
                Input::Merge { merge } => concrete.push(Concrete { parts: merge, merged: true }),
            }
        }
        let template = out_path.to_string_lossy().to_string();
        if concrete.len() > 1 && !template.contains("{stem}") && !template.contains("{index}") {
            return Err(PipelineError::Invalid(format!(
                "{} inputs would all write {template:?}: put {{stem}} or {{index}} in the write path",
                concrete.len()
            )));
        }
        Ok(concrete
            .into_iter()
            .enumerate()
            .map(|(index, input)| {
                let stem = input.parts.iter().map(|p| p.path.file_stem().unwrap_or_default().to_string_lossy().to_string()).collect::<Vec<_>>().join("+");
                let mut one = self.clone();
                if let Some(Node::Read { path, inputs, source, .. }) = one.nodes.get_mut(&read_id) {
                    if input.merged {
                        (*path, *inputs, *source) = (None, Some(vec![Input::Merge { merge: input.parts }]), None);
                    } else {
                        let part = input.parts.into_iter().next().expect("one part");
                        (*path, *inputs, *source) = (Some(part.path), None, part.source);
                    }
                }
                if let Some(Node::Write { path, .. }) = one.nodes.get_mut(&write_id) {
                    *path = PathBuf::from(template.replace("{stem}", &stem).replace("{index}", &index.to_string()));
                }
                one
            })
            .collect())
    }
}

impl PipelineSpec {
    /// Replaces what the read node reads (e.g. paths given on a command line): one plain path
    /// becomes `path`, anything else `inputs`. The read node's `source` is kept with a single path.
    pub fn set_inputs(&mut self, new: Vec<Input>) -> Result<(), PipelineError> {
        let id = self.chain()?.read.0.to_string();
        let Some(Node::Read { path, inputs, source, .. }) = self.nodes.get_mut(&id) else { unreachable!("read node") };
        match new.as_slice() {
            [] => return Err(PipelineError::Invalid("read: no inputs given".into())),
            [Input::Path(p)] => (*path, *inputs) = (Some(p.clone()), None),
            _ => (*path, *inputs, *source) = (None, Some(new), None),
        }
        Ok(())
    }

    /// Replaces the write node's path (`{stem}` / `{index}` still apply).
    pub fn set_output(&mut self, new: PathBuf) -> Result<(), PipelineError> {
        let id = self.chain()?.write.0.to_string();
        let Some(Node::Write { path, .. }) = self.nodes.get_mut(&id) else { unreachable!("write node") };
        *path = new;
        Ok(())
    }
}

/// `pattern` itself when it has no wildcard, else its matches (sorted) relative to `base_dir`.
fn expand_glob(base_dir: &Path, pattern: &Path) -> Result<Vec<PathBuf>, PipelineError> {
    let text = pattern.to_string_lossy();
    if !text.contains(['*', '?', '[']) {
        return Ok(vec![pattern.to_path_buf()]);
    }
    let full = base_dir.join(pattern);
    let mut found: Vec<PathBuf> = glob::glob(&full.to_string_lossy())
        .map_err(|e| PipelineError::Invalid(format!("read: bad pattern {text:?}: {e}")))?
        .filter_map(Result::ok)
        .collect();
    found.sort();
    if found.is_empty() {
        return Err(PipelineError::Invalid(format!("read: no file matches {text:?} (in {})", base_dir.display())));
    }
    Ok(found)
}

impl Node {
    /// The dspk-ops stage of a processing node (`None` for `read`, `write` and `resample`, which the
    /// executor runs itself).
    pub fn stage(&self) -> Result<Option<PipelineStage>, PipelineError> {
        let filter = |design: FilterDesign, mode: Mode, start: Start| {
            Ok(Some(PipelineStage::Filter(FilterSpec { design, mode: mode.into(), start: start.into() })))
        };
        match self {
            Node::Read { .. } | Node::Write { .. } => Ok(None),
            Node::Resample { up, down, .. } => {
                if *up == 0 || *down == 0 {
                    return Err(PipelineError::Invalid(format!("resample: up and down must be at least 1, got {up} / {down}")));
                }
                Ok(None)
            }
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
            Node::Fir { taps, centered, edge } => {
                if taps.is_empty() {
                    return Err(PipelineError::Invalid("fir: taps is empty".into()));
                }
                if *centered && taps.len() % 2 == 0 {
                    return Err(PipelineError::Invalid(format!("fir: centered needs an odd number of taps, got {}", taps.len())));
                }
                Ok(Some(PipelineStage::Fir { taps: taps.clone(), centered: *centered, edge: (*edge).into() }))
            }
            Node::Car {} => Ok(Some(PipelineStage::CommonAverageReference)),
            Node::Cmr {} => Ok(Some(PipelineStage::CommonMedianReference)),
            Node::GaussianSmooth { sigma_samples, edge } => Ok(Some(PipelineStage::GaussianSmooth { sigma_samples: *sigma_samples, edge: (*edge).into() })),
            Node::Median { width, edge } => Ok(Some(PipelineStage::Median { width: *width, edge: (*edge).into() })),
            Node::TeagerKaiser { edge } => Ok(Some(PipelineStage::TeagerKaiser { edge: (*edge).into() })),
        }
    }
}
