//! Pipelines on a live source: the same spec as for files, run block after block as data arrives.
//!
//! **The rule:** a live input accepts only ops whose output for a block is exact from that block
//! and the state carried from earlier ones. The check ([`check_live`]) names each refused node and
//! why; the device, the app and the CLI call the same function, so they refuse the same things.
//!
//! | [`Timing`] | Ops | Live |
//! |---|---|---|
//! | `Causal` | scale, baseline, clamp, rectify, square, sqrt, log10, CAR, CMR, Laplacian, remap, forward IIR / notch / SOS, crossings | yes |
//! | `History(n)` | causal FIR (n − 1 past samples) | yes: the samples before each block are carried |
//! | `Delay(n)` | centred FIR, median, Teager–Kaiser, Gaussian smoothing | yes: carried history, output `n` samples late (reported) |
//! | `Offline` | zero-phase filters, whitening (fitted on the recording) | never while recording |
//!
//! Resampling a live source is refused for now (its output rate differs from the input's).
//!
//! **The spec on a device** is a pipeline spec with two additions, taken out before the rest is
//! parsed as usual (so files and devices share one format):
//! - a top-level `"live": {"source": "<device source>", "block_ms": 10}`;
//! - on any node, `"tap": "<name>"` (or `{"name": …, "record": true}`): that node's output is
//!   published under the name, and recorded when `record` is set. A `write` node after a node
//!   records it too.
//!
//! The `read` node may name the device's source in `source` instead of `live.source`. Version 1
//! runs one chain without branches.

use std::collections::BTreeMap;

use dspk_core::{ComputeTarget, DspError, DspResult};
use dspk_ops::core::buffer;
use dspk_ops::crossings::{CrossingEvent, CrossingOptions, CrossingTracker};
use dspk_ops::pipeline::PipelineWorkspace;
use dspk_ops::{Pipeline, PipelineStage};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Mode, Node, PipelineSpec};

/// Block length of a chain that does not say (ms).
pub const DEFAULT_BLOCK_MS: f64 = 10.0;

/// When an op's output for a block is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timing {
    /// From the block and state carried from earlier blocks.
    Causal,
    /// Also needs this many samples before the block (not carried by stateful workspaces yet).
    History(usize),
    /// Also needs this many samples after each one: its output lags by them.
    Delay(usize),
    /// Needs the whole recording (zero-phase filters, a fit).
    Offline,
}

/// The `op` name of a node (`iir`, `car`, …).
pub fn op_name(node: &Node) -> String {
    serde_json::to_value(node).ok().and_then(|v| v.get("op").and_then(Value::as_str).map(str::to_string)).unwrap_or_default()
}

/// When `node`'s output is known, and why when it is not live.
pub fn timing(node: &Node) -> (Timing, &'static str) {
    let filter = |mode: &Mode| match mode {
        Mode::Forward => (Timing::Causal, ""),
        Mode::ForwardBackward => (Timing::Offline, "zero-phase (forward-backward) filtering needs future samples: set mode = \"forward\""),
    };
    match node {
        Node::Read { .. } | Node::Write { .. } | Node::Remap { .. } | Node::Crossings { .. } => (Timing::Causal, ""),
        Node::Scale { .. } | Node::SubtractBaseline { .. } | Node::Clamp { .. } | Node::Rectify {} | Node::Square {} | Node::Sqrt {} | Node::Log10 {} => {
            (Timing::Causal, "")
        }
        Node::Car {} | Node::Cmr {} | Node::Laplacian { .. } => (Timing::Causal, ""),
        Node::Iir { mode, .. } | Node::Notch { mode, .. } | Node::Sos { mode, .. } => filter(mode),
        Node::Whiten { .. } => (Timing::Offline, "whitening is fitted on the recording: not on a live source (a fit loaded from an earlier recording is planned)"),
        Node::Fir { taps, centered: false, .. } => (Timing::History(taps.len().saturating_sub(1)), "a FIR filter needs the samples before each block; carried history is planned"),
        Node::Fir { taps, centered: true, .. } => (Timing::Delay(taps.len() / 2), "a centred FIR filter needs samples after each one; delayed output is planned"),
        Node::Median { width, .. } => (Timing::Delay(width / 2), "a running median needs samples after each one; delayed output is planned"),
        Node::TeagerKaiser { .. } => (Timing::Delay(1), "Teager–Kaiser energy needs the next sample; delayed output is planned"),
        Node::GaussianSmooth { sigma_samples, .. } => {
            (Timing::Delay((sigma_samples * 4.0).ceil().max(1.0) as usize), "Gaussian smoothing is zero-phase: it needs samples after each one")
        }
        Node::Resample { .. } => (Timing::Delay(0), "resampling a live source is planned"),
    }
}

/// Why one node of a spec was refused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Refusal {
    /// Node id ("" for the spec as a whole).
    pub node: String,
    pub op: String,
    pub reason: String,
}

impl Refusal {
    fn whole(reason: impl Into<String>) -> Self {
        Self { node: String::new(), op: String::new(), reason: reason.into() }
    }
}

/// What a tap carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveTapKind {
    /// Blocks of a signal.
    Samples,
    /// Threshold-crossing events.
    Events,
}

/// A named output of a chain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveTap {
    pub name: String,
    /// The node it comes from.
    pub node: String,
    pub kind: LiveTapKind,
    /// Recorded next to the raw data.
    pub record: bool,
    /// Channel count of its output.
    pub channels: usize,
    /// Samples its frames lag the source (the look-ahead of the stencils before it).
    pub delay_samples: u64,
}

/// One device step of a chain, in order.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveStep {
    pub node: String,
    pub op: String,
    /// The device stage (`None`: nothing to run, e.g. a tap on a remap folded into the read).
    pub stage: Option<PipelineStage>,
}

/// A chain checked for a live input: what runs, in order, and what it publishes.
#[derive(Debug, Clone)]
pub struct LiveChain {
    /// The spec as configured (without its live additions).
    pub spec: PipelineSpec,
    /// The canonical JSON (with them), as a device reports it.
    pub spec_json: String,
    /// The device source the chain reads (`None`: the device's only one).
    pub source: Option<String>,
    pub block_ms: f64,
    /// Source channels read, in order (leading remaps fold into this).
    pub input_channels: Vec<usize>,
    pub steps: Vec<LiveStep>,
    /// The crossings node ending the chain, if any.
    pub crossings: Option<(String, CrossingParams)>,
    pub taps: Vec<LiveTap>,
    /// Samples the chain's output lags its input (0 until delayed ops are allowed).
    pub delay_samples: u64,
}

/// A crossings node's settings, before the sample rate turns times into samples.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossingParams {
    pub level: f32,
    pub release: Option<f32>,
    pub direction: crate::Direction,
    pub min_duration_ms: f64,
    pub refractory_ms: f64,
}

impl CrossingParams {
    pub fn options(&self, sample_rate: f64) -> CrossingOptions {
        let to_samples = |ms: f64| (ms * 1e-3 * sample_rate).round() as u64;
        CrossingOptions {
            levels: vec![self.level],
            releases: self.release.map(|r| vec![r]).unwrap_or_default(),
            direction: self.direction.into(),
            min_duration_samples: to_samples(self.min_duration_ms),
            refractory_samples: to_samples(self.refractory_ms),
        }
    }
}

/// The live additions of a spec.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveSettings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_ms: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
enum TapDecl {
    Name(String),
    Full {
        name: String,
        #[serde(default)]
        record: bool,
    },
}

/// Splits a device spec (JSON or TOML) into the pipeline spec, its taps (node id → name, record)
/// and its live settings.
pub fn parse_live_spec(text: &str) -> Result<(PipelineSpec, BTreeMap<String, (String, bool)>, LiveSettings), String> {
    let mut value: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(json_err) => {
            let toml: toml::Value = toml::from_str(text).map_err(|e| format!("neither JSON ({json_err}) nor TOML ({e})"))?;
            serde_json::to_value(toml).map_err(|e| e.to_string())?
        }
    };
    let obj = value.as_object_mut().ok_or("a spec is an object")?;
    let settings: LiveSettings = match obj.remove("live") {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("live: {e}"))?,
        None => LiveSettings::default(),
    };
    let mut taps = BTreeMap::new();
    if let Some(nodes) = obj.get_mut("nodes").and_then(Value::as_object_mut) {
        for (id, node) in nodes.iter_mut() {
            if let Some(tap) = node.as_object_mut().and_then(|n| n.remove("tap")) {
                let decl: TapDecl = serde_json::from_value(tap).map_err(|e| format!("node {id}: tap: {e}"))?;
                let (name, record) = match decl {
                    TapDecl::Name(n) => (n, false),
                    TapDecl::Full { name, record } => (name, record),
                };
                taps.insert(id.clone(), (name, record));
            }
        }
    }
    let spec: PipelineSpec = serde_json::from_value(value).map_err(|e| e.to_string())?;
    Ok((spec, taps, settings))
}

/// Checks a device spec against a live input of `channels` channels: the chain to run, or every
/// refusal (all of them, not only the first, so the app can mark each node).
pub fn check_live(text: &str, channels: usize) -> Result<LiveChain, Vec<Refusal>> {
    let (spec, mut tap_decls, settings) = parse_live_spec(text).map_err(|e| vec![Refusal::whole(e)])?;
    let mut refusals = Vec::new();

    // The chain, from its read
    let reads: Vec<&String> = spec.nodes.iter().filter(|(_, n)| matches!(n, Node::Read { .. })).map(|(id, _)| id).collect();
    let [read] = reads.as_slice() else {
        return Err(vec![Refusal::whole(format!("a live chain has one read node; this spec has {}", reads.len()))]);
    };
    let mut order = vec![(*read).clone()];
    loop {
        let last = order.last().cloned().unwrap_or_default();
        let next: Vec<&String> = spec.edges.iter().filter(|(from, _)| *from == last).map(|(_, to)| to).collect();
        match next.as_slice() {
            [] => break,
            [one] => {
                if order.contains(one) {
                    return Err(vec![Refusal::whole(format!("the edges loop back to {one}"))]);
                }
                order.push((*one).clone());
            }
            _ => {
                refusals.push(Refusal { node: last.clone(), op: op_name(&spec.nodes[&last]), reason: "branches are planned; a live chain is one line of nodes for now".into() });
                break;
            }
        }
    }
    for id in spec.nodes.keys() {
        if !order.contains(id) {
            refusals.push(Refusal { node: id.clone(), op: op_name(&spec.nodes[id]), reason: "not connected to the chain from the read".into() });
        }
    }

    // The read: the device's source and channels
    let Node::Read { path, inputs, source, channels: read_channels, start, stop } = &spec.nodes[*read] else { unreachable!() };
    if path.is_some() || inputs.is_some() || start.is_some() || stop.is_some() {
        refusals.push(Refusal { node: (*read).clone(), op: "read".into(), reason: "a live chain reads the device's source: no path, inputs, start or stop".into() });
    }
    let source_name = settings.source.clone().or_else(|| source.clone());
    let mut input_channels: Vec<usize> = read_channels.clone().unwrap_or_else(|| (0..channels).collect());
    if let Some(&bad) = input_channels.iter().find(|&&c| c >= channels) {
        refusals.push(Refusal { node: (*read).clone(), op: "read".into(), reason: format!("channel {bad} does not exist: the source has {channels}") });
        input_channels.retain(|&c| c < channels);
    }

    let mut steps = Vec::new();
    let mut crossings = None;
    let mut taps = Vec::new();
    let mut leading = true;
    let mut current = input_channels.len();
    let mut last_processing: Option<String> = None;
    let mut delay = 0u64;
    for id in order.iter().skip(1) {
        let node = &spec.nodes[id];
        let op = op_name(node);
        let refuse = |reason: String| Refusal { node: id.clone(), op: op.clone(), reason };
        let (t, why) = timing(node);
        // Ops with a bounded history or look-ahead run with the samples carried between blocks
        // (their delay is reported); the ones that need the whole recording, and resampling, do not
        if t == Timing::Offline || matches!(node, Node::Resample { .. }) {
            refusals.push(refuse(why.to_string()));
            continue;
        }
        if crossings.is_some() && !matches!(node, Node::Write { .. }) {
            refusals.push(refuse("crossings gives events: it ends the chain (only a write may follow)".into()));
            continue;
        }
        match node {
            Node::Write { .. } => {
                // Records the node before it
                if let Some(prev) = &last_processing {
                    match taps.iter_mut().find(|t: &&mut LiveTap| &t.node == prev) {
                        Some(tap) => tap.record = true,
                        None => {
                            let (name, _) = tap_decls.remove(prev).unwrap_or_else(|| (prev.clone(), true));
                            taps.push(LiveTap { name, node: prev.clone(), kind: kind_of(&crossings, prev), record: true, channels: current, delay_samples: delay });
                        }
                    }
                } else {
                    refusals.push(refuse("a write needs a processing node before it: the raw data is always recorded".into()));
                }
                continue;
            }
            Node::Remap { .. } => match node.channel_map(None).ok().flatten().map(|m| m.resolve(current)) {
                Some(Ok(order)) if leading => {
                    input_channels = order.into_iter().map(|i| input_channels[i]).collect();
                    current = input_channels.len();
                    steps.push(LiveStep { node: id.clone(), op: op.clone(), stage: None });
                }
                Some(Ok(order)) => {
                    let mut seen = order.clone();
                    seen.sort_unstable();
                    if seen != (0..current).collect::<Vec<_>>() {
                        refusals.push(refuse(format!("a remap after processing keeps every channel ({current}); drop channels in a remap right after the read")));
                    } else {
                        steps.push(LiveStep { node: id.clone(), op: op.clone(), stage: Some(PipelineStage::ReorderChannels { order }) });
                    }
                }
                Some(Err(e)) => refusals.push(refuse(e.to_string())),
                None => refusals.push(refuse("a remap on a device gives its pairs in the spec (files are not read)".into())),
            },
            Node::Crossings { level, release, direction, min_duration_ms, refractory_ms } => {
                if let Err(e) = node.stage() {
                    refusals.push(refuse(e.to_string()));
                } else {
                    crossings = Some((id.clone(), CrossingParams { level: *level, release: *release, direction: *direction, min_duration_ms: *min_duration_ms, refractory_ms: *refractory_ms }));
                }
                leading = false;
            }
            _ => {
                leading = false;
                match node.stage() {
                    Ok(Some(stage)) => {
                        if let PipelineStage::SurfaceLaplacian(lap) = &stage
                            && lap.num_channels != current
                        {
                            refusals.push(refuse(format!("the Laplacian's grid has {} electrodes; the chain has {current} channels here", lap.num_channels)));
                            continue;
                        }
                        if let Some((_, after)) = stencil_halo(&stage) {
                            delay += after as u64;
                        }
                        steps.push(LiveStep { node: id.clone(), op: op.clone(), stage: Some(stage) });
                    }
                    Ok(None) => steps.push(LiveStep { node: id.clone(), op: op.clone(), stage: None }),
                    Err(e) => refusals.push(refuse(e.to_string())),
                }
            }
        }
        last_processing = Some(id.clone());
        if let Some((name, record)) = tap_decls.remove(id) {
            taps.push(LiveTap { name, node: id.clone(), kind: kind_of(&crossings, id), record, channels: current, delay_samples: delay });
        }
    }
    // A tap on the read itself is the source's raw data, which every device publishes
    if let Some((name, _)) = tap_decls.remove(*read) {
        refusals.push(Refusal { node: (*read).clone(), op: "read".into(), reason: format!("tap {name}: the raw data is published already, under the source's name") });
    }
    for (node, (name, _)) in tap_decls {
        refusals.push(Refusal { node, op: String::new(), reason: format!("tap {name}: the node is not part of the chain") });
    }
    let mut names: Vec<&str> = taps.iter().map(|t| t.name.as_str()).collect();
    names.sort_unstable();
    if let Some(w) = names.windows(2).find(|w| w[0] == w[1]) {
        refusals.push(Refusal::whole(format!("two taps are named {}", w[0])));
    }
    let block_ms = settings.block_ms.unwrap_or(DEFAULT_BLOCK_MS);
    if !(block_ms > 0.0 && block_ms <= 10_000.0) {
        refusals.push(Refusal::whole(format!("live.block_ms {block_ms}: between 0 and 10 000 ms")));
    }
    if !refusals.is_empty() {
        return Err(refusals);
    }
    let spec_json = canonical_json(&spec, &taps, &settings);
    Ok(LiveChain { spec, spec_json, source: source_name, block_ms, input_channels, steps, crossings, taps, delay_samples: delay })
}

fn kind_of(crossings: &Option<(String, CrossingParams)>, node: &str) -> LiveTapKind {
    if crossings.as_ref().is_some_and(|(id, _)| id == node) { LiveTapKind::Events } else { LiveTapKind::Samples }
}

/// The spec with its live additions put back, as JSON.
fn canonical_json(spec: &PipelineSpec, taps: &[LiveTap], settings: &LiveSettings) -> String {
    let mut value = serde_json::to_value(spec).unwrap_or(Value::Null);
    if let Some(nodes) = value.get_mut("nodes").and_then(Value::as_object_mut) {
        for tap in taps {
            if let Some(node) = nodes.get_mut(&tap.node).and_then(Value::as_object_mut) {
                let decl = if tap.record { serde_json::json!({"name": tap.name, "record": true}) } else { Value::String(tap.name.clone()) };
                node.insert("tap".into(), decl);
            }
        }
    }
    if let Some(obj) = value.as_object_mut()
        && *settings != LiveSettings::default()
    {
        obj.insert("live".into(), serde_json::to_value(settings).unwrap_or(Value::Null));
    }
    serde_json::to_string(&value).unwrap_or_default()
}

/// One tap's output for one block.
#[derive(Debug, Clone, PartialEq)]
pub enum TapOutput {
    /// Channel-major `channels × samples` values.
    Samples { tap: usize, first_sample: u64, samples: usize, channels: usize, values: Vec<f32> },
    /// The events that ended in this block (sample indices of the run).
    Events { tap: usize, events: Vec<CrossingEvent> },
}

/// A stage that reads samples around each one (FIR, median, Teager–Kaiser, Gaussian smoothing):
/// how many before and after. `None` for the others (pointwise, spatial, forward filters, which
/// carry their own state).
fn stencil_halo(stage: &PipelineStage) -> Option<(usize, usize)> {
    match stage {
        PipelineStage::Fir { .. } | PipelineStage::Median { .. } | PipelineStage::TeagerKaiser { .. } | PipelineStage::GaussianSmooth { .. } => {
            stage.settling(1.0).ok().filter(|&(b, a)| b + a > 0)
        }
        _ => None,
    }
}

enum SegmentKind {
    /// Stages that carry their own state or need none, on the device as one stateful workspace.
    Stateful(Option<PipelineWorkspace<f32>>),
    /// One stage that reads `before` samples before and `after` after each one: run on the carried
    /// input (`history`, channel-major `channels × (before + after)`) followed by the block; its
    /// output is `after` samples late.
    Stencil { workspace: PipelineWorkspace<f32>, before: usize, after: usize, history: Vec<f32> },
}

/// Stages up to a tap.
struct Segment {
    kind: SegmentKind,
    /// The sample tap at its end (index into the chain's taps).
    tap: Option<usize>,
    /// The node of its last stage (a tap on that node attaches here).
    node: String,
    /// Samples its output lags the chain's input, counting the stencils before it.
    delay: u64,
}

/// Runs a [`LiveChain`] block after block: the filters carry their state, stencils carry the
/// samples they need from earlier blocks (their output comes later by their look-ahead, which
/// each tap's frames account for), each tap's output is read back once per block, the rest stays
/// on the device.
pub struct LiveRunner {
    chain: LiveChain,
    client: cubecl::prelude::Client,
    segments: Vec<Segment>,
    tracker: Option<(usize, CrossingTracker, CrossingOptions)>,
    /// Samples the crossings' input lags the chain's input.
    tracker_delay: u64,
    /// Source channels' gain and offset (stored → unit).
    gains: Vec<f32>,
    offsets: Vec<f32>,
    format: dspk_core::SampleFormat,
    source_channels: usize,
    block: usize,
}

impl LiveRunner {
    /// Prepares `chain` for a source of `gains.len()` channels stored as `format` at
    /// `sample_rate`, on `target`.
    pub fn new(chain: LiveChain, target: ComputeTarget, sample_rate: f64, format: dspk_core::SampleFormat, gains: Vec<f32>, offsets: Vec<f32>) -> DspResult<Self> {
        let client = target.client().map_err(|e| DspError::ComputeError(e.to_string()))?;
        let channels = chain.input_channels.len();
        let block = ((chain.block_ms / 1000.0 * sample_rate).round() as usize).max(1);
        let config = |e: dspk_ops::OpsError| DspError::InvalidConfig(e.to_string());
        let mut segments: Vec<Segment> = Vec::new();
        let mut stages: Vec<PipelineStage> = Vec::new();
        let mut delay = 0u64;
        let mut last_node = String::new();
        let close = |stages: &mut Vec<PipelineStage>, tap: Option<usize>, node: &str, delay: u64, segments: &mut Vec<Segment>| -> DspResult<()> {
            let workspace = if stages.is_empty() {
                None
            } else {
                let pipeline = Pipeline::with_stages(std::mem::take(stages));
                Some(PipelineWorkspace::new_stateful(client.clone(), pipeline, channels, block, sample_rate).map_err(config)?)
            };
            segments.push(Segment { kind: SegmentKind::Stateful(workspace), tap, node: node.to_string(), delay });
            Ok(())
        };
        for step in &chain.steps {
            if let Some(stage) = &step.stage {
                if let Some((before, after)) = stencil_halo(stage) {
                    if !stages.is_empty() {
                        close(&mut stages, None, &last_node, delay, &mut segments)?;
                    }
                    let pipeline = Pipeline::with_stages(vec![stage.clone()]);
                    let workspace = PipelineWorkspace::new(client.clone(), pipeline, channels, block + before + after, sample_rate).map_err(config)?;
                    delay += after as u64;
                    let history = vec![0.0; channels * (before + after)];
                    segments.push(Segment { kind: SegmentKind::Stencil { workspace, before, after, history }, tap: None, node: step.node.clone(), delay });
                } else {
                    stages.push(stage.clone());
                }
            }
            last_node = step.node.clone();
            if let Some(tap) = chain.taps.iter().position(|t| t.node == step.node && t.kind == LiveTapKind::Samples) {
                match segments.last_mut() {
                    // The stencil just made is this node: its output is the tap
                    Some(seg) if stages.is_empty() && seg.node == step.node && seg.tap.is_none() => seg.tap = Some(tap),
                    _ => close(&mut stages, Some(tap), &step.node, delay, &mut segments)?,
                }
            }
        }
        if !stages.is_empty() {
            close(&mut stages, None, &last_node, delay, &mut segments)?;
        }
        let tracker = match &chain.crossings {
            Some((node, params)) => {
                let tap = chain.taps.iter().position(|t| &t.node == node).unwrap_or(usize::MAX);
                let options = params.options(sample_rate);
                Some((tap, CrossingTracker::new(channels, &options).map_err(config)?, options))
            }
            None => None,
        };
        let source_channels = gains.len();
        Ok(Self { chain, client, segments, tracker, tracker_delay: delay, gains, offsets, format, source_channels, block })
    }

    pub fn chain(&self) -> &LiveChain {
        &self.chain
    }

    /// Samples the chain's last output lags its input (the stencils' look-ahead).
    pub fn delay(&self) -> u64 {
        self.tracker_delay
    }

    /// Runs one block of zeros through the chain and forgets it: kernels are compiled and tuned,
    /// buffers allocated, before the first real block (which would otherwise wait for them).
    pub fn warm(&mut self) -> DspResult<()> {
        let zeros = vec![0u8; self.source_channels * self.block * self.format.bytes()];
        self.process(&zeros, self.block, 0)?;
        self.reset()
    }

    /// Forgets every carried state: the next block starts as the first one.
    pub fn reset(&mut self) -> DspResult<()> {
        for seg in &mut self.segments {
            match &mut seg.kind {
                SegmentKind::Stateful(Some(ws)) => ws.reset(),
                SegmentKind::Stateful(None) => {}
                SegmentKind::Stencil { history, .. } => history.iter_mut().for_each(|v| *v = 0.0),
            }
        }
        if let Some((_, tracker, options)) = self.tracker.as_mut() {
            *tracker = CrossingTracker::new(self.chain.input_channels.len(), options).map_err(|e| DspError::InvalidConfig(e.to_string()))?;
        }
        Ok(())
    }

    /// Runs one block of the source (stored values, channel-major, every source channel) whose
    /// first sample is `first_sample`, and returns each tap's output. A tap after stencils starts
    /// its frames that many samples late (samples before the run's start are left out).
    pub fn process(&mut self, stored: &[u8], samples: usize, first_sample: u64) -> DspResult<Vec<TapOutput>> {
        let bytes = self.format.bytes();
        if stored.len() != self.source_channels * samples * bytes {
            return Err(DspError::ShapeMismatch { expected: vec![self.source_channels * samples * bytes], actual: vec![stored.len()] });
        }
        // The chain's channels, scaled, on the host (small blocks: one upload of f32)
        let channels = &self.chain.input_channels;
        let mut values = vec![0.0f32; channels.len() * samples];
        for (row, &c) in channels.iter().enumerate() {
            let src = &stored[c * samples * bytes..(c + 1) * samples * bytes];
            self.format.decode(src, &mut values[row * samples..(row + 1) * samples], self.gains[c], self.offsets[c]);
        }
        let n = channels.len();
        let mut outputs = Vec::new();
        let mut host = Some(values);
        let mut handle = None;
        let client = self.client.clone();
        let to_host = |handle: &mut Option<cubecl::server::Handle>, host: &mut Option<Vec<f32>>| -> Vec<f32> {
            match (handle.take(), host.take()) {
                (Some(h), _) => buffer::download_prefix::<f32>(&client, h, n * samples),
                (None, Some(v)) => v,
                (None, None) => unreachable!("a block always has an input"),
            }
        };
        for seg in &mut self.segments {
            match &mut seg.kind {
                SegmentKind::Stateful(Some(ws)) => {
                    let out = match (handle.take(), host.take()) {
                        (Some(h), _) => ws.process_handle(&h, samples),
                        (None, Some(v)) => ws.process_owned_chunk_in_vram(v, samples),
                        (None, None) => unreachable!("a block always has an input"),
                    };
                    handle = Some(out);
                }
                SegmentKind::Stateful(None) => {}
                SegmentKind::Stencil { workspace, before, after, history } => {
                    let input = to_host(&mut handle, &mut host);
                    let h = *before + *after;
                    let len = h + samples;
                    let mut buf = vec![0.0f32; n * len];
                    for c in 0..n {
                        buf[c * len..c * len + h].copy_from_slice(&history[c * h..(c + 1) * h]);
                        buf[c * len + h..(c + 1) * len].copy_from_slice(&input[c * samples..(c + 1) * samples]);
                    }
                    let mut out = vec![0.0f32; n * len];
                    workspace.process_chunk(&buf, len, &mut out);
                    // Output for the input `after` samples earlier: columns `before .. before + samples`
                    let mut result = vec![0.0f32; n * samples];
                    for c in 0..n {
                        result[c * samples..(c + 1) * samples].copy_from_slice(&out[c * len + *before..c * len + *before + samples]);
                        // The last `h` input samples are the next block's history
                        history[c * h..(c + 1) * h].copy_from_slice(&buf[(c + 1) * len - h..(c + 1) * len]);
                    }
                    host = Some(result);
                }
            }
            if let Some(tap) = seg.tap {
                let values = match (&handle, &host) {
                    (Some(h), _) => buffer::download_prefix::<f32>(&self.client, h.clone(), n * samples),
                    (None, Some(v)) => v.clone(),
                    (None, None) => unreachable!(),
                };
                if let Some((first, samples, values)) = trim_delayed(values, n, samples, first_sample, seg.delay) {
                    outputs.push(TapOutput::Samples { tap, first_sample: first, samples, channels: n, values });
                }
            }
        }
        if let Some((tap, tracker, _)) = self.tracker.as_mut() {
            let values = to_host(&mut handle, &mut host);
            if let Some((first, kept, values)) = trim_delayed(values, n, samples, first_sample, self.tracker_delay) {
                tracker.push_host_window(&values, kept, 0..kept, first);
            }
            let events = tracker.take_events();
            if *tap != usize::MAX {
                outputs.push(TapOutput::Events { tap: *tap, events });
            }
        }
        Ok(outputs)
    }
}

/// A block's output `delay` samples late: its first sample, and the samples from the run's start
/// on (`None` when all of it is before the start).
fn trim_delayed(values: Vec<f32>, channels: usize, samples: usize, first_sample: u64, delay: u64) -> Option<(u64, usize, Vec<f32>)> {
    let cut = delay.saturating_sub(first_sample).min(samples as u64) as usize;
    let kept = samples - cut;
    if kept == 0 {
        return None;
    }
    let first = (first_sample + cut as u64) - delay;
    if cut == 0 {
        return Some((first, kept, values));
    }
    let mut out = Vec::with_capacity(channels * kept);
    for c in 0..channels {
        out.extend_from_slice(&values[c * samples + cut..(c + 1) * samples]);
    }
    Some((first, kept, out))
}

/// A chain from a viewer's ops (pipeline nodes as JSON, in order, as a view's processing carries
/// them) over `channels` channels: checked as a device's chain is (live ops only), its output the
/// tap `view`.
pub fn chain_of_ops(ops: &[String], channels: usize, block_ms: f64) -> Result<LiveChain, Vec<Refusal>> {
    let mut nodes = serde_json::Map::new();
    nodes.insert("in".into(), serde_json::json!({"op": "read"}));
    let mut edges = Vec::new();
    let mut last = "in".to_string();
    for (i, op) in ops.iter().enumerate() {
        let mut node: Value = serde_json::from_str(op).map_err(|e| vec![Refusal::whole(format!("op {i}: not a pipeline node: {e}"))])?;
        let id = format!("op{i}");
        if i + 1 == ops.len()
            && let Some(obj) = node.as_object_mut()
        {
            obj.insert("tap".into(), Value::String("view".into()));
        }
        nodes.insert(id.clone(), node);
        edges.push(serde_json::json!([last, id]));
        last = id;
    }
    let spec = serde_json::json!({"dspk_pipeline": crate::SPEC_VERSION, "nodes": nodes, "edges": edges, "live": {"block_ms": block_ms}});
    check_live(&spec.to_string(), channels)
}

/// `spec_json` with one parameter of one node set to `value_json` (a parameter changed while
/// running). The op itself and its tap cannot be changed this way.
pub fn with_parameter(spec_json: &str, node: &str, parameter: &str, value_json: &str) -> Result<String, String> {
    if matches!(parameter, "op" | "tap") {
        return Err(format!("{node}.{parameter}: the op and its tap are the chain's shape: configure a new chain"));
    }
    let mut spec: Value = serde_json::from_str(spec_json).map_err(|e| e.to_string())?;
    let value: Value = serde_json::from_str(value_json).map_err(|e| format!("{node}.{parameter}: the value is not JSON: {e}"))?;
    let target = spec.get_mut("nodes").and_then(|n| n.get_mut(node)).and_then(Value::as_object_mut).ok_or_else(|| format!("no node {node}"))?;
    if !target.contains_key(parameter) && target.get("op").and_then(Value::as_str) == Some("read") {
        return Err(format!("{node}: the read's settings are the source's"));
    }
    target.insert(parameter.into(), value);
    Ok(spec.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(nodes: &str, edges: &str) -> String {
        format!(r#"{{"dspk_pipeline": 1, "nodes": {{{nodes}}}, "edges": [{edges}], "live": {{"source": "gen", "block_ms": 5}}}}"#)
    }

    #[test]
    fn a_causal_chain_with_taps() {
        let text = spec(
            r#""in": {"op": "read"}, "hp": {"op": "iir", "band": "highpass", "cutoff": [300], "mode": "forward", "tap": "filtered"},
               "ref": {"op": "car"}, "det": {"op": "crossings", "level": -50, "direction": "falling", "tap": "events"},
               "out": {"op": "write", "path": "events.csv", "format": "events"}"#,
            r#"["in", "hp"], ["hp", "ref"], ["ref", "det"], ["det", "out"]"#,
        );
        let chain = check_live(&text, 4).unwrap();
        assert_eq!(chain.source.as_deref(), Some("gen"));
        assert_eq!(chain.block_ms, 5.0);
        assert_eq!(chain.taps.len(), 2);
        assert_eq!((chain.taps[0].name.as_str(), chain.taps[0].kind), ("filtered", LiveTapKind::Samples));
        assert_eq!((chain.taps[1].kind, chain.taps[1].record), (LiveTapKind::Events, true));
        // The canonical JSON parses back to the same chain
        let again = check_live(&chain.spec_json, 4).unwrap();
        assert_eq!(again.taps, chain.taps);
    }

    #[test]
    fn refuses_every_node_that_needs_the_future_and_says_why() {
        let text = spec(
            r#""in": {"op": "read"}, "bp": {"op": "iir", "band": "bandpass", "cutoff": [300, 3000]},
               "med": {"op": "median", "width": 5}, "w": {"op": "whiten"}"#,
            r#"["in", "bp"], ["bp", "med"], ["med", "w"]"#,
        );
        let refusals = check_live(&text, 4).unwrap_err();
        let nodes: Vec<&str> = refusals.iter().map(|r| r.node.as_str()).collect();
        // The median runs live (carried history, a delay); the zero-phase filter and the fit do not
        assert_eq!(nodes, vec!["bp", "w"]);
        assert!(refusals[0].reason.contains("mode = \"forward\""));
    }

    #[test]
    fn refuses_branches_paths_and_unknown_taps() {
        let text = spec(
            r#""in": {"op": "read", "path": "x.bin"}, "a": {"op": "car"}, "b": {"op": "cmr", "tap": "b"}"#,
            r#"["in", "a"], ["in", "b"]"#,
        );
        let refusals = check_live(&text, 4).unwrap_err();
        assert!(refusals.iter().any(|r| r.reason.contains("branches")));
        assert!(refusals.iter().any(|r| r.node == "in" && r.reason.contains("no path")));
    }

    #[test]
    fn leading_remap_selects_channels() {
        let text = spec(
            r#""in": {"op": "read"}, "m": {"op": "remap", "pairs": [{"from": 3, "to": 0}, {"from": 1, "to": 1}]}, "s": {"op": "scale", "alpha": 2, "tap": "x2"}"#,
            r#"["in", "m"], ["m", "s"]"#,
        );
        let chain = check_live(&text, 4).unwrap();
        assert_eq!(chain.input_channels, vec![3, 1]);
        assert_eq!(chain.taps[0].channels, 2);
    }

    /// The chain run block by block gives what it gives on the whole signal at once (forward filters
    /// carry their state), on every runtime compiled in.
    #[test]
    fn blocks_equal_the_whole() {
        let text = spec(
            r#""in": {"op": "read"}, "hp": {"op": "iir", "band": "highpass", "cutoff": [20], "mode": "forward", "tap": "hp"}, "sq": {"op": "square", "tap": "power"}"#,
            r#"["in", "hp"], ["hp", "sq"]"#,
        );
        let (channels, total, rate) = (3usize, 600usize, 1000.0);
        let signal: Vec<f32> = (0..channels).flat_map(|c| (0..total).map(move |s| ((s as f32 * 0.05 + c as f32).sin() * 100.0).round())).collect();
        // int16 stored, gain 0.5
        let stored: Vec<u8> = signal.iter().flat_map(|&v| ((v / 0.5) as i16).to_le_bytes()).collect();
        let row = |data: &[u8], c: usize, a: usize, b: usize| data[(c * total + a) * 2..(c * total + b) * 2].to_vec();
        for target in ComputeTarget::for_tests() {
            let chain = check_live(&text, channels).unwrap();
            let run = |block: usize| {
                let mut runner = LiveRunner::new(chain.clone(), target, rate, dspk_core::SampleFormat::I16, vec![0.5; channels], vec![0.0; channels]).unwrap();
                let mut power = vec![0.0f32; channels * total];
                let mut s0 = 0;
                while s0 < total {
                    let n = block.min(total - s0);
                    let chunk: Vec<u8> = (0..channels).flat_map(|c| row(&stored, c, s0, s0 + n)).collect();
                    for out in runner.process(&chunk, n, s0 as u64).unwrap() {
                        if let TapOutput::Samples { tap: 1, values, .. } = out {
                            for c in 0..channels {
                                power[c * total + s0..c * total + s0 + n].copy_from_slice(&values[c * n..(c + 1) * n]);
                            }
                        }
                    }
                    s0 += n;
                }
                power
            };
            let whole = run(total);
            let blocks = run(37);
            for (a, b) in whole.iter().zip(&blocks) {
                assert!((a - b).abs() <= 1e-3 * a.abs().max(1.0), "{target:?}: {a} vs {b}");
            }
        }
    }
}

#[cfg(test)]
mod stencil_tests {
    use super::*;

    fn spec(nodes: &str, edges: &str) -> String {
        format!(r#"{{"dspk_pipeline": 1, "nodes": {{{nodes}}}, "edges": [{edges}], "live": {{"block_ms": 5}}}}"#)
    }

    /// Runs `chain` over `signal` (`channels × total`, f32 stored, gain 1) in blocks of `block`,
    /// and returns the tap's output and the sample its first value belongs to.
    fn run(chain: &LiveChain, target: ComputeTarget, signal: &[f32], channels: usize, total: usize, block: usize) -> (u64, Vec<Vec<f32>>) {
        let mut runner = LiveRunner::new(chain.clone(), target, 1000.0, dspk_core::SampleFormat::F32, vec![1.0; channels], vec![0.0; channels]).unwrap();
        let mut rows = vec![Vec::new(); channels];
        let mut first = None;
        let mut s0 = 0;
        while s0 < total {
            let n = block.min(total - s0);
            let chunk: Vec<u8> = (0..channels).flat_map(|c| signal[c * total + s0..c * total + s0 + n].iter().flat_map(|v| v.to_le_bytes())).collect();
            for out in runner.process(&chunk, n, s0 as u64).unwrap() {
                if let TapOutput::Samples { first_sample, samples, values, .. } = out {
                    first.get_or_insert(first_sample);
                    for c in 0..channels {
                        rows[c].extend_from_slice(&values[c * samples..(c + 1) * samples]);
                    }
                }
            }
            s0 += n;
        }
        (first.unwrap_or(0), rows)
    }

    /// FIR and median, with history carried and output delayed, give the same values in blocks of
    /// any size as in one block, on every runtime.
    #[test]
    fn history_and_delay_ops_are_exact_in_blocks() {
        let text = spec(
            r#""in": {"op": "read"}, "fir": {"op": "fir", "taps": [0.25, 0.5, 0.25]}, "med": {"op": "median", "width": 5, "tap": "out"}"#,
            r#"["in", "fir"], ["fir", "med"]"#,
        );
        let chain = check_live(&text, 2).unwrap();
        assert_eq!(chain.delay_samples, 2, "the median looks 2 ahead; the causal FIR none");
        let (channels, total) = (2usize, 400usize);
        let signal: Vec<f32> = (0..channels * total).map(|i| ((i as f32 * 0.37).sin() * 50.0).round()).collect();
        for target in ComputeTarget::for_tests() {
            let (first_whole, whole) = run(&chain, target, &signal, channels, total, total);
            let (first_blocks, blocks) = run(&chain, target, &signal, channels, total, 7);
            assert_eq!((first_whole, first_blocks), (0, 0));
            assert_eq!(whole, blocks, "{target:?}");
            assert_eq!(whole[0].len(), total - 2, "the last 2 samples wait for the next block");
        }
    }

    /// A delayed op's frames are labelled with the samples they belong to: a centred identity FIR
    /// gives back the input, sample for sample.
    #[test]
    fn delayed_frames_line_up_with_the_input() {
        let text = spec(r#""in": {"op": "read"}, "id": {"op": "fir", "taps": [0, 1, 0], "centered": true, "tap": "x"}"#, r#"["in", "id"]"#);
        let chain = check_live(&text, 1).unwrap();
        let signal: Vec<f32> = (0..50).map(|i| i as f32).collect();
        for target in ComputeTarget::for_tests() {
            let (first, rows) = run(&chain, target, &signal, 1, 50, 6);
            assert_eq!(first, 0);
            assert_eq!(rows[0], signal[..49].to_vec(), "{target:?}");
        }
    }

    #[test]
    fn viewer_ops_and_parameters() {
        let ops = vec![r#"{"op": "iir", "band": "highpass", "cutoff": [5], "mode": "forward"}"#.to_string(), r#"{"op": "rectify"}"#.to_string()];
        let chain = chain_of_ops(&ops, 3, 10.0).unwrap();
        assert_eq!(chain.taps.len(), 1);
        assert_eq!(chain.taps[0].name, "view");
        let zero_phase = vec![r#"{"op": "iir", "band": "highpass", "cutoff": [5]}"#.to_string()];
        assert!(chain_of_ops(&zero_phase, 3, 10.0).is_err(), "only live ops on a live view");

        let changed = with_parameter(&chain.spec_json, "op1", "nonexistent", "1").unwrap();
        assert!(check_live(&changed, 3).is_err(), "unknown fields are refused by the spec");
        let changed = with_parameter(&chain.spec_json, "op0", "cutoff", "[20]").unwrap();
        assert!(check_live(&changed, 3).is_ok());
        assert!(with_parameter(&chain.spec_json, "op0", "op", "\"car\"").is_err());
    }
}
