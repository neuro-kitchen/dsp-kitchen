//! One planner for a chain of pipeline nodes, whatever runs it: a view's preview, a batch run
//! ([`crate::run`]) or a live device ([`crate::live`]). The nodes become device stages by the same
//! rules everywhere, so a preview shows what a run writes and a device refuses what a run would
//! not do live; each use differs only where it must ([`Purpose`]).
//!
//! The rules ([`plan_chain`]):
//! - **Remaps by position.** Right after the read they fold into which channels are read (they may
//!   drop and reorder); between processing steps they reorder on the device and keep every channel;
//!   after the last processing step they choose the channels kept (they may drop), except live,
//!   where every remap after the read keeps every channel.
//! - **Whitening** is fitted on the signal of the steps before it (views, batch runs); a live
//!   chain refuses it (it needs the recording).
//! - **Resample and crossings** end the processing (only remaps may follow); crossings after
//!   resample is refused (detect at the signal's own rate). A view previews neither (it shows the
//!   signal before them); a live chain refuses resampling.
//! - **Live timing** ([`timing`]): ops that need the whole recording are refused; ops with a
//!   look-ahead run with carried samples and delay the output (summed in [`ChainPlan::delay`]).
//! - **A Laplacian's grid** must have as many electrodes as the chain has channels where it runs.
//!
//! Refusals name the node and why ([`Refusal`]); every refusal of a chain is reported at once.

use std::path::Path;

use dspk_core::ChannelMap;
use dspk_ops::crossings::CrossingOptions;
use dspk_ops::PipelineStage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Mode, Node, WhitenFit};

/// What a chain is planned for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    /// A view's preview: windows of a recording, drawn.
    View,
    /// A batch run: a recording processed and written.
    Batch,
    /// A live device: blocks as they arrive.
    Live,
}

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

/// Why one node of a chain was refused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Refusal {
    /// Node id ("" for the chain as a whole).
    pub node: String,
    pub op: String,
    pub reason: String,
}

impl Refusal {
    /// A refusal of the chain as a whole (the live checker's, for a spec it cannot read).
    #[cfg(all(feature = "live", not(target_family = "wasm")))]
    pub(crate) fn whole(reason: impl Into<String>) -> Self {
        Self { node: String::new(), op: String::new(), reason: reason.into() }
    }
}

/// Refusals as one line each, `node "id": reason` (a batch run's or a view's error).
pub fn refusals_text(refusals: &[Refusal]) -> String {
    refusals.iter().map(|r| if r.node.is_empty() { r.reason.clone() } else { format!("node {:?}: {}", r.node, r.reason) }).collect::<Vec<_>>().join("; ")
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

/// Samples before and after each one a stencil stage reads (FIR, median, Teager–Kaiser, Gaussian
/// smoothing), when it reads any.
pub(crate) fn stencil_halo(stage: &PipelineStage) -> Option<(usize, usize)> {
    match stage {
        PipelineStage::Fir { .. } | PipelineStage::Median { .. } | PipelineStage::TeagerKaiser { .. } | PipelineStage::GaussianSmooth { .. } => {
            stage.settling(1.0).ok().filter(|&(b, a)| b + a > 0)
        }
        _ => None,
    }
}

/// What one processing step does on the device.
#[derive(Debug, Clone)]
pub enum Step {
    /// A stage as the node gives it.
    Stage(PipelineStage),
    /// A remap between processing steps: `order` over the channels at that point (every one kept).
    Reorder { map: ChannelMap, order: Vec<usize> },
    /// A whitening, fitted on the signal of the steps before it.
    Whiten(WhitenFit),
}

/// One processing step of a planned chain.
#[derive(Debug, Clone)]
pub struct PlannedStep {
    /// The node's id in the chain.
    pub id: String,
    pub op: String,
    /// The node as JSON: identifies the step (two chains with the same keys share the prefix).
    pub key: String,
    pub step: Step,
    /// Channels after it.
    pub channels: usize,
    /// Samples the chain's output lags its input after it (live; 0 otherwise).
    pub delay: u64,
}

/// A chain as every use runs it.
#[derive(Debug, Clone, Default)]
pub struct ChainPlan {
    /// The input channels read, in order, through the remaps that open the chain.
    pub read: Vec<usize>,
    /// Those remaps, by node id.
    pub lead: Vec<(String, ChannelMap)>,
    pub steps: Vec<PlannedStep>,
    /// The resample node ending the processing (a batch run), by id.
    pub resample: Option<(String, Node)>,
    /// The crossings node ending the processing (a batch run, a live chain), by id.
    pub crossings: Option<(String, CrossingParams)>,
    /// The remaps after the last processing step (they may drop), by node id.
    pub tail: Vec<(String, ChannelMap)>,
    /// The rows of the last processing step kept, in order, through `tail` (`None`: all).
    pub keep: Option<Vec<usize>>,
    /// Samples the output lags the input (live).
    pub delay: u64,
}

impl ChainPlan {
    /// The device stages in order, a whitening as `whitening(fit, channels)` gives it (unfitted
    /// until it is fitted), with their keys.
    pub fn stages(&self, mut whitening: impl FnMut(&WhitenFit, usize) -> PipelineStage) -> Vec<(String, PipelineStage)> {
        let mut channels = self.read.len();
        self.steps
            .iter()
            .map(|s| {
                let stage = match &s.step {
                    Step::Stage(stage) => stage.clone(),
                    Step::Reorder { order, .. } => PipelineStage::ReorderChannels { order: order.clone() },
                    Step::Whiten(fit) => whitening(fit, channels),
                };
                channels = s.channels;
                (s.key.clone(), stage)
            })
            .collect()
    }

    /// The input channels each output row stands for: `read` through the reorders and `keep`.
    pub fn shown(&self) -> Vec<usize> {
        let mut current = self.read.clone();
        for s in &self.steps {
            if let Step::Reorder { order, .. } = &s.step {
                current = order.iter().map(|&j| current[j]).collect();
            }
        }
        match &self.keep {
            Some(keep) => keep.iter().map(|&j| current[j]).collect(),
            None => current,
        }
    }
}

/// Plans the chain of `nodes` (ids and nodes after the read, in order; no write nodes) over the
/// input channels `input`, for `purpose`. `base_dir` is where a remap's channel-map file is read
/// (`None`: files are not read, as on a device). The rules are in the module's documentation.
pub fn plan_chain(nodes: &[(String, Node)], input: &[usize], purpose: Purpose, base_dir: Option<&Path>) -> Result<ChainPlan, Vec<Refusal>> {
    let (plan, refusals) = plan_chain_all(nodes, input, purpose, base_dir);
    if refusals.is_empty() { Ok(plan) } else { Err(refusals) }
}

/// [`plan_chain`] with the plan of what was not refused, and the refusals (a live chain names
/// its taps on the nodes that were planned, and reports every refusal at once).
pub(crate) fn plan_chain_all(nodes: &[(String, Node)], input: &[usize], purpose: Purpose, base_dir: Option<&Path>) -> (ChainPlan, Vec<Refusal>) {
    let mut refusals = Vec::new();
    let mut plan = ChainPlan { read: input.to_vec(), ..Default::default() };
    let is_processing = |n: &Node| !matches!(n, Node::Remap { .. });
    let last_processing = nodes.iter().rposition(|(_, n)| is_processing(n));
    // Channels at the current point, and the delay so far
    let mut current = input.len();
    let mut leading = true;
    let mut ended = false;
    for (i, (id, node)) in nodes.iter().enumerate() {
        let op = op_name(node);
        let refuse = |reason: String| Refusal { node: id.clone(), op: op.clone(), reason };
        let key = serde_json::to_string(node).unwrap_or_default();
        if purpose == Purpose::Live {
            let (t, why) = timing(node);
            // Ops with a bounded history or look-ahead run with the samples carried between
            // blocks (their delay is reported); the ones that need the whole recording, and
            // resampling, do not
            if t == Timing::Offline || matches!(node, Node::Resample { .. }) {
                refusals.push(refuse(why.to_string()));
                continue;
            }
        }
        if let Node::Remap { .. } = node {
            // A live chain ends at crossings (its events are not channels to reorder)
            if ended && purpose == Purpose::Live {
                refusals.push(refuse("crossings gives events: it ends the chain (only a write may follow)".into()));
                continue;
            }
            let map = match node.channel_map(base_dir) {
                Ok(Some(map)) => map,
                Ok(None) => {
                    refusals.push(refuse("a remap on a device gives its pairs in the spec (files are not read)".into()));
                    continue;
                }
                Err(e) => {
                    refusals.push(refuse(e.to_string()));
                    continue;
                }
            };
            let order = match map.resolve(current) {
                Ok(order) => order,
                Err(e) => {
                    refusals.push(refuse(e.to_string()));
                    continue;
                }
            };
            let after_last = last_processing.is_none_or(|last| i > last);
            if leading {
                // Folds into the read: may drop and reorder
                plan.read = order.iter().map(|&j| plan.read[j]).collect();
                current = plan.read.len();
                plan.lead.push((id.clone(), map));
            } else if after_last && purpose != Purpose::Live {
                // The channels kept: may drop
                let base = plan.keep.take().unwrap_or_else(|| (0..current).collect());
                plan.keep = Some(order.iter().map(|&j| base[j]).collect());
                plan.tail.push((id.clone(), map));
            } else {
                // Between processing steps (and every remap after the read, live): a reorder
                let mut seen = order.clone();
                seen.sort_unstable();
                if seen != (0..current).collect::<Vec<_>>() {
                    let what = if purpose == Purpose::Live { "a remap after processing" } else { "a remap between processing nodes" };
                    refusals.push(refuse(format!("{what} keeps every channel ({current}; it reorders them): drop channels in a remap right after the read{}", if purpose == Purpose::Live { "" } else { " or after the last processing node" })));
                    continue;
                }
                plan.steps.push(PlannedStep { id: id.clone(), op, key, step: Step::Reorder { map, order }, channels: current, delay: plan.delay });
            }
            continue;
        }
        if ended {
            refusals.push(refuse(match &plan.crossings {
                Some(_) => "crossings gives events: it ends the chain (only remaps, or a write, may follow)".into(),
                None => "resample ends the processing: only remaps may follow".into(),
            }));
            continue;
        }
        leading = false;
        match node {
            Node::Resample { .. } => {
                if let Err(e) = node.stage() {
                    refusals.push(refuse(e.to_string()));
                    continue;
                }
                if purpose == Purpose::Batch && last_processing != Some(i) {
                    refusals.push(refuse("resample must be the last processing node before write (only remaps may follow)".into()));
                    continue;
                }
                // A view shows the signal at its own rate: nothing to preview
                if purpose == Purpose::Batch {
                    plan.resample = Some((id.clone(), node.clone()));
                }
                // Checked last above (a batch run); a view previews nothing of it
            }
            Node::Crossings { level, release, direction, min_duration_ms, refractory_ms } => {
                if let Err(e) = node.stage() {
                    refusals.push(refuse(e.to_string()));
                    continue;
                }
                if plan.resample.is_some() {
                    refusals.push(refuse("crossings after resample: detect on the signal at its own rate".into()));
                    continue;
                }
                if purpose == Purpose::Batch && last_processing != Some(i) {
                    refusals.push(refuse("crossings must be the last processing node before write (only remaps may follow)".into()));
                    continue;
                }
                // A view shows the signal crossings reads: nothing to preview
                if purpose != Purpose::View {
                    plan.crossings = Some((id.clone(), CrossingParams { level: *level, release: *release, direction: *direction, min_duration_ms: *min_duration_ms, refractory_ms: *refractory_ms }));
                    // A live chain ends here (what follows is refused); a batch run checked it
                    // is last above
                    ended = purpose == Purpose::Live;
                }
            }
            _ => {
                if let Some(fit) = node.whiten_fit() {
                    if let Err(e) = node.stage() {
                        refusals.push(refuse(e.to_string()));
                        continue;
                    }
                    plan.steps.push(PlannedStep { id: id.clone(), op, key, step: Step::Whiten(fit), channels: current, delay: plan.delay });
                    continue;
                }
                match node.stage() {
                    Ok(Some(stage)) => {
                        // A grid's operator is made for its electrodes
                        if let PipelineStage::SurfaceLaplacian(lap) = &stage
                            && lap.num_channels != current
                        {
                            refusals.push(refuse(format!("the Laplacian's grid is made for {} channels; the chain has {current} here", lap.num_channels)));
                            continue;
                        }
                        if purpose == Purpose::Live
                            && let Some((_, after)) = stencil_halo(&stage)
                        {
                            plan.delay += after as u64;
                        }
                        plan.steps.push(PlannedStep { id: id.clone(), op, key, step: Step::Stage(stage), channels: current, delay: plan.delay });
                    }
                    Ok(None) => {}
                    Err(e) => refusals.push(refuse(e.to_string())),
                }
            }
        }
    }
    (plan, refusals)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(json: &str) -> Node {
        serde_json::from_str(json).unwrap()
    }

    fn chain(nodes: &[&str]) -> Vec<(String, Node)> {
        nodes.iter().enumerate().map(|(i, j)| (format!("n{i}"), node(j))).collect()
    }

    // Output `to` reads input `from` (0-based)
    const REMAP_DROP: &str = r#"{"op": "remap", "base": 0, "pairs": [{"from": 2, "to": 0}, {"from": 0, "to": 1}]}"#;
    const REMAP_SWAP: &str = r#"{"op": "remap", "base": 0, "pairs": [{"from": 1, "to": 0}, {"from": 0, "to": 1}]}"#;
    const CAR: &str = r#"{"op": "car"}"#;
    const ZERO_PHASE: &str = r#"{"op": "iir", "band": "bandpass", "cutoff": [300.0, 3000.0], "order": 3, "mode": "forward_backward"}"#;
    const TK: &str = r#"{"op": "teager_kaiser"}"#;

    #[test]
    fn remaps_fold_into_the_read_reorder_between_steps_and_keep_at_the_end() {
        let nodes = chain(&[REMAP_DROP, CAR, REMAP_SWAP, CAR]);
        for purpose in [Purpose::View, Purpose::Batch, Purpose::Live] {
            let p = plan_chain(&nodes, &[10, 11, 12, 13], purpose, None).unwrap();
            assert_eq!(p.read, vec![12, 10], "{purpose:?}: the leading remap folds into the read");
            assert!(matches!(p.steps[1].step, Step::Reorder { ref order, .. } if order == &vec![1, 0]));
            assert_eq!(p.shown(), vec![10, 12]);
        }
        // After the last processing step a remap may drop (views and runs), not live
        let nodes = chain(&[CAR, r#"{"op": "remap", "base": 0, "pairs": [{"from": 1, "to": 0}]}"#]);
        let p = plan_chain(&nodes, &[5, 6], Purpose::Batch, None).unwrap();
        assert_eq!((p.keep.clone(), p.shown()), (Some(vec![1]), vec![6]));
        assert!(plan_chain(&nodes, &[5, 6], Purpose::Live, None).is_err());
    }

    #[test]
    fn each_purpose_refuses_what_it_cannot_run_and_says_why() {
        let nodes = chain(&[CAR, ZERO_PHASE, TK]);
        assert!(plan_chain(&nodes, &[0, 1], Purpose::View, None).is_ok());
        assert!(plan_chain(&nodes, &[0, 1], Purpose::Batch, None).is_ok());
        let refused = plan_chain(&nodes, &[0, 1], Purpose::Live, None).unwrap_err();
        assert_eq!(refused.len(), 1);
        assert_eq!(refused[0].node, "n1");
        assert!(refused[0].reason.contains("zero-phase"), "{refused:?}");
        // Live: Teager–Kaiser runs one sample late
        let live = plan_chain(&chain(&[CAR, TK]), &[0, 1], Purpose::Live, None).unwrap();
        assert_eq!(live.delay, 1);
    }

    #[test]
    fn crossings_ends_the_processing() {
        let crossings = r#"{"op": "crossings", "level": -50.0}"#;
        // A batch run names the crossings node; a live chain the node after it
        let refused = plan_chain(&chain(&[crossings, CAR]), &[0], Purpose::Batch, None).unwrap_err();
        assert_eq!(refused[0].node, "n0");
        assert!(refused[0].reason.contains("last processing node"), "{refused:?}");
        let refused = plan_chain(&chain(&[crossings, CAR]), &[0], Purpose::Live, None).unwrap_err();
        assert_eq!(refused[0].node, "n1");
        let p = plan_chain(&chain(&[CAR, crossings]), &[0], Purpose::Batch, None).unwrap();
        assert!(p.crossings.is_some());
        // A view previews the signal crossings reads
        let v = plan_chain(&chain(&[CAR, crossings]), &[0], Purpose::View, None).unwrap();
        assert!(v.crossings.is_none() && v.steps.len() == 1);
    }
}
