//! What can be put on the Pipeline canvas, grouped as the operations tree and the menu bar show
//! it (Photoshop style: one menu per kind of operation), and how a node reads on the canvas:
//! its title and its call in Kotlin Flow terms (`.bandpass(300.0, 6000.0, order = 3)`).

use dspk_pipeline::{Band, Mode, Node};
use gpui_kit::assets::IconName;
use serde_json::{json, Value};

use crate::domain::graph::NodeKind;
use crate::screens::explore::components::processing::op_summary;

/// What an item adds.
#[derive(Debug, Clone, PartialEq)]
pub enum Template {
    /// The recording open in Explore.
    OpenRecording,
    /// A recording picked in a file dialog.
    RecordingFile,
    /// An op (its node as spec JSON).
    Op(Value),
    View,
    Write,
}

#[derive(Debug, Clone)]
pub struct Item {
    pub label: &'static str,
    pub hint: &'static str,
    pub template: Template,
}

#[derive(Debug, Clone)]
pub struct Category {
    /// Also the menu's title.
    pub name: &'static str,
    pub icon: IconName,
    pub items: Vec<Item>,
}

fn op(label: &'static str, hint: &'static str, node: Value) -> Item {
    Item { label, hint, template: Template::Op(node) }
}

/// Every category, in tree and menu order.
pub fn catalog() -> Vec<Category> {
    vec![
        Category {
            name: "Sources",
            icon: IconName::AudioWaveform,
            items: vec![
                Item { label: "Open recording", hint: "The recording open in Explore", template: Template::OpenRecording },
                Item { label: "Recording from file…", hint: "Pick a file (SpikeGLX, raw .bin + .meta, NWB Zarr)", template: Template::RecordingFile },
            ],
        },
        Category {
            name: "Filter",
            icon: IconName::SlidersHorizontal,
            items: vec![
                op("Band-pass 300–6000 Hz", "Spike band (zero phase)", json!({"op": "iir", "band": "bandpass", "cutoff": [300.0, 6000.0]})),
                op("High-pass 300 Hz", "Removes the slow signal", json!({"op": "iir", "band": "highpass", "cutoff": [300.0]})),
                op("Low-pass 300 Hz", "LFP band", json!({"op": "iir", "band": "lowpass", "cutoff": [300.0]})),
                op("Low-pass 6000 Hz", "Removes high-frequency noise", json!({"op": "iir", "band": "lowpass", "cutoff": [6000.0]})),
                op("Band-stop 58–62 Hz", "Mains hum and a margin", json!({"op": "iir", "band": "bandstop", "cutoff": [58.0, 62.0]})),
                op("Notch 60 Hz", "Mains hum (Americas)", json!({"op": "notch", "freq": 60.0, "q": 30.0})),
                op("Notch 50 Hz", "Mains hum (Europe, Asia)", json!({"op": "notch", "freq": 50.0, "q": 30.0})),
            ],
        },
        Category {
            name: "Reference",
            icon: IconName::Rows3,
            items: vec![
                op("Common median reference", "Subtracts the median over channels, per sample", json!({"op": "cmr"})),
                op("Common average reference", "Subtracts the mean over channels, per sample", json!({"op": "car"})),
            ],
        },
        Category {
            name: "Smooth",
            icon: IconName::Activity,
            items: vec![
                op("Median (5 samples)", "Removes spikes of noise, keeps edges", json!({"op": "median", "width": 5})),
                op("Gaussian smoothing (σ 2 samples)", "Low-pass in time", json!({"op": "gaussian_smooth", "sigma_samples": 2.0})),
            ],
        },
        Category {
            name: "Transform",
            icon: IconName::Workflow,
            items: vec![
                op("Scale", "a · x + b", json!({"op": "scale", "alpha": 1.0, "beta": 0.0})),
                op("Subtract baseline", "x − baseline", json!({"op": "subtract_baseline", "baseline": 0.0})),
                op("Clamp ±1000", "Limits the values", json!({"op": "clamp", "min": -1000.0, "max": 1000.0})),
                op("Teager–Kaiser energy", "Instantaneous energy (spike detection)", json!({"op": "teager_kaiser"})),
            ],
        },
        Category {
            name: "Output",
            icon: IconName::Eye,
            items: vec![
                Item { label: "View", hint: "A stream shown in Explore", template: Template::View },
                Item { label: "Write", hint: "The chain written to a processed file (float32 Zarr)", template: Template::Write },
            ],
        },
    ]
}

/// `.bandpass(300.0, 6000.0, order = 3)` — the op as a Flow operator call.
pub fn flow_call(node: &Node) -> String {
    let n = |v: f64| if v.fract() == 0.0 { format!("{v:.1}") } else { format!("{v}") };
    match node {
        Node::Iir { band, cutoff, order, mode, .. } => {
            let phase = if *mode == Mode::ForwardBackward { "" } else { ", causal = true" };
            match (band, cutoff.as_slice()) {
                (Band::Bandpass, [lo, hi]) => format!(".bandpass({}, {}, order = {order}{phase})", n(*lo), n(*hi)),
                (Band::Bandstop, [lo, hi]) => format!(".bandstop({}, {}, order = {order}{phase})", n(*lo), n(*hi)),
                (Band::Highpass, [f]) => format!(".highpass({}, order = {order}{phase})", n(*f)),
                (Band::Lowpass, [f]) => format!(".lowpass({}, order = {order}{phase})", n(*f)),
                _ => ".iir(…)".into(),
            }
        }
        Node::Notch { freq, q, .. } => format!(".notch({}, q = {})", n(*freq), n(*q)),
        Node::Car {} => ".car()".into(),
        Node::Cmr {} => ".cmr()".into(),
        Node::Scale { alpha, beta } => format!(".map {{ it * {alpha} + {beta} }}"),
        Node::SubtractBaseline { baseline } => format!(".map {{ it - {baseline} }}"),
        Node::Clamp { min, max } => format!(".map {{ it.coerceIn({min}, {max}) }}"),
        Node::Sos { sections, .. } => format!(".sos(sections = {})", sections.len()),
        Node::Fir { taps, .. } => format!(".fir(taps = {})", taps.len()),
        Node::GaussianSmooth { sigma_samples, .. } => format!(".gaussianSmooth(sigma = {sigma_samples})"),
        Node::Median { width, .. } => format!(".median(width = {width})"),
        Node::TeagerKaiser { .. } => ".teagerKaiser()".into(),
        Node::Resample { up, down, .. } => format!(".resample({up}, {down})"),
        Node::Read { .. } => "flowOf(recording)".into(),
        Node::Write { .. } => ".collect(write)".into(),
    }
}

/// A node's title, its second line, and its icon on the canvas.
pub fn describe(kind: &NodeKind) -> (String, String, IconName) {
    match kind {
        NodeKind::Source { recording } => {
            let file = recording.path.file_name().map_or_else(|| recording.path.display().to_string(), |n| n.to_string_lossy().into_owned());
            let channels = recording.channels.as_ref().map_or_else(|| "all channels".to_string(), |c| format!("{} channels", c.len()));
            (file, format!("flowOf(recording) · {channels}"), IconName::AudioWaveform)
        }
        NodeKind::Op { node, .. } => {
            let summary = op_summary(node);
            let title = summary.split(" · ").next().unwrap_or(&summary).to_string();
            (title, flow_call(node), IconName::SlidersHorizontal)
        }
        NodeKind::View { name, .. } => (name.clone(), ".collect(view) · shown in Explore".into(), IconName::Eye),
        NodeKind::Write {} => ("Write".into(), ".collect(write) · processed file".into(), IconName::ArrowDown),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_ops_are_valid_view_ops() {
        for c in catalog() {
            for item in c.items {
                if let Template::Op(v) = item.template {
                    let node: Node = serde_json::from_value(v).unwrap_or_else(|e| panic!("{}: {e}", item.label));
                    assert!(node.stage().is_ok(), "{}", item.label);
                }
            }
        }
    }

    #[test]
    fn test_flow_calls() {
        let bp: Node = serde_json::from_value(json!({"op": "iir", "band": "bandpass", "cutoff": [300.0, 6000.0]})).unwrap();
        assert!(flow_call(&bp).starts_with(".bandpass(300.0, 6000.0, order = "));
        assert_eq!(flow_call(&Node::Cmr {}), ".cmr()");
    }
}
