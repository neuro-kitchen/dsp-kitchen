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
    /// A channel map, every channel of the input in order to start (edited in the inspector).
    Remap,
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

/// Every family, in tree and menu order. Each item is one op type of the spec (Author: "sub
/// families … like filter iir fir notch"); its values are set in the inspector, from presets.
pub fn catalog() -> Vec<Category> {
    vec![
        Category {
            name: "Sources",
            icon: IconName::AudioWaveform,
            items: vec![
                Item { label: "Open recording", hint: "The recording open in Explore", template: Template::OpenRecording },
                Item { label: "Recording from file…", hint: "SpikeGLX, raw .bin + .meta, NWB Zarr", template: Template::RecordingFile },
            ],
        },
        Category {
            name: "Channels",
            icon: IconName::ListOrdered,
            items: vec![Item { label: "Remap", hint: "Reorder, drop and name channels (a channel table, or a map file)", template: Template::Remap }],
        },
        Category {
            name: "Filter",
            icon: IconName::SlidersHorizontal,
            items: vec![
                op("IIR", "Butterworth, Chebyshev or Bessel; band-, high-, low-pass or band-stop", json!({"op": "iir", "band": "bandpass", "cutoff": [300.0, 6000.0]})),
                op("FIR", "Taps you give (a moving average to start)", json!({"op": "fir", "taps": [0.2, 0.2, 0.2, 0.2, 0.2], "centered": true})),
                op("SOS", "Second-order sections you give", json!({"op": "sos", "sections": [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]})),
                op("Notch", "One frequency out (mains hum)", json!({"op": "notch", "freq": 60.0, "q": 30.0})),
            ],
        },
        Category {
            name: "Reference",
            icon: IconName::Rows3,
            items: vec![
                op("CMR", "Common median reference", json!({"op": "cmr"})),
                op("CAR", "Common average reference", json!({"op": "car"})),
                op("Laplacian", "Each electrode minus its grid neighbours (HD-EMG, ECoG)", json!({"op": "laplacian", "grid": [4, 8]})),
                op("Whiten", "Decorrelates the channels (ZCA), fitted on the signal before it", json!({"op": "whiten"})),
            ],
        },
        Category {
            name: "Smooth",
            icon: IconName::Activity,
            items: vec![
                op("Median", "Running median in time", json!({"op": "median", "width": 5})),
                op("Gaussian", "Gaussian smoothing in time", json!({"op": "gaussian_smooth", "sigma_samples": 2.0})),
            ],
        },
        Category {
            name: "Transform",
            icon: IconName::Workflow,
            items: vec![
                op("Scale", "a · x + b", json!({"op": "scale", "alpha": 1.0, "beta": 0.0})),
                op("Subtract baseline", "x − baseline", json!({"op": "subtract_baseline", "baseline": 0.0})),
                op("Clamp", "Limits the values", json!({"op": "clamp", "min": -1000.0, "max": 1000.0})),
                op("Teager–Kaiser", "Instantaneous energy (spike detection)", json!({"op": "teager_kaiser"})),
                op("Rectify", "|x| (then a low-pass: an envelope)", json!({"op": "rectify"})),
                op("Square", "x² (instantaneous power)", json!({"op": "square"})),
                op("Square root", "√x (after a squared, smoothed signal: RMS)", json!({"op": "sqrt"})),
                op("Log₁₀", "log₁₀ |x| (energies over decades)", json!({"op": "log10"})),
            ],
        },
        Category {
            name: "Events",
            icon: IconName::ChartNoAxesColumn,
            items: vec![op(
                "Crossings",
                "Events where each channel goes past a level and back (onsets, offsets, peaks); then Write saves them as an event set",
                json!({"op": "crossings", "level": 50.0, "release": 25.0, "refractory_ms": 50.0}),
            )],
        },
        Category {
            name: "Rate",
            icon: IconName::ChartNoAxesColumn,
            items: vec![op("Resample", "up / down (polyphase FIR); the last op before Write", json!({"op": "resample", "up": 1, "down": 2}))],
        },
        Category {
            name: "Output",
            icon: IconName::Eye,
            items: vec![
                Item { label: "View", hint: "A stream shown in Explore", template: Template::View },
                Item { label: "Write", hint: "The chain written to a processed file (float32 Zarr); after Crossings, its events as an event set", template: Template::Write },
            ],
        },
    ]
}

/// The type name of an op node on the canvas (`IIR`, `CMR`, …).
pub fn type_name(node: &Node) -> &'static str {
    match node {
        Node::Iir { .. } => "IIR",
        Node::Fir { .. } => "FIR",
        Node::Sos { .. } => "SOS",
        Node::Notch { .. } => "Notch",
        Node::Car {} => "CAR",
        Node::Cmr {} => "CMR",
        Node::Median { .. } => "Median",
        Node::GaussianSmooth { .. } => "Gaussian",
        Node::Scale { .. } => "Scale",
        Node::SubtractBaseline { .. } => "Subtract baseline",
        Node::Clamp { .. } => "Clamp",
        Node::TeagerKaiser { .. } => "Teager–Kaiser",
        Node::Rectify {} => "Rectify",
        Node::Square {} => "Square",
        Node::Sqrt {} => "Square root",
        Node::Log10 {} => "Log₁₀",
        Node::Laplacian { .. } => "Laplacian",
        Node::Whiten { .. } => "Whiten",
        Node::Crossings { .. } => "Crossings",
        Node::Resample { .. } => "Resample",
        Node::Remap { .. } => "Remap",
        Node::Read { .. } => "Read",
        Node::Write { .. } => "Write",
    }
}

/// Presets of an op type: label and the node's values (`IIR`: the usual bands).
pub fn presets(node: &Node) -> Vec<(&'static str, Value)> {
    let iir = |band: &str, cutoff: Value, design: &str, order: u32| json!({"op": "iir", "band": band, "cutoff": cutoff, "design": design, "order": order});
    match node {
        Node::Iir { .. } => vec![
            ("Spikes · band-pass 300–6000 Hz (Butterworth 3)", iir("bandpass", json!([300.0, 6000.0]), "butterworth", 3)),
            ("Spikes · high-pass 300 Hz (Butterworth 3)", iir("highpass", json!([300.0]), "butterworth", 3)),
            ("LFP · low-pass 300 Hz (Butterworth 4)", iir("lowpass", json!([300.0]), "butterworth", 4)),
            ("EMG · band-pass 20–450 Hz (Butterworth 4)", iir("bandpass", json!([20.0, 450.0]), "butterworth", 4)),
            ("ECG · band-pass 0.5–40 Hz (Butterworth 2)", iir("bandpass", json!([0.5, 40.0]), "butterworth", 2)),
            ("EEG · band-pass 1–70 Hz (Butterworth 4)", iir("bandpass", json!([1.0, 70.0]), "butterworth", 4)),
            ("Mains · band-stop 58–62 Hz (Butterworth 2)", iir("bandstop", json!([58.0, 62.0]), "butterworth", 2)),
            ("Mains · band-stop 48–52 Hz (Butterworth 2)", iir("bandstop", json!([48.0, 52.0]), "butterworth", 2)),
        ],
        Node::Notch { .. } => vec![("Mains 60 Hz", json!({"op": "notch", "freq": 60.0, "q": 30.0})), ("Mains 50 Hz", json!({"op": "notch", "freq": 50.0, "q": 30.0})), ("Harmonic 120 Hz", json!({"op": "notch", "freq": 120.0, "q": 30.0}))],
        Node::Median { .. } => vec![("3 samples", json!({"op": "median", "width": 3})), ("5 samples", json!({"op": "median", "width": 5})), ("9 samples", json!({"op": "median", "width": 9}))],
        Node::GaussianSmooth { .. } => vec![("σ 1 sample", json!({"op": "gaussian_smooth", "sigma_samples": 1.0})), ("σ 2 samples", json!({"op": "gaussian_smooth", "sigma_samples": 2.0})), ("σ 5 samples", json!({"op": "gaussian_smooth", "sigma_samples": 5.0}))],
        Node::Fir { .. } => vec![("Derivative (first difference)", json!({"op": "fir", "taps": [1.0, -1.0], "centered": false})), ("Moving average, 5 taps", json!({"op": "fir", "taps": [0.2, 0.2, 0.2, 0.2, 0.2], "centered": true})), ("Moving average, 9 taps", json!({"op": "fir", "taps": [0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111], "centered": true}))],
        Node::Laplacian { .. } => vec![
            ("Grid 4 × 8 (32 electrodes)", json!({"op": "laplacian", "grid": [4, 8]})),
            ("Grid 8 × 4 (32 electrodes)", json!({"op": "laplacian", "grid": [8, 4]})),
            ("Grid 8 × 8 (64 electrodes)", json!({"op": "laplacian", "grid": [8, 8]})),
            ("Grid 16 × 4 (64 electrodes)", json!({"op": "laplacian", "grid": [16, 4]})),
        ],
        Node::Scale { .. } => vec![("Invert", json!({"op": "scale", "alpha": -1.0, "beta": 0.0})), ("µV → mV", json!({"op": "scale", "alpha": 0.001, "beta": 0.0}))],
        _ => Vec::new(),
    }
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
        Node::Crossings { level, .. } => format!(".crossings(level = {level})"),
        Node::Median { width, .. } => format!(".median(width = {width})"),
        Node::TeagerKaiser { .. } => ".teagerKaiser()".into(),
        Node::Rectify {} => ".map { abs(it) }".into(),
        Node::Square {} => ".map { it * it }".into(),
        Node::Sqrt {} => ".map { sqrt(it) }".into(),
        Node::Log10 {} => ".map { log10(abs(it)) }".into(),
        Node::Laplacian { grid: [r, c] } => format!(".laplacian(grid = {r} × {c})"),
        Node::Whiten { fit_s, .. } => format!(".whiten(fit = {fit_s} s)"),
        Node::Resample { up, down, .. } => format!(".resample({up}, {down})"),
        Node::Remap { pairs, path, .. } => match path {
            Some(p) => format!(".remap({:?})", p.display().to_string()),
            None => format!(".remap(pairs = {})", pairs.len()),
        },
        Node::Read { .. } => "flowOf(recording)".into(),
        Node::Write { .. } => ".collect(write)".into(),
    }
}

/// A node's title, its second line, and its icon on the canvas.
pub fn describe(kind: &NodeKind) -> (String, String, IconName) {
    match kind {
        NodeKind::Source { recording } => {
            let name = crate::domain::session::view_name(recording);
            let channels = recording.channels.as_ref().map_or_else(|| "all channels".to_string(), |c| format!("{} channels", c.len()));
            (name, format!("flowOf(recording) · {channels}"), IconName::AudioWaveform)
        }
        NodeKind::Op { node, .. } => (type_name(node).to_string(), op_summary(node), IconName::SlidersHorizontal),
        NodeKind::View { name, .. } => (name.clone(), ".collect(view) · shown in Explore".into(), IconName::Eye),
        NodeKind::Write {} => ("Write".into(), ".collect(write) · processed file".into(), IconName::ArrowDown),
        NodeKind::Recipe { recipe, overrides, .. } => {
            let own = if overrides.is_empty() { String::new() } else { format!(" · {} own value{}", overrides.len(), if overrides.len() == 1 { "" } else { "s" }) };
            (recipe.clone(), format!("recipe{own}"), IconName::Package)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_ops_and_presets_are_valid_view_ops() {
        for c in catalog() {
            for item in c.items {
                if let Template::Op(v) = item.template {
                    let node: Node = serde_json::from_value(v).unwrap_or_else(|e| panic!("{}: {e}", item.label));
                    assert!(node.stage().is_ok(), "{}", item.label);
                    for (label, p) in presets(&node) {
                        let n: Node = serde_json::from_value(p).unwrap_or_else(|e| panic!("{label}: {e}"));
                        assert!(n.stage().is_ok(), "{label}");
                        assert_eq!(type_name(&n), type_name(&node), "{label} keeps the type");
                    }
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
