//! What a node card shows of an operation's parameters, how a dragged number steps, and where
//! an operation is explained (its docs) and written (its code and GPU kernel). Plain functions
//! of a [`Node`]: the canvas draws them.

use dspk_pipeline::Node;
use kitchen_ui::Params as _;
use serde_json::Value;

use crate::ui::param_editor::{choices, label_of, trim_number, NodeParams};

/// One row of a node card: a field (or one element of a list field) and its value.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamRow {
    pub field: String,
    /// The element of a list field (`cutoff`'s low and high), or the whole field.
    pub index: Option<usize>,
    pub label: String,
    pub value: ParamValue,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    /// Dragged to step, clicked to type; `integer` steps by one.
    Number { value: f64, integer: bool },
    /// One of a few names (a menu).
    Choice { value: String, options: &'static [&'static str] },
    /// On or off.
    Flag(bool),
    /// Shown, edited in the inspector (a map of channels, a list of taps…).
    Text(String),
}

impl ParamRow {
    /// The value as the card shows it.
    pub fn shown(&self) -> String {
        match &self.value {
            ParamValue::Number { value, .. } => trim_number(*value),
            ParamValue::Choice { value, .. } => label_of(value),
            ParamValue::Flag(on) => if *on { "on" } else { "off" }.to_string(),
            ParamValue::Text(t) => t.clone(),
        }
    }

    /// A key naming this row of node `id` (element ids, the field being typed).
    pub fn key(&self, id: &str) -> String {
        match self.index {
            Some(i) => format!("{id}-{}-{i}", self.field),
            None => format!("{id}-{}", self.field),
        }
    }
}

/// Lists longer than this are shown as a count (edited in the inspector).
const LIST_ROWS_MAX: usize = 4;

/// The rows of `node`'s parameters: numbers first, then choices, switches and the rest, each in
/// its fields' order (alphabetical).
pub fn param_rows(node: &Node) -> Vec<ParamRow> {
    let Ok(Value::Object(map)) = serde_json::to_value(node) else { return Vec::new() };
    let mut rows = Vec::new();
    for (field, v) in map {
        if field == "op" {
            continue;
        }
        let row = |index: Option<usize>, label: String, value: ParamValue| ParamRow { field: field.clone(), index, label, value };
        match v {
            Value::Number(n) => rows.push(row(None, label_of(&field), ParamValue::Number { value: n.as_f64().unwrap_or(0.0), integer: n.is_i64() || n.is_u64() })),
            Value::Bool(b) => rows.push(row(None, label_of(&field), ParamValue::Flag(b))),
            Value::String(s) => rows.push(match choices(&field) {
                Some(options) => row(None, label_of(&field), ParamValue::Choice { value: s, options }),
                None => row(None, label_of(&field), ParamValue::Text(s)),
            }),
            Value::Array(a) if !a.is_empty() && a.len() <= LIST_ROWS_MAX && a.iter().all(Value::is_number) => {
                for (i, e) in a.iter().enumerate() {
                    let n = e.as_number().expect("checked");
                    let value = ParamValue::Number { value: n.as_f64().unwrap_or(0.0), integer: n.is_i64() || n.is_u64() };
                    rows.push(row(Some(i), NodeParams::element_label(&field, a.len(), i), value));
                }
            }
            Value::Array(a) => rows.push(row(None, label_of(&field), ParamValue::Text(format!("{} values", a.len())))),
            Value::Object(o) => rows.push(row(None, label_of(&field), ParamValue::Text(format!("{} entries", o.len())))),
            Value::Null => {}
        }
    }
    // What is tuned most first: numbers, then choices, switches, the rest (stable within each)
    rows.sort_by_key(|r| match r.value {
        ParamValue::Number { .. } => 0,
        ParamValue::Choice { .. } => 1,
        ParamValue::Flag(_) => 2,
        ParamValue::Text(_) => 3,
    });
    rows
}

/// `node` with `field` (element `index`) set to `value`, or `None` when that is not a valid node.
/// A band change keeps the cutoffs it can (as the inspector does).
pub fn with_value(node: &Node, field: &str, index: Option<usize>, value: Value) -> Option<Node> {
    let mut v = serde_json::to_value(node).ok()?;
    let choice = value.as_str().map(str::to_string);
    match index {
        None => *v.get_mut(field)? = value,
        Some(i) => *v.get_mut(field)?.get_mut(i)? = value,
    }
    if let Some(choice) = choice {
        NodeParams::on_choice(&mut v, field, &choice);
    }
    serde_json::from_value(v).ok()
}

/// One drag step of `value` in direction `dir` (−1 or +1): integers by one (not below one),
/// other numbers by a tenth of their order of magnitude (300 → 10, 6000 → 100, 0.5 → 0.01),
/// landing on a multiple of the step.
pub fn stepped(value: f64, integer: bool, dir: f32) -> f64 {
    if integer {
        return (value + dir as f64).max(1.0);
    }
    let step = nice_step(value);
    ((value + dir as f64 * step) / step).round() * step
}

fn nice_step(value: f64) -> f64 {
    let a = value.abs();
    if a < 1e-9 { 0.1 } else { 10f64.powf(a.log10().floor() - 1.0) }
}

/// Where an operation is explained and written.
#[derive(Debug, Clone, PartialEq)]
pub struct OpLinks {
    /// Its section of the documentation.
    pub docs: String,
    /// The code that runs it (dspk-ops, or the pipeline for reading and writing).
    pub source: String,
    /// The GPU kernel it launches, when it has its own.
    pub kernel: Option<String>,
}

/// The documentation site.
pub const DOCS_URL: &str = "https://neuro-kitchen.github.io/dsp-kitchen/";
/// The code, as released on `main`.
pub const CODE_URL: &str = "https://github.com/neuro-kitchen/dsp-kitchen/blob/main/";

/// Where `node` is explained and written.
pub fn op_links(node: &Node) -> OpLinks {
    // (docs page and section, source under crates/dspk-ops/src, kernel under the same)
    let (docs, source, kernel): (&str, &str, Option<&str>) = match node {
        Node::Iir { .. } | Node::Sos { .. } => ("operations/filters/#iir", "filter/iir/sos.rs", Some("filter/iir/kernels/sos.rs")),
        Node::Notch { .. } => ("operations/filters/#notch", "filter/design/notch.rs", Some("filter/iir/kernels/sos.rs")),
        Node::Fir { .. } => ("operations/filters/#fir", "filter/fir/conv.rs", Some("filter/fir/kernels/conv.rs")),
        Node::GaussianSmooth { .. } => ("operations/filters/#gaussian-smoothing", "filter/fir/gaussian.rs", Some("filter/fir/kernels/conv.rs")),
        Node::Median { .. } => ("operations/filters/#running-median", "filter/non_linear/median.rs", Some("filter/non_linear/kernels/median.rs")),
        Node::TeagerKaiser { .. } => ("operations/filters/#teagerkaiser-energy", "filter/non_linear/teager_kaiser.rs", Some("filter/non_linear/kernels/teager_kaiser.rs")),
        Node::Car {} | Node::Cmr {} => ("operations/referencing/#common-average-and-median-reference", "spatial/car.rs", Some("spatial/kernels/car.rs")),
        Node::Laplacian { .. } => ("operations/referencing/#surface-laplacian", "spatial/laplacian.rs", Some("spatial/kernels/sparse.rs")),
        Node::Whiten { .. } => ("operations/referencing/#whitening", "spatial/whitening.rs", Some("linalg/kernels/matmul.rs")),
        Node::Resample { .. } => ("operations/rate/#polyphase-resampling", "resampler/poly.rs", Some("resampler/kernels/poly.rs")),
        Node::Crossings { .. } => ("operations/statistics/#crossing-a-threshold", "crossings/mod.rs", Some("crossings/kernels/mod.rs")),
        Node::Scale { .. } | Node::SubtractBaseline { .. } => ("operations/per-sample/", "math/scaling.rs", Some("math/kernels/scale.rs")),
        Node::Clamp { .. } => ("operations/per-sample/", "math/clamp.rs", Some("math/kernels/clamp.rs")),
        Node::Rectify {} | Node::Square {} | Node::Sqrt {} | Node::Log10 {} => ("operations/per-sample/", "math/elementwise.rs", Some("math/kernels/elementwise.rs")),
        Node::Remap { .. } => ("operations/channels/", "core/kernels/gather.rs", Some("core/kernels/gather.rs")),
        Node::Read { .. } | Node::Write { .. } => ("operations/pipelines/", "", None),
    };
    let ops = |path: &str| format!("{CODE_URL}crates/dspk-ops/src/{path}");
    OpLinks {
        docs: format!("{DOCS_URL}{docs}"),
        // Reading and writing are the pipeline's own
        source: if source.is_empty() { format!("{CODE_URL}engine/dspk-pipeline/src/run.rs") } else { ops(source) },
        kernel: kernel.map(ops),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn node(v: Value) -> Node {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn a_filter_shows_one_row_per_value_and_each_cutoff() {
        let rows = param_rows(&node(json!({"op": "iir", "band": "bandpass", "cutoff": [300.0, 6000.0], "order": 5})));
        let shown: Vec<(String, String)> = rows.iter().map(|r| (r.label.clone(), r.shown())).collect();
        assert!(shown.contains(&("Low (Hz)".into(), "300".into())), "{shown:?}");
        assert!(shown.contains(&("High (Hz)".into(), "6000".into())), "{shown:?}");
        assert!(rows.iter().any(|r| r.field == "band" && matches!(r.value, ParamValue::Choice { .. })), "{shown:?}");
        assert!(rows.iter().any(|r| r.field == "order" && matches!(r.value, ParamValue::Number { integer: true, .. })), "{shown:?}");
        // No parameters, no rows
        assert!(param_rows(&node(json!({"op": "car"}))).is_empty());
        // The numbers come first: the cutoffs and the order fit in a card's rows
        assert!(rows[..3].iter().all(|r| matches!(r.value, ParamValue::Number { .. })), "{shown:?}");
    }

    #[test]
    fn a_value_set_on_a_row_makes_a_valid_node_or_none() {
        let iir = node(json!({"op": "iir", "band": "bandpass", "cutoff": [300.0, 6000.0], "order": 5}));
        let high = with_value(&iir, "cutoff", Some(1), json!(5000.0)).unwrap();
        assert_eq!(serde_json::to_value(&high).unwrap()["cutoff"], json!([300.0, 5000.0]));
        // A band change keeps one cutoff for a high pass
        let hp = with_value(&iir, "band", None, json!("highpass")).unwrap();
        assert_eq!(serde_json::to_value(&hp).unwrap()["cutoff"], json!([300.0]));
        // Not a number where one is needed: no node
        assert!(with_value(&iir, "order", None, json!("five")).is_none());
    }

    #[test]
    fn dragging_steps_by_a_tenth_of_the_magnitude() {
        assert_eq!(stepped(300.0, false, 1.0), 310.0);
        assert_eq!(stepped(6000.0, false, -1.0), 5900.0);
        assert!((stepped(0.5, false, 1.0) - 0.51).abs() < 1e-9);
        assert_eq!(stepped(5.0, true, 1.0), 6.0);
        assert_eq!(stepped(1.0, true, -1.0), 1.0, "integers stay at one or more");
    }

    #[test]
    fn every_op_has_docs_and_code() {
        for v in [json!({"op": "iir", "band": "lowpass", "cutoff": [100.0], "order": 2}), json!({"op": "car"}), json!({"op": "rectify"})] {
            let links = op_links(&node(v));
            assert!(links.docs.starts_with(DOCS_URL) && links.source.starts_with(CODE_URL));
            assert!(links.kernel.is_some());
        }
    }
}
