//! The parameters of one pipeline node, editable: kitchen-ui's parameter editor
//! (`kitchen_ui::ParamEditor`) on [`NodeParams`]. Numbers are inputs (Enter or leaving the field
//! applies them), choices menus, switches checkboxes. Who owns the node decides how a change is
//! applied ([`Commit`]: a session command), so the Processing panel and the Pipeline canvas's
//! inspector share it.

use std::rc::Rc;

use dspk_pipeline::Node;
use gpui_kit::{App, Context, Window};
use kitchen_ui::Params;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub use kitchen_ui::param_editor::{label_of, trim_number};

/// A pipeline node as the editor sees it: its JSON (`{"op": "iir", "band": …, "cutoff": […]}`).
/// A type of this crate, so kitchen-ui's [`Params`] can be implemented for it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeParams(pub Node);

impl Params for NodeParams {
    fn choices(field: &str) -> Option<&'static [&'static str]> {
        choices(field)
    }

    fn element_label(field: &str, len: usize, i: usize) -> String {
        match (field, len, i) {
            ("cutoff", 2, 0) => "Low (Hz)".to_string(),
            ("cutoff", 2, 1) => "High (Hz)".to_string(),
            ("cutoff", _, _) => "Cutoff (Hz)".to_string(),
            _ => format!("{} {}", label_of(field), i + 1),
        }
    }

    /// A band change keeps the cutoffs it can (one for low / high pass, two for band pass / stop).
    fn on_choice(value: &mut Value, field: &str, choice: &str) {
        if field != "band" {
            return;
        }
        let cut: Vec<f64> = value["cutoff"].as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
        let (lo, hi) = (cut.first().copied().unwrap_or(300.0), cut.last().copied().unwrap_or(6000.0));
        value["cutoff"] = match choice {
            "lowpass" => json!([hi]),
            "highpass" => json!([lo]),
            _ if cut.len() == 2 => json!(cut),
            _ => json!([lo, if hi > lo { hi } else { lo * 10.0 }]),
        };
    }
}

/// The editor of one node's parameters (a view: `cx.new(|_| ParamEditor::new())`).
pub type ParamEditor = kitchen_ui::ParamEditor<NodeParams>;

/// Sets new parameters on the node; an error is shown under the fields.
pub type Commit = Rc<dyn Fn(Node, &mut App) -> Result<(), String>>;

/// Choices of the enum parameters, by field name.
pub fn choices(field: &str) -> Option<&'static [&'static str]> {
    Some(match field {
        "band" => &["lowpass", "highpass", "bandpass", "bandstop"],
        "design" => &["butterworth", "chebyshev1", "bessel"],
        "mode" => &["forward_backward", "forward"],
        "start" => &["rest", "steady_state"],
        "edge" => &["zeros", "odd", "reflect", "nearest"],
        _ => return None,
    })
}

/// Editing a [`Node`] directly.
pub trait EditNode {
    /// Edits `node`, known as `key` (e.g. `v1/iir`); inputs are rebuilt when the key or the
    /// node's values change.
    fn set_node(&mut self, key: &str, node: &Node, commit: Commit, window: &mut Window, cx: &mut Context<Self>)
    where
        Self: Sized;
}

impl EditNode for ParamEditor {
    fn set_node(&mut self, key: &str, node: &Node, commit: Commit, window: &mut Window, cx: &mut Context<Self>) {
        let commit: kitchen_ui::Commit<NodeParams> = Rc::new(move |p: NodeParams, cx| commit(p.0, cx));
        self.set_target(key, &NodeParams(node.clone()), commit, window, cx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_labels_and_numbers() {
        assert_eq!(label_of("sigma_samples"), "Sigma samples");
        assert_eq!(trim_number(300.0), "300");
        assert_eq!(trim_number(0.25), "0.25");
        assert_eq!(NodeParams::element_label("cutoff", 2, 1), "High (Hz)");
    }

    #[test]
    fn a_band_change_keeps_the_cutoffs_it_can() {
        let mut v = json!({"op": "iir", "band": "bandpass", "cutoff": [300.0, 6000.0]});
        NodeParams::on_choice(&mut v, "band", "highpass");
        assert_eq!(v["cutoff"], json!([300.0]));
        NodeParams::on_choice(&mut v, "band", "bandstop");
        assert_eq!(v["cutoff"], json!([300.0, 3000.0]));
    }
}
