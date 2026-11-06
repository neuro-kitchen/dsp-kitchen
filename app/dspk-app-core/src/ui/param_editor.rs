//! The parameters of one op, editable: numbers as inputs (Enter or leaving the field applies
//! them), choices as menus, switches as checkboxes. Who owns the op decides how a change is
//! applied (`commit`: a session command), so the Processing panel and the Pipeline canvas's
//! inspector share it.

use std::rc::Rc;

use dspk_pipeline::Node;
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::{v_flex, ActiveTheme as _, Sizable as _};
use gpui_kit::{div, App, AppContext as _, Context, Entity, IntoElement, ParentElement as _, Render, SharedString, Styled as _, Subscription, Window};
use serde_json::{json, Value};

use super::widgets::{row, MenuSelect};

/// Sets new parameters on the op; an error is shown under the fields.
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

/// One editable number: its place in the node's JSON (`cutoff`, index 1).
struct Field {
    key: String,
    index: Option<usize>,
    label: String,
    input: Entity<InputState>,
}

#[derive(Default)]
pub struct ParamEditor {
    fields: Vec<Field>,
    /// What the fields were built for (key and node JSON).
    fields_for: String,
    field_subs: Vec<Subscription>,
    error: Option<String>,
    target: Option<(Node, Commit)>,
}

impl ParamEditor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Edits `node`, known as `key` (e.g. `v1/iir`); inputs are rebuilt when the key or the
    /// node's values change.
    pub fn set_target(&mut self, key: &str, node: &Node, commit: Commit, window: &mut Window, cx: &mut Context<Self>) {
        let value = serde_json::to_value(node).unwrap_or(Value::Null);
        self.target = Some((node.clone(), commit));
        let fresh = format!("{key}/{value}");
        if fresh == self.fields_for {
            return;
        }
        if !self.fields_for.starts_with(&format!("{key}/")) {
            self.error = None;
        }
        self.fields_for = fresh;
        self.fields.clear();
        self.field_subs.clear();
        let Value::Object(map) = &value else { return };
        for (k, v) in map {
            match v {
                Value::Number(n) => self.add_field(k, None, label_of(k), n.as_f64().unwrap_or(0.0), window, cx),
                Value::Array(a) if a.len() <= 2 && a.iter().all(Value::is_number) => {
                    for (i, n) in a.iter().enumerate() {
                        let label = match (k.as_str(), a.len(), i) {
                            ("cutoff", 2, 0) => "Low (Hz)".to_string(),
                            ("cutoff", 2, 1) => "High (Hz)".to_string(),
                            ("cutoff", _, _) => "Cutoff (Hz)".to_string(),
                            _ => format!("{} {}", label_of(k), i + 1),
                        };
                        self.add_field(k, Some(i), label, n.as_f64().unwrap_or(0.0), window, cx);
                    }
                }
                _ => {}
            }
        }
    }

    fn add_field(&mut self, key: &str, index: Option<usize>, label: String, v: f64, window: &mut Window, cx: &mut Context<Self>) {
        let input = cx.new(|cx| InputState::new(window, cx).default_value(trim_number(v)));
        let sub = cx.subscribe(&input, |this, _, e: &InputEvent, cx| {
            if matches!(e, InputEvent::PressEnter { .. } | InputEvent::Blur) {
                this.apply_fields(cx);
            }
        });
        self.field_subs.push(sub);
        self.fields.push(Field { key: key.to_string(), index, label, input });
    }

    /// The numbers typed in, on the op; a value that does not parse, or parameters the spec
    /// rejects, leave the op unchanged and say why.
    fn apply_fields(&mut self, cx: &mut Context<Self>) {
        let Some((node, _)) = &self.target else { return };
        let mut value = serde_json::to_value(node).unwrap_or(Value::Null);
        for f in &self.fields {
            let text = f.input.read(cx).value().to_string();
            let Ok(n) = text.trim().parse::<f64>() else {
                self.error = Some(format!("{}: {text:?} is not a number", f.label));
                cx.notify();
                return;
            };
            let slot = match f.index {
                Some(i) => value.get_mut(&f.key).and_then(|a| a.get_mut(i)),
                None => value.get_mut(&f.key),
            };
            if let Some(slot) = slot {
                // Integers stay integers (orders, widths)
                *slot = if slot.is_u64() || slot.is_i64() { json!(n.round() as i64) } else { json!(n) };
            }
        }
        self.set_value(value, cx);
    }

    fn set_value(&mut self, value: Value, cx: &mut Context<Self>) {
        let Some((node, commit)) = self.target.clone() else { return };
        match serde_json::from_value::<Node>(value) {
            Ok(new) if new == node => {}
            Ok(new) => self.error = commit(new, cx).err(),
            Err(e) => self.error = Some(e.to_string()),
        }
        cx.notify();
    }

    /// A new enum value (band, mode, …); a band change keeps the cutoffs it can (one for low /
    /// high pass, two for band pass / stop).
    fn set_choice(&mut self, key: &str, choice: &str, cx: &mut Context<Self>) {
        let Some((node, _)) = &self.target else { return };
        let mut value = serde_json::to_value(node).unwrap_or(Value::Null);
        value[key] = json!(choice);
        if key == "band" {
            let cut: Vec<f64> = value["cutoff"].as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
            let (lo, hi) = (cut.first().copied().unwrap_or(300.0), cut.last().copied().unwrap_or(6000.0));
            value["cutoff"] = match choice {
                "lowpass" => json!([hi]),
                "highpass" => json!([lo]),
                _ if cut.len() == 2 => json!(cut),
                _ => json!([lo, if hi > lo { hi } else { lo * 10.0 }]),
            };
        }
        self.set_value(value, cx);
    }

    fn set_bool(&mut self, key: &str, on: bool, cx: &mut Context<Self>) {
        let Some((node, _)) = &self.target else { return };
        let mut value = serde_json::to_value(node).unwrap_or(Value::Null);
        value[key] = json!(on);
        self.set_value(value, cx);
    }
}

impl Render for ParamEditor {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let (muted, danger) = (cx.theme().muted_foreground, cx.theme().danger);
        let Some((node, _)) = &self.target else { return v_flex() };
        let value = serde_json::to_value(node).unwrap_or(Value::Null);
        let mut out = v_flex().gap_2();
        for f in &self.fields {
            out = out.child(row(f.label.clone(), div().w_full().child(Input::new(&f.input).small()), cx));
        }
        if let Value::Object(map) = &value {
            for (k, v) in map {
                if let (Some(opts), Some(cur)) = (choices(k), v.as_str()) {
                    let this = cx.entity().downgrade();
                    let key = k.clone();
                    let names: Vec<SharedString> = opts.iter().map(|o| SharedString::from(o.replace('_', " "))).collect();
                    let sel = opts.iter().position(|o| *o == cur);
                    out = out.child(row(
                        label_of(k),
                        MenuSelect::new(SharedString::from(format!("param-{k}")), names, sel, move |i, _, cx| {
                            let _ = this.update(cx, |p, cx| p.set_choice(&key, opts[i], cx));
                        }),
                        cx,
                    ));
                } else if let Some(b) = v.as_bool() {
                    let this = cx.entity().downgrade();
                    let key = k.clone();
                    out = out.child(row(
                        label_of(k),
                        Checkbox::new(SharedString::from(format!("param-{k}"))).checked(b).on_click(move |on: &bool, _, cx| {
                            let on = *on;
                            let _ = this.update(cx, |p, cx| p.set_bool(&key, on, cx));
                        }),
                        cx,
                    ));
                } else if v.as_array().is_some_and(|a| a.len() > 2) {
                    out = out.child(row(label_of(k), div().text_xs().text_color(muted).child(format!("{} values (edit in the session file)", v.as_array().map_or(0, Vec::len))), cx));
                }
            }
        }
        if self.fields.is_empty() && !value.as_object().is_some_and(|m| m.keys().any(|k| k != "op")) {
            out = out.child(div().text_xs().text_color(muted).child("No parameters."));
        }
        out.children(self.error.clone().map(|e| div().text_xs().text_color(danger).child(e)))
    }
}

/// `sigma_samples` → `Sigma samples`
pub fn label_of(key: &str) -> String {
    let mut s = key.replace('_', " ");
    if let Some(c) = s.get_mut(0..1) {
        c.make_ascii_uppercase();
    }
    s
}

/// `300` rather than `300.0`, at most 6 decimals.
pub fn trim_number(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        let s = format!("{v:.6}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
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
    }
}
