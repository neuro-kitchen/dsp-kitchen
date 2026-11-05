//! Processing (right dock): the active view's op tree for the channel picked in Channels or on
//! the traces. The view's chain applies to every channel of its recording (one grouped array);
//! other processing for some channels: copy the view with those channels. Selecting an op shows
//! its parameters; an edit is a command on the session document, previewed live on the visible
//! window; writing the processed recording is a separate step. Plugins add their sections (e.g.
//! the sorting of the view).


use dspk_pipeline::{Band, Node};
use gpui_kit::base::dock::{Panel as BasePanel, PanelEvent};
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::dock::Panel;
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Disableable as _, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::TestSupportExt as _;
use gpui_kit::{
    div, px, AnyElement, App, AppContext as _, Context, Entity, EventEmitter, FocusHandle, Focusable, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render,
    SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};
use serde_json::{json, Value};

use crate::data::store::{AppEvent, Store};
use crate::domain::command::SessionCommand;
use crate::domain::session::{op_name, DisplayDoc, Show, ViewDoc};
use crate::screens::explore::ExploreVm;
use crate::ui::widgets::{icon_button, row, MenuSelect, PanelHeader, Section};

use super::panels::{side_panel, DockToggle};

/// Ops the Add menu offers: label and the node (defaults filled in by the spec).
pub fn presets() -> Vec<(&'static str, Value)> {
    vec![
        ("Band-pass 300–6000 Hz", json!({"op": "iir", "band": "bandpass", "cutoff": [300.0, 6000.0]})),
        ("High-pass 300 Hz", json!({"op": "iir", "band": "highpass", "cutoff": [300.0]})),
        ("Low-pass 6000 Hz", json!({"op": "iir", "band": "lowpass", "cutoff": [6000.0]})),
        ("Notch 60 Hz", json!({"op": "notch", "freq": 60.0, "q": 30.0})),
        ("Common median reference", json!({"op": "cmr"})),
        ("Common average reference", json!({"op": "car"})),
        ("Median (5 samples)", json!({"op": "median", "width": 5})),
        ("Gaussian smoothing (σ 2 samples)", json!({"op": "gaussian_smooth", "sigma_samples": 2.0})),
        ("Scale", json!({"op": "scale", "alpha": 1.0, "beta": 0.0})),
        ("Teager–Kaiser energy", json!({"op": "teager_kaiser"})),
    ]
}

/// Choices of the enum parameters, by field name.
fn choices(field: &str) -> Option<&'static [&'static str]> {
    Some(match field {
        "band" => &["lowpass", "highpass", "bandpass", "bandstop"],
        "design" => &["butterworth", "chebyshev1", "bessel"],
        "mode" => &["forward_backward", "forward"],
        "start" => &["rest", "steady_state"],
        "edge" => &["zeros", "odd", "reflect", "nearest"],
        _ => return None,
    })
}

/// `Band-pass 300–6000 Hz · order 3 · zero phase`
pub fn op_summary(node: &Node) -> String {
    let hz = |f: f64| if f >= 1000.0 && f % 1000.0 == 0.0 { format!("{} kHz", f / 1000.0) } else { format!("{f} Hz") };
    match node {
        Node::Iir { band, cutoff, order, mode, .. } => {
            let what = match (band, cutoff.as_slice()) {
                (Band::Bandpass, [lo, hi]) => format!("Band-pass {lo}–{}", hz(*hi)),
                (Band::Bandstop, [lo, hi]) => format!("Band-stop {lo}–{}", hz(*hi)),
                (Band::Highpass, [f]) => format!("High-pass {}", hz(*f)),
                (Band::Lowpass, [f]) => format!("Low-pass {}", hz(*f)),
                (b, _) => format!("{b:?}"),
            };
            let phase = if *mode == dspk_pipeline::Mode::ForwardBackward { "zero phase" } else { "causal" };
            format!("{what} · order {order} · {phase}")
        }
        Node::Notch { freq, q, .. } => format!("Notch {} (Q {q})", hz(*freq)),
        Node::Car {} => "Common average reference".into(),
        Node::Cmr {} => "Common median reference".into(),
        Node::Scale { alpha, beta } => format!("Scale ×{alpha} + {beta}"),
        Node::SubtractBaseline { baseline } => format!("Subtract {baseline}"),
        Node::Clamp { min, max } => format!("Clamp {min}…{max}"),
        Node::Sos { sections, .. } => format!("SOS filter ({} sections)", sections.len()),
        Node::Fir { taps, centered, .. } => format!("FIR ({} taps{})", taps.len(), if *centered { ", centred" } else { "" }),
        Node::GaussianSmooth { sigma_samples, .. } => format!("Gaussian smoothing σ {sigma_samples}"),
        Node::Median { width, .. } => format!("Median (width {width})"),
        Node::TeagerKaiser { .. } => "Teager–Kaiser energy".into(),
        other => op_name(other).to_string(),
    }
}

/// One editable number of the selected op: its place in the node's JSON (`cutoff`, index 1).
struct Field {
    key: String,
    index: Option<usize>,
    label: String,
    input: Entity<InputState>,
}

pub struct ProcessingPanel {
    store: Entity<Store>,
    dock: DockToggle,
    focus: FocusHandle,
    /// Inputs of the selected op, and what they were built for (view, op, node JSON).
    fields: Vec<Field>,
    fields_for: String,
    field_subs: Vec<Subscription>,
    error: Option<String>,
    _subs: Vec<Subscription>,
}

side_panel!(ProcessingPanel, "dspk-app.processing", "Processing");

impl ProcessingPanel {
    pub fn new(explore: Entity<ExploreVm>, dock: DockToggle, cx: &mut Context<Self>) -> Self {
        let store = explore.read(cx).store().clone();
        let subs = vec![cx.subscribe(&store, |this, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::Document(_) | AppEvent::Picked | AppEvent::JobChanged | AppEvent::RecordingChanged | AppEvent::SelectionChanged(_)) {
                this.error = None;
                cx.notify();
            }
        })];
        Self { store, dock, focus: cx.focus_handle(), fields: Vec::new(), fields_for: String::new(), field_subs: Vec::new(), error: None, _subs: subs }
    }

    fn header_extra(&self, cx: &App) -> Option<AnyElement> {
        let s = self.store.read(cx);
        let v = s.active_view()?;
        Some(div().text_xs().text_color(cx.theme().muted_foreground).truncate().child(v.name.clone()).into_any_element())
    }

    /// Inputs for the numbers of `node` (rebuilt when the view, the op or its values change).
    fn sync_fields(&mut self, view: &ViewDoc, op: &str, node: &Node, window: &mut Window, cx: &mut Context<Self>) {
        let value = serde_json::to_value(node).unwrap_or(Value::Null);
        let key = format!("{}/{op}/{value}", view.id);
        if key == self.fields_for {
            return;
        }
        self.fields_for = key;
        self.fields.clear();
        self.field_subs.clear();
        let Value::Object(map) = &value else { return };
        let add = |this: &mut Self, key: &str, index: Option<usize>, label: String, v: f64, window: &mut Window, cx: &mut Context<Self>| {
            let input = cx.new(|cx| InputState::new(window, cx).default_value(trim_number(v)));
            let sub = cx.subscribe(&input, |this, _, e: &InputEvent, cx| {
                if matches!(e, InputEvent::PressEnter { .. } | InputEvent::Blur) {
                    this.apply_fields(cx);
                }
            });
            this.field_subs.push(sub);
            this.fields.push(Field { key: key.to_string(), index, label, input });
        };
        for (k, v) in map {
            match v {
                Value::Number(n) => add(self, k, None, label_of(k), n.as_f64().unwrap_or(0.0), window, cx),
                Value::Array(a) if a.len() <= 2 && a.iter().all(Value::is_number) => {
                    for (i, n) in a.iter().enumerate() {
                        let label = match (k.as_str(), a.len(), i) {
                            ("cutoff", 2, 0) => "Low (Hz)".to_string(),
                            ("cutoff", 2, 1) => "High (Hz)".to_string(),
                            ("cutoff", _, _) => "Cutoff (Hz)".to_string(),
                            _ => format!("{} {}", label_of(k), i + 1),
                        };
                        add(self, k, Some(i), label, n.as_f64().unwrap_or(0.0), window, cx);
                    }
                }
                _ => {}
            }
        }
    }

    /// The numbers typed in, set on the selected op (a command); a value that does not parse,
    /// or parameters the spec rejects, leave the op unchanged and say why.
    fn apply_fields(&mut self, cx: &mut Context<Self>) {
        let (Some(view), Some(op)) = (self.store.read(cx).active_view().cloned(), self.store.read(cx).picked_op.clone()) else { return };
        let Some(node) = view.op(&op).map(|o| o.node.clone()) else { return };
        let mut value = serde_json::to_value(&node).unwrap_or(Value::Null);
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
        self.set_node(&view, &op, value, cx);
    }

    fn set_node(&mut self, view: &ViewDoc, op: &str, value: Value, cx: &mut Context<Self>) {
        match serde_json::from_value::<Node>(value) {
            Ok(node) => {
                if view.op(op).is_some_and(|o| o.node == node) {
                    return;
                }
                let result = self.store.update(cx, |s, cx| s.execute(SessionCommand::SetOp { view: view.id.clone(), op: op.to_string(), node }, cx));
                self.error = result.err().map(|e| e.to_string());
            }
            Err(e) => self.error = Some(e.to_string()),
        }
        cx.notify();
    }

    /// A new enum value (band, mode, …) for the selected op; a band change keeps the cutoffs
    /// it can (one for low / high pass, two for band pass / stop).
    fn set_choice(&mut self, key: &str, choice: &str, cx: &mut Context<Self>) {
        let (Some(view), Some(op)) = (self.store.read(cx).active_view().cloned(), self.store.read(cx).picked_op.clone()) else { return };
        let Some(node) = view.op(&op).map(|o| o.node.clone()) else { return };
        let mut value = serde_json::to_value(&node).unwrap_or(Value::Null);
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
        self.set_node(&view, &op, value, cx);
    }

    fn body(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = cx.theme();
        let (muted, danger, accent, border) = (t.muted_foreground, t.danger, t.accent, t.border);
        let s = self.store.read(cx);
        let Some(view) = s.active_view().cloned() else {
            return v_flex().id("processing-panel").test_support().size_full().p_4().text_sm().text_color(muted).child("Open a recording: its view and processing show here.").into_any_element();
        };
        let views: Vec<(String, String)> = s.doc.views.iter().map(|v| (v.id.clone(), v.name.clone())).collect();
        let source = s.view_source(&view);
        let total = s.sources().map_or(0, |ss| ss.entry(&source).channels);
        let picked = s.picked_channel;
        let picked_op = s.picked_op.clone().filter(|op| view.op(op).is_some());
        let label = picked.and_then(|c| s.sources().map(|ss| ss.get(&source)).and_then(|d| d.channel_label(c).map(str::to_string)));
        let window_samples = {
            let rate = s.sources().map_or(0.0, |ss| ss.entry(&source).sample_rate.rate_hz());
            (s.timeline.visible_window_sec * rate) as u64
        };
        let covers = s.preview_covers(&source, window_samples);
        let job = s.job.as_ref().map(|j| (j.label.clone(), j.fraction, j.line.clone()));
        let selection: Vec<usize> = s.selection(&source).to_vec();
        let in_view = view.recording.channels.as_ref().map_or(total, Vec::len);
        let id = view.id.clone();

        // View: which one, copy, close
        let names: Vec<SharedString> = views.iter().map(|(_, n)| n.clone().into()).collect();
        let ids: Vec<String> = views.iter().map(|(i, _)| i.clone()).collect();
        let selected = ids.iter().position(|i| *i == id);
        let store = self.store.clone();
        let view_pick = MenuSelect::new("processing-view", names, selected, move |i, _, cx| {
            let view = ids[i].clone();
            let _ = store.update(cx, |s, cx| s.execute(SessionCommand::SelectView { view }, cx));
        })
        .full_width();
        let copy = {
            let (store, id, sel) = (self.store.clone(), id.clone(), selection.clone());
            Button::new("copy-view")
                .outline()
                .small()
                .icon(IconName::Copy)
                .label("Copy with shown channels")
                .tooltip("A new view of the channels shown now, with this view's chain: edit it for those channels only")
                .on_click(move |_, _, cx| {
                    let channels = (!sel.is_empty()).then(|| sel.clone());
                    let _ = store.update(cx, |s, cx| s.execute(SessionCommand::CopyView { view: id.clone(), channels, name: None }, cx));
                })
        };
        let close = {
            let (store, id) = (self.store.clone(), id.clone());
            icon_button("close-view", IconName::Close, "Close this view (the recording stays open)")
                .on_click(move |_, _, cx| {
                    let _ = store.update(cx, |s, cx| s.execute(SessionCommand::CloseView { view: id.clone() }, cx));
                })
        };
        let recording = format!(
            "{} · {}",
            view.recording.path.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned()),
            match &view.recording.channels {
                Some(c) => format!("{} of {total} channels", c.len()),
                None => format!("all {total} channels"),
            }
        );

        // Raw / processed
        let show_store = self.store.clone();
        let show = |which: Show, text: &'static str| {
            let (store, id, display) = (show_store.clone(), id.clone(), view.display.clone());
            Button::new(SharedString::from(format!("show-{text}")))
                .small()
                .label(text)
                .map(|b| if display.show == which { b.primary() } else { b.outline() })
                .on_click(move |_, _, cx| {
                    let display = DisplayDoc { show: which, ..display.clone() };
                    let _ = store.update(cx, |s, cx| s.execute(SessionCommand::SetDisplay { view: id.clone(), display }, cx));
                })
        };
        let preview_note = match (view.display.show, covers) {
            (Show::Raw, _) => Some("Showing the recording as stored."),
            (Show::Processed, Some(false)) => Some("Zoomed out past the preview: showing raw. Zoom in to see the processing."),
            (Show::Processed, None) if view.ops.iter().any(|o| o.enabled) => Some("Preview off for this recording."),
            _ => None,
        };

        // The channel
        let channel_line = match picked {
            Some(c) => format!("Ch {c}{}", label.map(|l| format!(" · {l}")).unwrap_or_default()),
            None => "Click a channel (list or traces) to show its processing".to_string(),
        };

        // The op tree
        let n_ops = view.ops.len();
        let tree = v_flex()
            .gap_0p5()
            .child(h_flex().gap_2().h(px(26.)).child(div().size(px(8.)).rounded_full().bg(muted)).child(div().text_sm().child("Recording (raw)")))
            .children(view.ops.iter().enumerate().map(|(i, op)| {
                let is_sel = picked_op.as_deref() == Some(op.id.as_str());
                let (store, id, op_id, enabled) = (self.store.clone(), id.clone(), op.id.clone(), op.enabled);
                let pick = {
                    let (store, op_id) = (self.store.clone(), op.id.clone());
                    move |_: &gpui_kit::ClickEvent, _: &mut Window, cx: &mut App| store.update(cx, |s, cx| s.pick_op(Some(op_id.clone()), cx))
                };
                let mv = |dir: i64, icon: IconName, tip: &'static str| {
                    let (store, id, op_id) = (self.store.clone(), view.id.clone(), op.id.clone());
                    let to = (i as i64 + dir).clamp(0, n_ops as i64 - 1) as usize;
                    icon_button(SharedString::from(format!("op-{}-{tip}", op.id)), icon, tip).xsmall().disabled(to == i).on_click(move |_, _, cx| {
                        let _ = store.update(cx, |s, cx| s.execute(SessionCommand::MoveOp { view: id.clone(), op: op_id.clone(), index: to }, cx));
                    })
                };
                h_flex()
                    .id(SharedString::from(format!("op-row-{}", op.id)))
                    .h(px(28.))
                    .pl(px(14.))
                    .pr_1()
                    .gap_1()
                    .rounded_md()
                    .when(is_sel, |d| d.bg(accent))
                    .border_l_1()
                    .border_color(border)
                    .child(Checkbox::new(SharedString::from(format!("op-on-{}", op.id))).checked(enabled).on_click(move |on: &bool, _, cx| {
                        let _ = store.update(cx, |s, cx| s.execute(SessionCommand::EnableOp { view: id.clone(), op: op_id.clone(), enabled: *on }, cx));
                    }))
                    .child(div().flex_1().min_w_0().text_sm().truncate().cursor_pointer().when(!op.enabled, |d| d.text_color(muted)).child(op_summary(&op.node)).id(SharedString::from(format!("op-name-{}", op.id))).on_click(pick))
                    .child(mv(-1, IconName::ArrowUp, "Earlier"))
                    .child(mv(1, IconName::ArrowDown, "Later"))
                    .child({
                        let (store, id, op_id) = (self.store.clone(), view.id.clone(), op.id.clone());
                        icon_button(SharedString::from(format!("op-del-{}", op.id)), IconName::Close, "Remove").xsmall().on_click(move |_, _, cx| {
                            let _ = store.update(cx, |s, cx| s.execute(SessionCommand::RemoveOp { view: id.clone(), op: op_id.clone() }, cx));
                        })
                    })
            }));
        let add = {
            let (store, id) = (self.store.clone(), id.clone());
            Button::new("add-op").outline().small().icon(IconName::Plus).label("Add op").dropdown_menu(move |menu, _, _| {
                let mut menu = menu;
                for (label, value) in presets() {
                    let (store, id) = (store.clone(), id.clone());
                    menu = menu.item(PopupMenuItem::new(label).on_click(move |_, _, cx| {
                        let Ok(node) = serde_json::from_value::<Node>(value.clone()) else { return };
                        store.update(cx, |s, cx| {
                            if let Ok(_) = s.execute(SessionCommand::AddOp { view: id.clone(), node, index: None, id: None }, cx) {
                                let new = s.active_view().and_then(|v| v.ops.last()).map(|o| o.id.clone());
                                s.pick_op(new, cx);
                            }
                        });
                    }));
                }
                menu
            })
        };

        // Parameters of the selected op
        let params = picked_op.as_ref().and_then(|op| view.op(op).cloned()).map(|op| {
            self.sync_fields(&view, &op.id, &op.node, window, cx);
            let value = serde_json::to_value(&op.node).unwrap_or(Value::Null);
            let mut section = Section::new(format!("Parameters · {}", op_summary(&op.node)));
            for f in &self.fields {
                section = section.child(row(f.label.clone(), div().w(px(110.)).child(Input::new(&f.input).small()), cx));
            }
            if let Value::Object(map) = &value {
                for (k, v) in map {
                    if let (Some(opts), Some(cur)) = (choices(k), v.as_str()) {
                        let this = cx.entity().downgrade();
                        let key = k.clone();
                        let names: Vec<SharedString> = opts.iter().map(|o| SharedString::from(o.replace('_', " "))).collect();
                        let sel = opts.iter().position(|o| *o == cur);
                        section = section.child(row(
                            label_of(k),
                            MenuSelect::new(SharedString::from(format!("param-{k}")), names, sel, move |i, _, cx| {
                                let _ = this.update(cx, |p, cx| p.set_choice(&key, opts[i], cx));
                            }),
                            cx,
                        ));
                    } else if let Some(b) = v.as_bool() {
                        let this = cx.entity().downgrade();
                        let key = k.clone();
                        section = section.child(row(
                            label_of(k),
                            Checkbox::new(SharedString::from(format!("param-{k}"))).checked(b).on_click(move |on: &bool, _, cx| {
                                let on = *on;
                                let _ = this.update(cx, |p, cx| {
                                    let (Some(view), Some(op)) = (p.store.read(cx).active_view().cloned(), p.store.read(cx).picked_op.clone()) else { return };
                                    let Some(node) = view.op(&op).map(|o| o.node.clone()) else { return };
                                    let mut value = serde_json::to_value(&node).unwrap_or(Value::Null);
                                    value[key.as_str()] = json!(on);
                                    p.set_node(&view, &op, value, cx);
                                });
                            }),
                            cx,
                        ));
                    } else if v.as_array().is_some_and(|a| a.len() > 2) {
                        section = section.child(row(label_of(k), div().text_xs().text_color(muted).child(format!("{} values (edit in the session file)", v.as_array().map_or(0, Vec::len))), cx));
                    }
                }
            }
            section.child(div().text_xs().text_color(muted).child(format!("Applies to the {in_view} channels of this view. Enter applies a value.")))
        });

        // Output
        let output = match &job {
            Some((label, fraction, line)) => h_flex()
                .gap_2()
                .child(div().flex_1().min_w_0().text_xs().truncate().child(match fraction {
                    Some(f) => format!("{label} · {:.0} %", f * 100.0),
                    None => format!("{label} · {line}"),
                }))
                .child(Button::new("cancel-job").small().outline().label("Cancel").on_click(cx.listener(|this, _, _, cx| this.store.update(cx, |s, cx| s.cancel_job(cx)))))
                .into_any_element(),
            None => Button::new("write-view")
                .small()
                .outline()
                .icon(IconName::ArrowDown)
                .label("Write processed recording")
                .tooltip("The recording through this view's enabled ops, to <recording>.<view>.processed.zarr (float32, compressed)")
                .disabled(!view.ops.iter().any(|o| o.enabled))
                .on_click(cx.listener(|this, _, _, cx| this.store.update(cx, |s, cx| s.write_active_view(cx))))
                .into_any_element(),
        };

        v_flex()
            .id("processing-panel")
            .test_support()
            .size_full()
            .gap_3()
            .p_3()
            .overflow_y_scroll()
            .child(
                Section::new("View")
                    .child(view_pick)
                    .child(div().text_xs().text_color(muted).child(recording))
                    .child(h_flex().gap_1().child(copy).child(div().flex_1()).child(close)),
            )
            .child(
                Section::new("Processing")
                    .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(channel_line))
                    .child(h_flex().gap_1().child(show(Show::Processed, "Processed")).child(show(Show::Raw, "Raw")))
                    .children(preview_note.map(|n| div().text_xs().text_color(muted).child(n)))
                    .child(tree)
                    .child(add),
            )
            .children(params)
            .children(self.error.clone().map(|e| div().text_xs().text_color(danger).child(e)))
            .child(Section::new("Output").child(output))
            .into_any_element()
    }
}

/// `sigma_samples` → `Sigma samples`
fn label_of(key: &str) -> String {
    let mut s = key.replace('_', " ");
    if let Some(c) = s.get_mut(0..1) {
        c.make_ascii_uppercase();
    }
    s
}

/// `300` rather than `300.0`, at most 6 decimals.
fn trim_number(v: f64) -> String {
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
    fn test_presets_are_valid_view_ops() {
        for (label, value) in presets() {
            let node: Node = serde_json::from_value(value).unwrap_or_else(|e| panic!("{label}: {e}"));
            assert!(node.stage().is_ok(), "{label}");
            assert!(!op_summary(&node).is_empty());
        }
    }

    #[test]
    fn test_labels_and_numbers() {
        assert_eq!(label_of("sigma_samples"), "Sigma samples");
        assert_eq!(trim_number(300.0), "300");
        assert_eq!(trim_number(0.25), "0.25");
    }
}
