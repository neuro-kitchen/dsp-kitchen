//! Processing (right dock): the active view's op tree for the channel picked in Channels or on
//! the traces. The view's chain applies to every channel of its recording (one grouped array);
//! other processing for some channels: copy the view with those channels. Selecting an op shows
//! its parameters; an edit is a command on the session document, previewed live on the visible
//! window; writing the processed recording is a separate step. Plugins add their sections (e.g.
//! the sorting of the view).


use std::rc::Rc;

use dspk_pipeline::{Band, Node};
use gpui_kit::base::dock::{Panel as BasePanel, PanelEvent};
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::dock::Panel;
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Disableable as _, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::TestSupportExt as _;
use gpui_kit::{
    div, px, AnyElement, App, AppContext as _, Context, Entity, EventEmitter, FocusHandle, Focusable, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render,
    SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};
use serde_json::Value;

use crate::ui::param_editor::{Commit, ParamEditor};

use crate::data::store::{AppEvent, Store};
use crate::domain::command::SessionCommand;
use crate::domain::session::{op_name, DisplayDoc, Show};
use crate::screens::explore::ExploreVm;
use crate::ui::widgets::{icon_button, PanelHeader, Section};

use super::panels::{side_panel, DockToggle};

/// A family of the ops a view can run (the Pipeline catalog's, previewed live): its name and
/// each op's label and default node.
pub struct ViewOps {
    pub name: &'static str,
    pub ops: Vec<(&'static str, Value)>,
}

/// The catalog's ops a view previews (resampling, channel maps and outputs are the Pipeline's).
pub fn view_ops() -> Vec<ViewOps> {
    crate::screens::pipeline::catalog::catalog()
        .into_iter()
        .map(|c| ViewOps {
            name: c.name,
            ops: c
                .items
                .into_iter()
                .filter_map(|it| match it.template {
                    crate::screens::pipeline::catalog::Template::Op(v) => {
                        let node: Node = serde_json::from_value(v.clone()).ok()?;
                        (matches!(node.stage(), Ok(Some(_))) || node.whiten_fit().is_some()).then_some((it.label, v))
                    }
                    _ => None,
                })
                .collect(),
        })
        .filter(|f| !f.ops.is_empty())
        .collect()
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
        Node::Rectify {} => "Rectify |x|".into(),
        Node::Square {} => "Square x²".into(),
        Node::Sqrt {} => "Square root √x".into(),
        Node::Log10 {} => "Log₁₀ |x|".into(),
        Node::Laplacian { grid: [r, c] } => format!("Surface Laplacian, grid {r} × {c}"),
        Node::Whiten { fit_s, chunks, .. } => format!("Whitening (ZCA, fitted on {fit_s} s in {chunks} chunks)"),
        other => op_name(other).to_string(),
    }
}

pub struct ProcessingPanel {
    store: Entity<Store>,
    dock: DockToggle,
    focus: FocusHandle,
    /// The selected op's parameters.
    params: Entity<ParamEditor>,
    _subs: Vec<Subscription>,
}

side_panel!(ProcessingPanel, "dspk-app.processing", "Processing");

impl ProcessingPanel {
    pub fn new(explore: Entity<ExploreVm>, dock: DockToggle, cx: &mut Context<Self>) -> Self {
        let store = explore.read(cx).store().clone();
        let subs = vec![cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::Document(_) | AppEvent::Picked | AppEvent::JobChanged | AppEvent::RecordingChanged | AppEvent::SelectionChanged(_)) {
                cx.notify();
            }
        })];
        let params = cx.new(|_| ParamEditor::new());
        Self { store, dock, focus: cx.focus_handle(), params, _subs: subs }
    }

    fn header_extra(&self, cx: &App) -> Option<AnyElement> {
        let s = self.store.read(cx);
        let v = s.active_view()?;
        Some(div().text_xs().text_color(cx.theme().muted_foreground).truncate().child(v.name.clone()).into_any_element())
    }

    fn body(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let sections = self.sections(window, cx);
        v_flex().id("processing-panel").test_support().size_full().gap_3().p_3().overflow_y_scroll().children(sections).into_any_element()
    }

    /// The active view's processing (raw / processed, the op chain, the picked op's parameters)
    /// and its output: the Inspector's lower half.
    pub fn sections(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let t = cx.theme();
        let (muted, accent, border) = (t.muted_foreground, t.accent, t.border);
        let s = self.store.read(cx);
        let Some(view) = s.active_view().cloned() else {
            return vec![div().text_sm().text_color(muted).child("Open a recording: its view's processing shows here.").into_any_element()];
        };
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
        let in_view = view.recording.channels.as_ref().map_or(total, Vec::len);
        let id = view.id.clone();

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
            // The Pipeline's families (one catalog for both), with the ops a view previews; an op
            // with presets opens them (its default first)
            Button::new("add-op").outline().small().icon(IconName::Plus).label("Add op").dropdown_menu(move |menu, window, cx| {
                let add = |store: Entity<Store>, id: String, value: Value| {
                    move |_: &gpui_kit::ClickEvent, _: &mut Window, cx: &mut App| {
                        let Ok(node) = serde_json::from_value::<Node>(value.clone()) else { return };
                        store.update(cx, |s, cx| {
                            if s.execute(SessionCommand::AddOp { view: id.clone(), node, index: None, id: None }, cx).is_ok() {
                                let new = s.active_view().and_then(|v| v.ops.last()).map(|o| o.id.clone());
                                s.pick_op(new, cx);
                            }
                        });
                    }
                };
                let mut menu = menu;
                for family in view_ops() {
                    let (store, id) = (store.clone(), id.clone());
                    menu = menu.submenu(family.name, window, cx, move |m, window, cx| {
                        let mut m = m;
                        for (label, value) in &family.ops {
                            let node: Node = serde_json::from_value(value.clone()).expect("catalog ops are valid");
                            let presets = crate::screens::pipeline::catalog::presets(&node);
                            let (store, id, value) = (store.clone(), id.clone(), value.clone());
                            if presets.is_empty() {
                                m = m.item(PopupMenuItem::new(*label).on_click(add(store, id, value)));
                            } else {
                                m = m.submenu(*label, window, cx, move |sub, _, _| {
                                    let mut sub = sub.item(PopupMenuItem::new("Default").on_click(add(store.clone(), id.clone(), value.clone()))).separator();
                                    for (preset, v) in &presets {
                                        sub = sub.item(PopupMenuItem::new(*preset).on_click(add(store.clone(), id.clone(), v.clone())));
                                    }
                                    sub
                                });
                            }
                        }
                        m
                    });
                }
                menu
            })
        };

        // Parameters of the selected op
        let params = picked_op.as_ref().and_then(|op| view.op(op).cloned()).map(|op| {
            let commit: Commit = {
                let (store, view, op) = (self.store.clone(), view.id.clone(), op.id.clone());
                Rc::new(move |node, cx| store.update(cx, |s, cx| s.execute(SessionCommand::SetOp { view: view.clone(), op: op.clone(), node }, cx)).map(|_| ()).map_err(|e| e.to_string()))
            };
            self.params.update(cx, |p, cx| p.set_target(&format!("{}/{}", view.id, op.id), &op.node, commit, window, cx));
            Section::new("Parameters")
                .child(self.params.clone())
                .child(div().text_xs().text_color(muted).child(format!("Applies to the {in_view} channels of this view. Enter applies a value.")))
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

        let mut out = vec![
                Section::new("Processing")
                    .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(channel_line))
                    .child(h_flex().gap_1().child(show(Show::Processed, "Processed")).child(show(Show::Raw, "Raw")))
                    .children(preview_note.map(|n| div().text_xs().text_color(muted).child(n)))
                    .child(tree)
                    .child(add)
                    .into_any_element(),
        ];
        out.extend(params.map(IntoElement::into_any_element));
        out.push(Section::new("Output").child(output).into_any_element());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every op Explore offers previews, and its presets are ops of its type; the new
    /// per-sample ops are there.
    #[test]
    fn test_view_ops_preview_with_their_presets() {
        let mut labels = Vec::new();
        for family in view_ops() {
            for (label, value) in family.ops {
                let node: Node = serde_json::from_value(value).unwrap_or_else(|e| panic!("{label}: {e}"));
                assert!(matches!(node.stage(), Ok(Some(_))) || node.whiten_fit().is_some(), "{label}");
                assert!(!op_summary(&node).is_empty());
                for (preset, v) in crate::screens::pipeline::catalog::presets(&node) {
                    let p: Node = serde_json::from_value(v).unwrap_or_else(|e| panic!("{preset}: {e}"));
                    assert_eq!(op_name(&p), op_name(&node), "{preset}");
                }
                labels.push(label);
            }
        }
        for op in ["IIR", "Rectify", "Square", "Square root", "Log₁₀", "Teager–Kaiser", "Laplacian", "Whiten"] {
            assert!(labels.contains(&op), "{op} in {labels:?}");
        }
        assert!(!labels.contains(&"Resample"), "resampling is not previewed");
    }
}
