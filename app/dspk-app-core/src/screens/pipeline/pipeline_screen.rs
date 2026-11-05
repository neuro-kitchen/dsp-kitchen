//! The Pipeline workspace: the active view's processing drawn as an operator chain, the way a
//! Kotlin Flow reads — `flowOf(recording)`, then one operator per op (`.bandpass(300, 6000)`,
//! `.cmr()`), then `.collect(write)` — each a card, with the data's shape between them (channels ×
//! rate · sample type). Clicking an operator selects it: the Processing panel beside the chain
//! edits it. It is the same op chain as Explore's Processing (the session document's view).

use dspk_pipeline::{Band, Mode, Node};
use gpui_kit::component::button::Button;
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenu, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Disableable as _, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, px, App, AppContext as _, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString, StatefulInteractiveElement as _,
    Styled as _, Subscription, Window,
};

use crate::data::jobs::processed_path;
use crate::data::store::{AppEvent, Store};
use crate::domain::command::SessionCommand;
use crate::screens::explore::components::panels::DockToggle;
use crate::screens::explore::components::processing::{op_summary, presets, ProcessingPanel};
use crate::screens::explore::ExploreVm;
use crate::ui::widgets::EmptyState;

/// Width of the chain column and of the side panel (logical px).
const CHAIN_W: f32 = 560.;
const SIDE_W: f32 = 328.;

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

pub struct PipelineView {
    store: Entity<Store>,
    processing: Entity<ProcessingPanel>,
    _sub: Subscription,
}

impl PipelineView {
    pub fn new(store: Entity<Store>, explore: Entity<ExploreVm>, cx: &mut Context<Self>) -> Self {
        let toggle = DockToggle { placement: gpui_kit::base::dock::DockPlacement::Right, open: std::rc::Rc::new(|_| true), toggle: std::rc::Rc::new(|_, _| {}) };
        let processing = cx.new(|cx| ProcessingPanel::new(explore, toggle, cx));
        let sub = cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::Document(_) | AppEvent::Picked | AppEvent::RecordingChanged | AppEvent::JobChanged) {
                cx.notify();
            }
        });
        Self { store, processing, _sub: sub }
    }
}

/// A card of the chain: the operator call, what it does, and its controls.
fn card(id: impl Into<SharedString>, call: String, detail: String, selected: bool, cx: &App) -> gpui_kit::Stateful<gpui_kit::Div> {
    let t = cx.theme();
    div()
        .id(gpui_kit::ElementId::Name(id.into()))
        .w_full()
        .p_4()
        .rounded_lg()
        .border_1()
        .border_color(if selected { t.primary } else { t.border })
        .bg(t.background)
        .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(call))
        .child(div().mt_1().text_xs().text_color(t.muted_foreground).truncate().child(detail))
}

/// The arrow between two cards, with the data's shape.
fn connector(shape: String, cx: &App) -> impl IntoElement {
    let muted = cx.theme().muted_foreground;
    h_flex().h(px(40.)).pl_6().gap_3().child(div().w(px(1.)).h_full().bg(cx.theme().border)).child(div().text_xs().text_color(muted).child(shape))
}

impl Render for PipelineView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let s = self.store.read(cx);
        let Some(view) = s.active_view().cloned() else {
            return div()
                .size_full()
                .child(EmptyState::new(gpui_kit::assets::IconName::Workflow, "No view yet", "Open a recording: its processing shows here as a chain of operators."))
                .into_any_element();
        };
        let source = s.view_source(&view);
        let entry = s.sources().map(|ss| ss.entry(&source).clone());
        let picked = s.picked_op.clone();
        let running = s.job.is_some();
        let (channels, rate, format) = entry.as_ref().map_or((0, 0.0, "?"), |e| (e.channels, e.sample_rate.rate_hz(), e.format.name()));
        let channels = view.recording.channels.as_ref().map_or(channels, Vec::len);
        let rate = if rate >= 1000.0 { format!("{:.1} kHz", rate / 1000.0) } else { format!("{rate:.0} Hz") };
        let shape = |ty: &str| format!("{channels} ch × {rate} · {ty}");
        let file = view.recording.path.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        let output = processed_path(&view.recording.path, &view.id);
        let out_name = output.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        let enabled = view.ops.iter().any(|o| o.enabled);

        let mut chain = v_flex().w(px(CHAIN_W)).child(card("flow-source", "flowOf(recording)".into(), format!("{file} · {}", shape(format)), false, cx));
        let mut ty = format.to_string();
        for op in &view.ops {
            chain = chain.child(connector(shape(&ty), cx));
            if op.enabled {
                // Processing runs in float32 on the device
                ty = "float32".into();
            }
            let (store, id, op_id, on) = (self.store.clone(), view.id.clone(), op.id.clone(), op.enabled);
            let pick = {
                let (store, op_id) = (self.store.clone(), op.id.clone());
                move |_: &gpui_kit::ClickEvent, _: &mut Window, cx: &mut App| store.update(cx, |s, cx| s.pick_op(Some(op_id.clone()), cx))
            };
            let c = card(format!("flow-op-{}", op.id), flow_call(&op.node), op_summary(&op.node), picked.as_deref() == Some(op.id.as_str()), cx)
                .cursor_pointer()
                .when(!op.enabled, |d| d.opacity(0.5))
                .on_click(pick)
                .child(h_flex().mt_2().gap_2().child(Checkbox::new(SharedString::from(format!("flow-on-{}", op.id))).checked(on).label("Enabled").on_click(move |v: &bool, _, cx| {
                    let _ = store.update(cx, |s, cx| s.execute(SessionCommand::EnableOp { view: id.clone(), op: op_id.clone(), enabled: *v }, cx));
                })));
            chain = chain.child(c);
        }
        let add = {
            let (store, id) = (self.store.clone(), view.id.clone());
            Button::new("flow-add").outline().small().icon(IconName::Plus).label("Add operator").dropdown_menu(move |menu: PopupMenu, _, _| {
                let mut menu = menu;
                for (label, value) in presets() {
                    let (store, id) = (store.clone(), id.clone());
                    menu = menu.item(PopupMenuItem::new(label).on_click(move |_, _, cx| {
                        let Ok(node) = serde_json::from_value::<Node>(value.clone()) else { return };
                        store.update(cx, |s, cx| {
                            if s.execute(SessionCommand::AddOp { view: id.clone(), node, index: None, id: None }, cx).is_ok() {
                                let new = s.active_view().and_then(|v| v.ops.last()).map(|o| o.id.clone());
                                s.pick_op(new, cx);
                            }
                        });
                    }));
                }
                menu
            })
        };
        let write = Button::new("flow-write")
            .small()
            .outline()
            .label(if running { "Running…" } else { "Write" })
            .disabled(running || !enabled)
            .on_click({
                let store = self.store.clone();
                move |_, _, cx| store.update(cx, |s, cx| s.write_active_view(cx))
            });
        chain = chain
            .child(h_flex().h(px(48.)).pl_6().items_center().child(add))
            .child(connector(shape(&ty), cx))
            .child(card("flow-write-card", ".collect(write)".into(), format!("{out_name} · Zarr, compressed"), false, cx).child(h_flex().mt_2().child(write)));

        let t = cx.theme();
        h_flex()
            .id("pipeline-workspace")
            .size_full()
            .child(
                div()
                    .id("pipeline-chain")
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .overflow_y_scroll()
                    .child(
                        v_flex()
                            .items_center()
                            .py_8()
                            .gap_4()
                            .child(v_flex().w(px(CHAIN_W)).gap_1().child(div().text_lg().font_weight(FontWeight::SEMIBOLD).child(view.name.clone())).child(div().text_xs().text_color(t.muted_foreground).child("The view's processing as a chain of operators: click one to edit it on the right.")))
                            .child(chain),
                    ),
            )
            .child(div().w(px(SIDE_W)).flex_none().h_full().border_l_1().border_color(t.border).child(self.processing.clone()))
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_calls() {
        let bp: Node = serde_json::from_value(serde_json::json!({"op": "iir", "band": "bandpass", "cutoff": [300.0, 6000.0]})).unwrap();
        assert!(flow_call(&bp).starts_with(".bandpass(300.0, 6000.0, order = "));
        assert_eq!(flow_call(&Node::Cmr {}), ".cmr()");
    }
}
