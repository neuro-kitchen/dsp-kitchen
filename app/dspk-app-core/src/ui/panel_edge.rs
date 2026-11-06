//! Resizable side panels: a handle on a panel's inner edge (drag it; the pill shows while the
//! pointer is on it) and the listener the workspace root runs while it is dragged. Widths are
//! remembered per workspace and panel in the preferences (`panel_widths`).

use gpui_kit::component::resizable::{resize_handle, HandleEdge};
use gpui_kit::{div, App, AppContext as _, Axis, Context, DragMoveEvent, Entity, IntoElement, Render, SharedString, Window};

use crate::data::store::Store;

/// Narrowest and widest a side panel gets (logical px).
pub const MIN_PANEL: f32 = 180.;
pub const MAX_PANEL: f32 = 760.;

/// Where a panel sits: its handle is on the opposite edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
    /// A bottom panel (its height is resized).
    Bottom,
}

/// What a panel-edge drag carries: the panel's key in the preferences, its side, and how far
/// from the root's edge the panel starts (e.g. an icon bar).
#[derive(Debug, Clone)]
pub struct PanelResize {
    pub key: &'static str,
    pub side: Side,
    pub offset: f32,
}

impl Render for PanelResize {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
    }
}

/// The handle for a panel: a child of the panel's container (which must be `relative`).
pub fn panel_edge(key: &'static str, side: Side, offset: f32) -> impl IntoElement {
    let (axis, edge) = match side {
        Side::Left => (Axis::Horizontal, HandleEdge::Trailing),
        Side::Right => (Axis::Horizontal, HandleEdge::Leading),
        Side::Bottom => (Axis::Vertical, HandleEdge::Leading),
    };
    resize_handle::<PanelResize, PanelResize>(SharedString::from(format!("panel-edge-{key}")), axis)
        .inside(edge)
        .on_drag(PanelResize { key, side, offset }, |d, _, _, cx| cx.new(|_| d.clone()))
}

/// The new size of the dragged panel, from the pointer and the root's bounds.
pub fn follow_panel_drag(store: &Entity<Store>, e: &DragMoveEvent<PanelResize>, cx: &mut App) {
    let r = e.drag(cx).clone();
    let (b, p) = (e.bounds, e.event.position);
    let size = match r.side {
        Side::Left => (p.x - b.origin.x).as_f32() - r.offset,
        Side::Right => (b.origin.x + b.size.width - p.x).as_f32() - r.offset,
        Side::Bottom => (b.origin.y + b.size.height - p.y).as_f32() - r.offset,
    };
    let max = match r.side {
        Side::Bottom => (b.size.height.as_f32() * 0.8).max(MIN_PANEL),
        _ => MAX_PANEL.min(b.size.width.as_f32() * 0.45).max(MIN_PANEL),
    };
    let min = if r.side == Side::Bottom { 60.0 } else { MIN_PANEL };
    store.update(cx, |s, cx| s.set_panel_width(r.key, size.clamp(min, max), cx));
}
