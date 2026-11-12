//! Small pieces used by several views. The shared ones (a dropdown of fixed options, a titled
//! section, a labelled row, a slider, a stepper, an empty state, an icon button, the accent picker)
//! come from `kitchen_ui::widgets`; this app's own (side-panel headers and rails, workspace panes)
//! are here. Plain data in, no store.

use std::rc::Rc;

use gpui_kit::component::{h_flex, v_flex, IconName};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{div, px, AnyElement, App, Div, ElementId, FontWeight, IntoElement, ParentElement, RenderOnce, SharedString, Styled as _, Window};
pub use kitchen_ui::widgets::{icon_button, row, scrub_label, slider, stepper, swatch, AccentChanged, AccentPicker, EmptyState, MenuSelect, OnSlide, OnStep, Section, SCRUB_PX};

use super::theme::{self, color};

/// A region of a workspace (a tool column, the views, a side panel, a rail) as a card on the
/// chrome: rounded, its content clipped, on the panel colour. The row holding the cards sets
/// them [`theme::PANE_GAP`] apart: `.p(px(PANE_GAP)).gap(px(PANE_GAP)).bg(color::chrome())`.
pub fn pane() -> Div {
    div().rounded(px(theme::RADIUS_LG)).overflow_hidden().bg(color::panel())
}

/// Which window edge a side panel sits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    Left,
    Right,
    Bottom,
}

type OnToggle = Rc<dyn Fn(&mut Window, &mut App)>;

/// A side panel's header: its title and, at the window edge it sits on, the button that hides it
/// (so the button is always where the panel is, never beside something else).
#[derive(IntoElement)]
pub struct PanelHeader {
    id: ElementId,
    title: SharedString,
    edge: Edge,
    extra: Option<AnyElement>,
    open: bool,
    on_hide: OnToggle,
}

/// Height of a panel header (also what a closed bottom dock keeps).
pub const HEADER_HEIGHT: f32 = 29.;

impl PanelHeader {
    pub fn new(id: impl Into<ElementId>, title: impl Into<SharedString>, edge: Edge, on_hide: impl Fn(&mut Window, &mut App) + 'static) -> Self {
        Self { id: id.into(), title: title.into(), edge, extra: None, open: true, on_hide: Rc::new(on_hide) }
    }

    /// Whether the panel is shown (a folded bottom panel keeps its header, with a show button).
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }

    /// Something shown after the title (a count, a readout).
    pub fn extra(mut self, extra: impl IntoElement) -> Self {
        self.extra = Some(extra.into_any_element());
        self
    }
}

impl RenderOnce for PanelHeader {
    fn render(self, _: &mut Window, _: &mut App) -> impl IntoElement {
        let Self { id, title, edge, extra, open, on_hide } = self;
        let (icon, tip) = match (edge, open) {
            (Edge::Left, _) => (IconName::PanelLeftClose, format!("Hide {title}")),
            (Edge::Right, _) => (IconName::PanelRightClose, format!("Hide {title}")),
            (Edge::Bottom, true) => (IconName::ChevronDown, format!("Hide {title}")),
            (Edge::Bottom, false) => (IconName::ChevronUp, format!("Show {title}")),
        };
        let button = icon_button(id, icon, tip).on_click(move |_, window, cx| on_hide(window, cx));
        let title = div().text_sm().font_weight(FontWeight::SEMIBOLD).child(title);
        let left = edge == Edge::Left;
        let (lead, trail) = if left { (Some(button), None) } else { (None, Some(button)) };
        h_flex()
            .flex_none()
            .h(px(HEADER_HEIGHT))
            .px_1()
            .gap_1p5()
            .border_b_1()
            .border_color(color::border())
            .children(lead)
            .when(!left, |d| d.pl_2())
            .child(title)
            .children(extra)
            .child(div().flex_1())
            .children(trail)
    }
}

/// What a hidden side panel leaves at its window edge: a slim card with the button that shows it
/// again and the panel's icon, so it can always be found where it was.
pub fn rail(id: impl Into<ElementId>, title: &str, edge: Edge, on_show: impl Fn(&mut Window, &mut App) + 'static) -> impl IntoElement {
    let icon = match edge {
        Edge::Left => IconName::PanelLeftOpen,
        Edge::Right => IconName::PanelRightOpen,
        Edge::Bottom => IconName::ChevronUp,
    };
    let on_show = Rc::new(on_show);
    v_flex()
        .flex_none()
        .w(px(30.))
        .h_full()
        .py_1()
        .items_center()
        .rounded(px(theme::RADIUS_LG))
        .bg(color::panel())
        .child(icon_button(id, icon, format!("Show {title}")).on_click(move |_, window, cx| on_show(window, cx)))
}
