//! Small pieces used by several views: a dropdown of fixed options, a titled boxed section, a
//! labelled row, a centred empty state, side-panel headers and rails. Plain data in, no store.

use std::rc::Rc;

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::empty::{Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyMediaVariant, EmptyTitle};
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Icon, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{div, px, AnyElement, App, ElementId, FontWeight, IntoElement, ParentElement, RenderOnce, SharedString, Styled as _, Window};

type OnPick = Rc<dyn Fn(usize, &mut Window, &mut App)>;

/// A button showing the current option; pressing it lists the options.
#[derive(IntoElement)]
pub struct MenuSelect {
    id: ElementId,
    current: SharedString,
    options: Vec<SharedString>,
    selected: Option<usize>,
    on_pick: OnPick,
    full: bool,
}

impl MenuSelect {
    pub fn new(id: impl Into<ElementId>, options: Vec<SharedString>, selected: Option<usize>, on_pick: impl Fn(usize, &mut Window, &mut App) + 'static) -> Self {
        let current = selected.and_then(|i| options.get(i).cloned()).unwrap_or_else(|| "Choose…".into());
        Self { id: id.into(), current, options, selected, on_pick: Rc::new(on_pick), full: false }
    }

    /// Spans its row.
    pub fn full_width(mut self) -> Self {
        self.full = true;
        self
    }
}

impl RenderOnce for MenuSelect {
    fn render(self, _: &mut Window, _: &mut App) -> impl IntoElement {
        let Self { id, current, options, selected, on_pick, full } = self;
        Button::new(id).outline().small().label(current).dropdown_caret(true).cursor_pointer().when(full, |b| b.w_full()).dropdown_menu(move |menu, _, _| {
            let mut menu = menu.scrollable(true).max_h(px(360.));
            for (i, label) in options.iter().enumerate() {
                let on_pick = on_pick.clone();
                menu = menu.item(PopupMenuItem::new(label.clone()).checked(selected == Some(i)).on_click(move |_, window, cx| on_pick(i, window, cx)));
            }
            menu
        })
    }
}

/// A titled block: the title above a bordered box holding the block's rows.
#[derive(IntoElement)]
pub struct Section {
    title: SharedString,
    children: Vec<AnyElement>,
}

impl Section {
    pub fn new(title: impl Into<SharedString>) -> Self {
        Self { title: title.into(), children: Vec::new() }
    }
}

impl ParentElement for Section {
    fn extend(&mut self, elements: impl IntoIterator<Item = AnyElement>) {
        self.children.extend(elements);
    }
}

impl RenderOnce for Section {
    fn render(self, _: &mut Window, cx: &mut App) -> impl IntoElement {
        let t = cx.theme();
        v_flex()
            .gap_1p5()
            .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(t.muted_foreground).child(self.title))
            .child(v_flex().gap_2p5().p_2p5().rounded_lg().border_1().border_color(t.border).bg(t.background).children(self.children))
    }
}

/// `label    control` on one line.
pub fn row(label: impl Into<SharedString>, control: impl IntoElement, cx: &App) -> impl IntoElement {
    h_flex().gap_2().justify_between().child(div().text_sm().text_color(cx.theme().muted_foreground).child(label.into())).child(control)
}

/// Centred icon, title and a line of explanation, with optional actions under it.
#[derive(IntoElement)]
pub struct EmptyState {
    icon: Icon,
    title: SharedString,
    description: SharedString,
    actions: Vec<AnyElement>,
}

impl EmptyState {
    pub fn new(icon: impl Into<Icon>, title: impl Into<SharedString>, description: impl Into<SharedString>) -> Self {
        Self { icon: icon.into(), title: title.into(), description: description.into(), actions: Vec::new() }
    }
}

impl ParentElement for EmptyState {
    fn extend(&mut self, elements: impl IntoIterator<Item = AnyElement>) {
        self.actions.extend(elements);
    }
}

impl RenderOnce for EmptyState {
    fn render(self, _: &mut Window, _: &mut App) -> impl IntoElement {
        let header = EmptyHeader::new()
            .media(EmptyMedia::new().with_variant(EmptyMediaVariant::Icon).size_12().child(self.icon.size_6()))
            .title(EmptyTitle::new().text_lg().child(self.title))
            .description(EmptyDescription::new().text_sm().max_w(px(460.)).child(self.description));
        v_flex().size_full().items_center().justify_center().gap_4().child(Empty::new().flex_none().header(header)).child(v_flex().gap_2().items_center().children(self.actions))
    }
}

/// A small ghost icon button with a tooltip.
pub fn icon_button(id: impl Into<ElementId>, icon: impl Into<Icon>, tooltip: impl Into<SharedString>) -> Button {
    Button::new(id).ghost().small().icon(icon).tooltip(tooltip.into())
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
    fn render(self, _: &mut Window, cx: &mut App) -> impl IntoElement {
        let Self { id, title, edge, extra, open, on_hide } = self;
        let t = cx.theme();
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
            .border_color(t.border)
            .children(lead)
            .when(!left, |d| d.pl_2())
            .child(title)
            .children(extra)
            .child(div().flex_1())
            .children(trail)
    }
}

/// What a hidden side panel leaves at its window edge: a slim strip with the button that shows it
/// again and the panel's icon, so it can always be found where it was.
pub fn rail(id: impl Into<ElementId>, title: &str, edge: Edge, on_show: impl Fn(&mut Window, &mut App) + 'static, cx: &App) -> impl IntoElement {
    let t = cx.theme();
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
        .bg(t.sidebar)
        .map(|d| match edge {
            Edge::Left => d.border_r_1(),
            _ => d.border_l_1(),
        })
        .border_color(t.border)
        .child(icon_button(id, icon, format!("Show {title}")).on_click(move |_, window, cx| on_show(window, cx)))
}
