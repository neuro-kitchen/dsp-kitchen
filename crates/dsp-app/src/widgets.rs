//! Small pieces used by several views: a dropdown of fixed options, a titled boxed section, a
//! labelled row, a centred empty state. Plain data in, no store.

use std::rc::Rc;

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::empty::{Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyMediaVariant, EmptyTitle};
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Icon, Sizable as _};
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
