//! Search (Ctrl+K; Spotlight / Blender's F3): every action of the app by name — workspaces,
//! files, views, panels, undo, epochs, Settings, the Pipeline's operations and recipes —
//! filtered as you type, each row an icon tile, the name and a line saying what it does, under
//! group headings. ↑ ↓ move, Enter runs the highlighted one, a click runs any, Escape closes.
//! Opened from Process ▸ "Find an operation…", it lists the operations and recipes only.

use std::rc::Rc;

use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::assets::IconName;
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Icon};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, px, App, AppContext as _, Context, Entity, EventEmitter, FontWeight, InteractiveElement as _, IntoElement, KeyDownEvent, ParentElement as _, Render, SharedString,
    StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};

/// What a command does when run.
pub type Run = Rc<dyn Fn(&mut Window, &mut App)>;

/// A command: its group (`Go to`, `View`, `Filter`…), its name, what it does, its icon.
#[derive(Clone)]
pub struct Command {
    pub group: &'static str,
    pub name: String,
    /// What it does, or its key (the line under the name).
    pub hint: String,
    pub icon: IconName,
    pub run: Run,
}

pub enum PaletteEvent {
    Close,
    /// Close, then run this: the app runs it once the search is gone, with no view borrowed,
    /// so what it opens (Settings, a dialog, a workspace) keeps the focus.
    Run(Run),
}

pub struct CommandPalette {
    input: Entity<InputState>,
    commands: Vec<Command>,
    title: &'static str,
    /// The highlighted row (of the filtered list).
    cursor: usize,
    _subs: Vec<Subscription>,
}

impl EventEmitter<PaletteEvent> for CommandPalette {}

/// At most this many matches are listed.
const SHOWN: usize = 40;

impl CommandPalette {
    pub fn new(commands: Vec<Command>, title: &'static str, placeholder: &'static str, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let input = cx.new(|cx| InputState::new(window, cx).placeholder(placeholder));
        let subs = vec![cx.subscribe_in(&input, window, |this, _, e: &InputEvent, _, cx| match e {
            InputEvent::PressEnter { .. } => this.run_cursor(cx),
            InputEvent::Change => {
                this.cursor = 0;
                cx.notify()
            }
            _ => {}
        })];
        let focus = gpui_kit::Focusable::focus_handle(input.read(cx), cx);
        window.focus(&focus, cx);
        Self { input, commands, title, cursor: 0, _subs: subs }
    }

    /// Commands whose group, name and hint hold every word typed, in order of appearance.
    fn matches(&self, cx: &App) -> Vec<Command> {
        let q = self.input.read(cx).value().to_lowercase();
        let words: Vec<&str> = q.split_whitespace().collect();
        self.commands
            .iter()
            .filter(|c| {
                let hay = format!("{} {} {}", c.group, c.name, c.hint).to_lowercase();
                words.iter().all(|w| hay.contains(w))
            })
            .take(SHOWN)
            .cloned()
            .collect()
    }

    fn run_cursor(&mut self, cx: &mut Context<Self>) {
        if let Some(c) = self.matches(cx).into_iter().nth(self.cursor) {
            cx.emit(PaletteEvent::Run(c.run.clone()));
        }
    }

    fn on_key(&mut self, e: &KeyDownEvent, _: &mut Window, cx: &mut Context<Self>) {
        let n = self.matches(cx).len();
        match e.keystroke.key.as_str() {
            "down" => self.cursor = (self.cursor + 1).min(n.saturating_sub(1)),
            "up" => self.cursor = self.cursor.saturating_sub(1),
            "escape" => cx.emit(PaletteEvent::Close),
            "enter" => {
                self.run_cursor(cx);
                cx.stop_propagation();
                return;
            }
            _ => return,
        }
        cx.stop_propagation();
        cx.notify();
    }
}

impl Render for CommandPalette {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let (muted, hover, border, bg) = (t.muted_foreground, t.accent, t.border, t.popover);
        let found = self.matches(cx);
        let cursor = self.cursor.min(found.len().saturating_sub(1));
        let empty = found.is_empty();
        let mut list = v_flex().id("palette-rows").flex_1().min_h_0().overflow_y_scroll();
        let mut last_group = "";
        for (i, c) in found.into_iter().enumerate() {
            if c.group != last_group {
                last_group = c.group;
                list = list.child(div().px_2().pt(px(if i == 0 { 2. } else { 10. })).pb(px(2.)).text_xs().font_weight(FontWeight::SEMIBOLD).text_color(muted).child(c.group));
            }
            let run = c.run.clone();
            let me = cx.entity().downgrade();
            list = list.child(
                h_flex()
                    .id(("palette-row", i))
                    .min_h(px(48.))
                    .px_2()
                    .gap_3()
                    .rounded_md()
                    .cursor_pointer()
                    .when(i == cursor, |d| d.bg(hover))
                    .hover(|d| d.bg(hover))
                    .child(div().flex_none().size(px(32.)).rounded_md().border_1().border_color(border).flex().items_center().justify_center().child(Icon::new(c.icon)))
                    .child(
                        v_flex()
                            .flex_1()
                            .min_w_0()
                            .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).truncate().child(SharedString::from(c.name.clone())))
                            .child(div().text_xs().text_color(muted).truncate().child(c.hint.clone())),
                    )
                    .on_click(move |_, _, cx| {
                        let run = run.clone();
                        let _ = me.update(cx, |_, cx| cx.emit(PaletteEvent::Run(run)));
                    }),
            );
        }
        if empty {
            list = list.child(div().px_2().py_3().text_sm().text_color(muted).child("Nothing by that name"));
        }
        let me = cx.entity().downgrade();
        let close = cx.entity().downgrade();
        div()
            .id("palette-backdrop")
            .absolute()
            .top_0()
            .left_0()
            .size_full()
            .flex()
            .items_start()
            .justify_center()
            .pt(px(72.))
            .px_4()
            .bg(gpui_kit::hsla(0., 0., 0., 0.35))
            .on_click(move |_, _, cx| {
                let _ = me.update(cx, |_, cx| cx.emit(PaletteEvent::Close));
            })
            .child(
                v_flex()
                    .id("palette")
                    .occlude()
                    // Narrows with the window, never wider than reads well
                    .w_full()
                    .max_w(px(520.))
                    .max_h(gpui_kit::relative(0.8))
                    .p_2()
                    .gap_1()
                    .rounded_lg()
                    .border_1()
                    .border_color(border)
                    .bg(bg)
                    .shadow_lg()
                    .capture_key_down(cx.listener(Self::on_key))
                    .on_click(|_, _, cx| cx.stop_propagation())
                    .child(
                        h_flex()
                            .px_2()
                            .py_1()
                            .gap_2()
                            .items_baseline()
                            .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(self.title))
                            .child(div().text_xs().text_color(muted).child("↑ ↓ Enter · Esc"))
                            .child(div().flex_1())
                            .child(
                                div()
                                    .id("palette-close")
                                    .cursor_pointer()
                                    .rounded_sm()
                                    .p(px(2.))
                                    .hover(|d| d.bg(hover))
                                    .child(Icon::new(IconName::X))
                                    .on_click(move |_, _, cx| {
                                        let _ = close.update(cx, |_, cx| cx.emit(PaletteEvent::Close));
                                    }),
                            ),
                    )
                    .child(Input::new(&self.input))
                    .child(list),
            )
    }
}
