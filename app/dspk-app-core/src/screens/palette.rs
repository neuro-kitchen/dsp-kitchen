//! The command palette (Ctrl+K): every action of the app by name — workspaces, views, files,
//! undo, epochs, the Pipeline's operations and recipes — filtered as you type; Enter runs the
//! first, a click runs any, Escape closes.

use std::rc::Rc;

use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _};
use gpui_kit::{
    div, px, App, AppContext as _, Context, Entity, EventEmitter, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString,
    StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};

/// What a command does when run.
pub type Run = Rc<dyn Fn(&mut Window, &mut App)>;

/// A command: its group (`Go to`, `View`, `Pipeline`…), its name, what it does.
#[derive(Clone)]
pub struct Command {
    pub group: &'static str,
    pub name: String,
    pub run: Run,
}

pub enum PaletteEvent {
    Close,
}

pub struct CommandPalette {
    input: Entity<InputState>,
    commands: Vec<Command>,
    _subs: Vec<Subscription>,
}

impl EventEmitter<PaletteEvent> for CommandPalette {}

/// At most this many matches are listed.
const SHOWN: usize = 14;

impl CommandPalette {
    pub fn new(commands: Vec<Command>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let input = cx.new(|cx| InputState::new(window, cx).placeholder("Type a command, a view, an operation or a recipe…"));
        let subs = vec![cx.subscribe_in(&input, window, |this, _, e: &InputEvent, window, cx| match e {
            InputEvent::PressEnter { .. } => {
                if let Some(c) = this.matches(cx).into_iter().next() {
                    cx.emit(PaletteEvent::Close);
                    (c.run)(window, cx);
                }
            }
            InputEvent::Change => cx.notify(),
            _ => {}
        })];
        let focus = gpui_kit::Focusable::focus_handle(input.read(cx), cx);
        window.focus(&focus, cx);
        Self { input, commands, _subs: subs }
    }

    /// Commands whose group and name hold every word typed, in order of appearance.
    fn matches(&self, cx: &App) -> Vec<Command> {
        let q = self.input.read(cx).value().to_lowercase();
        let words: Vec<&str> = q.split_whitespace().collect();
        self.commands
            .iter()
            .filter(|c| {
                let hay = format!("{} {}", c.group, c.name).to_lowercase();
                words.iter().all(|w| hay.contains(w))
            })
            .take(SHOWN)
            .cloned()
            .collect()
    }
}

impl Render for CommandPalette {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let (muted, hover, border, bg) = (t.muted_foreground, t.accent, t.border, t.popover);
        let rows = self.matches(cx).into_iter().enumerate().map(|(i, c)| {
            let run = c.run.clone();
            let me = cx.entity().downgrade();
            h_flex()
                .id(("palette-row", i))
                .h(px(30.))
                .px_3()
                .gap_3()
                .rounded_md()
                .cursor_pointer()
                .hover(|d| d.bg(hover))
                .when_first(i == 0, |d| d.bg(hover))
                .child(div().w(px(90.)).flex_none().text_xs().text_color(muted).child(c.group))
                .child(div().flex_1().min_w_0().truncate().text_sm().child(SharedString::from(c.name.clone())))
                .on_click(move |_, window, cx| {
                    let _ = me.update(cx, |_, cx| cx.emit(PaletteEvent::Close));
                    run(window, cx);
                })
        });
        let me = cx.entity().downgrade();
        div()
            .id("palette-backdrop")
            .absolute()
            .top_0()
            .left_0()
            .size_full()
            .flex()
            .justify_center()
            .pt(px(90.))
            .bg(gpui_kit::hsla(0., 0., 0., 0.35))
            .on_click(move |_, _, cx| {
                let _ = me.update(cx, |_, cx| cx.emit(PaletteEvent::Close));
            })
            .on_key_down(cx.listener(|_, e: &gpui_kit::KeyDownEvent, _, cx| {
                if e.keystroke.key == "escape" {
                    cx.emit(PaletteEvent::Close);
                }
            }))
            .child(
                v_flex()
                    .id("palette")
                    .w(px(560.))
                    .max_h(px(520.))
                    .p_2()
                    .gap_1()
                    .rounded_lg()
                    .border_1()
                    .border_color(border)
                    .bg(bg)
                    .shadow_lg()
                    .on_click(|_, _, cx| cx.stop_propagation())
                    .child(Input::new(&self.input))
                    .child(v_flex().id("palette-rows").overflow_y_scroll().children(rows))
                    .child(div().px_1().pt_1().text_xs().text_color(muted).font_weight(FontWeight::NORMAL).child("Enter runs the first · Esc closes")),
            )
    }
}

trait WhenFirst: Sized {
    fn when_first(self, cond: bool, f: impl FnOnce(Self) -> Self) -> Self;
}

impl<T: Sized> WhenFirst for T {
    fn when_first(self, cond: bool, f: impl FnOnce(Self) -> Self) -> Self {
        if cond { f(self) } else { self }
    }
}
