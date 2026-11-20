//! The time views' own keys (their key context, `secondary` = Cmd on macOS, Ctrl elsewhere).
//! Every other key belongs to a command (`crate::commands`), bound by kitchen-shell's registry.
//!
//! Plot keys (pan, zoom, scroll, gain, play) work while a time view has focus (a click on it), so
//! typing in a text field never moves the recording.

use gpui_kit::{actions, App, KeyBinding};

/// Key context of a focused time view.
pub const PLOT: &str = "TimePlot";

actions!(
    dspk_app,
    [
        PlayPause,
        PanBack,
        PanForward,
        PageBack,
        PageForward,
        ZoomIn,
        ZoomOut,
        JumpStart,
        JumpEnd,
        ChannelUp,
        ChannelDown,
        ChannelPageUp,
        ChannelPageDown,
        GainUp,
        GainDown,
        /// Back from a channel shown alone (double-click) to the view's channels.
        LeaveSolo,
        ToggleLoop,
    ]
);

/// The time views' commands and their default keys: the `[plot]` section of `keybindings.toml`.
pub const PLOT_KEYS: &[(&str, &[&str])] = &[
    ("play_pause", &["space"]),
    ("pan_back", &["left"]),
    ("pan_forward", &["right"]),
    ("page_back", &["shift-left"]),
    ("page_forward", &["shift-right"]),
    ("zoom_in", &["+", "="]),
    ("zoom_out", &["-"]),
    ("jump_start", &["home"]),
    ("jump_end", &["end"]),
    ("channel_up", &["up"]),
    ("channel_down", &["down"]),
    ("channel_page_up", &["pageup"]),
    ("channel_page_down", &["pagedown"]),
    ("gain_up", &["]"]),
    ("gain_down", &["["]),
    ("leave_solo", &["escape"]),
    ("toggle_loop", &["l"]),
];

fn plot_action(name: &str) -> Box<dyn gpui_kit::Action> {
    match name {
        "play_pause" => Box::new(PlayPause),
        "pan_back" => Box::new(PanBack),
        "pan_forward" => Box::new(PanForward),
        "page_back" => Box::new(PageBack),
        "page_forward" => Box::new(PageForward),
        "zoom_in" => Box::new(ZoomIn),
        "zoom_out" => Box::new(ZoomOut),
        "jump_start" => Box::new(JumpStart),
        "jump_end" => Box::new(JumpEnd),
        "channel_up" => Box::new(ChannelUp),
        "channel_down" => Box::new(ChannelDown),
        "channel_page_up" => Box::new(ChannelPageUp),
        "channel_page_down" => Box::new(ChannelPageDown),
        "gain_up" => Box::new(GainUp),
        "gain_down" => Box::new(GainDown),
        "leave_solo" => Box::new(LeaveSolo),
        _ => Box::new(ToggleLoop),
    }
}

/// Binds the time views' keys: the defaults, or what `keybindings.toml` says in `[plot]`.
pub fn bind_keys(cx: &mut App) {
    let context = gpui_kit::KeyBindingContextPredicate::parse(PLOT).ok().map(std::rc::Rc::new);
    let load = |key: &str, name: &str, cx: &App| KeyBinding::load(key, plot_action(name), context.clone(), false, None, cx);
    let resolved = crate::data::settings::resolve("plot", PLOT_KEYS, |key| load(key, "play_pause", cx).is_ok());
    let mut bindings = Vec::new();
    for (name, keys) in &resolved {
        bindings.extend(keys.iter().filter_map(|key| load(key, name, cx).ok()));
    }
    cx.bind_keys(bindings);
}

/// Every binding as (keys, what it does), for the Keyboard & Mouse dialog.
pub const HELP: &[(&str, &[(&str, &str)])] = &[
    (
        "Anywhere",
        &[
            ("Ctrl+O", "Open a recording file"),
            ("Ctrl+Alt+O", "Open a recording folder (Zarr)"),
            ("Ctrl+Shift+O", "Search & add a GPU operation"),
            ("Ctrl+T / Ctrl+H", "Add a traces / heatmap view"),
            ("Ctrl+B / Ctrl+Alt+B / Ctrl+J", "Show or hide Channels / View settings / Timeline"),
            ("Ctrl+K", "Command palette: any action, view, operation or recipe"),
            ("Ctrl+Z / Ctrl+Shift+Z", "Undo / redo any change of the session (Curation: of the curation)"),
            ("F1", "This list"),
            ("Ctrl+Q", "Quit"),
        ],
    ),
    (
        "On a time view (click it first)",
        &[
            ("Drag · horizontal swipe · Shift+wheel", "Move in time"),
            ("Ctrl+wheel · + / −", "Zoom in time (wheel: at the pointer)"),
            ("Wheel · ↑ / ↓ · Page Up / Down", "Scroll channels"),
            ("Alt+wheel · [ / ]", "Gain"),
            ("← / →  (Shift: a whole window)", "Move in time"),
            ("Home / End", "Start / end of the recording"),
            ("Space · L", "Play / pause · loop"),
            ("Double-click a heatmap", "Show that channel in a traces view"),
            ("Double-click a trace · Esc", "Show that channel alone · back to all"),
            ("Drag a view's tab", "Snap it beside another view (edges) or onto its tabs"),
        ],
    ),
];
