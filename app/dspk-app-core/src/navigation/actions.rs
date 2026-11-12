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

/// Binds the time views' keys.
pub fn bind_keys(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("space", PlayPause, Some(PLOT)),
        KeyBinding::new("left", PanBack, Some(PLOT)),
        KeyBinding::new("right", PanForward, Some(PLOT)),
        KeyBinding::new("shift-left", PageBack, Some(PLOT)),
        KeyBinding::new("shift-right", PageForward, Some(PLOT)),
        KeyBinding::new("+", ZoomIn, Some(PLOT)),
        KeyBinding::new("=", ZoomIn, Some(PLOT)),
        KeyBinding::new("-", ZoomOut, Some(PLOT)),
        KeyBinding::new("home", JumpStart, Some(PLOT)),
        KeyBinding::new("end", JumpEnd, Some(PLOT)),
        KeyBinding::new("up", ChannelUp, Some(PLOT)),
        KeyBinding::new("down", ChannelDown, Some(PLOT)),
        KeyBinding::new("pageup", ChannelPageUp, Some(PLOT)),
        KeyBinding::new("pagedown", ChannelPageDown, Some(PLOT)),
        KeyBinding::new("]", GainUp, Some(PLOT)),
        KeyBinding::new("[", GainDown, Some(PLOT)),
        KeyBinding::new("escape", LeaveSolo, Some(PLOT)),
        KeyBinding::new("l", ToggleLoop, Some(PLOT)),
    ]);
}

/// Every binding as (keys, what it does), for the Keyboard & Mouse dialog.
pub const HELP: &[(&str, &[(&str, &str)])] = &[
    (
        "Anywhere",
        &[
            ("Ctrl+O", "Open a recording file"),
            ("Ctrl+Shift+O", "Open a recording folder (Zarr)"),
            ("Ctrl+T / Ctrl+H", "Add a traces / heatmap view"),
            ("Ctrl+B / Ctrl+Alt+B / Ctrl+J", "Show or hide Channels / View settings / Timeline"),
            ("Ctrl+K · F3", "Search: any action, view, operation or recipe by name · the operations only"),
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
