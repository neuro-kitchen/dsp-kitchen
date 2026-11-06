//! App actions and their key bindings (`secondary` = Cmd on macOS, Ctrl elsewhere).
//!
//! Plot keys (pan, zoom, scroll, gain, play) work while a time view has focus (a click on it), so
//! typing in a text field never moves the recording.

use gpui_kit::{actions, Action, App, KeyBinding};

/// Key context of a focused time view.
pub const PLOT: &str = "TimePlot";

actions!(
    dspk_app,
    [
        /// Choose a recording file and open it.
        OpenRecording,
        /// Open a procedural recording (no file needed).
        OpenSynthetic,
        AddTraces,
        AddHeatmap,
        ToggleChannels,
        ToggleSettings,
        ToggleTimeline,
        ShowHelp,
        ShowAbout,
        Quit,
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
        ToggleLoop,
        Undo,
        Redo,
        CommandPalette,
    ]
);

/// A global binding whose `secondary-` modifier is resolved for this OS (GPUI ≥ 0.3.11 resolves it
/// only in `KeyBinding::load`; `KeyBinding::new` refuses it).
fn os_binding<A: Action>(keys: &str, action: A, cx: &App) -> KeyBinding {
    KeyBinding::load(keys, Box::new(action), None, false, None, cx).unwrap_or_else(|e| panic!("key binding {keys:?}: {e:?}"))
}

pub fn bind_keys(cx: &mut App) {
    let global = [
        os_binding("secondary-o", OpenRecording, cx),
        os_binding("secondary-t", AddTraces, cx),
        os_binding("secondary-h", AddHeatmap, cx),
        os_binding("secondary-b", ToggleChannels, cx),
        os_binding("secondary-alt-b", ToggleSettings, cx),
        os_binding("secondary-j", ToggleTimeline, cx),
        os_binding("secondary-q", Quit, cx),
        os_binding("secondary-z", Undo, cx),
        os_binding("secondary-shift-z", Redo, cx),
        os_binding("secondary-y", Redo, cx),
        os_binding("secondary-k", CommandPalette, cx),
    ];
    cx.bind_keys(global);
    cx.bind_keys([
        KeyBinding::new("f1", ShowHelp, None),
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
        KeyBinding::new("l", ToggleLoop, Some(PLOT)),
    ]);
}

/// Every binding as (keys, what it does), for the Keyboard & Mouse dialog.
pub const HELP: &[(&str, &[(&str, &str)])] = &[
    (
        "Anywhere",
        &[
            ("Ctrl+O", "Open a recording"),
            ("Ctrl+T / Ctrl+H", "Add a traces / heatmap view"),
            ("Ctrl+B / Ctrl+Alt+B / Ctrl+J", "Show or hide Channels / View settings / Timeline"),
            ("Ctrl+K", "Command palette: any action, view, operation or recipe by name"),
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
            ("Drag a view's tab", "Snap it beside another view (edges) or onto its tabs"),
        ],
    ),
];
