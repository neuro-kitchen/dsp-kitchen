---
title: Keyboard and mouse
description: "Every shortcut of the app, also shown by F1."
sidebar:
  order: 4
---

## Anywhere

| Keys | Action |
|---|---|
| Ctrl+O | Open a recording file |
| Ctrl+Shift+O | Open a recording folder (Zarr) |
| Ctrl+T / Ctrl+H | Add a traces / heatmap view |
| Ctrl+B / Ctrl+Alt+B / Ctrl+J | Show or hide Channels / View settings / Timeline |
| Ctrl+K | Command palette: any action, view, operation or recipe by name |
| Ctrl+Z / Ctrl+Shift+Z | Undo / redo any change of the session (in Curation: of the curation) |
| F1 | This list |
| Ctrl+Q | Quit |

## On a time view (click it first)

| Input | Action |
|---|---|
| Drag · horizontal swipe · Shift+wheel | Move in time |
| Ctrl+wheel · + / − | Zoom in time (wheel: at the pointer) |
| Wheel · ↑ / ↓ · Page Up / Down | Scroll channels |
| Alt+wheel · [ / ] | Gain |
| ← / → (Shift: a whole window) | Move in time |
| Home / End | Start / end of the recording |
| Space · L | Play or pause · loop |
| Double-click a heatmap | Show that channel in a traces view |
| Drag a view's tab | Snap it beside another view (edges) or onto its tabs |

## In Curation

| Keys | Action |
|---|---|
| Alt+G / Alt+M / Alt+N | Label the selected units good / mua / noise |
| G | Merge the selected units |
| Ctrl+S | Save |

## Change the keys

`keybindings.toml`, in the app's folder beside [`settings.toml`](/dsp-kitchen/app/settings/#the-settings-file),
gives each command a key or a list of keys. The app writes it with every default the first time it
starts, and reads it at each start.

```toml
[plot]                       # the time views
play_pause = "space"
zoom_in = ["+", "="]

[commands]                   # every command of the menus and the search, by its id
file.open = "ctrl-o"

[curation]
merge = "g"
next_unit = ["down", "j"]
label_good = "alt-g"
```

Keys are written as `ctrl-shift-o`, `alt-g`, `space`, `left`, `f3`. `""` removes a binding;
deleting a line gives the default back. A command or a key the app does not know keeps the default
and is named in the status line, as is one key given to two commands of a section.
