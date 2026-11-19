---
title: Settings and servers
description: "Servers, the runtime, the results folder and the theme."
sidebar:
  order: 3
---

**File → Settings…** (Ctrl+,). Every value shows its default.

| Setting | What it does |
|---|---|
| Servers | Add, edit, remove and test the servers the app works with |
| Results folder | Where results fetched from a server go |
| Compute runtime | Which runtime the app uses |
| Memory | RAM the app keeps to be fast, for every open recording: decoded chunks, processed overviews, spike windows. Automatic is a quarter of the RAM (half of what a browser reports, at most 4 GiB). Past it, the least recently used goes and is read or computed again. Also `[compute] memory` in `settings.toml`, as `"8GiB"` |
| Theme | Light or dark, or follow the system. **View** also has "Dark plots in the light theme" |

![Settings with the compute runtime menu open: automatic, wgpu, CUDA or CPU](/dsp-kitchen/img/app/10_Settings-select-runtime.png)

## Add a server

You need a [running server](/dsp-kitchen/deploy/run/) and an
[identity file](/dsp-kitchen/deploy/clients/) for this client.

1. In Settings, **Add server**: a name, its address (`192.168.1.20:50051`), the name on its certificate
   and your identity file.
2. **Test** connects and reports what it finds.

## Open from a server

**File → Open from server…** picks a server and walks its data root. Open a recording to view it in
Explore (read on the server), or fetch a file or folder, such as a sorting, into the results folder.

A remote recording behaves like a local one. A sort started on it runs on the server, and its
progress shows like the app's other jobs.

## People

The person button at the right of the title bar says who works: a name, or **Guest**. Pick
someone from its menu, or **People…** to add one (a name, and optionally a lab and an
[ORCID iD](https://orcid.org)) or forget one. Names are kept in `people.json` in the app's folder;
there are no passwords.

The name is written on what you save: a curation's edits, and a recording's
[session metadata](/dsp-kitchen/app/explore/#session-and-subject) can list you as an
experimenter. A guest's edits say `guest`, with no name. The app records nothing about what you do:
**Analytics → This session** lists what this session made (sorts, pipeline runs, files saved),
and it is gone when the app closes.

## The settings file

The Settings screen writes `settings.toml` in the app's folder (`~/.config/dspk-app/` on Linux,
`%APPDATA%\dspk-app\` on Windows). You can edit it by hand: the app applies the change while it
runs, except the compute runtime, which takes effect at the next start. Comments you add are kept.

```toml
[appearance]
theme = "system"        # "system" | "dark" | "light"
accent = "app"          # "app" or a colour as "#rrggbb"
dark_plots = false      # keep plots dark in the light theme

[start]
reopen_last = false     # open the last recording when the app starts

[compute]
runtime = "automatic"   # "automatic" | "wgpu" | "cuda" | "cpu"

[remote]
results_dir = ""        # where results fetched from servers go; "" is ~/dspk/remote
```

Delete a line to go back to its default. A line the app does not understand keeps its default and
is named in the status line. Keys are in [`keybindings.toml`](/dsp-kitchen/app/keyboard/#change-the-keys),
beside it.
